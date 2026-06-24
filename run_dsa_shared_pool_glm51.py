# SPDX-License-Identifier: Apache-2.0
"""Manual end-to-end smoke test for GLM5.1 DSA shared KV pool.

Run this from the vLLM repository root on an Ascend NPU machine with the
patched vLLM and vLLM-Ascend branches installed or importable.

Example:
    GLM51_MODEL=/models/GLM-5-w4a8 \
    python run_dsa_shared_pool_glm51.py --tp 16

If vLLM-Ascend is checked out next to vLLM instead of installed, this script
will automatically prepend ../vllm-ascend to sys.path. You can also set:
    VLLM_ASCEND_REPO=/workspace/lmy/vllm-ascend
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path


DSA_ENV_DEFAULTS = {
    "VLLM_USE_V1": "1",
    "VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD": "1",
    "VLLM_ASCEND_DSA_UNBUNDLE": "1",
    "VLLM_ASCEND_DSA_TWO_GROUPS": "1",
    "VLLM_ASCEND_DSA_SHARED_POOL": "1",
    "VLLM_ASCEND_DSA_SHRINK_LATENT": "2",
    # Stage prefill latent outside NPU memory in this smoke test.
    "VLLM_ASCEND_DSA_OFFLOAD_BACKEND_DEVICE": "cpu",
    # Prefix caching is not supported by DSA two-group/shared-pool mode.
    "VLLM_ENABLE_PREFIX_CACHING": "0",
}


@dataclass
class RunConfig:
    model: str
    tp: int
    max_model_len: int
    max_num_seqs: int
    prompt_tokens: int
    max_tokens: int
    temperature: float
    gpu_memory_utilization: float
    dtype: str
    quantization: str | None
    enforce_eager: bool
    backend_device: str


def maybe_add_vllm_ascend_repo() -> None:
    configured = os.getenv("VLLM_ASCEND_REPO")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.append(Path(__file__).resolve().parent.parent / "vllm-ascend")

    for candidate in candidates:
        if (candidate / "vllm_ascend").is_dir():
            sys.path.insert(0, str(candidate))
            return


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Run GLM5.1 once with DSA shared KV pool enabled."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GLM51_MODEL"),
        help="GLM5.1 model path or HF id. Can also be set by GLM51_MODEL.",
    )
    parser.add_argument("--tp", type=int, default=int(os.getenv("TP_SIZE", "1")))
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=2304,
        help="Approximate minimum tokenizer length per prompt. Keep this above "
        "index_topk to exercise prefill shrink.",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=os.getenv("VLLM_QUANTIZATION"))
    parser.add_argument(
        "--no-enforce-eager",
        action="store_true",
        help="Allow graph mode. Eager is the default for bring-up stability.",
    )
    parser.add_argument(
        "--backend-device",
        choices=("cpu", "npu"),
        default=os.getenv("VLLM_ASCEND_DSA_OFFLOAD_BACKEND_DEVICE", "cpu"),
        help="Where the reference LMCache stand-in stores prefill latent.",
    )
    args = parser.parse_args()
    if not args.model:
        parser.error("missing --model or GLM51_MODEL")
    return RunConfig(
        model=args.model,
        tp=args.tp,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        prompt_tokens=args.prompt_tokens,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        quantization=args.quantization,
        enforce_eager=not args.no_enforce_eager,
        backend_device=args.backend_device,
    )


def apply_env(config: RunConfig) -> None:
    for key, value in DSA_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    os.environ["VLLM_ASCEND_DSA_OFFLOAD_BACKEND_DEVICE"] = config.backend_device


def require_npu() -> None:
    import torch

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("torch_npu is required for this GLM5.1 NPU test.") from exc
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("No available Ascend NPU was found.")


def get_index_topk(model: str) -> int:
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    text_config = getattr(hf_config, "text_config", hf_config)
    index_topk = getattr(text_config, "index_topk", None)
    if index_topk is None:
        raise RuntimeError(
            f"{model!r} does not expose index_topk; this is not a DSA sparse model."
        )
    return int(index_topk)


def make_prompts(model: str, target_tokens: int, batch_size: int) -> list[str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    seed = (
        "This is a deterministic DSA shared pool smoke-test context. "
        "The shared KV cache pool manages latent and indexer pages by bundle. "
        "After prefill, extra latent pages are released, while decode uses the "
        "indexer top-k selection to gather only the needed latent tokens. "
    )
    prompt = seed
    # Build a deterministic prompt long enough to trigger the post-prefill shrink.
    while len(tokenizer.encode(prompt, add_special_tokens=False)) < target_tokens:
        prompt += "\n" + seed
    prompt += "\nQuestion: What does this smoke test verify?"
    return [prompt for _ in range(batch_size)]


def run_generation(config: RunConfig) -> None:
    require_npu()
    index_topk = get_index_topk(config.model)
    if config.prompt_tokens <= index_topk:
        raise RuntimeError(
            f"--prompt-tokens={config.prompt_tokens} must be > index_topk={index_topk} "
            "to exercise DSA latent shrink."
        )
    if config.max_model_len <= config.prompt_tokens + config.max_tokens:
        raise RuntimeError(
            "--max-model-len must exceed prompt_tokens + max_tokens "
            f"({config.prompt_tokens + config.max_tokens})."
        )

    from vllm import LLM, SamplingParams

    prompts = make_prompts(config.model, config.prompt_tokens, config.max_num_seqs)
    sampling = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    llm_kwargs = {
        "model": config.model,
        "trust_remote_code": True,
        "tensor_parallel_size": config.tp,
        "max_model_len": config.max_model_len,
        "max_num_seqs": config.max_num_seqs,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "dtype": config.dtype,
        "enforce_eager": config.enforce_eager,
        "enable_prefix_caching": False,
    }
    if config.quantization:
        llm_kwargs["quantization"] = config.quantization

    print("[DSA-SHARED-E2E] config:")
    print(f"  model={config.model}")
    print(f"  tp={config.tp}")
    print(f"  index_topk={index_topk}")
    print(f"  prompt_tokens_target={config.prompt_tokens}")
    print(f"  backend_device={config.backend_device}")
    for key in sorted(DSA_ENV_DEFAULTS):
        print(f"  {key}={os.environ.get(key)}")

    llm = LLM(**llm_kwargs)
    outputs = llm.generate(prompts, sampling, use_tqdm=False)

    if len(outputs) != len(prompts):
        raise RuntimeError(f"Expected {len(prompts)} outputs, got {len(outputs)}.")
    for i, output in enumerate(outputs):
        if not output.outputs:
            raise RuntimeError(f"Request {i} returned no completion.")
        completion = output.outputs[0]
        token_ids = getattr(completion, "token_ids", None) or []
        text = completion.text
        print(f"[DSA-SHARED-E2E] request={i} tokens={len(token_ids)} text={text!r}")
        if len(token_ids) == 0 and not text:
            raise RuntimeError(f"Request {i} generated neither token ids nor text.")

    print("[DSA-SHARED-E2E] PASS")


def main() -> int:
    maybe_add_vllm_ascend_repo()
    config = parse_args()
    apply_env(config)
    try:
        run_generation(config)
    except Exception as exc:
        print(f"[DSA-SHARED-E2E] FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
