import importlib.util
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault(
    "VLLM_ENABLE_V1_MULTIPROCESSING",
    os.environ.get("TEST_V1_MULTIPROCESSING", "0"),
)

import time
from typing import Any

from huggingface_hub import snapshot_download, try_to_load_from_cache
from huggingface_hub.errors import LocalEntryNotFoundError
import torch
import vllm
from transformers import AutoTokenizer
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import KVTransferConfig

print(vllm.__file__)


def resolve_model_path(model_name: str) -> tuple[str, bool]:
    if os.path.isdir(model_name):
        return model_name, True

    cached_config = try_to_load_from_cache(
        repo_id=model_name,
        filename="config.json",
    )
    if not isinstance(cached_config, str):
        return model_name, False

    try:
        local_model_path = snapshot_download(
            repo_id=model_name,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        return model_name, False

    os.environ["HF_HUB_OFFLINE"] = "1"
    print(f"Using cached model from: {local_model_path}")
    return local_model_path, True


def build_kv_transfer_config() -> KVTransferConfig:
    kv_connector = os.environ.get("TEST_KV_CONNECTOR", "LMCacheConnectorV1")
    engine_id = os.environ.get("TEST_LMCACHE_ENGINE_ID", "test-lmcache-local")
    if kv_connector != "LMCacheConnectorV1":
        return KVTransferConfig(
            kv_connector=kv_connector,
            kv_role="kv_both",
            engine_id=engine_id,
            kv_connector_extra_config={},
        )

    use_native = os.environ.get("TEST_LMCACHE_USE_NATIVE", "0") == "1"
    if use_native and importlib.util.find_spec("lmcache.config") is None:
        raise RuntimeError(
            "TEST_LMCACHE_USE_NATIVE=1 requires an LMCache build that exposes "
            "'lmcache.config'. The installed LMCache package in this environment "
            "does not provide that API; use TEST_LMCACHE_USE_NATIVE=0 instead."
        )
    local_cpu_size = float(os.environ.get("TEST_LMCACHE_LOCAL_CPU_SIZE", "5.0"))
    chunk_size = int(os.environ.get("TEST_LMCACHE_CHUNK_SIZE", "256"))
    return KVTransferConfig(
        kv_connector=kv_connector,
        kv_role="kv_both",
        engine_id=engine_id,
        kv_connector_extra_config={
            "use_native": use_native,
            "lmcache.local_cpu": True,
            "lmcache.max_local_cpu_size": local_cpu_size,
            "lmcache.chunk_size": chunk_size,
        },
    )


def cleanup_lmcache() -> None:
    from lmcache.integration.vllm.utils import ENGINE_NAME
    from lmcache.v1.cache_engine import LMCacheEngineBuilder

    LMCacheEngineBuilder.destroy(ENGINE_NAME)


def cleanup_distributed() -> None:
    from vllm.distributed.parallel_state import destroy_distributed_environment

    destroy_distributed_environment()


def build_prompt(tokenizer: AutoTokenizer, request_index: int) -> str:
    shared_prefix_repeat = int(os.environ.get("TEST_SHARED_PREFIX_REPEAT", "0"))
    shared_prefix_text = os.environ.get(
        "TEST_SHARED_PREFIX_TEXT",
        "春天到了，万物复苏，微风轻拂，花香弥漫。",
    )
    shared_prefix = shared_prefix_text * shared_prefix_repeat
    user_content = os.environ.get(
        f"TEST_USER_PROMPT_{request_index}",
        os.environ.get("TEST_USER_PROMPT", "写一篇描写春天的小作文"),
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": shared_prefix + user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def run_request(
    engine: LLMEngine,
    tokenizer: AutoTokenizer,
    request_id: str,
    prompt: str,
    sampling_params: SamplingParams,
    *,
    print_output: bool,
    profile_decode_only: bool,
) -> dict[str, Any]:
    print(f"=== 添加请求 {request_id} ===")
    print(f"Prompt: {prompt}")
    print(f"Sampling params: {sampling_params}\n")

    engine.add_request(
        request_id=request_id,
        prompt=prompt,
        params=sampling_params,
    )
    print("✅ 请求已添加到等待队列\n")

    print("=== 开始执行引擎主循环 ===")
    step_count = 0
    generated_text = ""
    loop_start_time = time.perf_counter()
    prefill_elapsed = 0.0
    decode_start_time = None
    decode_end_time = None
    decode_range_id = None

    while engine.has_unfinished_requests():
        step_count += 1
        step_start_time = time.perf_counter()
        if print_output:
            print(f"\n--- 第 {step_count} 次 engine.step() ---")

        outputs = engine.step()
        if step_count == 1:
            if profile_decode_only:
                torch.cuda.synchronize()
            prefill_elapsed = time.perf_counter() - step_start_time
            if profile_decode_only:
                decode_range_id = torch.cuda.nvtx.range_start(
                    f"decode_only_{request_id}"
                )
            decode_start_time = time.perf_counter()

        for output in outputs:
            if output.finished:
                print(f"✅ 请求 {output.request_id} 完成")
                generated_text = output.outputs[0].text
            elif print_output:
                new_token = output.outputs[0].text
                print(f"生成token: {repr(new_token)}")

    if decode_range_id is not None:
        torch.cuda.synchronize()
        decode_end_time = time.perf_counter()
        torch.cuda.nvtx.range_end(decode_range_id)
    else:
        decode_end_time = time.perf_counter()

    loop_elapsed = decode_end_time - loop_start_time
    generated_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    decode_tokens = max(0, generated_tokens - 1) if step_count > 1 else 0
    decode_elapsed = (
        decode_end_time - decode_start_time if decode_start_time is not None else 0.0
    )
    prefill_tokens_per_second = (
        prompt_tokens / prefill_elapsed if prefill_elapsed > 0 else 0.0
    )
    decode_tokens_per_second = (
        decode_tokens / decode_elapsed if decode_elapsed > 0 else 0.0
    )

    print("\n=== 生成完成 ===")
    print(f"完整输出: {generated_text}")
    print(f"总step数: {step_count}")
    print(f"其中: 第1步 = prefill阶段，第2-{step_count}步 = decode阶段")
    print(f"生成token数: {generated_tokens}")
    print(f"耗时: {loop_elapsed:.3f}s")
    print(f"prefill token数: {prompt_tokens}")
    print(f"prefill耗时: {prefill_elapsed:.3f}s")
    print(f"prefill tokens/s: {prefill_tokens_per_second:.3f}")
    print(f"decode token数: {decode_tokens}")
    print(f"decode耗时: {decode_elapsed:.3f}s")
    print(f"decode tokens/s: {decode_tokens_per_second:.3f}")

    return {
        "request_id": request_id,
        "generated_text": generated_text,
        "step_count": step_count,
        "generated_tokens": generated_tokens,
        "prompt_tokens": prompt_tokens,
        "decode_tokens": decode_tokens,
        "loop_elapsed": loop_elapsed,
        "prefill_elapsed": prefill_elapsed,
        "decode_elapsed": decode_elapsed,
        "prefill_tokens_per_second": prefill_tokens_per_second,
        "decode_tokens_per_second": decode_tokens_per_second,
    }


def main():
    kv_connector = os.environ.get("TEST_KV_CONNECTOR", "LMCacheConnectorV1")
    sparse_flag = os.environ.get("TEST_USE_SPARSE_ATTENTION")
    if sparse_flag is None:
        sparse_flag = os.environ.get("TEST_USE_SPARSE_CLUSTER", "1")
    USE_SPARSE_ATTENTION = sparse_flag != "0"
    USE_LMCACHE = os.environ.get("TEST_USE_LMCACHE", "1") != "0"
    PROFILE_DECODE_ONLY = os.environ.get("TEST_PROFILE_DECODE_ONLY", "0") == "1"
    default_async_scheduling = "1"
    ASYNC_SCHEDULING = (
        os.environ.get("TEST_ASYNC_SCHEDULING", default_async_scheduling) == "1"
    )
    distributed_executor_backend = os.environ.get(
        "TEST_DISTRIBUTED_EXECUTOR_BACKEND",
        "uni",
    )
    gqa_topk_mode = os.environ.get("TEST_GQA_TOPK_MODE", "group_avg")
    model = "Qwen/Qwen2.5-7B-Instruct"
    context_length = int(os.environ.get("TEST_CONTEXT_LENGTH", "2048"))
    max_tokens = int(os.environ.get("TEST_MAX_TOKENS", str(context_length)))
    block_size = 16
    test_speed = os.environ.get("TEST_IGNORE_EOS", "1") == "1"
    default_enforce_eager = "0"
    enforce_eager = os.environ.get("TEST_ENFORCE_EAGER", default_enforce_eager) == "1"
    enable_prefix_caching = (
        os.environ.get(
            "TEST_ENABLE_PREFIX_CACHING",
            "0" if USE_LMCACHE and kv_connector == "LMCacheConnectorV1" else "1",
        )
        == "1"
    )
    request_count = int(
        os.environ.get("TEST_REQUEST_COUNT", "2" if USE_LMCACHE else "1")
    )
    gpu_memory_utilization = float(
        os.environ.get("TEST_GPU_MEMORY_UTILIZATION", "0.5")
    )
    print_output = False
    model_path, using_local_cache = resolve_model_path(model)
    if gqa_topk_mode not in ("head_union", "group_avg"):
        raise ValueError(
            f"TEST_GQA_TOPK_MODE must be 'head_union' or 'group_avg', got {gqa_topk_mode!r}"
        )
    if request_count < 1:
        raise ValueError(f"TEST_REQUEST_COUNT must be >= 1, got {request_count}")

    kv_transfer_config = build_kv_transfer_config() if USE_LMCACHE else None

    # ====================== 1. 引擎配置（和server端完全一致） ======================
    if USE_SPARSE_ATTENTION:
        engine_args = EngineArgs(
            # 用最小的模型调试，速度快，显存占用小
            model=model_path,
            tensor_parallel_size=1,
            dtype="bfloat16",
            max_model_len=context_length,
            block_size=block_size,
            max_num_seqs=1,
            enforce_eager=enforce_eager,
            distributed_executor_backend=distributed_executor_backend,
            gpu_memory_utilization=gpu_memory_utilization,
            async_scheduling=ASYNC_SCHEDULING,
            enable_prefix_caching=enable_prefix_caching,
            kv_transfer_config=kv_transfer_config,
            sparse_attention={
                "cluster_granularity": "token",
                "num_clusters": 64,
                "n_segment": 1,
                "nprobe": 5,
                "max_selected_tokens": 128,
                "static_pattern_end": 16,
                "static_pattern_start": 8,
                "gqa_topk_mode": gqa_topk_mode,
            },
        )
    else:
        engine_args = EngineArgs(
            # 用最小的模型调试，速度快，显存占用小
            model=model_path,
            tensor_parallel_size=1,
            dtype="bfloat16",
            max_model_len=context_length,
            block_size=block_size,
            max_num_seqs=1,
            enforce_eager=enforce_eager,
            distributed_executor_backend=distributed_executor_backend,
            gpu_memory_utilization=gpu_memory_utilization,
            async_scheduling=ASYNC_SCHEDULING,
            enable_prefix_caching=enable_prefix_caching,
            kv_transfer_config=kv_transfer_config,
        )

    # ====================== 2. 初始化引擎（server启动时做的事） ======================
    print("=== 初始化LLMEngine ===")
    engine = None
    try:
        engine = LLMEngine.from_engine_args(engine_args)
        print("✅ 引擎初始化完成\n")
        print(
            "LMCache:",
            "enabled" if USE_LMCACHE else "disabled",
            f"(native={kv_transfer_config.kv_connector_extra_config['use_native']})"
            if kv_transfer_config is not None
            and kv_transfer_config.kv_connector == "LMCacheConnectorV1" else "",
        )
        if kv_transfer_config is not None:
            print(f"KV connector: {kv_transfer_config.kv_connector}")
        print(
            "Debug config:",
            f"enforce_eager={enforce_eager}",
            f"async_scheduling={ASYNC_SCHEDULING}",
            f"v1_multiprocessing={os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']}",
            f"distributed_executor_backend={distributed_executor_backend}",
        )
        print(f"Prefix caching enabled: {enable_prefix_caching}")

        tokenizer = AutoTokenizer.from_pretrained(
            engine_args.model,
            local_files_only=using_local_cache,
        )
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=test_speed,
        )

        all_stats = []
        for request_index in range(request_count):
            request_id = f"debug_req_{request_index + 1:03d}"
            prompt = build_prompt(tokenizer, request_index)
            stats = run_request(
                engine,
                tokenizer,
                request_id,
                prompt,
                sampling_params,
                print_output=print_output,
                profile_decode_only=PROFILE_DECODE_ONLY,
            )
            all_stats.append(stats)
            if request_index + 1 < request_count:
                print("\n=== 准备下一次请求 ===\n")

        if len(all_stats) >= 2:
            first = all_stats[0]
            last = all_stats[-1]
            print("\n=== 请求间对比 ===")
            print(
                f"prefill耗时: {first['prefill_elapsed']:.3f}s -> "
                f"{last['prefill_elapsed']:.3f}s "
                f"(delta={last['prefill_elapsed'] - first['prefill_elapsed']:.3f}s)"
            )
            print(
                f"总耗时: {first['loop_elapsed']:.3f}s -> "
                f"{last['loop_elapsed']:.3f}s "
                f"(delta={last['loop_elapsed'] - first['loop_elapsed']:.3f}s)"
            )
            print(
                f"prefill tokens/s: {first['prefill_tokens_per_second']:.3f} -> "
                f"{last['prefill_tokens_per_second']:.3f}"
            )
            print(
                f"decode tokens/s: {first['decode_tokens_per_second']:.3f} -> "
                f"{last['decode_tokens_per_second']:.3f}"
            )
    finally:
        if engine is not None:
            engine.engine_core.shutdown()
        if (
            USE_LMCACHE
            and kv_transfer_config is not None
            and kv_transfer_config.kv_connector.startswith("LMCache")
        ):
            cleanup_lmcache()
        cleanup_distributed()


if __name__ == "__main__":
    main()