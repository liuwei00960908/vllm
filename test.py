import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time

from huggingface_hub import snapshot_download, try_to_load_from_cache
from huggingface_hub.errors import LocalEntryNotFoundError
import torch
import vllm
from transformers import AutoTokenizer
from vllm import EngineArgs, LLMEngine, SamplingParams

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


def main():
    USE_SPARSE_ATTENTION = os.environ.get("TEST_USE_SPARSE_ATTENTION", "1") != "0"
    PROFILE_DECODE_ONLY = os.environ.get("TEST_PROFILE_DECODE_ONLY", "0") == "1"
    ASYNC_SCHEDULING = os.environ.get("TEST_ASYNC_SCHEDULING", "1") == "1"
    gqa_topk_mode = os.environ.get("TEST_GQA_TOPK_MODE", "group_avg")
    model = "Qwen/Qwen2.5-7B-Instruct"
    context_length = int(os.environ.get("TEST_CONTEXT_LENGTH", "15000"))
    block_size = 16
    test_speed = os.environ.get("TEST_IGNORE_EOS", "1") == "1"
    enforce_eager = os.environ.get("TEST_ENFORCE_EAGER", "0") == "1"
    gpu_memory_utilization = float(
        os.environ.get("TEST_GPU_MEMORY_UTILIZATION", "0.5")
    )
    print_output = False
    model_path, using_local_cache = resolve_model_path(model)
    if gqa_topk_mode not in ("head_union", "group_avg"):
        raise ValueError(
            f"TEST_GQA_TOPK_MODE must be 'head_union' or 'group_avg', got {gqa_topk_mode!r}"
        )

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
            gpu_memory_utilization=gpu_memory_utilization,
            async_scheduling=ASYNC_SCHEDULING,
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
            gpu_memory_utilization=gpu_memory_utilization,
            async_scheduling=ASYNC_SCHEDULING,
        )

    # ====================== 2. 初始化引擎（server启动时做的事） ======================
    print("=== 初始化LLMEngine ===")
    engine = LLMEngine.from_engine_args(engine_args)
    print("✅ 引擎初始化完成\n")

    # ====================== 3. 硬编码请求（模拟server收到HTTP请求） ======================
    request_id = "debug_req_001"  # 字符串类型，和你之前问的一致
    tokenizer = AutoTokenizer.from_pretrained(
        engine_args.model,
        local_files_only=using_local_cache,
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "写一篇描写春天的小作文"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    sampling_params = SamplingParams(
        max_tokens=context_length,
        temperature=0.0,  # 固定输出，方便调试
        top_p=1.0,
        ignore_eos=test_speed,
    )

    print(f"=== 添加请求 {request_id} ===")
    print(f"Prompt: {prompt}")
    print(f"Sampling params: {sampling_params}\n")

    # ✅ server端接收请求后的入口函数！
    # 断点位置1：这里打断点，看请求怎么被处理
    engine.add_request(
        request_id=request_id,
        prompt=prompt,
        params=sampling_params,
    )
    print("✅ 请求已添加到等待队列\n")

    # ====================== 4. 主循环（模拟server的事件循环） ======================
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

        # ✅ 核心！每次step()会执行：
        # 1. Scheduler.schedule() 调度请求
        # 2. KVCacheManager.allocate_slots() 分配KV块
        # 3. GPU执行prefill或decode
        # 4. reshape_and_cache() 写入KV缓存
        # 5. KVCacheManager.cache_blocks() 缓存满块
        # 6. 采样生成token
        # 断点位置2：这里打断点，进入step()看完整流程
        outputs = engine.step()
        if step_count == 1:
            if PROFILE_DECODE_ONLY:
                torch.cuda.synchronize()
            prefill_elapsed = time.perf_counter() - step_start_time
            if PROFILE_DECODE_ONLY:
                decode_range_id = torch.cuda.nvtx.range_start("decode_only")
            decode_start_time = time.perf_counter()

        # 处理输出
        for output in outputs:
            if output.finished:
                print(f"✅ 请求 {output.request_id} 完成")
                generated_text = output.outputs[0].text
            else:
                # 打印本次生成的token
                if print_output:
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

    # ====================== 5. 输出最终结果 ======================
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


if __name__ == "__main__":
    main()