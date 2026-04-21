import vllm
print(vllm.__file__)

from vllm import EngineArgs, LLMEngine, SamplingParams
import torch

def main():
    USE_SPARSE_ATTENTION = True

    # ====================== 1. 引擎配置（和server端完全一致） ======================
    if USE_SPARSE_ATTENTION:
        engine_args = EngineArgs(
            # 用最小的模型调试，速度快，显存占用小
            model="Qwen/Qwen2-0.5B-Instruct",
            tensor_parallel_size=1,
            dtype="bfloat16",
            max_model_len=2048,
            block_size=16, 
            enforce_eager=True,
            gpu_memory_utilization=0.5,
            async_scheduling=False,
            sparse_attention={
                "cluster_granularity": "token",
                "num_clusters": 16,
                "n_segment": 1,
                "nprobe": 64,
                "max_selected_tokens": 128,
                "static_pattern_end": 16,
                "static_pattern_start": 8,
            }
        )
    else:
        engine_args = EngineArgs(
            # 用最小的模型调试，速度快，显存占用小
            model="Qwen/Qwen2-0.5B-Instruct",
            tensor_parallel_size=1,
            dtype="bfloat16",
            max_model_len=2048,
            block_size=16, 
            enforce_eager=True,
            gpu_memory_utilization=0.5,
            async_scheduling=True,
        )

    # ====================== 2. 初始化引擎（server启动时做的事） ======================
    print("=== 初始化LLMEngine ===")
    engine = LLMEngine.from_engine_args(engine_args)
    print("✅ 引擎初始化完成\n")

    # ====================== 3. 硬编码请求（模拟server收到HTTP请求） ======================
    request_id = "debug_req_001"  # 字符串类型，和你之前问的一致
    prompt = "你好，请介绍一下vLLM是什么？"
    sampling_params = SamplingParams(
        max_tokens=200,  # 生成20个token，足够看完整流程
        temperature=0.0,  # 固定输出，方便调试
        top_p=1.0,
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

    while engine.has_unfinished_requests():
        step_count += 1
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

        # 处理输出
        for output in outputs:
            if output.finished:
                print(f"✅ 请求 {output.request_id} 完成")
                generated_text = output.outputs[0].text
            else:
                # 打印本次生成的token
                new_token = output.outputs[0].text
                print(f"生成token: {repr(new_token)}")

    # ====================== 5. 输出最终结果 ======================
    print("\n=== 生成完成 ===")
    print(f"完整输出: {generated_text}")
    print(f"总step数: {step_count}")
    print(f"其中: 第1步 = prefill阶段，第2-{step_count}步 = decode阶段")

if __name__ == "__main__":
    main()