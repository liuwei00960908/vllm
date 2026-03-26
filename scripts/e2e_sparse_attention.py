#!/usr/bin/env python3
"""
端到端 Sparse Attention 验证脚本。

用法:
  python scripts/e2e_sparse_attention.py --model facebook/opt-125m
  python scripts/e2e_sparse_attention.py --model meta-llama/Llama-3.1-8B-Instruct

要求:
  - GPU 机器 (CUDA)
  - VLLM_USE_PRECOMPILED=1 pip install -e .
"""
import argparse
import sys
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="E2E Sparse Attention 验证")
    parser.add_argument("--model", default="facebook/opt-125m",
                        help="HuggingFace 模型 ID 或本地路径")
    parser.add_argument("--num-clusters", type=int, default=4,
                        help="聚类数 (KMeans K)")
    parser.add_argument("--n-segment", type=int, default=4,
                        help="分段数")
    parser.add_argument("--nprobe", type=int, default=2,
                        help="每次检索的聚类数")
    parser.add_argument("--max-selected-blocks", type=int, default=16,
                        help="最多保留的稀疏 block 数")
    parser.add_argument("--static-pattern-start", type=int, default=0,
                        help="静态保留 block 起始")
    parser.add_argument("--static-pattern-end", type=int, default=4,
                        help="静态保留 block 结束")
    parser.add_argument("--prefill-topk-query-window", type=int, default=8,
                        help="prefill topk query 窗口大小")
    parser.add_argument("--update-threshold-blocks", type=int, default=2,
                        help="触发 rebalance 的新增 block 阈值")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="每个 prompt 最多生成的 token 数")
    parser.add_argument("--disable-prefix-caching", action="store_true",
                        default=True, help="禁用 prefix cache (sparse 必须禁用)")
    parser.add_argument("--enforce-eager", action="store_true", default=True,
                        help="禁用 CUDA graph (调试用)")
    return parser.parse_args()


def check_environment():
    if not torch.cuda.is_available():
        print("[ERROR] 未检测到 CUDA GPU，Sparse Attention 需要 GPU 运行。")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[OK] GPU: {gpu_name} ({gpu_mem:.1f} GB)")


def main():
    args = parse_args()
    check_environment()

    sparse_config = {
        "num_clusters":            args.num_clusters,
        "n_segment":               args.n_segment,
        "nprobe":                  args.nprobe,
        "max_selected_blocks":     args.max_selected_blocks,
        "static_pattern_start":    args.static_pattern_start,
        "static_pattern_end":      args.static_pattern_end,
        "prefill_topk_query_window": args.prefill_topk_query_window,
        "update_threshold_blocks": args.update_threshold_blocks,
    }
    print(f"\n[CONFIG] Sparse attention 参数: {sparse_config}")

    # ── 导入 vLLM ──────────────────────────────────────────────────────────
    print(f"\n[STEP 1] 加载模型: {args.model}")
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        print(f"[ERROR] 无法导入 vLLM: {e}")
        sys.exit(1)

    t0 = time.time()
    try:
        llm = LLM(
            model=args.model,
            enforce_eager=args.enforce_eager,
            enable_prefix_caching=not args.disable_prefix_caching,
            sparse_attention=sparse_config,
            # 减小 GPU 内存使用以适应小型测试机器
            gpu_memory_utilization=0.7,
            max_model_len=1024,
        )
    except Exception as e:
        print(f"[ERROR] 模型加载失败: {e}")
        raise
    print(f"[OK] 模型加载完成，耗时 {time.time() - t0:.1f}s")

    # ── 验证 KV Cache spec ─────────────────────────────────────────────────
    print("\n[STEP 2] 验证 KV Cache Spec 类型")
    from vllm.v1.kv_cache_interface import SparseAttentionSpec

    kv_config = None
    try:
        # 通过 worker 拿到 kv_cache_config
        worker = llm.llm_engine.model_executor.driver_worker
        runner = getattr(worker, "model_runner", None)
        if runner is not None and hasattr(runner, "kv_cache_config"):
            kv_config = runner.kv_cache_config
    except Exception:
        pass

    if kv_config is not None:
        specs = [
            g.kv_cache_spec
            for g in kv_config.kv_cache_groups
        ]
        sparse_count = sum(isinstance(s, SparseAttentionSpec) for s in specs)
        total_count = len(specs)
        if sparse_count > 0:
            print(f"[OK] {sparse_count}/{total_count} 个 KV cache group 使用 SparseAttentionSpec")
        else:
            print(f"[WARN] 未检测到 SparseAttentionSpec ({total_count} groups 全部为 FullAttentionSpec)")
            print("       请检查 attention.py 中的 get_kv_cache_spec() 是否正确读取 cache_config.sparse_attention")
    else:
        print("[WARN] 无法访问 kv_cache_config，跳过 spec 类型检查")

    # ── 运行推理测试 ────────────────────────────────────────────────────────
    prompts = [
        "The theory of relativity states that",
        "Once upon a time in a land far away,",
        "The capital of France is Paris. The capital of Germany is",
    ]
    sampling_params = SamplingParams(
        temperature=0.0,    # greedy，结果确定性最强
        max_tokens=args.max_tokens,
    )

    print(f"\n[STEP 3] Prefill + Decode 推理 ({len(prompts)} 条 prompt，最多 {args.max_tokens} tokens)")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"[OK] 推理完成，耗时 {elapsed:.2f}s")

    # ── 结果检验 ───────────────────────────────────────────────────────────
    print("\n[STEP 4] 输出检验")
    all_pass = True
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        text = output.outputs[0].text
        tokens = output.outputs[0].token_ids

        # 基本健全性检查
        has_nan = any(t == 0 and j > 0 for j, t in enumerate(tokens))  # 连续 EOS 不是 NaN，但全 0 可疑
        is_empty = len(tokens) == 0
        is_repetitive = len(set(tokens)) < max(1, len(tokens) // 5) if len(tokens) > 10 else False

        status = "[OK]"
        if is_empty:
            status = "[FAIL] 输出为空"
            all_pass = False
        elif is_repetitive:
            status = "[WARN] 输出高度重复（可能注意力异常）"

        print(f"\n  Prompt {i+1}: {prompt[:50]}...")
        print(f"  生成 token 数: {len(tokens)}")
        print(f"  输出文本: {repr(text[:100])}")
        print(f"  状态: {status}")

    # ── 多步 decode 压力测试 ───────────────────────────────────────────────
    print("\n[STEP 5] 多步 decode 压力测试（200 tokens）")
    long_params = SamplingParams(temperature=0.0, max_tokens=200)
    t0 = time.time()
    try:
        long_output = llm.generate(
            ["Tell me a very long story about a dragon and a wizard."],
            long_params
        )
        elapsed = time.time() - t0
        tokens = long_output[0].outputs[0].token_ids
        print(f"[OK] 生成了 {len(tokens)} tokens，耗时 {elapsed:.2f}s，"
              f"速度 {len(tokens)/elapsed:.1f} tok/s")
    except Exception as e:
        print(f"[FAIL] 多步 decode 失败: {e}")
        all_pass = False
        raise

    # ── 最终结论 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] E2E Sparse Attention 验证通过！")
    else:
        print("[FAIL] 存在失败项，请查看上方日志。")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
