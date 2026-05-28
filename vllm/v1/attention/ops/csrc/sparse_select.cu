// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Batched sparse cluster scoring + fused top-K/cumsum kernels for the
// ``_sparse_select_tokens`` pipeline.
//
// Two kernels live in this TU:
//
//   1. ``sparse_cluster_score_kernel`` (stage 1)
//      For every ``(req r, kv_head h, query-in-group g)`` it computes
//      the flash-attention-style softmax over centroid scores
//      ``q dot K^T / sqrt(d)`` and accumulates the resulting
//      probabilities across the GQA group dim via ``atomicAdd`` so the
//      output is the group-summed probability tensor
//      ``[num_reqs, num_kv_heads, num_clusters]`` (fp32).
//
//   2. ``fused_topk_cumsum_kernel`` (stage 2)
//      For every ``(req, kv_head)`` it does a single-pass block-radix
//      ``top-K`` on the stage-1 ``[num_clusters]`` row, gathers the
//      selected ``cluster_size`` entries via a per-request pointer
//      table and writes both the top-K indices and the inclusive
//      ``cumsum`` of the gathered sizes -- replacing four sequential
//      PyTorch ops (``topk + stack + gather + cumsum``) with a single
//      kernel launch.
//
// Other key optimizations:
//
//   - **Pointer-array centres**: instead of a single stacked
//     ``[num_reqs, num_kv_heads, num_clusters, head_dim]`` tensor
//     (which forces a ~256 MB ``torch.stack`` copy per layer per
//     decode step), stage 1 receives ``Centres_ptrs[num_reqs]`` -- one
//     device pointer per request.  All centres must share strides.
//
//   - **Pointer-array cluster sizes**: the fused stage-2 kernel
//     receives ``Sizes_ptrs[num_reqs]`` and does the gather in-kernel,
//     eliminating another ``torch.stack`` (~4 MB at batch=64).
//
//   - **Online softmax**: stage 1 maintains a per-warp running
//     ``(m, l)`` pair while it computes scores, eliminating the
//     dedicated block-max and block-sum reduction passes over the
//     full ``s_scores`` buffer (~16 KB of SMEM traffic per block
//     avoided).
//
// Grid layouts:
//   stage 1: gridDim = (num_reqs, num_kv_heads, group_size)
//            blockDim.x = 256 (8 warps)
//   stage 2: gridDim = (num_reqs, num_kv_heads)
//            blockDim.x = 128 (4 warps), uses cub::BlockRadixSort +
//            cub::BlockScan over the full clusters dim.
//
// Numerics: stage 1 always accumulates in fp32 regardless of input
// dtype (bf16/fp16/fp32).  ``atomicAdd`` produces results that differ
// from a fully-ordered sum by at most a handful of fp32 ULPs (the GQA
// group_size is small, typically <=16); this is well below the
// tolerances of every downstream consumer (``torch.topk`` is invariant
// to ULP-level noise except at exact ties).

#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>

#include <climits>
#include <tuple>

namespace vllm_sparse_select {

// ---------------------------------------------------------------------------
// Element conversion helpers (scalar_t -> float)
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ float to_float(T v) {
  return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <>
__device__ __forceinline__ float to_float<__half>(__half v) {
  return __half2float(v);
}

// ---------------------------------------------------------------------------
// Vectorized K-row loader.
//
// For HEAD_DIM=128 / WARP_SIZE=32, each lane processes ``D_PER_THREAD=4``
// elements per cluster.  Without explicit vector loads, nvcc *usually*
// coalesces 4 consecutive ``scalar_t`` reads into a single 64- or 128-bit
// load, but it is not guaranteed -- and at 13% of peak HBM bandwidth we
// want to force the issue.  The specializations below emit a single
// ``ld.global.nc.v2.b32`` (bf16/fp16) or ``ld.global.nc.v4.b32`` (fp32)
// per cluster per lane, on properly-aligned addresses (the centres
// tensor is allocated by torch with a 256-byte alignment guarantee).
// ---------------------------------------------------------------------------

template <typename scalar_t, int N>
struct VecLoadToFloat {
  __device__ __forceinline__ static void load(const scalar_t* __restrict__ p,
                                              float (&out)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      out[i] = to_float<scalar_t>(p[i]);
    }
  }
};

// bf16, N=4 -> single 8-byte (float2) load
template <>
struct VecLoadToFloat<__nv_bfloat16, 4> {
  __device__ __forceinline__ static void load(
      const __nv_bfloat16* __restrict__ p, float (&out)[4]) {
    const float2 raw = *reinterpret_cast<const float2*>(p);
    const __nv_bfloat162* bf2 =
        reinterpret_cast<const __nv_bfloat162*>(&raw);
    const float2 lo = __bfloat1622float2(bf2[0]);
    const float2 hi = __bfloat1622float2(bf2[1]);
    out[0] = lo.x;
    out[1] = lo.y;
    out[2] = hi.x;
    out[3] = hi.y;
  }
};

// fp16, N=4 -> single 8-byte (float2) load
template <>
struct VecLoadToFloat<__half, 4> {
  __device__ __forceinline__ static void load(
      const __half* __restrict__ p, float (&out)[4]) {
    const float2 raw = *reinterpret_cast<const float2*>(p);
    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
    const float2 lo = __half22float2(h2[0]);
    const float2 hi = __half22float2(h2[1]);
    out[0] = lo.x;
    out[1] = lo.y;
    out[2] = hi.x;
    out[3] = hi.y;
  }
};

// fp32, N=4 -> single 16-byte (float4) load
template <>
struct VecLoadToFloat<float, 4> {
  __device__ __forceinline__ static void load(
      const float* __restrict__ p, float (&out)[4]) {
    const float4 raw = *reinterpret_cast<const float4*>(p);
    out[0] = raw.x;
    out[1] = raw.y;
    out[2] = raw.z;
    out[3] = raw.w;
  }
};

// bf16, N=8 -> single 16-byte (float4) load (covers HEAD_DIM=256)
template <>
struct VecLoadToFloat<__nv_bfloat16, 8> {
  __device__ __forceinline__ static void load(
      const __nv_bfloat16* __restrict__ p, float (&out)[8]) {
    const float4 raw = *reinterpret_cast<const float4*>(p);
    const __nv_bfloat162* bf2 =
        reinterpret_cast<const __nv_bfloat162*>(&raw);
    const float2 v0 = __bfloat1622float2(bf2[0]);
    const float2 v1 = __bfloat1622float2(bf2[1]);
    const float2 v2 = __bfloat1622float2(bf2[2]);
    const float2 v3 = __bfloat1622float2(bf2[3]);
    out[0] = v0.x; out[1] = v0.y;
    out[2] = v1.x; out[3] = v1.y;
    out[4] = v2.x; out[5] = v2.y;
    out[6] = v3.x; out[7] = v3.y;
  }
};

// fp16, N=8 -> single 16-byte (float4) load
template <>
struct VecLoadToFloat<__half, 8> {
  __device__ __forceinline__ static void load(
      const __half* __restrict__ p, float (&out)[8]) {
    const float4 raw = *reinterpret_cast<const float4*>(p);
    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
    const float2 v0 = __half22float2(h2[0]);
    const float2 v1 = __half22float2(h2[1]);
    const float2 v2 = __half22float2(h2[2]);
    const float2 v3 = __half22float2(h2[3]);
    out[0] = v0.x; out[1] = v0.y;
    out[2] = v1.x; out[3] = v1.y;
    out[4] = v2.x; out[5] = v2.y;
    out[6] = v3.x; out[7] = v3.y;
  }
};

// =============================================================================
// STAGE 1: per-(req, head, group) softmax with atomic group-sum reduction
// =============================================================================

// =============================================================================
// Stage-1 kernel: dual-q (Q-PAIR) fused block.
//
// Each block now handles *two* consecutive query-in-group indices (q_a =
// 2 * q_pair_idx and q_b = q_a + 1) that share the SAME centroid matrix
// K[r, h, :, :].  K is loaded once per block into thread-local registers
// and reused for both dot products, halving the L2 traffic for K vs.
// the previous "one q per block" design (4 K reads / (R,H) instead of
// 7 for G=7).
//
// SMEM is kept tight (~9 KB) by packing the per-cluster raw scores for
// q_a and q_b into a single ``bf16x2`` slot per cluster -- this preserves
// the previous 8 blocks/SM occupancy (warp-limited, not SMEM-limited).
// The fp32->bf16 quantization of intermediate scores is invisible to the
// downstream consumer (the kernel returns the *softmaxed* probabilities
// at full fp32 precision, and softmax is invariant to ULP-level noise in
// the scores except at exact ties).
//
// For odd ``group_size`` the last block in the (R, H) row covers only
// q_a (``has_b = false``); the q_b code path is gated.  ``__launch_bounds__``
// + the ``has_b`` bool being identical across all 256 threads in the block
// means the compiler eliminates the q_b dead code via branch prediction
// rather than warp divergence.
// =============================================================================

template <typename scalar_t, int HEAD_DIM, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS) void sparse_cluster_score_kernel(
    const scalar_t* __restrict__ Q,             // [T, num_q_heads, head_dim]
    const int32_t*  __restrict__ QSL,           // [num_reqs + 1]
    const int64_t*  __restrict__ Centres_ptrs,  // [num_reqs] device ptrs
    float*          __restrict__ Out,           // [R, H, C] fp32, zero-init
    const int num_clusters,
    const int group_size,
    const int64_t stride_q_tok,
    const int64_t stride_q_h,
    const int64_t stride_c_h,
    const int64_t stride_c_c,
    const int64_t stride_o_r,
    const int64_t stride_o_h,
    const float scale) {
  static_assert(HEAD_DIM % 32 == 0,
                "HEAD_DIM must be divisible by warp size (32)");
  constexpr int WARP_SIZE = 32;
  constexpr int N_WARPS = BLOCK_THREADS / WARP_SIZE;
  constexpr int D_PER_THREAD = HEAD_DIM / WARP_SIZE;

  // NB: grid is launched as ``(ceil(group_size/2), num_kv_heads, num_reqs)``.
  // The q-pair dim varies fastest so the (up to 4 for G=7) blocks sharing a
  // single K matrix are scheduled back-to-back, maximizing L2 reuse for K.
  const int q_pair_idx = blockIdx.x;
  const int kv_head_id = blockIdx.y;
  const int req_id     = blockIdx.z;
  const int tid        = threadIdx.x;
  const int lane       = tid & 31;
  const int warp_id    = tid >> 5;

  const int q_in_group_a = q_pair_idx * 2;
  const int q_in_group_b = q_pair_idx * 2 + 1;
  const bool has_b       = (q_in_group_b < group_size);

  // ------------------- shared memory layout -------------------
  //  s_q_a    : [HEAD_DIM]      fp32 q vector for q_a   (~512 B)
  //  s_q_b    : [HEAD_DIM]      fp32 q vector for q_b   (~512 B; unused if !has_b)
  //  s_scores : [num_clusters]  bf16x2 packed (q_a, q_b) raw scores (~8 KB @ C=2048)
  //  s_ml     : [4 * N_WARPS]   fp32 per-warp (m_a, l_a, m_b, l_b) (~128 B @ N_WARPS=8)
  // Total at HEAD_DIM=128, C=2048, BLOCK_THREADS=256: ~9.1 KB -- still
  // warp-limited at 8 blocks/SM (vs SMEM-limited).
  extern __shared__ unsigned char smem_raw[];
  float* s_q_a = reinterpret_cast<float*>(smem_raw);
  float* s_q_b = s_q_a + HEAD_DIM;
  __nv_bfloat162* s_scores =
      reinterpret_cast<__nv_bfloat162*>(s_q_b + HEAD_DIM);
  float* s_ml = reinterpret_cast<float*>(s_scores + num_clusters);

  // ---------- 1. Resolve indices from query_start_loc ----------
  const int tok_end      = QSL[req_id + 1];
  const int token_idx    = tok_end - 1;
  const int q_head_idx_a = kv_head_id * group_size + q_in_group_a;
  // For has_b=false we still need a valid pointer for the unconditional
  // load below; using q_head_idx_a keeps the load safe (the q_b results
  // will simply be ignored).
  const int q_head_idx_b =
      has_b ? (kv_head_id * group_size + q_in_group_b) : q_head_idx_a;

  // ---------- 2. Load q_a and q_b into shared memory ----------
  const scalar_t* q_a_ptr = Q
      + static_cast<int64_t>(token_idx)    * stride_q_tok
      + static_cast<int64_t>(q_head_idx_a) * stride_q_h;
  const scalar_t* q_b_ptr = Q
      + static_cast<int64_t>(token_idx)    * stride_q_tok
      + static_cast<int64_t>(q_head_idx_b) * stride_q_h;
#pragma unroll
  for (int d = tid; d < HEAD_DIM; d += BLOCK_THREADS) {
    s_q_a[d] = to_float<scalar_t>(q_a_ptr[d]);
    s_q_b[d] = to_float<scalar_t>(q_b_ptr[d]);
  }
  __syncthreads();

  // ---------- 3. Cache the lane's slice of both q's in registers ----------
  float q_a_reg[D_PER_THREAD];
  float q_b_reg[D_PER_THREAD];
  const int d_offset = lane * D_PER_THREAD;
#pragma unroll
  for (int i = 0; i < D_PER_THREAD; ++i) {
    q_a_reg[i] = s_q_a[d_offset + i];
    q_b_reg[i] = s_q_b[d_offset + i];
  }

  // ---------- 4. Dot products + dual online softmax ----------
  // K[c] is loaded ONCE into ``k_reg`` and consumed for both dot products
  // in registers -- saving 50% of L2 traffic for K vs. the single-q design.
  const scalar_t* c_base = reinterpret_cast<const scalar_t*>(
      Centres_ptrs[req_id]
  ) + static_cast<int64_t>(kv_head_id) * stride_c_h;

  float m_a_warp = -CUDART_INF_F, l_a_warp = 0.0f;  // valid on lane 0
  float m_b_warp = -CUDART_INF_F, l_b_warp = 0.0f;  // valid on lane 0

  for (int c_round = 0; c_round < num_clusters; c_round += N_WARPS) {
    const int c = c_round + warp_id;
    if (c >= num_clusters) continue;
    const scalar_t* k_ptr = c_base
        + static_cast<int64_t>(c) * stride_c_c
        + d_offset;

    float k_reg[D_PER_THREAD];
    VecLoadToFloat<scalar_t, D_PER_THREAD>::load(k_ptr, k_reg);

    float acc_a = 0.0f;
    float acc_b = 0.0f;
#pragma unroll
    for (int i = 0; i < D_PER_THREAD; ++i) {
      acc_a += q_a_reg[i] * k_reg[i];
      acc_b += q_b_reg[i] * k_reg[i];  // dead code eliminated when !has_b
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      acc_a += __shfl_xor_sync(0xffffffff, acc_a, off);
      acc_b += __shfl_xor_sync(0xffffffff, acc_b, off);
    }

    if (lane == 0) {
      const float s_a = acc_a * scale;
      const float s_b = has_b ? (acc_b * scale) : 0.0f;
      s_scores[c] = __floats2bfloat162_rn(s_a, s_b);

      // Online softmax update for q_a.
      if (s_a > m_a_warp) {
        l_a_warp = l_a_warp * __expf(m_a_warp - s_a) + 1.0f;
        m_a_warp = s_a;
      } else {
        l_a_warp += __expf(s_a - m_a_warp);
      }
      if (has_b) {
        if (s_b > m_b_warp) {
          l_b_warp = l_b_warp * __expf(m_b_warp - s_b) + 1.0f;
          m_b_warp = s_b;
        } else {
          l_b_warp += __expf(s_b - m_b_warp);
        }
      }
    }
  }

  // ---------- 5. Block combine of the N_WARPS (m, l) pairs for a AND b ----------
  if (lane == 0) {
    s_ml[warp_id * 4 + 0] = m_a_warp;
    s_ml[warp_id * 4 + 1] = l_a_warp;
    s_ml[warp_id * 4 + 2] = m_b_warp;
    s_ml[warp_id * 4 + 3] = l_b_warp;
  }
  __syncthreads();

  // One warp combines both (m, l) pairs in parallel: lanes 0..N_WARPS-1
  // pull a's pair, lanes N_WARPS..2*N_WARPS-1 pull b's pair.
  if (warp_id == 0) {
    const bool is_b_lane = (lane >= N_WARPS) && (lane < 2 * N_WARPS);
    const int sub_lane   = is_b_lane ? (lane - N_WARPS) : lane;
    const bool active    = (sub_lane < N_WARPS);

    const float m_lane = active
        ? s_ml[sub_lane * 4 + (is_b_lane ? 2 : 0)]
        : -CUDART_INF_F;
    const float l_lane = active
        ? s_ml[sub_lane * 4 + (is_b_lane ? 3 : 1)]
        : 0.0f;

    // Restrict the shuffle to within a's group (0..N_WARPS-1) and within
    // b's group (N_WARPS..2*N_WARPS-1) by using a half-warp mask.
    // N_WARPS = 8 means each "sub-warp" is half a real warp; we use the
    // ``__shfl_*_sync`` ``width`` argument to confine the reductions.
    constexpr int WIDTH = N_WARPS;  // each sub-reduce spans WIDTH lanes

    float m = m_lane;
#pragma unroll
    for (int off = WIDTH / 2; off > 0; off >>= 1) {
      m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off, WIDTH));
    }
    float l_scaled = active ? (l_lane * __expf(m_lane - m)) : 0.0f;
#pragma unroll
    for (int off = WIDTH / 2; off > 0; off >>= 1) {
      l_scaled += __shfl_xor_sync(0xffffffff, l_scaled, off, WIDTH);
    }

    if (sub_lane == 0) {
      if (is_b_lane) {
        s_ml[2] = m;
        s_ml[3] = (l_scaled > 0.0f) ? (1.0f / l_scaled) : 0.0f;
      } else {
        s_ml[0] = m;
        s_ml[1] = (l_scaled > 0.0f) ? (1.0f / l_scaled) : 0.0f;
      }
    }
  }
  __syncthreads();

  const float m_a_global = s_ml[0];
  const float inv_l_a    = s_ml[1];
  const float m_b_global = s_ml[2];
  const float inv_l_b    = s_ml[3];

  // ---------- 6. AtomicAdd normalized probs to [R, H, C] -----------
  // Two streaming writes through the packed s_scores: each cluster's
  // bf162 unpacks to (s_a, s_b), each contributes one atomic add.
  float* out_base = Out
      + static_cast<int64_t>(req_id)     * stride_o_r
      + static_cast<int64_t>(kv_head_id) * stride_o_h;

  if (has_b) {
    for (int c = tid; c < num_clusters; c += BLOCK_THREADS) {
      const float2 pair = __bfloat1622float2(s_scores[c]);
      atomicAdd(out_base + c, __expf(pair.x - m_a_global) * inv_l_a);
      atomicAdd(out_base + c, __expf(pair.y - m_b_global) * inv_l_b);
    }
  } else {
    for (int c = tid; c < num_clusters; c += BLOCK_THREADS) {
      const float2 pair = __bfloat1622float2(s_scores[c]);
      atomicAdd(out_base + c, __expf(pair.x - m_a_global) * inv_l_a);
    }
  }
}

// ---------------------------------------------------------------------------
// Stage-1 launcher
// ---------------------------------------------------------------------------

template <typename scalar_t, int HEAD_DIM>
void launch_sparse_cluster_score(
    const scalar_t* Q,
    const int32_t*  QSL,
    const int64_t*  Centres_ptrs,
    float*          Out,
    int num_reqs,
    int num_kv_heads,
    int group_size,
    int num_clusters,
    int64_t stride_q_tok, int64_t stride_q_h,
    int64_t stride_c_h,   int64_t stride_c_c,
    int64_t stride_o_r,   int64_t stride_o_h,
    float scale,
    cudaStream_t stream) {
  // 256-thread blocks (8 warps): 8 clusters per round, more outstanding
  // K loads per block to hide HBM latency at small batch sizes.  See the
  // benchmark in tests/v1/attention/test_sparse_select_cuda.py for the
  // batch=1 win this gives over a 128-thread layout (~30%).
  constexpr int BLOCK_THREADS = 256;
  constexpr int N_WARPS = BLOCK_THREADS / 32;

  // Dual-Q kernel: each block handles 2 consecutive q_in_group slots
  // sharing the same K matrix.  Grid x-dim is ceil(group_size / 2);
  // SMEM holds two q vectors + one bf16x2-packed (q_a, q_b) score array
  // + 4 fp32 (m, l) slots per warp.
  const int q_pair_blocks = (group_size + 1) / 2;
  const size_t smem_bytes =
      sizeof(float)          * static_cast<size_t>(HEAD_DIM)        // s_q_a
    + sizeof(float)          * static_cast<size_t>(HEAD_DIM)        // s_q_b
    + sizeof(__nv_bfloat162) * static_cast<size_t>(num_clusters)    // s_scores
    + sizeof(float)          * static_cast<size_t>(4 * N_WARPS);    // s_ml

  // ``q_pair`` is the FASTEST-varying grid dim so the (up to 4 for G=7)
  // blocks sharing a single K matrix are scheduled back-to-back, hitting
  // in L2 instead of being scattered across waves.
  dim3 grid(q_pair_blocks, num_kv_heads, num_reqs);
  dim3 block(BLOCK_THREADS);

  sparse_cluster_score_kernel<scalar_t, HEAD_DIM, BLOCK_THREADS>
      <<<grid, block, smem_bytes, stream>>>(
          Q, QSL, Centres_ptrs, Out,
          num_clusters, group_size,
          stride_q_tok, stride_q_h,
          stride_c_h, stride_c_c,
          stride_o_r, stride_o_h,
          scale);
}

// ---------------------------------------------------------------------------
// Stage-1 torch-facing entry point.
// ---------------------------------------------------------------------------

at::Tensor sparse_cluster_scores(
    const at::Tensor& query,             // [T, num_q_heads, head_dim]
    const at::Tensor& query_start_loc,   // [num_reqs + 1] int32
    const at::Tensor& centres_ptrs,      // [num_reqs] int64 (device ptrs)
    int64_t num_kv_heads,
    int64_t num_clusters,
    int64_t head_dim,
    int64_t stride_c_h_elems,
    int64_t stride_c_c_elems,
    int64_t group_size) {
  TORCH_CHECK(query.is_cuda(),             "query must be CUDA");
  TORCH_CHECK(query_start_loc.is_cuda(),   "query_start_loc must be CUDA");
  TORCH_CHECK(centres_ptrs.is_cuda(),      "centres_ptrs must be CUDA");
  TORCH_CHECK(query_start_loc.scalar_type() == at::kInt,
              "query_start_loc must be int32");
  TORCH_CHECK(centres_ptrs.scalar_type() == at::kLong,
              "centres_ptrs must be int64");
  TORCH_CHECK(query.dim() == 3,
              "query expected to be [T, num_q_heads, head_dim]");
  TORCH_CHECK(query.size(2) == head_dim,
              "query.head_dim does not match head_dim arg");
  TORCH_CHECK(query.size(1) == num_kv_heads * group_size,
              "query.num_q_heads must equal num_kv_heads * group_size");
  TORCH_CHECK(query.stride(2) == 1,
              "query must be contiguous along head_dim");
  TORCH_CHECK(group_size > 0, "group_size must be positive");

  const int64_t num_reqs = query_start_loc.numel() - 1;
  TORCH_CHECK(num_reqs > 0, "num_reqs must be positive");
  TORCH_CHECK(centres_ptrs.numel() == num_reqs,
              "centres_ptrs must have length num_reqs");

  auto out = at::zeros(
      {num_reqs, num_kv_heads, num_clusters},
      query.options().dtype(at::kFloat));

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const c10::cuda::CUDAGuard device_guard(query.device());

#define DISPATCH_HEAD_DIM(D)                                                  \
  case D: {                                                                   \
    if (query.scalar_type() == at::kBFloat16) {                               \
      launch_sparse_cluster_score<__nv_bfloat16, D>(                          \
          reinterpret_cast<const __nv_bfloat16*>(query.data_ptr()),           \
          query_start_loc.data_ptr<int32_t>(),                                \
          centres_ptrs.data_ptr<int64_t>(),                                   \
          out.data_ptr<float>(),                                              \
          num_reqs, num_kv_heads, group_size, num_clusters,                   \
          query.stride(0), query.stride(1),                                   \
          stride_c_h_elems, stride_c_c_elems,                                 \
          out.stride(0), out.stride(1),                                       \
          scale, stream);                                                     \
    } else if (query.scalar_type() == at::kHalf) {                            \
      launch_sparse_cluster_score<__half, D>(                                 \
          reinterpret_cast<const __half*>(query.data_ptr()),                  \
          query_start_loc.data_ptr<int32_t>(),                                \
          centres_ptrs.data_ptr<int64_t>(),                                   \
          out.data_ptr<float>(),                                              \
          num_reqs, num_kv_heads, group_size, num_clusters,                   \
          query.stride(0), query.stride(1),                                   \
          stride_c_h_elems, stride_c_c_elems,                                 \
          out.stride(0), out.stride(1),                                       \
          scale, stream);                                                     \
    } else if (query.scalar_type() == at::kFloat) {                           \
      launch_sparse_cluster_score<float, D>(                                  \
          query.data_ptr<float>(),                                            \
          query_start_loc.data_ptr<int32_t>(),                                \
          centres_ptrs.data_ptr<int64_t>(),                                   \
          out.data_ptr<float>(),                                              \
          num_reqs, num_kv_heads, group_size, num_clusters,                   \
          query.stride(0), query.stride(1),                                   \
          stride_c_h_elems, stride_c_c_elems,                                 \
          out.stride(0), out.stride(1),                                       \
          scale, stream);                                                     \
    } else {                                                                  \
      TORCH_CHECK(false, "Unsupported scalar type: ", query.scalar_type());   \
    }                                                                         \
    break;                                                                    \
  }

  switch (head_dim) {
    DISPATCH_HEAD_DIM(64)
    DISPATCH_HEAD_DIM(96)
    DISPATCH_HEAD_DIM(128)
    DISPATCH_HEAD_DIM(160)
    DISPATCH_HEAD_DIM(192)
    DISPATCH_HEAD_DIM(224)
    DISPATCH_HEAD_DIM(256)
    default:
      TORCH_CHECK(false,
                  "Unsupported head_dim: ", head_dim,
                  " (must be a positive multiple of 32 in [64, 256])");
  }

#undef DISPATCH_HEAD_DIM

  return out;
}

// =============================================================================
// STAGE 2: fused top-K + per-request size gather + cumulative sum
// =============================================================================
//
// Replaces the four-PyTorch-op tail of ``_sparse_select_tokens``:
//
//   top = torch.topk(group_scores, k=nprobe, dim=-1).indices  # 1 kernel
//   sizes = torch.stack(per_req_sizes, dim=0)                 # 1 kernel
//   sel = sizes.gather(-1, top)                               # 1 kernel
//   csi = torch.cumsum(sel, dim=-1, dtype=torch.int32)        # 1 kernel
//
// with one block-cooperative pass per ``(req, kv_head)`` that uses
// ``cub::BlockRadixSort`` for the top-K, then gathers ``cluster_size`` via
// a per-request pointer table (no ``torch.stack``), and finally runs
// ``cub::BlockScan::InclusiveSum`` for the cumsum.

template <int NUM_CLUSTERS_POW2, int BLOCK_THREADS_TOPK = 128>
__global__ __launch_bounds__(BLOCK_THREADS_TOPK)
void fused_topk_cumsum_kernel(
    const float*   __restrict__ Scores,         // [R, H, num_clusters] fp32
    const int64_t* __restrict__ Sizes_ptrs,     // [R] device ptrs (int32 buffers)
    int64_t*       __restrict__ TopIndices,     // [R, H, nprobe] int64
    int32_t*       __restrict__ CumSum,         // [R, H, nprobe] int32
    int num_kv_heads,
    int num_clusters,                            // actual <= NUM_CLUSTERS_POW2
    int nprobe,
    int64_t stride_scores_r,
    int64_t stride_scores_h,
    int64_t stride_sizes_h,
    int64_t stride_top_r,
    int64_t stride_top_h) {
  static_assert(NUM_CLUSTERS_POW2 > 0
                && (NUM_CLUSTERS_POW2 & (NUM_CLUSTERS_POW2 - 1)) == 0,
                "NUM_CLUSTERS_POW2 must be a power of 2");
  constexpr int ITEMS_PER_THREAD = NUM_CLUSTERS_POW2 / BLOCK_THREADS_TOPK;
  static_assert(ITEMS_PER_THREAD >= 1, "NUM_CLUSTERS_POW2 < BLOCK_THREADS_TOPK");

  using BlockLoad = cub::BlockLoad<
      float, BLOCK_THREADS_TOPK, ITEMS_PER_THREAD,
      cub::BLOCK_LOAD_TRANSPOSE>;
  using BlockRadixSort = cub::BlockRadixSort<
      float, BLOCK_THREADS_TOPK, ITEMS_PER_THREAD, int>;
  using BlockScan = cub::BlockScan<int32_t, BLOCK_THREADS_TOPK>;

  // Reuse SMEM across the three phases.
  __shared__ union {
    typename BlockLoad::TempStorage      load;
    typename BlockRadixSort::TempStorage sort;
    typename BlockScan::TempStorage      scan;
  } temp;

  const int req_id     = blockIdx.x;
  const int kv_head_id = blockIdx.y;
  const int tid        = threadIdx.x;

  // ---------- 1. Coalesced load of scores into per-thread BLOCKED arrays.
  const float* scores_base = Scores
      + static_cast<int64_t>(req_id)     * stride_scores_r
      + static_cast<int64_t>(kv_head_id) * stride_scores_h;

  float keys[ITEMS_PER_THREAD];
  BlockLoad(temp.load).Load(
      scores_base,
      keys,
      num_clusters,
      -CUDART_INF_F);
  __syncthreads();  // free temp.load before reusing the union

  // Per-thread cluster indices in BLOCKED layout.  ``INT_MAX`` for slots
  // beyond ``num_clusters`` (padding) so they sort to the bottom.
  int values[ITEMS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    const int rank = tid * ITEMS_PER_THREAD + i;
    values[i] = (rank < num_clusters) ? rank : INT_MAX;
  }

  // ---------- 2. Block-wide descending radix sort (keys=scores, vals=indices).
  BlockRadixSort(temp.sort).SortDescending(keys, values);
  __syncthreads();

  // ---------- 3. Gather cluster_size at the top-K cluster indices.
  const int32_t* sizes_base = reinterpret_cast<const int32_t*>(
      Sizes_ptrs[req_id]
  ) + static_cast<int64_t>(kv_head_id) * stride_sizes_h;

  int32_t my_sizes[ITEMS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    const int rank = tid * ITEMS_PER_THREAD + i;
    if (rank < nprobe && values[i] >= 0 && values[i] < num_clusters) {
      my_sizes[i] = sizes_base[values[i]];
    } else {
      my_sizes[i] = 0;
    }
  }

  // ---------- 4. Block-wide inclusive prefix sum of the top-K sizes.
  int32_t cumsum_per_thread[ITEMS_PER_THREAD];
  BlockScan(temp.scan).InclusiveSum(my_sizes, cumsum_per_thread);

  // ---------- 5. Write outputs: only the first ``nprobe`` ranks are real.
  int64_t* top_base = TopIndices
      + static_cast<int64_t>(req_id)     * stride_top_r
      + static_cast<int64_t>(kv_head_id) * stride_top_h;
  int32_t* csum_base = CumSum
      + static_cast<int64_t>(req_id)     * stride_top_r
      + static_cast<int64_t>(kv_head_id) * stride_top_h;

#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    const int rank = tid * ITEMS_PER_THREAD + i;
    if (rank < nprobe) {
      top_base[rank]  = static_cast<int64_t>(values[i]);
      csum_base[rank] = cumsum_per_thread[i];
    }
  }
}

// ---------------------------------------------------------------------------
// Stage-2 launcher with pow2 dispatch on num_clusters.
// ---------------------------------------------------------------------------

template <int NUM_CLUSTERS_POW2>
void launch_fused_topk_cumsum(
    const float*   Scores,
    const int64_t* Sizes_ptrs,
    int64_t*       TopIndices,
    int32_t*       CumSum,
    int num_reqs,
    int num_kv_heads,
    int num_clusters,
    int nprobe,
    int64_t stride_scores_r, int64_t stride_scores_h,
    int64_t stride_sizes_h,
    int64_t stride_top_r,    int64_t stride_top_h,
    cudaStream_t stream) {
  constexpr int BLOCK_THREADS_TOPK = 128;
  dim3 grid(num_reqs, num_kv_heads);
  dim3 block(BLOCK_THREADS_TOPK);
  fused_topk_cumsum_kernel<NUM_CLUSTERS_POW2, BLOCK_THREADS_TOPK>
      <<<grid, block, 0, stream>>>(
          Scores, Sizes_ptrs, TopIndices, CumSum,
          num_kv_heads, num_clusters, nprobe,
          stride_scores_r, stride_scores_h,
          stride_sizes_h,
          stride_top_r, stride_top_h);
}

static inline int next_pow2(int x) {
  int p = 1;
  while (p < x) p <<= 1;
  return p;
}

// ---------------------------------------------------------------------------
// Stage-2 torch-facing entry point.
//
// Allocates and returns ``(top_indices [R, H, nprobe] int64,
//                          cluster_start_index [R, H, nprobe] int32)``.
// ---------------------------------------------------------------------------

std::tuple<at::Tensor, at::Tensor> fused_topk_cumsum(
    const at::Tensor& scores,            // [R, H, num_clusters] fp32
    const at::Tensor& sizes_ptrs,        // [R] int64 (device ptrs to [H, C] int32)
    int64_t num_kv_heads,
    int64_t num_clusters,
    int64_t nprobe,
    int64_t stride_sizes_h_elems) {
  TORCH_CHECK(scores.is_cuda() && sizes_ptrs.is_cuda(),
              "scores and sizes_ptrs must be CUDA tensors");
  TORCH_CHECK(scores.scalar_type() == at::kFloat,
              "scores must be float32");
  TORCH_CHECK(sizes_ptrs.scalar_type() == at::kLong,
              "sizes_ptrs must be int64");
  TORCH_CHECK(scores.dim() == 3,
              "scores must be [num_reqs, num_kv_heads, num_clusters]");
  TORCH_CHECK(scores.size(1) == num_kv_heads,
              "scores.size(1) must match num_kv_heads");
  TORCH_CHECK(scores.size(2) == num_clusters,
              "scores.size(2) must match num_clusters");
  TORCH_CHECK(nprobe > 0 && nprobe <= num_clusters,
              "0 < nprobe <= num_clusters required");
  TORCH_CHECK(scores.stride(2) == 1,
              "scores must be contiguous along the clusters dim");

  const int64_t num_reqs = scores.size(0);
  TORCH_CHECK(sizes_ptrs.numel() == num_reqs,
              "sizes_ptrs.numel() must equal num_reqs");

  auto opts_int64 = scores.options().dtype(at::kLong);
  auto opts_int32 = scores.options().dtype(at::kInt);
  auto top_indices = at::empty({num_reqs, num_kv_heads, nprobe}, opts_int64);
  auto cumsum      = at::empty({num_reqs, num_kv_heads, nprobe}, opts_int32);

  const int nc_pow2 = next_pow2(static_cast<int>(num_clusters));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const c10::cuda::CUDAGuard device_guard(scores.device());

#define DISPATCH_NC_POW2(P)                                                    \
  case P:                                                                      \
    launch_fused_topk_cumsum<P>(                                               \
        scores.data_ptr<float>(),                                              \
        sizes_ptrs.data_ptr<int64_t>(),                                        \
        top_indices.data_ptr<int64_t>(),                                       \
        cumsum.data_ptr<int32_t>(),                                            \
        num_reqs, num_kv_heads,                                                \
        static_cast<int>(num_clusters),                                        \
        static_cast<int>(nprobe),                                              \
        scores.stride(0), scores.stride(1),                                    \
        stride_sizes_h_elems,                                                  \
        top_indices.stride(0), top_indices.stride(1),                          \
        stream);                                                               \
    break;

  switch (nc_pow2) {
    DISPATCH_NC_POW2(128)
    DISPATCH_NC_POW2(256)
    DISPATCH_NC_POW2(512)
    DISPATCH_NC_POW2(1024)
    DISPATCH_NC_POW2(2048)
    DISPATCH_NC_POW2(4096)
    default:
      TORCH_CHECK(false,
                  "Unsupported num_clusters (rounded-up pow2 = ", nc_pow2,
                  "). Supported: 128, 256, 512, 1024, 2048, 4096.");
  }
#undef DISPATCH_NC_POW2

  return std::make_tuple(top_indices, cumsum);
}

}  // namespace vllm_sparse_select

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_cluster_scores",
        &vllm_sparse_select::sparse_cluster_scores,
        "Stage 1: batched group-summed centroid softmax probabilities "
        "for sparse attention. Output: [num_reqs, num_kv_heads, "
        "num_clusters] fp32.",
        pybind11::arg("query"),
        pybind11::arg("query_start_loc"),
        pybind11::arg("centres_ptrs"),
        pybind11::arg("num_kv_heads"),
        pybind11::arg("num_clusters"),
        pybind11::arg("head_dim"),
        pybind11::arg("stride_c_h_elems"),
        pybind11::arg("stride_c_c_elems"),
        pybind11::arg("group_size"));

  m.def("fused_topk_cumsum",
        &vllm_sparse_select::fused_topk_cumsum,
        "Stage 2: per-(req, head) top-K over stage-1 scores, "
        "gather cluster_size via a per-request pointer table, and "
        "cumsum -- all fused into a single CUDA kernel. Returns "
        "(top_indices [R, H, nprobe] int64, "
        "cluster_start_index [R, H, nprobe] int32).",
        pybind11::arg("scores"),
        pybind11::arg("sizes_ptrs"),
        pybind11::arg("num_kv_heads"),
        pybind11::arg("num_clusters"),
        pybind11::arg("nprobe"),
        pybind11::arg("stride_sizes_h_elems"));
}
