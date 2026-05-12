#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "sparse_attention_common.h"

template <typename T>
__device__ __forceinline__ void strided_copy_vec(T* dst, const T* src, int dim, int start, int stride) {
    for (int d = start; d < dim; d += stride) {
        dst[d] = src[d];
    }
}

__device__ __forceinline__ int64_t plane_stride_elems(int total_blocks, int block_size, int dim) {
    return (int64_t)total_blocks * block_size * dim;
}

__device__ __forceinline__ int64_t block_elem_offset(int block_id, int block_off, int block_size, int dim) {
    return ((int64_t)block_id * block_size + block_off) * dim;
}

__device__ __forceinline__ int64_t hc_index(int h, int c, int C) {
    return (int64_t)h * C + c;
}

__device__ __forceinline__ int64_t compact_block_index(int h, int c, int bid_pos, int C, int maxB) {
    return (((int64_t)h * C + c) * maxB + bid_pos);
}

__device__ __forceinline__ int64_t temp_pos_index(int h, int c, int slot, int C, int block_size, int xy) {
    return ((((int64_t)h * C + c) * block_size + slot) * 2 + xy);
}

template <typename T, int MAX_HKV, int MAX_REMOVED>
__global__ void k_append_kv_to_clusters_persistent(
    const T* __restrict__ key,                        // [Nq, Hkv, dim]
    const T* __restrict__ value,                      // [Nq, Hkv, dim]
    const int32_t* __restrict__ label,                // [Nq, Hkv]

    const int32_t* __restrict__ temp_block_ids,       // [max_temp_blocks]
    const int32_t* __restrict__ free_block_ids,       // [max_free_block]

    T* __restrict__ block_storage,                    // [2, total_blocks, block_size, dim]
    int32_t* __restrict__ cluster_compact_block_ids,  // [Hkv, C, maxB]
    int32_t* __restrict__ cluster_temp_kv_pos,        // [Hkv, C, block_size, 2]
    int32_t* __restrict__ cluster_total_kv_counts,    // [Hkv, C]
    int32_t* __restrict__ temp_block_kv_counts,       // [1]
    int32_t* __restrict__ temp_block_kv_owner,        // [max_temp_blocks * block_size, 2]
    int32_t* __restrict__ used_free_block_count,      // [1]
    int32_t* __restrict__ error_code,                 // [1]

    int Nq, int Hkv, int C, int maxB,
    int total_blocks, int block_size, int dim,
    int max_temp_blocks, int max_free_block)
{
    if (blockIdx.x != 0) return;

    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    __shared__ int s_target_cid[MAX_HKV];

    __shared__ int s_full_count;
    __shared__ int s_full_h[MAX_HKV];
    __shared__ int s_full_c[MAX_HKV];

    __shared__ int s_removed_count;
    __shared__ int s_removed_gpos[MAX_REMOVED];

    __shared__ int s_fill_count;
    __shared__ int s_fill_dst_gpos[MAX_REMOVED];
    __shared__ int s_fill_src_gpos[MAX_REMOVED];

    __shared__ int s_temp_count_before;
    __shared__ int s_tail_base;

    const int64_t pstride = plane_stride_elems(total_blocks, block_size, dim);

    for (int q = 0; q < Nq; ++q) {
        if (tid < Hkv) {
            int cid = label[(int64_t)q * Hkv + tid];
            if (cid < 0 || cid >= C) {
                atomicCAS(error_code, 0, ERR_BAD_PARAM);
            } else {
                s_target_cid[tid] = cid;
            }
        }

        if (tid == 0) {
            s_full_count = 0;
            s_removed_count = 0;
            s_fill_count = 0;
        }
        __syncthreads();

        if (error_code[0] != ERR_OK) return;

        // ------------------------------------------------
        // Step 1: append 当前 q
        // ------------------------------------------------
        if (warp < Hkv) {
            const int h = warp;
            const int cid = s_target_cid[h];

            int gpos = -1;
            int tb_idx = -1;
            int tb_off = -1;
            int bid = -1;
            int slot = -1;
            int became_full = 0;

            if (lane == 0) {
                int old_total = atomicAdd(&cluster_total_kv_counts[hc_index(h, cid, C)], 1);
                slot = old_total % block_size;
                became_full = (((old_total + 1) % block_size) == 0);

                gpos = atomicAdd(temp_block_kv_counts, 1);
                tb_idx = gpos / block_size;
                tb_off = gpos % block_size;

                if (tb_idx < 0 || tb_idx >= max_temp_blocks) {
                    atomicCAS(error_code, 0, ERR_NO_FREE_BLOCK);
                    bid = -1;
                } else {
                    bid = temp_block_ids[tb_idx];
                    if (bid < 0 || bid >= total_blocks) {
                        atomicCAS(error_code, 0, ERR_BAD_PARAM);
                        bid = -1;
                    }
                }
            }

            gpos = __shfl_sync(FULL_MASK, gpos, 0);
            tb_idx = __shfl_sync(FULL_MASK, tb_idx, 0);
            tb_off = __shfl_sync(FULL_MASK, tb_off, 0);
            bid = __shfl_sync(FULL_MASK, bid, 0);

            if (bid >= 0) {
                int64_t src_base = ((int64_t)q * Hkv + h) * dim;
                const T* src_k = key + src_base;
                const T* src_v = value + src_base;

                int64_t dst_elem = block_elem_offset(bid, tb_off, block_size, dim);
                T* dst_k = block_storage + dst_elem;
                T* dst_v = block_storage + pstride + dst_elem;

                strided_copy_vec(dst_k, src_k, dim, lane, WARP_SIZE);
                strided_copy_vec(dst_v, src_v, dim, lane, WARP_SIZE);

                if (lane == 0) {
                    cluster_temp_kv_pos[temp_pos_index(h, cid, slot, C, block_size, 0)] = tb_idx;
                    cluster_temp_kv_pos[temp_pos_index(h, cid, slot, C, block_size, 1)] = tb_off;

                    temp_block_kv_owner[(int64_t)gpos * 2 + 0] = h * C + cid;
                    temp_block_kv_owner[(int64_t)gpos * 2 + 1] = slot;

                    if (became_full) {
                        int pos = atomicAdd(&s_full_count, 1);
                        if (pos < MAX_HKV) {
                            s_full_h[pos] = h;
                            s_full_c[pos] = cid;
                        } else {
                            atomicCAS(error_code, 0, ERR_BAD_PARAM);
                        }
                    }
                }
            }
        }
        __syncthreads();

        if (error_code[0] != ERR_OK) return;

        if (s_full_count == 0) {
            continue;
        }

        // ------------------------------------------------
        // Step 2: compact all newly-full clusters
        // ------------------------------------------------
        if (tid == 0) {
            s_temp_count_before = temp_block_kv_counts[0];
        }
        __syncthreads();

        for (int fc = warp; fc < s_full_count; fc += num_warps) {
            const int h = s_full_h[fc];
            const int cid = s_full_c[fc];

            int total_cnt = cluster_total_kv_counts[hc_index(h, cid, C)];
            int compact_pos = total_cnt / block_size - 1;

            if (compact_pos < 0 || compact_pos >= maxB) {
                if (lane == 0) atomicCAS(error_code, 0, ERR_NB_OVERFLOW);
                continue;
            }

            int64_t cb_idx = compact_block_index(h, cid, compact_pos, C, maxB);
            int dst_bid = -1;

            if (lane == 0) {
                int cur_bid = cluster_compact_block_ids[cb_idx];
                if (cur_bid >= 0) {
                    dst_bid = cur_bid;
                } else {
                    int fb = atomicAdd(used_free_block_count, 1);
                    if (fb >= max_free_block) {
                        atomicCAS(error_code, 0, ERR_NO_FREE_BLOCK);
                        dst_bid = -1;
                    } else {
                        int new_bid = free_block_ids[fb];
                        if (new_bid < 0 || new_bid >= total_blocks) {
                            atomicCAS(error_code, 0, ERR_BAD_PARAM);
                            dst_bid = -1;
                        } else {
                            cluster_compact_block_ids[cb_idx] = new_bid;
                            dst_bid = new_bid;
                        }
                    }
                }
            }
            dst_bid = __shfl_sync(FULL_MASK, dst_bid, 0);
            if (dst_bid < 0) continue;

            int removed_base = 0;
            if (lane == 0) {
                removed_base = atomicAdd(&s_removed_count, block_size);
                if (removed_base + block_size > MAX_REMOVED) {
                    atomicCAS(error_code, 0, ERR_BAD_PARAM);
                }
            }
            removed_base = __shfl_sync(FULL_MASK, removed_base, 0);

            if (error_code[0] != ERR_OK) continue;

            if (block_size <= WARP_SIZE) {
                const int lanes_per_slot = WARP_SIZE / block_size;
                const int slot = lane / lanes_per_slot;
                const int sub_lane = lane % lanes_per_slot;

                if (slot < block_size) {
                    int tb_idx_i = cluster_temp_kv_pos[temp_pos_index(h, cid, slot, C, block_size, 0)];
                    int tb_off_i = cluster_temp_kv_pos[temp_pos_index(h, cid, slot, C, block_size, 1)];

                    if (tb_idx_i < 0 || tb_off_i < 0 || tb_off_i >= block_size) {
                        if (sub_lane == 0) atomicCAS(error_code, 0, ERR_BAD_PARAM);
                    } else {
                        int src_bid = temp_block_ids[tb_idx_i];
                        if (src_bid < 0 || src_bid >= total_blocks) {
                            if (sub_lane == 0) atomicCAS(error_code, 0, ERR_BAD_PARAM);
                        } else {
                            int64_t src_elem = block_elem_offset(src_bid, tb_off_i, block_size, dim);
                            int64_t dst_elem = block_elem_offset(dst_bid, slot, block_size, dim);

                            const T* src_k = block_storage + src_elem;
                            const T* src_v = block_storage + pstride + src_elem;
                            T* dst_k = block_storage + dst_elem;
                            T* dst_v = block_storage + pstride + dst_elem;

                            strided_copy_vec(dst_k, src_k, dim, sub_lane, lanes_per_slot);
                            strided_copy_vec(dst_v, src_v, dim, sub_lane, lanes_per_slot);

                            if (sub_lane == 0) {
                                int gpos_i = tb_idx_i * block_size + tb_off_i;
                                s_removed_gpos[removed_base + slot] = gpos_i;

                                cluster_temp_kv_pos[temp_pos_index(h, cid, slot, C, block_size, 0)] = -1;
                                cluster_temp_kv_pos[temp_pos_index(h, cid, slot, C, block_size, 1)] = -1;

                                temp_block_kv_owner[(int64_t)gpos_i * 2 + 0] = -1;
                                temp_block_kv_owner[(int64_t)gpos_i * 2 + 1] = -1;
                            }
                        }
                    }
                }
            } else {
                for (int slot = lane; slot < block_size; slot += WARP_SIZE) {
                    int tb_idx_i = cluster_temp_kv_pos[temp_pos_index(h, cid, slot, C, block_size, 0)];
                    int tb_off_i = cluster_temp_kv_pos[temp_pos_index(h, cid, slot, C, block_size, 1)];

                    if (tb_idx_i < 0 || tb_off_i < 0 || tb_off_i >= block_size) {
                        atomicCAS(error_code, 0, ERR_BAD_PARAM);
                    } else {
                        int src_bid = temp_block_ids[tb_idx_i];
                        if (src_bid < 0 || src_bid >= total_blocks) {
                            atomicCAS(error_code, 0, ERR_BAD_PARAM);
                        } else {
                            int64_t src_elem = block_elem_offset(src_bid, tb_off_i, block_size, dim);
                            int64_t dst_elem = block_elem_offset(dst_bid, slot, block_size, dim);

                            const T* src_k = block_storage + src_elem;
                            const T* src_v = block_storage + pstride + src_elem;
                            T* dst_k = block_storage + dst_elem;
                            T* dst_v = block_storage + pstride + dst_elem;

                            strided_copy_vec(dst_k, src_k, dim, 0, 1);
                            strided_copy_vec(dst_v, src_v, dim, 0, 1);

                            int gpos_i = tb_idx_i * block_size + tb_off_i;
                            s_removed_gpos[removed_base + slot] = gpos_i;

                            cluster_temp_kv_pos[temp_pos_index(h, cid, slot, C, block_size, 0)] = -1;
                            cluster_temp_kv_pos[temp_pos_index(h, cid, slot, C, block_size, 1)] = -1;

                            temp_block_kv_owner[(int64_t)gpos_i * 2 + 0] = -1;
                            temp_block_kv_owner[(int64_t)gpos_i * 2 + 1] = -1;
                        }
                    }
                }
            }
        }
        __syncthreads();

        if (error_code[0] != ERR_OK) return;

        // ------------------------------------------------
        // Step 3: 构造真正需要 fill 的 (src,dst) 对
        //
        // tail_base = temp_count_before - removed_count
        // 只填 dst < tail_base 的 holes
        // src 只从 [tail_base, temp_count_before) 中挑有效 survivor
        // ------------------------------------------------
        if (tid == 0) {
            s_tail_base = s_temp_count_before - s_removed_count;
            s_fill_count = 0;

            // 收集需要填的 holes：只要落在前半有效区 [0, tail_base)
            for (int i = 0; i < s_removed_count; ++i) {
                int dst = s_removed_gpos[i];
                if (dst < s_tail_base) {
                    int pos = s_fill_count++;
                    s_fill_dst_gpos[pos] = dst;
                }
            }

            // 从尾段 [tail_base, temp_count_before) 收集仍存活的 src
            int k = 0;
            for (int src = s_tail_base; src < s_temp_count_before; ++src) {
                if (temp_block_kv_owner[(int64_t)src * 2 + 0] >= 0) {
                    if (k < s_fill_count) {
                        s_fill_src_gpos[k] = src;
                    }
                    ++k;
                }
            }

            if (k != s_fill_count) {
                atomicCAS(error_code, 0, ERR_BAD_PARAM);
            }
        }
        __syncthreads();

        if (error_code[0] != ERR_OK) return;

        // ------------------------------------------------
        // Step 4: fill holes
        //
        // 只搬 s_fill_count 对
        // s_removed_count 只用于最终缩短 temp 长度
        // ------------------------------------------------
        for (int fi = warp; fi < s_fill_count; fi += num_warps) {
            int dst_gpos = s_fill_dst_gpos[fi];
            int src_gpos = s_fill_src_gpos[fi];

            if (src_gpos != dst_gpos) {
                int src_tb_idx = src_gpos / block_size;
                int src_tb_off = src_gpos % block_size;
                int dst_tb_idx = dst_gpos / block_size;
                int dst_tb_off = dst_gpos % block_size;

                int src_bid = temp_block_ids[src_tb_idx];
                int dst_bid = temp_block_ids[dst_tb_idx];

                if (src_bid < 0 || src_bid >= total_blocks ||
                    dst_bid < 0 || dst_bid >= total_blocks) {
                    if (lane == 0) atomicCAS(error_code, 0, ERR_BAD_PARAM);
                } else {
                    int64_t src_elem = block_elem_offset(src_bid, src_tb_off, block_size, dim);
                    int64_t dst_elem = block_elem_offset(dst_bid, dst_tb_off, block_size, dim);

                    const T* src_k = block_storage + src_elem;
                    const T* src_v = block_storage + pstride + src_elem;
                    T* dst_k = block_storage + dst_elem;
                    T* dst_v = block_storage + pstride + dst_elem;

                    // 这里一个 fill pair 用整个 warp 沿 dim 协同 copy
                    strided_copy_vec(dst_k, src_k, dim, lane, WARP_SIZE);
                    strided_copy_vec(dst_v, src_v, dim, lane, WARP_SIZE);

                    if (lane == 0) {
                        int owner_cluster = temp_block_kv_owner[(int64_t)src_gpos * 2 + 0];
                        int owner_slot = temp_block_kv_owner[(int64_t)src_gpos * 2 + 1];

                        if (owner_cluster < 0 || owner_slot < 0 || owner_slot >= block_size) {
                            atomicCAS(error_code, 0, ERR_BAD_PARAM);
                        } else {
                            temp_block_kv_owner[(int64_t)dst_gpos * 2 + 0] = owner_cluster;
                            temp_block_kv_owner[(int64_t)dst_gpos * 2 + 1] = owner_slot;

                            int oh = owner_cluster / C;
                            int oc = owner_cluster % C;

                            cluster_temp_kv_pos[temp_pos_index(oh, oc, owner_slot, C, block_size, 0)] = dst_tb_idx;
                            cluster_temp_kv_pos[temp_pos_index(oh, oc, owner_slot, C, block_size, 1)] = dst_tb_off;
                        }
                    }
                }
            }
        }
        __syncthreads();

        if (error_code[0] != ERR_OK) return;

        if (tid == 0) {
            // 直接截短 temp 有效区间
            temp_block_kv_counts[0] -= s_removed_count;
        }
        __syncthreads();
    }
}

static int pick_warps(int rows) {
    if (rows <= 1) return 1;
    if (rows <= 2) return 2;
    if (rows <= 32) return 4;
    return 8;
}

extern "C" int append_kv_to_clusters_launcher_raw(
    const void* d_key,
    const void* d_value,
    int32_t storage_dtype,
    const int32_t* d_label,
    const int32_t* d_temp_block_ids,
    int32_t* d_temp_block_kv_counts,
    int32_t* d_temp_block_kv_owner,
    void* d_block_storage,
    int32_t* d_cluster_compact_block_ids,
    int32_t* d_cluster_temp_kv_pos,
    int32_t* d_cluster_total_kv_counts,
    const int32_t* d_free_block_ids,
    int32_t max_free_block,
    int32_t* d_used_free_block_count,
    int32_t* d_error_code,
    int Nq, int Hkv, int C, int maxB,
    int total_blocks, int block_size, int dim,
    int max_temp_blocks,
    cudaStream_t stream)
{
    if (Nq <= 0 || Hkv <= 0 || C <= 0 || maxB <= 0 ||
        total_blocks <= 0 || block_size <= 0 || dim <= 0 ||
        max_temp_blocks <= 0 || max_free_block <= 0) {
        return ERR_BAD_PARAM;
    }

    if (Hkv > 8) {
        return ERR_BAD_PARAM;
    }

    // shared memory 上界按 MAX_REMOVED = MAX_HKV * 256
    if (block_size > 256) {
        return ERR_BAD_PARAM;
    }

    cudaError_t cerr = cudaMemsetAsync(d_error_code, 0, sizeof(int32_t), stream);
    if (cerr != cudaSuccess) return ERR_LAUNCH;

    constexpr int MAX_HKV = 8;
    constexpr int MAX_REMOVED = 8 * 256;

    int warps = pick_warps(Nq * Hkv);
    int threads = warps * WARP_SIZE;

    if (storage_dtype == DTYPE_FP32) {
        k_append_kv_to_clusters_persistent<float, MAX_HKV, MAX_REMOVED>
            <<<1, threads, 0, stream>>>(
                (const float*)d_key,
                (const float*)d_value,
                d_label,
                d_temp_block_ids,
                d_free_block_ids,
                (float*)d_block_storage,
                d_cluster_compact_block_ids,
                d_cluster_temp_kv_pos,
                d_cluster_total_kv_counts,
                d_temp_block_kv_counts,
                d_temp_block_kv_owner,
                d_used_free_block_count,
                d_error_code,
                Nq, Hkv, C, maxB,
                total_blocks, block_size, dim,
                max_temp_blocks, max_free_block);
    } else if (storage_dtype == DTYPE_FP16) {
        k_append_kv_to_clusters_persistent<__half, MAX_HKV, MAX_REMOVED>
            <<<1, threads, 0, stream>>>(
                (const __half*)d_key,
                (const __half*)d_value,
                d_label,
                d_temp_block_ids,
                d_free_block_ids,
                (__half*)d_block_storage,
                d_cluster_compact_block_ids,
                d_cluster_temp_kv_pos,
                d_cluster_total_kv_counts,
                d_temp_block_kv_counts,
                d_temp_block_kv_owner,
                d_used_free_block_count,
                d_error_code,
                Nq, Hkv, C, maxB,
                total_blocks, block_size, dim,
                max_temp_blocks, max_free_block);
    } else if (storage_dtype == DTYPE_BF16) {
        k_append_kv_to_clusters_persistent<__nv_bfloat16, MAX_HKV, MAX_REMOVED>
            <<<1, threads, 0, stream>>>(
                (const __nv_bfloat16*)d_key,
                (const __nv_bfloat16*)d_value,
                d_label,
                d_temp_block_ids,
                d_free_block_ids,
                (__nv_bfloat16*)d_block_storage,
                d_cluster_compact_block_ids,
                d_cluster_temp_kv_pos,
                d_cluster_total_kv_counts,
                d_temp_block_kv_counts,
                d_temp_block_kv_owner,
                d_used_free_block_count,
                d_error_code,
                Nq, Hkv, C, maxB,
                total_blocks, block_size, dim,
                max_temp_blocks, max_free_block);
    } else {
        return ERR_UNSUPPORTED_DTYPE;
    }

    if (cudaGetLastError() != cudaSuccess) return ERR_LAUNCH;
    return ERR_OK;
}