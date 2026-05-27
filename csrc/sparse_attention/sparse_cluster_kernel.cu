#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <limits.h>

#include "sparse_attention_common.h"

template <typename T>
__device__ __forceinline__ void strided_copy_vec(
    T* dst,
    const T* src,
    int dim,
    int start,
    int stride,
    int64_t src_dim_stride = 1) {
    for (int d = start; d < dim; d += stride) {
        dst[d] = src[(int64_t)d * src_dim_stride];
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
    const T* __restrict__ key,                        // [Nq, Hkv, dim] (strided)
    const T* __restrict__ value,                      // [Nq, Hkv, dim] (strided)
    const int32_t* __restrict__ label,                // optional [Nq, Hkv]
    T* __restrict__ cluster_centers_T,                // optional [Hkv, dim, C]
    const T* __restrict__ mean,                       // optional [Hkv, dim]
    int32_t* __restrict__ cluster_center_count,       // optional [1]
    const int32_t* __restrict__ input_token_count,    // optional [1]

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
    const int32_t* __restrict__ steady_start_block_ids,  // optional [Hkv, steady_start_blocks]
    int steady_start_blocks,
    const int32_t* __restrict__ steady_end_block_ids,    // optional [Hkv, steady_end_blocks]
    int steady_end_blocks,
    int32_t* __restrict__ steady_state,                  // optional [4]
    int steady_start_capacity,
    int steady_end_capacity,

    int64_t key_stride0,
    int64_t key_stride1,
    int64_t key_stride2,
    int64_t value_stride0,
    int64_t value_stride1,
    int64_t value_stride2,
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
    __shared__ int s_valid_centers;
    __shared__ int s_effective_Nq;
    __shared__ int s_has_steady;
    __shared__ int s_steady_action;
    __shared__ int s_steady_pos;
    extern __shared__ float s_center_scores[];

    const int64_t pstride = plane_stride_elems(total_blocks, block_size, dim);

    if (tid == 0) {
        s_effective_Nq = Nq;
        if (input_token_count != nullptr) {
            const int requested = input_token_count[0];
            s_effective_Nq = min(max(requested, 0), Nq);
        }
        s_has_steady = steady_state != nullptr ? 1 : 0;
    }
    __syncthreads();

    for (int q = 0; q < s_effective_Nq; ++q) {
        if (s_has_steady) {
            if (tid == 0) {
                const int total_seen = steady_state[0];
                const int end_count = steady_state[2];
                const int end_start = steady_state[3];

                // 0=start, 1=end, 2=evict-old-end-then-append, 3=append-direct
                if (total_seen < steady_start_capacity) {
                    s_steady_action = 0;
                    s_steady_pos = total_seen;
                } else if (steady_end_capacity <= 0) {
                    s_steady_action = 3;
                    s_steady_pos = -1;
                } else if (end_count < steady_end_capacity) {
                    s_steady_action = 1;
                    s_steady_pos = end_count;
                } else {
                    s_steady_action = 2;
                    s_steady_pos = end_start;
                }
            }
            __syncthreads();

            if (s_steady_action <= 1) {
                if (warp < Hkv) {
                    const int h = warp;
                    const bool to_start = s_steady_action == 0;
                    const int pos = s_steady_pos;
                    const int block_idx = pos / block_size;
                    const int off = pos - block_idx * block_size;
                    const int num_blocks =
                        to_start ? steady_start_blocks : steady_end_blocks;
                    const int32_t* block_ids =
                        to_start ? steady_start_block_ids : steady_end_block_ids;
                    const int64_t src_k_base =
                        (int64_t)q * key_stride0 + (int64_t)h * key_stride1;
                    const int64_t src_v_base =
                        (int64_t)q * value_stride0 + (int64_t)h * value_stride1;
                    const T* src_k = key + src_k_base;
                    const T* src_v = value + src_v_base;

                    if (block_idx < 0 || block_idx >= num_blocks) {
                        if (lane == 0) atomicCAS(error_code, 0, ERR_BAD_PARAM);
                    } else {
                        const int bid = block_ids[(int64_t)h * num_blocks + block_idx];
                        if (bid < 0 || bid >= total_blocks) {
                            if (lane == 0) atomicCAS(error_code, 0, ERR_BAD_PARAM);
                        } else {
                            const int64_t dst_elem =
                                block_elem_offset(bid, off, block_size, dim);
                            T* dst_k = block_storage + dst_elem;
                            T* dst_v = block_storage + pstride + dst_elem;
                            strided_copy_vec(
                                dst_k, src_k, dim, lane, WARP_SIZE, key_stride2);
                            strided_copy_vec(
                                dst_v, src_v, dim, lane, WARP_SIZE, value_stride2);
                        }
                    }
                }
                __syncthreads();
                if (error_code[0] != ERR_OK) return;

                if (tid == 0) {
                    steady_state[0] += 1;
                    if (s_steady_action == 0) {
                        steady_state[1] += 1;
                    } else {
                        steady_state[2] += 1;
                    }
                }
                __syncthreads();
                continue;
            }
        }

        if (label == nullptr) {
            if (tid == 0) {
                if (cluster_center_count == nullptr) {
                    atomicCAS(error_code, 0, ERR_BAD_PARAM);
                    s_valid_centers = 0;
                } else {
                    s_valid_centers = min(max(cluster_center_count[0], 0), C);
                }
            }
            __syncthreads();
        }

        if (label != nullptr) {
            if (tid < Hkv) {
                int cid = label[(int64_t)q * Hkv + tid];
                if (cid < 0 || cid >= C) {
                    atomicCAS(error_code, 0, ERR_BAD_PARAM);
                } else {
                    s_target_cid[tid] = cid;
                }
            }
        } else {
            const int valid_centers = s_valid_centers;
            if (cluster_centers_T == nullptr || mean == nullptr) {
                if (tid == 0) atomicCAS(error_code, 0, ERR_BAD_PARAM);
            } else if (valid_centers < C) {
                if (warp < Hkv) {
                    const int h = warp;
                    const int action = s_has_steady ? s_steady_action : 3;
                    const int pos = s_steady_pos;
                    const int64_t key_base =
                        (int64_t)q * key_stride0 + (int64_t)h * key_stride1;
                    const int64_t center_base =
                        (int64_t)h * dim * C + valid_centers;
                    if (action == 2) {
                        const int block_idx = pos / block_size;
                        const int off = pos - block_idx * block_size;
                        if (block_idx < 0 || block_idx >= steady_end_blocks) {
                            if (lane == 0) atomicCAS(error_code, 0, ERR_BAD_PARAM);
                        } else {
                            const int bid = steady_end_block_ids[
                                (int64_t)h * steady_end_blocks + block_idx];
                            if (bid < 0 || bid >= total_blocks) {
                                if (lane == 0) {
                                    atomicCAS(error_code, 0, ERR_BAD_PARAM);
                                }
                            } else {
                                const int64_t src_elem =
                                    block_elem_offset(bid, off, block_size, dim);
                                const T* src_k = block_storage + src_elem;
                                for (int d = lane; d < dim; d += WARP_SIZE) {
                                    cluster_centers_T[center_base + (int64_t)d * C] =
                                        src_k[d];
                                }
                            }
                        }
                    } else {
                        for (int d = lane; d < dim; d += WARP_SIZE) {
                            cluster_centers_T[center_base + (int64_t)d * C] =
                                key[key_base + (int64_t)d * key_stride2];
                        }
                    }
                    if (lane == 0) {
                        s_target_cid[h] = valid_centers;
                    }
                }
            } else {
                // Decode appends classify the incoming key here so the
                // current cluster centers are sampled at the actual append
                // point.  This keeps the graph path independent of a Python
                // argmax branch and will remain correct when centers are
                // updated online.
                //
                // One warp computes one (head, cluster) score, with lanes
                // split across dim. This avoids serial dim loops inside a
                // single lane for the common dim=32/64 decode case.
                const int total_scores = Hkv * valid_centers;
                for (int score_idx = warp; score_idx < total_scores;
                     score_idx += num_warps) {
                    const int h = score_idx / valid_centers;
                    const int c = score_idx - h * valid_centers;
                    const int action = s_has_steady ? s_steady_action : 3;
                    const int pos = s_steady_pos;
                    const int64_t key_base =
                        (int64_t)q * key_stride0 + (int64_t)h * key_stride1;
                    const int64_t center_h_base = (int64_t)h * dim * C;
                    const int64_t mean_h_base = (int64_t)h * dim;
                    T score = T(0.0f);
                    if (action == 2) {
                        const int block_idx = pos / block_size;
                        const int off = pos - block_idx * block_size;
                        if (block_idx < 0 || block_idx >= steady_end_blocks) {
                            if (lane == 0) atomicCAS(error_code, 0, ERR_BAD_PARAM);
                        } else {
                            const int bid = steady_end_block_ids[
                                (int64_t)h * steady_end_blocks + block_idx];
                            if (bid < 0 || bid >= total_blocks) {
                                if (lane == 0) {
                                    atomicCAS(error_code, 0, ERR_BAD_PARAM);
                                }
                            } else {
                                const int64_t src_elem =
                                    block_elem_offset(bid, off, block_size, dim);
                                const T* src_k = block_storage + src_elem;
                                for (int d = lane; d < dim; d += WARP_SIZE) {
                                    const T key_v = src_k[d];
                                    const T mean_v = mean[mean_h_base + d];
                                    const T center_v =
                                        cluster_centers_T[center_h_base +
                                                          (int64_t)d * C + c];
                                    score = score + (key_v - mean_v) * center_v;
                                }
                            }
                        }
                    } else {
                        for (int d = lane; d < dim; d += WARP_SIZE) {
                            const T key_v =
                                key[key_base + (int64_t)d * key_stride2];
                            const T mean_v = mean[mean_h_base + d];
                            const T center_v =
                                cluster_centers_T[center_h_base +
                                                  (int64_t)d * C + c];
                            score = score + (key_v - mean_v) * center_v;
                        }
                    }

                    float score_f = static_cast<float>(score);
                    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                        score_f += __shfl_down_sync(FULL_MASK, score_f, offset);
                    }

                    if (lane == 0) {
                        s_center_scores[score_idx] = score_f;
                    }
                }
            }
            __syncthreads();

            if (valid_centers >= C && cluster_centers_T != nullptr &&
                mean != nullptr && warp < Hkv) {
                const int h = warp;
                float best_score = -3.4028234663852886e38f;
                int best_c = 0;
                for (int c = lane; c < valid_centers; c += WARP_SIZE) {
                    const float score =
                        s_center_scores[h * valid_centers + c];
                    if (score > best_score ||
                        (score == best_score && c < best_c)) {
                        best_score = score;
                        best_c = c;
                    }
                }
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                    const float other_score =
                        __shfl_down_sync(FULL_MASK, best_score, offset);
                    const int other_c =
                        __shfl_down_sync(FULL_MASK, best_c, offset);
                    if (other_score > best_score ||
                        (other_score == best_score && other_c < best_c)) {
                        best_score = other_score;
                        best_c = other_c;
                    }
                }
                if (lane == 0) {
                    s_target_cid[h] = best_c;
                }
            }
            __syncthreads();
            if (valid_centers < C && cluster_center_count != nullptr && tid == 0) {
                cluster_center_count[0] = valid_centers + 1;
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
            const int action = s_has_steady ? s_steady_action : 3;
            const int pos = s_steady_pos;
            int64_t src_k_base = (int64_t)q * key_stride0 + (int64_t)h * key_stride1;
            int64_t src_v_base = (int64_t)q * value_stride0 + (int64_t)h * value_stride1;
            const T* src_k = key + src_k_base;
            const T* src_v = value + src_v_base;
            int64_t src_k_stride = key_stride2;
            int64_t src_v_stride = value_stride2;
            bool src_valid = true;

            if (action == 2) {
                const int block_idx = pos / block_size;
                const int off = pos - block_idx * block_size;
                if (block_idx < 0 || block_idx >= steady_end_blocks) {
                    if (lane == 0) atomicCAS(error_code, 0, ERR_BAD_PARAM);
                    src_valid = false;
                } else {
                    const int steady_bid = steady_end_block_ids[
                        (int64_t)h * steady_end_blocks + block_idx];
                    if (steady_bid < 0 || steady_bid >= total_blocks) {
                        if (lane == 0) atomicCAS(error_code, 0, ERR_BAD_PARAM);
                        src_valid = false;
                    } else {
                        const int64_t src_elem =
                            block_elem_offset(steady_bid, off, block_size, dim);
                        src_k = block_storage + src_elem;
                        src_v = block_storage + pstride + src_elem;
                        src_k_stride = 1;
                        src_v_stride = 1;
                    }
                }
            }

            int gpos = -1;
            int tb_idx = -1;
            int tb_off = -1;
            int bid = -1;
            int slot = -1;
            int became_full = 0;

            if (lane == 0 && src_valid) {
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

            if (bid >= 0 && src_valid) {
                int64_t dst_elem = block_elem_offset(bid, tb_off, block_size, dim);
                T* dst_k = block_storage + dst_elem;
                T* dst_v = block_storage + pstride + dst_elem;

                strided_copy_vec(dst_k, src_k, dim, lane, WARP_SIZE, src_k_stride);
                strided_copy_vec(dst_v, src_v, dim, lane, WARP_SIZE, src_v_stride);

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

        if (s_full_count > 0) {
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

        if (s_has_steady && s_steady_action == 2) {
            if (warp < Hkv) {
                const int h = warp;
                const int pos = s_steady_pos;
                const int block_idx = pos / block_size;
                const int off = pos - block_idx * block_size;
                const int64_t src_k_base =
                    (int64_t)q * key_stride0 + (int64_t)h * key_stride1;
                const int64_t src_v_base =
                    (int64_t)q * value_stride0 + (int64_t)h * value_stride1;
                const T* src_k = key + src_k_base;
                const T* src_v = value + src_v_base;

                if (block_idx < 0 || block_idx >= steady_end_blocks) {
                    if (lane == 0) atomicCAS(error_code, 0, ERR_BAD_PARAM);
                } else {
                    const int bid = steady_end_block_ids[
                        (int64_t)h * steady_end_blocks + block_idx];
                    if (bid < 0 || bid >= total_blocks) {
                        if (lane == 0) atomicCAS(error_code, 0, ERR_BAD_PARAM);
                    } else {
                        const int64_t dst_elem =
                            block_elem_offset(bid, off, block_size, dim);
                        T* dst_k = block_storage + dst_elem;
                        T* dst_v = block_storage + pstride + dst_elem;
                        strided_copy_vec(
                            dst_k, src_k, dim, lane, WARP_SIZE, key_stride2);
                        strided_copy_vec(
                            dst_v, src_v, dim, lane, WARP_SIZE, value_stride2);
                    }
                }
            }
            __syncthreads();
            if (error_code[0] != ERR_OK) return;
        }

        if (tid == 0 && s_has_steady) {
            steady_state[0] += 1;
            if (s_steady_action == 2) {
                steady_state[3] = (steady_state[3] + 1) % steady_end_capacity;
            }
        }
        __syncthreads();
    }
}

template <typename T, int MAX_HKV>
__global__ void k_update_sparse_steady_kv(
    T* __restrict__ block_storage,
    const int32_t* __restrict__ steady_start_block_ids,
    int steady_start_blocks,
    const int32_t* __restrict__ steady_end_block_ids,
    int steady_end_blocks,
    int32_t* __restrict__ steady_state,
    T* __restrict__ evicted_key,
    T* __restrict__ evicted_value,
    int32_t* __restrict__ evicted_count,
    const T* __restrict__ key,
    const T* __restrict__ value,
    int64_t key_stride0,
    int64_t key_stride1,
    int64_t key_stride2,
    int64_t value_stride0,
    int64_t value_stride1,
    int64_t value_stride2,
    int Nq, int Hkv,
    int total_blocks, int block_size, int dim,
    int steady_start_capacity,
    int steady_end_capacity,
    int evicted_capacity,
    int32_t* __restrict__ error_code) {
    if (blockIdx.x != 0) return;

    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;

    __shared__ int s_action;
    __shared__ int s_pos;
    __shared__ int s_evict_idx;

    const int64_t pstride = plane_stride_elems(total_blocks, block_size, dim);

    if (tid == 0) {
        evicted_count[0] = 0;
        error_code[0] = ERR_OK;
    }
    __syncthreads();

    for (int q = 0; q < Nq; ++q) {
        if (tid == 0) {
            const int total_seen = steady_state[0];
            const int end_count = steady_state[2];
            const int end_start = steady_state[3];
            s_action = 0;  // 0=start, 1=end, 2=evict-old-end, 3=cluster-direct
            s_pos = total_seen;
            s_evict_idx = -1;

            if (total_seen < steady_start_capacity) {
                steady_state[1] = total_seen + 1;
            } else if (steady_end_capacity <= 0) {
                s_action = 3;
                s_pos = -1;
                s_evict_idx = evicted_count[0]++;
            } else if (end_count < steady_end_capacity) {
                s_action = 1;
                s_pos = end_count;
                steady_state[2] = end_count + 1;
            } else {
                s_action = 2;
                s_pos = end_start;
                s_evict_idx = evicted_count[0]++;
                steady_state[3] = (end_start + 1) % steady_end_capacity;
            }
            steady_state[0] = total_seen + 1;
            if (s_evict_idx >= evicted_capacity) {
                atomicCAS(error_code, 0, ERR_NO_FREE_BLOCK);
            }
        }
        __syncthreads();
        if (error_code[0] != ERR_OK) return;

        if (warp < Hkv) {
            const int h = warp;
            const int action = s_action;
            const int pos = s_pos;
            const int evict_idx = s_evict_idx;
            const int64_t src_k_base =
                (int64_t)q * key_stride0 + (int64_t)h * key_stride1;
            const int64_t src_v_base =
                (int64_t)q * value_stride0 + (int64_t)h * value_stride1;
            const T* src_k = key + src_k_base;
            const T* src_v = value + src_v_base;

            if (action == 2) {
                const int block_idx = pos / block_size;
                const int off = pos - block_idx * block_size;
                if (block_idx < 0 || block_idx >= steady_end_blocks) {
                    if (lane == 0) atomicCAS(error_code, 0, ERR_BAD_PARAM);
                } else {
                    const int bid =
                        steady_end_block_ids[(int64_t)h * steady_end_blocks + block_idx];
                    if (bid < 0 || bid >= total_blocks) {
                        if (lane == 0) atomicCAS(error_code, 0, ERR_BAD_PARAM);
                    } else {
                        const int64_t old_elem =
                            block_elem_offset(bid, off, block_size, dim);
                        const T* old_k = block_storage + old_elem;
                        const T* old_v = block_storage + pstride + old_elem;
                        T* ev_k = evicted_key + ((int64_t)evict_idx * Hkv + h) * dim;
                        T* ev_v = evicted_value + ((int64_t)evict_idx * Hkv + h) * dim;
                        strided_copy_vec(ev_k, old_k, dim, lane, WARP_SIZE);
                        strided_copy_vec(ev_v, old_v, dim, lane, WARP_SIZE);

                        T* dst_k = block_storage + old_elem;
                        T* dst_v = block_storage + pstride + old_elem;
                        strided_copy_vec(dst_k, src_k, dim, lane, WARP_SIZE, key_stride2);
                        strided_copy_vec(dst_v, src_v, dim, lane, WARP_SIZE, value_stride2);
                    }
                }
            } else if (action == 3) {
                T* ev_k = evicted_key + ((int64_t)evict_idx * Hkv + h) * dim;
                T* ev_v = evicted_value + ((int64_t)evict_idx * Hkv + h) * dim;
                strided_copy_vec(ev_k, src_k, dim, lane, WARP_SIZE, key_stride2);
                strided_copy_vec(ev_v, src_v, dim, lane, WARP_SIZE, value_stride2);
            } else {
                const bool to_start = action == 0;
                const int block_idx = pos / block_size;
                const int off = pos - block_idx * block_size;
                const int num_blocks =
                    to_start ? steady_start_blocks : steady_end_blocks;
                const int32_t* block_ids =
                    to_start ? steady_start_block_ids : steady_end_block_ids;
                if (block_idx < 0 || block_idx >= num_blocks) {
                    if (lane == 0) atomicCAS(error_code, 0, ERR_BAD_PARAM);
                } else {
                    const int bid = block_ids[(int64_t)h * num_blocks + block_idx];
                    if (bid < 0 || bid >= total_blocks) {
                        if (lane == 0) atomicCAS(error_code, 0, ERR_BAD_PARAM);
                    } else {
                        const int64_t dst_elem =
                            block_elem_offset(bid, off, block_size, dim);
                        T* dst_k = block_storage + dst_elem;
                        T* dst_v = block_storage + pstride + dst_elem;
                        strided_copy_vec(dst_k, src_k, dim, lane, WARP_SIZE, key_stride2);
                        strided_copy_vec(dst_v, src_v, dim, lane, WARP_SIZE, value_stride2);
                    }
                }
            }
        }
        __syncthreads();
        if (error_code[0] != ERR_OK) return;
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
    int64_t key_stride0,
    int64_t key_stride1,
    int64_t key_stride2,
    int64_t value_stride0,
    int64_t value_stride1,
    int64_t value_stride2,
    int32_t storage_dtype,
    const int32_t* d_label,
    void* d_cluster_centers_T,
    const void* d_mean,
    int32_t* d_cluster_center_count,
    const int32_t* d_input_token_count,
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
    const int32_t* d_steady_start_block_ids,
    int32_t steady_start_blocks,
    const int32_t* d_steady_end_block_ids,
    int32_t steady_end_blocks,
    int32_t* d_steady_state,
    int32_t steady_start_capacity,
    int32_t steady_end_capacity,
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
    if (d_steady_state != nullptr) {
        if (steady_start_capacity < 0 || steady_end_capacity < 0) {
            return ERR_BAD_PARAM;
        }
        if (steady_start_capacity > 0 &&
            (d_steady_start_block_ids == nullptr || steady_start_blocks <= 0)) {
            return ERR_BAD_PARAM;
        }
        if (steady_end_capacity > 0 &&
            (d_steady_end_block_ids == nullptr || steady_end_blocks <= 0)) {
            return ERR_BAD_PARAM;
        }
    }

    cudaError_t cerr = cudaMemsetAsync(d_error_code, 0, sizeof(int32_t), stream);
    if (cerr != cudaSuccess) return ERR_LAUNCH;

    constexpr int MAX_HKV = 8;
    constexpr int MAX_REMOVED = 8 * 256;

    int warps = (d_label == nullptr)
                    ? min(8, max(Hkv, Hkv * C))
                    : pick_warps(Nq * Hkv);
    int threads = warps * WARP_SIZE;
    size_t shared_bytes = (d_label == nullptr)
                              ? (size_t)Hkv * C * sizeof(float)
                              : 0;

    if (storage_dtype == DTYPE_FP32) {
        k_append_kv_to_clusters_persistent<float, MAX_HKV, MAX_REMOVED>
            <<<1, threads, shared_bytes, stream>>>(
                (const float*)d_key,
                (const float*)d_value,
                d_label,
                (float*)d_cluster_centers_T,
                (const float*)d_mean,
                d_cluster_center_count,
                d_input_token_count,
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
                d_steady_start_block_ids,
                steady_start_blocks,
                d_steady_end_block_ids,
                steady_end_blocks,
                d_steady_state,
                steady_start_capacity,
                steady_end_capacity,
                key_stride0,
                key_stride1,
                key_stride2,
                value_stride0,
                value_stride1,
                value_stride2,
                Nq, Hkv, C, maxB,
                total_blocks, block_size, dim,
                max_temp_blocks, max_free_block);
    } else if (storage_dtype == DTYPE_FP16) {
        k_append_kv_to_clusters_persistent<__half, MAX_HKV, MAX_REMOVED>
            <<<1, threads, shared_bytes, stream>>>(
                (const __half*)d_key,
                (const __half*)d_value,
                d_label,
                (__half*)d_cluster_centers_T,
                (const __half*)d_mean,
                d_cluster_center_count,
                d_input_token_count,
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
                d_steady_start_block_ids,
                steady_start_blocks,
                d_steady_end_block_ids,
                steady_end_blocks,
                d_steady_state,
                steady_start_capacity,
                steady_end_capacity,
                key_stride0,
                key_stride1,
                key_stride2,
                value_stride0,
                value_stride1,
                value_stride2,
                Nq, Hkv, C, maxB,
                total_blocks, block_size, dim,
                max_temp_blocks, max_free_block);
    } else if (storage_dtype == DTYPE_BF16) {
        k_append_kv_to_clusters_persistent<__nv_bfloat16, MAX_HKV, MAX_REMOVED>
            <<<1, threads, shared_bytes, stream>>>(
                (const __nv_bfloat16*)d_key,
                (const __nv_bfloat16*)d_value,
                d_label,
                (__nv_bfloat16*)d_cluster_centers_T,
                (const __nv_bfloat16*)d_mean,
                d_cluster_center_count,
                d_input_token_count,
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
                d_steady_start_block_ids,
                steady_start_blocks,
                d_steady_end_block_ids,
                steady_end_blocks,
                d_steady_state,
                steady_start_capacity,
                steady_end_capacity,
                key_stride0,
                key_stride1,
                key_stride2,
                value_stride0,
                value_stride1,
                value_stride2,
                Nq, Hkv, C, maxB,
                total_blocks, block_size, dim,
                max_temp_blocks, max_free_block);
    } else {
        return ERR_UNSUPPORTED_DTYPE;
    }

    if (cudaGetLastError() != cudaSuccess) return ERR_LAUNCH;
    return ERR_OK;
}

extern "C" int update_sparse_steady_kv_launcher_raw(
    void* d_block_storage,
    int32_t storage_dtype,
    const int32_t* d_steady_start_block_ids,
    int32_t steady_start_blocks,
    const int32_t* d_steady_end_block_ids,
    int32_t steady_end_blocks,
    int32_t* d_steady_state,
    void* d_evicted_key,
    void* d_evicted_value,
    int32_t* d_evicted_count,
    const void* d_key,
    const void* d_value,
    int64_t key_stride0,
    int64_t key_stride1,
    int64_t key_stride2,
    int64_t value_stride0,
    int64_t value_stride1,
    int64_t value_stride2,
    int Nq, int Hkv,
    int total_blocks, int block_size, int dim,
    int steady_start_capacity,
    int steady_end_capacity,
    int evicted_capacity,
    int32_t* d_error_code,
    cudaStream_t stream) {
    if (Nq <= 0 || Hkv <= 0 || Hkv > 8 ||
        total_blocks <= 0 || block_size <= 0 || dim <= 0 ||
        steady_start_capacity < 0 || steady_end_capacity < 0 ||
        evicted_capacity <= 0) {
        return ERR_BAD_PARAM;
    }
    if (steady_start_capacity > 0 && steady_start_blocks <= 0) {
        return ERR_BAD_PARAM;
    }
    if (steady_end_capacity > 0 && steady_end_blocks <= 0) {
        return ERR_BAD_PARAM;
    }

    constexpr int MAX_HKV = 8;
    const int threads = MAX_HKV * WARP_SIZE;
    if (storage_dtype == DTYPE_FP32) {
        k_update_sparse_steady_kv<float, MAX_HKV><<<1, threads, 0, stream>>>(
            (float*)d_block_storage,
            d_steady_start_block_ids,
            steady_start_blocks,
            d_steady_end_block_ids,
            steady_end_blocks,
            d_steady_state,
            (float*)d_evicted_key,
            (float*)d_evicted_value,
            d_evicted_count,
            (const float*)d_key,
            (const float*)d_value,
            key_stride0, key_stride1, key_stride2,
            value_stride0, value_stride1, value_stride2,
            Nq, Hkv, total_blocks, block_size, dim,
            steady_start_capacity, steady_end_capacity,
            evicted_capacity, d_error_code);
    } else if (storage_dtype == DTYPE_FP16) {
        k_update_sparse_steady_kv<__half, MAX_HKV><<<1, threads, 0, stream>>>(
            (__half*)d_block_storage,
            d_steady_start_block_ids,
            steady_start_blocks,
            d_steady_end_block_ids,
            steady_end_blocks,
            d_steady_state,
            (__half*)d_evicted_key,
            (__half*)d_evicted_value,
            d_evicted_count,
            (const __half*)d_key,
            (const __half*)d_value,
            key_stride0, key_stride1, key_stride2,
            value_stride0, value_stride1, value_stride2,
            Nq, Hkv, total_blocks, block_size, dim,
            steady_start_capacity, steady_end_capacity,
            evicted_capacity, d_error_code);
    } else if (storage_dtype == DTYPE_BF16) {
        k_update_sparse_steady_kv<__nv_bfloat16, MAX_HKV><<<1, threads, 0, stream>>>(
            (__nv_bfloat16*)d_block_storage,
            d_steady_start_block_ids,
            steady_start_blocks,
            d_steady_end_block_ids,
            steady_end_blocks,
            d_steady_state,
            (__nv_bfloat16*)d_evicted_key,
            (__nv_bfloat16*)d_evicted_value,
            d_evicted_count,
            (const __nv_bfloat16*)d_key,
            (const __nv_bfloat16*)d_value,
            key_stride0, key_stride1, key_stride2,
            value_stride0, value_stride1, value_stride2,
            Nq, Hkv, total_blocks, block_size, dim,
            steady_start_capacity, steady_end_capacity,
            evicted_capacity, d_error_code);
    } else {
        return ERR_UNSUPPORTED_DTYPE;
    }
    if (cudaGetLastError() != cudaSuccess) return ERR_LAUNCH;
    return ERR_OK;
}

template <typename T>
__global__ void k_sparse_select_topk_clusters(
    const T* __restrict__ query,             // [Nq,Hq,dim] (strided)
    const T* __restrict__ cluster_centers_T, // [Hkv,dim,C]
    const T* __restrict__ mean,              // [Hkv,dim]
    const int32_t* __restrict__ cluster_center_count, // [1]
    int32_t* __restrict__ top_clusters,      // [Nq,Hq,nprobe]
    int64_t query_stride0,
    int64_t query_stride1,
    int64_t query_stride2,
    int Nq,
    int Hq,
    int Hkv,
    int C,
    int dim,
    int nprobe) {
    const int row = blockIdx.x;
    const int rows = Nq * Hq;
    if (row >= rows) return;

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    extern __shared__ float scores[];

    const int q = row / Hq;
    const int hq = row - q * Hq;
    const int group_size = Hq / Hkv;
    const int hkv = hq / group_size;
    const int valid_centers = min(max(cluster_center_count[0], 0), C);

    const int64_t query_base =
        (int64_t)q * query_stride0 + (int64_t)hq * query_stride1;
    const int64_t mean_base = (int64_t)hkv * dim;
    const int64_t center_base = (int64_t)hkv * dim * C;

    for (int c = warp; c < valid_centers; c += warps_per_block) {
        T score = T(0.0f);
        for (int d = lane; d < dim; d += WARP_SIZE) {
            const T q_v = query[query_base + (int64_t)d * query_stride2];
            const T mean_v = mean[mean_base + d];
            const T center_v = cluster_centers_T[center_base + (int64_t)d * C + c];
            score = score + (q_v - mean_v) * center_v;
        }

        float score_f = static_cast<float>(score);
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            score_f += __shfl_down_sync(FULL_MASK, score_f, offset);
        }
        if (lane == 0) {
            scores[c] = score_f;
        }
    }
    __syncthreads();

    if (warp != 0) return;

    const int64_t out_base = (int64_t)row * nprobe;
    const int emit = min(nprobe, valid_centers);
    for (int k = 0; k < nprobe; ++k) {
        float best_score = -3.4028234663852886e38f;
        int best_c = INT_MAX;
        if (k < emit) {
            for (int c = lane; c < valid_centers; c += WARP_SIZE) {
                const float score = scores[c];
                if (score > best_score ||
                    (score == best_score && c < best_c)) {
                    best_score = score;
                    best_c = c;
                }
            }
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                const float other_score =
                    __shfl_down_sync(FULL_MASK, best_score, offset);
                const int other_c =
                    __shfl_down_sync(FULL_MASK, best_c, offset);
                if (other_score > best_score ||
                    (other_score == best_score && other_c < best_c)) {
                    best_score = other_score;
                    best_c = other_c;
                }
            }
        }
        if (lane == 0) {
            if (k < emit && best_c != INT_MAX) {
                top_clusters[out_base + k] = best_c;
                scores[best_c] = -3.4028234663852886e38f;
            } else {
                top_clusters[out_base + k] = -1;
            }
        }
        __syncwarp();
    }
}

extern "C" int sparse_select_topk_clusters_launcher_raw(
    const void* d_query,
    const void* d_cluster_centers_T,
    const void* d_mean,
    const int32_t* d_cluster_center_count,
    int32_t* d_top_clusters,
    int64_t query_stride0,
    int64_t query_stride1,
    int64_t query_stride2,
    int32_t storage_dtype,
    int Nq,
    int Hq,
    int Hkv,
    int C,
    int dim,
    int nprobe,
    cudaStream_t stream) {
    if (d_query == nullptr || d_cluster_centers_T == nullptr ||
        d_mean == nullptr || d_cluster_center_count == nullptr ||
        d_top_clusters == nullptr || Nq <= 0 || Hq <= 0 || Hkv <= 0 ||
        C <= 0 || dim <= 0 || nprobe <= 0 || Hq % Hkv != 0 ||
        nprobe > WARP_SIZE) {
        return ERR_BAD_PARAM;
    }

    const int rows = Nq * Hq;
    const int threads = 256;  // 8 warps/block, each warp scores clusters.
    const size_t smem = (size_t)C * sizeof(float);

    switch (storage_dtype) {
    case DTYPE_FP32:
        k_sparse_select_topk_clusters<float><<<rows, threads, smem, stream>>>(
            static_cast<const float*>(d_query),
            static_cast<const float*>(d_cluster_centers_T),
            static_cast<const float*>(d_mean),
            d_cluster_center_count,
            d_top_clusters,
            query_stride0,
            query_stride1,
            query_stride2,
            Nq,
            Hq,
            Hkv,
            C,
            dim,
            nprobe);
        break;
    case DTYPE_FP16:
        k_sparse_select_topk_clusters<__half><<<rows, threads, smem, stream>>>(
            static_cast<const __half*>(d_query),
            static_cast<const __half*>(d_cluster_centers_T),
            static_cast<const __half*>(d_mean),
            d_cluster_center_count,
            d_top_clusters,
            query_stride0,
            query_stride1,
            query_stride2,
            Nq,
            Hq,
            Hkv,
            C,
            dim,
            nprobe);
        break;
    case DTYPE_BF16:
        k_sparse_select_topk_clusters<__nv_bfloat16><<<rows, threads, smem, stream>>>(
            static_cast<const __nv_bfloat16*>(d_query),
            static_cast<const __nv_bfloat16*>(d_cluster_centers_T),
            static_cast<const __nv_bfloat16*>(d_mean),
            d_cluster_center_count,
            d_top_clusters,
            query_stride0,
            query_stride1,
            query_stride2,
            Nq,
            Hq,
            Hkv,
            C,
            dim,
            nprobe);
        break;
    default:
        return ERR_UNSUPPORTED_DTYPE;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ERR_LAUNCH;
    }
    return ERR_OK;
}
