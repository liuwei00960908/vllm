#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>

#include "sparse_attention_common.h"

enum class SparseBlockTableKernelKind : int {
    MetadataAndPlan = 0,
    CopyFromPlan = 1,
};

__host__ __device__ __forceinline__ int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

static inline int pick_threads(
    SparseBlockTableKernelKind kind,
    int rows,
    int nprobe = 0,
    int block_size = 0,
    int dim = 0) {
    (void)rows;
    (void)nprobe;
    (void)block_size;
    (void)dim;

    switch (kind) {
        case SparseBlockTableKernelKind::MetadataAndPlan:
            return 128;  // 4 warps/block, 1 warp = 1 row

        case SparseBlockTableKernelKind::CopyFromPlan:
            return 128;  // 4 warps/block, 1 warp = 1 plan entry
    }

    return 128;
}

static inline int pick_blocks(
    SparseBlockTableKernelKind kind,
    int rows,
    int threads,
    int nprobe = 0) {
    (void)nprobe;

    switch (kind) {
        case SparseBlockTableKernelKind::MetadataAndPlan: {
            const int warps_per_block = threads / WARP_SIZE;
            return ceil_div_int(rows, warps_per_block);
        }

        case SparseBlockTableKernelKind::CopyFromPlan:
            return 64;
    }

    return 1;
}

__device__ __forceinline__ void warp_set_error_and_trap(
    int32_t* err, int32_t code, int lane) {
    if (lane == 0) {
        int old = atomicCAS(err, 0, code);
        if (old == 0) {
            printf("\n================\nCUDA error_code=%d\n====================\n", code);
        }
    }
    // asm volatile("trap;");
}

template <typename T>
__device__ __forceinline__ void warp_copy_one_token(
    T* __restrict__ block_storage,
    int total_blocks, int block_size, int dim,
    int src_bid, int src_tok_offset,
    int dst_bid, int dst_tok_offset,
    int lane) {

    using Vec = int4;
    constexpr int VEC_BYTES = sizeof(Vec);
    constexpr int elems_per_vec = VEC_BYTES / (int)sizeof(T);

    const bool use_vec = ((dim % elems_per_vec) == 0);

    const int64_t plane_stride = (int64_t)total_blocks * block_size * dim;
    const int64_t src_base = (int64_t)src_bid * block_size * dim
                           + (int64_t)src_tok_offset * dim;
    const int64_t dst_base = (int64_t)dst_bid * block_size * dim
                           + (int64_t)dst_tok_offset * dim;

    #pragma unroll
    for (int plane = 0; plane < 2; ++plane) {
        T* src_plane = block_storage + plane * plane_stride + src_base;
        T* dst_plane = block_storage + plane * plane_stride + dst_base;

        if (use_vec) {
            const int n_vec = dim / elems_per_vec;
            const Vec* sv = reinterpret_cast<const Vec*>(src_plane);
            Vec* dv = reinterpret_cast<Vec*>(dst_plane);
            for (int i = lane; i < n_vec; i += WARP_SIZE) {
                dv[i] = sv[i];
            }
        } else {
            for (int i = lane; i < dim; i += WARP_SIZE) {
                dst_plane[i] = src_plane[i];
            }
        }
    }
}

__global__ void k_build_sparse_block_table_metadata_and_plan(
    const int32_t* __restrict__ top_clusters,
    const int32_t* __restrict__ cluster_compact_block_ids,
    const int32_t* __restrict__ cluster_temp_kv_pos,
    const int32_t* __restrict__ cluster_total_kv_counts,
    const int32_t* __restrict__ temp_block_ids,
    const int32_t* __restrict__ free_block_ids,
    const int32_t* __restrict__ steady_start_block_ids,
    int steady_start_blocks,
    const int32_t* __restrict__ steady_end_block_ids,
    int steady_end_blocks,
    const int32_t* __restrict__ steady_state,
    int32_t* __restrict__ out_block_table,
    int32_t* __restrict__ out_bt_len,
    int32_t* __restrict__ out_seqused_k,
    int32_t* __restrict__ out_row_free_base,
    int32_t* __restrict__ row_plan_base,
    int32_t* __restrict__ total_plan_count,
    int32_t* __restrict__ plan_row,
    int32_t* __restrict__ plan_src_tb_idx,
    int32_t* __restrict__ plan_src_tb_off,
    int32_t* __restrict__ error_code,
    int32_t* __restrict__ used_free_block_count,
    int NQ, int Hq, int nprobe,
    int Hkv, int C, int maxB,
    int total_blocks, int block_size,
    int max_bt_len,
    int max_free_block) {

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id_in_block = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int row = blockIdx.x * warps_per_block + warp_id_in_block;

    const int rows = NQ * Hq;
    if (row >= rows) return;

    const int64_t row_base = (int64_t)row * max_bt_len;

    // Fused replacement of torch::full(out_block_table, -1).
    for (int i = lane; i < max_bt_len; i += WARP_SIZE) {
        out_block_table[row_base + i] = -1;
    }

    if (error_code[0] != ERR_OK) return;

    const int q = row / Hq;
    const int hq = row - q * Hq;

    if (Hq % Hkv != 0) {
        warp_set_error_and_trap(error_code, ERR_HQ_HKV_MISMATCH, lane);
        return;
    }

    const int q_per_kv = Hq / Hkv;
    const int hkv = hq / q_per_kv;

    const int64_t top_base = ((int64_t)q * Hq + hq) * nprobe;
    const int64_t compact_base_h = (int64_t)hkv * C * maxB;
    const int64_t total_base_h = (int64_t)hkv * C;
    const int64_t temp_pos_base_h = (int64_t)hkv * C * block_size * 2;
    const int steady_start_count = steady_state[1];
    const int steady_end_count = steady_state[2];
    const int steady_start_full = steady_start_count / block_size;
    const int steady_start_tail = steady_start_count - steady_start_full * block_size;
    const int steady_end_full = steady_end_count / block_size;
    const int steady_end_tail = steady_end_count - steady_end_full * block_size;
    const int steady_full_len = steady_start_full + steady_end_full;
    const int steady_tail_sum = steady_start_tail + steady_end_tail;

    int my_full_nb = 0;
    int my_tail = 0;
    int my_seq = 0;
    int my_cid = -1;

    if (lane < nprobe) {
        my_cid = top_clusters[top_base + lane];
        if (my_cid >= 0 && my_cid < C) {
            const int total_cnt = cluster_total_kv_counts[total_base_h + my_cid];
            if (total_cnt > 0) {
                my_seq = total_cnt;
                my_full_nb = total_cnt / block_size;
                my_tail = total_cnt % block_size;

                if (my_full_nb > maxB) {
                    printf("\n===========\nmaxB=%d, my_full_nb=%d, cluster_total_kv_counts=%d, block_size=%d\n==========\n", 
                        maxB, my_full_nb, total_cnt, block_size);
                    warp_set_error_and_trap(error_code, ERR_NB_OVERFLOW, lane);
                    return;
                }
            }
        } else {
            my_cid = -1;
        }
    }

    int scan_full = my_full_nb;
    int scan_tail = my_tail;
    int scan_seq = my_seq;

    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        const int y_full = __shfl_up_sync(FULL_MASK, scan_full, offset);
        const int y_tail = __shfl_up_sync(FULL_MASK, scan_tail, offset);
        const int y_seq = __shfl_up_sync(FULL_MASK, scan_seq, offset);

        if (lane >= offset) {
            scan_full += y_full;
            scan_tail += y_tail;
            scan_seq += y_seq;
        }
    }

    const int my_write_base = scan_full - my_full_nb;
    const int my_tail_prefix = scan_tail - my_tail;

    int compact_len = 0;
    int temp_sum = 0;
    int seq_sum = 0;

    if (nprobe > 0) {
        compact_len = __shfl_sync(FULL_MASK, scan_full, nprobe - 1);
        temp_sum = __shfl_sync(FULL_MASK, scan_tail, nprobe - 1);
        seq_sum = __shfl_sync(FULL_MASK, scan_seq, nprobe - 1);
    }

    int free_base = 0;
    int base_plan = 0;
    int row_error = ERR_OK;

    if (lane == 0) {
        const int pack_sum = steady_tail_sum + temp_sum;
        const int free_blocks_needed = ceil_div_int(pack_sum, block_size);
        const int bt_len = steady_full_len + compact_len + free_blocks_needed;

        if (steady_start_full + (steady_start_tail > 0) > steady_start_blocks ||
            steady_end_full + (steady_end_tail > 0) > steady_end_blocks) {
            row_error = ERR_BAD_PARAM;
        } else if (bt_len > max_bt_len) {
            row_error = ERR_BT_LEN_OVERFLOW;
        } else {
            if (free_blocks_needed > 0) {
                free_base = atomicAdd(used_free_block_count, free_blocks_needed);
                if (free_base + free_blocks_needed > max_free_block) {
                    row_error = ERR_NO_FREE_BLOCK;
                }
            }

            if (row_error == ERR_OK && pack_sum > 0) {
                base_plan = atomicAdd(total_plan_count, pack_sum);
            }

            out_row_free_base[row] = free_base;
            row_plan_base[row] = base_plan;
            out_bt_len[row] = bt_len;
            out_seqused_k[row] = steady_start_count + steady_end_count + seq_sum;

            if (row_error == ERR_OK) {
                for (int i = 0; i < steady_start_full; ++i) {
                    const int bid =
                        steady_start_block_ids[(int64_t)hkv * steady_start_blocks + i];
                    if (bid < 0 || bid >= total_blocks) {
                        row_error = ERR_BAD_PARAM;
                        break;
                    }
                    out_block_table[row_base + i] = bid;
                }
            }
            if (row_error == ERR_OK) {
                for (int i = 0; i < steady_end_full; ++i) {
                    const int bid =
                        steady_end_block_ids[(int64_t)hkv * steady_end_blocks + i];
                    if (bid < 0 || bid >= total_blocks) {
                        row_error = ERR_BAD_PARAM;
                        break;
                    }
                    out_block_table[row_base + steady_start_full + i] = bid;
                }
            }
            if (row_error == ERR_OK) {
                for (int i = 0; i < free_blocks_needed; ++i) {
                    const int bid = free_block_ids[free_base + i];
                    if (bid < 0 || bid >= total_blocks) {
                        row_error = ERR_BAD_PARAM;
                        break;
                    }
                    out_block_table[row_base + steady_full_len + compact_len + i] =
                        bid;
                }
            }
        }
    }

    row_error = __shfl_sync(FULL_MASK, row_error, 0);
    free_base = __shfl_sync(FULL_MASK, free_base, 0);
    base_plan = __shfl_sync(FULL_MASK, base_plan, 0);

    if (row_error != ERR_OK) {
        warp_set_error_and_trap(error_code, row_error, lane);
        return;
    }

    const int lanes_per_cluster_raw = WARP_SIZE / nprobe;
    const int lanes_per_cluster = lanes_per_cluster_raw > 0 ? lanes_per_cluster_raw : 1;
    const int covered_lanes = lanes_per_cluster * nprobe;

    if (steady_start_tail > 0) {
        const int bid = steady_start_block_ids[
            (int64_t)hkv * steady_start_blocks + steady_start_full];
        if (bid < 0 || bid >= total_blocks) {
            warp_set_error_and_trap(error_code, ERR_BAD_PARAM, lane);
            return;
        }
        for (int slot = lane; slot < steady_start_tail; slot += WARP_SIZE) {
            const int p = base_plan + slot;
            plan_row[p] = row;
            plan_src_tb_idx[p] = bid;
            plan_src_tb_off[p] = slot;
        }
    }

    if (steady_end_tail > 0) {
        const int bid = steady_end_block_ids[
            (int64_t)hkv * steady_end_blocks + steady_end_full];
        if (bid < 0 || bid >= total_blocks) {
            warp_set_error_and_trap(error_code, ERR_BAD_PARAM, lane);
            return;
        }
        const int tail_base = base_plan + steady_start_tail;
        for (int slot = lane; slot < steady_end_tail; slot += WARP_SIZE) {
            const int p = tail_base + slot;
            plan_row[p] = row;
            plan_src_tb_idx[p] = bid;
            plan_src_tb_off[p] = slot;
        }
    }

    // Generate the per-token copy plan. Do this even when compact_len == 0,
    // because a row can have only tail tokens and no compact full blocks.
    if (lane < covered_lanes) {
        const int cluster_idx = lane / lanes_per_cluster;
        const int lane_in_cluster = lane - cluster_idx * lanes_per_cluster;

        const int cid = __shfl_sync(FULL_MASK, my_cid, cluster_idx);
        const int tail = __shfl_sync(FULL_MASK, my_tail, cluster_idx);
        const int tail_prefix = __shfl_sync(FULL_MASK, my_tail_prefix, cluster_idx);

        if (cid >= 0 && tail > 0) {
            const int64_t temp_pos_base =
                temp_pos_base_h + (int64_t)cid * block_size * 2;

            const int plan_base = base_plan + steady_tail_sum + tail_prefix;

            for (int slot = lane_in_cluster; slot < tail; slot += lanes_per_cluster) {
                const int tb_idx =
                    cluster_temp_kv_pos[temp_pos_base + (int64_t)slot * 2 + 0];
                const int tb_off =
                    cluster_temp_kv_pos[temp_pos_base + (int64_t)slot * 2 + 1];

                if (tb_idx < 0 || tb_off < 0 || tb_off >= block_size) {
                    warp_set_error_and_trap(error_code, ERR_BAD_PARAM, lane);
                    return;
                }

                const int src_bid = temp_block_ids[tb_idx];
                if (src_bid < 0 || src_bid >= total_blocks) {
                    warp_set_error_and_trap(error_code, ERR_BAD_PARAM, lane);
                    return;
                }

                const int p = plan_base + slot;
                plan_row[p] = row;
                plan_src_tb_idx[p] = src_bid;
                plan_src_tb_off[p] = tb_off;
            }
        }
    }

    // Write compact full blocks into the beginning of block_table.
    if (compact_len > 0 && lane < covered_lanes) {
        const int cluster_idx = lane / lanes_per_cluster;
        const int lane_in_cluster = lane - cluster_idx * lanes_per_cluster;

        const int cid = __shfl_sync(FULL_MASK, my_cid, cluster_idx);
        const int full_nb = __shfl_sync(FULL_MASK, my_full_nb, cluster_idx);
        const int write_base = __shfl_sync(FULL_MASK, my_write_base, cluster_idx);

        if (cid >= 0 && full_nb > 0) {
            const int64_t compact_base = compact_base_h + (int64_t)cid * maxB;

            for (int bi = lane_in_cluster; bi < full_nb; bi += lanes_per_cluster) {
                const int bid = cluster_compact_block_ids[compact_base + bi];
                if (bid < 0 || bid >= total_blocks) {
                    warp_set_error_and_trap(error_code, ERR_BAD_PARAM, lane);
                    return;
                }
                out_block_table[row_base + steady_full_len + write_base + bi] =
                    bid;
            }
        }
    }
}

template <typename T>
__global__ void k_build_sparse_block_table_copy_from_plan_src_only(
    const int32_t* __restrict__ total_plan_count_ptr,
    const int32_t* __restrict__ plan_row,
    const int32_t* __restrict__ plan_src_tb_idx,
    const int32_t* __restrict__ plan_src_tb_off,
    const int32_t* __restrict__ row_plan_base,
    const int32_t* __restrict__ row_free_base,
    const int32_t* __restrict__ temp_block_ids,
    const int32_t* __restrict__ free_block_ids,
    T* __restrict__ block_storage,
    int total_blocks,
    int block_size,
    int dim) {

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id_in_block = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    const int global_warp_id = blockIdx.x * warps_per_block + warp_id_in_block;
    const int warp_stride = gridDim.x * warps_per_block;

    const int total_plan_count = total_plan_count_ptr[0];

    for (int p = global_warp_id; p < total_plan_count; p += warp_stride) {
        const int row = plan_row[p];
        const int src_bid = plan_src_tb_idx[p];
        const int src_off = plan_src_tb_off[p];

        const int row_base_plan = row_plan_base[row];
        const int local_idx = p - row_base_plan;

        const int dst_local_block = local_idx / block_size;
        const int dst_off = local_idx - dst_local_block * block_size;
        const int dst_bid = free_block_ids[row_free_base[row] + dst_local_block];

        warp_copy_one_token<T>(
            block_storage,
            total_blocks, block_size, dim,
            src_bid, src_off,
            dst_bid, dst_off,
            lane);
    }
}

extern "C" int build_sparse_block_table_launcher_raw(
    const int32_t* d_top_clusters,
    const int32_t* d_cluster_compact_block_ids,
    const int32_t* d_cluster_temp_kv_pos,
    const int32_t* d_cluster_total_kv_counts,
    const int32_t* d_temp_block_ids,
    void* d_block_storage,
    int32_t storage_dtype,
    const int32_t* d_free_block_ids,
    int32_t max_free_block,
    const int32_t* d_steady_start_block_ids,
    int32_t steady_start_blocks,
    const int32_t* d_steady_end_block_ids,
    int32_t steady_end_blocks,
    const int32_t* d_steady_state,
    int32_t* d_out_block_table,
    int32_t* d_out_bt_len,
    int32_t* d_out_seqused_k,
    int32_t* d_state,
    int32_t* d_row_free_base,
    int32_t* d_row_plan_base,
    int32_t* d_plan_row,
    int32_t* d_plan_src_tb_idx,
    int32_t* d_plan_src_tb_off,
    int NQ, int Hq, int nprobe,
    int Hkv, int C, int maxB,
    int total_blocks, int block_size, int dim,
    int max_bt_len,
    cudaStream_t stream) {

    if (NQ <= 0 || Hq <= 0 || nprobe <= 0 ||
        Hkv <= 0 || C <= 0 || maxB <= 0 ||
        total_blocks <= 0 || block_size <= 0 || dim <= 0 ||
        max_bt_len <= 0 || max_free_block <= 0) {
        return ERR_BAD_PARAM;
    }

    if (Hq % Hkv != 0) return ERR_HQ_HKV_MISMATCH;
    if (nprobe > WARP_SIZE) return ERR_BAD_PARAM;

    const int rows = NQ * Hq;

    int32_t* d_error_code = d_state + 0;
    int32_t* d_used_free_block_count = d_state + 1;
    int32_t* d_total_plan_count = d_state + 2;

    cudaError_t cerr;
    cerr = cudaMemsetAsync(d_state, 0, 3 * sizeof(int32_t), stream);
    if (cerr != cudaSuccess) return ERR_LAUNCH;

    {
        const int threads = pick_threads(
            SparseBlockTableKernelKind::MetadataAndPlan,
            rows,
            nprobe,
            block_size,
            dim);

        const int blocks = pick_blocks(
            SparseBlockTableKernelKind::MetadataAndPlan,
            rows,
            threads,
            nprobe);

        k_build_sparse_block_table_metadata_and_plan<<<blocks, threads, 0, stream>>>(
            d_top_clusters,
            d_cluster_compact_block_ids,
            d_cluster_temp_kv_pos,
            d_cluster_total_kv_counts,
            d_temp_block_ids,
            d_free_block_ids,
            d_steady_start_block_ids,
            steady_start_blocks,
            d_steady_end_block_ids,
            steady_end_blocks,
            d_steady_state,
            d_out_block_table,
            d_out_bt_len,
            d_out_seqused_k,
            d_row_free_base,
            d_row_plan_base,
            d_total_plan_count,
            d_plan_row,
            d_plan_src_tb_idx,
            d_plan_src_tb_off,
            d_error_code,
            d_used_free_block_count,
            NQ, Hq, nprobe,
            Hkv, C, maxB,
            total_blocks, block_size,
            max_bt_len,
            max_free_block);

        if (cudaGetLastError() != cudaSuccess) return ERR_LAUNCH;
    }

    {
        const int threads = pick_threads(
            SparseBlockTableKernelKind::CopyFromPlan,
            rows,
            nprobe,
            block_size,
            dim);

        const int blocks = pick_blocks(
            SparseBlockTableKernelKind::CopyFromPlan,
            rows,
            threads,
            nprobe);

        if (storage_dtype == DTYPE_FP32) {
            k_build_sparse_block_table_copy_from_plan_src_only<float><<<blocks, threads, 0, stream>>>(
                d_total_plan_count,
                d_plan_row,
                d_plan_src_tb_idx,
                d_plan_src_tb_off,
                d_row_plan_base,
                d_row_free_base,
                d_temp_block_ids,
                d_free_block_ids,
                (float*)d_block_storage,
                total_blocks,
                block_size,
                dim);
        } else if (storage_dtype == DTYPE_FP16) {
            k_build_sparse_block_table_copy_from_plan_src_only<__half><<<blocks, threads, 0, stream>>>(
                d_total_plan_count,
                d_plan_row,
                d_plan_src_tb_idx,
                d_plan_src_tb_off,
                d_row_plan_base,
                d_row_free_base,
                d_temp_block_ids,
                d_free_block_ids,
                (__half*)d_block_storage,
                total_blocks,
                block_size,
                dim);
        } else if (storage_dtype == DTYPE_BF16) {
            k_build_sparse_block_table_copy_from_plan_src_only<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
                d_total_plan_count,
                d_plan_row,
                d_plan_src_tb_idx,
                d_plan_src_tb_off,
                d_row_plan_base,
                d_row_free_base,
                d_temp_block_ids,
                d_free_block_ids,
                (__nv_bfloat16*)d_block_storage,
                total_blocks,
                block_size,
                dim);
        } else {
            return ERR_UNSUPPORTED_DTYPE;
        }

        if (cudaGetLastError() != cudaSuccess) return ERR_LAUNCH;
    }

    return ERR_OK;
}