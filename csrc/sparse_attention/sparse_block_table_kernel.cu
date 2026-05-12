#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>

#include "sparse_attention_common.h"

enum class SparseBlockTableKernelKind : int {
    Metadata = 0,
    Allocate = 1,
    CopyPlan = 2,
    CopyFromPlan = 3,
    Append = 4,
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
        case SparseBlockTableKernelKind::Metadata:
            return 128;

        case SparseBlockTableKernelKind::Allocate:
            return 256;

        case SparseBlockTableKernelKind::CopyPlan:
            return 128;  // 4 warps/block, 1 warp = 1 row

        case SparseBlockTableKernelKind::CopyFromPlan:
            return 128;  // 4 warps/block, 1 warp = 1 plan entry

        case SparseBlockTableKernelKind::Append:
            return 256;
    }
    return 256;
}

static inline int pick_blocks(
    SparseBlockTableKernelKind kind,
    int rows,
    int threads,
    int nprobe = 0) {
    (void)nprobe;

    switch (kind) {
        case SparseBlockTableKernelKind::Metadata: {
            int warps_per_block = threads / WARP_SIZE;
            return ceil_div_int(rows, warps_per_block);
        }

        case SparseBlockTableKernelKind::Allocate:
            return ceil_div_int(rows, threads);

        case SparseBlockTableKernelKind::CopyPlan: {
            int warps_per_block = threads / WARP_SIZE;
            return ceil_div_int(rows, warps_per_block);
        }

        case SparseBlockTableKernelKind::CopyFromPlan:
            return 64;

        case SparseBlockTableKernelKind::Append:
            return ceil_div_int(rows, threads);
    }
    return ceil_div_int(rows, threads);
}

__device__ __forceinline__ void warp_set_error_and_trap(
    int32_t* err, int32_t code, int lane) {
    if (lane == 0) {
        int old = atomicCAS(err, 0, code);
        if (old == 0) {
            printf("CUDA error_code=%d\n", code);
        }
    }
    asm volatile("trap;");
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

__global__ void k_build_sparse_block_table_metadata(
    const int32_t* __restrict__ top_clusters,
    const int32_t* __restrict__ cluster_compact_block_ids,
    const int32_t* __restrict__ cluster_total_kv_counts,
    int32_t* __restrict__ out_block_table,
    int32_t* __restrict__ out_compact_len,
    int32_t* __restrict__ out_free_blocks_needed,
    int32_t* __restrict__ out_temp_token_count,
    int32_t* __restrict__ out_seqused_k,
    int32_t* __restrict__ error_code,
    int NQ, int Hq, int nprobe,
    int Hkv, int C, int maxB,
    int total_blocks, int block_size,
    int max_bt_len) {

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id_in_block = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int row = blockIdx.x * warps_per_block + warp_id_in_block;

    const int rows = NQ * Hq;
    if (row >= rows) return;

    const int q = row / Hq;
    const int hq = row % Hq;

    if (Hq % Hkv != 0) {
        warp_set_error_and_trap(error_code, ERR_HQ_HKV_MISMATCH, lane);
        return;
    }

    const int q_per_kv = Hq / Hkv;
    const int hkv = hq / q_per_kv;

    const int64_t top_base = ((int64_t)q * Hq + hq) * nprobe;
    const int64_t compact_base_h = (int64_t)hkv * C * maxB;
    const int64_t total_base_h = (int64_t)hkv * C;

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
                    warp_set_error_and_trap(error_code, ERR_NB_OVERFLOW, lane);
                    return;
                }
            }
        } else {
            my_cid = -1;
        }
    }

    const unsigned mask = __ballot_sync(FULL_MASK, lane < nprobe);

    int scan_full = my_full_nb;
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        int y = __shfl_up_sync(mask, scan_full, offset);
        if (lane >= offset) scan_full += y;
    }
    const int my_write_base = scan_full - my_full_nb;

    int seq_sum = my_seq;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        seq_sum += __shfl_down_sync(mask, seq_sum, offset);
    }

    int temp_sum = my_tail;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        temp_sum += __shfl_down_sync(mask, temp_sum, offset);
    }

    int compact_len = 0;
    if (nprobe > 0) {
        compact_len = __shfl_sync(mask, scan_full, nprobe - 1);
    }

    if (lane == 0) {
        const int free_blocks_needed = ceil_div_int(temp_sum, block_size);
        if (compact_len + free_blocks_needed > max_bt_len) {
            atomicCAS(error_code, 0, ERR_BT_LEN_OVERFLOW);
            return;
        }
        out_compact_len[row] = compact_len;
        out_free_blocks_needed[row] = free_blocks_needed;
        out_temp_token_count[row] = temp_sum;
        out_seqused_k[row] = seq_sum;
    }

    if (compact_len <= 0) return;

    const int lanes_per_cluster = max(1, WARP_SIZE / nprobe);
    const int covered_lanes = lanes_per_cluster * nprobe;

    if (lane < covered_lanes) {
        const int cluster_idx = lane / lanes_per_cluster;
        const int lane_in_cluster = lane % lanes_per_cluster;

        int cid = __shfl_sync(mask, my_cid, cluster_idx);
        int full_nb = __shfl_sync(mask, my_full_nb, cluster_idx);
        int write_base = __shfl_sync(mask, my_write_base, cluster_idx);

        if (cid >= 0 && full_nb > 0) {
            const int64_t compact_base = compact_base_h + (int64_t)cid * maxB;
            const int64_t row_base = (int64_t)row * max_bt_len;

            for (int bi = lane_in_cluster; bi < full_nb; bi += lanes_per_cluster) {
                const int bid = cluster_compact_block_ids[compact_base + bi];
                if (bid < 0 || bid >= total_blocks) {
                    atomicCAS(error_code, 0, ERR_BAD_PARAM);
                    return;
                }
                out_block_table[row_base + write_base + bi] = bid;
            }
        }
    }
}

__global__ void k_build_sparse_block_table_allocate(
    const int32_t* __restrict__ free_blocks_needed,
    int32_t* __restrict__ out_row_free_base,
    int32_t* __restrict__ used_free_block_count,
    int32_t* __restrict__ error_code,
    int rows,
    int max_free_block) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= rows) return;

    const int need = free_blocks_needed[tid];
    if (need < 0) {
        atomicCAS(error_code, 0, ERR_BAD_PARAM);
        return;
    }

    int base = 0;
    if (need > 0) {
        base = atomicAdd(used_free_block_count, need);
        if (base + need > max_free_block) {
            atomicCAS(error_code, 0, ERR_NO_FREE_BLOCK);
            return;
        }
    }
    out_row_free_base[tid] = base;
}

__global__ void k_build_sparse_block_table_copy_plan_src_only(
    const int32_t* __restrict__ top_clusters,
    const int32_t* __restrict__ cluster_temp_kv_pos,
    const int32_t* __restrict__ cluster_total_kv_counts,
    const int32_t* __restrict__ temp_block_ids,
    const int32_t* __restrict__ temp_token_count,
    int32_t* __restrict__ row_plan_base,
    int32_t* __restrict__ total_plan_count,
    int32_t* __restrict__ plan_row,
    int32_t* __restrict__ plan_src_tb_idx,
    int32_t* __restrict__ plan_src_tb_off,
    int32_t* __restrict__ error_code,
    int NQ, int Hq, int nprobe,
    int Hkv, int C,
    int total_blocks, int block_size) {

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id_in_block = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int row = blockIdx.x * warps_per_block + warp_id_in_block;
    const int rows = NQ * Hq;

    if (row >= rows) return;

    const int q = row / Hq;
    const int hq = row % Hq;

    if (Hq % Hkv != 0) {
        warp_set_error_and_trap(error_code, ERR_HQ_HKV_MISMATCH, lane);
        return;
    }

    const int q_per_kv = Hq / Hkv;
    const int hkv = hq / q_per_kv;

    const int64_t top_base = ((int64_t)q * Hq + hq) * nprobe;
    const int64_t total_base_h = (int64_t)hkv * C;
    const int64_t temp_pos_base_h = (int64_t)hkv * C * block_size * 2;

    int my_tail = 0;
    int my_cid = -1;
    int my_prefix = 0;

    if (lane < nprobe) {
        my_cid = top_clusters[top_base + lane];
        if (my_cid >= 0 && my_cid < C) {
            const int total_cnt = cluster_total_kv_counts[total_base_h + my_cid];
            if (total_cnt > 0) {
                my_tail = total_cnt % block_size;
            }
        } else {
            my_cid = -1;
        }
    }

    const unsigned mask = __ballot_sync(FULL_MASK, lane < nprobe);

    int scan = my_tail;
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        int y = __shfl_up_sync(mask, scan, offset);
        if (lane >= offset) scan += y;
    }
    my_prefix = scan - my_tail;

    int row_total = 0;
    if (nprobe > 0) {
        row_total = __shfl_sync(mask, scan, nprobe - 1);
    }

    int base_plan = 0;
    if (lane == 0) {
        const int expected = temp_token_count[row];
        if (row_total != expected) {
            warp_set_error_and_trap(error_code, ERR_BAD_PARAM, lane);
            return;
        }
        if (row_total > 0) {
            base_plan = atomicAdd(total_plan_count, row_total);
        }
        row_plan_base[row] = base_plan;
    }
    base_plan = __shfl_sync(FULL_MASK, base_plan, 0);

    if (row_total <= 0) return;

    const int lanes_per_cluster = max(1, WARP_SIZE / nprobe);
    const int covered_lanes = lanes_per_cluster * nprobe;

    if (lane < covered_lanes) {
        const int cluster_idx = lane / lanes_per_cluster;
        const int lane_in_cluster = lane % lanes_per_cluster;

        const int cid = __shfl_sync(FULL_MASK, my_cid, cluster_idx);
        const int tail = __shfl_sync(FULL_MASK, my_tail, cluster_idx);
        const int prefix = __shfl_sync(FULL_MASK, my_prefix, cluster_idx);

        if (cid >= 0 && tail > 0) {
            const int64_t temp_pos_base = temp_pos_base_h + (int64_t)cid * block_size * 2;
            const int plan_base = base_plan + prefix;

            for (int slot = lane_in_cluster; slot < tail; slot += lanes_per_cluster) {
                const int tb_idx = cluster_temp_kv_pos[temp_pos_base + (int64_t)slot * 2 + 0];
                const int tb_off = cluster_temp_kv_pos[temp_pos_base + (int64_t)slot * 2 + 1];

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
                plan_src_tb_idx[p] = tb_idx;
                plan_src_tb_off[p] = tb_off;
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
        const int tb_idx = plan_src_tb_idx[p];
        const int src_off = plan_src_tb_off[p];

        const int src_bid = temp_block_ids[tb_idx];
        const int row_base_plan = row_plan_base[row];
        const int local_idx = p - row_base_plan;

        const int dst_local_block = local_idx / block_size;
        const int dst_off = local_idx % block_size;
        const int dst_bid = free_block_ids[row_free_base[row] + dst_local_block];

        warp_copy_one_token<T>(
            block_storage,
            total_blocks, block_size, dim,
            src_bid, src_off,
            dst_bid, dst_off,
            lane);
    }
}

__global__ void k_build_sparse_block_table_append(
    const int32_t* __restrict__ compact_len,
    const int32_t* __restrict__ free_blocks_needed,
    const int32_t* __restrict__ row_free_base,
    const int32_t* __restrict__ free_block_ids,
    int32_t* __restrict__ out_block_table,
    int32_t* __restrict__ out_bt_len,
    int32_t* __restrict__ error_code,
    int rows,
    int total_blocks,
    int max_bt_len) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= rows) return;

    const int write0 = compact_len[tid];
    const int need = free_blocks_needed[tid];
    const int base = row_free_base[tid];

    if (write0 < 0 || need < 0 || write0 + need > max_bt_len) {
        atomicCAS(error_code, 0, ERR_BT_LEN_OVERFLOW);
        return;
    }

    int write = write0;
    for (int i = 0; i < need; ++i) {
        const int bid = free_block_ids[base + i];
        if (bid < 0 || bid >= total_blocks) {
            atomicCAS(error_code, 0, ERR_BAD_PARAM);
            return;
        }
        out_block_table[(int64_t)tid * max_bt_len + write] = bid;
        ++write;
    }
    out_bt_len[tid] = write;
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
    int32_t* d_out_block_table,
    int32_t* d_out_bt_len,
    int32_t* d_out_seqused_k,
    int32_t* d_state,
    int32_t* d_compact_len,
    int32_t* d_free_blocks_needed,
    int32_t* d_temp_token_count,
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
        const int threads = pick_threads(SparseBlockTableKernelKind::Metadata, rows, nprobe, block_size, dim);
        const int blocks = pick_blocks(SparseBlockTableKernelKind::Metadata, rows, threads, nprobe);

        k_build_sparse_block_table_metadata<<<blocks, threads, 0, stream>>>(
            d_top_clusters,
            d_cluster_compact_block_ids,
            d_cluster_total_kv_counts,
            d_out_block_table,
            d_compact_len,
            d_free_blocks_needed,
            d_temp_token_count,
            d_out_seqused_k,
            d_error_code,
            NQ, Hq, nprobe,
            Hkv, C, maxB,
            total_blocks, block_size,
            max_bt_len);
        if (cudaGetLastError() != cudaSuccess) return ERR_LAUNCH;
    }

    {
        const int threads = pick_threads(SparseBlockTableKernelKind::Allocate, rows, nprobe, block_size, dim);
        const int blocks = pick_blocks(SparseBlockTableKernelKind::Allocate, rows, threads, nprobe);

        k_build_sparse_block_table_allocate<<<blocks, threads, 0, stream>>>(
            d_free_blocks_needed,
            d_row_free_base,
            d_used_free_block_count,
            d_error_code,
            rows,
            max_free_block);
        if (cudaGetLastError() != cudaSuccess) return ERR_LAUNCH;
    }

    {
        const int threads = pick_threads(SparseBlockTableKernelKind::CopyPlan, rows, nprobe, block_size, dim);
        const int blocks = pick_blocks(SparseBlockTableKernelKind::CopyPlan, rows, threads, nprobe);

        k_build_sparse_block_table_copy_plan_src_only<<<blocks, threads, 0, stream>>>(
            d_top_clusters,
            d_cluster_temp_kv_pos,
            d_cluster_total_kv_counts,
            d_temp_block_ids,
            d_temp_token_count,
            d_row_plan_base,
            d_total_plan_count,
            d_plan_row,
            d_plan_src_tb_idx,
            d_plan_src_tb_off,
            d_error_code,
            NQ, Hq, nprobe,
            Hkv, C,
            total_blocks, block_size);
        if (cudaGetLastError() != cudaSuccess) return ERR_LAUNCH;
    }

    {
        const int threads = pick_threads(SparseBlockTableKernelKind::CopyFromPlan, rows, nprobe, block_size, dim);
        const int blocks = pick_blocks(SparseBlockTableKernelKind::CopyFromPlan, rows, threads, nprobe);

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

    {
        const int threads = pick_threads(SparseBlockTableKernelKind::Append, rows, nprobe, block_size, dim);
        const int blocks = pick_blocks(SparseBlockTableKernelKind::Append, rows, threads, nprobe);

        k_build_sparse_block_table_append<<<blocks, threads, 0, stream>>>(
            d_compact_len,
            d_free_blocks_needed,
            d_row_free_base,
            d_free_block_ids,
            d_out_block_table,
            d_out_bt_len,
            d_error_code,
            rows,
            total_blocks,
            max_bt_len);
        if (cudaGetLastError() != cudaSuccess) return ERR_LAUNCH;
    }

    return ERR_OK;
}