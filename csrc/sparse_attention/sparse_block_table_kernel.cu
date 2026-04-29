#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "sparse_attention_common.h"

__device__ __forceinline__ int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ void warp_set_error_and_trap(int32_t* err,
                                                        int32_t code,
                                                        int lane) {
    if (lane == 0) atomicCAS(err, 0, code);
    asm volatile("trap;");
}

template <typename T>
__device__ __forceinline__ void warp_copy_tokens(
    T* __restrict__ block_storage,
    int total_blocks, int block_size, int dim,
    int src_bid, int src_tok_offset,
    int dst_bid, int dst_tok_offset,
    int token_count, int lane) {

    using Vec = int4;
    constexpr int VEC_BYTES = sizeof(Vec);                       // 16
    constexpr int elems_per_vec = VEC_BYTES / (int)sizeof(T);    // fp16/bf16:8, fp32:4

    const bool use_vec = ((dim % elems_per_vec) == 0);

    const int elems_per_plane = token_count * dim;
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
            const int n_vec = elems_per_plane / elems_per_vec;
            const Vec* sv = reinterpret_cast<const Vec*>(src_plane);
            Vec*       dv = reinterpret_cast<Vec*>(dst_plane);
            for (int i = lane; i < n_vec; i += WARP_SIZE) {
                dv[i] = sv[i];
            }
            const int covered = n_vec * elems_per_vec;
            for (int i = covered + lane; i < elems_per_plane; i += WARP_SIZE) {
                dst_plane[i] = src_plane[i];
            }
        } else {
            for (int i = lane; i < elems_per_plane; i += WARP_SIZE) {
                dst_plane[i] = src_plane[i];
            }
        }
    }
}

template <typename T>
__global__ void k_build_and_pack_warp_per_row(
    const int32_t* __restrict__ top_clusters,      // [NQ,Hq,nprobe]
    const int32_t* __restrict__ cluster_block_ids, // [Hkv,C,maxB]
    const int32_t* __restrict__ cluster_sizes,     // [Hkv,C]
    T* __restrict__ block_storage,                 // [2,total_blocks,block_size,dim]
    const int32_t* __restrict__ free_block_ids,    // [max_free_block]
    int32_t max_free_block,
    int32_t* __restrict__ out_block_table,         // [rows,max_bt_len], prefilled -1
    int32_t* __restrict__ out_bt_len,              // [rows], prefilled 0
    int32_t* __restrict__ out_seqused_k,           // [rows], prefilled 0
    int32_t* __restrict__ used_free_block_count,   // scalar, prefilled 0
    int32_t* __restrict__ error_code,              // scalar, prefilled 0
    int NQ, int Hq, int nprobe,
    int Hkv, int C, int maxB,
    int total_blocks, int block_size, int dim,
    int max_bt_len) {

    const int lane             = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id_in_block = threadIdx.x / WARP_SIZE;
    const int warps_per_block  = blockDim.x / WARP_SIZE;
    const int row              = blockIdx.x * warps_per_block + warp_id_in_block;

    const int rows = NQ * Hq;
    if (row >= rows) return;

    const int hq      = row % Hq;
    const int q_per_kv = Hq / Hkv;
    const int hkv     = hq / q_per_kv;

    const int64_t top_base  = (int64_t)row * nprobe;
    const int64_t cs_base   = (int64_t)hkv * C;
    const int64_t cbid_base = (int64_t)hkv * C * maxB;

    int write         = 0;
    int seqused_k     = 0;
    int cur_free_bid  = -1;
    int cur_free_fill = 0;

    for (int k = 0; k < nprobe; ++k) {
        int cid = top_clusters[top_base + k];
        if (cid < 0 || cid >= C) {
            continue;
        }

        int sz = cluster_sizes[cs_base + cid];
        if (sz <= 0) continue;
        seqused_k += sz;

        int nb = ceil_div_int(sz, block_size);
        if (nb > maxB) {
            warp_set_error_and_trap(error_code, ERR_NB_OVERFLOW, lane);
            return;
        }

        int tail    = sz % block_size;
        int full_nb = (tail == 0) ? nb : (nb - 1);
        int64_t ids_base = cbid_base + (int64_t)cid * maxB;

        for (int bi = 0; bi < full_nb; ++bi) {
            if (write >= max_bt_len) {
                warp_set_error_and_trap(error_code, ERR_BT_LEN_OVERFLOW, lane);
                return;
            }
            int bid = cluster_block_ids[ids_base + bi];
            if (bid < 0 || bid >= total_blocks) {
                warp_set_error_and_trap(error_code, ERR_BAD_PARAM, lane);
                return;
            }
            if (lane == 0) {
                out_block_table[(int64_t)row * max_bt_len + write] = bid;
            }
            ++write;
        }

        if (tail != 0 && nb >= 1) {
            int src_bid = cluster_block_ids[ids_base + (nb - 1)];
            if (src_bid < 0 || src_bid >= total_blocks) {
                warp_set_error_and_trap(error_code, ERR_BAD_PARAM, lane);
                return;
            }

            int src_consumed = 0;
            while (src_consumed < tail) {
                if (cur_free_bid < 0 || cur_free_fill == block_size) {
                    if (cur_free_fill == block_size) {
                        if (write >= max_bt_len) {
                            warp_set_error_and_trap(error_code,
                                                    ERR_BT_LEN_OVERFLOW, lane);
                            return;
                        }
                        if (lane == 0) {
                            out_block_table[(int64_t)row * max_bt_len + write]
                                = cur_free_bid;
                        }
                        ++write;
                    }

                    int fb_idx = 0;
                    if (lane == 0) {
                        fb_idx = atomicAdd(used_free_block_count, 1);
                    }
                    fb_idx = __shfl_sync(FULL_MASK, fb_idx, 0);

                    if (fb_idx >= max_free_block) {
                        warp_set_error_and_trap(error_code,
                                                ERR_NO_FREE_BLOCK, lane);
                        return;
                    }
                    cur_free_bid = free_block_ids[fb_idx];
                    if (cur_free_bid < 0 || cur_free_bid >= total_blocks) {
                        warp_set_error_and_trap(error_code,
                                                ERR_BAD_PARAM, lane);
                        return;
                    }
                    cur_free_fill = 0;
                }

                int dst_room = block_size - cur_free_fill;
                int remain   = tail - src_consumed;
                int take     = (remain < dst_room) ? remain : dst_room;

                warp_copy_tokens<T>(
                    block_storage,
                    total_blocks, block_size, dim,
                    src_bid,      src_consumed,
                    cur_free_bid, cur_free_fill,
                    take, lane);

                src_consumed  += take;
                cur_free_fill += take;
            }
        }
    }

    if (cur_free_bid >= 0 && cur_free_fill > 0) {
        if (write >= max_bt_len) {
            warp_set_error_and_trap(error_code, ERR_BT_LEN_OVERFLOW, lane);
            return;
        }
        if (lane == 0) {
            out_block_table[(int64_t)row * max_bt_len + write] = cur_free_bid;
        }
        ++write;
    }

    if (lane == 0) {
        out_bt_len[row]    = write;
        out_seqused_k[row] = seqused_k;
    }
}

static int pick_warps_per_block(int rows) {
    if (rows < 32) return 1;
    return 4;
}

extern "C" int build_block_table_with_copy_launcher_raw(
    const int32_t* d_top_clusters,
    const int32_t* d_cluster_block_ids,
    const int32_t* d_cluster_sizes,
    void* d_block_storage,
    int32_t storage_dtype,
    const int32_t* d_free_block_ids,
    int32_t max_free_block,
    int32_t* d_out_block_table,
    int32_t* d_out_bt_len,
    int32_t* d_out_seqused_k,
    int32_t* d_used_free_block_count,
    int32_t* d_error_code,
    int NQ, int Hq, int nprobe,
    int Hkv, int C, int maxB,
    int total_blocks, int block_size, int dim,
    int max_bt_len,
    cudaStream_t stream) {

    if (NQ <= 0 || Hq <= 0 || nprobe <= 0 || Hkv <= 0 || C <= 0 || maxB <= 0 ||
        total_blocks <= 0 || block_size <= 0 || dim <= 0 || max_bt_len <= 0 ||
        max_free_block <= 0) {
        return ERR_BAD_PARAM;
    }
    if (Hq % Hkv != 0) return ERR_HQ_HKV_MISMATCH;

    const int rows            = NQ * Hq;
    const int warps_per_block = pick_warps_per_block(rows);
    const int threads         = warps_per_block * WARP_SIZE;
    const int blocks          = (rows + warps_per_block - 1) / warps_per_block;

    if (storage_dtype == DTYPE_FP32) {
        k_build_and_pack_warp_per_row<float>
            <<<blocks, threads, 0, stream>>>(
                d_top_clusters, d_cluster_block_ids, d_cluster_sizes,
                (float*)d_block_storage, d_free_block_ids, max_free_block,
                d_out_block_table, d_out_bt_len, d_out_seqused_k,
                d_used_free_block_count, d_error_code,
                NQ, Hq, nprobe, Hkv, C, maxB,
                total_blocks, block_size, dim, max_bt_len);
    } else if (storage_dtype == DTYPE_FP16) {
        k_build_and_pack_warp_per_row<__half>
            <<<blocks, threads, 0, stream>>>(
                d_top_clusters, d_cluster_block_ids, d_cluster_sizes,
                (__half*)d_block_storage, d_free_block_ids, max_free_block,
                d_out_block_table, d_out_bt_len, d_out_seqused_k,
                d_used_free_block_count, d_error_code,
                NQ, Hq, nprobe, Hkv, C, maxB,
                total_blocks, block_size, dim, max_bt_len);
    } else if (storage_dtype == DTYPE_BF16) {
        k_build_and_pack_warp_per_row<__nv_bfloat16>
            <<<blocks, threads, 0, stream>>>(
                d_top_clusters, d_cluster_block_ids, d_cluster_sizes,
                (__nv_bfloat16*)d_block_storage, d_free_block_ids, max_free_block,
                d_out_block_table, d_out_bt_len, d_out_seqused_k,
                d_used_free_block_count, d_error_code,
                NQ, Hq, nprobe, Hkv, C, maxB,
                total_blocks, block_size, dim, max_bt_len);
    } else {
        return ERR_UNSUPPORTED_DTYPE;
    }

    if (cudaGetLastError() != cudaSuccess) return ERR_LAUNCH;
    return ERR_OK;
}