#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "sparse_attention_common.h"

template <typename T>
__device__ __forceinline__ void write_vec(T* dst, const T* src, int dim, int lane) {
    for (int d = lane; d < dim; d += WARP_SIZE) {
        dst[d] = src[d];
    }
}

template <typename T>
__global__ void k_append_kv_to_clusters(
    // inputs
    const T* __restrict__ key,               // [Nq,Hkv,dim]
    const T* __restrict__ value,             // [Nq,Hkv,dim]
    const int32_t* __restrict__ label,       // [Nq,Hkv]
    const int32_t* __restrict__ free_block_ids, // [max_free_block]
    int32_t max_free_block,

    // in/out
    T* __restrict__ block_storage,           // [2,total_blocks,block_size,dim]
    int32_t* __restrict__ cluster_block_ids, // [Hkv,C,maxB]
    int32_t* __restrict__ cluster_sizes,     // [Hkv,C]
    int32_t* __restrict__ used_free_block_count, // scalar
    int32_t* __restrict__ error_code,            // scalar

    // shape
    int Nq, int Hkv, int C, int maxB,
    int total_blocks, int block_size, int dim) {

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id_in_block = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int row = blockIdx.x * warps_per_block + warp_id_in_block;

    const int rows = Nq * Hkv;
    if (row >= rows) return;

    const int q = row / Hkv;
    const int h = row % Hkv;

    // label
    const int cid = label[(int64_t)q * Hkv + h];
    if (cid < 0 || cid >= C) {
        if (lane == 0) atomicCAS(error_code, 0, ERR_BAD_PARAM);
        return;
    }

    const int64_t cs_idx = (int64_t)h * C + cid;
    const int64_t cb_base = ((int64_t)h * C + cid) * maxB;  // [maxB]

    int old_size = 0;
    if (lane == 0) {
        old_size = atomicAdd(&cluster_sizes[cs_idx], 1);
    }
    old_size = __shfl_sync(FULL_MASK, old_size, 0);

    const int token_idx = old_size;
    const int bid_pos = token_idx / block_size;
    const int tok_off = token_idx % block_size;

    if (bid_pos >= maxB) {
        if (lane == 0) atomicCAS(error_code, 0, ERR_NB_OVERFLOW);
        return;
    }

    int dst_bid = -1;
    if (lane == 0) {
        if (tok_off == 0) {
            int cur = cluster_block_ids[cb_base + bid_pos];
            if (cur >= 0) {
                dst_bid = cur;
            } else {
                int fb_idx = atomicAdd(used_free_block_count, 1);
                if (fb_idx >= max_free_block) {
                    atomicCAS(error_code, 0, ERR_NO_FREE_BLOCK);
                    dst_bid = -1;
                } else {
                    int new_bid = free_block_ids[fb_idx];
                    if (new_bid < 0 || new_bid >= total_blocks) {
                        atomicCAS(error_code, 0, ERR_BAD_PARAM);
                        dst_bid = -1;
                    } else {
                        cluster_block_ids[cb_base + bid_pos] = new_bid;
                        dst_bid = new_bid;
                    }
                }
            }
        } else {
            int cur = cluster_block_ids[cb_base + bid_pos];
            if (cur < 0 || cur >= total_blocks) {
                atomicCAS(error_code, 0, ERR_BAD_PARAM);
                dst_bid = -1;
            } else {
                dst_bid = cur;
            }
        }
    }
    dst_bid = __shfl_sync(FULL_MASK, dst_bid, 0);
    if (dst_bid < 0) return;

    const int64_t src_base = ((int64_t)q * Hkv + h) * dim;
    const T* src_k = key + src_base;
    const T* src_v = value + src_base;

    const int64_t plane_stride = (int64_t)total_blocks * block_size * dim;
    const int64_t dst_elem = ((int64_t)dst_bid * block_size + tok_off) * dim;

    T* dst_k = block_storage + 0 * plane_stride + dst_elem;
    T* dst_v = block_storage + 1 * plane_stride + dst_elem;

    write_vec<T>(dst_k, src_k, dim, lane);
    write_vec<T>(dst_v, src_v, dim, lane);
}

static int pick_warps_per_block(int rows) {
    if (rows < 32) return 1;
    return 4;
}

extern "C" int append_kv_to_clusters_launcher_raw(
    const void* d_key,
    const void* d_value,
    int32_t storage_dtype,
    const int32_t* d_label,
    const int32_t* d_free_block_ids,
    int32_t max_free_block,
    void* d_block_storage,
    int32_t* d_cluster_block_ids,
    int32_t* d_cluster_sizes,
    int32_t* d_used_free_block_count,
    int32_t* d_error_code,
    int Nq, int Hkv, int C, int maxB,
    int total_blocks, int block_size, int dim,
    cudaStream_t stream) {

    if (Nq <= 0 || Hkv <= 0 || C <= 0 || maxB <= 0 ||
        total_blocks <= 0 || block_size <= 0 || dim <= 0 ||
        max_free_block <= 0) {
        return ERR_BAD_PARAM;
    }

    const int rows = Nq * Hkv;
    const int warps = pick_warps_per_block(rows);
    const int threads = warps * WARP_SIZE;
    const int blocks = (rows + warps - 1) / warps;

    if (storage_dtype == DTYPE_FP32) {
        k_append_kv_to_clusters<float><<<blocks, threads, 0, stream>>>(
            (const float*)d_key, (const float*)d_value, d_label,
            d_free_block_ids, max_free_block,
            (float*)d_block_storage, d_cluster_block_ids, d_cluster_sizes,
            d_used_free_block_count, d_error_code,
            Nq, Hkv, C, maxB, total_blocks, block_size, dim);
    } else if (storage_dtype == DTYPE_FP16) {
        k_append_kv_to_clusters<__half><<<blocks, threads, 0, stream>>>(
            (const __half*)d_key, (const __half*)d_value, d_label,
            d_free_block_ids, max_free_block,
            (__half*)d_block_storage, d_cluster_block_ids, d_cluster_sizes,
            d_used_free_block_count, d_error_code,
            Nq, Hkv, C, maxB, total_blocks, block_size, dim);
    } else if (storage_dtype == DTYPE_BF16) {
        k_append_kv_to_clusters<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*)d_key, (const __nv_bfloat16*)d_value, d_label,
            d_free_block_ids, max_free_block,
            (__nv_bfloat16*)d_block_storage, d_cluster_block_ids, d_cluster_sizes,
            d_used_free_block_count, d_error_code,
            Nq, Hkv, C, maxB, total_blocks, block_size, dim);
    } else {
        return ERR_UNSUPPORTED_DTYPE;
    }

    if (cudaGetLastError() != cudaSuccess) return ERR_LAUNCH;
    return ERR_OK;
}