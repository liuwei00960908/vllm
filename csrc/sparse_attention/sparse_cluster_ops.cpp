#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "sparse_attention_common.h"

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
    cudaStream_t stream);

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
    cudaStream_t stream);

extern "C" int union_topk_clusters_by_kv_group_launcher_raw(
    const int32_t* d_top_clusters,
    int32_t* d_out_top_clusters,
    int Nq,
    int Hq,
    int Hkv,
    int num_clusters,
    int nprobe,
    int union_nprobe,
    cudaStream_t stream);

extern "C" int group_avg_topk_clusters_by_kv_group_launcher_raw(
    const void* d_query,
    const void* d_cluster_centers_T,
    const void* d_mean,
    const int32_t* d_cluster_center_count,
    float* d_head_scores,
    int32_t* d_out_top_clusters,
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
    cudaStream_t stream);

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
    cudaStream_t stream);

static inline int32_t map_storage_dtype(torch::ScalarType t) {
    if (t == torch::kFloat32) return DTYPE_FP32;
    if (t == torch::kFloat16) return DTYPE_FP16;
    if (t == torch::kBFloat16) return DTYPE_BF16;
    TORCH_CHECK(false, "Unsupported dtype: ", c10::toString(t));
}

static inline void check_cuda_contig(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

static inline void check_cuda(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
}

torch::Tensor append_kv_to_clusters_cuda(
    torch::Tensor block_storage,
    torch::Tensor cluster_compact_block_ids,
    torch::Tensor cluster_temp_kv_pos,
    torch::Tensor cluster_total_kv_counts,
    torch::Tensor temp_block_ids,
    torch::Tensor temp_block_kv_counts,
    torch::Tensor temp_block_kv_owner,
    torch::Tensor free_block_ids,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor label
) {
    check_cuda_contig(block_storage, "block_storage");
    check_cuda_contig(cluster_compact_block_ids, "cluster_compact_block_ids");
    check_cuda_contig(cluster_temp_kv_pos, "cluster_temp_kv_pos");
    check_cuda_contig(cluster_total_kv_counts, "cluster_total_kv_counts");
    check_cuda_contig(temp_block_ids, "temp_block_ids");
    check_cuda_contig(temp_block_kv_counts, "temp_block_kv_counts");
    check_cuda_contig(temp_block_kv_owner, "temp_block_kv_owner");
    check_cuda_contig(free_block_ids, "free_block_ids");
    check_cuda(key, "key");
    check_cuda(value, "value");
    check_cuda_contig(label, "label");

    TORCH_CHECK(cluster_compact_block_ids.scalar_type() == torch::kInt32,
                "cluster_compact_block_ids must be int32");
    TORCH_CHECK(cluster_temp_kv_pos.scalar_type() == torch::kInt32,
                "cluster_temp_kv_pos must be int32");
    TORCH_CHECK(cluster_total_kv_counts.scalar_type() == torch::kInt32,
                "cluster_total_kv_counts must be int32");
    TORCH_CHECK(temp_block_ids.scalar_type() == torch::kInt32,
                "temp_block_ids must be int32");
    TORCH_CHECK(temp_block_kv_counts.scalar_type() == torch::kInt32,
                "temp_block_kv_counts must be int32");
    TORCH_CHECK(temp_block_kv_owner.scalar_type() == torch::kInt32,
                "temp_block_kv_owner must be int32");
    TORCH_CHECK(free_block_ids.scalar_type() == torch::kInt32,
                "free_block_ids must be int32");
    TORCH_CHECK(label.scalar_type() == torch::kInt32,
                "label must be int32");

    TORCH_CHECK(block_storage.dim() == 4 && block_storage.size(0) == 2,
                "block_storage must be [2,total_blocks,block_size,dim]");
    TORCH_CHECK(cluster_compact_block_ids.dim() == 3,
                "cluster_compact_block_ids must be [Hkv,C,maxB]");
    TORCH_CHECK(cluster_temp_kv_pos.dim() == 4 && cluster_temp_kv_pos.size(3) == 2,
                "cluster_temp_kv_pos must be [Hkv,C,block_size,2]");
    TORCH_CHECK(cluster_total_kv_counts.dim() == 2,
                "cluster_total_kv_counts must be [Hkv,C]");
    TORCH_CHECK(temp_block_ids.dim() == 1,
                "temp_block_ids must be [max_temp_blocks]");
    TORCH_CHECK(temp_block_kv_counts.dim() == 1 && temp_block_kv_counts.size(0) == 1,
                "temp_block_kv_counts must be [1]");
    TORCH_CHECK(temp_block_kv_owner.dim() == 2 && temp_block_kv_owner.size(1) == 2,
                "temp_block_kv_owner must be [max_temp_blocks * block_size,2]");
    TORCH_CHECK(free_block_ids.dim() == 1,
                "free_block_ids must be [max_free_block]");
    TORCH_CHECK(key.dim() == 3 && value.dim() == 3,
                "key/value must be [Nq,Hkv,dim]");
    TORCH_CHECK(label.dim() == 2,
                "label must be [Nq,Hkv]");

    TORCH_CHECK(key.scalar_type() == block_storage.scalar_type(),
                "key dtype must equal block_storage dtype");
    TORCH_CHECK(value.scalar_type() == block_storage.scalar_type(),
                "value dtype must equal block_storage dtype");
    TORCH_CHECK(key.sizes() == value.sizes(), "key/value shape mismatch");
    TORCH_CHECK(key.stride(0) >= 0 && key.stride(1) >= 0 && key.stride(2) >= 0,
                "key strides must be non-negative");
    TORCH_CHECK(value.stride(0) >= 0 && value.stride(1) >= 0 && value.stride(2) >= 0,
                "value strides must be non-negative");

    const int Nq = static_cast<int>(key.size(0));
    const int Hkv = static_cast<int>(key.size(1));
    const int dim = static_cast<int>(key.size(2));

    TORCH_CHECK(label.size(0) == Nq && label.size(1) == Hkv, "label shape mismatch");

    TORCH_CHECK(cluster_compact_block_ids.size(0) == Hkv,
                "cluster_compact_block_ids Hkv mismatch");
    TORCH_CHECK(cluster_temp_kv_pos.size(0) == Hkv,
                "cluster_temp_kv_pos Hkv mismatch");
    TORCH_CHECK(cluster_total_kv_counts.size(0) == Hkv,
                "cluster_total_kv_counts Hkv mismatch");

    const int C = static_cast<int>(cluster_compact_block_ids.size(1));
    const int maxB = static_cast<int>(cluster_compact_block_ids.size(2));
    const int total_blocks = static_cast<int>(block_storage.size(1));
    const int block_size = static_cast<int>(block_storage.size(2));
    const int max_temp_blocks = static_cast<int>(temp_block_ids.size(0));
    const int max_free_block = static_cast<int>(free_block_ids.size(0));

    TORCH_CHECK(cluster_temp_kv_pos.size(1) == C, "cluster_temp_kv_pos C mismatch");
    TORCH_CHECK(cluster_total_kv_counts.size(1) == C, "cluster_total_kv_counts C mismatch");
    TORCH_CHECK(cluster_temp_kv_pos.size(2) == block_size,
                "cluster_temp_kv_pos third dim must equal block_size");
    TORCH_CHECK(static_cast<int>(block_storage.size(3)) == dim, "dim mismatch");

    TORCH_CHECK(temp_block_kv_owner.size(0) == max_temp_blocks * block_size,
                "temp_block_kv_owner first dim must be max_temp_blocks * block_size");

    auto i32_opts = cluster_total_kv_counts.options();
    auto used_free_block_count = torch::zeros({1}, i32_opts);
    auto error_code = torch::zeros({1}, i32_opts);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(block_storage));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int rc = append_kv_to_clusters_launcher_raw(
        key.data_ptr(),
        value.data_ptr(),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        map_storage_dtype(block_storage.scalar_type()),
        label.data_ptr<int32_t>(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        temp_block_ids.data_ptr<int32_t>(),
        temp_block_kv_counts.data_ptr<int32_t>(),
        temp_block_kv_owner.data_ptr<int32_t>(),
        block_storage.data_ptr(),
        cluster_compact_block_ids.data_ptr<int32_t>(),
        cluster_temp_kv_pos.data_ptr<int32_t>(),
        cluster_total_kv_counts.data_ptr<int32_t>(),
        free_block_ids.data_ptr<int32_t>(),
        max_free_block,
        used_free_block_count.data_ptr<int32_t>(),
        error_code.data_ptr<int32_t>(),
        nullptr,
        0,
        nullptr,
        0,
        nullptr,
        0,
        0,
        Nq, Hkv, C, maxB,
        total_blocks, block_size, dim,
        max_temp_blocks,
        stream);

    TORCH_CHECK(rc == ERR_OK, "append_kv_to_clusters launcher failed, rc=", rc);

    return used_free_block_count;
}

torch::Tensor append_kv_to_clusters_inplace_cuda(
    torch::Tensor block_storage,
    torch::Tensor cluster_compact_block_ids,
    torch::Tensor cluster_temp_kv_pos,
    torch::Tensor cluster_total_kv_counts,
    torch::Tensor temp_block_ids,
    torch::Tensor temp_block_kv_counts,
    torch::Tensor temp_block_kv_owner,
    torch::Tensor free_block_ids,
    torch::Tensor used_free_block_count,
    torch::Tensor error_code,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor label
) {
    check_cuda_contig(block_storage, "block_storage");
    check_cuda_contig(cluster_compact_block_ids, "cluster_compact_block_ids");
    check_cuda_contig(cluster_temp_kv_pos, "cluster_temp_kv_pos");
    check_cuda_contig(cluster_total_kv_counts, "cluster_total_kv_counts");
    check_cuda_contig(temp_block_ids, "temp_block_ids");
    check_cuda_contig(temp_block_kv_counts, "temp_block_kv_counts");
    check_cuda_contig(temp_block_kv_owner, "temp_block_kv_owner");
    check_cuda_contig(free_block_ids, "free_block_ids");
    check_cuda_contig(used_free_block_count, "used_free_block_count");
    check_cuda_contig(error_code, "error_code");
    check_cuda(key, "key");
    check_cuda(value, "value");
    check_cuda_contig(label, "label");

    TORCH_CHECK(used_free_block_count.scalar_type() == torch::kInt32,
                "used_free_block_count must be int32");
    TORCH_CHECK(error_code.scalar_type() == torch::kInt32,
                "error_code must be int32");
    TORCH_CHECK(used_free_block_count.numel() == 1,
                "used_free_block_count must be [1]");
    TORCH_CHECK(error_code.numel() == 1,
                "error_code must be [1]");
    TORCH_CHECK(key.dim() == 3 && value.dim() == 3,
                "key/value must be [Nq,Hkv,dim]");
    TORCH_CHECK(key.scalar_type() == block_storage.scalar_type(),
                "key dtype must equal block_storage dtype");
    TORCH_CHECK(value.scalar_type() == block_storage.scalar_type(),
                "value dtype must equal block_storage dtype");
    TORCH_CHECK(key.sizes() == value.sizes(), "key/value shape mismatch");
    TORCH_CHECK(key.stride(0) >= 0 && key.stride(1) >= 0 && key.stride(2) >= 0,
                "key strides must be non-negative");
    TORCH_CHECK(value.stride(0) >= 0 && value.stride(1) >= 0 && value.stride(2) >= 0,
                "value strides must be non-negative");

    const int Nq = static_cast<int>(key.size(0));
    const int Hkv = static_cast<int>(key.size(1));
    const int dim = static_cast<int>(key.size(2));
    const int C = static_cast<int>(cluster_compact_block_ids.size(1));
    const int maxB = static_cast<int>(cluster_compact_block_ids.size(2));
    const int total_blocks = static_cast<int>(block_storage.size(1));
    const int block_size = static_cast<int>(block_storage.size(2));
    const int max_temp_blocks = static_cast<int>(temp_block_ids.size(0));
    const int max_free_block = static_cast<int>(free_block_ids.size(0));

    const at::cuda::OptionalCUDAGuard device_guard(device_of(block_storage));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int rc = append_kv_to_clusters_launcher_raw(
        key.data_ptr(),
        value.data_ptr(),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        map_storage_dtype(block_storage.scalar_type()),
        label.data_ptr<int32_t>(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        temp_block_ids.data_ptr<int32_t>(),
        temp_block_kv_counts.data_ptr<int32_t>(),
        temp_block_kv_owner.data_ptr<int32_t>(),
        block_storage.data_ptr(),
        cluster_compact_block_ids.data_ptr<int32_t>(),
        cluster_temp_kv_pos.data_ptr<int32_t>(),
        cluster_total_kv_counts.data_ptr<int32_t>(),
        free_block_ids.data_ptr<int32_t>(),
        max_free_block,
        used_free_block_count.data_ptr<int32_t>(),
        error_code.data_ptr<int32_t>(),
        nullptr,
        0,
        nullptr,
        0,
        nullptr,
        0,
        0,
        Nq, Hkv, C, maxB,
        total_blocks, block_size, dim,
        max_temp_blocks,
        stream);

    TORCH_CHECK(rc == ERR_OK, "append_kv_to_clusters launcher failed, rc=", rc);
    return used_free_block_count;
}

torch::Tensor append_kv_to_clusters_by_centers_inplace_cuda(
    torch::Tensor block_storage,
    torch::Tensor cluster_compact_block_ids,
    torch::Tensor cluster_temp_kv_pos,
    torch::Tensor cluster_total_kv_counts,
    torch::Tensor temp_block_ids,
    torch::Tensor temp_block_kv_counts,
    torch::Tensor temp_block_kv_owner,
    torch::Tensor free_block_ids,
    torch::Tensor used_free_block_count,
    torch::Tensor error_code,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor cluster_centers_T,
    torch::Tensor mean,
    torch::Tensor cluster_center_count,
    torch::Tensor input_token_count
) {
    check_cuda_contig(block_storage, "block_storage");
    check_cuda_contig(cluster_compact_block_ids, "cluster_compact_block_ids");
    check_cuda_contig(cluster_temp_kv_pos, "cluster_temp_kv_pos");
    check_cuda_contig(cluster_total_kv_counts, "cluster_total_kv_counts");
    check_cuda_contig(temp_block_ids, "temp_block_ids");
    check_cuda_contig(temp_block_kv_counts, "temp_block_kv_counts");
    check_cuda_contig(temp_block_kv_owner, "temp_block_kv_owner");
    check_cuda_contig(free_block_ids, "free_block_ids");
    check_cuda_contig(used_free_block_count, "used_free_block_count");
    check_cuda_contig(error_code, "error_code");
    check_cuda(key, "key");
    check_cuda(value, "value");
    check_cuda_contig(cluster_centers_T, "cluster_centers_T");
    check_cuda_contig(mean, "mean");
    check_cuda_contig(cluster_center_count, "cluster_center_count");
    check_cuda_contig(input_token_count, "input_token_count");

    TORCH_CHECK(used_free_block_count.scalar_type() == torch::kInt32,
                "used_free_block_count must be int32");
    TORCH_CHECK(error_code.scalar_type() == torch::kInt32,
                "error_code must be int32");
    TORCH_CHECK(used_free_block_count.numel() == 1,
                "used_free_block_count must be [1]");
    TORCH_CHECK(error_code.numel() == 1,
                "error_code must be [1]");
    TORCH_CHECK(cluster_center_count.scalar_type() == torch::kInt32,
                "cluster_center_count must be int32");
    TORCH_CHECK(cluster_center_count.numel() == 1,
                "cluster_center_count must be [1]");
    TORCH_CHECK(input_token_count.scalar_type() == torch::kInt32,
                "input_token_count must be int32");
    TORCH_CHECK(input_token_count.numel() == 1,
                "input_token_count must be [1]");
    TORCH_CHECK(key.dim() == 3 && value.dim() == 3,
                "key/value must be [Nq,Hkv,dim]");
    TORCH_CHECK(key.scalar_type() == block_storage.scalar_type(),
                "key dtype must equal block_storage dtype");
    TORCH_CHECK(value.scalar_type() == block_storage.scalar_type(),
                "value dtype must equal block_storage dtype");
    TORCH_CHECK(cluster_centers_T.scalar_type() == block_storage.scalar_type(),
                "cluster_centers_T dtype must equal block_storage dtype");
    TORCH_CHECK(mean.scalar_type() == block_storage.scalar_type(),
                "mean dtype must equal block_storage dtype");
    TORCH_CHECK(key.sizes() == value.sizes(), "key/value shape mismatch");
    TORCH_CHECK(key.stride(0) >= 0 && key.stride(1) >= 0 && key.stride(2) >= 0,
                "key strides must be non-negative");
    TORCH_CHECK(value.stride(0) >= 0 && value.stride(1) >= 0 && value.stride(2) >= 0,
                "value strides must be non-negative");

    const int Nq = static_cast<int>(key.size(0));
    const int Hkv = static_cast<int>(key.size(1));
    const int dim = static_cast<int>(key.size(2));
    const int C = static_cast<int>(cluster_compact_block_ids.size(1));
    const int maxB = static_cast<int>(cluster_compact_block_ids.size(2));
    const int total_blocks = static_cast<int>(block_storage.size(1));
    const int block_size = static_cast<int>(block_storage.size(2));
    const int max_temp_blocks = static_cast<int>(temp_block_ids.size(0));
    const int max_free_block = static_cast<int>(free_block_ids.size(0));

    TORCH_CHECK(cluster_centers_T.dim() == 3,
                "cluster_centers_T must be [Hkv,dim,C]");
    TORCH_CHECK(cluster_centers_T.size(0) == Hkv &&
                cluster_centers_T.size(1) == dim &&
                cluster_centers_T.size(2) == C,
                "cluster_centers_T shape mismatch");
    TORCH_CHECK(mean.dim() == 2 && mean.size(0) == Hkv && mean.size(1) == dim,
                "mean must be [Hkv,dim]");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(block_storage));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int rc = append_kv_to_clusters_launcher_raw(
        key.data_ptr(),
        value.data_ptr(),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        map_storage_dtype(block_storage.scalar_type()),
        nullptr,
        cluster_centers_T.data_ptr(),
        mean.data_ptr(),
        cluster_center_count.data_ptr<int32_t>(),
        input_token_count.data_ptr<int32_t>(),
        temp_block_ids.data_ptr<int32_t>(),
        temp_block_kv_counts.data_ptr<int32_t>(),
        temp_block_kv_owner.data_ptr<int32_t>(),
        block_storage.data_ptr(),
        cluster_compact_block_ids.data_ptr<int32_t>(),
        cluster_temp_kv_pos.data_ptr<int32_t>(),
        cluster_total_kv_counts.data_ptr<int32_t>(),
        free_block_ids.data_ptr<int32_t>(),
        max_free_block,
        used_free_block_count.data_ptr<int32_t>(),
        error_code.data_ptr<int32_t>(),
        nullptr,
        0,
        nullptr,
        0,
        nullptr,
        0,
        0,
        Nq, Hkv, C, maxB,
        total_blocks, block_size, dim,
        max_temp_blocks,
        stream);

    TORCH_CHECK(rc == ERR_OK,
                "append_kv_to_clusters_by_centers launcher failed, rc=", rc);
    return used_free_block_count;
}

torch::Tensor append_kv_to_clusters_by_centers_with_steady_inplace_cuda(
    torch::Tensor block_storage,
    torch::Tensor cluster_compact_block_ids,
    torch::Tensor cluster_temp_kv_pos,
    torch::Tensor cluster_total_kv_counts,
    torch::Tensor temp_block_ids,
    torch::Tensor temp_block_kv_counts,
    torch::Tensor temp_block_kv_owner,
    torch::Tensor free_block_ids,
    torch::Tensor used_free_block_count,
    torch::Tensor error_code,
    torch::Tensor steady_start_block_ids,
    torch::Tensor steady_end_block_ids,
    torch::Tensor steady_state,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor cluster_centers_T,
    torch::Tensor mean,
    torch::Tensor cluster_center_count,
    int64_t steady_start_capacity,
    int64_t steady_end_capacity
) {
    check_cuda_contig(block_storage, "block_storage");
    check_cuda_contig(cluster_compact_block_ids, "cluster_compact_block_ids");
    check_cuda_contig(cluster_temp_kv_pos, "cluster_temp_kv_pos");
    check_cuda_contig(cluster_total_kv_counts, "cluster_total_kv_counts");
    check_cuda_contig(temp_block_ids, "temp_block_ids");
    check_cuda_contig(temp_block_kv_counts, "temp_block_kv_counts");
    check_cuda_contig(temp_block_kv_owner, "temp_block_kv_owner");
    check_cuda_contig(free_block_ids, "free_block_ids");
    check_cuda_contig(used_free_block_count, "used_free_block_count");
    check_cuda_contig(error_code, "error_code");
    check_cuda_contig(steady_start_block_ids, "steady_start_block_ids");
    check_cuda_contig(steady_end_block_ids, "steady_end_block_ids");
    check_cuda_contig(steady_state, "steady_state");
    check_cuda(key, "key");
    check_cuda(value, "value");
    check_cuda_contig(cluster_centers_T, "cluster_centers_T");
    check_cuda_contig(mean, "mean");
    check_cuda_contig(cluster_center_count, "cluster_center_count");

    TORCH_CHECK(used_free_block_count.scalar_type() == torch::kInt32,
                "used_free_block_count must be int32");
    TORCH_CHECK(error_code.scalar_type() == torch::kInt32,
                "error_code must be int32");
    TORCH_CHECK(steady_start_block_ids.scalar_type() == torch::kInt32,
                "steady_start_block_ids must be int32");
    TORCH_CHECK(steady_end_block_ids.scalar_type() == torch::kInt32,
                "steady_end_block_ids must be int32");
    TORCH_CHECK(steady_state.scalar_type() == torch::kInt32,
                "steady_state must be int32");
    TORCH_CHECK(used_free_block_count.numel() == 1,
                "used_free_block_count must be [1]");
    TORCH_CHECK(error_code.numel() == 1,
                "error_code must be [1]");
    TORCH_CHECK(cluster_center_count.scalar_type() == torch::kInt32,
                "cluster_center_count must be int32");
    TORCH_CHECK(cluster_center_count.numel() == 1,
                "cluster_center_count must be [1]");
    TORCH_CHECK(steady_state.dim() == 1 && steady_state.numel() >= 4,
                "steady_state must be [>=4]");
    TORCH_CHECK(key.dim() == 3 && value.dim() == 3,
                "key/value must be [Nq,Hkv,dim]");
    TORCH_CHECK(key.scalar_type() == block_storage.scalar_type(),
                "key dtype must equal block_storage dtype");
    TORCH_CHECK(value.scalar_type() == block_storage.scalar_type(),
                "value dtype must equal block_storage dtype");
    TORCH_CHECK(cluster_centers_T.scalar_type() == block_storage.scalar_type(),
                "cluster_centers_T dtype must equal block_storage dtype");
    TORCH_CHECK(mean.scalar_type() == block_storage.scalar_type(),
                "mean dtype must equal block_storage dtype");
    TORCH_CHECK(key.sizes() == value.sizes(), "key/value shape mismatch");
    TORCH_CHECK(key.stride(0) >= 0 && key.stride(1) >= 0 && key.stride(2) >= 0,
                "key strides must be non-negative");
    TORCH_CHECK(value.stride(0) >= 0 && value.stride(1) >= 0 &&
                    value.stride(2) >= 0,
                "value strides must be non-negative");

    const int Nq = static_cast<int>(key.size(0));
    const int Hkv = static_cast<int>(key.size(1));
    const int dim = static_cast<int>(key.size(2));
    const int C = static_cast<int>(cluster_compact_block_ids.size(1));
    const int maxB = static_cast<int>(cluster_compact_block_ids.size(2));
    const int total_blocks = static_cast<int>(block_storage.size(1));
    const int block_size = static_cast<int>(block_storage.size(2));
    const int max_temp_blocks = static_cast<int>(temp_block_ids.size(0));
    const int max_free_block = static_cast<int>(free_block_ids.size(0));

    TORCH_CHECK(cluster_centers_T.dim() == 3,
                "cluster_centers_T must be [Hkv,dim,C]");
    TORCH_CHECK(cluster_centers_T.size(0) == Hkv &&
                    cluster_centers_T.size(1) == dim &&
                    cluster_centers_T.size(2) == C,
                "cluster_centers_T shape mismatch");
    TORCH_CHECK(mean.dim() == 2 && mean.size(0) == Hkv && mean.size(1) == dim,
                "mean must be [Hkv,dim]");
    TORCH_CHECK(steady_start_block_ids.dim() == 2 &&
                    steady_start_block_ids.size(0) == Hkv,
                "steady_start_block_ids must be [Hkv,start_blocks]");
    TORCH_CHECK(steady_end_block_ids.dim() == 2 &&
                    steady_end_block_ids.size(0) == Hkv,
                "steady_end_block_ids must be [Hkv,end_blocks]");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(block_storage));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int rc = append_kv_to_clusters_launcher_raw(
        key.data_ptr(),
        value.data_ptr(),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        map_storage_dtype(block_storage.scalar_type()),
        nullptr,
        cluster_centers_T.data_ptr(),
        mean.data_ptr(),
        cluster_center_count.data_ptr<int32_t>(),
        nullptr,
        temp_block_ids.data_ptr<int32_t>(),
        temp_block_kv_counts.data_ptr<int32_t>(),
        temp_block_kv_owner.data_ptr<int32_t>(),
        block_storage.data_ptr(),
        cluster_compact_block_ids.data_ptr<int32_t>(),
        cluster_temp_kv_pos.data_ptr<int32_t>(),
        cluster_total_kv_counts.data_ptr<int32_t>(),
        free_block_ids.data_ptr<int32_t>(),
        max_free_block,
        used_free_block_count.data_ptr<int32_t>(),
        error_code.data_ptr<int32_t>(),
        steady_start_block_ids.data_ptr<int32_t>(),
        static_cast<int32_t>(steady_start_block_ids.size(1)),
        steady_end_block_ids.data_ptr<int32_t>(),
        static_cast<int32_t>(steady_end_block_ids.size(1)),
        steady_state.data_ptr<int32_t>(),
        static_cast<int32_t>(steady_start_capacity),
        static_cast<int32_t>(steady_end_capacity),
        Nq, Hkv, C, maxB,
        total_blocks, block_size, dim,
        max_temp_blocks,
        stream);

    TORCH_CHECK(rc == ERR_OK,
                "append_kv_to_clusters_by_centers_with_steady launcher failed, rc=", rc);
    return used_free_block_count;
}

std::vector<torch::Tensor> update_sparse_steady_kv_inplace_cuda(
    torch::Tensor block_storage,
    torch::Tensor steady_start_block_ids,
    torch::Tensor steady_end_block_ids,
    torch::Tensor steady_state,
    torch::Tensor evicted_key,
    torch::Tensor evicted_value,
    torch::Tensor evicted_count,
    torch::Tensor key,
    torch::Tensor value,
    int64_t steady_start_capacity,
    int64_t steady_end_capacity
) {
    check_cuda_contig(block_storage, "block_storage");
    check_cuda_contig(steady_start_block_ids, "steady_start_block_ids");
    check_cuda_contig(steady_end_block_ids, "steady_end_block_ids");
    check_cuda_contig(steady_state, "steady_state");
    check_cuda_contig(evicted_key, "evicted_key");
    check_cuda_contig(evicted_value, "evicted_value");
    check_cuda_contig(evicted_count, "evicted_count");
    check_cuda(key, "key");
    check_cuda(value, "value");

    TORCH_CHECK(steady_start_block_ids.scalar_type() == torch::kInt32,
                "steady_start_block_ids must be int32");
    TORCH_CHECK(steady_end_block_ids.scalar_type() == torch::kInt32,
                "steady_end_block_ids must be int32");
    TORCH_CHECK(steady_state.scalar_type() == torch::kInt32,
                "steady_state must be int32");
    TORCH_CHECK(evicted_count.scalar_type() == torch::kInt32,
                "evicted_count must be int32");
    TORCH_CHECK(evicted_count.numel() == 1, "evicted_count must be [1]");

    TORCH_CHECK(block_storage.dim() == 4 && block_storage.size(0) == 2,
                "block_storage must be [2,total_blocks,block_size,dim]");
    TORCH_CHECK(steady_start_block_ids.dim() == 2,
                "steady_start_block_ids must be [Hkv,start_blocks]");
    TORCH_CHECK(steady_end_block_ids.dim() == 2,
                "steady_end_block_ids must be [Hkv,end_blocks]");
    TORCH_CHECK(steady_state.dim() == 1 && steady_state.numel() >= 4,
                "steady_state must be [>=4]");
    TORCH_CHECK(key.dim() == 3 && value.dim() == 3,
                "key/value must be [Nq,Hkv,dim]");
    TORCH_CHECK(evicted_key.dim() == 3 && evicted_value.dim() == 3,
                "evicted_key/evicted_value must be [cap,Hkv,dim]");

    TORCH_CHECK(key.scalar_type() == block_storage.scalar_type(),
                "key dtype must equal block_storage dtype");
    TORCH_CHECK(value.scalar_type() == block_storage.scalar_type(),
                "value dtype must equal block_storage dtype");
    TORCH_CHECK(evicted_key.scalar_type() == block_storage.scalar_type(),
                "evicted_key dtype must equal block_storage dtype");
    TORCH_CHECK(evicted_value.scalar_type() == block_storage.scalar_type(),
                "evicted_value dtype must equal block_storage dtype");
    TORCH_CHECK(key.sizes() == value.sizes(), "key/value shape mismatch");
    TORCH_CHECK(evicted_key.sizes() == evicted_value.sizes(),
                "evicted_key/evicted_value shape mismatch");
    TORCH_CHECK(key.stride(0) >= 0 && key.stride(1) >= 0 && key.stride(2) >= 0,
                "key strides must be non-negative");
    TORCH_CHECK(value.stride(0) >= 0 && value.stride(1) >= 0 && value.stride(2) >= 0,
                "value strides must be non-negative");

    const int Nq = static_cast<int>(key.size(0));
    const int Hkv = static_cast<int>(key.size(1));
    const int dim = static_cast<int>(key.size(2));

    TORCH_CHECK(static_cast<int>(value.size(1)) == Hkv &&
                static_cast<int>(value.size(2)) == dim,
                "value shape mismatch");
    TORCH_CHECK(steady_start_block_ids.size(0) == Hkv,
                "steady_start_block_ids Hkv mismatch");
    TORCH_CHECK(steady_end_block_ids.size(0) == Hkv,
                "steady_end_block_ids Hkv mismatch");
    TORCH_CHECK(evicted_key.size(1) == Hkv && evicted_key.size(2) == dim,
                "evicted key shape mismatch");

    auto error_code = torch::empty({1}, steady_state.options());
    const int total_blocks = static_cast<int>(block_storage.size(1));
    const int block_size = static_cast<int>(block_storage.size(2));

    const at::cuda::OptionalCUDAGuard device_guard(device_of(block_storage));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int rc = update_sparse_steady_kv_launcher_raw(
        block_storage.data_ptr(),
        map_storage_dtype(block_storage.scalar_type()),
        steady_start_block_ids.data_ptr<int32_t>(),
        static_cast<int32_t>(steady_start_block_ids.size(1)),
        steady_end_block_ids.data_ptr<int32_t>(),
        static_cast<int32_t>(steady_end_block_ids.size(1)),
        steady_state.data_ptr<int32_t>(),
        evicted_key.data_ptr(),
        evicted_value.data_ptr(),
        evicted_count.data_ptr<int32_t>(),
        key.data_ptr(),
        value.data_ptr(),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        Nq, Hkv,
        total_blocks, block_size, dim,
        static_cast<int>(steady_start_capacity),
        static_cast<int>(steady_end_capacity),
        static_cast<int>(evicted_key.size(0)),
        error_code.data_ptr<int32_t>(),
        stream);
    TORCH_CHECK(rc == ERR_OK,
                "update_sparse_steady_kv launcher failed, rc=", rc);
    return {evicted_key, evicted_value, evicted_count};
}

torch::Tensor sparse_select_topk_clusters_out_cuda(
    torch::Tensor query,
    torch::Tensor cluster_centers_T,
    torch::Tensor mean,
    torch::Tensor cluster_center_count,
    int64_t nprobe,
    torch::Tensor out_top_clusters
) {
    check_cuda(query, "query");
    check_cuda_contig(cluster_centers_T, "cluster_centers_T");
    check_cuda_contig(mean, "mean");
    check_cuda_contig(cluster_center_count, "cluster_center_count");
    check_cuda_contig(out_top_clusters, "out_top_clusters");

    TORCH_CHECK(query.dim() == 3, "query must be [Nq,Hq,dim]");
    TORCH_CHECK(cluster_centers_T.dim() == 3,
                "cluster_centers_T must be [Hkv,dim,C]");
    TORCH_CHECK(mean.dim() == 2, "mean must be [Hkv,dim]");
    TORCH_CHECK(cluster_center_count.scalar_type() == torch::kInt32,
                "cluster_center_count must be int32");
    TORCH_CHECK(cluster_center_count.numel() == 1,
                "cluster_center_count must be [1]");
    TORCH_CHECK(out_top_clusters.scalar_type() == torch::kInt32,
                "out_top_clusters must be int32");

    TORCH_CHECK(query.scalar_type() == cluster_centers_T.scalar_type(),
                "query dtype must equal cluster_centers_T dtype");
    TORCH_CHECK(mean.scalar_type() == query.scalar_type(),
                "mean dtype must equal query dtype");
    TORCH_CHECK(query.stride(0) >= 0 && query.stride(1) >= 0 &&
                query.stride(2) >= 0, "query strides must be non-negative");

    const int Nq = static_cast<int>(query.size(0));
    const int Hq = static_cast<int>(query.size(1));
    const int dim = static_cast<int>(query.size(2));
    const int Hkv = static_cast<int>(cluster_centers_T.size(0));
    const int C = static_cast<int>(cluster_centers_T.size(2));
    const int nprobe_i = static_cast<int>(nprobe);

    TORCH_CHECK(cluster_centers_T.size(1) == dim,
                "cluster_centers_T dim mismatch");
    TORCH_CHECK(mean.size(0) == Hkv && mean.size(1) == dim,
                "mean shape mismatch");
    TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0, "Hq must be divisible by Hkv");
    TORCH_CHECK(nprobe_i > 0 && nprobe_i <= C && nprobe_i <= 32,
                "nprobe must be in (0, min(C, 32)]");
    TORCH_CHECK(out_top_clusters.dim() == 3 &&
                out_top_clusters.size(0) == Nq &&
                out_top_clusters.size(1) == Hq &&
                out_top_clusters.size(2) == nprobe_i,
                "out_top_clusters must be [Nq,Hq,nprobe]");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int rc = sparse_select_topk_clusters_launcher_raw(
        query.data_ptr(),
        cluster_centers_T.data_ptr(),
        mean.data_ptr(),
        cluster_center_count.data_ptr<int32_t>(),
        out_top_clusters.data_ptr<int32_t>(),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        map_storage_dtype(query.scalar_type()),
        Nq,
        Hq,
        Hkv,
        C,
        dim,
        nprobe_i,
        stream);

    TORCH_CHECK(rc == ERR_OK,
                "sparse_select_topk_clusters launcher failed, rc=", rc);
    return out_top_clusters;
}

torch::Tensor group_avg_topk_clusters_by_kv_group_out_cuda(
    torch::Tensor query,
    torch::Tensor cluster_centers_T,
    torch::Tensor mean,
    torch::Tensor cluster_center_count,
    int64_t nprobe,
    torch::Tensor head_scores,
    torch::Tensor out_top_clusters
) {
    check_cuda(query, "query");
    check_cuda_contig(cluster_centers_T, "cluster_centers_T");
    check_cuda_contig(mean, "mean");
    check_cuda_contig(cluster_center_count, "cluster_center_count");
    check_cuda_contig(head_scores, "head_scores");
    check_cuda_contig(out_top_clusters, "out_top_clusters");

    TORCH_CHECK(query.dim() == 3, "query must be [Nq,Hq,dim]");
    TORCH_CHECK(cluster_centers_T.dim() == 3,
                "cluster_centers_T must be [Hkv,dim,C]");
    TORCH_CHECK(mean.dim() == 2, "mean must be [Hkv,dim]");
    TORCH_CHECK(cluster_center_count.scalar_type() == torch::kInt32,
                "cluster_center_count must be int32");
    TORCH_CHECK(cluster_center_count.numel() == 1,
                "cluster_center_count must be [1]");
    TORCH_CHECK(head_scores.scalar_type() == torch::kFloat32,
                "head_scores must be float32");
    TORCH_CHECK(out_top_clusters.scalar_type() == torch::kInt32,
                "out_top_clusters must be int32");

    TORCH_CHECK(query.scalar_type() == cluster_centers_T.scalar_type(),
                "query dtype must equal cluster_centers_T dtype");
    TORCH_CHECK(mean.scalar_type() == query.scalar_type(),
                "mean dtype must equal query dtype");
    TORCH_CHECK(query.stride(0) >= 0 && query.stride(1) >= 0 &&
                query.stride(2) >= 0, "query strides must be non-negative");

    const int Nq = static_cast<int>(query.size(0));
    const int Hq = static_cast<int>(query.size(1));
    const int dim = static_cast<int>(query.size(2));
    const int Hkv = static_cast<int>(cluster_centers_T.size(0));
    const int C = static_cast<int>(cluster_centers_T.size(2));
    const int nprobe_i = static_cast<int>(nprobe);

    TORCH_CHECK(cluster_centers_T.size(1) == dim,
                "cluster_centers_T dim mismatch");
    TORCH_CHECK(mean.size(0) == Hkv && mean.size(1) == dim,
                "mean shape mismatch");
    TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0, "Hq must be divisible by Hkv");
    TORCH_CHECK(nprobe_i > 0 && nprobe_i <= C && nprobe_i <= 32,
                "nprobe must be in (0, min(C, 32)]");
    TORCH_CHECK(head_scores.dim() == 3 &&
                head_scores.size(0) == Nq &&
                head_scores.size(1) == Hq &&
                head_scores.size(2) == C,
                "head_scores must be [Nq,Hq,C]");
    TORCH_CHECK(out_top_clusters.dim() == 3 &&
                out_top_clusters.size(0) == Nq &&
                out_top_clusters.size(1) == Hkv &&
                out_top_clusters.size(2) == nprobe_i,
                "out_top_clusters must be [Nq,Hkv,nprobe]");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int rc = group_avg_topk_clusters_by_kv_group_launcher_raw(
        query.data_ptr(),
        cluster_centers_T.data_ptr(),
        mean.data_ptr(),
        cluster_center_count.data_ptr<int32_t>(),
        head_scores.data_ptr<float>(),
        out_top_clusters.data_ptr<int32_t>(),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        map_storage_dtype(query.scalar_type()),
        Nq,
        Hq,
        Hkv,
        C,
        dim,
        nprobe_i,
        stream);

    TORCH_CHECK(rc == ERR_OK,
                "group_avg_topk_clusters_by_kv_group launcher failed, rc=", rc);
    return out_top_clusters;
}

torch::Tensor union_topk_clusters_by_kv_group_out_cuda(
    torch::Tensor top_clusters,
    int64_t hkv,
    int64_t num_clusters,
    torch::Tensor out_top_clusters
) {
    check_cuda_contig(top_clusters, "top_clusters");
    check_cuda_contig(out_top_clusters, "out_top_clusters");

    TORCH_CHECK(top_clusters.scalar_type() == torch::kInt32,
                "top_clusters must be int32");
    TORCH_CHECK(out_top_clusters.scalar_type() == torch::kInt32,
                "out_top_clusters must be int32");
    TORCH_CHECK(top_clusters.dim() == 3,
                "top_clusters must be [Nq,Hq,nprobe]");
    TORCH_CHECK(out_top_clusters.dim() == 3,
                "out_top_clusters must be [Nq,Hkv,union_nprobe]");

    const int Nq = static_cast<int>(top_clusters.size(0));
    const int Hq = static_cast<int>(top_clusters.size(1));
    const int nprobe = static_cast<int>(top_clusters.size(2));
    const int Hkv = static_cast<int>(hkv);
    const int C = static_cast<int>(num_clusters);
    const int union_nprobe = static_cast<int>(out_top_clusters.size(2));

    TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0, "Hq must be divisible by Hkv");
    TORCH_CHECK(C > 0, "num_clusters must be positive");
    TORCH_CHECK(nprobe > 0, "nprobe must be positive");
    TORCH_CHECK(union_nprobe > 0 && union_nprobe <= C,
                "union_nprobe must be in (0, num_clusters]");
    TORCH_CHECK(out_top_clusters.size(0) == Nq &&
                    out_top_clusters.size(1) == Hkv,
                "out_top_clusters shape mismatch");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(top_clusters));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int rc = union_topk_clusters_by_kv_group_launcher_raw(
        top_clusters.data_ptr<int32_t>(),
        out_top_clusters.data_ptr<int32_t>(),
        Nq,
        Hq,
        Hkv,
        C,
        nprobe,
        union_nprobe,
        stream);

    TORCH_CHECK(rc == ERR_OK,
                "union_topk_clusters_by_kv_group launcher failed, rc=", rc);
    return out_top_clusters;
}

TORCH_LIBRARY_FRAGMENT(_C, ops) {
    ops.def(
        "append_kv_to_clusters("
        "  Tensor block_storage,"
        "  Tensor cluster_compact_block_ids,"
        "  Tensor cluster_temp_kv_pos,"
        "  Tensor cluster_total_kv_counts,"
        "  Tensor temp_block_ids,"
        "  Tensor temp_block_kv_counts,"
        "  Tensor temp_block_kv_owner,"
        "  Tensor free_block_ids,"
        "  Tensor key,"
        "  Tensor value,"
        "  Tensor label"
        ") -> Tensor");
    ops.def(
        "append_kv_to_clusters_inplace("
        "  Tensor block_storage,"
        "  Tensor cluster_compact_block_ids,"
        "  Tensor cluster_temp_kv_pos,"
        "  Tensor cluster_total_kv_counts,"
        "  Tensor temp_block_ids,"
        "  Tensor temp_block_kv_counts,"
        "  Tensor temp_block_kv_owner,"
        "  Tensor free_block_ids,"
        "  Tensor used_free_block_count,"
        "  Tensor error_code,"
        "  Tensor key,"
        "  Tensor value,"
        "  Tensor label"
        ") -> Tensor");
    ops.def(
        "append_kv_to_clusters_by_centers_inplace("
        "  Tensor block_storage,"
        "  Tensor cluster_compact_block_ids,"
        "  Tensor cluster_temp_kv_pos,"
        "  Tensor cluster_total_kv_counts,"
        "  Tensor temp_block_ids,"
        "  Tensor temp_block_kv_counts,"
        "  Tensor temp_block_kv_owner,"
        "  Tensor free_block_ids,"
        "  Tensor used_free_block_count,"
        "  Tensor error_code,"
        "  Tensor key,"
        "  Tensor value,"
        "  Tensor cluster_centers_T,"
        "  Tensor mean,"
        "  Tensor cluster_center_count,"
        "  Tensor input_token_count"
        ") -> Tensor");
    ops.def(
        "append_kv_to_clusters_by_centers_with_steady_inplace("
        "  Tensor block_storage,"
        "  Tensor cluster_compact_block_ids,"
        "  Tensor cluster_temp_kv_pos,"
        "  Tensor cluster_total_kv_counts,"
        "  Tensor temp_block_ids,"
        "  Tensor temp_block_kv_counts,"
        "  Tensor temp_block_kv_owner,"
        "  Tensor free_block_ids,"
        "  Tensor used_free_block_count,"
        "  Tensor error_code,"
        "  Tensor steady_start_block_ids,"
        "  Tensor steady_end_block_ids,"
        "  Tensor steady_state,"
        "  Tensor key,"
        "  Tensor value,"
        "  Tensor cluster_centers_T,"
        "  Tensor mean,"
        "  Tensor cluster_center_count,"
        "  int steady_start_capacity,"
        "  int steady_end_capacity"
        ") -> Tensor");
    ops.def(
        "update_sparse_steady_kv_inplace("
        "  Tensor block_storage,"
        "  Tensor steady_start_block_ids,"
        "  Tensor steady_end_block_ids,"
        "  Tensor steady_state,"
        "  Tensor evicted_key,"
        "  Tensor evicted_value,"
        "  Tensor evicted_count,"
        "  Tensor key,"
        "  Tensor value,"
        "  int steady_start_capacity,"
        "  int steady_end_capacity"
        ") -> Tensor[]");
    ops.def(
        "sparse_select_topk_clusters_out("
        "  Tensor query,"
        "  Tensor cluster_centers_T,"
        "  Tensor mean,"
        "  Tensor cluster_center_count,"
        "  int nprobe,"
        "  Tensor out_top_clusters"
        ") -> Tensor");
    ops.def(
        "union_topk_clusters_by_kv_group_out("
        "  Tensor top_clusters,"
        "  int hkv,"
        "  int num_clusters,"
        "  Tensor out_top_clusters"
        ") -> Tensor");
    ops.def(
        "group_avg_topk_clusters_by_kv_group_out("
        "  Tensor query,"
        "  Tensor cluster_centers_T,"
        "  Tensor mean,"
        "  Tensor cluster_center_count,"
        "  int nprobe,"
        "  Tensor head_scores,"
        "  Tensor out_top_clusters"
        ") -> Tensor");
}

TORCH_LIBRARY_IMPL(_C, CUDA, m) {
    m.impl("append_kv_to_clusters", &append_kv_to_clusters_cuda);
    m.impl("append_kv_to_clusters_inplace", &append_kv_to_clusters_inplace_cuda);
    m.impl("append_kv_to_clusters_by_centers_inplace",
           &append_kv_to_clusters_by_centers_inplace_cuda);
    m.impl("append_kv_to_clusters_by_centers_with_steady_inplace",
           &append_kv_to_clusters_by_centers_with_steady_inplace_cuda);
    m.impl("update_sparse_steady_kv_inplace",
           &update_sparse_steady_kv_inplace_cuda);
    m.impl("sparse_select_topk_clusters_out",
           &sparse_select_topk_clusters_out_cuda);
    m.impl("union_topk_clusters_by_kv_group_out",
           &union_topk_clusters_by_kv_group_out_cuda);
    m.impl("group_avg_topk_clusters_by_kv_group_out",
           &group_avg_topk_clusters_by_kv_group_out_cuda);
}
