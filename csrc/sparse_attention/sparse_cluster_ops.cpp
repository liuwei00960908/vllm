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
        Nq, Hkv, C, maxB,
        total_blocks, block_size, dim,
        max_temp_blocks,
        stream);

    TORCH_CHECK(rc == ERR_OK, "append_kv_to_clusters launcher failed, rc=", rc);
    return used_free_block_count;
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
}

TORCH_LIBRARY_IMPL(_C, CUDA, m) {
    m.impl("append_kv_to_clusters", &append_kv_to_clusters_cuda);
    m.impl("append_kv_to_clusters_inplace", &append_kv_to_clusters_inplace_cuda);
}