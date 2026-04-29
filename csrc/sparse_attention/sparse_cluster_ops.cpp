#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "sparse_attention_common.h"

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

torch::Tensor append_kv_to_clusters_cuda(
    torch::Tensor block_storage,      // [2,total_blocks,block_size,dim], in/out
    torch::Tensor cluster_block_ids,  // [Hkv,C,maxB], in/out
    torch::Tensor cluster_sizes,      // [Hkv,C], in/out
    torch::Tensor free_block_ids,     // [max_free_block]
    torch::Tensor key,                // [Nq,Hkv,dim]
    torch::Tensor value,              // [Nq,Hkv,dim]
    torch::Tensor label               // [Nq,Hkv]
) {
  check_cuda_contig(block_storage, "block_storage");
  check_cuda_contig(cluster_block_ids, "cluster_block_ids");
  check_cuda_contig(cluster_sizes, "cluster_sizes");
  check_cuda_contig(free_block_ids, "free_block_ids");
  check_cuda_contig(key, "key");
  check_cuda_contig(value, "value");
  check_cuda_contig(label, "label");

  TORCH_CHECK(cluster_block_ids.scalar_type() == torch::kInt32,
              "cluster_block_ids must be int32");
  TORCH_CHECK(cluster_sizes.scalar_type() == torch::kInt32,
              "cluster_sizes must be int32");
  TORCH_CHECK(free_block_ids.scalar_type() == torch::kInt32,
              "free_block_ids must be int32");
  TORCH_CHECK(label.scalar_type() == torch::kInt32, "label must be int32");

  TORCH_CHECK(block_storage.dim() == 4 && block_storage.size(0) == 2,
              "block_storage must be [2,total_blocks,block_size,dim]");
  TORCH_CHECK(cluster_block_ids.dim() == 3,
              "cluster_block_ids must be [Hkv,C,maxB]");
  TORCH_CHECK(cluster_sizes.dim() == 2, "cluster_sizes must be [Hkv,C]");
  TORCH_CHECK(free_block_ids.dim() == 1, "free_block_ids must be [max_free_block]");
  TORCH_CHECK(key.dim() == 3 && value.dim() == 3, "key/value must be [Nq,Hkv,dim]");
  TORCH_CHECK(label.dim() == 2, "label must be [Nq,Hkv]");

  TORCH_CHECK(key.scalar_type() == block_storage.scalar_type(),
              "key dtype must equal block_storage dtype");
  TORCH_CHECK(value.scalar_type() == block_storage.scalar_type(),
              "value dtype must equal block_storage dtype");
  TORCH_CHECK(key.sizes() == value.sizes(), "key/value shape mismatch");

  const int Nq = static_cast<int>(key.size(0));
  const int Hkv = static_cast<int>(key.size(1));
  const int dim = static_cast<int>(key.size(2));

  TORCH_CHECK(label.size(0) == Nq && label.size(1) == Hkv, "label shape mismatch");
  TORCH_CHECK(cluster_block_ids.size(0) == Hkv &&
              cluster_sizes.size(0) == Hkv, "Hkv mismatch");

  const int C = static_cast<int>(cluster_block_ids.size(1));
  const int maxB = static_cast<int>(cluster_block_ids.size(2));
  TORCH_CHECK(cluster_sizes.size(1) == C, "C mismatch");

  TORCH_CHECK(static_cast<int>(block_storage.size(3)) == dim, "dim mismatch");
  const int total_blocks = static_cast<int>(block_storage.size(1));
  const int block_size = static_cast<int>(block_storage.size(2));
  const int max_free_block = static_cast<int>(free_block_ids.size(0));

  auto i32_opts = cluster_sizes.options();
  auto used_free_block_count = torch::zeros({1}, i32_opts);
  auto error_code = torch::zeros({1}, i32_opts);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(block_storage));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int rc = append_kv_to_clusters_launcher_raw(
      key.data_ptr(),
      value.data_ptr(),
      map_storage_dtype(block_storage.scalar_type()),
      label.data_ptr<int32_t>(),
      free_block_ids.data_ptr<int32_t>(),
      max_free_block,
      block_storage.data_ptr(),
      cluster_block_ids.data_ptr<int32_t>(),
      cluster_sizes.data_ptr<int32_t>(),
      used_free_block_count.data_ptr<int32_t>(),
      error_code.data_ptr<int32_t>(),
      Nq, Hkv, C, maxB, total_blocks, block_size, dim,
      stream);

  TORCH_CHECK(rc == ERR_OK, "append_kv_to_clusters launcher failed, rc=", rc);

  return used_free_block_count;
}

TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "append_kv_to_clusters("
      "  Tensor block_storage,"
      "  Tensor cluster_block_ids,"
      "  Tensor cluster_sizes,"
      "  Tensor free_block_ids,"
      "  Tensor key,"
      "  Tensor value,"
      "  Tensor label"
      ") -> Tensor");
}

TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("append_kv_to_clusters", &append_kv_to_clusters_cuda);
}