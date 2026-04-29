#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <vector>

#include "sparse_attention_common.h"

extern "C" int build_block_table_with_copy_launcher_raw(
    const int32_t* d_top_clusters, const int32_t* d_cluster_block_ids,
    const int32_t* d_cluster_sizes, void* d_block_storage,
    int32_t storage_dtype, const int32_t* d_free_block_ids,
    int32_t max_free_block, int32_t* d_out_block_table, int32_t* d_out_bt_len,
    int32_t* d_out_seqused_k, int32_t* d_used_free_block_count,
    int32_t* d_error_code, int NQ, int Hq, int nprobe, int Hkv, int C, int maxB,
    int total_blocks, int block_size, int dim, int max_bt_len, cudaStream_t stream);

static inline int32_t map_storage_dtype(torch::ScalarType t) {
  if (t == torch::kFloat32) return DTYPE_FP32;
  if (t == torch::kFloat16) return DTYPE_FP16;
  if (t == torch::kBFloat16) return DTYPE_BF16;
  TORCH_CHECK(false, "Unsupported block_storage dtype: ", c10::toString(t));
}

static inline void check_cuda_contiguous(const torch::Tensor& x,
                                         const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

std::vector<torch::Tensor> build_sparse_block_table_cuda(
    torch::Tensor top_clusters,       // [NQ,Hq,nprobe]           int32
    torch::Tensor cluster_block_ids,  // [Hkv,C,maxB]             int32
    torch::Tensor cluster_sizes,      // [Hkv,C]                  int32
    torch::Tensor block_storage,      // [2,total_blocks,bs,dim]  fp16/bf16/fp32
    torch::Tensor free_block_ids,     // [max_free_block]         int32
    int64_t max_bt_len) {
  check_cuda_contiguous(top_clusters, "top_clusters");
  check_cuda_contiguous(cluster_block_ids, "cluster_block_ids");
  check_cuda_contiguous(cluster_sizes, "cluster_sizes");
  check_cuda_contiguous(block_storage, "block_storage");
  check_cuda_contiguous(free_block_ids, "free_block_ids");

  TORCH_CHECK(top_clusters.scalar_type() == torch::kInt32,
              "top_clusters must be int32");
  TORCH_CHECK(cluster_block_ids.scalar_type() == torch::kInt32,
              "cluster_block_ids must be int32");
  TORCH_CHECK(cluster_sizes.scalar_type() == torch::kInt32,
              "cluster_sizes must be int32");
  TORCH_CHECK(free_block_ids.scalar_type() == torch::kInt32,
              "free_block_ids must be int32");

  TORCH_CHECK(top_clusters.dim() == 3,
              "top_clusters shape must be [NQ,Hq,nprobe]");
  TORCH_CHECK(cluster_block_ids.dim() == 3,
              "cluster_block_ids shape must be [Hkv,C,maxB]");
  TORCH_CHECK(cluster_sizes.dim() == 2, "cluster_sizes shape must be [Hkv,C]");
  TORCH_CHECK(block_storage.dim() == 4,
              "block_storage shape must be [2,total_blocks,block_size,dim]");
  TORCH_CHECK(free_block_ids.dim() == 1,
              "free_block_ids shape must be [max_free_block]");
  TORCH_CHECK(max_bt_len > 0, "max_bt_len must be > 0");

  const int NQ = static_cast<int>(top_clusters.size(0));
  const int Hq = static_cast<int>(top_clusters.size(1));
  const int nprobe = static_cast<int>(top_clusters.size(2));

  const int Hkv = static_cast<int>(cluster_block_ids.size(0));
  const int C = static_cast<int>(cluster_block_ids.size(1));
  const int maxB = static_cast<int>(cluster_block_ids.size(2));

  TORCH_CHECK(cluster_sizes.size(0) == Hkv && cluster_sizes.size(1) == C,
              "cluster_sizes shape mismatch with cluster_block_ids");

  TORCH_CHECK(block_storage.size(0) == 2, "block_storage first dim must be 2");
  const int total_blocks = static_cast<int>(block_storage.size(1));
  const int block_size = static_cast<int>(block_storage.size(2));
  const int dim = static_cast<int>(block_storage.size(3));

  const int max_free_block = static_cast<int>(free_block_ids.size(0));

  auto i32_opts = top_clusters.options().dtype(torch::kInt32);
  const int rows = NQ * Hq;

  auto out_block_table = torch::full({rows, max_bt_len}, -1, i32_opts);
  auto out_bt_len = torch::zeros({rows}, i32_opts);
  auto out_seqused_k = torch::zeros({rows}, i32_opts);
  auto used_free_block_count = torch::zeros({1}, i32_opts);
  auto error_code = torch::zeros({1}, i32_opts);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(top_clusters));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int rc = build_block_table_with_copy_launcher_raw(
      top_clusters.data_ptr<int32_t>(), cluster_block_ids.data_ptr<int32_t>(),
      cluster_sizes.data_ptr<int32_t>(), block_storage.data_ptr(),
      map_storage_dtype(block_storage.scalar_type()),
      free_block_ids.data_ptr<int32_t>(), max_free_block,
      out_block_table.data_ptr<int32_t>(), out_bt_len.data_ptr<int32_t>(),
      out_seqused_k.data_ptr<int32_t>(),
      used_free_block_count.data_ptr<int32_t>(), error_code.data_ptr<int32_t>(),
      NQ, Hq, nprobe, Hkv, C, maxB, total_blocks, block_size, dim,
      static_cast<int>(max_bt_len), stream);

  TORCH_CHECK(rc == ERR_OK,
              "build_block_table_with_copy_launcher_raw failed, rc=", rc);

  return {
      out_block_table.view({NQ, Hq, max_bt_len}),
      out_bt_len.view({NQ, Hq}),
      out_seqused_k.view({NQ, Hq}),
      used_free_block_count,
  };
}

TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "build_sparse_block_table("
      "  Tensor top_clusters,"
      "  Tensor cluster_block_ids,"
      "  Tensor cluster_sizes,"
      "  Tensor block_storage,"
      "  Tensor free_block_ids,"
      "  int max_bt_len"
      ") -> Tensor[]");
}

TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("build_sparse_block_table", &build_sparse_block_table_cuda);
}