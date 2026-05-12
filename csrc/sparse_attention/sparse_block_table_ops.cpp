#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <vector>
#include <unordered_map>
#include <mutex>

#include "sparse_attention_common.h"

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
    int32_t* d_state,                 // [3] => error_code, used_free_block_count, total_plan_count
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
    cudaStream_t stream);

static inline int32_t map_storage_dtype(torch::ScalarType t) {
    if (t == torch::kFloat32) return DTYPE_FP32;
    if (t == torch::kFloat16) return DTYPE_FP16;
    if (t == torch::kBFloat16) return DTYPE_BF16;
    TORCH_CHECK(false, "Unsupported block_storage dtype: ", c10::toString(t));
}

static inline void check_cuda_contiguous(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

struct SparseBlockTableWorkspace {
    torch::Tensor state;               // [3]
    torch::Tensor compact_len;         // [rows_capacity]
    torch::Tensor free_blocks_needed;  // [rows_capacity]
    torch::Tensor temp_token_count;    // [rows_capacity]
    torch::Tensor row_free_base;       // [rows_capacity]
    torch::Tensor row_plan_base;       // [rows_capacity]
    torch::Tensor plan_row;            // [plan_capacity]
    torch::Tensor plan_src_tb_idx;     // [plan_capacity]
    torch::Tensor plan_src_tb_off;     // [plan_capacity]
    int64_t rows_capacity = 0;
    int64_t plan_capacity = 0;
};

static std::unordered_map<int, SparseBlockTableWorkspace> g_workspaces;
static std::mutex g_workspaces_mu;

static SparseBlockTableWorkspace& get_workspace(
    int device_index,
    int64_t rows_needed,
    int64_t plan_needed,
    const torch::TensorOptions& i32_opts) {

    std::lock_guard<std::mutex> lock(g_workspaces_mu);
    auto& ws = g_workspaces[device_index];

    if (!ws.state.defined()) {
        ws.state = torch::empty({3}, i32_opts);
    }

    if (ws.rows_capacity < rows_needed) {
        int64_t new_rows = std::max<int64_t>(rows_needed, std::max<int64_t>(ws.rows_capacity * 2, 1024));
        ws.compact_len = torch::empty({new_rows}, i32_opts);
        ws.free_blocks_needed = torch::empty({new_rows}, i32_opts);
        ws.temp_token_count = torch::empty({new_rows}, i32_opts);
        ws.row_free_base = torch::empty({new_rows}, i32_opts);
        ws.row_plan_base = torch::empty({new_rows}, i32_opts);
        ws.rows_capacity = new_rows;
    }

    if (ws.plan_capacity < plan_needed) {
        int64_t new_plan = std::max<int64_t>(plan_needed, std::max<int64_t>(ws.plan_capacity * 2, 4096));
        ws.plan_row = torch::empty({new_plan}, i32_opts);
        ws.plan_src_tb_idx = torch::empty({new_plan}, i32_opts);
        ws.plan_src_tb_off = torch::empty({new_plan}, i32_opts);
        ws.plan_capacity = new_plan;
    }

    return ws;
}

std::vector<torch::Tensor> build_sparse_block_table_cuda(
    torch::Tensor top_clusters,
    torch::Tensor cluster_compact_block_ids,
    torch::Tensor cluster_temp_kv_pos,
    torch::Tensor cluster_total_kv_counts,
    torch::Tensor temp_block_ids,
    torch::Tensor block_storage,
    torch::Tensor free_block_ids,
    int64_t max_bt_len) {

    check_cuda_contiguous(top_clusters, "top_clusters");
    check_cuda_contiguous(cluster_compact_block_ids, "cluster_compact_block_ids");
    check_cuda_contiguous(cluster_temp_kv_pos, "cluster_temp_kv_pos");
    check_cuda_contiguous(cluster_total_kv_counts, "cluster_total_kv_counts");
    check_cuda_contiguous(temp_block_ids, "temp_block_ids");
    check_cuda_contiguous(block_storage, "block_storage");
    check_cuda_contiguous(free_block_ids, "free_block_ids");

    TORCH_CHECK(top_clusters.scalar_type() == torch::kInt32, "top_clusters must be int32");
    TORCH_CHECK(cluster_compact_block_ids.scalar_type() == torch::kInt32, "cluster_compact_block_ids must be int32");
    TORCH_CHECK(cluster_temp_kv_pos.scalar_type() == torch::kInt32, "cluster_temp_kv_pos must be int32");
    TORCH_CHECK(cluster_total_kv_counts.scalar_type() == torch::kInt32, "cluster_total_kv_counts must be int32");
    TORCH_CHECK(temp_block_ids.scalar_type() == torch::kInt32, "temp_block_ids must be int32");
    TORCH_CHECK(free_block_ids.scalar_type() == torch::kInt32, "free_block_ids must be int32");

    TORCH_CHECK(top_clusters.dim() == 3, "top_clusters shape must be [NQ,Hq,nprobe]");
    TORCH_CHECK(cluster_compact_block_ids.dim() == 3, "cluster_compact_block_ids shape must be [Hkv,C,maxB]");
    TORCH_CHECK(cluster_temp_kv_pos.dim() == 4, "cluster_temp_kv_pos shape must be [Hkv,C,block_size,2]");
    TORCH_CHECK(cluster_total_kv_counts.dim() == 2, "cluster_total_kv_counts shape must be [Hkv,C]");
    TORCH_CHECK(temp_block_ids.dim() == 1, "temp_block_ids shape must be [max_temp_blocks]");
    TORCH_CHECK(block_storage.dim() == 4, "block_storage shape must be [2,total_blocks,block_size,dim]");
    TORCH_CHECK(free_block_ids.dim() == 1, "free_block_ids shape must be [max_free_block]");

    TORCH_CHECK(max_bt_len > 0, "max_bt_len must be > 0");

    const int NQ = static_cast<int>(top_clusters.size(0));
    const int Hq = static_cast<int>(top_clusters.size(1));
    const int nprobe = static_cast<int>(top_clusters.size(2));

    TORCH_CHECK(nprobe <= WARP_SIZE,
                "nprobe must be <= WARP_SIZE, got nprobe=", nprobe,
                ", WARP_SIZE=", WARP_SIZE);

    const int Hkv = static_cast<int>(cluster_compact_block_ids.size(0));
    const int C = static_cast<int>(cluster_compact_block_ids.size(1));
    const int maxB = static_cast<int>(cluster_compact_block_ids.size(2));

    TORCH_CHECK(cluster_total_kv_counts.size(0) == Hkv &&
                cluster_total_kv_counts.size(1) == C,
                "cluster_total_kv_counts shape mismatch");

    TORCH_CHECK(cluster_temp_kv_pos.size(0) == Hkv &&
                cluster_temp_kv_pos.size(1) == C,
                "cluster_temp_kv_pos shape mismatch");

    TORCH_CHECK(cluster_temp_kv_pos.size(3) == 2,
                "cluster_temp_kv_pos last dim must be 2");

    TORCH_CHECK(block_storage.size(0) == 2,
                "block_storage first dim must be 2");

    const int total_blocks = static_cast<int>(block_storage.size(1));
    const int block_size = static_cast<int>(block_storage.size(2));
    const int dim = static_cast<int>(block_storage.size(3));

    TORCH_CHECK(cluster_temp_kv_pos.size(2) == block_size,
                "cluster_temp_kv_pos third dim must equal block_size");

    const int max_free_block = static_cast<int>(free_block_ids.size(0));
    const int rows = NQ * Hq;

    const int64_t max_plan_count_64 =
        static_cast<int64_t>(rows) * static_cast<int64_t>(nprobe) * static_cast<int64_t>(block_size - 1);
    TORCH_CHECK(max_plan_count_64 > 0 && max_plan_count_64 <= INT32_MAX,
                "max_plan_count invalid: ", max_plan_count_64);

    auto i32_opts = top_clusters.options().dtype(torch::kInt32);

    auto out_block_table = torch::full({rows, max_bt_len}, -1, i32_opts);
    auto out_bt_len = torch::zeros({rows}, i32_opts);
    auto out_seqused_k = torch::zeros({rows}, i32_opts);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(top_clusters));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int device_index = top_clusters.get_device();

    auto& ws = get_workspace(device_index, rows, max_plan_count_64, i32_opts);

    int rc = build_sparse_block_table_launcher_raw(
        top_clusters.data_ptr<int32_t>(),
        cluster_compact_block_ids.data_ptr<int32_t>(),
        cluster_temp_kv_pos.data_ptr<int32_t>(),
        cluster_total_kv_counts.data_ptr<int32_t>(),
        temp_block_ids.data_ptr<int32_t>(),
        block_storage.data_ptr(),
        map_storage_dtype(block_storage.scalar_type()),
        free_block_ids.data_ptr<int32_t>(),
        max_free_block,
        out_block_table.data_ptr<int32_t>(),
        out_bt_len.data_ptr<int32_t>(),
        out_seqused_k.data_ptr<int32_t>(),
        ws.state.data_ptr<int32_t>(),
        ws.compact_len.data_ptr<int32_t>(),
        ws.free_blocks_needed.data_ptr<int32_t>(),
        ws.temp_token_count.data_ptr<int32_t>(),
        ws.row_free_base.data_ptr<int32_t>(),
        ws.row_plan_base.data_ptr<int32_t>(),
        ws.plan_row.data_ptr<int32_t>(),
        ws.plan_src_tb_idx.data_ptr<int32_t>(),
        ws.plan_src_tb_off.data_ptr<int32_t>(),
        NQ, Hq, nprobe,
        Hkv, C, maxB,
        total_blocks, block_size, dim,
        static_cast<int>(max_bt_len),
        stream);

    TORCH_CHECK(rc == ERR_OK,
                "build_sparse_block_table_launcher_raw failed, rc=", rc);

    auto used_free_block_count = ws.state.slice(0, 1, 2);

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
        "  Tensor cluster_compact_block_ids,"
        "  Tensor cluster_temp_kv_pos,"
        "  Tensor cluster_total_kv_counts,"
        "  Tensor temp_block_ids,"
        "  Tensor block_storage,"
        "  Tensor free_block_ids,"
        "  int max_bt_len"
        ") -> Tensor[]");
}

TORCH_LIBRARY_IMPL(_C, CUDA, m) {
    m.impl("build_sparse_block_table", &build_sparse_block_table_cuda);
}