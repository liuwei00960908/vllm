#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <mutex>
#include <unordered_map>
#include <vector>

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
    const int32_t* d_steady_start_block_ids,
    int32_t steady_start_blocks,
    const int32_t* d_steady_end_block_ids,
    int32_t steady_end_blocks,
    const int32_t* d_steady_state,
    int32_t* d_out_block_table,
    int32_t* d_out_bt_len,
    int32_t* d_out_seqused_k,
    int32_t* d_state,                 // [3] => error_code, used_free_block_count, total_plan_count
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
    torch::Tensor state;               // [3] => error_code, used_free_block_count, total_plan_count
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
        const int64_t new_rows = std::max<int64_t>(
            rows_needed,
            std::max<int64_t>(ws.rows_capacity * 2, 1024));

        ws.row_free_base = torch::empty({new_rows}, i32_opts);
        ws.row_plan_base = torch::empty({new_rows}, i32_opts);
        ws.rows_capacity = new_rows;
    }

    if (ws.plan_capacity < plan_needed) {
        const int64_t new_plan = std::max<int64_t>(
            plan_needed,
            std::max<int64_t>(ws.plan_capacity * 2, 4096));

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
    int64_t max_bt_len);

static void build_sparse_block_table_cuda_impl(
    torch::Tensor top_clusters,
    torch::Tensor cluster_compact_block_ids,
    torch::Tensor cluster_temp_kv_pos,
    torch::Tensor cluster_total_kv_counts,
    torch::Tensor temp_block_ids,
    torch::Tensor block_storage,
    torch::Tensor free_block_ids,
    torch::Tensor steady_start_block_ids,
    torch::Tensor steady_end_block_ids,
    torch::Tensor steady_state,
    int64_t max_bt_len,
    torch::Tensor out_block_table,
    torch::Tensor out_bt_len,
    torch::Tensor out_seqused_k,
    torch::Tensor workspace_state,
    torch::Tensor workspace_row_free_base,
    torch::Tensor workspace_row_plan_base,
    torch::Tensor workspace_plan_row,
    torch::Tensor workspace_plan_src_tb_idx,
    torch::Tensor workspace_plan_src_tb_off) {

    check_cuda_contiguous(top_clusters, "top_clusters");
    check_cuda_contiguous(cluster_compact_block_ids, "cluster_compact_block_ids");
    check_cuda_contiguous(cluster_temp_kv_pos, "cluster_temp_kv_pos");
    check_cuda_contiguous(cluster_total_kv_counts, "cluster_total_kv_counts");
    check_cuda_contiguous(temp_block_ids, "temp_block_ids");
    check_cuda_contiguous(block_storage, "block_storage");
    check_cuda_contiguous(free_block_ids, "free_block_ids");
    check_cuda_contiguous(steady_start_block_ids, "steady_start_block_ids");
    check_cuda_contiguous(steady_end_block_ids, "steady_end_block_ids");
    check_cuda_contiguous(steady_state, "steady_state");
    check_cuda_contiguous(out_block_table, "out_block_table");
    check_cuda_contiguous(out_bt_len, "out_bt_len");
    check_cuda_contiguous(out_seqused_k, "out_seqused_k");
    check_cuda_contiguous(workspace_state, "workspace_state");
    check_cuda_contiguous(workspace_row_free_base, "workspace_row_free_base");
    check_cuda_contiguous(workspace_row_plan_base, "workspace_row_plan_base");
    check_cuda_contiguous(workspace_plan_row, "workspace_plan_row");
    check_cuda_contiguous(workspace_plan_src_tb_idx, "workspace_plan_src_tb_idx");
    check_cuda_contiguous(workspace_plan_src_tb_off, "workspace_plan_src_tb_off");

    TORCH_CHECK(top_clusters.scalar_type() == torch::kInt32, "top_clusters must be int32");
    TORCH_CHECK(cluster_compact_block_ids.scalar_type() == torch::kInt32, "cluster_compact_block_ids must be int32");
    TORCH_CHECK(cluster_temp_kv_pos.scalar_type() == torch::kInt32, "cluster_temp_kv_pos must be int32");
    TORCH_CHECK(cluster_total_kv_counts.scalar_type() == torch::kInt32, "cluster_total_kv_counts must be int32");
    TORCH_CHECK(temp_block_ids.scalar_type() == torch::kInt32, "temp_block_ids must be int32");
    TORCH_CHECK(free_block_ids.scalar_type() == torch::kInt32, "free_block_ids must be int32");
    TORCH_CHECK(steady_start_block_ids.scalar_type() == torch::kInt32,
                "steady_start_block_ids must be int32");
    TORCH_CHECK(steady_end_block_ids.scalar_type() == torch::kInt32,
                "steady_end_block_ids must be int32");
    TORCH_CHECK(steady_state.scalar_type() == torch::kInt32,
                "steady_state must be int32");

    TORCH_CHECK(top_clusters.dim() == 3, "top_clusters shape must be [NQ,Hq,nprobe]");
    TORCH_CHECK(cluster_compact_block_ids.dim() == 3, "cluster_compact_block_ids shape must be [Hkv,C,maxB]");
    TORCH_CHECK(cluster_temp_kv_pos.dim() == 4, "cluster_temp_kv_pos shape must be [Hkv,C,block_size,2]");
    TORCH_CHECK(cluster_total_kv_counts.dim() == 2, "cluster_total_kv_counts shape must be [Hkv,C]");
    TORCH_CHECK(temp_block_ids.dim() == 1, "temp_block_ids shape must be [max_temp_blocks]");
    TORCH_CHECK(block_storage.dim() == 4, "block_storage shape must be [2,total_blocks,block_size,dim]");
    TORCH_CHECK(free_block_ids.dim() == 1, "free_block_ids shape must be [max_free_block]");
    TORCH_CHECK(steady_start_block_ids.dim() == 2,
                "steady_start_block_ids shape must be [Hkv,start_blocks]");
    TORCH_CHECK(steady_end_block_ids.dim() == 2,
                "steady_end_block_ids shape must be [Hkv,end_blocks]");
    TORCH_CHECK(steady_state.dim() == 1 && steady_state.numel() >= 4,
                "steady_state shape must be [>=4]");

    TORCH_CHECK(max_bt_len > 0, "max_bt_len must be > 0");
    TORCH_CHECK(max_bt_len <= INT32_MAX, "max_bt_len too large: ", max_bt_len);

    const int NQ = static_cast<int>(top_clusters.size(0));
    const int Hq = static_cast<int>(top_clusters.size(1));
    const int nprobe = static_cast<int>(top_clusters.size(2));

    TORCH_CHECK(nprobe > 0, "nprobe must be > 0");
    TORCH_CHECK(nprobe <= WARP_SIZE,
                "nprobe must be <= WARP_SIZE, got nprobe=", nprobe,
                ", WARP_SIZE=", WARP_SIZE);

    const int Hkv = static_cast<int>(cluster_compact_block_ids.size(0));
    const int C = static_cast<int>(cluster_compact_block_ids.size(1));
    const int maxB = static_cast<int>(cluster_compact_block_ids.size(2));

    TORCH_CHECK(cluster_total_kv_counts.size(0) == Hkv &&
                cluster_total_kv_counts.size(1) == C,
                "cluster_total_kv_counts shape mismatch");
    TORCH_CHECK(steady_start_block_ids.size(0) == Hkv,
                "steady_start_block_ids Hkv mismatch");
    TORCH_CHECK(steady_end_block_ids.size(0) == Hkv,
                "steady_end_block_ids Hkv mismatch");

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

    TORCH_CHECK(block_size > 0, "block_size must be > 0");

    TORCH_CHECK(cluster_temp_kv_pos.size(2) == block_size,
                "cluster_temp_kv_pos third dim must equal block_size");

    const int max_free_block = static_cast<int>(free_block_ids.size(0));
    const int rows = NQ * Hq;
    const int steady_blocks_per_head =
        static_cast<int>(steady_start_block_ids.size(1) +
                         steady_end_block_ids.size(1));

    const int64_t max_plan_count_64 =
        static_cast<int64_t>(rows) *
        static_cast<int64_t>(nprobe + steady_blocks_per_head) *
        static_cast<int64_t>(std::max(block_size - 1, 0));

    TORCH_CHECK(max_plan_count_64 >= 0 && max_plan_count_64 <= INT32_MAX,
                "max_plan_count invalid: ", max_plan_count_64);

    const int64_t plan_capacity_needed = std::max<int64_t>(max_plan_count_64, 1);

    TORCH_CHECK(out_block_table.scalar_type() == torch::kInt32 &&
                    out_block_table.numel() == static_cast<int64_t>(rows) * max_bt_len,
                "out_block_table must have shape [rows, max_bt_len]");
    TORCH_CHECK(out_bt_len.scalar_type() == torch::kInt32 &&
                    out_bt_len.numel() == rows,
                "out_bt_len must have shape [rows]");
    TORCH_CHECK(out_seqused_k.scalar_type() == torch::kInt32 &&
                    out_seqused_k.numel() == rows,
                "out_seqused_k must have shape [rows]");
    TORCH_CHECK(workspace_state.scalar_type() == torch::kInt32 &&
                    workspace_state.numel() >= 3,
                "workspace_state must have shape [>=3]");
    TORCH_CHECK(workspace_row_free_base.scalar_type() == torch::kInt32 &&
                    workspace_row_free_base.numel() >= rows,
                "workspace_row_free_base capacity too small");
    TORCH_CHECK(workspace_row_plan_base.scalar_type() == torch::kInt32 &&
                    workspace_row_plan_base.numel() >= rows,
                "workspace_row_plan_base capacity too small");
    TORCH_CHECK(workspace_plan_row.scalar_type() == torch::kInt32 &&
                    workspace_plan_row.numel() >= plan_capacity_needed,
                "workspace_plan_row capacity too small");
    TORCH_CHECK(workspace_plan_src_tb_idx.scalar_type() == torch::kInt32 &&
                    workspace_plan_src_tb_idx.numel() >= plan_capacity_needed,
                "workspace_plan_src_tb_idx capacity too small");
    TORCH_CHECK(workspace_plan_src_tb_off.scalar_type() == torch::kInt32 &&
                    workspace_plan_src_tb_off.numel() >= plan_capacity_needed,
                "workspace_plan_src_tb_off capacity too small");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(top_clusters));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

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
        steady_start_block_ids.data_ptr<int32_t>(),
        static_cast<int32_t>(steady_start_block_ids.size(1)),
        steady_end_block_ids.data_ptr<int32_t>(),
        static_cast<int32_t>(steady_end_block_ids.size(1)),
        steady_state.data_ptr<int32_t>(),
        out_block_table.data_ptr<int32_t>(),
        out_bt_len.data_ptr<int32_t>(),
        out_seqused_k.data_ptr<int32_t>(),
        workspace_state.data_ptr<int32_t>(),
        workspace_row_free_base.data_ptr<int32_t>(),
        workspace_row_plan_base.data_ptr<int32_t>(),
        workspace_plan_row.data_ptr<int32_t>(),
        workspace_plan_src_tb_idx.data_ptr<int32_t>(),
        workspace_plan_src_tb_off.data_ptr<int32_t>(),
        NQ, Hq, nprobe,
        Hkv, C, maxB,
        total_blocks, block_size, dim,
        static_cast<int>(max_bt_len),
        stream);

    TORCH_CHECK(rc == ERR_OK,
                "build_sparse_block_table_launcher_raw failed, rc=", rc);
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
    auto i32_opts = top_clusters.options().dtype(torch::kInt32);
    const int NQ = static_cast<int>(top_clusters.size(0));
    const int Hq = static_cast<int>(top_clusters.size(1));
    const int rows = NQ * Hq;
    const int64_t plan_capacity_needed = std::max<int64_t>(
        static_cast<int64_t>(rows) *
            static_cast<int64_t>(top_clusters.size(2)) *
            static_cast<int64_t>(
                std::max(static_cast<int>(block_storage.size(2)) - 1, 0)),
        1);
    auto out_block_table = torch::empty({rows, max_bt_len}, i32_opts);
    auto out_bt_len = torch::empty({rows}, i32_opts);
    auto out_seqused_k = torch::empty({rows}, i32_opts);
    const int Hkv = static_cast<int>(cluster_compact_block_ids.size(0));
    auto empty_steady_blocks = torch::empty({Hkv, 0}, i32_opts);
    auto empty_steady_state = torch::zeros({4}, i32_opts);
    const int device_index = top_clusters.get_device();
    auto& ws = get_workspace(device_index, rows, plan_capacity_needed, i32_opts);

    build_sparse_block_table_cuda_impl(
        top_clusters,
        cluster_compact_block_ids,
        cluster_temp_kv_pos,
        cluster_total_kv_counts,
        temp_block_ids,
        block_storage,
        free_block_ids,
        empty_steady_blocks,
        empty_steady_blocks,
        empty_steady_state,
        max_bt_len,
        out_block_table,
        out_bt_len,
        out_seqused_k,
        ws.state,
        ws.row_free_base,
        ws.row_plan_base,
        ws.plan_row,
        ws.plan_src_tb_idx,
        ws.plan_src_tb_off);

    auto used_free_block_count = ws.state.slice(0, 1, 2);

    return {
        out_block_table.view({NQ, Hq, max_bt_len}),
        out_bt_len.view({NQ, Hq}),
        out_seqused_k.view({NQ, Hq}),
        used_free_block_count,
    };
}

std::vector<torch::Tensor> build_sparse_block_table_out_cuda(
    torch::Tensor top_clusters,
    torch::Tensor cluster_compact_block_ids,
    torch::Tensor cluster_temp_kv_pos,
    torch::Tensor cluster_total_kv_counts,
    torch::Tensor temp_block_ids,
    torch::Tensor block_storage,
    torch::Tensor free_block_ids,
    torch::Tensor steady_start_block_ids,
    torch::Tensor steady_end_block_ids,
    torch::Tensor steady_state,
    int64_t max_bt_len,
    torch::Tensor out_block_table,
    torch::Tensor out_bt_len,
    torch::Tensor out_seqused_k,
    torch::Tensor workspace_state,
    torch::Tensor workspace_row_free_base,
    torch::Tensor workspace_row_plan_base,
    torch::Tensor workspace_plan_row,
    torch::Tensor workspace_plan_src_tb_idx,
    torch::Tensor workspace_plan_src_tb_off) {
    const int NQ = static_cast<int>(top_clusters.size(0));
    const int Hq = static_cast<int>(top_clusters.size(1));
    build_sparse_block_table_cuda_impl(
        top_clusters,
        cluster_compact_block_ids,
        cluster_temp_kv_pos,
        cluster_total_kv_counts,
        temp_block_ids,
        block_storage,
        free_block_ids,
        steady_start_block_ids,
        steady_end_block_ids,
        steady_state,
        max_bt_len,
        out_block_table,
        out_bt_len,
        out_seqused_k,
        workspace_state,
        workspace_row_free_base,
        workspace_row_plan_base,
        workspace_plan_row,
        workspace_plan_src_tb_idx,
        workspace_plan_src_tb_off);
    return {
        out_block_table.view({NQ, Hq, max_bt_len}),
        out_bt_len.view({NQ, Hq}),
        out_seqused_k.view({NQ, Hq}),
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
    ops.def(
        "build_sparse_block_table_out("
        "  Tensor top_clusters,"
        "  Tensor cluster_compact_block_ids,"
        "  Tensor cluster_temp_kv_pos,"
        "  Tensor cluster_total_kv_counts,"
        "  Tensor temp_block_ids,"
        "  Tensor block_storage,"
        "  Tensor free_block_ids,"
        "  Tensor steady_start_block_ids,"
        "  Tensor steady_end_block_ids,"
        "  Tensor steady_state,"
        "  int max_bt_len,"
        "  Tensor out_block_table,"
        "  Tensor out_bt_len,"
        "  Tensor out_seqused_k,"
        "  Tensor workspace_state,"
        "  Tensor workspace_row_free_base,"
        "  Tensor workspace_row_plan_base,"
        "  Tensor workspace_plan_row,"
        "  Tensor workspace_plan_src_tb_idx,"
        "  Tensor workspace_plan_src_tb_off"
        ") -> Tensor[]");
}

TORCH_LIBRARY_IMPL(_C, CUDA, m) {
    m.impl("build_sparse_block_table", &build_sparse_block_table_cuda);
    m.impl("build_sparse_block_table_out", &build_sparse_block_table_out_cuda);
}