# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Batched CUDA implementation of ``_sparse_select_tokens`` (stage-1 kernel).

The original code path inside ``GPUModelRunner._build_sparse_runtime_q_head_gather``
loops over requests in Python and calls into a sequence of small PyTorch
ops (``bmm`` + ``softmax`` + ``topk`` + ``cumsum``) for each request.  At
batch sizes of a few dozen the loop dispatch overhead dominates.

This module exposes a single batched entry point ``sparse_select_tokens``
that:

1. invokes a custom CUDA kernel computing, for every
   ``(req, kv_head, q_in_group)`` triplet, the softmax over centroid
   scores ``q @ K^T / sqrt(d)`` and **directly accumulates** the
   probabilities across the GQA group via ``atomicAdd`` so the output
   is the group-summed ``[num_reqs, num_kv_heads, num_clusters]``
   distribution (no separate ``.sum(dim=group)`` kernel needed);
2. accepts per-request centres as a **list** (or pre-built pointer
   table) so no ``torch.stack`` copy is required;
3. runs ``torch.topk`` for the ``nprobe`` selection and a final
   ``int32`` ``cumsum`` of the selected ``cluster_size`` rows.

Grid layout in the CUDA kernel is ``(num_reqs, num_kv_heads, group_size)``
which matches the request: even when ``group_size`` is not a power of two
(e.g. ``7`` for Qwen3-style GQA), each query head still gets its own
block.  Inside each block the dot product is parallelized across the warp
in a flash-attention style, with a single block-level softmax over the
2048 cluster dimension and ~9 KB of shared memory per block.
"""

from __future__ import annotations

import math
import os
import threading
from typing import List, Sequence, Tuple

import torch
from torch.utils.cpp_extension import load as _torch_load_ext

__all__ = [
    "sparse_select_tokens",
    "sparse_cluster_scores",
    "fused_topk_cumsum",
    "batched_sparse_select_dynamic_only",
    "reference_sparse_select_tokens",
]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CSRC_PATH = os.path.join(_THIS_DIR, "csrc", "sparse_select.cu")

_EXT = None
_EXT_LOCK = threading.Lock()


def _get_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    with _EXT_LOCK:
        if _EXT is not None:
            return _EXT
        extra_cuda_cflags = [
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ]
        _EXT = _torch_load_ext(
            name="vllm_sparse_select_ext",
            sources=[_CSRC_PATH],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=False,
        )
        return _EXT


def _ptr_table_from_stacked(stacked: torch.Tensor) -> torch.Tensor:
    """Synthesize an int64 ``[num_reqs]`` pointer table from a stacked tensor.

    The table is built on GPU in a single ``arange`` kernel launch so the
    stacked-tensor and list-of-tensors paths converge on the same kernel.
    """
    num_reqs = stacked.size(0)
    elem_size = stacked.element_size()
    stride_r_bytes = stacked.stride(0) * elem_size
    offsets = torch.arange(
        num_reqs, dtype=torch.int64, device=stacked.device
    ) * stride_r_bytes
    return offsets + stacked.data_ptr()


def _ptr_table_from_list(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Build an int64 ``[num_reqs]`` pointer table from per-request tensors.

    A small (``num_reqs * 8`` bytes) host-to-device copy.  All input
    tensors must share dtype and strides.
    """
    assert len(tensors) > 0, "expected at least one tensor"
    device = tensors[0].device
    # Build on CPU then async-copy.  ``tensors`` is small enough that the
    # Python list comprehension is fine.
    return torch.tensor(
        [t.data_ptr() for t in tensors],
        dtype=torch.int64,
        device=device,
    )


def fused_topk_cumsum(
    group_scores: torch.Tensor,
    sizes_ptrs: torch.Tensor,
    num_kv_heads: int,
    num_clusters: int,
    nprobe: int,
    stride_sizes_h_elems: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stage 2: fused top-K + per-request size gather + cumulative sum.

    Replaces the four-PyTorch-op tail of ``_sparse_select_tokens``
    (``torch.topk`` + ``torch.stack`` + ``torch.gather`` + ``torch.cumsum``)
    with a single CUDA kernel that does the top-K via
    ``cub::BlockRadixSort``, gathers ``cluster_size`` via a per-request
    ``int64`` device pointer table (no ``torch.stack`` copy), and runs
    ``cub::BlockScan::InclusiveSum`` for the cumsum.

    Args:
      group_scores:         ``[num_reqs, num_kv_heads, num_clusters]`` ``float32``
                            (the stage-1 output of ``sparse_cluster_scores``).
      sizes_ptrs:           ``[num_reqs]`` ``int64`` device pointer table where
                            ``sizes_ptrs[r]`` points to a contiguous
                            ``[num_kv_heads, num_clusters]`` ``int32`` tensor
                            (all per-request size tensors must share strides).
      stride_sizes_h_elems: per-head stride of the size tensors, in elements.

    Returns:
      top_indices:          ``[num_reqs, num_kv_heads, nprobe]`` ``int64``
                            (indices sorted by descending score).
      cluster_start_index:  ``[num_reqs, num_kv_heads, nprobe]`` ``int32``
                            (inclusive cumsum of the gathered sizes).
    """
    return _get_ext().fused_topk_cumsum(
        group_scores,
        sizes_ptrs,
        int(num_kv_heads),
        int(num_clusters),
        int(nprobe),
        int(stride_sizes_h_elems),
    )


def sparse_cluster_scores(
    query: torch.Tensor,
    query_start_loc: torch.Tensor,
    cluster_centres: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Stage 1: group-summed softmax probabilities, ``[num_reqs, num_kv_heads, num_clusters]``.

    Args:
      query:           ``[total_tokens, num_q_heads, head_dim]`` (bf16/fp16/fp32).
                       Must be contiguous along the head_dim axis.
      query_start_loc: ``[num_reqs + 1]`` ``int32`` cumulative query lengths
                       (the canonical ``self.query_start_loc.gpu[: num_reqs + 1]``).
      cluster_centres: ``[num_reqs, num_kv_heads, num_clusters, head_dim]``
                       (same dtype as ``query``, contiguous along head_dim).
      group_size:      ``num_q_heads // num_kv_heads`` (the GQA group size,
                       e.g. 7 for Qwen3-32B).

    Returns:
      ``[num_reqs, num_kv_heads, num_clusters]`` ``float32`` tensor of
      probabilities summed across the GQA group dim.  Each
      ``[req, kv_head, :]`` row sums to ``group_size`` (not 1).
    """
    assert cluster_centres.dim() == 4, (
        "stacked cluster_centres expected as [R, H, C, D]"
    )
    assert cluster_centres.stride(3) == 1, (
        "cluster_centres must be contiguous along head_dim"
    )
    num_kv_heads = cluster_centres.size(1)
    num_clusters = cluster_centres.size(2)
    head_dim = cluster_centres.size(3)
    ptr_table = _ptr_table_from_stacked(cluster_centres)
    return _get_ext().sparse_cluster_scores(
        query,
        query_start_loc,
        ptr_table,
        int(num_kv_heads),
        int(num_clusters),
        int(head_dim),
        int(cluster_centres.stride(1)),
        int(cluster_centres.stride(2)),
        int(group_size),
    )


def sparse_select_tokens(
    query: torch.Tensor,
    query_start_loc: torch.Tensor,
    cluster_centres: torch.Tensor,
    cluster_size: torch.Tensor,
    nprobe: int,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """End-to-end batched cluster selection (stacked inputs, test-friendly).

    Calls the stage-1 ``sparse_cluster_scores`` kernel followed by the
    stage-2 ``fused_topk_cumsum`` kernel.  No intermediate ``torch.topk``
    / ``torch.stack`` / ``torch.gather`` / ``torch.cumsum`` calls.

    Args:
      query:           ``[total_tokens, num_q_heads, head_dim]``.
      query_start_loc: ``[num_reqs + 1]`` ``int32``.
      cluster_centres: ``[num_reqs, num_kv_heads, num_clusters, head_dim]``.
      cluster_size:    ``[num_reqs, num_kv_heads, num_clusters]`` ``int32``
                       tensor of per-cluster member counts.
      nprobe:          number of clusters to pick per (req, kv_head).
      group_size:      GQA group size.

    Returns:
      top_clusters:        ``[num_reqs, num_kv_heads, nprobe]`` ``int64``.
      cluster_start_index: ``[num_reqs, num_kv_heads, nprobe]`` ``int32``.
    """
    assert cluster_size.dtype == torch.int32, (
        "cluster_size must be int32; cast on the caller side if needed"
    )
    assert cluster_size.dim() == 3
    assert cluster_size.stride(2) == 1, (
        "cluster_size must be contiguous along the clusters dim"
    )
    num_kv_heads = cluster_centres.size(1)
    num_clusters = cluster_centres.size(2)

    group_scores = sparse_cluster_scores(
        query, query_start_loc, cluster_centres, group_size
    )
    sizes_ptrs = _ptr_table_from_stacked(cluster_size)
    return fused_topk_cumsum(
        group_scores,
        sizes_ptrs,
        num_kv_heads=num_kv_heads,
        num_clusters=num_clusters,
        nprobe=int(nprobe),
        stride_sizes_h_elems=int(cluster_size.stride(1)),
    )


def batched_sparse_select_dynamic_only(
    q_flat: torch.Tensor,
    full_query_start_loc_gpu: torch.Tensor,
    active_batch_indices: torch.Tensor,
    per_req_centres: List[torch.Tensor],
    per_req_sizes: List[torch.Tensor],
    nprobe: int,
    group_size: int,
    centres_ptrs: torch.Tensor | None = None,
    sizes_ptrs: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integration-friendly wrapper for ``GPUModelRunner``.

    Designed to be a drop-in replacement for the original Python
    ``for req`` loop inside ``_build_sparse_runtime_q_head_gather``.

    * ``q_flat`` is ``query.view(-1, num_q_heads, head_dim)``.
    * ``full_query_start_loc_gpu`` is
      ``self.query_start_loc.gpu[: num_reqs + 1]`` (int32).
    * ``active_batch_indices`` lists, in batch order, the positions of
      requests that go through the *dynamic* cluster-scoring path.
      When this length equals ``len(full_query_start_loc_gpu) - 1`` the
      compacted-QSL construction is skipped (fast path).
    * ``per_req_centres`` / ``per_req_sizes`` are the per-request
      ``layer_stats.cluster_centres`` / ``cluster_size`` tensors.  They
      are consumed *without* a ``torch.stack`` copy -- the centres are
      passed by pointer to the kernel.
    * ``centres_ptrs`` / ``sizes_ptrs`` optionally supply the int64
      ``[num_active]`` device pointer tables directly.  When provided
      (the caller has cached them across decode steps), the two blocking
      ``torch.tensor(list, device=cuda)`` host->device copies are skipped.
      The per-request tensors are still used for shape/stride metadata.

    Returns ``(top_clusters, cluster_start_index)`` with shapes
    ``[num_active, num_kv_heads, nprobe]`` (int64) and
    ``[num_active, num_kv_heads, nprobe]`` (int32) respectively.
    """
    device = q_flat.device
    num_active = len(per_req_centres)
    assert num_active == len(per_req_sizes), (
        "per_req_centres/per_req_sizes length mismatch"
    )
    assert num_active > 0, "expected at least one active request"
    assert active_batch_indices.numel() == num_active, (
        "active_batch_indices length mismatch"
    )
    assert active_batch_indices.device == device, (
        "active_batch_indices must live on the same device as q_flat"
    )

    num_full = int(full_query_start_loc_gpu.numel() - 1)

    # Fast path: active subset *is* the whole batch -- reuse the full
    # QSL directly and skip the compacted-QSL construction (saves 2
    # kernel launches plus a small allocation; ~30 us at small batches).
    if num_active == num_full:
        qsl_to_use = full_query_start_loc_gpu
    else:
        compacted_qsl = torch.empty(
            num_active + 1, dtype=torch.int32, device=device
        )
        compacted_qsl[0] = 0
        compacted_qsl[1:] = full_query_start_loc_gpu[
            active_batch_indices.to(torch.int64) + 1
        ]
        qsl_to_use = compacted_qsl

    # Build the pointer table from the per-request centres list -- no
    # ``torch.stack`` copy (~170 us savings at batch=64 vs. stacking).
    # The caller may pass a cached table to skip this blocking H2D copy
    # entirely (the centres tensors are stable across a request's decode).
    if centres_ptrs is None:
        centres_ptrs = _ptr_table_from_list(per_req_centres)

    first_c = per_req_centres[0]
    num_kv_heads, num_clusters, head_dim = first_c.shape
    stride_c_h = int(first_c.stride(0))
    stride_c_c = int(first_c.stride(1))

    # Stage 1: kernel writes [R, H, C] (group-summed via atomicAdd).
    group_scores = _get_ext().sparse_cluster_scores(
        q_flat,
        qsl_to_use,
        centres_ptrs,
        int(num_kv_heads),
        int(num_clusters),
        int(head_dim),
        stride_c_h,
        stride_c_c,
        int(group_size),
    )

    # Stage 2: fused top-K + per-request size gather + cumsum, all in
    # one kernel.  ``per_req_sizes`` is consumed via a pointer table
    # (no ``torch.stack`` of the int32 sizes tensors, no ``torch.gather``
    # / ``torch.cumsum`` follow-ups).
    first_s = per_req_sizes[0]
    assert first_s.dtype == torch.int32, (
        "per_req_sizes elements must be int32"
    )
    assert first_s.stride(1) == 1, (
        "per_req_sizes elements must be contiguous along the clusters dim"
    )
    if sizes_ptrs is None:
        sizes_ptrs = _ptr_table_from_list(per_req_sizes)
    return fused_topk_cumsum(
        group_scores,
        sizes_ptrs,
        num_kv_heads=num_kv_heads,
        num_clusters=num_clusters,
        nprobe=int(nprobe),
        stride_sizes_h_elems=int(first_s.stride(0)),
    )


def reference_sparse_select_tokens(
    query: torch.Tensor,
    query_start_loc: torch.Tensor,
    cluster_centres: torch.Tensor,
    cluster_size: torch.Tensor,
    nprobe: int,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference matching the original ``_sparse_select_tokens``."""
    num_reqs = query_start_loc.numel() - 1
    num_kv_heads, num_clusters, head_dim = cluster_centres.shape[1:]

    last_idx = query_start_loc[1:].long() - 1
    q_last = query[last_idx]
    q_grouped = q_last.view(num_reqs, num_kv_heads, group_size, head_dim)

    q32 = q_grouped.float()
    c32 = cluster_centres.float()
    scores = torch.einsum("rhgd,rhcd->rhgc", q32, c32) / math.sqrt(head_dim)
    probs = torch.softmax(scores, dim=-1)
    group_scores = probs.sum(dim=2)

    top_clusters = torch.topk(group_scores, k=int(nprobe), dim=-1).indices
    sel_sizes = cluster_size.gather(-1, top_clusters)
    cluster_start_index = torch.cumsum(sel_sizes, dim=-1, dtype=torch.int32)
    return top_clusters, cluster_start_index
