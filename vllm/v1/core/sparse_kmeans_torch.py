# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU (or device) segment K-Means for sparse KV prefill indexing.

The torch path mirrors ``sparse_kv_cache_manager._segment_kmeans`` on centered
features. The optional Triton path directly calls the paged
``segment_k_means_paged`` operator on raw K/V cache for token prefill.
"""

from __future__ import annotations

import os
import time

import torch

from vllm.v1.attention.ops.triton_segment_kmeans import (
    segment_k_means_paged,
)

from vllm.logger import init_logger

logger = init_logger(__name__)


def _as_batched_kv_cache(
    kv_cache: torch.Tensor,
    name: str,
) -> tuple[torch.Tensor, bool]:
    if kv_cache.dim() == 3:
        return kv_cache.unsqueeze(2), True
    if kv_cache.dim() == 4:
        return kv_cache, False
    raise ValueError(
        f"{name} must be [B, S, D] or [B, S, H, D], "
        f"got {tuple(kv_cache.shape)}"
    )

def prefill_cluster_meta_from_kv_cache_device(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    num_tokens: int,
    *,
    value_cache: torch.Tensor | None = None,
    num_clusters: int,
    n_segment: int,
    is_centered: bool,
    n_iter: int = 15,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Build prefill K-Means metadata from raw paged KV cache.

    When Triton is enabled and ``value_cache`` is provided, token-granularity
    prefill (``is_centered=False``) calls the optimized
    ``segment_k_means_paged`` entry directly. Centered/block feature callers
    use the torch fallback.
    """

    key_cache, _ = _as_batched_kv_cache(kv_cache, "kv_cache")
    value_cache_b, _ = _as_batched_kv_cache(value_cache, "value_cache")

    if block_ids.dim() != 1:
        raise ValueError(
            "block_ids must be [num_selected_blocks], got "
            f"{tuple(block_ids.shape)}"
        )

    (
        centres,
        labels,
        clusters,
        cluster_size,
        value_sum,
    ) = segment_k_means_paged(
        key_cache,
        value_cache_b,
        block_ids,
        int(num_tokens),
        int(num_clusters),
        block_size=int(key_cache.shape[1]),
        num_iters=int(n_iter),
        num_segments=int(n_segment),
    )

    raw = {
        "cluster_centres": centres,
        "cluster_size": cluster_size,
        "clusters": clusters,
        "value_sum": value_sum,
    }
    return raw

