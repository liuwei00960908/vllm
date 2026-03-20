# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from vllm.config.kv_transfer import KVTransferConfig
from vllm.v1.topk_history.kmeans import kmeans_select_representatives
from vllm.v1.topk_history.types import SparseSelectionResult


@dataclass
class TopKHistoryManager:
    """Tracks per-request sparse KV selection for PD transfer (experimental).

    **Prefill (scheduler-side fallback)**: without GPU key vectors, selects
    prefix / suffix tokens plus evenly-spaced indices from the middle pool.

    **With key vectors (worker hook, future)**: runs k-means representatives
    merged with forced prefix/suffix (and sliding-window policy in decode).
    """

    topk_clusters: int = 64
    prefix_keep: int = 8
    tail_keep: int = 16
    sliding_window: int = 512
    kmeans_iters: int = 5
    # Per-request optional state for decode-time window merges (logical pool).
    _pools: dict[str, list[int]] = field(default_factory=dict)

    @classmethod
    def from_kv_config(cls, cfg: KVTransferConfig) -> TopKHistoryManager:
        return cls(
            topk_clusters=cfg.sparse_kv_topk_k,
            prefix_keep=cfg.sparse_kv_prefix_keep,
            tail_keep=cfg.sparse_kv_tail_keep,
            sliding_window=cfg.sparse_kv_sliding_window,
            kmeans_iters=cfg.sparse_kv_kmeans_iters,
        )

    def forced_logical_indices(self, seq_len: int) -> list[int]:
        if seq_len <= 0:
            return []
        p = min(self.prefix_keep, seq_len)
        t = min(self.tail_keep, seq_len)
        head = list(range(0, p))
        tail_start = max(0, seq_len - t)
        tail = list(range(tail_start, seq_len))
        return sorted(set(head + tail))

    def middle_pool(self, seq_len: int, forced: set[int]) -> list[int]:
        return [i for i in range(seq_len) if i not in forced]

    def plan_prefill_logical_indices(self, seq_len: int) -> list[int]:
        """Heuristic selection when key vectors are not available (e.g. on
        scheduler)."""
        forced = set(self.forced_logical_indices(seq_len))
        budget = max(0, min(self.topk_clusters, seq_len) - len(forced))
        pool = self.middle_pool(seq_len, forced)
        if budget <= 0 or not pool:
            return sorted(forced)

        step = max(1, math.ceil(len(pool) / budget))
        extra = pool[::step][:budget]
        return sorted(forced.union(extra))

    def select_with_keys(
        self,
        keys: torch.Tensor,
        *,
        layer_name: str | None = None,
        forced_indices: list[int] | None = None,
    ) -> SparseSelectionResult:
        """Select indices using k-means on RoPE (or other) key features."""
        seq_len = keys.shape[0]
        forced_list = (
            forced_indices
            if forced_indices is not None
            else self.forced_logical_indices(seq_len)
        )
        forced_t = torch.tensor(forced_list, dtype=torch.long, device=keys.device)
        extra_k = max(0, self.topk_clusters - len(set(forced_list)))
        indices_t = kmeans_select_representatives(
            keys,
            extra_k,
            forced_indices=forced_t,
            num_iters=self.kmeans_iters,
        )
        out = SparseSelectionResult(indices=indices_t, per_layer_indices={})
        if layer_name is not None:
            out.per_layer_indices[layer_name] = indices_t
        return out

    def register_decode_pool_merged(
        self,
        request_id: str,
        spilled_logical_indices: list[int],
    ) -> None:
        """Append tokens that left the sliding window into the global pool."""
        pool = self._pools.setdefault(request_id, [])
        pool.extend(spilled_logical_indices)

    def pop_request_state(self, request_id: str) -> None:
        self._pools.pop(request_id, None)
