# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SparseKVManager – RetroInfer-style dynamic sparse attention (block or token units).

When ``SparseAttentionSpec.cluster_granularity == "token"``, clustering and
Top-K run on **per-token** key features; selected tokens are mapped to the
minimal set of logical KV blocks for paging.  FlashAttention still reads all
slots inside each loaded block (no per-token mask in the kernel yet), so FLOPs
per loaded block are unchanged; only **which** blocks are resident changes.

Algorithm (block-granularity default; token mode uses the same pipeline on rows)
--------------------------------------------------------------------------------

Prefill
  1. Model runner extracts **block-level key features** per attention layer
     (same ``SparseAttentionSpec`` group): ``layer → [num_blocks, D]`` mean K
     per block (tokens and KV-heads averaged within the layer).  Query vectors
     are likewise **per layer** (mean Q per layer).  A legacy path still accepts
     a single ``[num_blocks, D]`` / ``[D]`` array (treated as one synthetic
     layer ``__flat__``).

  2. ``indexing(req_id, block_features, mean_value_features)``
     - For **each layer** independently: mean-center that layer's block
       features, run **Segment K-Means**, restore centroids, accumulate value
       sums.  Cluster ids and centroids are not shared across layers.

  3. ``select(req_id, query_vec, ...)`` runs TopK **independently per layer**:
     each layer gets ``steady_zone ∪ retrieve_zone`` capped by
     ``max_selected_blocks`` using **that layer's** cluster scores only.
     The **union** of all per-layer logical indices is stored for
     ``allocate_new_blocks`` so every layer's chosen history blocks are
     resident in GPU memory; the model runner builds a **per-layer** sparse
     block table so each forward only attends within that layer's subset.

Decode
  - When ``SparseAttentionSpec.refresh_topk_each_decode`` is True (default),
    each step's query vector re-runs TopK against the current cluster index
    (same as prefill), always merging the steady zone
    (``static_pattern_start`` / ``static_pattern_end``).
  - When ``refresh_topk_each_decode`` is False, the cached prefill TopK
    selection can be reused until a dynamic index update invalidates it.
  - ``rebalance(req_id, new_block_feature, new_block_value_feature)`` tracks
    newly written decode blocks.  After ``update_threshold_blocks`` new blocks
    accumulate, a fresh Segment K-Means is run on them and new centroids are
    appended (mirroring RetroInfer's ``_update_kv_cache``).  The prefill TopK
    cache is then invalidated and re-computed.

Block allocation (per decode step)
  - ``remove_skipped_blocks`` – no-op (deferred to allocate_new_blocks).
  - ``get_num_blocks_to_allocate`` – returns ``max(|selected|+1 − |old|, 0)``.
  - ``allocate_new_blocks`` – frees all old blocks, allocates ``|selected|+1``
    fresh GPU slots (the +1 is for the current decode token).

Steady Zone
  - First ``n_sink_blocks`` blocks (attention sinks) and last ``n_recent_blocks``
    blocks (local window) are **always** included in the selection regardless
    of cluster scores.

TODO(estimation-zone)
---------------------
The Estimation Zone from RetroInfer is NOT implemented yet.  The required
pieces are already stored:

  ``_cluster_value_sum[req_id]``  → ``[n_clusters, D]``
      sum of block mean-V vectors for each cluster.
  ``_cluster_size[req_id]``       → ``[n_clusters]`` int
      number of blocks in each cluster.

To enable the Estimation Zone:
  1. In ``select()``: after choosing the top-nprobe retrieve clusters, pick
     the next ``estimation_zone_size`` clusters as the estimation set.
     Return their centroid and value_sum to the model runner via
     ``ModelRunnerOutput.sparse_estimation_zone``.
  2. In the attention kernel: compute
       ``est_output = softmax(q · C_est^T / sqrt(d)) × (V_sum_est / size_est)``
     using only the centroid as K-representative and the mean V.
  3. Merge with retrieve zone flash-attention output via log-sum-exp (lse).
  4. Add ``estimation_zone_size: int`` to ``SparseAttentionSpec`` and wire
     ``max_compute_cluster_num = nprobe + estimation_zone_size``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

import time

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import KVCacheSpec, SparseAttentionSpec
from vllm.v1.request import Request

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)

# Single-array API (tests / callers that pass one ``[num_blocks, D]`` matrix)
# is stored under this synthetic layer name.
SPARSE_LEGACY_FLAT_LAYER = "__flat__"

# Per-KV-head clustering uses ``{layer}##kv{idx}``; per-query-head selection / Q
# vectors use ``{layer}##qh{idx}``. Plain ``layer`` keys are normalized to
# ``layer##kv0`` / ``layer##qh0`` for backward compatibility with tests.
SPARSE_KV_KEY = "##kv"
SPARSE_QH_KEY = "##qh"

_SPARSE_FREE_PREFILL_AFTER_SAVE = (
    int(os.getenv("VLLM_SPARSE_FREE_PREFILL_AFTER_SAVE", "0")) == 1
)


def sparse_kv_unit_key(layer: str, kv_idx: int) -> str:
    return f"{layer}{SPARSE_KV_KEY}{kv_idx}"


def sparse_qh_unit_key(layer: str, qh_idx: int) -> str:
    return f"{layer}{SPARSE_QH_KEY}{qh_idx}"


# Normalized keys for the legacy single-matrix ``__flat__`` indexing path.
SPARSE_LEGACY_FLAT_KV_KEY = sparse_kv_unit_key(SPARSE_LEGACY_FLAT_LAYER, 0)
SPARSE_LEGACY_FLAT_QH_KEY = sparse_qh_unit_key(SPARSE_LEGACY_FLAT_LAYER, 0)


def parse_sparse_kv_key(key: str) -> tuple[str, int] | None:
    if SPARSE_KV_KEY not in key:
        return None
    layer, _, rest = key.partition(SPARSE_KV_KEY)
    try:
        return layer, int(rest)
    except ValueError:
        return None


def parse_sparse_qh_key(key: str) -> tuple[str, int] | None:
    if SPARSE_QH_KEY not in key:
        return None
    layer, _, rest = key.partition(SPARSE_QH_KEY)
    try:
        return layer, int(rest)
    except ValueError:
        return None


def _normalize_kv_feature_map(m: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k, v in m.items():
        if SPARSE_KV_KEY in k:
            out[k] = v
        else:
            out[sparse_kv_unit_key(k, 0)] = v
    return out


def _empty_b2c() -> np.ndarray:
    return np.empty((0,), dtype=np.int32)


def _empty_feats() -> np.ndarray:
    return np.empty((0, 0), dtype=np.float32)


def _grow_feats(buf: np.ndarray, size: int, row: np.ndarray
                ) -> tuple[np.ndarray, int]:
    """Append one ``[D]`` row to a ``[capacity, D]`` buffer, growing 2× when full.

    Returns ``(buf, size)`` where ``buf[:size]`` is the valid slice.  Decode
    rebalance calls this once per step; geometric growth keeps amortised cost
    O(1) instead of O(N) per append (``np.concatenate``) – the previous
    implementation re-copied every layer's full ``[N_prompt, D]`` buffer per
    decode step (~1.4 GB of memcpy per step for 112 KV heads × 24k tokens).
    """
    row_2d = np.asarray(row, dtype=np.float32).reshape(-1)
    d = int(row_2d.shape[0])
    if buf.ndim != 2 or buf.shape[1] != d:
        # Reinitialise with a small capacity – only happens if the state was
        # constructed from an empty placeholder.
        capacity = 16
        new_buf = np.empty((capacity, d), dtype=np.float32)
        new_buf[0] = row_2d
        return new_buf, 1
    capacity = buf.shape[0]
    if size < capacity:
        buf[size] = row_2d
        return buf, size + 1
    new_capacity = max(capacity * 2, size + 1)
    new_buf = np.empty((new_capacity, d), dtype=np.float32)
    new_buf[:size] = buf[:size]
    new_buf[size] = row_2d
    return new_buf, size + 1


def _grow_b2c(buf: np.ndarray, size: int, value: int
              ) -> tuple[np.ndarray, int]:
    """Append a single int to a growable int32 buffer (2× geometric growth)."""
    capacity = int(buf.shape[0]) if buf.ndim == 1 else 0
    if size < capacity:
        buf[size] = np.int32(value)
        return buf, size + 1
    new_capacity = max(capacity * 2, size + 1, 16)
    new_buf = np.empty((new_capacity,), dtype=np.int32)
    if size > 0:
        new_buf[:size] = buf[:size]
    new_buf[size] = np.int32(value)
    return new_buf, size + 1


class _SparseLayerIndexState:
    """Per-(request, attention-layer) clustering and feature buffers.

    ``block_to_cluster`` / ``all_block_features`` / ``all_value_features`` are
    ndarray-backed with preallocated capacity and ``*_size`` cursors so
    decode-step appends are amortised O(1).  The old implementation stored
    one ``[D]`` row per token as a Python list of small ``np.ndarray``
    objects plus a parallel ``list[int]`` – that triggered millions of
    per-element allocations inside ``indexing()`` (~7–8 s per prompt for
    N≈24 k × 112 KV units) and concat-on-append turned decode ``rebalance``
    into an O(N²) copy loop.  Consolidated ndarray + cursor growth collapses
    both paths to single large allocations.

    Public attribute surface (``all_block_features``, ``all_value_features``,
    ``block_to_cluster``) returns a valid-slice view so ``len(...)`` and
    indexing work identically to the old list-backed layout.
    """

    __slots__ = (
        "cluster_centres",
        "cluster_value_sum",
        "cluster_size",
        "_b2c_buf",
        "_b2c_size",
        "_abf_buf",
        "_abf_size",
        "_avf_buf",
        "_avf_size",
        "mean_key",
        "decode_block_buffer",
        "decode_value_buffer",
    )

    def __init__(
        self,
        cluster_centres: np.ndarray,
        cluster_value_sum: np.ndarray,
        cluster_size: np.ndarray,
        block_to_cluster: np.ndarray | None = None,
        all_block_features: np.ndarray | None = None,
        all_value_features: np.ndarray | None = None,
        mean_key: np.ndarray | None = None,
    ) -> None:
        self.cluster_centres = cluster_centres
        self.cluster_value_sum = cluster_value_sum
        self.cluster_size = cluster_size

        b2c = _empty_b2c() if block_to_cluster is None else np.asarray(
            block_to_cluster, dtype=np.int32
        )
        self._b2c_buf = b2c
        self._b2c_size = int(b2c.shape[0]) if b2c.ndim == 1 else 0

        abf = _empty_feats() if all_block_features is None else np.ascontiguousarray(
            all_block_features, dtype=np.float32
        )
        self._abf_buf = abf
        self._abf_size = int(abf.shape[0]) if abf.ndim == 2 else 0

        avf = _empty_feats() if all_value_features is None else np.ascontiguousarray(
            all_value_features, dtype=np.float32
        )
        self._avf_buf = avf
        self._avf_size = int(avf.shape[0]) if avf.ndim == 2 else 0

        self.mean_key = mean_key
        self.decode_block_buffer = []
        self.decode_value_buffer = []

    # -- block_to_cluster ---------------------------------------------------
    @property
    def block_to_cluster(self) -> np.ndarray:
        return self._b2c_buf[:self._b2c_size]

    @block_to_cluster.setter
    def block_to_cluster(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.int32)
        self._b2c_buf = arr
        self._b2c_size = int(arr.shape[0]) if arr.ndim == 1 else 0

    def append_b2c(self, value: int) -> None:
        self._b2c_buf, self._b2c_size = _grow_b2c(
            self._b2c_buf, self._b2c_size, int(value)
        )

    def write_b2c_range(self, start: int, values: np.ndarray) -> None:
        """Overwrite ``b2c[start:start+len(values)]``; grow if needed."""
        values = np.asarray(values, dtype=np.int32).reshape(-1)
        end = start + int(values.shape[0])
        if end > int(self._b2c_buf.shape[0]):
            new_capacity = max(int(self._b2c_buf.shape[0]) * 2, end, 16)
            new_buf = np.empty((new_capacity,), dtype=np.int32)
            if self._b2c_size > 0:
                new_buf[:self._b2c_size] = self._b2c_buf[:self._b2c_size]
            self._b2c_buf = new_buf
        self._b2c_buf[start:end] = values
        if end > self._b2c_size:
            self._b2c_size = end

    # -- all_block_features -------------------------------------------------
    @property
    def all_block_features(self) -> np.ndarray:
        return self._abf_buf[:self._abf_size]

    @all_block_features.setter
    def all_block_features(self, value: np.ndarray) -> None:
        arr = np.ascontiguousarray(value, dtype=np.float32)
        self._abf_buf = arr
        self._abf_size = int(arr.shape[0]) if arr.ndim == 2 else 0

    def append_block_feature(self, row: np.ndarray) -> None:
        self._abf_buf, self._abf_size = _grow_feats(
            self._abf_buf, self._abf_size, row
        )

    # -- all_value_features -------------------------------------------------
    @property
    def all_value_features(self) -> np.ndarray:
        return self._avf_buf[:self._avf_size]

    @all_value_features.setter
    def all_value_features(self, value: np.ndarray) -> None:
        arr = np.ascontiguousarray(value, dtype=np.float32)
        self._avf_buf = arr
        self._avf_size = int(arr.shape[0]) if arr.ndim == 2 else 0

    def append_value_feature(self, row: np.ndarray) -> None:
        self._avf_buf, self._avf_size = _grow_feats(
            self._avf_buf, self._avf_size, row
        )

    def reserve_capacity(self, extra_rows: int) -> None:
        """Pre-grow the ``block_to_cluster`` buffer so the next ``extra_rows``
        appends don't trigger a grow memcpy.

        Only touches ``_b2c_buf`` because that is the single row-aligned
        buffer whose *content* is read on the hot path (by
        ``_retrieve_zone_one_layer``).  ``all_block_features`` and
        ``all_value_features`` store no downstream-readable content – see
        ``set_prefill_with_capacity`` below for the cheaper lazy path used on
        those two.
        """
        if extra_rows <= 0:
            return
        need_b2c = self._b2c_size + extra_rows
        if int(self._b2c_buf.shape[0]) < need_b2c:
            new_buf = np.empty((need_b2c,), dtype=np.int32)
            if self._b2c_size > 0:
                new_buf[:self._b2c_size] = self._b2c_buf[:self._b2c_size]
            self._b2c_buf = new_buf

    def set_prefill_block_feature_shape(
        self, n_rows: int, d: int, extra_rows: int
    ) -> None:
        """Allocate the ``all_block_features`` backing buffer without copying
        prefill content.

        The logical size is set to ``n_rows`` but the first ``n_rows`` rows of
        the buffer hold uninitialised memory – every caller of
        ``all_block_features`` in production touches it only via ``len(...)``
        (``_num_index_units`` / ``_dynamic_update_layer`` / idempotence
        guard).  This avoids the ~1.4 GB of useless memcpy that
        ``reserve_capacity`` used to pay while still giving decode rebalances
        ``extra_rows`` of free appends.
        """
        total = int(n_rows) + max(int(extra_rows), 0)
        # ``np.empty`` leaves pages unmapped; physical RSS stays zero until a
        # row is actually written, which only happens for decode-appended
        # rows (positions ``n_rows`` onward).
        self._abf_buf = np.empty((total, int(d)), dtype=np.float32)
        self._abf_size = int(n_rows)

    def set_prefill_value_feature_shape(
        self, n_rows: int, d: int, extra_rows: int
    ) -> None:
        """Same as ``set_prefill_block_feature_shape`` but for ``all_value_features``."""
        total = int(n_rows) + max(int(extra_rows), 0)
        self._avf_buf = np.empty((total, int(d)), dtype=np.float32)
        self._avf_size = int(n_rows)


# ---------------------------------------------------------------------------
# CPU K-Means primitives (numpy only, dot-product / attention-score style)
# ---------------------------------------------------------------------------

def _kmeans_dot(
    features: np.ndarray,
    k: int,
    n_iter: int = 15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    K-Means using dot-product similarity on **mean-centered** features.

    Dot product on centered data is equivalent to minimising the sum of
    squared distances (L2-KMeans), which is the correct distance for
    attention-score cluster selection (RetroInfer §3).

    Args:
        features: ``[N, D]`` float32, **already mean-centered**.
        k:        Number of clusters (clamped to N).
        n_iter:   Lloyd iterations.
        seed:     RNG seed.

    Returns:
        centres: ``[k, D]`` float32.
        labels:  ``[N]`` int32 – cluster assignment per point.
    """
    N, D = features.shape
    k = min(k, N)
    if k == 0:
        return np.zeros((0, D), dtype=np.float32), np.zeros(N, dtype=np.int32)

    rng = np.random.default_rng(seed)
    centres = features[rng.choice(N, k, replace=False)].astype(np.float32).copy()
    feat = features.astype(np.float32)

    labels = np.zeros(N, dtype=np.int32)
    for _ in range(n_iter):
        # Maximise dot product ↔ assign to nearest centre after centering.
        sims = feat @ centres.T          # [N, k]
        labels = np.argmax(sims, axis=1).astype(np.int32)

        new_centres = np.zeros_like(centres)
        counts = np.bincount(labels, minlength=k).astype(np.float32)
        np.add.at(new_centres, labels, feat)
        has_members = counts > 0
        new_centres[has_members] /= counts[has_members, np.newaxis]
        new_centres[~has_members] = centres[~has_members]  # keep old centre
        centres = new_centres

    return centres, labels


def _segment_kmeans(
    features: np.ndarray,
    n_clusters: int,
    n_segments: int,
    n_iter: int = 15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment K-Means: divide the block sequence into ``n_segments`` position-
    aligned slices and run independent K-Means within each slice.

    This preserves temporal locality: blocks from the same era of the prompt
    are clustered together, preventing early and late tokens from merging into
    a single centroid (which would wash out their individual signatures).

    Mean-centering must be applied by the caller *before* this function;
    centroids are returned in the centered space.

    Args:
        features:   ``[N, D]`` float32 – mean-centered block key features.
        n_clusters: Target total centroids (distributed as evenly as possible).
        n_segments: Number of position segments.
        n_iter:     K-Means iterations per segment.
        seed:       Base RNG seed (each segment gets ``seed + seg``).

    Returns:
        centres:  ``[total_k, D]`` float32 – all segment centroids stacked.
        labels:   ``[N]`` int32 – global cluster id per block (0-indexed,
                  contiguous across segments).
        sizes:    ``[total_k]`` int32 – number of blocks per cluster.
    """
    N, D = features.shape
    if N == 0:
        return (
            np.zeros((0, D), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
        )

    n_segments = min(n_segments, N)
    k_per_seg = max(1, n_clusters // n_segments)

    all_centres: list[np.ndarray] = []
    labels = np.zeros(N, dtype=np.int32)
    cluster_offset = 0

    # Distribute blocks as evenly as possible across segments.
    # Last segment absorbs remainder.
    seg_starts = [i * (N // n_segments) for i in range(n_segments)]
    seg_ends = seg_starts[1:] + [N]

    for seg_idx, (start, end) in enumerate(zip(seg_starts, seg_ends)):
        seg_feat = features[start:end]
        k = min(k_per_seg, len(seg_feat))
        if k == 0:
            continue

        centres_seg, labels_seg = _kmeans_dot(
            seg_feat, k, n_iter=n_iter, seed=seed + seg_idx
        )
        all_centres.append(centres_seg)
        labels[start:end] = labels_seg + cluster_offset
        cluster_offset += k

    if not all_centres:
        empty_k = 0
        return (
            np.zeros((empty_k, D), dtype=np.float32),
            np.zeros(N, dtype=np.int32),
            np.zeros(empty_k, dtype=np.int32),
        )

    all_centres_arr = np.vstack(all_centres).astype(np.float32)
    sizes = np.bincount(labels, minlength=len(all_centres_arr)).astype(np.int32)
    return all_centres_arr, labels, sizes


# ---------------------------------------------------------------------------
# SparseKVManager
# ---------------------------------------------------------------------------

class SparseKVManager(FullAttentionManager):
    """
    ``SingleTypeKVCacheManager`` implementing RetroInfer-style sparse attention
    adapted for vLLM's paged block layout.

    See module docstring for the full algorithm description.

    Per-request CPU state
    ---------------------
    ``_layer_states``         ``layer_name → _SparseLayerIndexState`` – per-layer
                              centroids, ``block_to_cluster``, feature lists,
                              and decode buffers for dynamic K-Means updates.
    ``_pending_query``        ``layer_name → [D]`` – last Q vectors per layer.
    ``_selected_block_indices`` list[int] – **union** of per-layer logical blocks
                              for the next decode step (for physical allocation).
    ``_selected_block_indices_by_layer`` ``layer → list[int]`` – per-layer
                              capped selection used by the model runner.
    ``_prefill_topk_ready``   bool – whether per-layer prefill TopK cache is valid.
    ``_prefill_selected``     cached **union** selection (compat / allocation).
    ``_prefill_selected_by_layer`` cached per-layer selection after prefill.
    """

    def __init__(
        self,
        kv_cache_spec: SparseAttentionSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> None:
        super().__init__(
            kv_cache_spec,
            block_pool,
            enable_caching,
            kv_cache_group_id,
            dcp_world_size,
            pcp_world_size,
        )
        assert isinstance(kv_cache_spec, SparseAttentionSpec)
        self._spec: SparseAttentionSpec = kv_cache_spec

        # ── per-request, per-attention-layer clustering (CPU state; prefill
        #    K-Means may run on GPU in the model runner) ───────────────────
        self._layer_states: dict[str, dict[str, _SparseLayerIndexState]] = {}

        # ── selection state ───────────────────────────────────────────────
        self._pending_query: dict[str, dict[str, np.ndarray]] = {}
        self._selected_block_indices: dict[str, list[int]] = {}
        self._selected_block_indices_by_layer: dict[str, dict[str, list[int]]] = {}
        # Per-step retrieve-zone logical block indices (steady-zone excluded).
        self._selected_retrieve_block_indices: dict[str, list[int]] = {}
        self._selected_retrieve_block_indices_by_layer: dict[
            str, dict[str, list[int]]
        ] = {}
        # prefill TopK cache
        self._prefill_topk_ready: dict[str, bool] = {}
        self._prefill_selected: dict[str, list[int]] = {}
        self._prefill_selected_by_layer: dict[str, dict[str, list[int]]] = {}
        # Token-granularity selection (global token indices), per layer.
        self._selected_token_indices_by_layer: dict[str, dict[str, list[int]]] = {}
        self._prefill_selected_tokens_by_layer: dict[str, dict[str, list[int]]] = {}

        self._current_step: int = 0

        # Reuse existing sparse debug switch used by GPUModelRunner.
        self._debug_state_transitions: bool = bool(
            int(os.getenv("VLLM_SPARSE_DEBUG_DECODE_TOKENS", "0"))
        )
        # Probe logs are disabled by default; enable explicitly when needed.
        self._sparse_probe_info_enabled: bool = (
            int(os.getenv("VLLM_SPARSE_PROBE_INFO", "0")) == 1
        )
        # Per-step compact trace (req_id -> latest key state in this step).
        self._step_trace: dict[str, dict[str, object]] = {}

        # Prompt token count at end of prefill (token-granularity indexing only).
        self._prefill_token_count: dict[str, int] = {}
        # One-shot sparse probe per request to avoid log spam.
        self._first_select_probe_done: set[str] = set()

        # ── physical block tracking (Bug 1 fix) ──────────────────────────
        # _prefill_blocks[req_id]: permanent list of all physical blocks
        # allocated during prefill (indexed by logical block index 0..N-1).
        # Never freed until the request is fully freed.
        self._prefill_blocks: dict[str, list[KVCacheBlock]] = {}
        # _decode_block[req_id]: the active decode block. It is reused across
        # steps until full, then replaced.
        self._decode_block: dict[str, KVCacheBlock | None] = {}
        # Number of decode tokens already written into _decode_block.
        self._decode_block_fill: dict[str, int] = {}
        # Last observed num_tokens_main_model to estimate decode tokens
        # scheduled in the current step.
        self._last_num_tokens_main_model: dict[str, int] = {}

        self._prefill_offloaded: set[str] = set()
        self._scratch_blocks: dict[str, list[KVCacheBlock]] = {}
        self._decode_history_blocks: dict[str, list[KVCacheBlock]] = {}

    def _debug_log_state(self, request_id: str, phase: str, **kwargs) -> None:
        if not self._debug_state_transitions:
            return
        trace = self._step_trace.setdefault(request_id, {})
        trace["phase"] = phase
        trace["step"] = self._current_step
        for key in (
            "prefill_topk_ready",
            "selected_count",
            "selected_logical_blocks",
            "retrieve_logical_blocks",
            "selected_preview",
            "used_prefill_cache",
            "total_blocks",
            "buffered_blocks",
            "threshold",
            "added_clusters",
            "total_clusters",
        ):
            if key in kwargs:
                trace[key] = kwargs[key]
        payload = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
        logger.info(
            "Sparse state trace: step=%d req_id=%s phase=%s %s",
            self._current_step,
            request_id,
            phase,
            payload,
        )

    # ------------------------------------------------------------------
    # Required abstract-method implementations
    # ------------------------------------------------------------------

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        # Sparse attention does not participate in prefix caching: blocks are
        # selected dynamically and re-allocated each decode step.
        return tuple([] for _ in range(len(kv_cache_group_ids)))

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0

    # ------------------------------------------------------------------
    # Block allocation overrides
    # ------------------------------------------------------------------

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        # Returning 0 prevents the base-class remove_skipped_blocks from
        # treating any prefix as "outside the window".
        return 0

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        # No-op: freeing old decode-step blocks is deferred to
        # allocate_new_blocks so that the free-pool check in
        # KVCacheManager.allocate_slots remains correct.
        return

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int:
        # Offloaded requests skip the prefill branch even on the first decode
        # step: their prefill blocks were already freed, so the standard
        # prefill calc would over-allocate (re-allocating the entire prompt).
        if (
            request_id not in self.num_cached_block
            and request_id not in self._prefill_offloaded
        ):
            # Prefill: standard page-aligned calculation.
            return super().get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks,
                total_computed_tokens,
                num_tokens_main_model,
            )
        step_tokens = self._estimate_decode_tokens_this_step(
            request_id, num_tokens_main_model
        )
        extra = 0
        if (
            request_id in self._prefill_offloaded
            and request_id not in self._scratch_blocks
        ):
            extra = self._scratch_block_count()
        cur_decode = self._decode_block.get(request_id)
        if cur_decode is None or cur_decode.is_null:
            return extra + 1
        fill = self._decode_block_fill.get(request_id, 0)
        remaining = max(0, self.block_size - fill)
        return extra + (1 if step_tokens > remaining else 0)

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        if request_id in self.num_cached_block:
            # Running (decode) request: no prefix-cache hits possible.
            assert len(new_computed_blocks) == 0
            return
        # New (prefill) request: sparse attention ignores prefix hits.
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        # Keep this request in prefill mode for allocate_new_blocks().
        # If we set num_cached_block here, sparse manager will treat it as decode
        # immediately and only allocate one active decode block, which is wrong
        # for "new request + external computed tokens" load path.
        #
        # Transition to decode is handled later by cache_blocks(), after the
        # prefill-stage allocation has created enough slots.

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
    ) -> list[KVCacheBlock]:
        # Offloaded requests jump straight to the decode-offload allocator,
        # bypassing the prefill branch even on their first decode step.
        if request_id in self._prefill_offloaded:
            req_blocks = self.req_to_blocks[request_id]
            ret = self._allocate_new_blocks_offloaded(
                request_id, req_blocks, num_tokens_main_model
            )
            return ret

        if request_id not in self.num_cached_block:
            # Prefill: allocate sequentially for all prompt tokens.
            return super().allocate_new_blocks(
                request_id, num_tokens, num_tokens_main_model
            )

        req_blocks = self.req_to_blocks[request_id]

        if request_id not in self._prefill_blocks:
            self._prefill_blocks[request_id] = list(req_blocks)

        # Map selected logical indices → physical historical blocks.
        # The history list contains original prefill blocks followed by
        # finalized decode blocks in chronological order.
        prefill_blocks = self._prefill_blocks[request_id]
        if self.delegates_token_selection_to_runner():
            # Token compact gather selects tokens inside GPUModelRunner.
            # Keep the scheduler row chronological/full so token_id //
            # block_size indexes the same way it does for full attention.
            selected = range(len(prefill_blocks))
        else:
            selected = self._selected_block_indices.get(request_id, [])
        physical_selected = [
            prefill_blocks[i]
            for i in selected
            if i < len(prefill_blocks)
        ]

        step_tokens = self._estimate_decode_tokens_this_step(
            request_id, num_tokens_main_model
        )

        cur_decode = self._decode_block.get(request_id)
        fill = self._decode_block_fill.get(request_id, 0)
        allocated_new_decode = False
        if cur_decode is None or cur_decode.is_null:
            new_decode = self.block_pool.get_new_blocks(1)[0]
            self._decode_block[request_id] = new_decode
            self._decode_block_fill[request_id] = 0
            cur_decode = new_decode
            fill = 0
            allocated_new_decode = True
        elif fill + step_tokens > self.block_size:
            # Current decode block is full for this step; freeze it into
            # sparse-selectable history and start a fresh decode block.
            self._prefill_blocks[request_id].append(cur_decode)
            new_decode = self.block_pool.get_new_blocks(1)[0]
            self._decode_block[request_id] = new_decode
            self._decode_block_fill[request_id] = 0
            cur_decode = new_decode
            fill = 0
            allocated_new_decode = True

        # Rebuild req_to_blocks: [selected prefill blocks..., decode block].
        req_blocks.clear()
        req_blocks.extend(physical_selected)
        req_blocks.append(cur_decode)

        if allocated_new_decode:
            self.new_block_ids.append(cur_decode.block_id)
        self._decode_block_fill[request_id] = min(
            self.block_size, fill + step_tokens
        )
        self._last_num_tokens_main_model[request_id] = num_tokens_main_model
        self.num_cached_block[request_id] = 0

        # Return the FULL new block list so the model runner can rebuild
        # the block table row via add_row (not just append the new block).
        return list(req_blocks)

    def _allocate_new_blocks_offloaded(
        self,
        request_id: str,
        req_blocks: list[KVCacheBlock],
        num_tokens_main_model: int,
    ) -> list[KVCacheBlock]:
        if request_id not in self._scratch_blocks:
            n = self._scratch_block_count()
            new = self.block_pool.get_new_blocks(n) if n > 0 else []
            self._scratch_blocks[request_id] = new
            for b in new:
                self.new_block_ids.append(b.block_id)
            self._decode_history_blocks.setdefault(request_id, [])

        scratch = self._scratch_blocks[request_id]
        history = self._decode_history_blocks.setdefault(request_id, [])

        step_tokens = self._estimate_decode_tokens_this_step(
            request_id, num_tokens_main_model
        )
        cur_decode = self._decode_block.get(request_id)
        fill = self._decode_block_fill.get(request_id, 0)
        allocated_new_decode = False
        if cur_decode is None or cur_decode.is_null:
            cur_decode = self.block_pool.get_new_blocks(1)[0]
            self._decode_block[request_id] = cur_decode
            self._decode_block_fill[request_id] = 0
            fill = 0
            allocated_new_decode = True
        elif fill + step_tokens > self.block_size:
            history.append(cur_decode)
            cur_decode = self.block_pool.get_new_blocks(1)[0]
            self._decode_block[request_id] = cur_decode
            self._decode_block_fill[request_id] = 0
            fill = 0
            allocated_new_decode = True

        # Tail layout: LMCache-loaded scratch (selected prompt KV, block-aligned
        # and filled) occupies the FRONT of the row, the resident decode region
        # (history + active decode block) follows it.  This matches LMCache's
        # append-order block tracking (token_start_index=0) so no block-order
        # surgery is needed; ``seqused_k = n_scratch_tokens + D'`` then covers
        # [scratch | decode] contiguously and the decode tokens enter FA.
        req_blocks.clear()
        req_blocks.extend(scratch)
        req_blocks.extend(history)
        req_blocks.append(cur_decode)

        if allocated_new_decode:
            self.new_block_ids.append(cur_decode.block_id)
        self._decode_block_fill[request_id] = min(
            self.block_size, fill + step_tokens
        )
        self._last_num_tokens_main_model[request_id] = num_tokens_main_model
        self.num_cached_block[request_id] = 0
        return list(req_blocks)

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        # Sparse blocks are not entered into the prefix-hash cache.
        # Advance the pointer so repeated calls are idempotent.
        self.num_cached_block[request.request_id] = (
            num_tokens // self.block_size
        )

    def free(self, request_id: str) -> None:
        # Free non-selected prefill blocks that are no longer referenced by
        # req_to_blocks (they were "held" to preserve KV data but not picked
        # by the latest selection round).
        prefill_blocks = self._prefill_blocks.pop(request_id, [])
        current_block_ids = {
            id(b) for b in self.req_to_blocks.get(request_id, [])
        }
        orphaned = [
            b for b in prefill_blocks
            if id(b) not in current_block_ids and not b.is_null
        ]
        if orphaned:
            self.block_pool.free_blocks(reversed(orphaned))

        # Clear decode block ref (the block itself is in req_to_blocks and
        # will be freed by super().free()).
        self._decode_block.pop(request_id, None)
        self._decode_block_fill.pop(request_id, None)
        self._last_num_tokens_main_model.pop(request_id, None)
        self._prefill_offloaded.discard(request_id)
        scratch_blocks = self._scratch_blocks.pop(request_id, [])
        history_blocks = self._decode_history_blocks.pop(request_id, [])
        orphan_offload = [
            b for b in (*scratch_blocks, *history_blocks)
            if id(b) not in current_block_ids and not b.is_null
        ]
        if orphan_offload:
            self.block_pool.free_blocks(reversed(orphan_offload))

        super().free(request_id)
        self._layer_states.pop(request_id, None)
        self._pending_query.pop(request_id, None)
        self._selected_block_indices.pop(request_id, None)
        self._selected_block_indices_by_layer.pop(request_id, None)
        self._selected_retrieve_block_indices.pop(request_id, None)
        self._selected_retrieve_block_indices_by_layer.pop(request_id, None)
        self._prefill_topk_ready.pop(request_id, None)
        self._prefill_selected.pop(request_id, None)
        self._prefill_selected_by_layer.pop(request_id, None)
        self._prefill_selected_tokens_by_layer.pop(request_id, None)
        self._selected_token_indices_by_layer.pop(request_id, None)
        self._step_trace.pop(request_id, None)
        self._prefill_token_count.pop(request_id, None)
        self._first_select_probe_done.discard(request_id)

    def free_prefill_blocks_after_save(self, request_id: str) -> int:
        if not _SPARSE_FREE_PREFILL_AFTER_SAVE:
            return 0
        if request_id in self._prefill_offloaded:
            return 0
        blocks = (
            self._prefill_blocks.get(request_id)
            or self.req_to_blocks.get(request_id, [])
        )
        to_free = [b for b in blocks if not b.is_null]
        if to_free:
            self.block_pool.free_blocks(reversed(to_free))
        self._prefill_offloaded.add(request_id)
        self._prefill_blocks.pop(request_id, None)
        req_blocks = self.req_to_blocks.get(request_id)
        if req_blocks is not None:
            req_blocks.clear()
        logger.info(                                                                                  
            "[sparse-offload] freed %d prefill blocks for req=%s; "                                   
            "block_pool free=%d/%d",                                                                  
            len(to_free),                                                                      
            request_id,                                                                               
            self.block_pool.get_num_free_blocks(),                                                    
            self.block_pool.num_gpu_blocks,                                                           
       )                                  
        return len(to_free)

    def is_prefill_offloaded(self, request_id: str) -> bool:
        return request_id in self._prefill_offloaded

    def _scratch_block_count(self) -> int:
        # Scratch holds exactly what LMCache's clustered transfer loads:
        # ``effective_max_selected_tokens`` (= budget) dynamic cluster tokens
        # at slots [0, budget).  Static head/tail are NOT separately loaded
        # (budget is "steady + retrieve").  Sizing the scratch to budget --
        # which is block-aligned for the usual block sizes -- makes the
        # resident decode region start on a block boundary in PROPER decode
        # blocks (right after the loaded scratch).  Over-sizing to
        # static+budget leaves slack scratch blocks that the decode region
        # then "borrows", mis-aligning the manager's decode-block accounting
        # against the runner's decode slot_mapping and corrupting decode KV.
        spec = self._spec
        return cdiv(spec.effective_max_selected_tokens, self.block_size)

    def get_scratch_blocks(self, request_id: str) -> list[KVCacheBlock]:
        return self._scratch_blocks.get(request_id, [])

    def _estimate_decode_tokens_this_step(
        self, request_id: str, num_tokens_main_model: int
    ) -> int:
        """Estimate decode tokens scheduled for this request in current step."""
        prev = self._last_num_tokens_main_model.get(request_id)
        if prev is None:
            return 1
        delta = num_tokens_main_model - prev
        # Keep at least one token to preserve decode fast-path assumptions.
        return max(1, delta)

    def new_step_starts(self) -> None:
        if self._debug_state_transitions and self._step_trace:
            for req_id in sorted(self._step_trace):
                trace = self._step_trace[req_id]
                logger.info(
                    "Sparse step summary: step=%d req_id=%s phase=%r "
                    "prefill_topk_ready=%r used_prefill_cache=%r "
                    "selected_count=%r total_blocks=%r buffered=%r/%r "
                    "added_clusters=%r total_clusters=%r selected_preview=%r",
                    self._current_step,
                    req_id,
                    trace.get("phase"),
                    trace.get("prefill_topk_ready"),
                    trace.get("used_prefill_cache"),
                    trace.get("selected_count"),
                    trace.get("total_blocks"),
                    trace.get("buffered_blocks"),
                    trace.get("threshold"),
                    trace.get("added_clusters"),
                    trace.get("total_clusters"),
                    trace.get("selected_preview"),
                )
            self._step_trace.clear()
        self._current_step += 1

    # ------------------------------------------------------------------
    # Public sparse-management API
    # Called by KVCacheManager which is called by the Scheduler.
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_block_features_map(
        block_features: np.ndarray | dict[str, np.ndarray],
        block_value_features: np.ndarray | dict[str, np.ndarray] | None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
        if isinstance(block_features, np.ndarray):
            u0 = sparse_kv_unit_key(SPARSE_LEGACY_FLAT_LAYER, 0)
            bf = {u0: block_features}
            bv = (
                None
                if block_value_features is None
                else {u0: block_value_features}
            )
            return bf, bv
        if block_value_features is not None and not isinstance(
            block_value_features, dict
        ):
            raise ValueError(
                "sparse block_value_features must be a dict[str, ndarray] "
                "when block_features is per-layer dict"
            )
        bf = _normalize_kv_feature_map(dict(block_features))
        bv = (
            None
            if block_value_features is None
            else _normalize_kv_feature_map(dict(block_value_features))
        )
        return bf, bv

    def _layer_map_for_request(
        self, request_id: str
    ) -> dict[str, _SparseLayerIndexState]:
        return self._layer_states.setdefault(request_id, {})

    def _token_mode(self) -> bool:
        return self._spec.cluster_granularity == "token"

    def delegates_token_selection_to_runner(self) -> bool:
        """Whether decode token selection is owned by the GPU runner.

        In token mode with compact KV gather, the runner builds per-query-head
        token selections directly from GPU-resident Q vectors and centroids.
        The scheduler side should therefore avoid a second CPU TopK/steady-zone
        pass and keep allocation metadata full-attention-like.
        """
        return self._token_mode() and bool(self._spec.use_compact_kv_gather)

    def _num_index_units(self, request_id: str) -> int:
        """Number of clustered rows (blocks or tokens) for this request."""
        ls = self._layer_states.get(request_id, {})
        if not ls:
            return 0
        base_units = len(next(iter(ls.values())).all_block_features)
        if not self._token_mode():
            return base_units
        # Token mode must expose decode history tokens too (not only prefill
        # indexed rows), otherwise tail/static windows miss the newest decode
        # tokens and selection lags behind by recent tokens.
        p_count = int(self._prefill_token_count.get(request_id, base_units))
        bsz = int(self.block_size)
        prefill_blocks = self._prefill_blocks.get(request_id, [])
        n_prefill_blocks = cdiv(p_count, bsz)
        finalized_decode_blocks = max(0, len(prefill_blocks) - n_prefill_blocks)
        decode_hist_tokens = finalized_decode_blocks * bsz
        decode_active_tokens = int(self._decode_block_fill.get(request_id, 0))
        return max(base_units, p_count + decode_hist_tokens + decode_active_tokens)

    @staticmethod
    def _global_token_to_logical_block(
        global_tok: int, prefill_tokens: int, block_size: int
    ) -> int:
        if global_tok < prefill_tokens:
            return global_tok // block_size
        d = global_tok - prefill_tokens
        n_pb = cdiv(prefill_tokens, block_size)
        return n_pb + (d // block_size)

    @staticmethod
    def _logical_block_first_global(
        logical_block: int, prefill_tokens: int, block_size: int
    ) -> int:
        """Smallest global token index stored in chronological logical block ``logical_block``."""
        n_pb = cdiv(prefill_tokens, block_size)
        if logical_block < n_pb:
            return logical_block * block_size
        d_block = logical_block - n_pb
        return prefill_tokens + d_block * block_size

    def _tokens_to_history_logical_blocks(
        self, request_id: str, token_indices: Iterable[int]
    ) -> list[int]:
        """Map global token indices to historical logical block ids (decode slot excluded).

        Vectorised to eliminate the per-token Python dispatch: called once per
        ``(select, query-head)`` with a budget-sized token set (~128-512 items)
        across 784 query heads per decode step – the original ``for g in ...``
        loop dominated ``sparse_update_query_vectors_ms``.

        Fast path: when ``token_indices`` is already a numpy ndarray (the
        common case from the batched select path), use ``np.asarray`` which
        is a bulk view / dtype-copy – no per-element Python iteration.
        ``np.fromiter`` is reserved for list/set inputs where bulk copy is
        not possible.
        """
        p_count = self._prefill_token_count.get(request_id)
        if p_count is None:
            return []
        bsz = int(self.block_size)
        pb = self._prefill_blocks.get(request_id, [])
        if isinstance(token_indices, np.ndarray):
            toks = token_indices.astype(np.int64, copy=False)
        else:
            # Iterable fallback: np.fromiter goes per-element but this only
            # fires for list/set/range callers, which are small (<= budget).
            toks = np.fromiter(token_indices, dtype=np.int64)
        if toks.size == 0:
            return []
        # Equivalent to ``_global_token_to_logical_block`` per element:
        #   if g < p_count:  lb = g // bsz
        #   else:            lb = cdiv(p_count, bsz) + (g - p_count) // bsz
        n_pb = (p_count + bsz - 1) // bsz  # cdiv
        in_prompt = toks < p_count
        lb = np.where(
            in_prompt,
            toks // bsz,
            n_pb + (toks - p_count) // bsz,
        )
        if pb:
            n_hist = len(pb)
            lb = lb[lb < n_hist]
        if lb.size == 0:
            return []
        return np.unique(lb).tolist()

    def _steady_block_set(self, request_id: str) -> set[int]:
        """Logical history blocks that intersect the steady (sink + local) token zones."""
        n_units = self._num_index_units(request_id)
        if n_units == 0:
            return set()
        bsz = self.block_size
        if self._token_mode():
            p_count = self._prefill_token_count.get(request_id, n_units)
            steady_toks: set[int] = set()
            head = min(self._spec.static_pattern_start, n_units)
            steady_toks.update(range(head))
            tail0 = max(0, n_units - self._spec.static_pattern_end)
            steady_toks.update(range(tail0, n_units))
            out: set[int] = set()
            for t in steady_toks:
                out.add(self._global_token_to_logical_block(t, p_count, bsz))
            return out
        n_sink = self._spec.n_sink_blocks
        n_recent = self._spec.n_recent_blocks
        sink_blocks = set(range(min(n_sink, n_units)))
        recent_blocks = set(range(max(0, n_units - n_recent), n_units))
        return sink_blocks | recent_blocks

    def _coerce_query_by_qh(
        self,
        request_id: str,
        query_vector: np.ndarray | dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Map scheduler / runner query payloads to ``layer##qh{i}`` vectors."""
        ls = self._layer_states.get(request_id, {})
        if isinstance(query_vector, dict):
            out: dict[str, np.ndarray] = {}
            for k, v in query_vector.items():
                if SPARSE_QH_KEY in k:
                    out[k] = np.asarray(v, dtype=np.float32)
                else:
                    out[sparse_qh_unit_key(k, 0)] = np.asarray(v, dtype=np.float32)
            return out
        q = np.asarray(query_vector, dtype=np.float32)
        if not ls:
            return {sparse_qh_unit_key(SPARSE_LEGACY_FLAT_LAYER, 0): q}
        layers: set[str] = set()
        for k in ls:
            pk = parse_sparse_kv_key(k)
            if pk is not None:
                layers.add(pk[0])
            elif SPARSE_QH_KEY not in k and SPARSE_KV_KEY not in k:
                layers.add(k)
        if not layers:
            return {sparse_qh_unit_key(SPARSE_LEGACY_FLAT_LAYER, 0): q}
        return {sparse_qh_unit_key(layer, 0): q for layer in sorted(layers)}

    @staticmethod
    def _sorted_kv_indices_for_layer(
        layer: str, ls: dict[str, _SparseLayerIndexState]
    ) -> list[int]:
        idxs: list[int] = []
        for k in ls:
            pk = parse_sparse_kv_key(k)
            if pk is not None and pk[0] == layer:
                idxs.append(pk[1])
        return sorted(idxs)

    @staticmethod
    def _qh_to_kv_index(qh_idx: int, num_q: int, num_kv: int) -> int:
        if num_kv <= 1:
            return 0
        q_per_kv = max(1, num_q // num_kv)
        return min(qh_idx // q_per_kv, num_kv - 1)

    def _retrieve_zone_one_layer(
        self,
        st: _SparseLayerIndexState,
        q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select retrieve-zone block indices and their per-block cluster scores.

        Returns ``(retrieve_idx, per_block_scores)``:
            - ``retrieve_idx``   int64 ndarray of ascending block indices
              (equivalent to ``sorted(set)`` of the old set return).
            - ``per_block_scores`` float32 ndarray, same length, each entry is
              ``scores_c[b2c[bidx]]`` for the corresponding ``bidx``.

        Downstream (``_select_one_layer_topk_*``) consumes these directly as
        ndarrays – avoiding the old ``set[int]`` / ``dict[int, float]``
        materialisation that forced per-element Python boxing at ~19 M
        elements/decode (784 q-heads × 24 k tokens).
        """
        centres = st.cluster_centres
        b2c = st.block_to_cluster
        empty_idx = np.empty((0,), dtype=np.int64)
        empty_scores = np.empty((0,), dtype=np.float32)
        if len(centres) == 0 or b2c.size == 0:
            return empty_idx, empty_scores
        q = np.asarray(q, dtype=np.float32)
        d = int(q.shape[-1])
        scores_c = (q @ centres.T) / np.sqrt(max(d, 1))
        nprobe = min(self._spec.nprobe, len(centres))
        top_cluster_ids_np = np.argpartition(scores_c, -nprobe)[-nprobe:]
        mask = np.isin(b2c, top_cluster_ids_np, assume_unique=False)
        retrieve_idx = np.nonzero(mask)[0].astype(np.int64, copy=False)
        if retrieve_idx.size == 0:
            return empty_idx, empty_scores
        per_block_scores = scores_c[b2c[retrieve_idx]].astype(
            np.float32, copy=False
        )
        return retrieve_idx, per_block_scores

    @staticmethod
    def _union_sorted_block_indices(
        by_layer: dict[str, list[int]],
    ) -> list[int]:
        return sorted({b for blocks in by_layer.values() for b in blocks})

    def _select_one_layer_topk_blocks(
        self,
        total_blocks: int,
        steady_np: np.ndarray,
        steady_set: set[int],
        st: _SparseLayerIndexState,
        q: np.ndarray,
        budget: int,
    ) -> tuple[list[int], list[int]]:
        """
        Steady zone + retrieve zone for one layer, capped by ``budget`` blocks.

        ``steady_np`` is the caller-hoisted sorted ndarray form of
        ``steady_set`` (same contents); both are accepted so that per-layer
        ndarray ops can skip the set→ndarray conversion that would otherwise
        fire per q-head (784× per decode step).

        Returns:
            (sorted full selection, sorted retrieve-only logical block indices).
        """

        if len(st.cluster_centres) == 0 or st.block_to_cluster.size == 0:
            fallback = np.arange(
                max(0, total_blocks - budget), total_blocks, dtype=np.int64
            )
            combined = np.union1d(fallback, steady_np)[:budget]
            sel_list = combined.tolist()
            retr_np = np.setdiff1d(combined, steady_np, assume_unique=True)
            return sel_list, retr_np.tolist()

        retr_np, scores_np = self._retrieve_zone_one_layer(st, q)
        combined = np.union1d(retr_np, steady_np)
        if combined.size > budget:
            # Drop steady members from retrieve, then keep top (budget -
            # |steady|) retrieve entries by score.  This matches the old
            # semantics: steady is always included, retrieve members
            # compete for the remaining slots.
            is_steady_in_retr = np.isin(retr_np, steady_np, assume_unique=True)
            non_steady_mask = ~is_steady_in_retr
            non_steady = retr_np[non_steady_mask]
            ns_scores = scores_np[non_steady_mask]
            cap = max(0, budget - int(steady_np.size))
            if cap > 0 and non_steady.size > cap:
                # Stable argsort on descending scores: deterministic tie-break
                # (smaller block index wins on equal score) – a valid refinement
                # of the original ``sorted(retr - steady, key=score, reverse=True)``
                # which iterated a Python set (non-deterministic order).
                order = np.argsort(-ns_scores, kind="stable")[:cap]
                top_non_steady = non_steady[order]
            else:
                top_non_steady = non_steady[:cap]
            combined = np.union1d(steady_np, top_non_steady)
        retr_only = np.setdiff1d(retr_np, steady_np, assume_unique=True)
        return combined.tolist(), retr_only.tolist()

    def _select_one_layer_topk_tokens(
        self,
        request_id: str,
        total_tokens: int,
        steady_np: np.ndarray,
        steady_tokens: set[int],
        st: _SparseLayerIndexState,
        q: np.ndarray,
        budget: int,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Token-granularity Top-K.

        ``steady_np`` is the caller-hoisted sorted ndarray form of
        ``steady_tokens`` (same contents); both are accepted to avoid
        re-building either per q-head iteration.

        Returns:
            (selected logical blocks, retrieve-only blocks, sorted global token ids).
        """

        if len(st.cluster_centres) == 0 or st.block_to_cluster.size == 0:
            fallback = np.arange(
                max(0, total_tokens - budget), total_tokens, dtype=np.int64
            )
            combined_t = np.union1d(fallback, steady_np)[:budget]
            sel_t_list = combined_t.tolist()
            retr_minus_steady = np.setdiff1d(combined_t, steady_np,
                                             assume_unique=True)
            sel_bl = self._tokens_to_history_logical_blocks(
                request_id, combined_t
            )
            retr_bl = self._tokens_to_history_logical_blocks(
                request_id, retr_minus_steady
            )
            return sel_bl, retr_bl, sel_t_list

        retr_np, scores_np = self._retrieve_zone_one_layer(st, q)
        combined_t = np.union1d(retr_np, steady_np)
        if combined_t.size > budget:
            is_steady_in_retr = np.isin(retr_np, steady_np, assume_unique=True)
            non_steady_mask = ~is_steady_in_retr
            non_steady = retr_np[non_steady_mask]
            ns_scores = scores_np[non_steady_mask]
            cap = max(0, budget - int(steady_np.size))
            if cap > 0 and non_steady.size > cap:
                # Stable argsort on descending scores: deterministic tie-break
                # (smaller token index wins on equal score) – a valid refinement
                # of the original ``sorted(retr - steady, key=score, reverse=True)``
                # which iterated a Python set (non-deterministic order).
                order = np.argsort(-ns_scores, kind="stable")[:cap]
                top_non_steady = non_steady[order]
            else:
                top_non_steady = non_steady[:cap]
            combined_t = np.union1d(steady_np, top_non_steady)
        retr_t_np = np.setdiff1d(retr_np, steady_np, assume_unique=True)
        sel_bl = self._tokens_to_history_logical_blocks(request_id, combined_t)
        retr_bl = self._tokens_to_history_logical_blocks(request_id, retr_t_np)
        return sel_bl, retr_bl, combined_t.tolist()

    def _select_kv_head_batched_topk_tokens(
        self,
        request_id: str,
        n_units: int,
        steady_np: np.ndarray,
        st: _SparseLayerIndexState,
        queries: np.ndarray,
        budget: int,
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
        """Batched equivalent of ``_select_one_layer_topk_tokens`` over G
        query heads that share the same layer state ``st``.

        ``queries`` is a stacked ``[G, D]`` float32 array.  Batching replaces
        G independent ``q @ centres.T`` matmuls, G ``argpartition`` calls, and
        G ``np.isin`` scans with a single matmul, one batched argpartition,
        and one fancy-index into a ``[G, N]`` mask – same arithmetic, far
        less Python / numpy dispatch overhead.

        Returns ``(sel_bl_list, retr_bl_list, sel_t_list)``, each length G.
        """
        G = int(queries.shape[0])
        centres = st.cluster_centres
        b2c = st.block_to_cluster

        # Fallback: no cluster info → every q head shares the same tail window.
        if len(centres) == 0 or b2c.size == 0:
            fallback = np.arange(
                max(0, n_units - budget), n_units, dtype=np.int64
            )
            combined_t = np.union1d(fallback, steady_np)[:budget]
            sel_t_list_one = combined_t.tolist()
            retr_minus_steady = np.setdiff1d(
                combined_t, steady_np, assume_unique=True
            )
            sel_bl = self._tokens_to_history_logical_blocks(
                request_id, combined_t
            )
            retr_bl = self._tokens_to_history_logical_blocks(
                request_id, retr_minus_steady
            )
            return (
                [sel_bl] * G,
                [retr_bl] * G,
                [sel_t_list_one] * G,
            )

        d = int(queries.shape[-1])
        # One matmul for all G heads: [G, D] @ [D, n_k] -> [G, n_k].
        scores_all = (queries @ centres.T) / np.sqrt(max(d, 1))
        n_k = int(centres.shape[0])
        nprobe = min(self._spec.nprobe, n_k)

        # Batched top-nprobe per row.  ``argpartition`` picks the top nprobe
        # unsorted per-row – we only need membership to build the cluster
        # mask, so sorted order is irrelevant here.
        top_ids_all = np.argpartition(
            scores_all, -nprobe, axis=-1
        )[:, -nprobe:]  # [G, nprobe]

        # Build [G, n_k] cluster mask via row-scatter.  ``put_along_axis``
        # writes True at ``top_ids_all`` positions for each row.
        cluster_mask = np.zeros((G, n_k), dtype=bool)
        np.put_along_axis(cluster_mask, top_ids_all, True, axis=-1)

        # retrieve_mask_all[g, t] = cluster_mask[g, b2c[t]] – one fancy-index
        # on the column axis yields the full [G, N] retrieve mask.  Replaces
        # the per-q ``np.isin(b2c, top_cluster_ids_g)`` scan.
        retrieve_mask_all = cluster_mask[:, b2c]  # [G, N] bool

        sel_bl_list: list[list[int]] = []
        retr_bl_list: list[list[int]] = []
        sel_t_list: list[list[int]] = []
        steady_size = int(steady_np.size)
        cap_base = max(0, budget - steady_size)

        # Precompute request-level bookkeeping once (avoids ``self.*.get(...)``
        # dict lookups inside ``_tokens_to_history_logical_blocks`` × 2 × G =
        # 14 calls per group, ~1.5k per step).
        p_count = self._prefill_token_count.get(request_id)
        bsz = int(self.block_size)
        n_pb = (int(p_count) + bsz - 1) // bsz if p_count is not None else 0
        pb = self._prefill_blocks.get(request_id, [])
        n_hist = len(pb) if pb else None

        def _to_logical(toks: np.ndarray) -> list[int]:
            """Inline, hoisted-state version of ``_tokens_to_history_logical_blocks``."""
            if p_count is None or toks.size == 0:
                return []
            in_prompt = toks < p_count
            lb = np.where(
                in_prompt, toks // bsz, n_pb + (toks - p_count) // bsz
            )
            if n_hist is not None:
                lb = lb[lb < n_hist]
            if lb.size == 0:
                return []
            return np.unique(lb).tolist()

        for g in range(G):
            retr_np = np.nonzero(retrieve_mask_all[g])[0].astype(
                np.int64, copy=False
            )
            if retr_np.size == 0:
                combined_t = steady_np
                retr_t_np = np.empty((0,), dtype=np.int64)
            else:
                # Searchsorted-based membership: steady_np is small and sorted
                # so this is O(R log S) vs ``np.isin``'s sort-based
                # O((R+S) log (R+S)).  ``retr_np`` is already ascending
                # (``np.nonzero`` output), so ``non_steady`` stays sorted.
                if steady_size == 0:
                    is_steady_in_retr = np.zeros(retr_np.shape, dtype=bool)
                    overlap = 0
                else:
                    idx = np.searchsorted(steady_np, retr_np)
                    idx_clamped = np.minimum(idx, steady_size - 1)
                    is_steady_in_retr = steady_np[idx_clamped] == retr_np
                    overlap = int(is_steady_in_retr.sum())
                non_steady_mask = ~is_steady_in_retr
                combined_size = retr_np.size + steady_size - overlap

                if combined_size > budget:
                    non_steady = retr_np[non_steady_mask]
                    # Per-block score = score of its cluster under this query.
                    ns_scores = scores_all[g, b2c[non_steady]].astype(
                        np.float32, copy=False
                    )
                    if cap_base > 0 and non_steady.size > cap_base:
                        order = np.argsort(
                            -ns_scores, kind="stable"
                        )[:cap_base]
                        top_non_steady = non_steady[order]
                    else:
                        top_non_steady = non_steady[:cap_base]
                    # ``top_non_steady`` is disjoint from ``steady_np`` (the
                    # non-steady mask guarantees it) so simple concat + sort
                    # gives the final sorted union; the combined length is at
                    # most ``budget`` so the sort is cheap.
                    combined_t = np.sort(
                        np.concatenate([steady_np, top_non_steady])
                    )
                else:
                    # No cap needed: merge sorted-disjoint ``steady_np`` with
                    # sorted ``non_steady``.  Sort on concat of two sorted
                    # disjoint arrays with ``combined_size <= budget`` is
                    # cheap (bounded by budget).
                    non_steady = retr_np[non_steady_mask]
                    combined_t = np.sort(
                        np.concatenate([steady_np, non_steady])
                    )
                # ``retr_t_np = retr_np \ steady_np`` – already sorted.
                retr_t_np = retr_np[non_steady_mask]
            sel_bl_list.append(_to_logical(combined_t))
            retr_bl_list.append(_to_logical(retr_t_np))
            sel_t_list.append(combined_t.tolist())

        return sel_bl_list, retr_bl_list, sel_t_list

    def _per_layer_fresh_topk(
        self,
        request_id: str,
        query_by_qh: dict[str, np.ndarray],
        budget: int,
    ) -> tuple[
        dict[str, list[int]], dict[str, list[int]], dict[str, list[int]] | None
    ]:
        """
        Independent TopK + budget per **query head** (``layer##qh{i}`` keys).

        Clustering state remains per **KV head** (``layer##kv{j}``); each query
        head maps to one KV head via GQA grouping.
        """
        n_units = self._num_index_units(request_id)
        if n_units == 0:
            return {}, {}, None

        ls = self._layer_states.get(request_id, {})
        if not ls:
            return {}, {}, None

        layers_from_qh: dict[str, list[tuple[int, str]]] = {}
        for qk, _q in query_by_qh.items():
            parsed = parse_sparse_qh_key(qk)
            if parsed is None:
                continue
            layer_name, qh_idx = parsed
            layers_from_qh.setdefault(layer_name, []).append((qh_idx, qk))
        for lst in layers_from_qh.values():
            lst.sort(key=lambda x: x[0])

        if self._token_mode():
            head = min(self._spec.static_pattern_start, n_units)
            steady_tokens = set(range(head))
            steady_tokens.update(
                range(max(0, n_units - self._spec.static_pattern_end), n_units)
            )
            # Hoist the set→ndarray conversion out of the per-q_head loop so
            # all 784 calls into ``_select_one_layer_topk_tokens`` share it.
            steady_tokens_np = np.fromiter(
                steady_tokens, dtype=np.int64, count=len(steady_tokens)
            )
            steady_tokens_np.sort()
            if all(len(st.cluster_centres) == 0 for st in ls.values()):
                fallback = set(range(max(0, n_units - budget), n_units))
                sel_t = sorted(fallback | steady_tokens)[:budget]
                retr_t = sorted(set(sel_t) - steady_tokens)
                sel_bl = self._tokens_to_history_logical_blocks(request_id, sel_t)
                retr_bl = self._tokens_to_history_logical_blocks(request_id, retr_t)
                tok = {qk: list(sel_t) for qk in query_by_qh}
                return (
                    {qk: sel_bl for qk in query_by_qh},
                    {qk: retr_bl for qk in query_by_qh},
                    tok,
                )

            selected_by_qh: dict[str, list[int]] = {}
            retrieve_by_qh: dict[str, list[int]] = {}
            tokens_by_qh: dict[str, list[int]] = {}
            for layer_name, qh_list in layers_from_qh.items():
                kv_sorted = self._sorted_kv_indices_for_layer(layer_name, ls)
                num_kv = len(kv_sorted)
                if num_kv == 0:
                    continue
                num_q = max(qh for qh, _ in qh_list) + 1 if qh_list else 1
                # Group q_heads by the kv_head they map to – all q_heads in
                # the same group share the same ``st`` (centres + b2c) so we
                # can do one batched matmul/mask per group instead of G
                # independent per-q ops.
                group_qks: dict[int, list[str]] = {}
                group_qs: dict[int, list[np.ndarray]] = {}
                for qh_idx, qk in qh_list:
                    q = query_by_qh.get(qk)
                    if q is None:
                        continue
                    kv_slot = self._qh_to_kv_index(qh_idx, num_q, num_kv)
                    kv_actual = kv_sorted[kv_slot]
                    group_qks.setdefault(kv_actual, []).append(qk)
                    group_qs.setdefault(kv_actual, []).append(q)
                for kv_actual, qks_in_group in group_qks.items():
                    st_key = sparse_kv_unit_key(layer_name, kv_actual)
                    st = ls.get(st_key)
                    if st is None:
                        continue
                    qs_in_group = group_qs[kv_actual]
                    Q = np.stack(qs_in_group, axis=0).astype(
                        np.float32, copy=False
                    )
                    sel_bl_list, retr_bl_list, sel_t_list = (
                        self._select_kv_head_batched_topk_tokens(
                            request_id, n_units, steady_tokens_np,
                            st, Q, budget,
                        )
                    )
                    for qk, sel_bl, retr_bl, sel_t in zip(
                        qks_in_group, sel_bl_list, retr_bl_list, sel_t_list,
                        strict=True,
                    ):
                        selected_by_qh[qk] = sel_bl
                        retrieve_by_qh[qk] = retr_bl
                        tokens_by_qh[qk] = sel_t
            if self._sparse_probe_info_enabled and request_id not in self._first_select_probe_done:
                qh_items = list(tokens_by_qh.items())
                n_heads = len(qh_items)
                if n_heads > 0:
                    tail_start = max(0, n_units - 256)
                    tail_cov_heads = sum(
                        1 for _, toks in qh_items if any(t >= tail_start for t in toks)
                    )
                    tok_lens = [len(toks) for _, toks in qh_items]
                    logger.info(
                        "[SparseProbe:first_recall] req_id=%s token_mode=1 "
                        "index_units=%d qh_selected=%d tail256_heads=%d "
                        "tok_per_head(min/avg/max)=%d/%.1f/%d",
                        request_id,
                        int(n_units),
                        n_heads,
                        tail_cov_heads,
                        int(min(tok_lens)),
                        float(sum(tok_lens)) / float(max(1, n_heads)),
                        int(max(tok_lens)),
                    )
            return selected_by_qh, retrieve_by_qh, tokens_by_qh

        n_sink = self._spec.n_sink_blocks
        n_recent = self._spec.n_recent_blocks
        sink_set = set(range(min(n_sink, n_units)))
        recent_set = set(range(max(0, n_units - n_recent), n_units))
        steady_set = sink_set | recent_set
        steady_set_np = np.fromiter(
            steady_set, dtype=np.int64, count=len(steady_set)
        )
        steady_set_np.sort()

        if all(len(st.cluster_centres) == 0 for st in ls.values()):
            fallback = set(range(max(0, n_units - budget), n_units))
            sel = sorted(fallback | steady_set)[:budget]
            retr = sorted(set(sel) - steady_set)
            return (
                {qk: list(sel) for qk in query_by_qh},
                {qk: list(retr) for qk in query_by_qh},
                None,
            )

        selected_by_qh: dict[str, list[int]] = {}
        retrieve_by_qh: dict[str, list[int]] = {}
        for layer_name, qh_list in layers_from_qh.items():
            kv_sorted = self._sorted_kv_indices_for_layer(layer_name, ls)
            num_kv = len(kv_sorted)
            if num_kv == 0:
                continue
            num_q = max(qh for qh, _ in qh_list) + 1 if qh_list else 1
            for qh_idx, qk in qh_list:
                q = query_by_qh.get(qk)
                if q is None:
                    continue
                kv_slot = self._qh_to_kv_index(qh_idx, num_q, num_kv)
                kv_actual = kv_sorted[kv_slot]
                st_key = sparse_kv_unit_key(layer_name, kv_actual)
                st = ls.get(st_key)
                if st is None:
                    continue
                sel, retr = self._select_one_layer_topk_blocks(
                    n_units, steady_set_np, steady_set, st, q, budget
                )
                selected_by_qh[qk] = sel
                retrieve_by_qh[qk] = retr
        return selected_by_qh, retrieve_by_qh, None

    def _apply_steady_and_cap_tokens_per_layer(
        self,
        request_id: str,
        cached_tokens: dict[str, list[int]],
        token_cap: int,
    ) -> dict[str, list[int]]:
        """Re-merge steady tokens and trim; returns per-layer sorted global token ids."""
        n_units = self._num_index_units(request_id)
        head = min(self._spec.static_pattern_start, n_units)
        steady_tokens = set(range(head))
        steady_tokens.update(
            range(max(0, n_units - self._spec.static_pattern_end), n_units)
        )

        out: dict[str, list[int]] = {}
        for layer_name, cached in cached_tokens.items():
            combined = set(cached) | steady_tokens
            if len(combined) <= token_cap:
                sel_t = sorted(combined)
            else:
                non_steady = sorted(combined - steady_tokens)
                bud = max(0, token_cap - len(steady_tokens))
                sel_t = sorted(steady_tokens | set(non_steady[:bud]))
            out[layer_name] = sel_t
        return out

    def get_chrono_phys_block_ids(self, request_id: str) -> list[int]:
        """Chronological physical KV block ids (prefill + finalized decode + active decode)."""
        out: list[int] = []
        for b in self._prefill_blocks.get(request_id, []):
            if not b.is_null:
                out.append(int(b.block_id))
        dec = self._decode_block.get(request_id)
        if dec is not None and not dec.is_null:
            out.append(int(dec.block_id))
        # Phase-transition safety: if decode bookkeeping has not materialized
        # (_prefill_blocks/_decode_block still empty) but req_to_blocks already
        # carries a valid sparse row, use that row as chronological fallback.
        if not out:
            row = self.req_to_blocks.get(request_id, [])
            for b in row:
                if not b.is_null:
                    out.append(int(b.block_id))
        return out

    def _apply_steady_and_cap_per_layer(
        self,
        request_id: str,
        cached_by_layer: dict[str, list[int]],
        block_cap: int,
    ) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        """Re-merge steady zone and trim using each layer's cached prefill selection."""
        steady_set = self._steady_block_set(request_id)

        selected_by_layer: dict[str, list[int]] = {}
        retrieve_by_layer: dict[str, list[int]] = {}
        for layer_name, cached in cached_by_layer.items():
            combined = set(cached) | steady_set
            if len(combined) <= block_cap:
                sel = sorted(combined)
            else:
                non_steady = sorted(combined - steady_set)
                budget = max(0, block_cap - len(steady_set))
                sel = sorted(steady_set | set(non_steady[:budget]))
            selected_by_layer[layer_name] = sel
            retrieve_by_layer[layer_name] = sorted(set(sel) - steady_set)
        return selected_by_layer, retrieve_by_layer

    def _indexing_one_layer(
        self,
        request_id: str,
        layer_name: str,
        block_features: np.ndarray,
        block_value_features: np.ndarray | None,
        precomputed: dict[str, np.ndarray] | None = None,
    ) -> int:
        num_blocks, d = block_features.shape
        feat = block_features.astype(np.float32)
        if precomputed is not None:
            centres = np.asarray(precomputed["cluster_centres"], dtype=np.float32)
            labels_arr = np.asarray(precomputed["block_to_cluster"], dtype=np.int32)
            sizes = np.asarray(precomputed["cluster_size"], dtype=np.int32)
            mean_key = np.asarray(precomputed["mean_key"], dtype=np.float32)
            if centres.ndim != 2 or centres.shape[1] != d:
                raise ValueError(
                    f"sparse indexing: bad cluster_centres shape for "
                    f"{layer_name!r}: {centres.shape} vs D={d}"
                )
            if mean_key.shape != (d,):
                raise ValueError(
                    f"sparse indexing: bad mean_key shape for {layer_name!r}"
                )
            if labels_arr.shape != (num_blocks,):
                raise ValueError(
                    f"sparse indexing: block_to_cluster length mismatch for "
                    f"{layer_name!r}: {labels_arr.shape[0]} vs num_blocks={num_blocks}"
                )
            n_k = centres.shape[0]
            if sizes.shape != (n_k,):
                raise ValueError(
                    f"sparse indexing: cluster_size length mismatch for "
                    f"{layer_name!r}: {sizes.shape[0]} vs K={n_k}"
                )
        else:
            mean_key = feat.mean(axis=0)
            centered = feat - mean_key
            k = min(self._spec.num_clusters, num_blocks)
            centres, labels_arr, sizes = _segment_kmeans(
                centered,
                n_clusters=k,
                n_segments=min(self._spec.n_segment, num_blocks),
            )
            centres = centres + mean_key
            n_k = len(centres)
        if block_value_features is not None:
            vfeat = block_value_features.astype(np.float32)
            # ``np.add.at`` is needed because multiple tokens share a cluster id;
            # plain ``value_sum[labels_arr] += vfeat`` would only keep the last
            # write per duplicate index.  When vfeat is all zeros (fallback
            # path below), the scatter-add reduces to zero-plus-zero, so skip
            # the scan entirely.
            value_sum = np.zeros((n_k, d), dtype=np.float32)
            np.add.at(value_sum, labels_arr, vfeat)
            all_value = vfeat
        else:
            # Skip the ``np.zeros_like(feat)`` allocation + the ``np.add.at``
            # scan over ``[N_prompt, D]`` zeros.  For token-mode prefill with
            # N=24640 × D=128 × 112 KV units this was ~1.4 GB of zero-fill +
            # ~350 M futile scatter-adds per indexing() call (several seconds).
            value_sum = np.zeros((n_k, d), dtype=np.float32)
            all_value = _empty_feats()

        # ``labels_arr`` is already int32/int64 from either the precomputed
        # path or _segment_kmeans; cast to a stable int32 so downstream
        # vectorisation (``np.isin(b2c, ...)``) doesn't fall into a mixed-dtype
        # slow path.
        b2c_arr = np.ascontiguousarray(labels_arr, dtype=np.int32)

        # Reserve decode headroom so the first rebalance after prefill doesn't
        # trigger an N→2N grow memcpy across every KV unit (~1.4 GB / step
        # across 112 heads for N=24640).  Size is capped so worst-case extra
        # memory is bounded even for very long prompts.
        reserve = min(max(num_blocks // 4, 1024), 4096)

        state = _SparseLayerIndexState(
            cluster_centres=centres,
            cluster_value_sum=value_sum,
            cluster_size=sizes,
            block_to_cluster=b2c_arr,
            # DO NOT pass ``all_block_features=feat`` / ``all_value_features=vfeat``
            # here: copying N_prompt × D × 4 B per KV unit (×112 heads) costs
            # ~1.4 GB of memcpy during prefill and has no downstream value
            # since the content of these two ndarrays is never read in
            # production code (only ``len(...)`` is consumed).  We allocate an
            # oversized empty buffer below and set the logical size so
            # ``len(...)`` matches the old behaviour and decode rebalances get
            # pre-reserved headroom without paying the copy.
            mean_key=mean_key,
        )
        state.reserve_capacity(reserve)
        state.set_prefill_block_feature_shape(num_blocks, d, reserve)
        # ``all_value_features`` storage is only meaningful when the caller
        # passed ``block_value_features``.  In the (common) ``None`` case we
        # leave it as the empty placeholder so ``append_value_feature`` takes
        # the reinit branch at its first call — same observable ``len`` growth
        # as before with zero cost at indexing time.
        if block_value_features is not None:
            state.set_prefill_value_feature_shape(num_blocks, d, reserve)
        self._layer_map_for_request(request_id)[layer_name] = state
        return n_k

    def indexing(
        self,
        request_id: str,
        block_features: np.ndarray | dict[str, np.ndarray],
        block_value_features: np.ndarray | dict[str, np.ndarray] | None = None,
        *,
        prefill_cluster_meta: dict[str, dict[str, np.ndarray]] | None = None,
    ) -> None:
        """
        Build per-layer cluster indices from prefill features.

        ``block_features`` may be a single matrix (legacy) or per-layer dict.
        Row count is ``num_blocks`` when ``cluster_granularity == "block"``, or
        ``num_prompt_tokens`` when ``cluster_granularity == "token"``.

        If ``prefill_cluster_meta`` provides per-layer centroids and labels
        (e.g. from GPU K-Means in the model runner), CPU K-Means is skipped
        for those layers.
        """
        bf_map, bv_map = self._normalize_block_features_map(
            block_features, block_value_features
        )
        if not bf_map:
            return
        n_per_layer = {name: arr.shape[0] for name, arr in bf_map.items()}
        if len(set(n_per_layer.values())) != 1:
            raise ValueError(
                f"sparse indexing: inconsistent num_blocks across layers "
                f"for req {request_id}: {n_per_layer!r}"
            )
        num_blocks = next(iter(n_per_layer.values()))
        if num_blocks == 0:
            logger.debug("sparse indexing: req %s has 0 blocks – skipped", request_id)
            return

        # Idempotence guard: if the request was already indexed with the same
        # row count, skip the rebuild.  Defense-in-depth for the case where
        # the worker accidentally re-emits ``sparse_block_features`` (see
        # ``_sparse_prefill_emitted`` guard in gpu_model_runner).  Rebuilding
        # from scratch costs O(num_blocks * num_layers) Python allocations
        # (~7s for a 24k-token prompt × 112 KV units).
        existing_ls = self._layer_states.get(request_id)
        if existing_ls:
            any_state = next(iter(existing_ls.values()), None)
            # ``len(all_block_features)`` works for both list[ndarray] and a
            # single 2D ndarray, so this guard is stable across the upcoming
            # storage vectorisation.
            if (
                any_state is not None
                and len(any_state.all_block_features) == num_blocks
            ):
                logger.debug(
                    "sparse indexing: req %s already indexed at %d units – "
                    "skipping idempotent rebuild",
                    request_id,
                    num_blocks,
                )
                return

        self._layer_states[request_id] = {}
        total_k = 0
        for layer_name, feat_arr in bf_map.items():
            v_layer = None if bv_map is None else bv_map.get(layer_name)
            pmeta = (
                None
                if prefill_cluster_meta is None
                else prefill_cluster_meta.get(layer_name)
            )
            total_k += self._indexing_one_layer(
                request_id, layer_name, feat_arr, v_layer, precomputed=pmeta
            )

        if self._token_mode():
            self._prefill_token_count[request_id] = int(num_blocks)
        else:
            self._prefill_token_count.pop(request_id, None)

        self._prefill_topk_ready.pop(request_id, None)
        self._prefill_selected_tokens_by_layer.pop(request_id, None)
        self._debug_log_state(
            request_id,
            "indexing_done",
            num_blocks=num_blocks,
            num_clusters=total_k,
            prefill_topk_ready=self._prefill_topk_ready.get(request_id, None),
        )

        logger.debug(
            "sparse indexing: req %s – %d blocks × %d layers → %d total clusters "
            "(%d segments/layer)",
            request_id,
            num_blocks,
            len(bf_map),
            total_k,
            self._spec.n_segment,
        )
        if self._sparse_probe_info_enabled:
            logger.info(
                "[SparseProbe] indexing req_id=%s layers=%d units=%d "
                "total_clusters=%d token_mode=%s",
                request_id,
                len(bf_map),
                num_blocks,
                total_k,
                self._token_mode(),
            )

    def select(
        self,
        request_id: str,
        query_vector: np.ndarray | dict[str, np.ndarray],
        num_blocks: int,
        *,
        ignore_prefill_topk_cache: bool = False,
    ) -> list[int]:
        """
        Choose logical block indices for the next decode step.

        ``num_blocks`` is the per-layer selection budget: logical **blocks** in
        block mode, **tokens** when ``cluster_granularity == "token"``.

        When ``ignore_prefill_topk_cache`` is True, any valid prefill TopK
        cache is skipped and selection uses ``query_vector`` (decode path
        aligned with prefill TopK).
        """
        n_units = self._num_index_units(request_id)
        steady_blocks = self._steady_block_set(request_id)
        block_cap = self._spec.max_blocks_for_sparse()

        used_prefill_cache = False
        if self._prefill_topk_ready.get(request_id, False) and (
            not ignore_prefill_topk_cache
        ):
            cached_bl = self._prefill_selected_by_layer.get(request_id, {})
            cached_tok = self._prefill_selected_tokens_by_layer.get(request_id, {})
            if cached_bl:
                if self._token_mode() and cached_tok:
                    sel_tok = self._apply_steady_and_cap_tokens_per_layer(
                        request_id,
                        cached_tok,
                        self._spec.sparse_selection_budget(),
                    )
                    sel_bl = {
                        ln: self._tokens_to_history_logical_blocks(request_id, tlist)
                        for ln, tlist in sel_tok.items()
                    }
                    head = min(self._spec.static_pattern_start, n_units)
                    steady_tok = set(range(head))
                    steady_tok.update(
                        range(
                            max(0, n_units - self._spec.static_pattern_end),
                            n_units,
                        )
                    )
                    retr_bl = {
                        ln: self._tokens_to_history_logical_blocks(
                            request_id, sorted(set(sel_tok[ln]) - steady_tok)
                        )
                        for ln in sel_tok
                    }
                    self._selected_token_indices_by_layer[request_id] = sel_tok
                elif self._token_mode():
                    # Token-sparse decode requires per-layer token indices for
                    # compact gather. If prefill cache has block-level entries
                    # but token-level cache is unavailable, recompute TopK from
                    # current query vectors instead of silently degrading to
                    # block-only selection.
                    q_by_qh = self._coerce_query_by_qh(request_id, query_vector)
                    sel_bl, retr_bl, tok_bl = self._per_layer_fresh_topk(
                        request_id, q_by_qh, num_blocks
                    )
                    if tok_bl is not None:
                        self._selected_token_indices_by_layer[request_id] = tok_bl
                    else:
                        self._selected_token_indices_by_layer.pop(request_id, None)
                else:
                    sel_bl, retr_bl = self._apply_steady_and_cap_per_layer(
                        request_id, cached_bl, block_cap
                    )
                    self._selected_token_indices_by_layer.pop(request_id, None)
                result = self._union_sorted_block_indices(sel_bl)
                self._selected_block_indices_by_layer[request_id] = sel_bl
                self._selected_retrieve_block_indices_by_layer[request_id] = retr_bl
                self._selected_block_indices[request_id] = result
                self._selected_retrieve_block_indices[request_id] = sorted(
                    set(result) - steady_blocks
                )
                used_prefill_cache = True
                self._debug_log_state(
                    request_id,
                    "select_done",
                    total_blocks=n_units,
                    num_blocks_cap=num_blocks,
                    selected_count=len(result),
                    selected_logical_blocks=len(result),
                    retrieve_logical_blocks=len(
                        self._selected_retrieve_block_indices.get(request_id, [])
                    ),
                    selected_preview=result[:16],
                    used_prefill_cache=used_prefill_cache,
                    prefill_topk_ready=self._prefill_topk_ready.get(request_id),
                )
                if self._sparse_probe_info_enabled:
                    logger.info(
                        "[SparseProbe] select req_id=%s used_prefill_cache=%s "
                        "union_selected=%d per_layer=%d "
                        "selected_logical_blocks=%d retrieve_logical_blocks=%d",
                        request_id,
                        used_prefill_cache,
                        len(result),
                        len(sel_bl),
                        len(result),
                        len(self._selected_retrieve_block_indices.get(request_id, [])),
                    )
                return result

        q_by_qh = self._coerce_query_by_qh(request_id, query_vector)
        sel_bl, retr_bl, tok_bl = self._per_layer_fresh_topk(
            request_id, q_by_qh, num_blocks
        )
        if tok_bl is not None:
            self._selected_token_indices_by_layer[request_id] = tok_bl
        else:
            self._selected_token_indices_by_layer.pop(request_id, None)
        result = self._union_sorted_block_indices(sel_bl)
        self._selected_block_indices_by_layer[request_id] = sel_bl
        self._selected_retrieve_block_indices_by_layer[request_id] = retr_bl
        self._selected_block_indices[request_id] = result
        self._selected_retrieve_block_indices[request_id] = sorted(
            set(result) - steady_blocks
        )
        self._debug_log_state(
            request_id,
            "select_done",
            total_blocks=n_units,
            num_blocks_cap=num_blocks,
            selected_count=len(result),
            selected_logical_blocks=len(result),
            retrieve_logical_blocks=len(
                self._selected_retrieve_block_indices.get(request_id, [])
            ),
            selected_preview=result[:16],
            used_prefill_cache=used_prefill_cache,
            prefill_topk_ready=self._prefill_topk_ready.get(request_id),
        )
        if self._sparse_probe_info_enabled:
            logger.info(
                "[SparseProbe] select req_id=%s used_prefill_cache=%s "
                "union_selected=%d per_layer=%d "
                "selected_logical_blocks=%d retrieve_logical_blocks=%d",
                request_id,
                used_prefill_cache,
                len(result),
                len(sel_bl),
                len(result),
                len(self._selected_retrieve_block_indices.get(request_id, [])),
            )
            if self._token_mode():
                tok_by_layer = self._selected_token_indices_by_layer.get(
                    request_id, {}
                )
                qh0_key: str | None = None
                for k in sorted(tok_by_layer.keys()):
                    if k.endswith("##qh0"):
                        qh0_key = k
                        break
                if qh0_key is not None:
                    toks0 = [int(t) for t in tok_by_layer.get(qh0_key, [])]
                    p_count = int(self._prefill_token_count.get(request_id, n_units))
                    bsz = int(self.block_size)
                    lb0 = [
                        int(self._global_token_to_logical_block(t, p_count, bsz))
                        for t in toks0
                    ]
                    lb_hist: dict[int, int] = {}
                    for lb in lb0:
                        lb_hist[lb] = lb_hist.get(lb, 0) + 1
                    lb_hist_head = sorted(lb_hist.items(), key=lambda kv: kv[0])[:6]
                    tail_n = int(self._spec.static_pattern_end)
                    tail_expected = list(
                        range(max(0, int(n_units) - tail_n), int(n_units))
                    )
                    tok_set = set(toks0)
                    tail_hit = [t for t in tail_expected if t in tok_set]
                    tail_miss = [t for t in tail_expected if t not in tok_set]
                    logger.info(
                        "[SparseProbe:select_tokens] req_id=%s qh0=%s "
                        "n_units=%d p_count=%d tok_all=%s lb_all=%s lb_hist=%s "
                        "tail_expected=%s tail_hit=%s tail_miss=%s",
                        request_id,
                        qh0_key,
                        int(n_units),
                        p_count,
                        toks0,
                        lb0,
                        lb_hist_head,
                        tail_expected,
                        tail_hit,
                        tail_miss,
                    )
            if request_id not in self._first_select_probe_done:
                self._first_select_probe_done.add(request_id)
                if self._token_mode():
                    tok_by_layer = self._selected_token_indices_by_layer.get(
                        request_id, {}
                    )
                    n_units_i = int(n_units)
                    tail_start = max(0, n_units_i - 256)
                    layer_items = sorted(tok_by_layer.items())
                    preview_layers = layer_items[:2]
                    if preview_layers:
                        layer_probe = "; ".join(
                            (
                                f"{ln}:sel={len(toks)} "
                                f"tail256={sum(1 for t in toks if t >= tail_start)} "
                                f"min={int(min(toks)) if toks else -1} "
                                f"max={int(max(toks)) if toks else -1}"
                            )
                            for ln, toks in preview_layers
                        )
                    else:
                        layer_probe = "no_token_selection"
                    logger.info(
                        "[SparseProbe:first_select] req_id=%s token_mode=1 "
                        "n_units=%d query_heads=%d selected_heads=%d "
                        "union_blocks=%d %s",
                        request_id,
                        n_units_i,
                        len(q_by_qh),
                        len(sel_bl),
                        len(result),
                        layer_probe,
                    )
                else:
                    logger.info(
                        "[SparseProbe:first_select] req_id=%s token_mode=0 "
                        "n_units=%d query_heads=%d selected_heads=%d "
                        "union_blocks=%d",
                        request_id,
                        int(n_units),
                        len(q_by_qh),
                        len(sel_bl),
                        len(result),
                    )
        return result

    def update_query_vector(
        self,
        request_id: str,
        query_vec: np.ndarray | dict[str, np.ndarray],
    ) -> None:
        """
        Store per-layer query vectors for the next ``select()`` call.

        ``query_vec`` may be a single ``[D]`` vector (broadcast to all indexed
        layers) or ``layer_name → [D]``.
        """
        q_by_qh = self._coerce_query_by_qh(request_id, query_vec)
        self._pending_query[request_id] = q_by_qh
        q0 = next(iter(q_by_qh.values()))
        self._debug_log_state(
            request_id,
            "query_updated",
            query_dim=int(q0.shape[-1]) if q0.ndim > 0 else 0,
            has_index=(request_id in self._layer_states),
            prefill_topk_ready=self._prefill_topk_ready.get(request_id, None),
        )

        if (
            request_id in self._layer_states
            and request_id not in self._prefill_topk_ready
            and self._spec.prefill_topk_query_window > 0
            and not self._spec.refresh_topk_each_decode
        ):
            self._compute_prefill_topk(request_id, q_by_qh)

    def get_pending_query(
        self, request_id: str
    ) -> dict[str, np.ndarray] | None:
        """Return pending per-layer query vectors, or ``None``."""
        return self._pending_query.get(request_id)

    def rebalance(
        self,
        request_id: str,
        new_block_feature: np.ndarray | dict[str, np.ndarray],
        new_block_value_feature: np.ndarray | dict[str, np.ndarray] | None = None,
    ) -> None:
        """
        Absorb a new decode block into **each** layer's sparse index.

        Pass a dict ``layer_name → [D]`` for layer-specific mean-K (and V), or
        a single vector to broadcast to every indexed layer (legacy).
        """
        ls = self._layer_states.get(request_id, {})
        if not ls:
            return

        if isinstance(new_block_feature, np.ndarray):
            feat_map = {k: new_block_feature for k in ls}
            v_map = (
                None
                if new_block_value_feature is None
                else {k: new_block_value_feature for k in ls}
            )
        else:
            feat_map = _normalize_kv_feature_map(dict(new_block_feature))
            v_map = (
                None
                if new_block_value_feature is None
                else _normalize_kv_feature_map(dict(new_block_value_feature))
            )

        max_buf = 0
        for unit_key, st in ls.items():
            if parse_sparse_kv_key(unit_key) is None:
                continue
            feat = feat_map.get(unit_key)
            if feat is None:
                continue
            vfeat = None if v_map is None else v_map.get(unit_key)
            max_buf = max(
                max_buf,
                self._rebalance_one_layer(request_id, unit_key, st, feat, vfeat),
            )

        threshold = (
            self._spec.update_threshold_tokens
            if self._token_mode()
            else self._spec.update_threshold_blocks
        )
        self._debug_log_state(
            request_id,
            "rebalance_buffered",
            buffered_blocks=max_buf,
            threshold=threshold,
            total_blocks=self._num_index_units(request_id),
        )
        if self._sparse_probe_info_enabled:
            logger.info(
                "[SparseProbe] rebalance req_id=%s layers=%d buffered=%d/%d",
                request_id,
                len(ls),
                max_buf,
                threshold,
            )

        for unit_key, st in ls.items():
            if parse_sparse_kv_key(unit_key) is None:
                continue
            if len(st.decode_block_buffer) >= threshold:
                self._dynamic_update_layer(request_id, unit_key, st)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebalance_one_layer(
        self,
        request_id: str,
        layer_name: str,
        st: _SparseLayerIndexState,
        new_block_feature: np.ndarray,
        new_block_value_feature: np.ndarray | None,
    ) -> int:
        feat = np.asarray(new_block_feature, dtype=np.float32)
        vfeat = (
            np.asarray(new_block_value_feature, dtype=np.float32)
            if new_block_value_feature is not None
            else np.zeros_like(feat)
        )
        # Amortised O(1) append into the preallocated history buffers (see
        # ``_grow_feats``/``_grow_b2c``).  Previous ``np.concatenate`` per
        # decode step cost ~1.4 GB/step of memcpy across 112 KV heads ×
        # ``[N_prompt, D]`` buffers – showed up as +570 ms/step in the
        # ``sparse_post_decode_rebalance_ms`` trace.
        st.append_block_feature(feat)
        st.append_value_feature(vfeat)
        st.decode_block_buffer.append(feat.copy())
        st.decode_value_buffer.append(vfeat.copy())

        centres = st.cluster_centres
        if len(centres) > 0:
            mean_key = st.mean_key if st.mean_key is not None else np.zeros_like(feat)
            centered_feat = feat - mean_key
            centred_c = centres - mean_key
            nearest = int(np.argmax(centered_feat @ centred_c.T))
        else:
            nearest = 0
        st.append_b2c(nearest)

        return len(st.decode_block_buffer)

    def _compute_prefill_topk(
        self,
        request_id: str,
        query_by_qh: dict[str, np.ndarray],
    ) -> None:
        cap = self._spec.sparse_selection_budget()
        sel_bl, _, tok_bl = self._per_layer_fresh_topk(request_id, query_by_qh, cap)
        self._prefill_selected_by_layer[request_id] = sel_bl
        if tok_bl is not None:
            self._prefill_selected_tokens_by_layer[request_id] = tok_bl
        else:
            self._prefill_selected_tokens_by_layer.pop(request_id, None)
        union_sel = self._union_sorted_block_indices(sel_bl)
        self._prefill_selected[request_id] = union_sel
        self._prefill_topk_ready[request_id] = True
        self._debug_log_state(
            request_id,
            "prefill_topk_cached",
            selected_count=len(union_sel),
            selected_preview=union_sel[:16],
            prefill_topk_ready=self._prefill_topk_ready.get(request_id),
        )
        logger.debug(
            "sparse prefill TopK: req %s – cached %d layers, %d union blocks",
            request_id,
            len(sel_bl),
            len(union_sel),
        )

    def _dynamic_update_layer(
        self,
        request_id: str,
        layer_name: str,
        st: _SparseLayerIndexState,
    ) -> None:
        buf = st.decode_block_buffer
        vbuf = st.decode_value_buffer
        if not buf:
            return

        feat = np.stack(buf, axis=0).astype(np.float32)
        vfeat = np.stack(vbuf, axis=0).astype(np.float32)
        m, d = feat.shape

        mean_key_new = feat.mean(axis=0)
        centered = feat - mean_key_new

        k_new = max(1, m // 16)
        k_new = (k_new // max(1, 32)) * 32
        k_new = max(k_new, 1)
        k_new = min(k_new, m)

        centres_new, labels_new, sizes_new = _segment_kmeans(
            centered, n_clusters=k_new, n_segments=1
        )
        centres_new = centres_new + mean_key_new

        n_k_new = len(centres_new)
        vsum_new = np.zeros((n_k_new, d), dtype=np.float32)
        np.add.at(vsum_new, labels_new, vfeat)

        n_existing = len(st.cluster_centres)
        st.cluster_centres = np.vstack([st.cluster_centres, centres_new])
        st.cluster_value_sum = np.vstack([st.cluster_value_sum, vsum_new])
        st.cluster_size = np.concatenate([st.cluster_size, sizes_new])

        # labels_new indexes the fresh sub-clusters; remap to the global
        # cluster-id space (after ``n_existing``) and write into block_to_cluster.
        # ``write_b2c_range`` handles both in-place overwrite and capacity
        # growth – in the common case (decode ``rebalance`` appends one row
        # per step so ``len(b2c) == n_total``) all ``m`` positions fall into
        # the in-place overwrite branch (O(m) scalar write, no memcpy of the
        # full buffer).
        n_total = len(st.all_block_features)
        new_slot = np.asarray(labels_new, dtype=np.int32) + n_existing
        st.write_b2c_range(n_total - m, new_slot)

        st.decode_block_buffer = []
        st.decode_value_buffer = []

        self._prefill_topk_ready[request_id] = False
        self._prefill_selected.pop(request_id, None)
        self._prefill_selected_by_layer.pop(request_id, None)
        self._prefill_selected_tokens_by_layer.pop(request_id, None)
        self._debug_log_state(
            request_id,
            "dynamic_update_done",
            buffered_blocks=m,
            added_clusters=n_k_new,
            total_clusters=len(st.cluster_centres),
            prefill_topk_ready=self._prefill_topk_ready.get(request_id),
            layer=layer_name,
        )

        logger.debug(
            "sparse dynamic update: req %s layer=%s – added %d clusters "
            "(total %d for layer), prefill TopK invalidated",
            request_id,
            layer_name,
            n_k_new,
            len(st.cluster_centres),
        )
