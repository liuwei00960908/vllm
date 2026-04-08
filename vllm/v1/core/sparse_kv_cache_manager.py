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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

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


@dataclass
class _SparseLayerIndexState:
    """Per-(request, attention-layer) clustering and feature buffers."""

    cluster_centres: np.ndarray
    cluster_value_sum: np.ndarray
    cluster_size: np.ndarray
    block_to_cluster: list[int]
    all_block_features: list[np.ndarray] = field(default_factory=list)
    all_value_features: list[np.ndarray] = field(default_factory=list)
    mean_key: np.ndarray | None = None
    decode_block_buffer: list[np.ndarray] = field(default_factory=list)
    decode_value_buffer: list[np.ndarray] = field(default_factory=list)


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
        # Probe logs visible only when VLLM_LOGGING_LEVEL=DEBUG.
        self._sparse_probe_info_enabled: bool = (
            os.getenv("VLLM_LOGGING_LEVEL", "").upper() == "DEBUG"
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
        if request_id not in self.num_cached_block:
            # Prefill: standard page-aligned calculation.
            return super().get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks,
                total_computed_tokens,
                num_tokens_main_model,
            )
        # Decode: allocate a new block only when there is no active decode
        # block, or when the active decode block does not have enough room.
        step_tokens = self._estimate_decode_tokens_this_step(
            request_id, num_tokens_main_model
        )
        cur_decode = self._decode_block.get(request_id)
        if cur_decode is None or cur_decode.is_null:
            return 1
        fill = self._decode_block_fill.get(request_id, 0)
        remaining = max(0, self.block_size - fill)
        return 1 if step_tokens > remaining else 0

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
        if request_id not in self.num_cached_block:
            # Prefill: allocate sequentially for all prompt tokens.
            return super().allocate_new_blocks(
                request_id, num_tokens, num_tokens_main_model
            )

        # ── Decode ────────────────────────────────────────────────────────
        # Key principle (Bug 1 fix): we NEVER free historical blocks during
        # decode so their KV data stays selectable. This includes prefill
        # blocks and finalized decode blocks from previous steps.
        req_blocks = self.req_to_blocks[request_id]

        # On the first decode call, snapshot all prefill blocks.
        if request_id not in self._prefill_blocks:
            self._prefill_blocks[request_id] = list(req_blocks)

        # Map selected logical indices → physical historical blocks.
        # The history list contains original prefill blocks followed by
        # finalized decode blocks in chronological order.
        prefill_blocks = self._prefill_blocks[request_id]
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

    def _num_index_units(self, request_id: str) -> int:
        """Number of clustered rows (blocks or tokens) for this request."""
        ls = self._layer_states.get(request_id, {})
        if not ls:
            return 0
        return len(next(iter(ls.values())).all_block_features)

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
        """Map global token indices to historical logical block ids (decode slot excluded)."""
        p_count = self._prefill_token_count.get(request_id)
        if p_count is None:
            return []
        bsz = self.block_size
        pb = self._prefill_blocks.get(request_id, [])
        out: set[int] = set()
        if pb:
            n_hist = len(pb)
            for g in token_indices:
                lb = self._global_token_to_logical_block(int(g), p_count, bsz)
                if lb < n_hist:
                    out.add(lb)
        else:
            for g in token_indices:
                out.add(self._global_token_to_logical_block(int(g), p_count, bsz))
        return sorted(out)

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
    ) -> tuple[set[int], dict[int, float]]:
        centres = st.cluster_centres
        b2c = st.block_to_cluster
        if len(centres) == 0 or not b2c:
            return set(), {}
        q = np.asarray(q, dtype=np.float32)
        d = int(q.shape[-1])
        scores_c = (q @ centres.T) / np.sqrt(max(d, 1))
        nprobe = min(self._spec.nprobe, len(centres))
        top_cluster_ids = set(
            int(c) for c in np.argpartition(scores_c, -nprobe)[-nprobe:]
        )
        retrieve: set[int] = set()
        block_scores: dict[int, float] = {}
        for bidx, cid in enumerate(b2c):
            if cid in top_cluster_ids:
                retrieve.add(bidx)
                block_scores[bidx] = float(scores_c[cid])
        return retrieve, block_scores

    @staticmethod
    def _union_sorted_block_indices(
        by_layer: dict[str, list[int]],
    ) -> list[int]:
        return sorted({b for blocks in by_layer.values() for b in blocks})

    def _select_one_layer_topk_blocks(
        self,
        total_blocks: int,
        steady_set: set[int],
        st: _SparseLayerIndexState,
        q: np.ndarray,
        budget: int,
    ) -> tuple[list[int], list[int]]:
        """
        Steady zone + retrieve zone for one layer, capped by ``budget`` blocks.

        Returns:
            (sorted full selection, sorted retrieve-only logical block indices).
        """
        if len(st.cluster_centres) == 0 or not st.block_to_cluster:
            fallback = set(range(max(0, total_blocks - budget), total_blocks))
            combined = fallback | steady_set
            sel = sorted(combined)[:budget]
            return sel, sorted(set(sel) - steady_set)

        retr, unit_scores = self._retrieve_zone_one_layer(st, q)
        combined = steady_set | retr
        if len(combined) > budget:
            non_steady = sorted(
                retr - steady_set,
                key=lambda b: unit_scores.get(b, float("-inf")),
                reverse=True,
            )
            cap = max(0, budget - len(steady_set))
            combined = steady_set | set(non_steady[:cap])
        sel = sorted(combined)
        return sel, sorted(retr - steady_set)

    def _select_one_layer_topk_tokens(
        self,
        request_id: str,
        total_tokens: int,
        steady_tokens: set[int],
        st: _SparseLayerIndexState,
        q: np.ndarray,
        budget: int,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Token-granularity Top-K.

        Returns:
            (selected logical blocks, retrieve-only blocks, sorted global token ids).
        """
        if len(st.cluster_centres) == 0 or not st.block_to_cluster:
            fallback = set(range(max(0, total_tokens - budget), total_tokens))
            combined_t = fallback | steady_tokens
            sel_t = sorted(combined_t)[:budget]
            sel_bl = self._tokens_to_history_logical_blocks(request_id, sel_t)
            retr_bl = self._tokens_to_history_logical_blocks(
                request_id, set(sel_t) - steady_tokens
            )
            return sel_bl, retr_bl, sel_t

        retr, unit_scores = self._retrieve_zone_one_layer(st, q)
        combined_t = steady_tokens | retr
        if len(combined_t) > budget:
            non_steady = sorted(
                retr - steady_tokens,
                key=lambda t: unit_scores.get(t, float("-inf")),
                reverse=True,
            )
            cap = max(0, budget - len(steady_tokens))
            combined_t = steady_tokens | set(non_steady[:cap])
        sel_t = sorted(combined_t)
        retr_t = sorted(retr - steady_tokens)
        sel_bl = self._tokens_to_history_logical_blocks(request_id, sel_t)
        retr_bl = self._tokens_to_history_logical_blocks(request_id, retr_t)
        return sel_bl, retr_bl, sel_t

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
                    sel_bl, retr_bl, sel_t = self._select_one_layer_topk_tokens(
                        request_id, n_units, steady_tokens, st, q, budget
                    )
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
                    n_units, steady_set, st, q, budget
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
        else:
            vfeat = np.zeros_like(feat)
        value_sum = np.zeros((n_k, d), dtype=np.float32)
        np.add.at(value_sum, labels_arr, vfeat)

        state = _SparseLayerIndexState(
            cluster_centres=centres,
            cluster_value_sum=value_sum,
            cluster_size=sizes,
            block_to_cluster=labels_arr.tolist(),
            all_block_features=[feat[i].copy() for i in range(num_blocks)],
            all_value_features=[vfeat[i].copy() for i in range(num_blocks)],
            mean_key=mean_key,
        )
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
                    logger.info(
                        "[SparseProbe:select_tokens] req_id=%s qh0=%s "
                        "n_units=%d p_count=%d tok_head=%s lb_head=%s lb_hist=%s",
                        request_id,
                        qh0_key,
                        int(n_units),
                        p_count,
                        toks0[:16],
                        lb0[:16],
                        lb_hist_head,
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
        st.all_block_features.append(feat.copy())
        st.all_value_features.append(vfeat.copy())
        st.decode_block_buffer.append(feat.copy())
        st.decode_value_buffer.append(vfeat.copy())

        centres = st.cluster_centres
        b2c = st.block_to_cluster
        if len(centres) > 0 and b2c is not None:
            mean_key = st.mean_key if st.mean_key is not None else np.zeros_like(feat)
            centered_feat = feat - mean_key
            centred_c = centres - mean_key
            nearest = int(np.argmax(centered_feat @ centred_c.T))
            b2c.append(nearest)
        else:
            b2c.append(0)

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

        b2c = st.block_to_cluster
        n_total = len(st.all_block_features)
        for i, lbl in enumerate(labels_new.tolist()):
            idx = n_total - m + i
            if idx < len(b2c):
                b2c[idx] = n_existing + lbl
            else:
                b2c.append(n_existing + lbl)

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
