# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SparseKVManager – RetroInfer-style block-level dynamic sparse attention.

Algorithm (matches RetroInfer, adapted for vLLM's paged block layout)
----------------------------------------------------------------------

Prefill
  1. Model runner extracts **block-level key features**:
       ``block_features[b]`` = mean K over all tokens in block b  → ``[D]``
     and the last-N query vectors for prefill TopK caching:
       ``query_vectors[req_id]`` = mean Q per KV-head group for last N prompt
     tokens → ``[N, D]`` (or ``[D]`` if N=1).

  2. ``indexing(req_id, block_features, mean_value_features)``
     - Mean-centers the block features:  ``feat -= feat.mean(axis=0)``
     - Runs **Segment K-Means** over the blocks (``n_segment`` position
       segments, ``num_clusters`` centroids distributed evenly).
     - Restores centering before storing centroids.
     - Also stores ``_cluster_value_sum`` (sum of mean-V per cluster).
       This is the hook for the Estimation Zone (currently unused).

  3. ``select(req_id, query_vec, ...)`` is called immediately after indexing
     (with the prefill query) to compute and **cache** the block selection.
     The cache is reused for all subsequent decode steps.

Decode
  - The cached selection from prefill is used directly → no per-token TopK.
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

from collections import defaultdict
from collections.abc import Sequence
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
    ``_cluster_centres``      [n_total_clusters, D] – uncentered centroids.
    ``_cluster_value_sum``    [n_total_clusters, D] – sum of block mean-V per cluster.
                               (stored for Estimation Zone; currently unused).
    ``_cluster_size``         [n_total_clusters] int – blocks per cluster.
    ``_block_to_cluster``     list[int] – logical block index → cluster id.
    ``_all_block_features``   list[np.ndarray] – accumulated block key features.
    ``_all_value_features``   list[np.ndarray] – accumulated block mean-V features.
    ``_mean_key``             [D] – global key mean subtracted before clustering.
    ``_pending_query``        [D] – last decoded/prefill query for next select().
    ``_selected_block_indices`` list[int] – blocks chosen for next decode step.
    ``_prefill_topk_ready``   bool – whether prefill TopK cache is valid.
    ``_decode_block_buffer``  list[np.ndarray] – features buffered for index update.
    ``_decode_value_buffer``  list[np.ndarray] – value features buffered for update.
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

        # ── per-request clustering state (all CPU numpy) ──────────────────
        self._cluster_centres: dict[str, np.ndarray] = {}
        # TODO(estimation-zone): _cluster_value_sum is stored but not yet
        # consumed by any attention computation.  See module docstring for
        # the steps needed to enable the Estimation Zone.
        self._cluster_value_sum: dict[str, np.ndarray] = {}
        self._cluster_size: dict[str, np.ndarray] = {}
        # logical block index → cluster id (grows as decode blocks arrive)
        self._block_to_cluster: dict[str, list[int]] = {}
        # all block key/value features accumulated since prefill
        self._all_block_features: dict[str, list[np.ndarray]] = {}
        self._all_value_features: dict[str, list[np.ndarray]] = {}
        # mean key used for centering (restored before storing centroids)
        self._mean_key: dict[str, np.ndarray] = {}

        # ── selection state ───────────────────────────────────────────────
        self._pending_query: dict[str, np.ndarray] = {}
        self._selected_block_indices: dict[str, list[int]] = {}
        # prefill TopK cache
        self._prefill_topk_ready: dict[str, bool] = {}
        self._prefill_selected: dict[str, list[int]] = {}

        # ── dynamic index update state ────────────────────────────────────
        # buffer for decode-phase blocks pending a fresh segment K-Means
        self._decode_block_buffer: dict[str, list[np.ndarray]] = {}
        self._decode_value_buffer: dict[str, list[np.ndarray]] = {}

        self._current_step: int = 0

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
        # Decode: net new blocks needed from the free pool after freeing the
        # old allocation and re-allocating for the new selection.
        #
        #   num_needed = |selected| + 1    (selected history + current token)
        #   num_old    = current len(req_to_blocks)   (will all be freed)
        #   net_new    = max(num_needed - num_old, 0)
        selected = self._selected_block_indices.get(request_id, [])
        num_needed = len(selected) + 1
        num_old = len(self.req_to_blocks.get(request_id, []))
        return max(num_needed - num_old, 0)

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
        self.num_cached_block[request_id] = 0

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

        # Decode: free ALL old blocks, then allocate a fresh set for the
        # current selection.
        req_blocks = self.req_to_blocks[request_id]
        if req_blocks:
            non_null = [b for b in req_blocks if not b.is_null]
            self.block_pool.free_blocks(reversed(non_null))
            req_blocks.clear()

        selected = self._selected_block_indices.get(request_id, [])
        num_needed = len(selected) + 1  # history slots + current-token slot

        new_blocks = self.block_pool.get_new_blocks(num_needed)
        req_blocks.extend(new_blocks)
        self.new_block_ids.extend(b.block_id for b in new_blocks)
        self.num_cached_block[request_id] = 0
        return new_blocks

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        # Sparse blocks are not entered into the prefix-hash cache.
        # Advance the pointer so repeated calls are idempotent.
        self.num_cached_block[request.request_id] = (
            num_tokens // self.block_size
        )

    def free(self, request_id: str) -> None:
        super().free(request_id)
        self._cluster_centres.pop(request_id, None)
        self._cluster_value_sum.pop(request_id, None)
        self._cluster_size.pop(request_id, None)
        self._block_to_cluster.pop(request_id, None)
        self._all_block_features.pop(request_id, None)
        self._all_value_features.pop(request_id, None)
        self._mean_key.pop(request_id, None)
        self._pending_query.pop(request_id, None)
        self._selected_block_indices.pop(request_id, None)
        self._prefill_topk_ready.pop(request_id, None)
        self._prefill_selected.pop(request_id, None)
        self._decode_block_buffer.pop(request_id, None)
        self._decode_value_buffer.pop(request_id, None)

    def new_step_starts(self) -> None:
        self._current_step += 1

    # ------------------------------------------------------------------
    # Public sparse-management API
    # Called by KVCacheManager which is called by the Scheduler.
    # ------------------------------------------------------------------

    def indexing(
        self,
        request_id: str,
        block_features: np.ndarray,
        block_value_features: np.ndarray | None = None,
    ) -> None:
        """
        Build the cluster index from prefill block features.

        Called once after prefill completes.  The model runner extracts
        per-block key features (mean K over tokens in the block) and passes
        them as ``block_features``.

        Steps
        -----
        1. Compute global mean; mean-center the features.
        2. Run Segment K-Means to get centroids and assignments.
        3. Restore centering (add mean back to centroids).
        4. Compute per-cluster value_sum (for future Estimation Zone).
        5. Mark prefill TopK as not yet ready (will be triggered by
           ``sparse_update_query_vectors`` immediately after).

        Args:
            request_id:
                The request ID.
            block_features:
                ``[num_blocks, D]`` float32 CPU array – mean K per block.
                D may be ``head_dim`` (averaged across KV heads) or
                ``num_kv_heads * head_dim`` (flattened per-head features).
            block_value_features:
                ``[num_blocks, D]`` float32 CPU array – mean V per block
                (optional; stored for the Estimation Zone TODO).
        """
        num_blocks, D = block_features.shape
        if num_blocks == 0:
            logger.debug("sparse indexing: req %s has 0 blocks – skipped", request_id)
            return

        feat = block_features.astype(np.float32)
        # --- 1. Mean-center ---
        mean_key = feat.mean(axis=0)          # [D]
        centered = feat - mean_key            # [num_blocks, D]

        # --- 2. Segment K-Means on centered features ---
        k = min(self._spec.num_clusters, num_blocks)
        centres, labels, sizes = _segment_kmeans(
            centered,
            n_clusters=k,
            n_segments=min(self._spec.n_segment, num_blocks),
        )

        # --- 3. Restore centering ---
        centres = centres + mean_key          # uncentered centroids

        # --- 4. Value sum per cluster (TODO: Estimation Zone) ---
        n_k = len(centres)
        if block_value_features is not None:
            vfeat = block_value_features.astype(np.float32)
        else:
            vfeat = np.zeros_like(feat)
        value_sum = np.zeros((n_k, D), dtype=np.float32)
        np.add.at(value_sum, labels, vfeat)

        # --- 5. Store ---
        self._cluster_centres[request_id] = centres          # [n_k, D]
        self._cluster_value_sum[request_id] = value_sum      # [n_k, D]
        self._cluster_size[request_id] = sizes               # [n_k]
        self._block_to_cluster[request_id] = labels.tolist() # [num_blocks]
        self._all_block_features[request_id] = [feat[i] for i in range(num_blocks)]
        self._all_value_features[request_id] = [vfeat[i] for i in range(num_blocks)]
        self._mean_key[request_id] = mean_key
        self._prefill_topk_ready[request_id] = False
        self._decode_block_buffer[request_id] = []
        self._decode_value_buffer[request_id] = []

        logger.debug(
            "sparse indexing: req %s – %d blocks → %d clusters (%d segments)",
            request_id, num_blocks, n_k, self._spec.n_segment,
        )

    def select(
        self,
        request_id: str,
        query_vector: np.ndarray,
        num_blocks: int,
    ) -> list[int]:
        """
        Choose the block indices to load for the next decode step.

        Selection logic
        ---------------
        1. **Steady zone** (always included): first ``n_sink_blocks`` and last
           ``n_recent_blocks`` logical block indices.
        2. **Prefill TopK cache** (if valid): return the cached selection from
           after-prefill (merged with the current steady zone).
        3. **Fresh TopK search**: compute attention scores
           ``score[c] = q · centroid[c] / sqrt(D)`` and take the top
           ``nprobe`` clusters; collect all their block indices.
        4. **Budget cap**: if total > ``num_blocks``, keep the steady zone and
           trim the retrieve zone by cluster score.

        Args:
            request_id: The request ID.
            query_vector:
                ``[D]`` float32 query (typically the last token's mean Q
                across the KV-head group, averaged over heads).
            num_blocks:
                Hard cap on the returned list length.

        Returns:
            Sorted list of **logical** block indices (0-based).
        """
        total_blocks = len(self._all_block_features.get(request_id, []))

        # ── Steady zone ────────────────────────────────────────────────
        n_sink = self._spec.n_sink_blocks
        n_recent = self._spec.n_recent_blocks
        sink_set = set(range(min(n_sink, total_blocks)))
        recent_set = set(range(max(0, total_blocks - n_recent), total_blocks))
        steady_set = sink_set | recent_set

        # ── Prefill TopK cache ─────────────────────────────────────────
        if self._prefill_topk_ready.get(request_id, False):
            cached = self._prefill_selected.get(request_id, [])
            if cached:
                result = sorted(set(cached) | steady_set)
                result = result[:num_blocks]
                self._selected_block_indices[request_id] = result
                return result

        # ── Fresh TopK search ──────────────────────────────────────────
        centres = self._cluster_centres.get(request_id)
        block_to_cluster = self._block_to_cluster.get(request_id)

        if centres is None or block_to_cluster is None or len(centres) == 0:
            # No clustering yet (edge case: very short prompt).  Fall back to
            # the most-recent blocks plus the steady zone.
            fallback = set(range(max(0, total_blocks - num_blocks), total_blocks))
            result = sorted(fallback | steady_set)[:num_blocks]
            self._selected_block_indices[request_id] = result
            return result

        # Attention-score TopK: q · centroid / sqrt(D)
        # This matches RetroInfer's batch_gemm_softmax intent.  We skip the
        # softmax here because we only need relative ordering, not probabilities.
        q = query_vector.astype(np.float32)
        D = q.shape[-1]
        scores = (q @ centres.T) / np.sqrt(max(D, 1))  # [n_clusters]

        # Select top-nprobe clusters (retrieve zone).
        nprobe = min(self._spec.nprobe, len(centres))
        # argpartition is O(n) vs O(n log n) for argsort; fine for small n_clusters.
        top_cluster_ids = set(
            int(c) for c in np.argpartition(scores, -nprobe)[-nprobe:]
        )

        retrieve_blocks = {
            bidx
            for bidx, cid in enumerate(block_to_cluster)
            if cid in top_cluster_ids
        }

        # ── Budget cap ────────────────────────────────────────────────
        combined = steady_set | retrieve_blocks
        if len(combined) > num_blocks:
            # Keep all steady-zone blocks; trim non-steady by cluster score.
            non_steady = sorted(
                retrieve_blocks - steady_set,
                key=lambda b: scores[block_to_cluster[b]],
                reverse=True,
            )
            budget = max(0, num_blocks - len(steady_set))
            combined = steady_set | set(non_steady[:budget])

        result = sorted(combined)
        self._selected_block_indices[request_id] = result
        return result

    def update_query_vector(
        self,
        request_id: str,
        query_vec: np.ndarray,
    ) -> None:
        """
        Store the query vector produced at the current step for use in the
        **next** step's ``select()`` call.

        If the prefill TopK window is enabled and ``indexing()`` has already
        run for this request but the TopK cache is not yet ready, this call
        also computes and caches the prefill selection.

        Args:
            request_id: The request ID.
            query_vec:  ``[D]`` float32 query (mean Q per KV-head group).
        """
        self._pending_query[request_id] = query_vec

        # If indexing completed but prefill TopK hasn't been computed yet,
        # compute it now using this query (the last prompt token's query).
        if (
            request_id in self._cluster_centres
            and not self._prefill_topk_ready.get(request_id, False)
            and self._spec.prefill_topk_query_window > 0
        ):
            self._compute_prefill_topk(request_id, query_vec)

    def get_pending_query(self, request_id: str) -> np.ndarray | None:
        """Return the pending query vector, or ``None`` if not yet set."""
        return self._pending_query.get(request_id)

    def rebalance(
        self,
        request_id: str,
        new_block_feature: np.ndarray,
        new_block_value_feature: np.ndarray | None = None,
    ) -> None:
        """
        Absorb a newly generated decode block into the sparse index.

        The new block's feature is buffered.  Once ``update_threshold_blocks``
        blocks have accumulated, a fresh Segment K-Means is run on the buffer
        and the resulting centroids are appended to the existing index
        (matching RetroInfer's ``_update_kv_cache`` logic).

        After a dynamic update the prefill TopK cache is invalidated, and
        ``select()`` will re-compute from scratch on the next decode step
        (i.e., the selection is refreshed using the current query).

        Args:
            request_id:
                The request ID.
            new_block_feature:
                ``[D]`` float32 mean-K feature of the newly written block.
            new_block_value_feature:
                ``[D]`` float32 mean-V feature (optional; stored for TODO).
        """
        feat = new_block_feature.astype(np.float32)
        vfeat = (
            new_block_value_feature.astype(np.float32)
            if new_block_value_feature is not None
            else np.zeros_like(feat)
        )

        # Always register the new block in the global feature lists.
        self._all_block_features.setdefault(request_id, []).append(feat)
        self._all_value_features.setdefault(request_id, []).append(vfeat)

        # Buffer the feature for the pending bulk update.
        buf = self._decode_block_buffer.setdefault(request_id, [])
        vbuf = self._decode_value_buffer.setdefault(request_id, [])
        buf.append(feat)
        vbuf.append(vfeat)

        # Assign to nearest existing cluster (for block_to_cluster bookkeeping).
        centres = self._cluster_centres.get(request_id)
        b2c = self._block_to_cluster.get(request_id)
        if centres is not None and b2c is not None and len(centres) > 0:
            mean_key = self._mean_key.get(request_id, np.zeros(feat.shape))
            centered_feat = feat - mean_key
            centred_C = centres - mean_key
            nearest = int(np.argmax(centered_feat @ centred_C.T))
            b2c.append(nearest)
        else:
            # Fallback: assign to cluster 0 (index will be rebuilt shortly).
            b2c_list = self._block_to_cluster.setdefault(request_id, [])
            b2c_list.append(0)

        # ── Trigger bulk index update ─────────────────────────────────
        threshold = self._spec.update_threshold_blocks
        if len(buf) >= threshold:
            self._dynamic_update(request_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_prefill_topk(
        self, request_id: str, query_vec: np.ndarray
    ) -> None:
        """
        Compute and cache the prefill cluster selection using the provided
        query vector.

        In RetroInfer this uses an average of the last
        ``prefill_topk_query_window`` prompt query vectors.  Here we receive
        a single pre-averaged vector from the model runner (the averaging is
        done on the GPU side, which is more efficient than passing N vectors).

        The cached selection is stored in ``_prefill_selected`` and the flag
        ``_prefill_topk_ready`` is set to ``True``.
        """
        num_blocks = len(self._all_block_features.get(request_id, []))
        selected = self.select(
            request_id,
            query_vec,
            num_blocks=self._spec.max_selected_blocks,
        )
        self._prefill_selected[request_id] = selected
        self._prefill_topk_ready[request_id] = True
        logger.debug(
            "sparse prefill TopK: req %s – cached %d blocks",
            request_id,
            len(selected),
        )

    def _dynamic_update(self, request_id: str) -> None:
        """
        Re-run Segment K-Means on the accumulated decode block buffer and
        append the resulting centroids to the existing index.

        This mirrors RetroInfer's ``_update_kv_cache``:
          - Extract the buffered block features.
          - Mean-center, cluster, restore.
          - Append new centroids / value_sum / sizes / block_to_cluster entries.
          - Invalidate the prefill TopK cache so the next ``select()`` does a
            fresh search with the newly added centroids.
        """
        buf = self._decode_block_buffer.get(request_id, [])
        vbuf = self._decode_value_buffer.get(request_id, [])
        if not buf:
            return

        feat = np.stack(buf, axis=0).astype(np.float32)  # [M, D]
        vfeat = np.stack(vbuf, axis=0).astype(np.float32)
        M, D = feat.shape

        mean_key_new = feat.mean(axis=0)
        centered = feat - mean_key_new

        k_new = max(1, M // 16)  # ~16 blocks per cluster, like RetroInfer
        k_new = (k_new // max(1, 32)) * 32  # round to multiple of 32
        k_new = max(k_new, 1)
        k_new = min(k_new, M)

        centres_new, labels_new, sizes_new = _segment_kmeans(
            centered, n_clusters=k_new, n_segments=1
        )
        centres_new = centres_new + mean_key_new  # restore centering

        n_k_new = len(centres_new)
        vsum_new = np.zeros((n_k_new, D), dtype=np.float32)
        np.add.at(vsum_new, labels_new, vfeat)

        # ── Append to existing index ──────────────────────────────────
        existing_centres = self._cluster_centres.get(request_id)
        if existing_centres is not None:
            n_existing = len(existing_centres)
            self._cluster_centres[request_id] = np.vstack(
                [existing_centres, centres_new]
            )
            self._cluster_value_sum[request_id] = np.vstack([
                self._cluster_value_sum[request_id], vsum_new
            ])
            self._cluster_size[request_id] = np.concatenate([
                self._cluster_size[request_id], sizes_new
            ])
            # The last M entries of block_to_cluster were placeholders; update.
            b2c = self._block_to_cluster[request_id]
            n_total = len(self._all_block_features[request_id])
            for i, lbl in enumerate(labels_new.tolist()):
                idx = n_total - M + i
                if idx < len(b2c):
                    b2c[idx] = n_existing + lbl
                else:
                    b2c.append(n_existing + lbl)
        else:
            # First time (shouldn't happen, but be safe).
            self._cluster_centres[request_id] = centres_new
            self._cluster_value_sum[request_id] = vsum_new
            self._cluster_size[request_id] = sizes_new
            self._block_to_cluster[request_id] = labels_new.tolist()

        # Clear buffer.
        self._decode_block_buffer[request_id] = []
        self._decode_value_buffer[request_id] = []

        # Invalidate prefill TopK so next select() searches fresh.
        self._prefill_topk_ready[request_id] = False
        self._prefill_selected.pop(request_id, None)

        logger.debug(
            "sparse dynamic update: req %s – added %d clusters "
            "(total %d), prefill TopK invalidated",
            request_id,
            n_k_new,
            len(self._cluster_centres[request_id]),
        )
