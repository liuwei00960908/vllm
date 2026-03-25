# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for SparseKVManager and related components.

Coverage
--------
* SparseAttentionSpec – properties, merge(), is_uniform_type()
* _kmeans_dot         – convergence, empty input, k > N clamp
* _segment_kmeans     – segment boundaries, label range, sizes sum
* SparseKVManager.indexing()         – centering, cluster count, value_sum
* SparseKVManager.select()           – steady zone guarantee, TopK, budget cap,
                                       prefill TopK cache hit/miss
* SparseKVManager.update_query_vector() – prefill TopK trigger after indexing
* SparseKVManager.rebalance()        – buffer fill, dynamic update, cache
                                       invalidation
* SparseKVManager allocation helpers – get_num_blocks_to_allocate,
                                       allocate_new_blocks (prefill vs decode)
* SparseKVManager.free()             – state cleanup
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.sparse_kv_cache_manager import (
    SparseKVManager,
    _kmeans_dot,
    _segment_kmeans,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, SparseAttentionSpec

pytestmark = pytest.mark.cpu_test

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BLOCK_SIZE = 16
HEAD_DIM = 32  # keep small for fast tests
NUM_KV_HEADS = 4


def make_spec(**kwargs) -> SparseAttentionSpec:
    defaults = dict(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_DIM,
        dtype=torch.float16,
        num_clusters=8,
        n_segment=4,
        nprobe=4,
        static_pattern_start=0,
        static_pattern_end=0,
        prefill_topk_query_window=4,
        update_threshold_blocks=8,
        max_selected_blocks=32,
    )
    defaults.update(kwargs)
    return SparseAttentionSpec(**defaults)


def make_manager(spec: SparseAttentionSpec | None = None,
                 num_gpu_blocks: int = 256) -> SparseKVManager:
    if spec is None:
        spec = make_spec()
    pool = BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=False,
        hash_block_size=BLOCK_SIZE,
    )
    return SparseKVManager(
        kv_cache_spec=spec,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
    )


def random_features(n_blocks: int, seed: int = 0) -> np.ndarray:
    """Return [n_blocks, HEAD_DIM] float32 random block features."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_blocks, HEAD_DIM)).astype(np.float32)


def _do_indexing(mgr: SparseKVManager, req_id: str,
                 n_blocks: int = 32, seed: int = 0) -> np.ndarray:
    feat = random_features(n_blocks, seed=seed)
    mgr.indexing(req_id, feat)
    return feat


def _register_new_request(mgr: SparseKVManager, req_id: str) -> None:
    """Simulate the state that exists at the START of a decode step
    (i.e. after prefill has allocated some blocks)."""
    mgr.req_to_blocks[req_id] = []
    mgr.num_cached_block[req_id] = 0


# ---------------------------------------------------------------------------
# SparseAttentionSpec
# ---------------------------------------------------------------------------

class TestSparseAttentionSpec:

    def test_n_sink_blocks_zero(self):
        spec = make_spec(static_pattern_start=0, block_size=BLOCK_SIZE)
        assert spec.n_sink_blocks == 0

    def test_n_sink_blocks_aligned(self):
        spec = make_spec(static_pattern_start=32, block_size=16)
        assert spec.n_sink_blocks == 2

    def test_n_sink_blocks_unaligned_rounds_up(self):
        spec = make_spec(static_pattern_start=24, block_size=16)
        assert spec.n_sink_blocks == 2

    def test_n_recent_blocks(self):
        spec = make_spec(static_pattern_end=48, block_size=16)
        assert spec.n_recent_blocks == 3

    def test_n_recent_blocks_partial(self):
        spec = make_spec(static_pattern_end=20, block_size=16)
        assert spec.n_recent_blocks == 2

    def test_merge_preserves_sparse_params(self):
        spec = make_spec(num_clusters=16, n_segment=4, nprobe=8,
                         static_pattern_start=32, static_pattern_end=16,
                         update_threshold_blocks=32)
        merged = SparseAttentionSpec.merge([spec, spec])
        assert merged.num_clusters == 16
        assert merged.n_segment == 4
        assert merged.nprobe == 8
        assert merged.static_pattern_start == 32
        assert merged.static_pattern_end == 16
        assert merged.update_threshold_blocks == 32

    def test_max_memory_usage_proportional_to_selected_blocks(self):
        spec = make_spec(max_selected_blocks=64)
        # should be (64+1) * page_size_bytes
        assert spec.max_memory_usage_bytes(None) == 65 * spec.page_size_bytes


# ---------------------------------------------------------------------------
# _kmeans_dot
# ---------------------------------------------------------------------------

class TestKmeansDot:

    def test_returns_correct_shapes(self):
        feat = random_features(20)
        centres, labels = _kmeans_dot(feat, k=4)
        assert centres.shape == (4, HEAD_DIM)
        assert labels.shape == (20,)

    def test_labels_in_range(self):
        feat = random_features(50)
        _, labels = _kmeans_dot(feat, k=6)
        assert labels.min() >= 0
        assert labels.max() < 6

    def test_k_clamped_to_n(self):
        feat = random_features(3)
        centres, labels = _kmeans_dot(feat, k=10)
        assert centres.shape[0] == 3
        assert labels.max() < 3

    def test_empty_input(self):
        feat = np.zeros((0, HEAD_DIM), dtype=np.float32)
        centres, labels = _kmeans_dot(feat, k=4)
        assert centres.shape == (0, HEAD_DIM)
        assert labels.shape == (0,)

    def test_single_point(self):
        feat = random_features(1)
        centres, labels = _kmeans_dot(feat, k=1)
        assert centres.shape == (1, HEAD_DIM)
        assert labels[0] == 0

    def test_well_separated_clusters_assigned_correctly(self):
        """Points in clearly separated groups should go to the right cluster."""
        rng = np.random.default_rng(42)
        group_a = rng.standard_normal((10, HEAD_DIM)).astype(np.float32) + 10.0
        group_b = rng.standard_normal((10, HEAD_DIM)).astype(np.float32) - 10.0
        feat = np.vstack([group_a, group_b])
        _, labels = _kmeans_dot(feat, k=2, n_iter=30)
        # All 10 points from each group should share the same label.
        assert len(set(labels[:10])) == 1
        assert len(set(labels[10:])) == 1
        assert labels[0] != labels[10]

    def test_dtype_is_float32(self):
        feat = random_features(10).astype(np.float64)
        centres, _ = _kmeans_dot(feat, k=2)
        assert centres.dtype == np.float32


# ---------------------------------------------------------------------------
# _segment_kmeans
# ---------------------------------------------------------------------------

class TestSegmentKmeans:

    def test_returns_correct_shapes(self):
        feat = random_features(40)
        centres, labels, sizes = _segment_kmeans(feat, n_clusters=8, n_segments=4)
        assert centres.ndim == 2 and centres.shape[1] == HEAD_DIM
        assert labels.shape == (40,)
        assert sizes.shape == (len(centres),)

    def test_sizes_sum_to_n(self):
        feat = random_features(48)
        centres, labels, sizes = _segment_kmeans(feat, n_clusters=8, n_segments=4)
        assert int(sizes.sum()) == 48

    def test_labels_contiguous_global_ids(self):
        feat = random_features(32)
        centres, labels, sizes = _segment_kmeans(feat, n_clusters=8, n_segments=4)
        assert labels.min() >= 0
        assert labels.max() < len(centres)

    def test_early_segment_blocks_dont_mix_with_late_segment(self):
        """Blocks in segment 0 and segment 1 should have different cluster ids
        (since segment K-Means runs independently per segment)."""
        feat = random_features(64)
        n_seg = 2
        k_total = 4  # 2 per segment
        _, labels, _ = _segment_kmeans(feat, n_clusters=k_total, n_segments=n_seg)
        seg0_labels = set(labels[:32].tolist())
        seg1_labels = set(labels[32:].tolist())
        # With 2 clusters per segment, seg0 uses ids 0-1 and seg1 uses ids 2-3.
        assert seg0_labels.isdisjoint(seg1_labels)

    def test_n_segments_clamped_to_n_blocks(self):
        feat = random_features(2)
        centres, labels, sizes = _segment_kmeans(feat, n_clusters=8, n_segments=10)
        assert int(sizes.sum()) == 2

    def test_empty_input(self):
        feat = np.zeros((0, HEAD_DIM), dtype=np.float32)
        centres, labels, sizes = _segment_kmeans(feat, n_clusters=4, n_segments=2)
        assert len(centres) == 0
        assert len(labels) == 0


# ---------------------------------------------------------------------------
# SparseKVManager.indexing()
# ---------------------------------------------------------------------------

class TestIndexing:

    def test_cluster_count_at_most_num_clusters(self):
        spec = make_spec(num_clusters=8, n_segment=4)
        mgr = make_manager(spec)
        feat = random_features(32)
        mgr.indexing("req0", feat)
        assert len(mgr._cluster_centres["req0"]) <= spec.num_clusters

    def test_block_to_cluster_length_matches_n_blocks(self):
        mgr = make_manager()
        feat = random_features(20)
        mgr.indexing("req0", feat)
        assert len(mgr._block_to_cluster["req0"]) == 20

    def test_block_to_cluster_ids_in_range(self):
        mgr = make_manager()
        n_blocks = 24
        feat = random_features(n_blocks)
        mgr.indexing("req0", feat)
        n_clusters = len(mgr._cluster_centres["req0"])
        b2c = mgr._block_to_cluster["req0"]
        assert all(0 <= c < n_clusters for c in b2c)

    def test_cluster_sizes_sum_to_n_blocks(self):
        mgr = make_manager()
        n_blocks = 32
        feat = random_features(n_blocks)
        mgr.indexing("req0", feat)
        assert int(mgr._cluster_size["req0"].sum()) == n_blocks

    def test_mean_key_stored(self):
        mgr = make_manager()
        feat = random_features(16)
        mgr.indexing("req0", feat)
        expected_mean = feat.mean(axis=0)
        np.testing.assert_allclose(mgr._mean_key["req0"], expected_mean,
                                   rtol=1e-5, atol=1e-5)

    def test_centroids_near_group_means(self):
        """Centroids should reflect the actual cluster means after centering."""
        mgr = make_manager(make_spec(num_clusters=2, n_segment=1))
        # Two clearly separated groups of 8 blocks each.
        rng = np.random.default_rng(0)
        g0 = (rng.standard_normal((8, HEAD_DIM)) + 5.0).astype(np.float32)
        g1 = (rng.standard_normal((8, HEAD_DIM)) - 5.0).astype(np.float32)
        feat = np.vstack([g0, g1])
        mgr.indexing("req0", feat)
        # With 2 clusters on clearly separated data the centroid norms should
        # differ – a basic sanity check that centering was restored.
        centres = mgr._cluster_centres["req0"]  # uncentered
        assert centres.shape[0] == 2
        norms = np.linalg.norm(centres, axis=1)
        assert abs(norms[0] - norms[1]) > 1.0  # should differ substantially

    def test_value_sum_shape(self):
        """value_sum must have the same shape as centroids."""
        mgr = make_manager()
        feat = random_features(16)
        vfeat = random_features(16, seed=99)
        mgr.indexing("req0", feat, block_value_features=vfeat)
        centres = mgr._cluster_centres["req0"]
        vsum = mgr._cluster_value_sum["req0"]
        assert vsum.shape == centres.shape

    def test_empty_features_noop(self):
        mgr = make_manager()
        feat = np.zeros((0, HEAD_DIM), dtype=np.float32)
        mgr.indexing("req0", feat)
        assert "req0" not in mgr._cluster_centres

    def test_prefill_topk_ready_is_false_after_indexing(self):
        """Prefill TopK must not be marked ready until a query is provided."""
        mgr = make_manager()
        _do_indexing(mgr, "req0")
        assert mgr._prefill_topk_ready.get("req0", False) is False

    def test_all_block_features_stored(self):
        mgr = make_manager()
        n = 20
        feat = random_features(n)
        mgr.indexing("req0", feat)
        assert len(mgr._all_block_features["req0"]) == n

    def test_decode_buffers_cleared_on_reindex(self):
        mgr = make_manager()
        feat = random_features(16)
        mgr.indexing("req0", feat)
        # Buffers should be empty after fresh indexing.
        assert mgr._decode_block_buffer["req0"] == []
        assert mgr._decode_value_buffer["req0"] == []


# ---------------------------------------------------------------------------
# SparseKVManager.select()
# ---------------------------------------------------------------------------

class TestSelect:

    def _prepare(self, n_blocks=32, n_sink=0, n_recent=0, seed=0):
        spec = make_spec(
            static_pattern_start=n_sink * BLOCK_SIZE,
            static_pattern_end=n_recent * BLOCK_SIZE,
            nprobe=4,
            num_clusters=8,
            n_segment=4,
            max_selected_blocks=16,
        )
        mgr = make_manager(spec)
        feat = random_features(n_blocks, seed=seed)
        mgr.indexing("req0", feat)
        return mgr, feat

    def test_result_is_sorted(self):
        mgr, feat = self._prepare()
        q = np.random.default_rng(1).standard_normal(HEAD_DIM).astype(np.float32)
        result = mgr.select("req0", q, num_blocks=16)
        assert result == sorted(result)

    def test_result_within_valid_range(self):
        n = 32
        mgr, feat = self._prepare(n_blocks=n)
        q = np.ones(HEAD_DIM, dtype=np.float32)
        result = mgr.select("req0", q, num_blocks=16)
        assert all(0 <= b < n for b in result)

    def test_result_honours_budget(self):
        mgr, feat = self._prepare()
        q = np.ones(HEAD_DIM, dtype=np.float32)
        budget = 10
        result = mgr.select("req0", q, num_blocks=budget)
        assert len(result) <= budget

    def test_steady_zone_sink_always_included(self):
        n_sink = 2
        mgr, feat = self._prepare(n_blocks=32, n_sink=n_sink)
        q = np.zeros(HEAD_DIM, dtype=np.float32)  # neutral query
        result = mgr.select("req0", q, num_blocks=16)
        for b in range(n_sink):
            assert b in result, f"Sink block {b} missing from selection"

    def test_steady_zone_recent_always_included(self):
        n_blocks = 32
        n_recent = 3
        mgr, feat = self._prepare(n_blocks=n_blocks, n_recent=n_recent)
        q = np.zeros(HEAD_DIM, dtype=np.float32)
        result = mgr.select("req0", q, num_blocks=16)
        for b in range(n_blocks - n_recent, n_blocks):
            assert b in result, f"Recent block {b} missing from selection"

    def test_steady_zone_respects_budget_by_trimming_retrieve(self):
        """When steady zone alone exceeds budget, result must still be within budget."""
        n_sink = 5
        n_recent = 5
        mgr, feat = self._prepare(n_blocks=32, n_sink=n_sink, n_recent=n_recent)
        q = np.zeros(HEAD_DIM, dtype=np.float32)
        budget = 6
        result = mgr.select("req0", q, num_blocks=budget)
        assert len(result) <= budget

    def test_selected_stored_in_state(self):
        mgr, feat = self._prepare()
        q = np.ones(HEAD_DIM, dtype=np.float32)
        returned = mgr.select("req0", q, num_blocks=16)
        assert mgr._selected_block_indices["req0"] == returned

    def test_fallback_on_missing_clusters(self):
        """select() before indexing must not crash; fallback to recent blocks."""
        mgr = make_manager()
        mgr._all_block_features["req0"] = [np.zeros(HEAD_DIM)] * 20
        q = np.ones(HEAD_DIM, dtype=np.float32)
        result = mgr.select("req0", q, num_blocks=8)
        assert len(result) <= 8
        assert all(0 <= b < 20 for b in result)

    def test_prefill_topk_cache_is_used(self):
        """After update_query_vector() triggers TopK cache, select must use it."""
        mgr, feat = self._prepare(n_blocks=32)
        q = np.ones(HEAD_DIM, dtype=np.float32)
        # Trigger prefill TopK caching.
        mgr.update_query_vector("req0", q)
        assert mgr._prefill_topk_ready["req0"] is True
        cached = mgr._prefill_selected["req0"][:]

        # Calling select with a *different* query should still return the cache.
        q2 = -np.ones(HEAD_DIM, dtype=np.float32)
        result = mgr.select("req0", q2, num_blocks=16)
        assert sorted(result) == sorted(set(cached) | set(result))

    def test_select_without_prefill_cache_does_fresh_search(self):
        spec = make_spec(prefill_topk_query_window=0)  # disable prefill cache
        mgr = make_manager(spec)
        feat = random_features(32)
        mgr.indexing("req0", feat)
        # Even after supplying a query, TopK cache must not be set.
        q = np.ones(HEAD_DIM, dtype=np.float32)
        mgr.update_query_vector("req0", q)
        assert not mgr._prefill_topk_ready.get("req0", False)
        # select should still return a valid result.
        result = mgr.select("req0", q, num_blocks=16)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# SparseKVManager.update_query_vector()
# ---------------------------------------------------------------------------

class TestUpdateQueryVector:

    def test_pending_query_stored(self):
        mgr = make_manager()
        q = np.ones(HEAD_DIM, dtype=np.float32)
        mgr.update_query_vector("req0", q)
        assert mgr.get_pending_query("req0") is not None
        np.testing.assert_array_equal(mgr.get_pending_query("req0"), q)

    def test_triggers_prefill_topk_after_indexing(self):
        mgr = make_manager()
        _do_indexing(mgr, "req0")
        assert not mgr._prefill_topk_ready.get("req0", False)
        q = np.ones(HEAD_DIM, dtype=np.float32)
        mgr.update_query_vector("req0", q)
        assert mgr._prefill_topk_ready["req0"] is True
        assert len(mgr._prefill_selected["req0"]) > 0

    def test_does_not_trigger_before_indexing(self):
        mgr = make_manager()
        q = np.ones(HEAD_DIM, dtype=np.float32)
        mgr.update_query_vector("req0", q)
        assert not mgr._prefill_topk_ready.get("req0", False)

    def test_topk_cache_window_zero_disables_trigger(self):
        spec = make_spec(prefill_topk_query_window=0)
        mgr = make_manager(spec)
        _do_indexing(mgr, "req0")
        q = np.ones(HEAD_DIM, dtype=np.float32)
        mgr.update_query_vector("req0", q)
        assert not mgr._prefill_topk_ready.get("req0", False)


# ---------------------------------------------------------------------------
# SparseKVManager.rebalance()
# ---------------------------------------------------------------------------

class TestRebalance:

    def test_new_block_appended_to_all_features(self):
        mgr = make_manager()
        _do_indexing(mgr, "req0", n_blocks=16)
        n_before = len(mgr._all_block_features["req0"])
        feat = np.ones(HEAD_DIM, dtype=np.float32)
        mgr.rebalance("req0", feat)
        assert len(mgr._all_block_features["req0"]) == n_before + 1

    def test_buffer_grows_until_threshold(self):
        threshold = 4
        spec = make_spec(update_threshold_blocks=threshold, num_clusters=4,
                         n_segment=2)
        mgr = make_manager(spec)
        _do_indexing(mgr, "req0", n_blocks=16)

        for i in range(threshold - 1):
            feat = np.ones(HEAD_DIM, dtype=np.float32) * i
            mgr.rebalance("req0", feat)
            assert len(mgr._decode_block_buffer["req0"]) == i + 1

    def test_dynamic_update_triggered_at_threshold(self):
        threshold = 4
        spec = make_spec(update_threshold_blocks=threshold, num_clusters=4,
                         n_segment=2)
        mgr = make_manager(spec)
        _do_indexing(mgr, "req0", n_blocks=16)
        n_clusters_before = len(mgr._cluster_centres["req0"])

        for _ in range(threshold):
            mgr.rebalance("req0",
                          np.random.default_rng(0).standard_normal(
                              HEAD_DIM).astype(np.float32))

        # Buffer should be cleared after the update.
        assert mgr._decode_block_buffer["req0"] == []
        # New centroids must have been appended.
        assert len(mgr._cluster_centres["req0"]) > n_clusters_before

    def test_prefill_topk_invalidated_after_dynamic_update(self):
        threshold = 4
        spec = make_spec(update_threshold_blocks=threshold, num_clusters=4,
                         n_segment=2, prefill_topk_query_window=4)
        mgr = make_manager(spec)
        _do_indexing(mgr, "req0", n_blocks=16)
        # Trigger prefill TopK caching.
        q = np.ones(HEAD_DIM, dtype=np.float32)
        mgr.update_query_vector("req0", q)
        assert mgr._prefill_topk_ready["req0"] is True

        # Accumulate enough blocks to trigger a dynamic update.
        for _ in range(threshold):
            mgr.rebalance("req0", np.ones(HEAD_DIM, dtype=np.float32))

        # Cache must be invalidated.
        assert not mgr._prefill_topk_ready.get("req0", False)

    def test_block_to_cluster_grows(self):
        mgr = make_manager()
        _do_indexing(mgr, "req0", n_blocks=16)
        n_before = len(mgr._block_to_cluster["req0"])
        mgr.rebalance("req0", np.zeros(HEAD_DIM, dtype=np.float32))
        assert len(mgr._block_to_cluster["req0"]) == n_before + 1

    def test_rebalance_without_prior_indexing(self):
        """rebalance() before indexing must not crash."""
        mgr = make_manager()
        feat = np.ones(HEAD_DIM, dtype=np.float32)
        mgr.rebalance("req0", feat)  # should not raise
        assert len(mgr._all_block_features["req0"]) == 1


# ---------------------------------------------------------------------------
# Allocation helpers
# ---------------------------------------------------------------------------

class TestAllocationHelpers:

    # ── get_num_blocks_to_allocate ──────────────────────────────────────

    def test_get_num_blocks_decode_no_prior_blocks(self):
        """Decode step with empty req_to_blocks: need len(selected)+1 blocks."""
        mgr = make_manager()
        _do_indexing(mgr, "req0", n_blocks=16)
        _register_new_request(mgr, "req0")
        # Manually set a selection of 4 blocks.
        mgr._selected_block_indices["req0"] = [0, 1, 2, 3]
        n = mgr.get_num_blocks_to_allocate(
            "req0", num_tokens=1, new_computed_blocks=[],
            total_computed_tokens=0, num_tokens_main_model=1,
        )
        assert n == 5  # 4 selected + 1 new decode slot

    def test_get_num_blocks_decode_with_existing_blocks(self):
        """Net-new = max(needed - existing, 0)."""
        mgr = make_manager()
        pool = mgr.block_pool
        _do_indexing(mgr, "req0", n_blocks=16)
        # Simulate 10 existing blocks already allocated.
        blocks = pool.get_new_blocks(10)
        mgr.req_to_blocks["req0"] = blocks
        mgr.num_cached_block["req0"] = 0
        mgr._selected_block_indices["req0"] = list(range(7))  # 7 selected
        # Need 8 slots, have 10 → 0 new blocks required.
        n = mgr.get_num_blocks_to_allocate(
            "req0", num_tokens=1, new_computed_blocks=[],
            total_computed_tokens=0, num_tokens_main_model=1,
        )
        assert n == 0

    def test_get_num_blocks_decode_need_more_than_existing(self):
        mgr = make_manager()
        pool = mgr.block_pool
        _do_indexing(mgr, "req0", n_blocks=16)
        blocks = pool.get_new_blocks(3)
        mgr.req_to_blocks["req0"] = blocks
        mgr.num_cached_block["req0"] = 0
        # Select 8 blocks → need 9 slots, have 3 → net +6.
        mgr._selected_block_indices["req0"] = list(range(8))
        n = mgr.get_num_blocks_to_allocate(
            "req0", num_tokens=1, new_computed_blocks=[],
            total_computed_tokens=0, num_tokens_main_model=1,
        )
        assert n == 6

    # ── allocate_new_blocks (decode) ────────────────────────────────────

    def test_allocate_new_blocks_decode_frees_old(self):
        mgr = make_manager()
        pool = mgr.block_pool
        _do_indexing(mgr, "req0", n_blocks=16)
        # Give the request 5 old blocks.
        old_blocks = pool.get_new_blocks(5)
        mgr.req_to_blocks["req0"] = list(old_blocks)
        mgr.num_cached_block["req0"] = 0
        old_ids = {b.block_id for b in old_blocks}
        mgr._selected_block_indices["req0"] = [0, 1, 2]

        mgr.allocate_new_blocks("req0", num_tokens=1,
                                num_tokens_main_model=1)

        new_block_ids = {b.block_id for b in mgr.req_to_blocks["req0"]}
        # Old blocks must have been freed (not appearing in the new set).
        assert new_block_ids.isdisjoint(old_ids)

    def test_allocate_new_blocks_decode_correct_count(self):
        mgr = make_manager()
        pool = mgr.block_pool
        _do_indexing(mgr, "req0", n_blocks=16)
        mgr.req_to_blocks["req0"] = []
        mgr.num_cached_block["req0"] = 0
        selected = [0, 1, 2, 3, 4]
        mgr._selected_block_indices["req0"] = selected

        mgr.allocate_new_blocks("req0", num_tokens=1,
                                num_tokens_main_model=1)

        # Should have len(selected)+1 fresh blocks.
        assert len(mgr.req_to_blocks["req0"]) == len(selected) + 1

    def test_allocate_new_blocks_decode_resets_num_cached(self):
        mgr = make_manager()
        pool = mgr.block_pool
        _do_indexing(mgr, "req0", n_blocks=16)
        mgr.req_to_blocks["req0"] = pool.get_new_blocks(3)
        mgr.num_cached_block["req0"] = 3
        mgr._selected_block_indices["req0"] = [0, 1]

        mgr.allocate_new_blocks("req0", num_tokens=1,
                                num_tokens_main_model=1)
        assert mgr.num_cached_block["req0"] == 0

    # ── cache_blocks ────────────────────────────────────────────────────

    def test_cache_blocks_does_not_raise(self):
        """cache_blocks is a lightweight tracker; it must not crash."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher
        from vllm.utils.hashing import sha256
        from vllm.v1.core.kv_cache_utils import init_none_hash
        init_none_hash(sha256)
        block_hasher = get_request_block_hasher(BLOCK_SIZE, sha256)
        sp = SamplingParams(max_tokens=10)
        sp.update_from_generation_config({}, 0)
        req = Request("req0", [0] * 32, sp, None, None, block_hasher)
        mgr = make_manager()
        mgr.req_to_blocks["req0"] = []
        mgr.num_cached_block["req0"] = 0
        mgr.cache_blocks(req, num_tokens=32)
        assert mgr.num_cached_block["req0"] == 32 // BLOCK_SIZE


# ---------------------------------------------------------------------------
# SparseKVManager.free()
# ---------------------------------------------------------------------------

class TestFree:

    def test_free_removes_all_sparse_state(self):
        mgr = make_manager()
        _do_indexing(mgr, "req0", n_blocks=16)
        q = np.ones(HEAD_DIM, dtype=np.float32)
        mgr.update_query_vector("req0", q)
        mgr.rebalance("req0", q)
        _register_new_request(mgr, "req0")

        mgr.free("req0")

        assert "req0" not in mgr._cluster_centres
        assert "req0" not in mgr._cluster_value_sum
        assert "req0" not in mgr._cluster_size
        assert "req0" not in mgr._block_to_cluster
        assert "req0" not in mgr._all_block_features
        assert "req0" not in mgr._all_value_features
        assert "req0" not in mgr._mean_key
        assert "req0" not in mgr._pending_query
        assert "req0" not in mgr._selected_block_indices
        assert "req0" not in mgr._prefill_topk_ready
        assert "req0" not in mgr._prefill_selected
        assert "req0" not in mgr._decode_block_buffer
        assert "req0" not in mgr._decode_value_buffer

    def test_free_is_idempotent(self):
        """Calling free() twice must not raise."""
        mgr = make_manager()
        _do_indexing(mgr, "req0")
        _register_new_request(mgr, "req0")
        mgr.free("req0")
        mgr.free("req0")  # should not raise


# ---------------------------------------------------------------------------
# Integration: indexing → update_query_vector → select → rebalance cycle
# ---------------------------------------------------------------------------

class TestIntegrationCycle:

    def test_full_prefill_to_decode_cycle(self):
        """
        Simulate one request going through:
          1. indexing (after prefill)
          2. update_query_vector (last prompt token's Q → caches prefill TopK)
          3. select (uses cached TopK)
          4. rebalance (absorb one decode block)
          5. select again (still uses cache until threshold)
        """
        n_blocks = 32
        spec = make_spec(num_clusters=8, n_segment=4, nprobe=4,
                         update_threshold_blocks=16, max_selected_blocks=12)
        mgr = make_manager(spec)
        feat = random_features(n_blocks)
        mgr.indexing("req0", feat)

        # Provide a query to trigger prefill TopK.
        q = np.ones(HEAD_DIM, dtype=np.float32)
        mgr.update_query_vector("req0", q)
        assert mgr._prefill_topk_ready["req0"]

        # First decode select must return cached selection.
        sel1 = mgr.select("req0", q, num_blocks=12)
        assert len(sel1) <= 12
        assert sel1 == sorted(sel1)

        # Rebalance with a new decode block.
        decode_feat = np.random.default_rng(7).standard_normal(
            HEAD_DIM).astype(np.float32)
        mgr.rebalance("req0", decode_feat)
        assert len(mgr._all_block_features["req0"]) == n_blocks + 1

        # Second decode select with the same query; cache still valid.
        sel2 = mgr.select("req0", q, num_blocks=12)
        assert sel1 == sel2  # unchanged until dynamic update

    def test_dynamic_update_refreshes_selection(self):
        """After threshold decode blocks, a new select() returns a fresh result."""
        threshold = 4
        spec = make_spec(num_clusters=4, n_segment=2,
                         update_threshold_blocks=threshold,
                         prefill_topk_query_window=4, max_selected_blocks=12)
        mgr = make_manager(spec)
        feat = random_features(16)
        mgr.indexing("req0", feat)

        q = np.ones(HEAD_DIM, dtype=np.float32)
        mgr.update_query_vector("req0", q)
        sel_before = mgr.select("req0", q, num_blocks=12)[:]

        # Feed threshold decode blocks to trigger dynamic update.
        for i in range(threshold):
            rng = np.random.default_rng(i + 100)
            mgr.rebalance("req0", rng.standard_normal(HEAD_DIM).astype(np.float32))

        # Cache must be invalidated.
        assert not mgr._prefill_topk_ready.get("req0", False)
        # select() now does a fresh search; result may differ.
        sel_after = mgr.select("req0", q, num_blocks=12)
        # The result is still valid (sorted, within range, within budget).
        n_total = len(mgr._all_block_features["req0"])
        assert all(0 <= b < n_total for b in sel_after)
        assert sel_after == sorted(sel_after)
        assert len(sel_after) <= 12

    def test_multi_request_isolation(self):
        """Two concurrent requests must not share any state."""
        mgr = make_manager()
        feat_a = random_features(16, seed=1)
        feat_b = random_features(16, seed=2)
        mgr.indexing("req_a", feat_a)
        mgr.indexing("req_b", feat_b)

        qa = np.ones(HEAD_DIM, dtype=np.float32)
        qb = -np.ones(HEAD_DIM, dtype=np.float32)
        mgr.update_query_vector("req_a", qa)
        mgr.update_query_vector("req_b", qb)

        sel_a = mgr.select("req_a", qa, num_blocks=8)
        sel_b = mgr.select("req_b", qb, num_blocks=8)

        # Each request's selection is independently stored.
        assert mgr._selected_block_indices["req_a"] == sel_a
        assert mgr._selected_block_indices["req_b"] == sel_b

        # Freeing one request must not affect the other.
        _register_new_request(mgr, "req_a")
        mgr.free("req_a")
        assert "req_b" in mgr._cluster_centres
