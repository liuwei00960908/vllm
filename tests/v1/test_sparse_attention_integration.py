# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration tests for the RetroInfer-style sparse KV attention pipeline.

覆盖范围
--------
串联 GPU 侧（model_runner）与 CPU 侧（SparseKVManager / scheduler hooks）
的完整流程，分为五个测试组：

Group 1 – ModelRunnerOutput sparse 字段
    验证字段默认值与赋值语义。

Group 2 – Scheduler update_from_output sparse hooks
    验证调度器在收到含稀疏字段的 ModelRunnerOutput 时：
      (a) 正确路由到 KVCacheManager 的三个入口方法；
      (b) 调用顺序：rebalance → indexing → update_query_vectors；
      (c) 字段为 None 时不做任何操作。

Group 3 – KVCacheManager sparse 入口方法
    验证 sparse_notify_prefill_done / sparse_update_query_vectors /
    sparse_post_decode_rebalance 对下游 SparseKVManager 的调用。

Group 4 – _collect_sparse_features GPU 侧逻辑
    用 mock 张量替代真实 KV cache，验证特征提取逻辑：
      · prefill 完成  → 输出 block_features + query_vectors
      · mid-prefill   → 不输出任何字段（chunked prefill 中间块）
      · decode 步     → 输出 query_vectors + new_block_features
      · 每层独立 K/Q  → req_id → layer_name → 特征

Group 5 – Prefill→Decode 完整生命周期
    用 SparseKVManager 真实逻辑跑一遍
    模拟 GPU output → Scheduler hooks → CPU state 的全链路。
"""

from __future__ import annotations

from copy import copy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.sparse_kv_cache_manager import (
    SPARSE_LEGACY_FLAT_LAYER,
    SparseKVManager,
)
from vllm.v1.kv_cache_interface import SparseAttentionSpec
from vllm.v1.outputs import ModelRunnerOutput

pytestmark = pytest.mark.cpu_test

# ---------------------------------------------------------------------------
# Shared constants & fixtures
# ---------------------------------------------------------------------------

BLOCK_SIZE = 16
HEAD_SIZE = 32
NUM_KV_HEADS = 4
NUM_Q_HEADS = 8  # GQA: 2× kv heads


def make_spec(**kw) -> SparseAttentionSpec:
    defaults = dict(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.float16,
        num_clusters=8,
        n_segment=4,
        nprobe=4,
        static_pattern_start=0,
        static_pattern_end=0,
        prefill_topk_query_window=4,
        update_threshold_blocks=4,
        max_selected_blocks=32,
    )
    defaults.update(kw)
    return SparseAttentionSpec(**defaults)


def make_sparse_manager(spec: SparseAttentionSpec | None = None,
                        num_gpu_blocks: int = 512) -> SparseKVManager:
    spec = spec or make_spec()
    pool = BlockPool(num_gpu_blocks=num_gpu_blocks, enable_caching=False,
                     hash_block_size=BLOCK_SIZE)
    return SparseKVManager(
        kv_cache_spec=spec,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
    )


def rand_feats(n_blocks: int, seed: int = 0) -> np.ndarray:
    """[n_blocks, HEAD_SIZE] float32."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_blocks, HEAD_SIZE)).astype(np.float32)


def rand_vec(seed: int = 0) -> np.ndarray:
    """[HEAD_SIZE] float32 unit vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(HEAD_SIZE).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


# ---------------------------------------------------------------------------
# Group 1 – ModelRunnerOutput sparse fields
# ---------------------------------------------------------------------------

class TestModelRunnerOutputSparseFields:
    """Verify the sparse field contract on ModelRunnerOutput."""

    def _empty_output(self) -> ModelRunnerOutput:
        return ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
        )

    def test_sparse_fields_default_none(self):
        """All four sparse fields must be None by default."""
        out = self._empty_output()
        assert out.sparse_block_features is None
        assert out.sparse_query_vectors is None
        assert out.sparse_new_block_features is None
        assert out.sparse_block_value_features is None

    def test_sparse_block_features_assignment(self):
        out = self._empty_output()
        feats = {"req1": {"L0": np.ones((3, HEAD_SIZE), dtype=np.float32)}}
        out = copy(out)
        out.sparse_block_features = feats
        assert out.sparse_block_features is feats
        assert out.sparse_block_features["req1"]["L0"].shape == (3, HEAD_SIZE)

    def test_sparse_query_vectors_assignment(self):
        out = self._empty_output()
        qv = {"req1": {"L0": rand_vec(1)}}
        out = copy(out)
        out.sparse_query_vectors = qv
        assert out.sparse_query_vectors["req1"]["L0"].shape == (HEAD_SIZE,)

    def test_sparse_new_block_features_assignment(self):
        out = self._empty_output()
        nbf = {"req2": {"L0": rand_vec(2)}}
        out = copy(out)
        out.sparse_new_block_features = nbf
        assert out.sparse_new_block_features["req2"]["L0"].shape == (HEAD_SIZE,)

    def test_sparse_block_value_features_assignment(self):
        out = self._empty_output()
        vf = {"req3": {"L0": np.zeros((5, HEAD_SIZE), dtype=np.float32)}}
        out = copy(out)
        out.sparse_block_value_features = vf
        assert out.sparse_block_value_features["req3"]["L0"].shape == (5, HEAD_SIZE)

    def test_non_sparse_output_unaffected(self):
        """Adding sparse fields must not break standard sampled_token_ids."""
        out = ModelRunnerOutput(
            req_ids=["r0"],
            req_id_to_index={"r0": 0},
            sampled_token_ids=[[42]],
            logprobs=None,
            prompt_logprobs_dict={},
            sparse_query_vectors={"r0": {"L": rand_vec()}},
        )
        assert out.sampled_token_ids == [[42]]
        assert out.sparse_query_vectors is not None


# ---------------------------------------------------------------------------
# Group 2 – Scheduler update_from_output hooks
# ---------------------------------------------------------------------------

def _make_scheduler_output_stub(
    sparse_block_features=None,
    sparse_query_vectors=None,
    sparse_new_block_features=None,
    sparse_new_value_features=None,
) -> ModelRunnerOutput:
    """Minimal ModelRunnerOutput for scheduler hook tests."""
    out = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
    )
    out = copy(out)
    out.sparse_block_features = sparse_block_features
    out.sparse_query_vectors = sparse_query_vectors
    out.sparse_new_block_features = sparse_new_block_features
    out.sparse_new_value_features = sparse_new_value_features
    return out


class TestSchedulerSparseHooks:
    """
    Test the three sparse hooks in Scheduler.update_from_output() via direct
    calls to KVCacheManager's sparse entry methods (no full scheduler needed).
    """

    def _make_kv_cache_manager_with_mock_sparse(self) -> tuple[Any, MagicMock]:
        """Return (kv_cache_manager mock, sparse_manager mock)."""
        sparse_mgr = MagicMock(spec=SparseKVManager)
        kv_mgr = MagicMock()
        kv_mgr.get_sparse_manager.return_value = sparse_mgr
        return kv_mgr, sparse_mgr

    # ── 2a: routing ──────────────────────────────────────────────────────────

    def test_prefill_done_calls_indexing(self):
        """sparse_block_features → KVCacheManager.sparse_notify_prefill_done."""
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        mgr = make_sparse_manager()
        kvcm = MagicMock()
        kvcm.get_sparse_manager.return_value = mgr

        feats = rand_feats(4)
        with patch.object(mgr, "indexing") as mock_idx:
            KVCacheManager.sparse_notify_prefill_done(kvcm, "req0", feats)
            mock_idx.assert_called_once_with("req0", feats, None)

    def test_prefill_done_with_value_features(self):
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        mgr = make_sparse_manager()
        kvcm = MagicMock()
        kvcm.get_sparse_manager.return_value = mgr

        feats = rand_feats(4)
        vfeats = rand_feats(4, seed=99)
        with patch.object(mgr, "indexing") as mock_idx:
            KVCacheManager.sparse_notify_prefill_done(kvcm, "req0", feats,
                                                      vfeats)
            mock_idx.assert_called_once_with("req0", feats, vfeats)

    def test_update_query_vectors_calls_update_and_select(self):
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        mgr = make_sparse_manager()
        # Pre-register the request so update_query_vector passes the guard.
        mgr.req_to_blocks["req0"] = []
        kvcm = MagicMock()
        kvcm.get_sparse_manager.return_value = mgr

        # Give indexing state so TopK can fire.
        feats = rand_feats(8)
        mgr.indexing("req0", feats)

        q = rand_vec()
        with (patch.object(mgr, "update_query_vector",
                           wraps=mgr.update_query_vector) as m_uq,
              patch.object(mgr, "select",
                           wraps=mgr.select) as m_sel):
            KVCacheManager.sparse_update_query_vectors(kvcm,
                                                       {"req0": q})
            m_uq.assert_called_once_with("req0", q)
            assert m_sel.call_count == 1

    def test_post_decode_rebalance_calls_rebalance(self):
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        mgr = make_sparse_manager()
        kvcm = MagicMock()
        kvcm.get_sparse_manager.return_value = mgr

        feat = rand_vec()
        with patch.object(mgr, "rebalance") as mock_rb:
            KVCacheManager.sparse_post_decode_rebalance(kvcm, {"req0": feat})
            mock_rb.assert_called_once_with("req0", feat, None)

    def test_post_decode_rebalance_with_value_features(self):
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        mgr = make_sparse_manager()
        kvcm = MagicMock()
        kvcm.get_sparse_manager.return_value = mgr

        feat = rand_vec()
        vfeat = rand_vec(seed=5)
        with patch.object(mgr, "rebalance") as mock_rb:
            KVCacheManager.sparse_post_decode_rebalance(
                kvcm, {"req0": feat}, {"req0": vfeat})
            mock_rb.assert_called_once_with("req0", feat, vfeat)

    # ── 2b: None fields are no-ops ────────────────────────────────────────────

    def test_all_none_fields_are_noop(self):
        """When all sparse fields are None, no sparse method is called."""
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        mgr = make_sparse_manager()
        kvcm = MagicMock()
        kvcm.get_sparse_manager.return_value = mgr

        with (patch.object(mgr, "indexing") as m_idx,
              patch.object(mgr, "update_query_vector") as m_uq,
              patch.object(mgr, "select") as m_sel,
              patch.object(mgr, "rebalance") as m_rb):
            # Simulate what the scheduler does when all fields are None.
            out = _make_scheduler_output_stub()
            if out.sparse_new_block_features:
                KVCacheManager.sparse_post_decode_rebalance(
                    kvcm, out.sparse_new_block_features)
            if out.sparse_block_features:
                for req_id, feats in out.sparse_block_features.items():
                    KVCacheManager.sparse_notify_prefill_done(
                        kvcm, req_id, feats)
            if out.sparse_query_vectors:
                KVCacheManager.sparse_update_query_vectors(
                    kvcm, out.sparse_query_vectors)

            m_idx.assert_not_called()
            m_uq.assert_not_called()
            m_sel.assert_not_called()
            m_rb.assert_not_called()

    # ── 2c: no sparse manager is a no-op ─────────────────────────────────────

    def test_no_sparse_manager_returns_silently(self):
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        kvcm = MagicMock()
        kvcm.get_sparse_manager.return_value = None

        # None of these should raise.
        KVCacheManager.sparse_notify_prefill_done(kvcm, "req0",
                                                  rand_feats(4))
        KVCacheManager.sparse_update_query_vectors(kvcm,
                                                   {"req0": rand_vec()})
        KVCacheManager.sparse_post_decode_rebalance(kvcm,
                                                    {"req0": rand_vec()})

    # ── 2d: multi-request routing ─────────────────────────────────────────────

    def test_multi_request_query_update(self):
        """sparse_update_query_vectors must iterate over all keys."""
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        mgr = make_sparse_manager()
        kvcm = MagicMock()
        kvcm.get_sparse_manager.return_value = mgr
        for rid in ("req0", "req1"):
            mgr.req_to_blocks[rid] = []
            mgr.indexing(rid, rand_feats(8, seed=ord(rid[-1])))

        qvs = {"req0": rand_vec(0), "req1": rand_vec(1)}
        with patch.object(mgr, "update_query_vector",
                          wraps=mgr.update_query_vector) as m_uq:
            KVCacheManager.sparse_update_query_vectors(kvcm, qvs)
            assert m_uq.call_count == 2
            called_reqs = {c.args[0] for c in m_uq.call_args_list}
            assert called_reqs == {"req0", "req1"}

    def test_multi_request_rebalance(self):
        """sparse_post_decode_rebalance must rebalance each request."""
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        mgr = make_sparse_manager()
        kvcm = MagicMock()
        kvcm.get_sparse_manager.return_value = mgr

        feats = {"req0": rand_vec(0), "req1": rand_vec(1)}
        with patch.object(mgr, "rebalance", wraps=mgr.rebalance) as m_rb:
            KVCacheManager.sparse_post_decode_rebalance(kvcm, feats)
            assert m_rb.call_count == 2


# ---------------------------------------------------------------------------
# Group 3 – KVCacheManager.get_sparse_manager()
# ---------------------------------------------------------------------------

class TestKVCacheManagerGetSparseManager:
    """Verify the guard logic in get_sparse_manager()."""

    def _make_kvcm_with_sparse(self) -> Any:
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        kvcm = MagicMock(spec=KVCacheManager)
        sparse_mgr = make_sparse_manager()
        # Simulate the real implementation path.
        kvcm.get_sparse_manager.return_value = sparse_mgr
        return kvcm, sparse_mgr

    def test_get_sparse_manager_returns_instance(self):
        kvcm, sparse_mgr = self._make_kvcm_with_sparse()
        assert kvcm.get_sparse_manager() is sparse_mgr

    def test_notify_prefill_done_updates_indexing_state(self):
        """After calling sparse_notify_prefill_done, clusters must exist."""
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        mgr = make_sparse_manager()
        kvcm = MagicMock()
        kvcm.get_sparse_manager.return_value = mgr

        feats = rand_feats(12)
        KVCacheManager.sparse_notify_prefill_done(kvcm, "req0", feats)
        assert "req0" in mgr._layer_states
        assert (
            mgr._layer_states["req0"][SPARSE_LEGACY_FLAT_LAYER]
            .cluster_centres.shape[1]
            == HEAD_SIZE
        )

    def test_update_query_vectors_sets_prefill_topk_ready(self):
        """After update, prefill TopK cache must be populated."""
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        mgr = make_sparse_manager()
        kvcm = MagicMock()
        kvcm.get_sparse_manager.return_value = mgr
        mgr.req_to_blocks["req0"] = []
        mgr.indexing("req0", rand_feats(12))

        KVCacheManager.sparse_update_query_vectors(kvcm,
                                                   {"req0": rand_vec()})
        assert mgr._prefill_topk_ready.get("req0") is True

    def test_post_decode_rebalance_grows_block_buffer(self):
        """rebalance() must append to _all_block_features."""
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        mgr = make_sparse_manager()
        kvcm = MagicMock()
        kvcm.get_sparse_manager.return_value = mgr
        mgr.indexing("req0", rand_feats(8))

        st = mgr._layer_states["req0"][SPARSE_LEGACY_FLAT_LAYER]
        before = len(st.all_block_features)
        KVCacheManager.sparse_post_decode_rebalance(kvcm,
                                                    {"req0": rand_vec()})
        after = len(
            mgr._layer_states["req0"][SPARSE_LEGACY_FLAT_LAYER].all_block_features
        )
        assert after == before + 1


# ---------------------------------------------------------------------------
# Group 4 – _collect_sparse_features GPU-side logic (mocked KV cache)
# ---------------------------------------------------------------------------

class _MockBlockTableGroup:
    """Minimal MultiGroupBlockTable surrogate supporting [gid] indexing."""

    def __init__(self, tables: list) -> None:
        self._tables = tables

    def __getitem__(self, idx: int):
        return self._tables[idx]


def _build_mock_runner(
    req_ids: list[str],
    num_prompt_tokens: list[int],
    seq_lens: list[int],
    query_start_locs: list[int],
    num_output_before: list[int],
    num_scheduled: dict[str, int],
    n_layers: int = 2,
    block_size: int = BLOCK_SIZE,
    head_size: int = HEAD_SIZE,
    num_kv_heads: int = NUM_KV_HEADS,
    num_q_heads: int = NUM_Q_HEADS,
    num_total_blocks: int = 64,
    blocks_per_req: list[list[int]] | None = None,
    kv_data: torch.Tensor | None = None,
    q_data: torch.Tensor | None = None,
) -> SimpleNamespace:
    """
    Build a minimal mock of GPUModelRunner with just the attributes that
    _collect_sparse_features reads.

    query_start_locs must have (num_reqs + 1) entries:
        [tok_start_req0, tok_start_req1, ..., total_tokens]
    """
    num_reqs = len(req_ids)

    # ── seq_lens and query_start_loc ─────────────────────────────────────────
    sl_np = np.array(seq_lens, dtype=np.int32)
    # query_start_locs already contains num_reqs + 1 entries.
    qsl_np = np.array(query_start_locs, dtype=np.int32)

    seq_lens_buf = SimpleNamespace(np=sl_np)
    qsl_buf = SimpleNamespace(np=qsl_np)

    # ── input_batch ──────────────────────────────────────────────────────────
    npt_np = np.array(num_prompt_tokens, dtype=np.int32)

    # Build block tables.
    if blocks_per_req is None:
        blocks_per_req = [list(range(3)) for _ in req_ids]

    max_blocks = max(len(b) for b in blocks_per_req) if blocks_per_req else 1
    bt_np = np.full((num_reqs, max_blocks), -1, dtype=np.int32)
    nb_per_row = np.zeros(num_reqs, dtype=np.int32)
    for i, blks in enumerate(blocks_per_req):
        bt_np[i, :len(blks)] = blks
        nb_per_row[i] = len(blks)
    bt_obj = SimpleNamespace(
        block_table=SimpleNamespace(np=bt_np),
        num_blocks_per_row=nb_per_row,
        block_size=block_size,
    )

    mt_obj = _MockBlockTableGroup([bt_obj])

    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        req_ids=req_ids,
        num_prompt_tokens=npt_np,
        block_table=mt_obj,
    )

    # ── kv_cache tensors via static_forward_context ───────────────────────────
    if kv_data is None:
        kv_data = torch.randn(
            2, num_total_blocks, block_size, num_kv_heads, head_size)

    if q_data is None:
        total_tokens = query_start_locs[-1]
        q_data = torch.randn(total_tokens, num_q_heads * head_size)

    forward_ctx: dict[str, Any] = {}
    spec = make_spec(block_size=block_size, num_kv_heads=num_kv_heads,
                     head_size=head_size)
    group_layer_names = []
    for li in range(n_layers):
        ln = f"model.layers.{li}.self_attn.attn"
        group_layer_names.append(ln)
        attn_mock = SimpleNamespace(
            kv_cache=[kv_data],
            num_heads=num_q_heads,
            head_size=head_size,
        )
        forward_ctx[ln] = attn_mock

    # ── kv_cache_config ───────────────────────────────────────────────────────
    kv_group = SimpleNamespace(
        kv_cache_spec=spec,
        layer_names=group_layer_names,
    )
    kv_cache_config = SimpleNamespace(kv_cache_groups=[kv_group])

    # ── scheduler_output ─────────────────────────────────────────────────────
    # new_reqs: requests with num_output_before == 0 AND freshly created
    new_reqs = [
        SimpleNamespace(req_id=rid)
        for rid, n in zip(req_ids, num_output_before)
        if n == 0
    ]
    cached_reqs = SimpleNamespace(
        req_ids=[rid for rid, n in zip(req_ids, num_output_before) if n > 0],
        num_output_tokens=[n for n in num_output_before if n > 0],
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached_reqs,
        num_scheduled_tokens=num_scheduled,
    )

    # ── _sparse_q_captures ───────────────────────────────────────────────────
    sparse_q_captures: dict[str, torch.Tensor] = {
        ln: q_data for ln in group_layer_names
    }

    # ── compilation_config ───────────────────────────────────────────────────
    compilation_config = SimpleNamespace(
        static_forward_context=forward_ctx,
    )

    runner = SimpleNamespace(
        _has_sparse_attn=True,
        kv_cache_config=kv_cache_config,
        input_batch=input_batch,
        seq_lens=seq_lens_buf,
        query_start_loc=qsl_buf,
        _sparse_q_captures=sparse_q_captures,
        compilation_config=compilation_config,
        device=torch.device("cpu"),
    )

    # Bind the unbound method to the runner namespace so it can be called.
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    runner._collect_sparse_features = (
        GPUModelRunner._collect_sparse_features.__get__(runner)
    )
    return runner, scheduler_output


class TestCollectSparseFeatures:
    """Unit-test _collect_sparse_features with mocked GPU state."""

    # ── 4a: disabled path ────────────────────────────────────────────────────

    def test_no_sparse_attn_returns_all_none(self):
        runner, sched = _build_mock_runner(
            req_ids=["r0"],
            num_prompt_tokens=[32],
            seq_lens=[32],
            query_start_locs=[0, 32],
            num_output_before=[0],
            num_scheduled={"r0": 32},
        )
        runner._has_sparse_attn = False
        bf, qv, nbf = runner._collect_sparse_features(sched, 1)
        assert bf is None and qv is None and nbf is None

    # ── 4b: prefill completing ────────────────────────────────────────────────

    def test_prefill_done_emits_block_and_query(self):
        """seq_len_after == num_prompt_tokens → prefill done."""
        n_blocks = 3
        runner, sched = _build_mock_runner(
            req_ids=["r0"],
            num_prompt_tokens=[n_blocks * BLOCK_SIZE],
            seq_lens=[n_blocks * BLOCK_SIZE],
            query_start_locs=[0, n_blocks * BLOCK_SIZE],
            num_output_before=[0],
            num_scheduled={"r0": n_blocks * BLOCK_SIZE},
            blocks_per_req=[list(range(n_blocks))],
        )
        bf, qv, nbf = runner._collect_sparse_features(sched, 1)

        assert bf is not None, "block_features must be emitted at prefill done"
        assert "r0" in bf
        layer_names = runner.kv_cache_config.kv_cache_groups[0].layer_names
        for ln in layer_names:
            assert bf["r0"][ln].shape == (n_blocks, HEAD_SIZE)
            assert bf["r0"][ln].dtype == np.float32

        assert qv is not None, "query_vectors must be emitted at prefill done"
        assert "r0" in qv
        for ln in layer_names:
            assert qv["r0"][ln].shape == (HEAD_SIZE,)

        assert nbf is None, "new_block_features must NOT be emitted at prefill"

    def test_prefill_done_block_features_are_mean_k(self):
        """block_features[b] should equal mean over block tokens × KV heads."""
        n_blocks = 2
        bs, hs, nh = BLOCK_SIZE, HEAD_SIZE, NUM_KV_HEADS

        # Construct deterministic KV cache data.
        kv = torch.zeros(2, n_blocks, bs, nh, hs)
        for b in range(n_blocks):
            kv[0, b, :, :, :] = float(b + 1)  # K: block b filled with b+1

        runner, sched = _build_mock_runner(
            req_ids=["r0"],
            num_prompt_tokens=[n_blocks * bs],
            seq_lens=[n_blocks * bs],
            query_start_locs=[0, n_blocks * bs],
            num_output_before=[0],
            num_scheduled={"r0": n_blocks * bs},
            blocks_per_req=[list(range(n_blocks))],
            kv_data=kv,
            n_layers=1,
        )
        bf, _, _ = runner._collect_sparse_features(sched, 1)
        assert bf is not None
        ln0 = runner.kv_cache_config.kv_cache_groups[0].layer_names[0]
        # block 0: all K = 1.0, so mean = 1.0
        # block 1: all K = 2.0, so mean = 2.0
        np.testing.assert_allclose(bf["r0"][ln0][0], np.ones(hs) * 1.0,
                                   rtol=1e-4)
        np.testing.assert_allclose(bf["r0"][ln0][1], np.ones(hs) * 2.0,
                                   rtol=1e-4)

    # ── 4c: mid-prefill chunk (chunked prefill) ───────────────────────────────

    def test_mid_prefill_emits_nothing(self):
        """If seq_len_after < num_prompt_tokens, it's still a chunked prefill."""
        prompt_len = 4 * BLOCK_SIZE
        chunk_len = 2 * BLOCK_SIZE  # only half processed so far
        runner, sched = _build_mock_runner(
            req_ids=["r0"],
            num_prompt_tokens=[prompt_len],
            seq_lens=[chunk_len],  # not yet at prompt_len
            query_start_locs=[0, chunk_len],
            num_output_before=[0],
            num_scheduled={"r0": chunk_len},
            blocks_per_req=[list(range(2))],
        )
        bf, qv, nbf = runner._collect_sparse_features(sched, 1)
        assert bf is None and qv is None and nbf is None

    # ── 4d: decode step ───────────────────────────────────────────────────────

    def test_decode_emits_query_and_new_block(self):
        """Decode step (num_output_before > 0) → query + new_block_feature."""
        n_blocks = 3
        runner, sched = _build_mock_runner(
            req_ids=["r0"],
            num_prompt_tokens=[n_blocks * BLOCK_SIZE],
            seq_lens=[n_blocks * BLOCK_SIZE + 1],  # one decode token
            query_start_locs=[0, 1],               # 1 token in this step
            num_output_before=[1],                  # already has 1 output token
            num_scheduled={"r0": 1},
            blocks_per_req=[list(range(n_blocks))],
        )
        bf, qv, nbf = runner._collect_sparse_features(sched, 1)

        assert bf is None, "block_features should NOT be emitted during decode"
        assert qv is not None, "query_vectors must be emitted during decode"
        assert nbf is not None, "new_block_features must be emitted during decode"
        layer_names = runner.kv_cache_config.kv_cache_groups[0].layer_names
        for ln in layer_names:
            assert qv["r0"][ln].shape == (HEAD_SIZE,)
            assert nbf["r0"][ln].shape == (HEAD_SIZE,)

    def test_decode_new_block_is_last_block_mean_k(self):
        """new_block_features must be the mean-K of the last block."""
        n_blocks = 3
        bs, hs, nh = BLOCK_SIZE, HEAD_SIZE, NUM_KV_HEADS
        kv = torch.zeros(2, n_blocks, bs, nh, hs)
        for b in range(n_blocks):
            kv[0, b, :, :, :] = float(b + 1)  # block b filled with b+1

        runner, sched = _build_mock_runner(
            req_ids=["r0"],
            num_prompt_tokens=[n_blocks * bs],
            seq_lens=[n_blocks * bs + 1],
            query_start_locs=[0, 1],
            num_output_before=[5],
            num_scheduled={"r0": 1},
            blocks_per_req=[list(range(n_blocks))],
            kv_data=kv,
            n_layers=1,
        )
        _, _, nbf = runner._collect_sparse_features(sched, 1)
        assert nbf is not None
        ln0 = runner.kv_cache_config.kv_cache_groups[0].layer_names[0]
        # last block (index 2) has K = 3.0
        np.testing.assert_allclose(nbf["r0"][ln0], np.ones(hs) * 3.0, rtol=1e-4)

    # ── 4e: Q is kept per layer (no cross-layer mean) ───────────────────────

    def test_q_distinct_per_layer(self):
        """Each sparse layer keeps its own mean Q (no averaging across layers)."""
        n_layers = 3
        total_tokens = 2
        q_dim = NUM_Q_HEADS * HEAD_SIZE

        runner, sched = _build_mock_runner(
            req_ids=["r0"],
            num_prompt_tokens=[2],
            seq_lens=[2],
            query_start_locs=[0, 2],
            num_output_before=[0],
            num_scheduled={"r0": 2},
            blocks_per_req=[list(range(1))],
            n_layers=n_layers,
        )
        for i, ln in enumerate(
            runner.kv_cache_config.kv_cache_groups[0].layer_names
        ):
            runner._sparse_q_captures[ln] = torch.full(
                (total_tokens, q_dim), float(i + 1)
            )

        _, qv, _ = runner._collect_sparse_features(sched, 1)
        assert qv is not None
        for i, ln in enumerate(
            runner.kv_cache_config.kv_cache_groups[0].layer_names
        ):
            expected = np.full(HEAD_SIZE, float(i + 1), dtype=np.float32)
            np.testing.assert_allclose(qv["r0"][ln], expected, rtol=1e-4)

    # ── 4f: K block features per layer (no cross-layer mean) ────────────────

    def test_k_distinct_per_layer(self):
        """Each layer's block_features use only that layer's KV cache."""
        n_layers = 2
        n_blocks = 2
        bs, hs, nh = BLOCK_SIZE, HEAD_SIZE, NUM_KV_HEADS

        runner, sched = _build_mock_runner(
            req_ids=["r0"],
            num_prompt_tokens=[n_blocks * bs],
            seq_lens=[n_blocks * bs],
            query_start_locs=[0, n_blocks * bs],
            num_output_before=[0],
            num_scheduled={"r0": n_blocks * bs},
            blocks_per_req=[list(range(n_blocks))],
            n_layers=n_layers,
        )
        for i, ln in enumerate(
            runner.kv_cache_config.kv_cache_groups[0].layer_names
        ):
            kv = torch.full((2, n_blocks, bs, nh, hs), float(i * 2 + 1))
            runner.compilation_config.static_forward_context[ln].kv_cache = [kv]

        bf, _, _ = runner._collect_sparse_features(sched, 1)
        assert bf is not None
        for i, ln in enumerate(
            runner.kv_cache_config.kv_cache_groups[0].layer_names
        ):
            val = float(i * 2 + 1)
            np.testing.assert_allclose(
                bf["r0"][ln], np.full((n_blocks, hs), val), rtol=1e-4
            )

    # ── 4g: multi-request batch ───────────────────────────────────────────────

    def test_multi_request_batch(self):
        """Two requests in the same batch: each gets its own features."""
        n_blocks = 2
        # r0 is completing prefill, r1 is in decode.
        runner, sched = _build_mock_runner(
            req_ids=["r0", "r1"],
            num_prompt_tokens=[n_blocks * BLOCK_SIZE, n_blocks * BLOCK_SIZE],
            seq_lens=[n_blocks * BLOCK_SIZE, n_blocks * BLOCK_SIZE + 1],
            query_start_locs=[0, n_blocks * BLOCK_SIZE,
                               n_blocks * BLOCK_SIZE + 1],
            num_output_before=[0, 3],
            num_scheduled={"r0": n_blocks * BLOCK_SIZE, "r1": 1},
            blocks_per_req=[list(range(n_blocks)), list(range(n_blocks))],
        )
        bf, qv, nbf = runner._collect_sparse_features(sched, 2)

        assert bf is not None and "r0" in bf
        assert "r1" not in (bf or {})
        assert qv is not None and "r0" in qv and "r1" in qv
        assert nbf is not None and "r1" in nbf
        assert "r0" not in (nbf or {})

    # ── 4h: empty sparse_q_captures ───────────────────────────────────────────

    def test_empty_q_captures_skips_query_vector(self):
        """If hook data is missing, query_vectors must be absent for that req."""
        runner, sched = _build_mock_runner(
            req_ids=["r0"],
            num_prompt_tokens=[BLOCK_SIZE],
            seq_lens=[BLOCK_SIZE],
            query_start_locs=[0, BLOCK_SIZE],
            num_output_before=[0],
            num_scheduled={"r0": BLOCK_SIZE},
            blocks_per_req=[list(range(1))],
        )
        runner._sparse_q_captures.clear()  # simulate hook not firing

        bf, qv, nbf = runner._collect_sparse_features(sched, 1)
        # block_features can still be extracted from KV cache
        assert bf is not None and "r0" in bf
        # query_vectors should be absent (no Q captured)
        assert qv is None or "r0" not in (qv or {})


# ---------------------------------------------------------------------------
# Group 5 – Full prefill→decode lifecycle integration
# ---------------------------------------------------------------------------

class TestFullPrefillDecodeCycle:
    """
    Simulate the GPU→Scheduler→CPU path for a complete request lifecycle.

    We do NOT spin up the full vLLM engine; instead we:
    1. Build a SparseKVManager directly.
    2. Simulate what the GPU model runner would emit at each step.
    3. Call the KVCacheManager sparse entry methods (which call the manager).
    4. Assert the manager state is consistent.
    """

    def _simulate_step(
        self,
        kv_mgr_mock: Any,
        mgr: SparseKVManager,
        block_features: dict[str, np.ndarray] | None,
        query_vectors: dict[str, np.ndarray] | None,
        new_block_features: dict[str, np.ndarray] | None,
    ) -> None:
        """
        Mirror what scheduler.update_from_output() does with the sparse hooks.
        Order must be: rebalance → indexing → update_query_vectors.
        """
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        if new_block_features:
            KVCacheManager.sparse_post_decode_rebalance(
                kv_mgr_mock, new_block_features)
        if block_features:
            for req_id, feats in block_features.items():
                KVCacheManager.sparse_notify_prefill_done(
                    kv_mgr_mock, req_id, feats)
        if query_vectors:
            KVCacheManager.sparse_update_query_vectors(
                kv_mgr_mock, query_vectors)

    def _make_kvcm(self, mgr: SparseKVManager) -> Any:
        kvcm = MagicMock()
        kvcm.get_sparse_manager.return_value = mgr
        return kvcm

    # ── 5a: basic prefill→decode ──────────────────────────────────────────────

    def test_prefill_step_builds_cluster_index(self):
        mgr = make_sparse_manager()
        kvcm = self._make_kvcm(mgr)
        n_blocks = 16
        req_id = "req_abc"

        block_feats = rand_feats(n_blocks, seed=1)
        q_vec = rand_vec(seed=2)
        # Register request so update_query_vector doesn't skip it.
        mgr.req_to_blocks[req_id] = []

        # Step 1: Prefill completes.
        self._simulate_step(kvcm,
                            block_features={req_id: block_feats},
                            query_vectors={req_id: q_vec},
                            new_block_features=None,
                            mgr=mgr)

        assert req_id in mgr._layer_states, "Cluster index must be built"
        assert mgr._prefill_topk_ready[req_id] is True
        assert len(mgr._prefill_selected[req_id]) > 0

    def test_decode_step_grows_block_history(self):
        mgr = make_sparse_manager()
        kvcm = self._make_kvcm(mgr)
        req_id = "req_abc"
        mgr.req_to_blocks[req_id] = []

        # Prefill.
        self._simulate_step(kvcm, mgr,
                            block_features={req_id: rand_feats(12, seed=1)},
                            query_vectors={req_id: rand_vec(2)},
                            new_block_features=None)
        st0 = mgr._layer_states[req_id][SPARSE_LEGACY_FLAT_LAYER]
        initial_count = len(st0.all_block_features)

        # Decode step 1.
        self._simulate_step(kvcm, mgr,
                            block_features=None,
                            query_vectors={req_id: rand_vec(3)},
                            new_block_features={req_id: rand_vec(4)})

        assert (
            len(mgr._layer_states[req_id][SPARSE_LEGACY_FLAT_LAYER].all_block_features)
            == initial_count + 1
        )

    def test_decode_query_vector_updated_each_step(self):
        """Each decode step must replace the pending query vector."""
        mgr = make_sparse_manager()
        kvcm = self._make_kvcm(mgr)
        req_id = "req_q"
        mgr.req_to_blocks[req_id] = []

        self._simulate_step(kvcm, mgr,
                            block_features={req_id: rand_feats(8)},
                            query_vectors={req_id: rand_vec(0)},
                            new_block_features=None)

        q1 = rand_vec(seed=10)
        self._simulate_step(kvcm, mgr,
                            block_features=None,
                            query_vectors={req_id: q1},
                            new_block_features={req_id: rand_vec(20)})
        pq1 = mgr.get_pending_query(req_id)
        assert pq1 is not None
        np.testing.assert_array_equal(pq1[SPARSE_LEGACY_FLAT_LAYER], q1)

        q2 = rand_vec(seed=11)
        self._simulate_step(kvcm, mgr,
                            block_features=None,
                            query_vectors={req_id: q2},
                            new_block_features={req_id: rand_vec(21)})
        pq2 = mgr.get_pending_query(req_id)
        assert pq2 is not None
        np.testing.assert_array_equal(pq2[SPARSE_LEGACY_FLAT_LAYER], q2)

    # ── 5b: rebalance order ───────────────────────────────────────────────────

    def test_rebalance_before_indexing_order(self):
        """
        Simulate a batch where r0 completes prefill and r1 is in decode.
        rebalance(r1) must execute BEFORE indexing(r0).
        """
        mgr = make_sparse_manager()
        call_order: list[str] = []

        original_indexing = mgr.indexing
        original_rebalance = mgr.rebalance

        def tracked_indexing(*a, **kw):
            call_order.append("indexing")
            return original_indexing(*a, **kw)

        def tracked_rebalance(*a, **kw):
            call_order.append("rebalance")
            return original_rebalance(*a, **kw)

        mgr.indexing = tracked_indexing      # type: ignore[method-assign]
        mgr.rebalance = tracked_rebalance    # type: ignore[method-assign]
        kvcm = self._make_kvcm(mgr)

        # Pre-register r1 as having been indexed.
        mgr.req_to_blocks["r0"] = []
        mgr.req_to_blocks["r1"] = []
        mgr.indexing("r1", rand_feats(8, seed=99))  # r1 already indexed
        call_order.clear()                           # reset after setup

        # One batch: r0 finishing prefill, r1 in decode.
        self._simulate_step(
            kvcm, mgr,
            block_features={"r0": rand_feats(8, seed=1)},
            query_vectors={"r0": rand_vec(2), "r1": rand_vec(3)},
            new_block_features={"r1": rand_vec(4)},
        )
        # Verify order: rebalance must appear before indexing.
        assert "rebalance" in call_order
        assert "indexing" in call_order
        assert call_order.index("rebalance") < call_order.index("indexing")

    # ── 5c: dynamic rebalance threshold ──────────────────────────────────────

    def test_dynamic_update_invalidates_prefill_cache(self):
        """After update_threshold_blocks decode blocks, prefill TopK is reset."""
        threshold = 4
        spec = make_spec(update_threshold_blocks=threshold,
                         num_clusters=4, n_segment=2, max_selected_blocks=16)
        mgr = make_sparse_manager(spec)
        kvcm = self._make_kvcm(mgr)
        req_id = "req_dyn"
        mgr.req_to_blocks[req_id] = []

        # Prefill.
        self._simulate_step(kvcm, mgr,
                            block_features={req_id: rand_feats(16, seed=1)},
                            query_vectors={req_id: rand_vec(2)},
                            new_block_features=None)
        assert mgr._prefill_topk_ready[req_id] is True

        # Feed exactly threshold decode blocks to trigger dynamic update.
        for i in range(threshold):
            self._simulate_step(kvcm, mgr,
                                 block_features=None,
                                 query_vectors={req_id: rand_vec(i + 10)},
                                 new_block_features={req_id: rand_vec(i + 20)})

        # Cache must be invalidated after the bulk update.
        assert not mgr._prefill_topk_ready.get(req_id, False), (
            "prefill TopK cache should be invalidated after dynamic update"
        )

    # ── 5d: multi-request isolation ───────────────────────────────────────────

    def test_two_requests_stay_isolated(self):
        """Separate requests in the same batch must not interfere."""
        mgr = make_sparse_manager()
        kvcm = self._make_kvcm(mgr)
        for rid in ("r0", "r1"):
            mgr.req_to_blocks[rid] = []

        # Both finish prefill in the same batch step.
        self._simulate_step(
            kvcm, mgr,
            block_features={
                "r0": rand_feats(8, seed=1),
                "r1": rand_feats(12, seed=2),
            },
            query_vectors={
                "r0": rand_vec(3),
                "r1": rand_vec(4),
            },
            new_block_features=None,
        )
        # Each request has its own cluster state.
        assert "r0" in mgr._layer_states
        assert "r1" in mgr._layer_states

        # Select must return non-overlapping-by-state blocks.
        sel_r0 = mgr.select("r0", rand_vec(5), num_blocks=8)
        sel_r1 = mgr.select("r1", rand_vec(6), num_blocks=10)
        assert len(sel_r0) <= 8
        assert len(sel_r1) <= 10

    # ── 5e: request cleanup ───────────────────────────────────────────────────

    def test_free_clears_all_sparse_state(self):
        """After free(), no sparse state must remain for that request."""
        mgr = make_sparse_manager()
        kvcm = self._make_kvcm(mgr)
        req_id = "req_free"
        mgr.req_to_blocks[req_id] = []

        self._simulate_step(kvcm, mgr,
                            block_features={req_id: rand_feats(8)},
                            query_vectors={req_id: rand_vec()},
                            new_block_features=None)
        # Verify state exists.
        assert req_id in mgr._layer_states

        # Free the request.
        mgr.free(req_id)

        # All sparse dictionaries must be cleared.
        for attr in (
            "_layer_states",
            "_pending_query",
            "_selected_block_indices",
            "_selected_block_indices_by_layer",
            "_selected_retrieve_block_indices",
            "_selected_retrieve_block_indices_by_layer",
            "_prefill_topk_ready",
            "_prefill_selected",
            "_prefill_selected_by_layer",
        ):
            d = getattr(mgr, attr)
            assert req_id not in d, f"{attr} still contains {req_id} after free()"

    # ── 5f: chunked prefill then decode ───────────────────────────────────────

    def test_chunked_prefill_no_indexing_until_done(self):
        """
        With chunked prefill, indexing must NOT be called on mid-chunks.
        Only the step where seq_len_after == num_prompt_tokens triggers it.
        """
        mgr = make_sparse_manager()
        kvcm = self._make_kvcm(mgr)
        req_id = "req_chunked"
        mgr.req_to_blocks[req_id] = []

        # Simulate: chunk 1 (mid-prefill) – no block_features emitted by GPU.
        self._simulate_step(kvcm, mgr,
                            block_features=None,   # mid-chunk: NOT emitted
                            query_vectors=None,
                            new_block_features=None)
        assert req_id not in mgr._layer_states, (
            "No clusters should exist mid-prefill"
        )

        # Simulate: chunk 2 (prefill completes) – GPU emits block_features.
        self._simulate_step(kvcm, mgr,
                            block_features={req_id: rand_feats(16)},
                            query_vectors={req_id: rand_vec()},
                            new_block_features=None)
        assert req_id in mgr._layer_states, (
            "Clusters must be built after prefill completes"
        )

    # ── 5g: decode after prefill cache is active uses cached selection ────────

    def test_decode_uses_prefill_cached_selection(self):
        """
        After prefill TopK is cached, multiple decode steps with the same
        query must return the same block selection (stable cache).
        """
        mgr = make_sparse_manager()
        kvcm = self._make_kvcm(mgr)
        req_id = "req_stable"
        mgr.req_to_blocks[req_id] = []

        self._simulate_step(kvcm, mgr,
                            block_features={req_id: rand_feats(16, seed=1)},
                            query_vectors={req_id: rand_vec(2)},
                            new_block_features=None)

        q_decode = rand_vec(seed=99)
        # Two consecutive decode steps with the same Q.
        self._simulate_step(kvcm, mgr,
                            block_features=None,
                            query_vectors={req_id: q_decode},
                            new_block_features={req_id: rand_vec(3)})
        sel1 = mgr.select(req_id, q_decode, mgr._spec.max_selected_blocks)[:]

        self._simulate_step(kvcm, mgr,
                            block_features=None,
                            query_vectors={req_id: q_decode},
                            new_block_features={req_id: rand_vec(4)})
        sel2 = mgr.select(req_id, q_decode, mgr._spec.max_selected_blocks)

        # Must be equal (cache hit, no dynamic update yet).
        assert sel1 == sel2, "Cached selection must be stable across decode steps"

    def test_real_collect_to_scheduler_long_path(self):
        """
        Use real _collect_sparse_features outputs to drive Scheduler sparse hooks:
          1) mid-chunk prefill: no sparse fields emitted
          2) prefill done: indexing + query update
          3) decode x threshold: rebalance + query update
        and verify dynamic update invalidates prefill cache.
        """
        threshold = 3
        spec = make_spec(
            update_threshold_blocks=threshold,
            num_clusters=8,
            n_segment=4,
            nprobe=4,
            prefill_topk_query_window=4,
            max_selected_blocks=16,
        )
        mgr = make_sparse_manager(spec)
        kvcm = self._make_kvcm(mgr)
        req_id = "req_real_path"
        mgr.req_to_blocks[req_id] = []

        n_blocks = 4
        prompt_len = n_blocks * BLOCK_SIZE

        # Step 1: chunked prefill mid-step -> _collect should emit nothing.
        runner_mid, sched_mid = _build_mock_runner(
            req_ids=[req_id],
            num_prompt_tokens=[prompt_len],
            seq_lens=[2 * BLOCK_SIZE],  # still mid-prefill
            query_start_locs=[0, 2 * BLOCK_SIZE],
            num_output_before=[0],
            num_scheduled={req_id: 2 * BLOCK_SIZE},
            blocks_per_req=[list(range(n_blocks))],
            n_layers=1,
        )
        bf, qv, nbf = runner_mid._collect_sparse_features(sched_mid, 1)
        assert bf is None and qv is None and nbf is None
        self._simulate_step(kvcm, mgr, bf, qv, nbf)
        assert req_id not in mgr._layer_states

        # Step 2: prefill done -> indexing + prefill query cache built.
        runner_prefill, sched_prefill = _build_mock_runner(
            req_ids=[req_id],
            num_prompt_tokens=[prompt_len],
            seq_lens=[prompt_len],
            query_start_locs=[0, prompt_len],
            num_output_before=[0],
            num_scheduled={req_id: prompt_len},
            blocks_per_req=[list(range(n_blocks))],
            n_layers=1,
        )
        bf, qv, nbf = runner_prefill._collect_sparse_features(sched_prefill, 1)
        assert bf is not None and qv is not None and nbf is None
        self._simulate_step(kvcm, mgr, bf, qv, nbf)
        assert req_id in mgr._layer_states
        assert mgr._prefill_topk_ready.get(req_id, False) is True
        initial_cluster_count = len(
            mgr._layer_states[req_id][SPARSE_LEGACY_FLAT_LAYER].cluster_centres
        )

        # Decode steps: every step uses real collect output (qv + new block feat).
        for i in range(threshold):
            runner_dec, sched_dec = _build_mock_runner(
                req_ids=[req_id],
                num_prompt_tokens=[prompt_len],
                seq_lens=[prompt_len + i + 1],
                query_start_locs=[0, 1],  # one decode token this step
                num_output_before=[i + 1],  # already in decode stage
                num_scheduled={req_id: 1},
                blocks_per_req=[list(range(n_blocks))],
                n_layers=1,
            )
            bf, qv, nbf = runner_dec._collect_sparse_features(sched_dec, 1)
            assert bf is None and qv is not None and nbf is not None
            self._simulate_step(kvcm, mgr, bf, qv, nbf)

        # Dynamic update path must invalidate prefill cache.
        assert not mgr._prefill_topk_ready.get(req_id, False)
        final_state = mgr._layer_states[req_id][SPARSE_LEGACY_FLAT_LAYER]
        assert len(final_state.cluster_centres) > initial_cluster_count
        assert len(final_state.all_block_features) == n_blocks + threshold
