# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DSALatentManager (replay B1b).

Covers the release-chain semantics of the shrink-latent latent-group
manager: release waits for committed receipts only (progress-based removal
stays a no-op), scratch prefix survives every release, releases are
append-only/idempotent, and bundle alignment rounds the release start.
"""

import os
from contextlib import contextmanager

import pytest
import torch

from vllm.v1.core.kv_cache_interface import MLAAttentionSpec
from vllm.v1.core.single_type_kv_cache_manager import (
    DSALatentManager,
    FullAttentionManager,
    get_manager_for_kv_cache_spec,
)


@contextmanager
def _dsa_env(stage: str | None):
    names = (
        "VLLM_ASCEND_DSA_UNBUNDLE",
        "VLLM_ASCEND_DSA_TWO_GROUPS",
        "VLLM_ASCEND_DSA_SHRINK_LATENT",
    )
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ["VLLM_ASCEND_DSA_UNBUNDLE"] = "1"
        os.environ["VLLM_ASCEND_DSA_TWO_GROUPS"] = "1"
        if stage is None:
            os.environ.pop("VLLM_ASCEND_DSA_SHRINK_LATENT", None)
        else:
            os.environ["VLLM_ASCEND_DSA_SHRINK_LATENT"] = stage
        yield
    finally:
        for name, old in saved.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


class _FakeBlock:
    def __init__(self, block_id: int):
        self.block_id = block_id


class _FakeFreeQueue:
    def popleft_n(self, n):
        return []


class _FakeBlockPool:
    """Minimal block pool: records freed blocks, optional bundle width."""

    def __init__(self, blocks_per_bundle: int = 1):
        self.blocks_per_bundle = blocks_per_bundle
        self.null_block = _FakeBlock(-1)
        self.freed: list = []
        self.free_block_queue = _FakeFreeQueue()

    def free_blocks(self, blocks):
        self.freed.extend(blocks)

    def get_new_blocks(self, n):
        return [_FakeBlock(i) for i in range(n)]


def _manager(block_size: int = 128, scratch_blocks: int = 16,
             blocks_per_bundle: int = 1) -> tuple[DSALatentManager, _FakeBlockPool]:
    pool = _FakeBlockPool(blocks_per_bundle)
    spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
    )
    manager = DSALatentManager(spec, block_pool=pool, kv_cache_group_id=0)
    manager.scratch_blocks = scratch_blocks
    return manager, pool


def _alloc(manager, request_id: str, num_blocks: int):
    """Fake a resident request block table of real blocks."""
    manager.req_to_blocks[request_id] = [
        _FakeBlock(i) for i in range(num_blocks)
    ]


def test_release_waits_for_committed_receipt():
    manager, pool = _manager()
    _alloc(manager, "r1", 100)
    # Progress-based removal must be a no-op even for a fully computed prefix.
    manager.remove_skipped_blocks("r1", total_computed_tokens=12800)
    assert pool.freed == []
    assert len([b for b in manager.req_to_blocks["r1"] if b != manager._null_block]) == 100


def test_release_respects_scratch_prefix():
    manager, pool = _manager(scratch_blocks=16)
    _alloc(manager, "r1", 100)
    freed = manager.remove_saved_decode_window_blocks("r1", committed_end=100 * 128)
    assert freed == 100 - 16
    blocks = manager.req_to_blocks["r1"]
    # Scratch prefix intact, everything above it is a null hole.
    assert all(b != manager._null_block for b in blocks[:16])
    assert all(b == manager._null_block for b in blocks[16:])


def test_release_is_append_only_and_idempotent():
    manager, pool = _manager(scratch_blocks=16)
    _alloc(manager, "r1", 100)
    first = manager.remove_saved_decode_window_blocks("r1", 50 * 128)
    second = manager.remove_saved_decode_window_blocks("r1", 50 * 128)  # replay
    later = manager.remove_saved_decode_window_blocks("r1", 80 * 128)   # grows
    assert (first, second, later) == (34, 0, 30)
    assert len(pool.freed) == 64  # 50-block frontier + 30 more, no double free


def test_short_prompt_release_inert():
    # committed_end below the scratch prefix frees nothing by design.
    manager, pool = _manager(scratch_blocks=16)
    _alloc(manager, "r1", 10)
    assert manager.remove_saved_decode_window_blocks("r1", 10 * 128) == 0
    assert pool.freed == []


def test_non_block_aligned_committed_end_raises():
    manager, _ = _manager(scratch_blocks=16)
    _alloc(manager, "r1", 100)
    with pytest.raises(ValueError):
        manager.remove_saved_decode_window_blocks("r1", 100)


def test_bundle_alignment_rounds_release_start():
    # 2 blocks/bundle: scratch 16 stays 16 (already aligned); a scratch of
    # 15 rounds UP to 16 (never release into the scratch bundle).
    manager, _ = _manager(scratch_blocks=15, blocks_per_bundle=2)
    _alloc(manager, "r1", 32)
    freed = manager.remove_saved_decode_window_blocks("r1", 32 * 128)
    assert freed == 32 - 16
    assert manager.req_to_blocks["r1"][15] != manager._null_block


def test_unknown_request_is_safe_noop():
    manager, pool = _manager()
    assert manager.remove_saved_decode_window_blocks("ghost", 4096) == 0
    assert pool.freed == []


def test_factory_selects_dsa_latent_manager():
    spec = MLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=576,
        dtype=torch.bfloat16,
    )
    with _dsa_env("2"):
        manager = get_manager_for_kv_cache_spec(
            spec, max_num_batched_tokens=4096, max_model_len=10000,
            block_pool=_FakeBlockPool(), kv_cache_group_id=0,
        )
        assert isinstance(manager, DSALatentManager)
        assert isinstance(manager, FullAttentionManager)


def test_factory_off_paths_stay_official():
    spec = MLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=576,
        dtype=torch.bfloat16,
    )
    indexer_spec = MLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=128,
        dtype=torch.bfloat16,
    )
    with _dsa_env("0"):
        m = get_manager_for_kv_cache_spec(
            spec, max_num_batched_tokens=4096, max_model_len=10000,
            block_pool=_FakeBlockPool(), kv_cache_group_id=0,
        )
        assert type(m) is FullAttentionManager
    with _dsa_env("1"):  # stage 1: read path only, no release manager swap
        m = get_manager_for_kv_cache_spec(
            spec, max_num_batched_tokens=4096, max_model_len=10000,
            block_pool=_FakeBlockPool(), kv_cache_group_id=0,
        )
        assert type(m) is FullAttentionManager
    with _dsa_env("2"):  # indexer head stays plain
        m = get_manager_for_kv_cache_spec(
            indexer_spec, max_num_batched_tokens=4096, max_model_len=10000,
            block_pool=_FakeBlockPool(), kv_cache_group_id=1,
        )
        assert type(m) is FullAttentionManager


if __name__ == "__main__":
    pytest.main([__file__])
