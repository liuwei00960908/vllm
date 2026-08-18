#
# Copyright (c) 2025 The vLLM team.
# This file is a part of the vllm project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unit tests for the DSA shared bundle pool (DSA replay Step 5 / 5a).
# Geometry/allocator cases ported from the fork's
# cpu_tests/test_dsa_shared_pool.py (vllm-dsa-two-groups@4575d8a12);
# shared config / reconcile cases added for the replay slice.
#

import os
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.v1.core.dsa_shared_pool import (
    DSASharedBlockLayout,
    DSASharedBlockOwner,
    DSASharedBundleAllocator,
)
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_configs,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

_DSA_ENVS = (
    "VLLM_ASCEND_DSA_UNBUNDLE",
    "VLLM_ASCEND_DSA_TWO_GROUPS",
    "VLLM_ASCEND_DSA_SHARED_POOL",
    "VLLM_ASCEND_DSA_SHRINK_LATENT",
)


@contextmanager
def _dsa_shared_env():
    saved = {name: os.environ.get(name) for name in _DSA_ENVS}
    os.environ["VLLM_ASCEND_DSA_UNBUNDLE"] = "1"
    os.environ["VLLM_ASCEND_DSA_TWO_GROUPS"] = "1"
    os.environ["VLLM_ASCEND_DSA_SHARED_POOL"] = "1"
    os.environ["VLLM_ASCEND_DSA_SHRINK_LATENT"] = "0"
    try:
        yield
    finally:
        for name, old in saved.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


@dataclass(frozen=True, kw_only=True)
class _DSAMLASpec(MLAAttentionSpec):
    sparse_head_dim: tuple[int, ...] | None = None


def _make_specs(num_latent: int, num_indexer: int) -> dict:
    latent_spec = _DSAMLASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        sparse_head_dim=(512, 64),
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
    )
    indexer_spec = _DSAMLASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        sparse_head_dim=(128,),
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
    )
    specs = {}
    for i in range(num_latent):
        specs[f"model.layers.{i}.self_attn.attn"] = latent_spec
    for i in range(num_indexer):
        specs[f"model.layers.{i}.self_attn.indexer.k_cache"] = indexer_spec
    return specs


def _vllm_config() -> VllmConfig:
    return VllmConfig(model_config=ModelConfig(max_model_len=10000))


# ---------------------------------------------------------------------------
# Geometry / allocator (ported from fork cpu_tests/test_dsa_shared_pool.py)
# ---------------------------------------------------------------------------


def test_layout_uses_128_dim_base_pages():
    layout = DSASharedBlockLayout(
        latent_page_size_bytes=147456,
        indexer_page_size_bytes=32768,
        capacity_bundles=8,
    )
    assert layout.bundle_page_size_bytes == 294912
    assert layout.latent_blocks_per_bundle == 2
    assert layout.indexer_blocks_per_bundle == 9
    assert layout.nope_pages_per_bundle == 8
    assert layout.pe_pages_per_bundle == 1


def test_layout_maps_latent_and_indexer_blocks_to_same_bundle():
    layout = DSASharedBlockLayout(
        latent_page_size_bytes=147456,
        indexer_page_size_bytes=32768,
        capacity_bundles=8,
    )
    for bundle_id in range(1, layout.capacity_bundles + 1):
        latent_ids = layout.block_ids_for_bundle(
            DSASharedBlockOwner.LATENT, bundle_id
        )
        indexer_ids = layout.block_ids_for_bundle(
            DSASharedBlockOwner.INDEXER, bundle_id
        )
        # latent: [2b, 2b+1]; indexer: nope 8 pages + pe 1 page
        assert latent_ids == (bundle_id * 2, bundle_id * 2 + 1)
        assert len(indexer_ids) == 9


def test_bundle_id_reverse_mapping():
    layout = DSASharedBlockLayout(
        latent_page_size_bytes=147456,
        indexer_page_size_bytes=32768,
        capacity_bundles=8,
    )
    assert (
        layout.bundle_id_for_block(DSASharedBlockOwner.LATENT, 7) == 3
    )
    assert (
        layout.bundle_id_for_block(DSASharedBlockOwner.INDEXER, 50) == 6
    )


def test_allocator_reuses_latent_freed_bundles_for_indexer():
    allocator = DSASharedBundleAllocator(
        DSASharedBlockLayout(147456, 32768, capacity_bundles=8)
    )
    bundles = allocator.allocate(DSASharedBlockOwner.LATENT, 4)
    assert bundles == (1, 2, 3, 4)
    allocator.free(DSASharedBlockOwner.LATENT, (2, 3))
    # The freed bundles are immediately claimable by the other owner.
    reused = allocator.allocate(DSASharedBlockOwner.INDEXER, 2)
    assert set(reused) == {2, 3}


def test_allocator_prefers_smallest_contiguous_range():
    allocator = DSASharedBundleAllocator(
        DSASharedBlockLayout(147456, 32768, capacity_bundles=12)
    )
    allocator.allocate(DSASharedBlockOwner.LATENT, 3)  # 1..3
    allocator.free(DSASharedBlockOwner.LATENT, (2,))  # free {2}, ranges {2}, {4..12}
    allocator.allocate(DSASharedBlockOwner.LATENT, 2)  # best-fit should NOT use {2}
    # After taking 4..5, bundle 2 remains free for a 1-bundle allocation.
    single = allocator.allocate(DSASharedBlockOwner.INDEXER, 1)
    assert single == (2,)


def test_allocator_rejects_wrong_owner_and_duplicate_free():
    allocator = DSASharedBundleAllocator(
        DSASharedBlockLayout(147456, 32768, capacity_bundles=4)
    )
    allocator.allocate(DSASharedBlockOwner.LATENT, 2)
    with pytest.raises(ValueError):
        allocator.free(DSASharedBlockOwner.INDEXER, (1,))
    with pytest.raises(ValueError):
        allocator.free(DSASharedBlockOwner.LATENT, (1, 1))


def test_bundle_count_rounds_up():
    allocator = DSASharedBundleAllocator(
        DSASharedBlockLayout(147456, 32768, capacity_bundles=4)
    )
    assert allocator.bundle_count_for_blocks(DSASharedBlockOwner.LATENT, 3) == 2
    assert allocator.bundle_count_for_blocks(DSASharedBlockOwner.INDEXER, 9) == 1
    assert allocator.bundle_count_for_blocks(DSASharedBlockOwner.INDEXER, 10) == 2


# ---------------------------------------------------------------------------
# Shared config / reconcile (replay-slice additions)
# ---------------------------------------------------------------------------


def test_shared_config_pairs_and_sizes():
    with _dsa_shared_env():
        specs = _make_specs(78, 78)
        groups = get_kv_cache_groups(_vllm_config(), specs)
        available = 10 * 2**30
        config = get_kv_cache_config_from_groups(_vllm_config(), groups, available)

    bundle_page = 294912
    expected_bundles = available // (78 * bundle_page) - 1
    assert config.num_blocks == expected_bundles
    assert config.num_blocks_per_group is None  # shared mode: single scalar
    assert len(config.kv_cache_tensors) == 78  # one per pair, NOT per layer
    tensor_size = (expected_bundles + 1) * bundle_page
    for tensor in config.kv_cache_tensors:
        assert tensor.size == tensor_size
        assert len(tensor.shared_by) == 2  # latent + indexer share the slab
        assert any("indexer" in n for n in tensor.shared_by)
        assert any("indexer" not in n for n in tensor.shared_by)
    assert sum(t.size for t in config.kv_cache_tensors) <= available


def test_shared_reconcile_single_rank_no_shrink():
    with _dsa_shared_env():
        specs = _make_specs(78, 78)
        configs = get_kv_cache_configs(_vllm_config(), [specs], [10 * 2**30])
    bundle_page = 294912
    tensor_size = configs[0].num_blocks * bundle_page  # after (nb+1) shrink? no-op
    # Single rank: nothing shrinks; tensor remains (nb+1) * bundle_page.
    assert configs[0].kv_cache_tensors[0].size == (configs[0].num_blocks + 1) * bundle_page


def test_shared_reconcile_two_ranks_proportional():
    with _dsa_shared_env():
        specs = _make_specs(78, 78)
        configs = get_kv_cache_configs(
            _vllm_config(), [specs, specs], [10 * 2**30, 8 * 2**30]
        )
    bundle_page = 294912
    big, small = sorted(configs, key=lambda c: c.num_blocks, reverse=True)
    assert big.num_blocks > small.num_blocks
    # Every config ends at the SAME min bundle count; tensors shrank by (old+1)->(new+1).
    assert big.num_blocks == small.num_blocks
    expected_size = (small.num_blocks + 1) * bundle_page
    for config in (big, small):
        for tensor in config.kv_cache_tensors:
            assert tensor.size == expected_size
