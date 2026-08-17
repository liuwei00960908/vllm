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
# Unit tests for the DSA two-groups replay slice (replay Step 3 / 3a-3d):
# grouping, per-group capacity, cross-worker reconcile and memory planning.
# All DSA branches are env-gated; with the env unset the official behavior
# is exercised (existing upstream tests cover that path).
#

import os
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_config_from_groups,
    get_kv_cache_configs,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

_DSA_ENVS = (
    "VLLM_ASCEND_DSA_UNBUNDLE",
    "VLLM_ASCEND_DSA_TWO_GROUPS",
    "VLLM_ASCEND_DSA_SHARED_POOL",
    "VLLM_ASCEND_DSA_SHRINK_LATENT",
)


@contextmanager
def _dsa_two_groups_env():
    saved = {name: os.environ.get(name) for name in _DSA_ENVS}
    os.environ["VLLM_ASCEND_DSA_UNBUNDLE"] = "1"
    os.environ["VLLM_ASCEND_DSA_TWO_GROUPS"] = "1"
    os.environ["VLLM_ASCEND_DSA_SHARED_POOL"] = "0"
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
    """Stand-in for AscendMLAAttentionSpec (sparse_head_dim) without pulling
    in the vllm-ascend plugin in core unit tests."""

    sparse_head_dim: tuple[int, ...] | None = None


def _make_spec(head_size: int, sparse_head_dim):
    return _DSAMLASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=head_size,
        sparse_head_dim=sparse_head_dim,
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
    )


def _make_specs(num_latent: int, num_indexer: int) -> dict[str, KVCacheSpec]:
    latent_spec = _make_spec(576, (512, 64))
    indexer_spec = _make_spec(128, (128,))
    specs = {}
    for i in range(num_latent):
        specs[f"model.layers.{i}.self_attn.attn"] = latent_spec
    for i in range(num_indexer):
        specs[f"model.layers.{i}.self_attn.indexer.k_cache"] = indexer_spec
    return specs


def _vllm_config(max_model_len: int = 10000) -> VllmConfig:
    return VllmConfig(
        model_config=SimpleNamespace(
            max_model_len=max_model_len,
            original_max_model_len=max_model_len,
        ),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
            max_num_batched_tokens=4096,
            max_model_len=max_model_len,
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None,
            gpu_memory_utilization=0.9,
            block_size=128,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )


def _check_two_groups(groups):
    assert len(groups) == 2
    assert (
        groups[0].kv_cache_spec.page_size_bytes > groups[1].kv_cache_spec.page_size_bytes
    )
    assert all(
        isinstance(g.kv_cache_spec, MLAAttentionSpec) for g in groups
    )
    assert not isinstance(groups[0].kv_cache_spec, UniformTypeKVCacheSpecs)


def test_dsa_two_groups_splits_glm51_geometry():
    with _dsa_two_groups_env():
        specs = _make_specs(78, 78)
        groups = get_kv_cache_groups(_vllm_config(), specs)
    _check_two_groups(groups)
    assert len(groups[0].layer_names) == 78  # latent first
    assert len(groups[1].layer_names) == 78  # indexer
    assert set(groups[0].layer_names + groups[1].layer_names) == set(specs.keys())


def test_dsa_two_groups_splits_glm52_geometry():
    # GLM-5.2 unbundle geometry: 79 latent + 22 indexer (MTP1) still forms
    # exactly two groups.
    with _dsa_two_groups_env():
        specs = _make_specs(79, 22)
        groups = get_kv_cache_groups(_vllm_config(), specs)
    _check_two_groups(groups)
    assert len(groups[0].layer_names) == 79
    assert len(groups[1].layer_names) == 22


def test_dsa_two_groups_requires_exactly_two_layouts():
    with _dsa_two_groups_env():
        specs = _make_specs(78, 78)
        specs["extra.latent.alt"] = _make_spec(192, (128, 64))
        with pytest.raises(ValueError):
            get_kv_cache_groups(_vllm_config(), specs)


def test_dsa_two_groups_rejects_non_mla():
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    with _dsa_two_groups_env():
        specs = _make_specs(78, 78)
        specs["other.attn"] = FullAttentionSpec(
            block_size=128,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        with pytest.raises(ValueError):
            get_kv_cache_groups(_vllm_config(), specs)


def test_dsa_two_groups_config_per_group():
    with _dsa_two_groups_env():
        specs = _make_specs(78, 78)
        groups = get_kv_cache_groups(_vllm_config(), specs)
        available = 10 * 2**30  # 10 GiB
        config = get_kv_cache_config_from_groups(_vllm_config(), groups, available)

    assert config.num_blocks_per_group == [config.num_blocks, config.num_blocks]
    assert len(config.kv_cache_tensors) == 156
    # num_blocks = available // (78*147456 + 78*32768)
    per_block = 78 * 147456 + 78 * 32768
    assert config.num_blocks == available // per_block
    assert all(
        isinstance(t, KVCacheTensor) for t in config.kv_cache_tensors
    )
    assert sum(t.size for t in config.kv_cache_tensors) <= available
    # Every layer has its own tensor sized to its own page.
    latent_tensor_size = config.kv_cache_tensors[0].size
    assert latent_tensor_size == config.num_blocks * 147456


def test_dsa_two_groups_configs_reconcile_per_group():
    with _dsa_two_groups_env():
        specs = _make_specs(78, 78)
        vllm_config = _vllm_config()
        groups = get_kv_cache_groups(vllm_config, specs)
        # Two workers with different available memory.
        configs = get_kv_cache_configs(
            vllm_config,
            [specs, specs],
            [10 * 2**30, 8 * 2**30],
        )
    assert len(configs) == 2
    assert configs[0].num_blocks_per_group == configs[1].num_blocks_per_group
    # The smaller worker wins: 8 GiB budget.
    per_block = 78 * 147456 + 78 * 32768
    expected_min = (8 * 2**30) // per_block
    assert configs[1].num_blocks_per_group == [expected_min, expected_min]
    assert configs[0].num_blocks == max(configs[0].num_blocks_per_group)
    # Tensor divisibility must survive the shrink (the original scalar-min
    # reconcile would break on per-group sizing).
    for config in configs:
        for tensor in config.kv_cache_tensors:
            assert tensor.size % 147456 == 0 or tensor.size % 32768 == 0


def test_dsa_two_groups_full_core_chain():
    # Core call chain: registry check -> configs -> scheduler config ->
    # coordinator (no-prefix-caching branch). DSA two groups must survive
    # every stage; the runtime block allocation (coordinator.allocate_slots)
    # is exercised end-to-end in Step 4 on the NPU machine.
    with _dsa_two_groups_env():
        vllm_config = _vllm_config()
        specs = _make_specs(78, 78)
        configs = get_kv_cache_configs(vllm_config, [specs], [10 * 2**30])
        scheduler_config = generate_scheduler_kv_cache_config(configs)
        assert scheduler_config.num_blocks == configs[0].num_blocks

        coordinator = get_kv_cache_coordinator(
            kv_cache_config=scheduler_config,
            max_model_len=vllm_config.model_config.max_model_len,
            max_num_batched_tokens=4096,
            use_eagle=False,
            enable_caching=False,
            enable_kv_cache_events=False,
            dcp_world_size=1,
            pcp_world_size=1,
            scheduler_block_size=128,
            hash_block_size=128,
        )
        assert coordinator is not None
