# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DSA KV-organization composite env gates (replay V1).

Covers default values, the dependency chain (unbundle -> two-groups ->
shared-pool) and the reverse-dependency guarantee: enabling a downstream
variable alone must remain a no-op.
"""

import os
from contextlib import contextmanager

import pytest

from vllm.v1.kv_cache_interface import (
    dsa_shared_pool_enabled,
    dsa_shrink_stage,
    dsa_two_groups_enabled,
    dsa_unbundle_enabled,
)


@contextmanager
def _dsa_env(**kwargs: str | None):
    """Temporarily set/unset the DSA env variables."""
    names = (
        "VLLM_ASCEND_DSA_UNBUNDLE",
        "VLLM_ASCEND_DSA_TWO_GROUPS",
        "VLLM_ASCEND_DSA_SHARED_POOL",
        "VLLM_ASCEND_DSA_SHRINK_LATENT",
    )
    saved = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            value = kwargs.get(name)
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, old in saved.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


def test_defaults_all_off():
    with _dsa_env():
        assert not dsa_unbundle_enabled()
        assert not dsa_two_groups_enabled()
        assert not dsa_shared_pool_enabled()


def test_unbundle_alone():
    with _dsa_env(VLLM_ASCEND_DSA_UNBUNDLE="1"):
        assert dsa_unbundle_enabled()
        assert not dsa_two_groups_enabled()
        assert not dsa_shared_pool_enabled()


def test_two_groups_requires_unbundle():
    # Reverse dependency: TWO_GROUPS=1 without UNBUNDLE must stay a no-op.
    with _dsa_env(VLLM_ASCEND_DSA_TWO_GROUPS="1"):
        assert not dsa_unbundle_enabled()
        assert not dsa_two_groups_enabled()
        assert not dsa_shared_pool_enabled()


def test_two_groups_full_chain():
    with _dsa_env(
        VLLM_ASCEND_DSA_UNBUNDLE="1",
        VLLM_ASCEND_DSA_TWO_GROUPS="1",
    ):
        assert dsa_unbundle_enabled()
        assert dsa_two_groups_enabled()
        # shared pool raw default is "1": effective once two-groups is on.
        assert dsa_shared_pool_enabled()


def test_shared_pool_requires_two_groups():
    # Raw default "1" must be suppressed when two-groups is off.
    with _dsa_env(VLLM_ASCEND_DSA_UNBUNDLE="1"):
        assert not dsa_shared_pool_enabled()


def test_shared_pool_explicit_disable():
    with _dsa_env(
        VLLM_ASCEND_DSA_UNBUNDLE="1",
        VLLM_ASCEND_DSA_TWO_GROUPS="1",
        VLLM_ASCEND_DSA_SHARED_POOL="0",
    ):
        assert dsa_two_groups_enabled()
        assert not dsa_shared_pool_enabled()


def test_invalid_values_are_off():
    # Non-"1" values are treated as disabled (fork semantics: exact "1" match).
    with _dsa_env(VLLM_ASCEND_DSA_UNBUNDLE="true"):
        assert not dsa_unbundle_enabled()


def test_shrink_defaults_off():
    with _dsa_env():
        assert dsa_shrink_stage() == 0


def test_shrink_requires_two_groups():
    # Reverse dependency: SHRINK_LATENT=2 without the two-group chain is a
    # no-op (fork semantics: shrink operates on the latent group).
    with _dsa_env(VLLM_ASCEND_DSA_SHRINK_LATENT="2"):
        assert dsa_shrink_stage() == 0
    with _dsa_env(VLLM_ASCEND_DSA_UNBUNDLE="1", VLLM_ASCEND_DSA_SHRINK_LATENT="2"):
        assert dsa_shrink_stage() == 0


def test_shrink_stages():
    with _dsa_env(
        VLLM_ASCEND_DSA_UNBUNDLE="1",
        VLLM_ASCEND_DSA_TWO_GROUPS="1",
        VLLM_ASCEND_DSA_SHRINK_LATENT="1",
    ):
        assert dsa_shrink_stage() == 1
    with _dsa_env(
        VLLM_ASCEND_DSA_UNBUNDLE="1",
        VLLM_ASCEND_DSA_TWO_GROUPS="1",
        VLLM_ASCEND_DSA_SHRINK_LATENT="2",
    ):
        assert dsa_shrink_stage() == 2
    with _dsa_env(
        VLLM_ASCEND_DSA_UNBUNDLE="1",
        VLLM_ASCEND_DSA_TWO_GROUPS="1",
        VLLM_ASCEND_DSA_SHRINK_LATENT="3",
    ):
        assert dsa_shrink_stage() == 3


def test_shrink_invalid_value_falls_back_to_zero():
    # Fail-safe: a malformed stage value never enables the release chain.
    with _dsa_env(
        VLLM_ASCEND_DSA_UNBUNDLE="1",
        VLLM_ASCEND_DSA_TWO_GROUPS="1",
        VLLM_ASCEND_DSA_SHRINK_LATENT="yes",
    ):
        assert dsa_shrink_stage() == 0
    with _dsa_env(
        VLLM_ASCEND_DSA_UNBUNDLE="1",
        VLLM_ASCEND_DSA_TWO_GROUPS="1",
        VLLM_ASCEND_DSA_SHRINK_LATENT="",
    ):
        assert dsa_shrink_stage() == 0


if __name__ == "__main__":
    pytest.main([__file__])
