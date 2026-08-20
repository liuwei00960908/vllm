# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DSA shrink config plumbing (replay B1a).

Covers `_dsa_shrink_config_kwargs`: the KVCacheConfig scratch-sizing inputs
(dsa_index_topk / dsa_num_speculative_tokens) are only populated for
two-group DSA models whose hf config exposes index_topk; everything else
keeps the inert defaults ({} -> None/0).
"""

import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from vllm.v1.core.kv_cache_utils import _dsa_shrink_config_kwargs


@contextmanager
def _dsa_env(**kwargs: str | None):
    names = (
        "VLLM_ASCEND_DSA_UNBUNDLE",
        "VLLM_ASCEND_DSA_TWO_GROUPS",
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


def _vllm_config(index_topk=None, num_speculative_tokens=None):
    hf_text_config = (
        None if index_topk is None else SimpleNamespace(index_topk=index_topk)
    )
    model_config = SimpleNamespace(hf_text_config=hf_text_config)
    speculative_config = (
        None
        if num_speculative_tokens is None
        else SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
    )
    return SimpleNamespace(
        model_config=model_config,
        speculative_config=speculative_config,
    )


def test_inert_when_two_groups_off():
    with _dsa_env():
        assert _dsa_shrink_config_kwargs(_vllm_config(index_topk=2048)) == {}


def test_inert_when_index_topk_missing():
    with _dsa_env(
        VLLM_ASCEND_DSA_UNBUNDLE="1", VLLM_ASCEND_DSA_TWO_GROUPS="1"
    ):
        assert _dsa_shrink_config_kwargs(_vllm_config(index_topk=None)) == {}
        # hf_text_config entirely absent (non-MLA model).
        cfg = SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=None),
            speculative_config=None,
        )
        assert _dsa_shrink_config_kwargs(cfg) == {}


def test_kwargs_for_glm51_mtp_off():
    with _dsa_env(
        VLLM_ASCEND_DSA_UNBUNDLE="1", VLLM_ASCEND_DSA_TWO_GROUPS="1"
    ):
        kwargs = _dsa_shrink_config_kwargs(_vllm_config(index_topk=2048))
        assert kwargs == {
            "dsa_index_topk": 2048,
            "dsa_num_speculative_tokens": 0,
        }


def test_kwargs_with_speculative_tokens():
    with _dsa_env(
        VLLM_ASCEND_DSA_UNBUNDLE="1", VLLM_ASCEND_DSA_TWO_GROUPS="1"
    ):
        kwargs = _dsa_shrink_config_kwargs(
            _vllm_config(index_topk=2048, num_speculative_tokens=1)
        )
        assert kwargs["dsa_index_topk"] == 2048
        assert kwargs["dsa_num_speculative_tokens"] == 1


def test_two_groups_without_unbundle_stays_inert():
    # Reverse dependency: TWO_GROUPS alone must not populate shrink kwargs.
    with _dsa_env(VLLM_ASCEND_DSA_TWO_GROUPS="1"):
        assert _dsa_shrink_config_kwargs(_vllm_config(index_topk=2048)) == {}


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
