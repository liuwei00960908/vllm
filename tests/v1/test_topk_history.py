# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.v1.topk_history import (
    TopKHistoryManager,
    kmeans_select_representatives,
    logical_indices_to_physical_slots,
)


def test_logical_indices_to_physical_slots():
    block_ids = [10, 11]
    block_size = 16
    # token 0 -> 10*16+0 = 160, token 17 -> block 1 offset 1 -> 11*16+1 = 177
    slots = logical_indices_to_physical_slots(block_ids, block_size, [0, 17, 17])
    assert slots.tolist() == [160, 177]


def test_topk_manager_prefill_respects_prefix_tail():
    mgr = TopKHistoryManager(
        topk_clusters=32,
        prefix_keep=4,
        tail_keep=3,
    )
    seq_len = 100
    idx = mgr.plan_prefill_logical_indices(seq_len)
    assert 0 in idx and 1 in idx and 2 in idx and 3 in idx
    assert 97 in idx and 98 in idx and 99 in idx
    assert len(idx) <= 32


def test_kmeans_includes_forced():
    torch.manual_seed(0)
    keys = torch.randn(20, 8)
    forced = torch.tensor([0, 5, 10], dtype=torch.long)
    out = kmeans_select_representatives(keys, k=5, forced_indices=forced)
    assert set(forced.tolist()).issubset(set(out))
