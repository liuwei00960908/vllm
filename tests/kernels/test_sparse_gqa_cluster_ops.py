# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm._custom_ops import (
    group_avg_topk_clusters_by_kv_group_out,
    sparse_select_topk_clusters_out,
    union_topk_clusters_by_kv_group_out,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


def _run_group_avg_topk(
    query: torch.Tensor,
    cluster_centers_t: torch.Tensor,
    mean: torch.Tensor,
    cluster_center_count: torch.Tensor,
    nprobe: int,
) -> torch.Tensor:
    head_scores = torch.empty(
        (query.shape[0], query.shape[1], cluster_centers_t.shape[2]),
        dtype=torch.float32,
        device=query.device,
    )
    out = torch.empty(
        (query.shape[0], cluster_centers_t.shape[0], nprobe),
        dtype=torch.int32,
        device=query.device,
    )
    group_avg_topk_clusters_by_kv_group_out(
        query,
        cluster_centers_t,
        mean,
        cluster_center_count,
        nprobe,
        head_scores,
        out,
    )
    torch.accelerator.synchronize()
    return out


def _ref_group_avg_topk(
    query: torch.Tensor,
    cluster_centers_t: torch.Tensor,
    mean: torch.Tensor,
    valid_centers: int,
    nprobe: int,
) -> torch.Tensor:
    nq, hq, dim = query.shape
    hkv = cluster_centers_t.shape[0]
    c = cluster_centers_t.shape[2]
    q_per_kv = hq // hkv

    centered_query = query.reshape(nq, hkv, q_per_kv, dim) - mean[None, :, None, :]
    raw = torch.einsum("ngqd,gdc->ngqc", centered_query.float(),
                       cluster_centers_t.float())
    raw /= math.sqrt(dim)
    if valid_centers < c:
        raw[..., valid_centers:] = float("-inf")

    probs = torch.softmax(raw, dim=-1)
    grouped_scores = probs.mean(dim=2)
    out = torch.full((nq, hkv, nprobe), -1, dtype=torch.int32, device=query.device)
    emit = min(nprobe, valid_centers)
    if emit > 0:
        out[..., :emit] = torch.topk(grouped_scores, k=emit, dim=-1).indices.to(
            torch.int32
        )
    return out


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_group_avg_topk_matches_per_head_topk_when_q_per_kv_is_one() -> None:
    set_random_seed(0)
    torch.set_default_device("cuda:0")

    nq, hq, hkv, dim, num_clusters, nprobe = 2, 2, 2, 8, 6, 3
    query = torch.randn((nq, hq, dim), dtype=torch.float16)
    cluster_centers_t = torch.randn((hkv, dim, num_clusters), dtype=torch.float16)
    mean = torch.randn((hkv, dim), dtype=torch.float16)
    cluster_center_count = torch.tensor([num_clusters], dtype=torch.int32, device="cuda")

    grouped_topk = _run_group_avg_topk(
        query, cluster_centers_t, mean, cluster_center_count, nprobe
    )
    per_head_topk = torch.empty((nq, hq, nprobe), dtype=torch.int32, device="cuda")
    sparse_select_topk_clusters_out(
        query,
        cluster_centers_t,
        mean,
        cluster_center_count,
        nprobe,
        per_head_topk,
    )

    assert torch.equal(grouped_topk, per_head_topk.reshape(nq, hkv, nprobe))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_group_avg_topk_keeps_grouped_selection_within_nprobe() -> None:
    torch.set_default_device("cuda:0")

    nprobe = 1
    query = torch.tensor(
        [
            [
                [6.0, 0.0, 0.0, 0.0],
                [0.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 3.0],
            ]
        ],
        dtype=torch.float16,
        device="cuda",
    )
    cluster_centers_t = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ],
        dtype=torch.float16,
        device="cuda",
    ).transpose(1, 2).contiguous()
    mean = torch.zeros((2, 4), dtype=torch.float16, device="cuda")
    cluster_center_count = torch.tensor([4], dtype=torch.int32, device="cuda")

    head_topk = torch.empty((1, 4, nprobe), dtype=torch.int32, device="cuda")
    sparse_select_topk_clusters_out(
        query,
        cluster_centers_t,
        mean,
        cluster_center_count,
        nprobe,
        head_topk,
    )
    union_out = torch.empty((1, 2, 2), dtype=torch.int32, device="cuda")
    union_topk_clusters_by_kv_group_out(head_topk, 2, 4, union_out)
    grouped_topk = _run_group_avg_topk(
        query, cluster_centers_t, mean, cluster_center_count, nprobe
    )
    ref = _ref_group_avg_topk(
        query, cluster_centers_t, mean, valid_centers=4, nprobe=nprobe
    )

    union_counts = (union_out >= 0).sum(dim=-1)
    grouped_counts = (grouped_topk >= 0).sum(dim=-1)
    assert int(union_counts[0, 0]) > nprobe
    assert torch.equal(grouped_counts, torch.full_like(grouped_counts, nprobe))
    assert torch.equal(grouped_topk, ref)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_group_avg_topk_masks_invalid_centers() -> None:
    set_random_seed(1)
    torch.set_default_device("cuda:0")

    nq, hq, hkv, dim, num_clusters, nprobe = 1, 4, 2, 8, 5, 4
    valid_centers = 2
    query = torch.randn((nq, hq, dim), dtype=torch.bfloat16)
    cluster_centers_t = torch.randn((hkv, dim, num_clusters), dtype=torch.bfloat16)
    mean = torch.randn((hkv, dim), dtype=torch.bfloat16)
    cluster_center_count = torch.tensor([valid_centers], dtype=torch.int32, device="cuda")

    grouped_topk = _run_group_avg_topk(
        query, cluster_centers_t, mean, cluster_center_count, nprobe
    )
    ref = _ref_group_avg_topk(
        query,
        cluster_centers_t,
        mean,
        valid_centers=valid_centers,
        nprobe=nprobe,
    )

    assert torch.equal(grouped_topk, ref)
    assert torch.all((grouped_topk[..., :valid_centers] >= 0)
                     & (grouped_topk[..., :valid_centers] < valid_centers))
    assert torch.all(grouped_topk[..., valid_centers:] == -1)
