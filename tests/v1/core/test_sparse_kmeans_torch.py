# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.core.sparse_kmeans_torch import (
    prefill_cluster_meta_from_features_torch,
    prefill_cluster_meta_from_features_torch_batched,
    segment_kmeans_centered_torch,
)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_prefill_cluster_meta_batched_matches_per_head(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    dev = torch.device(device)
    torch.manual_seed(0)
    h, n, d = 4, 128, 32
    feat = torch.randn(h, n, d, device=dev, dtype=torch.float32)
    num_clusters = 24
    n_segment = 4

    batched = prefill_cluster_meta_from_features_torch(
        feat, num_clusters=num_clusters, n_segment=n_segment, seed=123
    )
    for i in range(h):
        one = prefill_cluster_meta_from_features_torch(
            feat[i], num_clusters=num_clusters, n_segment=n_segment, seed=123
        )
        torch.testing.assert_close(
            batched["cluster_centres"][i], one["cluster_centres"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            batched["block_to_cluster"][i].to(torch.int64),
            one["block_to_cluster"].to(torch.int64),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            batched["cluster_size"][i].to(torch.float32),
            one["cluster_size"].to(torch.float32),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            batched["mean_key"][i], one["mean_key"], rtol=0, atol=0
        )


def test_prefill_cluster_meta_preserves_input_rank() -> None:
    torch.manual_seed(1)
    feat = torch.randn(64, 16, dtype=torch.float32)
    args = dict(num_clusters=12, n_segment=3, seed=99)
    single = prefill_cluster_meta_from_features_torch(feat, **args)
    batched = prefill_cluster_meta_from_features_torch(feat.unsqueeze(0), **args)
    torch.testing.assert_close(single["cluster_centres"], batched["cluster_centres"][0])
    torch.testing.assert_close(
        single["block_to_cluster"].to(torch.int64),
        batched["block_to_cluster"][0].to(torch.int64),
    )
    torch.testing.assert_close(
        single["cluster_size"].to(torch.float32),
        batched["cluster_size"][0].to(torch.float32),
    )
    torch.testing.assert_close(single["mean_key"], batched["mean_key"][0])


def test_compat_batched_wrapper_matches_unified_prefill() -> None:
    torch.manual_seed(2)
    feat = torch.randn(3, 32, 8, dtype=torch.float32)
    args = dict(num_clusters=8, n_segment=2, seed=11)
    unified = prefill_cluster_meta_from_features_torch(feat, **args)
    wrapper = prefill_cluster_meta_from_features_torch_batched(feat, **args)
    for key in unified:
        torch.testing.assert_close(unified[key], wrapper[key])


def test_segment_kmeans_centered_preserves_input_rank() -> None:
    torch.manual_seed(3)
    feat = torch.randn(2, 48, 12, dtype=torch.float32)
    centered = feat - feat.mean(dim=1, keepdim=True)
    batched = segment_kmeans_centered_torch(
        centered, n_clusters=10, n_segments=2, seed=7
    )
    single = segment_kmeans_centered_torch(
        centered[0], n_clusters=10, n_segments=2, seed=7
    )
    torch.testing.assert_close(batched[0][0], single[0])
    torch.testing.assert_close(
        batched[1][0].to(torch.int64), single[1].to(torch.int64)
    )
    torch.testing.assert_close(
        batched[2][0].to(torch.float32), single[2].to(torch.float32)
    )
