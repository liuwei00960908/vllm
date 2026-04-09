# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.core.sparse_kmeans_torch import (
    prefill_cluster_meta_from_features_torch,
    prefill_cluster_meta_from_features_torch_batched,
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

    batched = prefill_cluster_meta_from_features_torch_batched(
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
        torch.testing.assert_close(batched["mean_key"][i], one["mean_key"], rtol=0, atol=0)


def test_prefill_cluster_meta_single_delegates_to_batched() -> None:
    torch.manual_seed(1)
    feat = torch.randn(64, 16, dtype=torch.float32)
    args = dict(num_clusters=12, n_segment=3, seed=99)
    direct = prefill_cluster_meta_from_features_torch(feat, **args)
    via = prefill_cluster_meta_from_features_torch_batched(feat.unsqueeze(0), **args)
    torch.testing.assert_close(direct["cluster_centres"], via["cluster_centres"][0])
    torch.testing.assert_close(
        direct["block_to_cluster"].to(torch.int64),
        via["block_to_cluster"][0].to(torch.int64),
    )
    torch.testing.assert_close(
        direct["cluster_size"].to(torch.float32),
        via["cluster_size"][0].to(torch.float32),
    )
    torch.testing.assert_close(direct["mean_key"], via["mean_key"][0])
