# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU (or device) segment K-Means for sparse KV prefill indexing.

Mirrors ``sparse_kv_cache_manager._segment_kmeans`` on centered features:
Lloyd iterations with dot-product assignment (equivalent to L2 K-Means on
mean-centered vectors).  Intended to run in the model-runner process while
features still live on device, avoiding large CPU matmuls.
"""

from __future__ import annotations

import os

import torch

_PREFILL_CLUSTER_DEVICE = os.environ.get(
    "VLLM_SPARSE_PREFILL_CLUSTER_DEVICE", "auto"
).lower()


def sparse_prefill_cluster_use_device_kmeans(feat: torch.Tensor) -> bool:
    """Whether to cluster ``feat`` on its current device (typically CUDA)."""
    if _PREFILL_CLUSTER_DEVICE == "cpu":
        return False
    if _PREFILL_CLUSTER_DEVICE == "cuda":
        return feat.is_cuda
    # auto
    return feat.is_cuda


def _kmeans_dot_torch(
    features: torch.Tensor,
    k: int,
    n_iter: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd K-Means on **mean-centered** features (dot-product assignment)."""
    n, d = features.shape
    k = min(int(k), int(n))
    if k == 0:
        zc = features.new_zeros((0, d))
        zl = torch.zeros(n, dtype=torch.int64, device=features.device)
        return zc, zl

    g = torch.Generator(device=features.device)
    g.manual_seed(int(seed))
    perm = torch.randperm(n, generator=g, device=features.device)[:k]
    centres = features[perm].to(dtype=torch.float32).clone()
    feat = features.to(dtype=torch.float32)

    labels = torch.zeros(n, dtype=torch.int64, device=features.device)
    for _ in range(n_iter):
        sims = feat @ centres.T
        labels = sims.argmax(dim=1)
        new_centres = torch.zeros_like(centres)
        counts = torch.bincount(labels, minlength=k).to(dtype=torch.float32)
        new_centres.index_add_(0, labels, feat)
        has = counts > 0
        new_centres[has] /= counts[has].unsqueeze(1)
        new_centres[~has] = centres[~has]
        centres = new_centres
    return centres, labels


def segment_kmeans_centered_torch(
    features_centered: torch.Tensor,
    n_clusters: int,
    n_segments: int,
    n_iter: int = 15,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Segment K-Means on **already mean-centered** features.

    Returns:
        centres_c: ``[total_k, D]`` float32 in centered space.
        labels:    ``[N]`` int64 global cluster ids.
        sizes:     ``[total_k]`` int32 cluster sizes.
    """
    n, d = features_centered.shape
    if n == 0:
        zc = features_centered.new_zeros((0, d))
        zl = torch.zeros(0, dtype=torch.int64, device=features_centered.device)
        zs = torch.zeros(0, dtype=torch.int32, device=features_centered.device)
        return zc, zl, zs

    n_seg = min(int(n_segments), int(n))
    k_per_seg = max(1, int(n_clusters) // n_seg)

    seg_starts = [i * (n // n_seg) for i in range(n_seg)]
    seg_ends = seg_starts[1:] + [n]

    all_centres: list[torch.Tensor] = []
    labels = torch.zeros(n, dtype=torch.int64, device=features_centered.device)
    cluster_offset = 0

    fc = features_centered.to(dtype=torch.float32)
    for seg_idx, (start, end) in enumerate(zip(seg_starts, seg_ends, strict=False)):
        seg_feat = fc[start:end]
        seg_n = seg_feat.shape[0]
        k = min(k_per_seg, seg_n)
        if k == 0:
            continue
        centres_seg, labels_seg = _kmeans_dot_torch(
            seg_feat, k, n_iter=n_iter, seed=seed + seg_idx
        )
        all_centres.append(centres_seg)
        labels[start:end] = labels_seg + cluster_offset
        cluster_offset += k

    if not all_centres:
        zc = fc.new_zeros((0, d))
        zs = torch.zeros(0, dtype=torch.int32, device=features_centered.device)
        return zc, labels, zs

    all_c = torch.cat(all_centres, dim=0)
    sizes = torch.bincount(labels, minlength=all_c.shape[0]).to(torch.int32)
    return all_c, labels, sizes


def prefill_cluster_meta_from_features_torch(
    feat: torch.Tensor,
    num_clusters: int,
    n_segment: int,
    n_iter: int = 15,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """
    Mean-center, run segment K-Means, return tensors in **original** key space.

    Keys: ``cluster_centres`` (float32 [K,D]), ``block_to_cluster`` (int64 [N]),
    ``cluster_size`` (int32 [K]), ``mean_key`` (float32 [D]).
    """
    f = feat.to(dtype=torch.float32)
    mean_key = f.mean(dim=0)
    centered = f - mean_key
    k = min(int(num_clusters), int(f.shape[0]))
    n_seg = min(int(n_segment), int(f.shape[0]))
    centres_c, labels, sizes = segment_kmeans_centered_torch(
        centered, n_clusters=k, n_segments=n_seg, n_iter=n_iter, seed=seed
    )
    centres = centres_c + mean_key
    return {
        "cluster_centres": centres,
        "block_to_cluster": labels,
        "cluster_size": sizes,
        "mean_key": mean_key,
    }
