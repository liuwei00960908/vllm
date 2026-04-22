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


def _kmeans_dot_torch_batched(
    features: torch.Tensor,
    k: int,
    n_iter: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd K-Means on **mean-centered** features, batched over leading dim.

    Args:
        features: ``[H, N, D]`` float – one independent K-Means per row ``h``.

    Returns:
        centres: ``[H, k, D]``, labels: ``[H, N]`` int64.
    """
    h, n, d = features.shape
    k = min(int(k), int(n))
    if k == 0:
        zc = features.new_zeros((h, 0, d))
        zl = torch.zeros(h, n, dtype=torch.int64, device=features.device)
        return zc, zl

    g = torch.Generator(device=features.device)
    g.manual_seed(int(seed))
    perm = torch.randperm(n, generator=g, device=features.device)[:k]
    feat = features.to(dtype=torch.float32)
    centres = feat[:, perm, :].clone()

    labels = torch.zeros(h, n, dtype=torch.int64, device=features.device)
    for _ in range(n_iter):
        sims = torch.bmm(feat, centres.transpose(1, 2))
        labels = sims.argmax(dim=-1)
        new_centres = torch.zeros_like(centres)
        for hi in range(h):
            lh = labels[hi]
            fh = feat[hi]
            ch = centres[hi]
            nc = new_centres[hi]
            counts = torch.bincount(lh, minlength=k).to(dtype=torch.float32)
            nc.index_add_(0, lh, fh)
            has = counts > 0
            nc[has] /= counts[has].unsqueeze(1)
            nc[~has] = ch[~has]
        centres = new_centres
    return centres, labels


def _kmeans_dot_torch(
    features: torch.Tensor,
    k: int,
    n_iter: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd K-Means on **mean-centered** features (dot-product assignment)."""
    c_b, l_b = _kmeans_dot_torch_batched(
        features.unsqueeze(0), k, n_iter=n_iter, seed=seed
    )
    return c_b[0], l_b[0]


def _as_batched_features(
    features: torch.Tensor,
    name: str,
) -> tuple[torch.Tensor, bool]:
    if features.dim() == 2:
        return features.unsqueeze(0), True
    if features.dim() == 3:
        return features, False
    raise ValueError(
        f"{name} must be [N, D] or [H, N, D], got {tuple(features.shape)}"
    )


def _maybe_squeeze(tensor: torch.Tensor, squeeze: bool) -> torch.Tensor:
    return tensor[0] if squeeze else tensor


def segment_kmeans_centered_torch(
    features_centered: torch.Tensor,
    n_clusters: int,
    n_segments: int,
    n_iter: int = 15,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Segment K-Means on **already mean-centered** features.

    Args:
        features_centered: ``[N, D]`` or ``[H, N, D]``.

    Returns:
        For 2D input, ``centres_c`` is ``[total_k, D]``, ``labels`` is ``[N]``,
        and ``sizes`` is ``[total_k]``.  For 3D input, the same tensors keep the
        leading head dimension: ``[H, total_k, D]``, ``[H, N]``, ``[H, total_k]``.
    """
    features_centered, squeeze = _as_batched_features(
        features_centered, "features_centered"
    )
    h, n, d = features_centered.shape
    if n == 0:
        zc = features_centered.new_zeros((h, 0, d))
        zl = torch.zeros(h, 0, dtype=torch.int64, device=features_centered.device)
        zs = torch.zeros(h, 0, dtype=torch.int32, device=features_centered.device)
        return _maybe_squeeze(zc, squeeze), _maybe_squeeze(
            zl, squeeze
        ), _maybe_squeeze(zs, squeeze)

    n_seg = min(int(n_segments), int(n))
    k_per_seg = max(1, int(n_clusters) // n_seg)

    seg_starts = [i * (n // n_seg) for i in range(n_seg)]
    seg_ends = seg_starts[1:] + [n]

    all_centres: list[torch.Tensor] = []
    labels = torch.zeros(h, n, dtype=torch.int64, device=features_centered.device)
    cluster_offset = 0

    fc = features_centered.to(dtype=torch.float32)
    for seg_idx, (start, end) in enumerate(zip(seg_starts, seg_ends, strict=False)):
        seg_feat = fc[:, start:end, :]
        seg_n = seg_feat.shape[1]
        k = min(k_per_seg, seg_n)
        if k == 0:
            continue
        centres_seg, labels_seg = _kmeans_dot_torch_batched(
            seg_feat, k, n_iter=n_iter, seed=seed + seg_idx
        )
        all_centres.append(centres_seg)
        labels[:, start:end] = labels_seg + cluster_offset
        cluster_offset += k

    if not all_centres:
        zc = fc.new_zeros((h, 0, d))
        zs = torch.zeros(h, 0, dtype=torch.int32, device=features_centered.device)
        return _maybe_squeeze(zc, squeeze), _maybe_squeeze(
            labels, squeeze
        ), _maybe_squeeze(zs, squeeze)

    all_c = torch.cat(all_centres, dim=1)
    total_k = all_c.shape[1]
    sizes = torch.zeros(h, total_k, dtype=torch.int32, device=features_centered.device)
    for hi in range(h):
        sizes[hi] = torch.bincount(labels[hi], minlength=total_k).to(torch.int32)
    return _maybe_squeeze(all_c, squeeze), _maybe_squeeze(
        labels, squeeze
    ), _maybe_squeeze(sizes, squeeze)


def segment_kmeans_centered_torch_batched(
    features_centered: torch.Tensor,
    n_clusters: int,
    n_segments: int,
    n_iter: int = 15,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for callers that still pass batched features."""
    if features_centered.dim() != 3:
        raise ValueError(
            "features_centered must be [H, N, D], "
            f"got {tuple(features_centered.shape)}"
        )
    return segment_kmeans_centered_torch(
        features_centered,
        n_clusters=n_clusters,
        n_segments=n_segments,
        n_iter=n_iter,
        seed=seed,
    )


def prefill_cluster_meta_from_features_torch(
    feat: torch.Tensor,
    num_clusters: int,
    n_segment: int,
    n_iter: int = 15,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """
    Mean-center, run segment K-Means, return tensors in **original** key space.

    Args:
        feat: ``[N, D]`` or ``[H, N, D]``. With 3D input, ``H`` independent
        heads are clustered in one batched K-Means call.

    Keys:
        For 2D input, ``cluster_centres`` is float32 ``[K, D]``,
        ``block_to_cluster`` is int64 ``[N]``, ``cluster_size`` is int32
        ``[K]``, and ``mean_key`` is float32 ``[D]``. For 3D input, the same
        tensors keep the leading head dimension.
    """
    feat, squeeze = _as_batched_features(feat, "feat")
    f = feat.to(dtype=torch.float32)
    _h, n_tokens, _d = f.shape
    mean_key = f.mean(dim=1)
    centered = f - mean_key.unsqueeze(1)
    k = min(int(num_clusters), int(n_tokens))
    n_seg = min(int(n_segment), int(n_tokens))
    centres_c, labels, sizes = segment_kmeans_centered_torch(
        centered, n_clusters=k, n_segments=n_seg, n_iter=n_iter, seed=seed
    )
    centres = centres_c + mean_key.unsqueeze(1)
    return {
        "cluster_centres": _maybe_squeeze(centres, squeeze),
        "block_to_cluster": _maybe_squeeze(labels, squeeze),
        "cluster_size": _maybe_squeeze(sizes, squeeze),
        "mean_key": _maybe_squeeze(mean_key, squeeze),
    }


def prefill_cluster_meta_from_features_torch_batched(
    feat: torch.Tensor,
    num_clusters: int,
    n_segment: int,
    n_iter: int = 15,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Compatibility wrapper for callers that still pass batched features."""
    if feat.dim() != 3:
        raise ValueError(f"feat must be [H, N, D], got {tuple(feat.shape)}")
    return prefill_cluster_meta_from_features_torch(
        feat,
        num_clusters=num_clusters,
        n_segment=n_segment,
        n_iter=n_iter,
        seed=seed,
    )
