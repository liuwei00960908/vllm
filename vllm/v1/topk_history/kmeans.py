# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import torch


def kmeans_select_representatives(
    keys: torch.Tensor,
    k: int,
    *,
    forced_indices: torch.Tensor | None = None,
    num_iters: int = 5,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Pick up to ``k`` token indices using Lloyd k-means on key vectors.

    After clustering, one representative token per cluster is chosen (token in
    that cluster closest to the cluster centroid). All ``forced_indices`` are
    always included in the result. Remaining budget is filled with cluster reps.

    Args:
        keys: Tensor ``[seq_len, dim]`` (typically RoPE-applied keys projected
            to 2-D for distance, or a flattened head slice).
        k: Target number of **additional** cluster representatives before
            merging with forced set; effective cap is ``seq_len``.
        forced_indices: Optional 1-D integer tensor of logical indices that must
            appear in the output.
        num_iters: Lloyd iterations.
        generator: Optional RNG (used to break ties / init on CPU or CUDA).

    Returns:
        Sorted unique logical indices (Python ``list[int]``).
    """
    if keys.dim() != 2:
        raise ValueError(f"keys must be 2-D [seq_len, dim], got {keys.shape}")
    seq_len, _dim = keys.shape
    if seq_len == 0:
        return []
    device = keys.device
    dtype = keys.dtype

    forced: set[int] = set()
    if forced_indices is not None and forced_indices.numel() > 0:
        fi = forced_indices.detach().to(device=torch.device("cpu"), dtype=torch.long)
        forced.update(int(x) for x in fi.tolist() if 0 <= int(x) < seq_len)

    k_eff = max(0, min(int(k), seq_len))
    if k_eff == 0 and not forced:
        return []

    x = keys.detach().to(dtype=torch.float32)

    if k_eff == 0:
        return sorted(forced)

    # Initialize centroids: deterministic strided picks + small noise optional.
    if generator is not None:
        init_idx = torch.randperm(seq_len, device=device, generator=generator)[
            :k_eff
        ]
    else:
        step = max(1, seq_len // max(k_eff, 1))
        init_idx = torch.arange(0, seq_len, step, device=device)[:k_eff]
        if init_idx.numel() < k_eff:
            extra = torch.arange(0, k_eff, device=device) % seq_len
            init_idx = torch.unique(torch.cat([init_idx, extra], dim=0))[:k_eff]

    centroids = x[init_idx].clone()
    if centroids.shape[0] < k_eff:
        pad = k_eff - centroids.shape[0]
        centroids = torch.cat(
            [centroids, x[:pad]], dim=0
        )

    # Lloyd iterations
    for _ in range(num_iters):
        dist = torch.cdist(x, centroids, p=2.0)  # [seq_len, k_eff]
        assign = dist.argmin(dim=1)  # [seq_len]
        new_centroids = []
        for c in range(k_eff):
            mask = assign == c
            if mask.any():
                new_centroids.append(x[mask].mean(dim=0))
            else:
                # empty cluster: reuse old centroid
                new_centroids.append(centroids[c])
        centroids = torch.stack(new_centroids, dim=0)

    dist = torch.cdist(x, centroids, p=2.0)
    assign = dist.argmin(dim=1)

    chosen: set[int] = set(forced)
    for c in range(k_eff):
        mask = assign == c
        if not mask.any():
            continue
        idxs = mask.nonzero(as_tuple=False).squeeze(-1)
        sub = dist[idxs, c]
        best_local = int(idxs[int(sub.argmin())].item())
        chosen.add(best_local)

    return sorted(chosen)
