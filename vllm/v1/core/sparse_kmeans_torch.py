# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU (or device) segment K-Means for sparse KV prefill indexing.

The torch path mirrors ``sparse_kv_cache_manager._segment_kmeans`` on centered
features. The optional Triton path directly calls the paged
``segment_k_means_paged`` operator on raw K/V cache for token prefill.
"""

from __future__ import annotations

import os
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_PREFILL_CLUSTER_DEVICE = os.environ.get(
    "VLLM_SPARSE_PREFILL_CLUSTER_DEVICE", "auto"
).lower()
_PREFILL_CLUSTER_TRITON = (
    os.environ.get("VLLM_SPARSE_TRITON_KMEANS", "0").lower()
    in ("1", "true", "yes", "on")
)
_PREFILL_CLUSTER_TRITON_DEBUG = (
    os.environ.get("VLLM_SPARSE_TRITON_KMEANS_DEBUG", "0").lower()
    in ("1", "true", "yes", "on")
)
_TRITON_KMEANS_HEAD_DIMS = {32, 64, 128}


def sparse_prefill_cluster_use_device_kmeans(feat: torch.Tensor) -> bool:
    """Whether to cluster ``feat`` on its current device (typically CUDA)."""
    if _PREFILL_CLUSTER_DEVICE == "cpu":
        return False
    if _PREFILL_CLUSTER_DEVICE == "cuda":
        return feat.is_cuda
    # auto
    return feat.is_cuda


def _triton_kmeans_skip_reason(feat: torch.Tensor) -> str | None:
    if not _PREFILL_CLUSTER_TRITON:
        return "VLLM_SPARSE_TRITON_KMEANS is not enabled"
    if not sparse_prefill_cluster_use_device_kmeans(feat):
        return (
            f"device clustering disabled or tensor is not on the requested "
            f"device: VLLM_SPARSE_PREFILL_CLUSTER_DEVICE="
            f"{_PREFILL_CLUSTER_DEVICE!r}, is_cuda={feat.is_cuda}"
        )
    if not feat.is_cuda:
        return "tensor is not CUDA"
    if feat.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return f"unsupported dtype {feat.dtype}"
    head_dim = int(feat.shape[-1]) if feat.dim() > 0 else -1
    if head_dim not in _TRITON_KMEANS_HEAD_DIMS:
        return (
            f"unsupported head_dim {head_dim}; supported="
            f"{sorted(_TRITON_KMEANS_HEAD_DIMS)}"
        )
    return None


def _trace_triton_kmeans(
    source: str,
    feat: torch.Tensor,
    *,
    use_triton: bool,
    reason: str | None = None,
) -> None:
    if not _PREFILL_CLUSTER_TRITON_DEBUG:
        return
    shape = tuple(feat.shape)
    if use_triton:
        logger.info_once(
            "[SparseTritonKMeans] source=%s use_triton=1 shape=%s "
            "dtype=%s device=%s",
            source,
            shape,
            feat.dtype,
            feat.device,
        )
    else:
        logger.info_once(
            "[SparseTritonKMeans] source=%s use_triton=0 reason=%s "
            "shape=%s dtype=%s device=%s",
            source,
            reason,
            shape,
            feat.dtype,
            feat.device,
        )


def sparse_prefill_cluster_use_triton_kmeans(feat: torch.Tensor) -> bool:
    """Whether to use the Triton K-Means kernels for ``feat``.

    This is deliberately opt-in and only gates Triton-compatible CUDA tensors.
    The active Triton prefill path is the paged ``segment_k_means_paged`` entry.
    """
    return _triton_kmeans_skip_reason(feat) is None


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


def _as_batched_kv_cache(
    kv_cache: torch.Tensor,
    name: str,
) -> tuple[torch.Tensor, bool]:
    if kv_cache.dim() == 3:
        return kv_cache.unsqueeze(2), True
    if kv_cache.dim() == 4:
        return kv_cache, False
    raise ValueError(
        f"{name} must be [B, S, D] or [B, S, H, D], "
        f"got {tuple(kv_cache.shape)}"
    )


def kmeans_features_from_kv_cache_torch(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    num_tokens: int,
    *,
    is_centered: bool,
) -> torch.Tensor:
    """Build K-Means input features directly from a paged K cache.

    Args:
        kv_cache: K cache, ``[num_blocks, block_size, num_heads, head_dim]``.
            A legacy squeezed ``[num_blocks, block_size, head_dim]`` layout is
            also accepted and treated as one KV head.
        block_ids: Physical block ids for this request, ``[num_selected_blocks]``.
        num_tokens: Number of valid request tokens covered by ``block_ids``.
        is_centered: If True, return one key-centre feature per selected block
            by averaging valid token slots in that block. If False, return one
            feature per valid token.

    Returns:
        Batched per-head features ``[num_heads, N, head_dim]``.
    """
    kv_cache, _ = _as_batched_kv_cache(kv_cache, "kv_cache")
    if block_ids.dim() != 1:
        raise ValueError(
            f"block_ids must be [num_selected_blocks], got {tuple(block_ids.shape)}"
        )

    num_blocks = int(block_ids.numel())
    block_size = int(kv_cache.shape[1])
    num_heads = int(kv_cache.shape[2])
    head_dim = int(kv_cache.shape[3])
    max_tokens = num_blocks * block_size
    valid_tokens = max(0, min(int(num_tokens), max_tokens))
    if valid_tokens == 0 or num_blocks == 0:
        return kv_cache.new_zeros((num_heads, 0, head_dim), dtype=torch.float32)

    if is_centered:
        valid_blocks = (valid_tokens + block_size - 1) // block_size
        k_blocks = kv_cache[block_ids[:valid_blocks]].to(dtype=torch.float32)
        out = kv_cache.new_empty(
            (num_heads, valid_blocks, head_dim), dtype=torch.float32
        )
        full_blocks = valid_tokens // block_size
        tail_tokens = valid_tokens % block_size
        if full_blocks:
            out[:, :full_blocks, :] = k_blocks[:full_blocks].mean(dim=1).transpose(
                0, 1
            )
        if tail_tokens:
            out[:, full_blocks, :] = k_blocks[full_blocks, :tail_tokens].mean(
                dim=0
            )
        return out.contiguous()

    k_blocks = kv_cache[block_ids].to(dtype=torch.float32)
    return (
        k_blocks.reshape(num_blocks * block_size, num_heads, head_dim)[
            :valid_tokens
        ]
        .transpose(0, 1)
        .contiguous()
    )


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


def prefill_cluster_meta_from_features_device(
    feat: torch.Tensor,
    num_clusters: int,
    n_segment: int,
    n_iter: int = 15,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Device K-Means from feature tensors.

    The direct Triton path starts from paged K/V cache, so feature tensors keep
    using the torch implementation.
    """
    skip_reason = _triton_kmeans_skip_reason(feat) or (
        "segment_k_means_paged requires paged kv_cache/value_cache inputs"
    )
    _trace_triton_kmeans("features", feat, use_triton=False, reason=skip_reason)
    return prefill_cluster_meta_from_features_torch(
        feat,
        num_clusters=num_clusters,
        n_segment=n_segment,
        n_iter=n_iter,
        seed=seed,
    )


def value_sum_from_kv_cache_torch(
    v_cache: torch.Tensor,
    block_ids: torch.Tensor,
    num_tokens: int,
    labels: torch.Tensor,
    num_clusters: int,
    *,
    middle_start: int = 0,
    middle_end: int | None = None,
) -> torch.Tensor:
    """Bucket-sum V rows by cluster directly from a paged V cache.

    Mirrors retroinfer's ``value_sum`` accumulator: for each K-Means cluster
    ``c``, the sum over all clustered-token V vectors that landed in ``c``.
    Used by the retroinfer-style "estimation zone" to approximate the softmax
    contribution of clusters not selected into the retrieval zone, via the
    identity ``softmax_like_score(c) ≈ Q · (value_sum[c] / cluster_size[c])``
    weighted by ``log(cluster_size[c])``.

    The read path intentionally mirrors ``kmeans_features_from_kv_cache_torch``
    so that reuse of ``block_ids`` stays one ``kv_cache[block_ids]`` gather –
    the V-side computation adds O(N·D) FLOPs beyond the K-side but does not
    introduce any extra block-table indirection.  Computation is kept in
    fp32 to keep the accumulator numerically stable across arbitrarily long
    prompts (retroinfer does the same via CUTLASS fp32 epilogue).

    Args:
        v_cache: V cache, ``[num_blocks, block_size, num_heads, head_dim]``.
            Legacy 3D ``[num_blocks, block_size, head_dim]`` is accepted as
            one KV head.
        block_ids: Physical block ids for this request.
        num_tokens: Number of valid request tokens covered by ``block_ids``.
        labels: Cluster label per request token, ``[H, N]`` or ``[N]`` int.
        num_clusters: ``K`` – output leading size.
        middle_start: First token index to include.  Tokens in
            ``[0, middle_start)`` are treated as the retroinfer "steady head
            zone" and excluded from the value_sum accumulation (they are
            always merged into the executable buffer, so letting them
            contribute to an estimated cluster score would double-count).
        middle_end: Exclusive upper bound; defaults to ``num_tokens``.
            ``[middle_end, num_tokens)`` is the steady tail zone.

    Returns:
        ``[H, K, head_dim]`` fp32 value_sum tensor (squeezed to ``[K, D]``
        when ``labels`` was 1D).
    """
    v_cache, kv_squeezed = _as_batched_kv_cache(v_cache, "v_cache")
    if block_ids.dim() != 1:
        raise ValueError(
            f"block_ids must be [num_selected_blocks], got {tuple(block_ids.shape)}"
        )

    num_blocks = int(block_ids.numel())
    block_size = int(v_cache.shape[1])
    num_heads = int(v_cache.shape[2])
    head_dim = int(v_cache.shape[3])
    max_tokens = num_blocks * block_size
    valid_tokens = max(0, min(int(num_tokens), max_tokens))

    lo = max(0, int(middle_start))
    hi = valid_tokens if middle_end is None else min(int(middle_end), valid_tokens)
    lo = min(lo, hi)
    m = hi - lo

    labels, lbl_squeezed = _as_batched_features(
        labels.unsqueeze(-1), "labels"
    )
    # labels is now [H, N, 1]; drop trailing dim.
    labels = labels.squeeze(-1)
    h_labels = int(labels.shape[0])

    if h_labels != num_heads:
        raise ValueError(
            f"labels head dim ({h_labels}) must match v_cache num_heads "
            f"({num_heads})"
        )

    out = v_cache.new_zeros(
        (num_heads, int(num_clusters), head_dim), dtype=torch.float32
    )
    if m <= 0 or num_blocks == 0 or num_heads == 0 or num_clusters == 0:
        return _maybe_squeeze(out, lbl_squeezed and kv_squeezed)

    # Gather the V rows covering [lo, hi) once – **in cache dtype** so
    # we don't carry a transient ``[M, H, D]`` fp32 copy of the whole
    # V-cache slice through the rest of prefill (observed: 770 MiB OOM
    # at long prompts on a 24 GiB card).  The fp32 upcast is deferred to
    # the per-head slice inside the scatter loop so peak is ``[M, D]``
    # fp32 instead of ``[M, H, D]``.  ``index_add_`` on a non-contiguous
    # (strided) fp32 source is well-supported by PyTorch's CUDA
    # implementation – no need for the pre-loop ``transpose+contiguous``.
    v_blocks = v_cache[block_ids]
    v_flat = v_blocks.reshape(num_blocks * block_size, num_heads, head_dim)[
        lo:hi
    ]  # [M, H, D]  (cache dtype)
    lab_slice = labels[:, lo:hi].to(torch.int64)  # [H, M]

    # Per-head scatter-add: one bincount-style accumulation per head.
    # The fp32 cast happens on the per-head slice so only ``[M, D]``
    # fp32 is live at any time (~M*D*4 bytes, vs the prior
    # ``[M, H, D]`` fp32 = M*H*D*4).
    for h in range(num_heads):
        out[h].index_add_(
            0, lab_slice[h], v_flat[:, h, :].to(dtype=torch.float32)
        )

    return _maybe_squeeze(out, lbl_squeezed and kv_squeezed)


def prefill_cluster_meta_from_kv_cache_torch(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    num_tokens: int,
    *,
    num_clusters: int,
    n_segment: int,
    is_centered: bool,
    n_iter: int = 15,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Build prefill K-Means metadata directly from raw paged K cache.

    ``is_centered=True`` clusters block key centres; ``is_centered=False``
    clusters individual token keys. The returned ``features`` tensor is the
    exact ``[H, N, D]`` feature matrix used for clustering so callers can copy it
    to CPU without re-reading the KV cache.
    """
    features = kmeans_features_from_kv_cache_torch(
        kv_cache,
        block_ids,
        num_tokens,
        is_centered=is_centered,
    )
    raw = prefill_cluster_meta_from_features_torch(
        features,
        num_clusters=num_clusters,
        n_segment=n_segment,
        n_iter=n_iter,
        seed=seed,
    )
    raw["features"] = features
    return raw


def prefill_cluster_meta_from_kv_cache_device(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    num_tokens: int,
    *,
    value_cache: torch.Tensor | None = None,
    num_clusters: int,
    n_segment: int,
    is_centered: bool,
    n_iter: int = 15,
    seed: int = 42,
    return_features: bool = True,
) -> dict[str, torch.Tensor]:
    """Build prefill K-Means metadata from raw paged KV cache.

    When Triton is enabled and ``value_cache`` is provided, token-granularity
    prefill (``is_centered=False``) calls the optimized
    ``segment_k_means_paged`` entry directly. Centered/block feature callers
    use the torch fallback.
    """
    skip_reason = _triton_kmeans_skip_reason(kv_cache)
    if skip_reason is None and not is_centered and value_cache is not None:
        _trace_triton_kmeans("kv_cache_paged", kv_cache, use_triton=True)
        t0 = time.perf_counter() if _PREFILL_CLUSTER_TRITON_DEBUG else None
        try:
            from vllm.v1.attention.ops.triton_segment_kmeans import (
                segment_k_means_paged,
            )

            key_cache, _ = _as_batched_kv_cache(kv_cache, "kv_cache")
            value_cache_b, _ = _as_batched_kv_cache(value_cache, "value_cache")
            if tuple(value_cache_b.shape) != tuple(key_cache.shape):
                raise ValueError(
                    "value_cache shape must match kv_cache shape for "
                    f"segment_k_means_paged, got {tuple(value_cache_b.shape)} "
                    f"vs {tuple(key_cache.shape)}"
                )
            if block_ids.dim() != 1:
                raise ValueError(
                    "block_ids must be [num_selected_blocks], got "
                    f"{tuple(block_ids.shape)}"
                )

            (
                centres,
                labels,
                clusters,
                cluster_size,
                value_sum,
            ) = segment_k_means_paged(
                key_cache,
                value_cache_b,
                block_ids,
                int(num_tokens),
                int(num_clusters),
                block_size=int(key_cache.shape[1]),
                num_iters=int(n_iter),
                num_segments=int(n_segment),
            )
            num_heads = int(key_cache.shape[2])
            head_dim = int(key_cache.shape[3])
            if return_features:
                features = kmeans_features_from_kv_cache_torch(
                    key_cache,
                    block_ids,
                    num_tokens,
                    is_centered=False,
                )
            else:
                # Token compact legacy TopK only needs the feature tensor's
                # shape/device to initialize online state storage. The content
                # is deliberately not copied into that state.
                features = key_cache.new_empty(
                    (num_heads, int(num_tokens), head_dim)
                )
            raw = {
                "cluster_centres": centres.to(dtype=torch.float32),
                "block_to_cluster": labels,
                "cluster_size": cluster_size.to(dtype=torch.int32),
                # Direct paged op ignores mean-centering; keep downstream
                # decode assignment in the same uncentered coordinate system.
                "mean_key": torch.zeros(
                    (num_heads, head_dim),
                    dtype=torch.float32,
                    device=key_cache.device,
                ),
                "features": features,
                "clusters": clusters,
                "value_sum": value_sum,
            }
            if t0 is not None:
                logger.info(
                    "[SparseTritonKMeans] source=kv_cache_paged done=1 "
                    "elapsed_ms=%.3f kv_shape=%s feature_shape=%s "
                    "centres_shape=%s labels_shape=%s dtype=%s device=%s",
                    (time.perf_counter() - t0) * 1000.0,
                    tuple(key_cache.shape),
                    tuple(features.shape),
                    tuple(centres.shape),
                    tuple(labels.shape),
                    key_cache.dtype,
                    key_cache.device,
                )
            return raw
        except Exception as err:
            if _PREFILL_CLUSTER_TRITON_DEBUG:
                logger.warning_once(
                    "[SparseTritonKMeans] source=kv_cache_paged fallback=1 "
                    "kv_shape=%s dtype=%s device=%s error=%s",
                    tuple(kv_cache.shape),
                    kv_cache.dtype,
                    kv_cache.device,
                    err,
                )
            logger.warning_once(
                "Sparse paged Triton K-Means from KV cache failed; falling back "
                "to torch K-Means. error=%s",
                err,
            )
    else:
        reason = skip_reason
        if reason is None:
            reason = (
                "segment_k_means_paged requires is_centered=False and "
                "value_cache"
            )
        _trace_triton_kmeans(
            "kv_cache_paged", kv_cache, use_triton=False, reason=reason
        )

    return prefill_cluster_meta_from_kv_cache_torch(
        kv_cache,
        block_ids,
        num_tokens,
        num_clusters=num_clusters,
        n_segment=n_segment,
        is_centered=is_centered,
        n_iter=n_iter,
        seed=seed,
    )


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
