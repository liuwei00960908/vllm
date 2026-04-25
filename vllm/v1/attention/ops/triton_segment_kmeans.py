# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _triton_assign_kernel(
    K, X, S, C, M,  # data, centroids, data_sum, data_cnt, max_idx
    stride_kz, stride_kn, stride_kd,
    stride_xz, stride_xk, stride_xd,
    stride_sz, stride_sk, stride_sd,
    stride_cz, stride_ck,
    stride_mz, stride_mn,
    num_tokens, num_centroids,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N
    batch_idx = tl.program_id(1)

    if start_n >= num_tokens:
        return

    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    n_mask = offs_n < num_tokens

    # blockwise skip batch, token-wise broadcast, dim-wise broadcast
    k_ptrs = K + batch_idx * stride_kz + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    x_ptrs = X + batch_idx * stride_xz + offs_k[None, :] * stride_xk + offs_d[:, None] * stride_xd
    s_ptrs = S + batch_idx * stride_sz + offs_d[None, :] * stride_sd
    c_ptrs = C + batch_idx * stride_cz
    m_ptrs = M + batch_idx * stride_mz + offs_n * stride_mn

    k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.) # [BLOCK_N, BLOCK_D]
    max_val = tl.zeros([BLOCK_N], dtype=tl.float32) - float("inf")
    max_idx = tl.zeros([BLOCK_N], dtype=tl.int32)

    for start_k in tl.range(0, num_centroids, BLOCK_K):
        k_mask = start_k + offs_k < num_centroids
        x = tl.load(x_ptrs, mask=k_mask[None, :], other=0.) # [BLOCK_D, BLOCK_K]
        ip = tl.dot(k, x).to(tl.float32)    # [BLOCK_N, BLOCK_K]
        ip = tl.where(k_mask[None, :], ip, -float("inf"))
        tmp_max_val, tmp_max_idx = tl.max(ip, axis=1, return_indices=True)
        tmp_max_idx += start_k
        max_idx = tl.where(tmp_max_val > max_val, tmp_max_idx, max_idx)
        max_val = tl.maximum(tmp_max_val, max_val)
        x_ptrs += BLOCK_K * stride_xk

    tl.store(m_ptrs, max_idx, mask=n_mask)
    tl.atomic_add(s_ptrs + max_idx[:, None] * stride_sk, k.to(tl.float32), mask=n_mask[:, None], sem='relaxed')
    tl.atomic_add(c_ptrs + max_idx * stride_ck, tl.zeros_like(max_idx) + 1, mask=n_mask, sem='relaxed')


@triton.jit
def _triton_update_kernel(
    X, S, C,  # centroids, data_sum, data_cnt
    stride_xz, stride_xk, stride_xd,
    stride_sz, stride_sk, stride_sd,
    stride_cz, stride_ck,
    num_centroids,
    BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    NORMORLIZE: tl.constexpr,
):
    start_k = tl.program_id(0) * BLOCK_K
    batch_idx = tl.program_id(1)

    offs_k = start_k + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    k_mask = offs_k < num_centroids

    x_ptrs = X + batch_idx * stride_xz + offs_k[:, None] * stride_xk + offs_d[None, :] * stride_xd
    s_ptrs = S + batch_idx * stride_sz + offs_k[:, None] * stride_sk + offs_d[None, :] * stride_sd
    c_ptrs = C + batch_idx * stride_cz + offs_k[:, None] * stride_ck

    s = tl.load(s_ptrs, mask=k_mask[:, None], other=0.) # [BLOCK_K, BLOCK_D]
    c = tl.load(c_ptrs, mask=k_mask[:, None], other=0)
    x_mask = c > 0
    x = s / c
    if NORMORLIZE:
        x /= tl.sqrt(tl.sum(x * x, axis=-1, keep_dims=True))

    tl.store(x_ptrs, x.to(X.type.element_ty), mask=x_mask)


def _triton_k_means_train(
    data: torch.Tensor,             # [batch_size, num_tokens, dim]
    centroids: torch.Tensor,        # [batch_size, num_centroids, dim]
    max_idx: torch.Tensor = None,   # [batch_size, num_tokens]
    normalize_centroids: bool = True,
    return_indices: bool = False,
    return_counts: bool = False,
):
    batch_size, num_tokens, dim = data.shape
    num_centroids = centroids.shape[1]
    data_sum = torch.zeros_like(centroids, dtype=torch.float32)
    data_cnt = torch.zeros((batch_size, num_centroids), dtype=torch.int32, device=data.device)
    if max_idx is None:
        max_idx = torch.empty((batch_size, num_tokens), dtype=torch.int32, device=data.device)
    # assert max_idx.shape == (batch_size, num_tokens)
    block_N = 128
    block_K = 32
    # assert dim in [32, 64, 128]
    _triton_assign_kernel[(triton.cdiv(num_tokens, block_N), batch_size, 1)](
        data, centroids, data_sum, data_cnt, max_idx,
        data.stride(0), data.stride(1), data.stride(2),
        centroids.stride(0), centroids.stride(1), centroids.stride(2),
        data_sum.stride(0), data_sum.stride(1), data_sum.stride(2),
        data_cnt.stride(0), data_cnt.stride(1),
        max_idx.stride(0), max_idx.stride(1),
        num_tokens, num_centroids,
        BLOCK_N=block_N, BLOCK_K=block_K, BLOCK_D=dim,
        num_warps=4, num_stages=2,
    )
    block_K = 128
    _triton_update_kernel[(triton.cdiv(num_centroids, block_K), batch_size, 1)](
        centroids, data_sum, data_cnt,
        centroids.stride(0), centroids.stride(1), centroids.stride(2),
        data_sum.stride(0), data_sum.stride(1), data_sum.stride(2),
        data_cnt.stride(0), data_cnt.stride(1),
        num_centroids,
        BLOCK_K=block_K, BLOCK_D=dim,
        NORMORLIZE=normalize_centroids,
        num_warps=4, num_stages=1,
    )
    if return_indices:
        if return_counts:
            return centroids, max_idx, data_cnt
        return centroids, max_idx, data_cnt.max().item()
    return centroids


def segment_kmeans_centered_triton(
    features_centered: torch.Tensor,
    n_clusters: int,
    n_segments: int,
    n_iter: int = 15,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Segment K-Means for already mean-centered features.

    This is the Triton implementation of
    ``sparse_kmeans_torch.segment_kmeans_centered_torch``. The surrounding
    algorithm intentionally matches the torch reference: same segment split,
    same random initialization, dot-product assignment, empty-cluster retention,
    and no centroid normalization.
    """
    if features_centered.dim() != 3:
        raise ValueError(
            "features_centered must be [H, N, D], "
            f"got {tuple(features_centered.shape)}"
        )
    if not features_centered.is_cuda:
        raise ValueError("Triton segment K-Means requires CUDA tensors")

    h, n, d = features_centered.shape
    if n == 0:
        zc = features_centered.new_zeros((h, 0, d), dtype=torch.float32)
        zl = torch.zeros(h, 0, dtype=torch.int64,
                         device=features_centered.device)
        zs = torch.zeros(h, 0, dtype=torch.int32,
                         device=features_centered.device)
        return zc, zl, zs

    n_seg = min(int(n_segments), int(n))
    k_per_seg = max(1, int(n_clusters) // n_seg)
    seg_starts = [i * (n // n_seg) for i in range(n_seg)]
    seg_ends = seg_starts[1:] + [n]

    fc = features_centered.to(dtype=torch.float32).contiguous()
    all_centres: list[torch.Tensor] = []
    labels = torch.zeros(h, n, dtype=torch.int64, device=fc.device)
    cluster_offset = 0

    for seg_idx, (start, end) in enumerate(
        zip(seg_starts, seg_ends, strict=False)
    ):
        seg_feat = fc[:, start:end, :].contiguous()
        seg_n = int(seg_feat.shape[1])
        k = min(k_per_seg, seg_n)
        if k == 0:
            continue

        g = torch.Generator(device=fc.device)
        g.manual_seed(int(seed) + seg_idx)
        perm = torch.randperm(seg_n, generator=g, device=fc.device)[:k]
        centres = seg_feat[:, perm, :].contiguous().clone()
        labels_seg = torch.zeros(h, seg_n, dtype=torch.int32, device=fc.device)

        for it in range(max(int(n_iter), 0)):
            is_last = it == int(n_iter) - 1
            if is_last:
                centres, labels_seg, _ = _triton_k_means_train(
                    seg_feat,
                    centres,
                    max_idx=labels_seg,
                    normalize_centroids=False,
                    return_indices=True,
                    return_counts=True,
                )
            else:
                centres = _triton_k_means_train(
                    seg_feat,
                    centres,
                    max_idx=labels_seg,
                    normalize_centroids=False,
                    return_indices=False,
                )

        all_centres.append(centres)
        labels[:, start:end] = labels_seg.to(torch.int64) + cluster_offset
        cluster_offset += k

    if not all_centres:
        zc = fc.new_zeros((h, 0, d))
        zs = torch.zeros(h, 0, dtype=torch.int32, device=fc.device)
        return zc, labels, zs

    all_c = torch.cat(all_centres, dim=1)
    total_k = int(all_c.shape[1])
    sizes = torch.zeros(h, total_k, dtype=torch.int32, device=fc.device)
    for hi in range(h):
        sizes[hi] = torch.bincount(
            labels[hi], minlength=total_k
        ).to(torch.int32)
    return all_c, labels, sizes


@triton.jit
def _triton_reverse_index_kernel(
    M, I, C,  # max_idx, clusters, cluster_size
    stride_mz, stride_mn,
    stride_iz, stride_ik, stride_in,
    stride_cz, stride_ck,
    num_tokens,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N
    batch_idx = tl.program_id(1)

    if start_n >= num_tokens:
        return
    
    offs_n = start_n + tl.arange(0, BLOCK_N)
    n_mask = offs_n < num_tokens

    m_ptrs = M + batch_idx * stride_mz + offs_n * stride_mn
    i_ptrs = I + batch_idx * stride_iz
    c_ptrs = C + batch_idx * stride_cz

    max_idx = tl.load(m_ptrs, mask=n_mask, other=0)
    cnt = tl.atomic_add(c_ptrs + max_idx * stride_ck, tl.zeros_like(max_idx) + 1, mask=n_mask, sem='relaxed')
    tl.store(i_ptrs + max_idx * stride_ik + cnt * stride_in, offs_n, mask=n_mask)


def triton_reverse_index(
    max_idx: torch.Tensor,  # [batch_size, num_tokens]
    num_centroids: int,
    max_cluster_size: int,
):
    batch_size, num_tokens = max_idx.shape
    clusters = torch.zeros((batch_size, num_centroids, max_cluster_size), dtype=torch.int32, device=max_idx.device)
    cluster_size = torch.zeros((batch_size, num_centroids), dtype=torch.int32, device=max_idx.device)
    block_N = 128
    _triton_reverse_index_kernel[(triton.cdiv(num_tokens, block_N), batch_size, 1)](
        max_idx, clusters, cluster_size,
        max_idx.stride(0), 
        max_idx.stride(1),
        clusters.stride(0), clusters.stride(1), clusters.stride(2),
        cluster_size.stride(0), cluster_size.stride(1),
        num_tokens, BLOCK_N=block_N,
        num_warps=4, num_stages=1,
    )
    return clusters, cluster_size


@triton.jit
def _triton_index_add_kernel(
    V, S, M,  # data, sum, max_idx
    stride_vz, stride_vn, stride_vd,
    stride_sz, stride_sk, stride_sd,
    stride_mz, stride_mn,
    num_tokens,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N
    batch_idx = tl.program_id(1)

    if start_n >= num_tokens:
        return

    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    n_mask = offs_n < num_tokens

    v_ptrs = V + batch_idx * stride_vz + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    s_ptrs = S + batch_idx * stride_sz + offs_d[None, :] * stride_sd
    m_ptrs = M + batch_idx * stride_mz + offs_n * stride_mn

    v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.)
    max_idx = tl.load(m_ptrs, mask=n_mask, other=0)

    tl.atomic_add(s_ptrs + max_idx[:, None] * stride_sk, v.to(S.type.element_ty), mask=n_mask[:, None], sem='relaxed')


def triton_index_add(
    value: torch.Tensor,    # [batch_size, num_tokens, head_dim]
    max_idx: torch.Tensor,  # [batch_size, num_tokens]
    num_centroids: int,
):
    batch_size, num_tokens, dim = value.shape
    value_sum = torch.zeros((batch_size, num_centroids, dim), dtype=torch.float32, device=value.device)
    block_N = 128
    _triton_index_add_kernel[(triton.cdiv(num_tokens, block_N), batch_size, 1)](
        value, value_sum, max_idx,
        value.stride(0), value.stride(1), value.stride(2),
        value_sum.stride(0), value_sum.stride(1), value_sum.stride(2),
        max_idx.stride(0), max_idx.stride(1),
        num_tokens,
        BLOCK_N=block_N, BLOCK_D=dim,
    )
    return value_sum.to(value.dtype)


def segment_k_means(
    key: torch.Tensor,    # [batch_size(=1)*num_heads, num_tokens, head_dim]
    value: torch.Tensor,  # [batch_size(=1)*num_heads, num_tokens, head_dim]
    num_centroids: int,
    num_iters: int = 10,
    num_segments: int = 1
):
    num_groups, num_tokens, head_dim = key.shape

    # initialize centroids uniformly
    centroid_indices = torch.arange(num_centroids, dtype=torch.float32, device=key.device) * (num_tokens / num_centroids)
    centroid_indices += num_tokens / num_centroids / 2
    centroid_indices = centroid_indices.to(torch.int64)
    # Clamp to valid token range to avoid out-of-bounds access.
    if num_tokens > 0:
        centroid_indices = torch.clamp(centroid_indices, max=num_tokens - 1)
    centroids = torch.index_select(key, dim=1, index=centroid_indices)

    assert num_centroids % num_segments == 0
    num_tokens_per_segment = num_tokens // num_segments
    num_centroids_per_segment = num_centroids // num_segments
    data = key[:, :num_tokens_per_segment * num_segments].reshape((-1, num_tokens_per_segment, head_dim))
    centroids = centroids.reshape((-1, num_centroids_per_segment, head_dim))
    max_idx = torch.empty((data.shape[0], data.shape[1]), dtype=torch.int32, device=data.device)
    for _ in range(num_iters - 1):
        centroids = _triton_k_means_train(data, centroids, max_idx=max_idx, normalize_centroids=True, return_indices=False)

    data = key.reshape((-1, num_tokens, head_dim))
    centroids = centroids.reshape((-1, num_centroids, head_dim))
    centroids, max_idx, max_cluster_size = _triton_k_means_train(data, centroids, normalize_centroids=False, return_indices=True)

    value_sum = triton_index_add(value.reshape((-1, num_tokens, head_dim)), max_idx, num_centroids)
    clusters, cluster_size = triton_reverse_index(max_idx, num_centroids, max_cluster_size)

    return centroids, max_idx, clusters, cluster_size, value_sum


@triton.jit
def _triton_index_add_kernel_paged(
    V,                      # V: [block_num, block_size, num_heads, head_dim]
    B, S, M,                # S: output sum [num_heads, num_centroids, head_dim]
    stride_vp, stride_vb, stride_vh, stride_vd,
    stride_bid,
    stride_sz, stride_sk, stride_sd,
    stride_mz, stride_mn,
    num_tokens,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N
    head_idx = tl.program_id(1)   # head 索引

    if start_n >= num_tokens:
        return

    offs_n = start_n + tl.arange(0, BLOCK_N)
    n_mask = offs_n < num_tokens
    safe_offs_n = tl.where(n_mask, offs_n, 0)

    logical_block = safe_offs_n // BLOCK_S
    offset_in_block = safe_offs_n % BLOCK_S

    bid_ptrs = B + logical_block * stride_bid
    physical_block = tl.load(bid_ptrs, mask=n_mask)   # [BLOCK_N]

    offs_d = tl.arange(0, BLOCK_D)
    v_ptrs = (V + physical_block[:, None] * stride_vp +
              offset_in_block[:, None] * stride_vb +
              head_idx * stride_vh +
              offs_d[None, :] * stride_vd)
    v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.)   # [BLOCK_N, BLOCK_D]

    m_ptrs = M + head_idx * stride_mz + offs_n * stride_mn
    max_idx = tl.load(m_ptrs, mask=n_mask, other=0)       # [BLOCK_N]

    s_ptrs = S + head_idx * stride_sz + offs_d[None, :] * stride_sd
    tl.atomic_add(s_ptrs + max_idx[:, None] * stride_sk, v.to(S.type.element_ty), mask=n_mask[:, None], sem='relaxed')


def triton_index_add_paged(
    value_physical: torch.Tensor,  # [block_num, block_size, num_heads, head_dim]
    block_ids: torch.Tensor,       # [num_logical_blocks]
    max_idx: torch.Tensor,         # [num_heads, num_tokens]
    num_centroids: int,
    block_size: int = 16,
) -> torch.Tensor:
    """
    Accumulation on paged value per cluster
    
    Args:
        value_physical: [P, B, H, D]
        block_ids: L = num_logical_blocks
        max_idx: [H, L*B]
        num_centroids: number of clusters
        block_size: vllm block size 16
    
    Returns:
        value_sum: [H, num_centroids, D]
    """
    block_num, B, num_heads, head_dim = value_physical.shape
    assert B == block_size
    num_tokens = max_idx.shape[1]

    value_sum = torch.zeros((num_heads, num_centroids, head_dim), dtype=torch.float32, device=value_physical.device)

    block_N = 128
    grid = (triton.cdiv(num_tokens, block_N), num_heads)

    _triton_index_add_kernel_paged[grid](
        value_physical, block_ids, value_sum, max_idx,
        value_physical.stride(0), value_physical.stride(1),
        value_physical.stride(2), value_physical.stride(3),
        block_ids.stride(0),
        value_sum.stride(0), value_sum.stride(1), value_sum.stride(2),
        max_idx.stride(0), max_idx.stride(1),
        num_tokens,
        BLOCK_S=block_size,
        BLOCK_N=block_N,
        BLOCK_D=head_dim,
        num_warps=4,
        num_stages=2,
    )
    return value_sum.to(value_physical.dtype)


@triton.jit
def _triton_assign_kernel_paged(
    K, block_ids,           # K: [block_num, block_size, num_heads, head_dim]
    X, S, C, M, N,          # centroids, data_sum, data_cnt, max_idx, num_tokens
    stride_kp, stride_ks, stride_kh, stride_kd,
    stride_bid_seg, stride_bid_block,
    stride_xz, stride_xk, stride_xd,
    stride_sz, stride_sk, stride_sd,
    stride_cz, stride_ck,
    stride_mz, stride_mn,
    stride_ns,
    num_centroids,
    num_segments,
    BLOCK_N: tl.constexpr,  # token num of each thread block (128)
    BLOCK_S: tl.constexpr,  # vllm block_size = 16
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N
    batch_idx = tl.program_id(1)

    # centroids[heads * segments, ...]
    head_idx = batch_idx // num_segments
    segment_idx = batch_idx % num_segments

    num_tokens = tl.load(N + segment_idx * stride_ns)
    if start_n >= num_tokens:
        return

    offs_n = start_n + tl.arange(0, BLOCK_N)
    n_mask = offs_n < num_tokens
    offs_n = tl.where(n_mask, offs_n, 0)
    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)

    # load block indices
    logical_block = offs_n // BLOCK_S   # [BLOCK_N] = [block0, ..., block0, block1, ..., block1, ...]
    offset_in_block = offs_n % BLOCK_S  # [BLOCK_N] = [0, 1, ..., 15, 0, 1, ..., 15, ...]
    bid_ptrs = block_ids + segment_idx * stride_bid_seg + logical_block * stride_bid_block
    physical_block = tl.load(bid_ptrs, mask=n_mask, other=0)  # [BLOCK_N]

    # K: [P, BLOCK_S, H, D]
    # k_ptrs: [BLOCK_N, BLOCK_D]
    # x_ptrs: [BLOCK_D, BLOCK_K]
    k_ptrs = K + physical_block[:, None] * stride_kp + offset_in_block[:, None] * stride_ks + head_idx * stride_kh + offs_d[None, :] * stride_kd
    x_ptrs = X + batch_idx * stride_xz + offs_k[None, :] * stride_xk + offs_d[:, None] * stride_xd
    s_ptrs = S + batch_idx * stride_sz + offs_d[None, :] * stride_sd
    c_ptrs = C + batch_idx * stride_cz
    m_ptrs = M + batch_idx * stride_mz + offs_n * stride_mn

    k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.)
    max_val = tl.zeros([BLOCK_N], dtype=tl.float32) - float("inf")
    max_idx = tl.zeros([BLOCK_N], dtype=tl.int32)

    for start_k in tl.range(0, num_centroids, BLOCK_K):
        k_mask = start_k + offs_k < num_centroids
        x = tl.load(x_ptrs, mask=k_mask[None, :], other=0.) # [BLOCK_D, BLOCK_K]
        ip = tl.dot(k, x).to(tl.float32)    # [BLOCK_N, BLOCK_K]
        ip = tl.where(k_mask[None, :], ip, -float("inf"))
        tmp_max_val, tmp_max_idx = tl.max(ip, axis=1, return_indices=True)
        tmp_max_idx += start_k
        max_idx = tl.where(tmp_max_val > max_val, tmp_max_idx, max_idx)
        max_val = tl.maximum(tmp_max_val, max_val)
        x_ptrs += BLOCK_K * stride_xk

    tl.store(m_ptrs, max_idx, mask=n_mask)
    tl.atomic_add(s_ptrs + max_idx[:, None] * stride_sk, k.to(tl.float32), mask=n_mask[:, None])
    tl.atomic_add(c_ptrs + max_idx * stride_ck, 1, mask=n_mask)


def _triton_k_means_train_paged(
    data: torch.Tensor,             # [block_num, block_size, num_heads, head_dim]
    block_ids: torch.Tensor,        # [num_segments, num_logical_blocks]
    centroids: torch.Tensor,        # [batch_size, num_centroids, dim]
    num_tokens: torch.Tensor,       # [num_segments]
    num_segments: int,
    max_idx: torch.Tensor = None,   # [batch_size, num_tokens]
    normalize_centroids: bool = True,
    return_indices: bool = False,
    return_counts: bool = False,
):
    block_num, block_size, num_heads, dim = data.shape
    batch_size, num_centroids, _ = centroids.shape
    assert batch_size == num_heads * num_segments
    num_segments_tensor = num_tokens.size(0)
    assert num_segments_tensor == num_segments, f"num_segments_tensor = {num_segments_tensor}, num_segments = {num_segments}"

    data_sum = torch.zeros_like(centroids, dtype=torch.float32)
    data_cnt = torch.zeros((batch_size, num_centroids), dtype=torch.int32, device=data.device)
    max_tokens = num_tokens.max().item()
    if max_idx is None:
        max_idx = torch.empty((batch_size, max_tokens), dtype=torch.int32, device=data.device)
    block_N = max(64, block_size)
    if block_N % block_size != 0:
        block_N = triton.cdiv(block_N, block_size) * block_size
    block_K = 32
    block_S = block_size
    assert block_N % block_S == 0, f"block_N = {block_N}, block_S = {block_S}"
    num_warps = block_N // block_S
    if num_warps not in (1, 2, 4, 8):
        num_warps = 4
    # assert dim in [32, 64, 128]
    _triton_assign_kernel_paged[(triton.cdiv(max_tokens, block_N), batch_size, 1)](
        data, block_ids, centroids, data_sum, data_cnt, max_idx, num_tokens,
        data.stride(0), data.stride(1), data.stride(2), data.stride(3),
        block_ids.stride(0), block_ids.stride(1),
        centroids.stride(0), centroids.stride(1), centroids.stride(2),
        data_sum.stride(0), data_sum.stride(1), data_sum.stride(2),
        data_cnt.stride(0), data_cnt.stride(1),
        max_idx.stride(0), max_idx.stride(1),
        num_tokens.stride(0),
        num_centroids, num_segments,
        BLOCK_N=block_N, BLOCK_S=block_S, BLOCK_K=block_K, BLOCK_D=dim,
        num_warps=num_warps, num_stages=1,
    )
    block_K = 128
    _triton_update_kernel[(triton.cdiv(num_centroids, block_K), batch_size, 1)](
        centroids, data_sum, data_cnt,
        centroids.stride(0), centroids.stride(1), centroids.stride(2),
        data_sum.stride(0), data_sum.stride(1), data_sum.stride(2),
        data_cnt.stride(0), data_cnt.stride(1),
        num_centroids,
        BLOCK_K=block_K, BLOCK_D=dim,
        NORMORLIZE=normalize_centroids,
        num_warps=4, num_stages=1,
    )
    if return_indices:
        if return_counts:
            return centroids, max_idx, data_cnt
        return centroids, max_idx, data_cnt.max().item()
    return centroids


def segment_k_means_paged(
    key: torch.Tensor,          # [block_num, block_size, num_heads, head_dim]
    value: torch.Tensor,          # [block_num, block_size, num_heads, head_dim]
    block_ids: torch.Tensor,    # [num_logical_blocks]
    num_tokens: int,
    num_centroids: int,
    block_size: int = 16,
    num_iters: int = 10,
    num_segments: int = 1
):
    """
    Args:
        key:        full contiguous key block pool of vllm
        value:      full contiguous value block pool of vllm
        block_ids:  indices of the selected blocks
    Returns:
        centroids:      [num_heads, num_centroids, head_dim] centroid of each cluster
        max_idx:        [num_heads, num_tokens] centroid each token belongs to
        clusters:       [num_heads, num_centroids, max_cluster_size] tokens each cluster contains
        cluster_size:   [num_heads, num_centroids] num tokens of each cluster
        value_sum:      [num_heads, num_centroids, head_dim] sum of value of each cluster and head
    """

    block_num, block_size_phy, num_heads, head_dim = key.shape
    assert block_size_phy == block_size
    num_logical_blocks = block_ids.size(0)
    num_logical_tokens = num_logical_blocks * block_size_phy
    assert num_logical_tokens >= num_tokens and num_tokens > 0
    assert block_ids.min() >= 0 and block_ids.max() < block_num
    num_segments = max(1, min(int(num_segments), num_logical_blocks,
                              int(num_tokens)))

    # initialize centroids uniformly
    centroid_indices = torch.arange(num_centroids, dtype=torch.float32, device=key.device) * (num_tokens / num_centroids)
    centroid_indices += num_tokens / num_centroids / 2
    centroid_indices = centroid_indices.to(torch.int64)
    # Clamp to valid token range to avoid out-of-bounds access.
    if num_tokens > 0:
        centroid_indices = torch.clamp(centroid_indices, max=num_tokens - 1)

    logical_blocks = centroid_indices // block_size
    assert logical_blocks.max() <= num_logical_blocks - 1
    offsets = centroid_indices % block_size     # [num_centroids]
    physical_blocks = block_ids[logical_blocks] # [num_centroids]
    # Here create new tensor for centroids
    centroids = key[physical_blocks, offsets, :, :]     # [num_centroids, num_heads, head_dim]

    # -----------------------------------------------------------------------------------------------------------------#
    # segmentation on tokens, blocks, centroids
    # enlarge batch by num_segments time

    # block_ids segmemted
    max_blocks_per_segment = (num_logical_blocks + num_segments - 1) // num_segments   # ceiling
    block_ids_seg_pad = torch.zeros((num_segments * max_blocks_per_segment), dtype=torch.int32, device=block_ids.device)
    block_ids_seg_pad[:num_logical_blocks].copy_(block_ids)
    block_ids_seg = block_ids_seg_pad.view(num_segments, max_blocks_per_segment)

    # tokens segmemted
    max_tokens_per_segment = max_blocks_per_segment * block_size
    tokens_per_segment = torch.full((num_segments,), fill_value=max_tokens_per_segment, device=block_ids.device)
    tokens_per_segment[-1] = num_tokens - (num_segments - 1) * max_tokens_per_segment

    # centroids segmented
    assert num_centroids % num_segments == 0, f"num_centroids = {num_centroids}, num_segments = {num_segments}"
    centroids_per_segment = num_centroids // num_segments
    centroids = centroids.view(num_segments, centroids_per_segment, num_heads, head_dim)
    centroids = centroids.permute(2, 0, 1, 3).contiguous()  # [num_heads, num_segments, centroids_per_segment, head_dim]
    centroids = centroids.view(num_heads * num_segments, centroids_per_segment, head_dim)

    # max_idx segmemted
    max_idx = torch.empty((num_heads * num_segments, max_tokens_per_segment), dtype=torch.int32, device=key.device)

    # -----------------------------------------------------------------------------------------------------------------#
    # construct centroids
    for _ in range(num_iters - 1):
        # [num_heads * num_segments, centroids_per_segment, head_dim]
        centroids = _triton_k_means_train_paged(key, block_ids_seg, centroids, tokens_per_segment, num_segments,
            max_idx=max_idx, normalize_centroids=True, return_indices=False)

    # Final run, cluster each token
    centroids = centroids.view(num_heads, num_segments * centroids_per_segment, head_dim)
    centroids, max_idx, max_cluster_size = _triton_k_means_train_paged(
        key, block_ids.unsqueeze(0), centroids, torch.tensor([num_tokens], device=block_ids.device), 1,
        normalize_centroids=False, return_indices=True
    )

    clusters, cluster_size = triton_reverse_index(max_idx, num_centroids, max_cluster_size)

    value_sum = triton_index_add_paged(value, block_ids, max_idx, num_centroids, block_size)

    return centroids, max_idx, clusters, cluster_size, value_sum
