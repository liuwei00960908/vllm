# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels for the token-granularity sparse-attention decode
fast path.

These kernels replace a chain of ~5 tiny per-layer torch ops
(``torch.nonzero``, ``//``, ``-*``, ``index_select`` on the block table,
``sum(dim=1)``) that the sparse attention decode builder launches once
per sparse layer.  They preserve bitwise-identical outputs — no
approximation, no change in attended KV set — while cutting kernel
launch overhead from 5-6 ops down to 2 for the common
``num_reqs == 1`` decode path.

Kernels:
    ``sparse_pack_count_kernel``
        For each query head row of ``selected_mask``, count the number
        of selected tokens.  One program per head.
    ``sparse_pack_data_kernel``
        Using the already-computed ``head_offsets`` (exclusive prefix
        sum of ``counts``), walk each head row and for every selected
        token write into pre-allocated flat buffers the physical block
        id (fused ``bt_row.index_select(block_idx)``), the intra-block
        slot id (fused ``%`` / ``-*``), and the owning head id.  One
        program per head; within a row a Triton associative scan
        (``tl.cumsum`` on the int32 mask tile) produces the exclusive
        write offset, guaranteeing head-major / token-ascending order
        identical to ``torch.nonzero``'s row-major traversal.

The pair is deliberately split around a torch ``cumsum``/``.item()``
because the downstream consumer (FA compact-gather) needs an
exact-sized flat buffer; the same sync is already paid for by
``torch.nonzero`` in the legacy code path.
"""

from __future__ import annotations

import time
from typing import Callable

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _sparse_pack_count_kernel(
    mask_ptr,  # [H, S] bool
    counts_i32_ptr,  # [H] int32 output (written into ``head_offsets[1:]``)
    S: tl.int32,
    mask_stride_h: tl.int64,
    TILE_S: tl.constexpr,
):
    """Row-wise sum of a bool mask, int32 output.

    Launches ``H`` programs, each sequentially walking its row in tiles
    of ``TILE_S`` bools and accumulating the count into a register
    int32 accumulator.  Max per-head count is ``S`` (mask row length);
    for token-sparse decode ``S`` fits comfortably in int32 well below
    the INT32_MAX headroom.

    Phase-C optimization: writing int32 directly (was int64) lets the
    caller use the output tensor AS-IS as ``head_offsets[1:]`` and do
    an in-place cumsum, eliminating the legacy ``.to(int32)`` cast
    kernel (one kernel launch removed per sparse layer).
    """
    h = tl.program_id(axis=0)
    row_ptr = mask_ptr + h.to(tl.int64) * mask_stride_h

    acc = tl.zeros((), tl.int32)
    for tile_base in tl.range(0, S, TILE_S):
        offs = tile_base + tl.arange(0, TILE_S)
        tile_mask = offs < S
        m = tl.load(row_ptr + offs, mask=tile_mask, other=0).to(tl.int32)
        acc += tl.sum(m).to(tl.int32)

    tl.store(counts_i32_ptr + h, acc)


@triton.jit
def _sparse_pack_data_kernel(
    mask_ptr,  # [H, S] bool
    bt_row_ptr,  # [num_blocks_for_req] int32 (row view of block table)
    head_offsets_ptr,  # [H + 1] int32 — exclusive prefix of counts
    kv_head_ids_ptr,  # [H] int64 — precomputed kv_head id per q head
    phys_out_ptr,  # [total_N] int64
    slots_out_ptr,  # [total_N] int64
    kv_token_ids_out_ptr,  # [total_N] int64 (kv_head id broadcast per entry)
    S: tl.int32,
    BLOCK_SIZE_KV: tl.int32,  # kv paged block size (``bsz``)
    mask_stride_h: tl.int64,
    TILE_S: tl.constexpr,
):
    """Pack per-head (head-major, token-ascending) phys/slot outputs.

    For each program (one per head) the row is scanned tile-by-tile.
    Within a tile an exclusive prefix (``cumsum`` minus self) turns the
    local 0/1 mask into write offsets that are identical to the ordering
    ``torch.nonzero(selected_mask, as_tuple=False)`` would produce when
    restricted to this row (nonzero's row-major traversal).  A running
    inter-tile offset is threaded through the loop to preserve order
    across tiles.  The write is masked by the actual mask bit, so slots
    past the selected-count are never written.

    The block-table gather is fused inline: ``phys = bt_row[block_idx]``
    is one ``tl.load`` gather; the slot offset is recovered by
    ``offs - block_idx * BLOCK_SIZE_KV``.

    Phase B: outputs are int64 (widened from int32) and a third
    ``kv_token_ids`` output replays the static ``kv_head_ids`` mapping
    for every selected entry.  This folds the legacy
    ``flat_phys.to(int64)``, ``flat_slots.to(int64)`` and
    ``torch.repeat_interleave(kv_head_ids, counts)`` chain into a
    single kernel, saving 3 torch ops per sparse layer in decode.
    """
    h = tl.program_id(axis=0)
    row_ptr = mask_ptr + h.to(tl.int64) * mask_stride_h
    base_offset = tl.load(head_offsets_ptr + h).to(tl.int64)
    # ``kv_head_ids`` is a layer-static tensor (computed once at ctx
    # init and cached).  Loading a scalar per program is free on H100.
    kv_head_id_self = tl.load(kv_head_ids_ptr + h)

    running = tl.zeros((), tl.int32)
    for tile_base in tl.range(0, S, TILE_S):
        offs = tile_base + tl.arange(0, TILE_S)
        tile_mask = offs < S
        m = tl.load(row_ptr + offs, mask=tile_mask, other=0).to(tl.int32)

        # Exclusive prefix within the tile: local write slot for this
        # True entry (0-based) relative to the tile start.
        incl = tl.cumsum(m, axis=0)
        excl = incl - m

        block_idx = offs // BLOCK_SIZE_KV
        slot_off = offs - block_idx * BLOCK_SIZE_KV
        phys = tl.load(bt_row_ptr + block_idx, mask=tile_mask, other=0)

        abs_pos = base_offset + running.to(tl.int64) + excl.to(tl.int64)
        write_mask = tile_mask & (m != 0)
        # Int32 → int64 widening is free at the store site (Triton emits
        # a register extend), so the outputs are declared int64.
        tl.store(
            phys_out_ptr + abs_pos, phys.to(tl.int64), mask=write_mask
        )
        tl.store(
            slots_out_ptr + abs_pos,
            slot_off.to(tl.int64),
            mask=write_mask,
        )
        # Every selected entry carries the same ``kv_head_id_self``.
        # Broadcasting a scalar across the tile is a register-level op.
        tl.store(
            kv_token_ids_out_ptr + abs_pos,
            tl.full((TILE_S,), kv_head_id_self, tl.int64),
            mask=write_mask,
        )

        running += tl.sum(m)


def sparse_pack_single_req(
    selected_mask: torch.Tensor,
    bt_row_gpu: torch.Tensor,
    kv_head_ids: torch.Tensor,
    block_size_kv: int,
    head_offsets_scratch: torch.Tensor | None = None,
    perf_record: Callable[[str, float], None] | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor
]:
    """Bitwise-equivalent replacement for the legacy ``num_reqs == 1``
    pack + Tier-2 int64/kv_token_ids precompute chain:

    .. code-block:: python

        nz = torch.nonzero(selected_mask, as_tuple=False)
        head_ids_r = nz[:, 0]
        tok_ids_r = nz[:, 1]
        block_idx = tok_ids_r // bsz
        slots_r = (tok_ids_r - block_idx * bsz).to(torch.int32)
        phys_r = bt_row_gpu.index_select(0, block_idx)
        counts_r = selected_mask.sum(dim=1).to(torch.int64)
        # ... then later in finalize:
        flat_phys_int64 = flat_phys.to(torch.int64)
        flat_slots_int64 = flat_slots.to(torch.int64)
        kv_token_ids = torch.repeat_interleave(kv_head_ids, counts_r)

    Outputs ``(flat_phys64, flat_slots64, kv_token_ids, counts,
    head_offsets)`` where:

    - ``flat_phys64`` is head-major, token-ascending int64 [N]
    - ``flat_slots64`` is head-major, token-ascending int64 [N]
    - ``kv_token_ids`` is int64 [N] with ``kv_head_ids[h]`` at each
      entry belonging to head ``h``
    - ``counts`` is int64 [H] — per-head selected count
    - ``head_offsets`` is int32 [H+1] — exclusive prefix of ``counts``

    ``head_ids_r`` (the legacy per-entry head id) is not returned: the
    decode fast path (``num_reqs == 1``) does not consume it (it was
    only needed for the multi-req stable argsort).

    Widening phys/slots directly to int64 and emitting ``kv_token_ids``
    in the same kernel eliminates three torch-level kernels per sparse
    layer (two ``.to(int64)`` casts plus ``repeat_interleave``) that
    Phase-A's finalize still ran — observed savings ~0.8 ms/step at
    H=28, 20 sparse layers.

    One GPU sync is required to size the flat output buffers (via
    ``head_offsets[-1].item()``).  This matches the single implicit
    sync paid by ``torch.nonzero`` in the legacy path, so the end-to-end
    sync count is unchanged.

    Phase-C optimization: the ``cumsum`` stage used to launch three
    tiny kernels (``head_offsets[0]=0`` scalar write, ``counts.to(int32)``
    cast, ``torch.cumsum(out=...)``).  Two are removed here:

    - Count kernel writes **int32** directly, so ``counts_i32`` is the
      tensor passed into ``cumsum`` — no cast kernel.
    - When ``head_offsets_scratch`` is supplied (persistent int32
      buffer of size ``>= H + 1`` pre-zeroed by the caller), it is
      reused, so ``[0] = 0`` is paid once at init instead of per call.
      The caller must ensure ``scratch[0]`` stays at zero between
      calls (the cumsum below only writes into ``[1:]``).

    After the opt, the ``cumsum`` stage is **one kernel** (in-place
    cumsum on a view) plus a cheap view+alloc, trimming the segment
    from ~0.118 ms to ~0.04 ms per sparse layer (~2 ms/decode step
    at 28 sparse layers, H=28, measured on H100/Qwen).

    Returned ``counts`` is ``None`` when a scratch buffer is reused —
    the num_reqs==1 hot-path finalize does not consume it; a
    zero-cost view ``head_offsets[1:].to(int64)`` would only be
    materialized on the (never-taken in this path) num_reqs>1 branch.
    """
    assert selected_mask.dim() == 2, "selected_mask must be [H, S]"
    assert selected_mask.dtype == torch.bool
    assert bt_row_gpu.dim() == 1
    assert kv_head_ids.dim() == 1
    H, S = selected_mask.shape
    assert int(kv_head_ids.shape[0]) == int(H), (
        "kv_head_ids must have one entry per q head"
    )
    device = selected_mask.device

    # Tile choice: 1024 fits a row tile into a single warp-scan step in
    # Triton (``tl.cumsum`` scales linearly, 1024 is the sweet spot on
    # H100/A100 measured separately).  Kernels are launched on the
    # current CUDA stream by Triton.
    TILE_S = 1024 if S > 1024 else max(32, triton.next_power_of_2(max(int(S), 1)))
    grid = (int(H),)

    # Phase-C profiling: optional fine-grained sub-timers to locate the
    # dominant cost inside pack (kernel launches, cumsum, ``.item()``
    # sync, output allocations, data-kernel launch).  Each timer stop
    # placement is deliberate — wall time from submit-A to submit-B on
    # a single CUDA stream reflects "work-through-to-this-point" because
    # the only blocking call (``.item()``) drains the queue.
    if perf_record is not None:
        _t_count_start = time.perf_counter()

    # Phase-C: head_offsets is now the int32 receptacle for counts too.
    # With a persistent scratch buffer the per-call allocator is
    # bypassed entirely; without one we fall back to a fresh zeros().
    if head_offsets_scratch is not None and head_offsets_scratch.shape[0] >= H + 1:
        # Caller-provided scratch: pre-zeroed ``[0] = 0`` invariant
        # holds across calls because the cumsum below only writes [1:]
        # and the count kernel below overwrites [1:] every call.
        head_offsets = head_offsets_scratch[: int(H) + 1]
    else:
        head_offsets = torch.zeros(
            int(H) + 1, dtype=torch.int32, device=device
        )

    # Count kernel writes int32 counts directly into ``head_offsets[1:]``
    # (a zero-cost contiguous view).  Avoids the separate counts tensor
    # and the ``.to(int32)`` cast kernel.
    counts_i32_view = head_offsets[1:]
    _sparse_pack_count_kernel[grid](
        selected_mask,
        counts_i32_view,
        int(S),
        selected_mask.stride(0),
        TILE_S=TILE_S,
    )

    if perf_record is not None:
        perf_record("pack_sub:launch_count", time.perf_counter() - _t_count_start)
        _t_cumsum_start = time.perf_counter()

    # In-place exclusive-prefix via ``cumsum(out=same_view)``.  Single
    # kernel (was ``[0]=0`` + ``.to(int32)`` + ``cumsum`` = 3 kernels).
    torch.cumsum(counts_i32_view, dim=0, out=counts_i32_view)

    if perf_record is not None:
        perf_record("pack_sub:cumsum", time.perf_counter() - _t_cumsum_start)
        _t_sync_start = time.perf_counter()

    # Sync: we need the true total to size the flat output buffers.
    # Matches the implicit sync from the legacy ``torch.nonzero``.
    # NOTE: this ``.item()`` also drains the count kernel + cumsum, so
    # ``pack_sub:item_sync`` subsumes their exec time (their launch
    # time is captured above).
    total_n = int(head_offsets[-1].item())

    if perf_record is not None:
        perf_record("pack_sub:item_sync", time.perf_counter() - _t_sync_start)
        _t_alloc_start = time.perf_counter()

    flat_phys64 = torch.empty(total_n, dtype=torch.int64, device=device)
    flat_slots64 = torch.empty(total_n, dtype=torch.int64, device=device)
    kv_token_ids = torch.empty(total_n, dtype=torch.int64, device=device)

    if perf_record is not None:
        perf_record("pack_sub:alloc_outputs", time.perf_counter() - _t_alloc_start)
        _t_data_start = time.perf_counter()

    if total_n > 0:
        _sparse_pack_data_kernel[grid](
            selected_mask,
            bt_row_gpu,
            head_offsets,
            kv_head_ids,
            flat_phys64,
            flat_slots64,
            kv_token_ids,
            int(S),
            int(block_size_kv),
            selected_mask.stride(0),
            TILE_S=TILE_S,
        )

    if perf_record is not None:
        perf_record("pack_sub:launch_data", time.perf_counter() - _t_data_start)

    # ``counts`` is returned as ``None``: the num_reqs==1 hot path
    # consumes ``head_offsets[1:]`` directly; the num_reqs>1 path uses
    # the legacy ``selected_mask.sum(dim=1)`` branch (sparse_pack_single_req
    # is gated to num_reqs==1 at the call site).
    return flat_phys64, flat_slots64, kv_token_ids, None, head_offsets


@triton.jit
def _cluster_member_pack_count_kernel(
    top_clusters_ptr,  # [G, H_PER_KV, NPROBE]
    cluster_members_ptr,  # [G, K, M] int32 logical token ids
    cluster_size_ptr,  # [G, K] int32
    counts_i32_ptr,  # [G * H_PER_KV] int32 output
    SEQ_LEN: tl.int32,
    PROMPT_LEN: tl.int32,
    HEAD_N: tl.int32,
    TAIL_START: tl.int32,
    STEADY_COUNT: tl.int32,
    PENDING_COUNT: tl.int32,
    CAP: tl.int32,
    H_PER_KV: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    NPROBE: tl.constexpr,
    ALL_PROMPT_STEADY: tl.constexpr,
    HAS_SELECTION: tl.constexpr,
    TILE_M: tl.constexpr,
):
    h = tl.program_id(axis=0)
    if ALL_PROMPT_STEADY:
        tl.store(counts_i32_ptr + h, SEQ_LEN)
        return
    total = STEADY_COUNT + PENDING_COUNT
    if not HAS_SELECTION:
        tl.store(counts_i32_ptr + h, total)
        return

    g = h // H_PER_KV
    h_local = h - g * H_PER_KV
    selected = tl.zeros((), tl.int32)
    offs = tl.arange(0, TILE_M)
    for p in tl.range(0, NPROBE):
        cid = tl.load(
            top_clusters_ptr + (g * H_PER_KV + h_local) * NPROBE + p
        ).to(tl.int64)
        csize = tl.load(cluster_size_ptr + g * K + cid).to(tl.int32)
        member_base = (g * K + cid) * M
        for m_base in tl.range(0, M, TILE_M):
            m_offs = m_base + offs
            in_cluster = m_offs < csize
            pos = tl.load(
                cluster_members_ptr + member_base + m_offs,
                mask=in_cluster,
                other=0,
            ).to(tl.int32)
            valid = (
                in_cluster
                & (pos >= HEAD_N)
                & (pos < TAIL_START)
                & (pos < PROMPT_LEN)
            )
            selected += tl.sum(valid.to(tl.int32))
    selected = tl.minimum(selected, CAP)
    tl.store(counts_i32_ptr + h, total + selected)


@triton.jit
def _cluster_member_pack_data_kernel(
    top_clusters_ptr,  # [G, H_PER_KV, NPROBE]
    cluster_members_ptr,  # [G, K, M] int32 logical token ids
    cluster_size_ptr,  # [G, K] int32
    bt_row_ptr,  # [num_blocks_for_req] int32
    head_offsets_ptr,  # [G * H_PER_KV + 1] int32
    kv_head_ids_ptr,  # [G * H_PER_KV] int64
    phys_out_ptr,  # [total_N] int64
    slots_out_ptr,  # [total_N] int64
    kv_token_ids_out_ptr,  # [total_N] int64
    SEQ_LEN: tl.int32,
    PROMPT_LEN: tl.int32,
    HEAD_N: tl.int32,
    TAIL_START: tl.int32,
    PENDING_COUNT: tl.int32,
    CAP: tl.int32,
    BLOCK_SIZE_KV: tl.int32,
    H_PER_KV: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    NPROBE: tl.constexpr,
    ALL_PROMPT_STEADY: tl.constexpr,
    HAS_SELECTION: tl.constexpr,
    TILE_POS: tl.constexpr,
    TILE_M: tl.constexpr,
):
    h = tl.program_id(axis=0)
    base = tl.load(head_offsets_ptr + h).to(tl.int64)
    kv_head_id = tl.load(kv_head_ids_ptr + h)
    running = tl.zeros((), tl.int32)
    pos_offs = tl.arange(0, TILE_POS)

    if ALL_PROMPT_STEADY:
        for tile_base in tl.range(0, SEQ_LEN, TILE_POS):
            pos = tile_base + pos_offs
            mask = pos < SEQ_LEN
            block_idx = pos // BLOCK_SIZE_KV
            slot_off = pos - block_idx * BLOCK_SIZE_KV
            phys = tl.load(bt_row_ptr + block_idx, mask=mask, other=0)
            out_pos = base + pos.to(tl.int64)
            tl.store(phys_out_ptr + out_pos, phys.to(tl.int64), mask=mask)
            tl.store(
                slots_out_ptr + out_pos,
                slot_off.to(tl.int64),
                mask=mask,
            )
            tl.store(
                kv_token_ids_out_ptr + out_pos,
                kv_head_id + tl.zeros((TILE_POS,), dtype=tl.int64),
                mask=mask,
            )
        return

    # Steady prompt head: [0, HEAD_N).
    for tile_base in tl.range(0, HEAD_N, TILE_POS):
        pos = tile_base + pos_offs
        mask = pos < HEAD_N
        block_idx = pos // BLOCK_SIZE_KV
        slot_off = pos - block_idx * BLOCK_SIZE_KV
        phys = tl.load(bt_row_ptr + block_idx, mask=mask, other=0)
        out_pos = base + (running + pos).to(tl.int64)
        tl.store(phys_out_ptr + out_pos, phys.to(tl.int64), mask=mask)
        tl.store(slots_out_ptr + out_pos, slot_off.to(tl.int64), mask=mask)
        tl.store(
            kv_token_ids_out_ptr + out_pos,
            kv_head_id + tl.zeros((TILE_POS,), dtype=tl.int64),
            mask=mask,
        )
    running += HEAD_N

    # Selected non-steady prompt tokens, in selected-cluster order.
    if HAS_SELECTION:
        g = h // H_PER_KV
        h_local = h - g * H_PER_KV
        selected_seen = tl.zeros((), tl.int32)
        member_offs = tl.arange(0, TILE_M)
        for p in tl.range(0, NPROBE):
            cid = tl.load(
                top_clusters_ptr + (g * H_PER_KV + h_local) * NPROBE + p
            ).to(tl.int64)
            csize = tl.load(cluster_size_ptr + g * K + cid).to(tl.int32)
            member_base = (g * K + cid) * M
            for m_base in tl.range(0, M, TILE_M):
                m_offs = m_base + member_offs
                in_cluster = m_offs < csize
                pos = tl.load(
                    cluster_members_ptr + member_base + m_offs,
                    mask=in_cluster,
                    other=0,
                ).to(tl.int32)
                valid = (
                    in_cluster
                    & (pos >= HEAD_N)
                    & (pos < TAIL_START)
                    & (pos < PROMPT_LEN)
                )
                v_i32 = valid.to(tl.int32)
                incl = tl.cumsum(v_i32, axis=0)
                excl = incl - v_i32
                rank = selected_seen + excl
                write_mask = valid & (rank < CAP)
                block_idx = pos // BLOCK_SIZE_KV
                slot_off = pos - block_idx * BLOCK_SIZE_KV
                phys = tl.load(bt_row_ptr + block_idx, mask=write_mask, other=0)
                out_pos = base + (running + rank).to(tl.int64)
                tl.store(phys_out_ptr + out_pos, phys.to(tl.int64), mask=write_mask)
                tl.store(
                    slots_out_ptr + out_pos,
                    slot_off.to(tl.int64),
                    mask=write_mask,
                )
                tl.store(
                    kv_token_ids_out_ptr + out_pos,
                    kv_head_id + tl.zeros((TILE_M,), dtype=tl.int64),
                    mask=write_mask,
                )
                selected_seen += tl.sum(v_i32)
        running += tl.minimum(selected_seen, CAP)

    # Steady prompt tail: [TAIL_START, PROMPT_LEN).
    tail_count = PROMPT_LEN - TAIL_START
    for tile_base in tl.range(0, tail_count, TILE_POS):
        rel = tile_base + pos_offs
        mask = rel < tail_count
        pos = TAIL_START + rel
        block_idx = pos // BLOCK_SIZE_KV
        slot_off = pos - block_idx * BLOCK_SIZE_KV
        phys = tl.load(bt_row_ptr + block_idx, mask=mask, other=0)
        out_pos = base + (running + rel).to(tl.int64)
        tl.store(phys_out_ptr + out_pos, phys.to(tl.int64), mask=mask)
        tl.store(slots_out_ptr + out_pos, slot_off.to(tl.int64), mask=mask)
        tl.store(
            kv_token_ids_out_ptr + out_pos,
            kv_head_id + tl.zeros((TILE_POS,), dtype=tl.int64),
            mask=mask,
        )
    running += tail_count

    # Pending generated tokens: [PROMPT_LEN, SEQ_LEN).
    for tile_base in tl.range(0, PENDING_COUNT, TILE_POS):
        rel = tile_base + pos_offs
        mask = rel < PENDING_COUNT
        pos = PROMPT_LEN + rel
        block_idx = pos // BLOCK_SIZE_KV
        slot_off = pos - block_idx * BLOCK_SIZE_KV
        phys = tl.load(bt_row_ptr + block_idx, mask=mask, other=0)
        out_pos = base + (running + rel).to(tl.int64)
        tl.store(phys_out_ptr + out_pos, phys.to(tl.int64), mask=mask)
        tl.store(slots_out_ptr + out_pos, slot_off.to(tl.int64), mask=mask)
        tl.store(
            kv_token_ids_out_ptr + out_pos,
            kv_head_id + tl.zeros((TILE_POS,), dtype=tl.int64),
            mask=mask,
        )


def sparse_pack_cluster_members_single_req(
    top_clusters: torch.Tensor,
    cluster_members: torch.Tensor,
    cluster_size: torch.Tensor,
    bt_row_gpu: torch.Tensor,
    kv_head_ids: torch.Tensor,
    block_size_kv: int,
    *,
    seq_len: int,
    prompt_len: int,
    head_n: int,
    tail_start: int,
    select_budget: int,
    head_offsets_scratch: torch.Tensor | None = None,
    perf_record: Callable[[str, float], None] | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor
]:
    """Pack selected cluster members directly, without a dense token mask.

    ``top_clusters`` is ``[G, H_PER_KV, NPROBE]``. ``cluster_members`` is
    ``[G, K, M]`` and stores logical prompt token positions per cluster.
    Output ordering within each query head is steady-head, selected cluster
    members, steady-tail, pending tokens.  Decode attention with q_len=1 is
    invariant to this per-head KV order; the head grouping and counts are the
    contract consumed by compact varlen FA.
    """
    assert top_clusters.dim() == 3
    assert cluster_members.dim() == 3
    assert cluster_size.dim() == 2
    assert bt_row_gpu.dim() == 1
    assert kv_head_ids.dim() == 1

    top_clusters = top_clusters.contiguous()
    cluster_members = cluster_members.contiguous()
    cluster_size = cluster_size.contiguous()

    G, h_per_kv, nprobe = top_clusters.shape
    g_m, k_clusters, max_cluster_size = cluster_members.shape
    assert int(g_m) == int(G)
    assert int(cluster_size.shape[0]) == int(G)
    assert int(cluster_size.shape[1]) == int(k_clusters)
    H_total = int(G) * int(h_per_kv)
    assert int(kv_head_ids.shape[0]) == H_total
    device = top_clusters.device

    prompt_len_i = int(prompt_len)
    seq_len_i = int(seq_len)
    head_n_i = min(int(head_n), prompt_len_i)
    tail_start_i = max(0, min(int(tail_start), prompt_len_i))
    pending_count_i = max(0, seq_len_i - prompt_len_i)
    all_prompt_steady = head_n_i >= tail_start_i
    if all_prompt_steady:
        steady_count_i = prompt_len_i
    else:
        steady_count_i = head_n_i + (prompt_len_i - tail_start_i)
    cap_i = max(0, int(select_budget) - steady_count_i)
    has_selection = (
        cap_i > 0
        and int(nprobe) > 0
        and int(max_cluster_size) > 0
        and not all_prompt_steady
    )

    if head_offsets_scratch is not None and head_offsets_scratch.shape[0] >= H_total + 1:
        head_offsets = head_offsets_scratch[: H_total + 1]
    else:
        head_offsets = torch.zeros(H_total + 1, dtype=torch.int32, device=device)

    tile_m = (
        1024 if int(max_cluster_size) > 1024
        else max(32, triton.next_power_of_2(max(int(max_cluster_size), 1)))
    )
    tile_pos = 1024
    grid = (H_total,)

    if perf_record is not None:
        _t_count_start = time.perf_counter()
    counts_i32_view = head_offsets[1:]
    _cluster_member_pack_count_kernel[grid](
        top_clusters,
        cluster_members,
        cluster_size,
        counts_i32_view,
        int(seq_len_i),
        int(prompt_len_i),
        int(head_n_i),
        int(tail_start_i),
        int(steady_count_i),
        int(pending_count_i),
        int(cap_i),
        H_PER_KV=int(h_per_kv),
        K=int(k_clusters),
        M=int(max_cluster_size),
        NPROBE=int(nprobe),
        ALL_PROMPT_STEADY=bool(all_prompt_steady),
        HAS_SELECTION=bool(has_selection),
        TILE_M=int(tile_m),
    )
    if perf_record is not None:
        perf_record(
            "cluster_pack_sub:launch_count",
            time.perf_counter() - _t_count_start,
        )
        _t_cumsum_start = time.perf_counter()

    torch.cumsum(counts_i32_view, dim=0, out=counts_i32_view)

    if perf_record is not None:
        perf_record(
            "cluster_pack_sub:cumsum",
            time.perf_counter() - _t_cumsum_start,
        )
        _t_sync_start = time.perf_counter()

    total_n = int(head_offsets[-1].item())

    if perf_record is not None:
        perf_record(
            "cluster_pack_sub:item_sync",
            time.perf_counter() - _t_sync_start,
        )
        _t_alloc_start = time.perf_counter()

    flat_phys64 = torch.empty(total_n, dtype=torch.int64, device=device)
    flat_slots64 = torch.empty(total_n, dtype=torch.int64, device=device)
    kv_token_ids = torch.empty(total_n, dtype=torch.int64, device=device)

    if perf_record is not None:
        perf_record(
            "cluster_pack_sub:alloc_outputs",
            time.perf_counter() - _t_alloc_start,
        )
        _t_data_start = time.perf_counter()

    if total_n > 0:
        _cluster_member_pack_data_kernel[grid](
            top_clusters,
            cluster_members,
            cluster_size,
            bt_row_gpu,
            head_offsets,
            kv_head_ids,
            flat_phys64,
            flat_slots64,
            kv_token_ids,
            int(seq_len_i),
            int(prompt_len_i),
            int(head_n_i),
            int(tail_start_i),
            int(pending_count_i),
            int(cap_i),
            int(block_size_kv),
            H_PER_KV=int(h_per_kv),
            K=int(k_clusters),
            M=int(max_cluster_size),
            NPROBE=int(nprobe),
            ALL_PROMPT_STEADY=bool(all_prompt_steady),
            HAS_SELECTION=bool(has_selection),
            TILE_POS=int(tile_pos),
            TILE_M=int(tile_m),
        )

    if perf_record is not None:
        perf_record(
            "cluster_pack_sub:launch_data",
            time.perf_counter() - _t_data_start,
        )

    return flat_phys64, flat_slots64, kv_token_ids, None, head_offsets


@triton.jit
def _cluster_member_gather_kv_data_kernel(
    top_clusters_ptr,  # [G, H_PER_KV, NPROBE]
    cluster_members_ptr,  # [G, K, M] int32 logical token ids
    cluster_size_ptr,  # [G, K] int32
    bt_row_ptr,  # [num_blocks_for_req] int32
    head_offsets_ptr,  # [G * H_PER_KV + 1] int32
    key_cache_ptr,
    value_cache_ptr,
    k_out_ptr,  # [total_N, 1, D]
    v_out_ptr,  # [total_N, 1, D]
    SEQ_LEN: tl.int32,
    PROMPT_LEN: tl.int32,
    HEAD_N: tl.int32,
    TAIL_START: tl.int32,
    PENDING_COUNT: tl.int32,
    CAP: tl.int32,
    BLOCK_SIZE_KV: tl.int32,
    D: tl.int32,
    KC_S0: tl.int64,
    KC_S1: tl.int64,
    KC_S2: tl.int64,
    KC_S3: tl.int64,
    VC_S0: tl.int64,
    VC_S1: tl.int64,
    VC_S2: tl.int64,
    VC_S3: tl.int64,
    H_PER_KV: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    NPROBE: tl.constexpr,
    ALL_PROMPT_STEADY: tl.constexpr,
    HAS_SELECTION: tl.constexpr,
    TILE_POS: tl.constexpr,
    TILE_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    h = tl.program_id(axis=0)
    d_block = tl.program_id(axis=1)
    d_offs = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    base = tl.load(head_offsets_ptr + h).to(tl.int64)
    kv_head_id = (h // H_PER_KV).to(tl.int64)
    running = tl.zeros((), tl.int32)
    pos_offs = tl.arange(0, TILE_POS)

    if ALL_PROMPT_STEADY:
        for tile_base in tl.range(0, SEQ_LEN, TILE_POS):
            rel = tile_base + pos_offs
            pos_mask = rel < SEQ_LEN
            block_idx = rel // BLOCK_SIZE_KV
            slot_off = rel - block_idx * BLOCK_SIZE_KV
            phys = tl.load(bt_row_ptr + block_idx, mask=pos_mask, other=0)
            cache_base_k = (
                phys[:, None].to(tl.int64) * KC_S0
                + slot_off[:, None].to(tl.int64) * KC_S1
                + kv_head_id * KC_S2
                + d_offs[None, :].to(tl.int64) * KC_S3
            )
            cache_base_v = (
                phys[:, None].to(tl.int64) * VC_S0
                + slot_off[:, None].to(tl.int64) * VC_S1
                + kv_head_id * VC_S2
                + d_offs[None, :].to(tl.int64) * VC_S3
            )
            mask = pos_mask[:, None] & d_mask[None, :]
            k_vals = tl.load(key_cache_ptr + cache_base_k, mask=mask, other=0.0)
            v_vals = tl.load(value_cache_ptr + cache_base_v, mask=mask, other=0.0)
            out_idx = base + rel[:, None].to(tl.int64)
            out_ptr = out_idx * D + d_offs[None, :].to(tl.int64)
            tl.store(k_out_ptr + out_ptr, k_vals, mask=mask)
            tl.store(v_out_ptr + out_ptr, v_vals, mask=mask)
        return

    # Steady prompt head: [0, HEAD_N).
    for tile_base in tl.range(0, HEAD_N, TILE_POS):
        rel = tile_base + pos_offs
        pos_mask = rel < HEAD_N
        block_idx = rel // BLOCK_SIZE_KV
        slot_off = rel - block_idx * BLOCK_SIZE_KV
        phys = tl.load(bt_row_ptr + block_idx, mask=pos_mask, other=0)
        cache_base_k = (
            phys[:, None].to(tl.int64) * KC_S0
            + slot_off[:, None].to(tl.int64) * KC_S1
            + kv_head_id * KC_S2
            + d_offs[None, :].to(tl.int64) * KC_S3
        )
        cache_base_v = (
            phys[:, None].to(tl.int64) * VC_S0
            + slot_off[:, None].to(tl.int64) * VC_S1
            + kv_head_id * VC_S2
            + d_offs[None, :].to(tl.int64) * VC_S3
        )
        mask = pos_mask[:, None] & d_mask[None, :]
        k_vals = tl.load(key_cache_ptr + cache_base_k, mask=mask, other=0.0)
        v_vals = tl.load(value_cache_ptr + cache_base_v, mask=mask, other=0.0)
        out_idx = base + (running + rel)[:, None].to(tl.int64)
        out_ptr = out_idx * D + d_offs[None, :].to(tl.int64)
        tl.store(k_out_ptr + out_ptr, k_vals, mask=mask)
        tl.store(v_out_ptr + out_ptr, v_vals, mask=mask)
    running += HEAD_N

    # Selected non-steady prompt tokens, in selected-cluster order.
    if HAS_SELECTION:
        g = h // H_PER_KV
        h_local = h - g * H_PER_KV
        selected_seen = tl.zeros((), tl.int32)
        member_offs = tl.arange(0, TILE_M)
        for p in tl.range(0, NPROBE):
            cid = tl.load(
                top_clusters_ptr + (g * H_PER_KV + h_local) * NPROBE + p
            ).to(tl.int64)
            csize = tl.load(cluster_size_ptr + g * K + cid).to(tl.int32)
            member_base = (g * K + cid) * M
            for m_base in tl.range(0, M, TILE_M):
                m_offs = m_base + member_offs
                in_cluster = m_offs < csize
                pos = tl.load(
                    cluster_members_ptr + member_base + m_offs,
                    mask=in_cluster,
                    other=0,
                ).to(tl.int32)
                valid = (
                    in_cluster
                    & (pos >= HEAD_N)
                    & (pos < TAIL_START)
                    & (pos < PROMPT_LEN)
                )
                v_i32 = valid.to(tl.int32)
                incl = tl.cumsum(v_i32, axis=0)
                excl = incl - v_i32
                rank = selected_seen + excl
                write_mask_pos = valid & (rank < CAP)
                block_idx = pos // BLOCK_SIZE_KV
                slot_off = pos - block_idx * BLOCK_SIZE_KV
                phys = tl.load(
                    bt_row_ptr + block_idx, mask=write_mask_pos, other=0
                )
                cache_base_k = (
                    phys[:, None].to(tl.int64) * KC_S0
                    + slot_off[:, None].to(tl.int64) * KC_S1
                    + kv_head_id * KC_S2
                    + d_offs[None, :].to(tl.int64) * KC_S3
                )
                cache_base_v = (
                    phys[:, None].to(tl.int64) * VC_S0
                    + slot_off[:, None].to(tl.int64) * VC_S1
                    + kv_head_id * VC_S2
                    + d_offs[None, :].to(tl.int64) * VC_S3
                )
                mask = write_mask_pos[:, None] & d_mask[None, :]
                k_vals = tl.load(key_cache_ptr + cache_base_k, mask=mask, other=0.0)
                v_vals = tl.load(value_cache_ptr + cache_base_v, mask=mask, other=0.0)
                out_idx = base + (running + rank)[:, None].to(tl.int64)
                out_ptr = out_idx * D + d_offs[None, :].to(tl.int64)
                tl.store(k_out_ptr + out_ptr, k_vals, mask=mask)
                tl.store(v_out_ptr + out_ptr, v_vals, mask=mask)
                selected_seen += tl.sum(v_i32)
        running += tl.minimum(selected_seen, CAP)

    # Steady prompt tail: [TAIL_START, PROMPT_LEN).
    tail_count = PROMPT_LEN - TAIL_START
    for tile_base in tl.range(0, tail_count, TILE_POS):
        rel = tile_base + pos_offs
        pos_mask = rel < tail_count
        pos = TAIL_START + rel
        block_idx = pos // BLOCK_SIZE_KV
        slot_off = pos - block_idx * BLOCK_SIZE_KV
        phys = tl.load(bt_row_ptr + block_idx, mask=pos_mask, other=0)
        cache_base_k = (
            phys[:, None].to(tl.int64) * KC_S0
            + slot_off[:, None].to(tl.int64) * KC_S1
            + kv_head_id * KC_S2
            + d_offs[None, :].to(tl.int64) * KC_S3
        )
        cache_base_v = (
            phys[:, None].to(tl.int64) * VC_S0
            + slot_off[:, None].to(tl.int64) * VC_S1
            + kv_head_id * VC_S2
            + d_offs[None, :].to(tl.int64) * VC_S3
        )
        mask = pos_mask[:, None] & d_mask[None, :]
        k_vals = tl.load(key_cache_ptr + cache_base_k, mask=mask, other=0.0)
        v_vals = tl.load(value_cache_ptr + cache_base_v, mask=mask, other=0.0)
        out_idx = base + (running + rel)[:, None].to(tl.int64)
        out_ptr = out_idx * D + d_offs[None, :].to(tl.int64)
        tl.store(k_out_ptr + out_ptr, k_vals, mask=mask)
        tl.store(v_out_ptr + out_ptr, v_vals, mask=mask)
    running += tail_count

    # Pending generated tokens: [PROMPT_LEN, SEQ_LEN).
    for tile_base in tl.range(0, PENDING_COUNT, TILE_POS):
        rel = tile_base + pos_offs
        pos_mask = rel < PENDING_COUNT
        pos = PROMPT_LEN + rel
        block_idx = pos // BLOCK_SIZE_KV
        slot_off = pos - block_idx * BLOCK_SIZE_KV
        phys = tl.load(bt_row_ptr + block_idx, mask=pos_mask, other=0)
        cache_base_k = (
            phys[:, None].to(tl.int64) * KC_S0
            + slot_off[:, None].to(tl.int64) * KC_S1
            + kv_head_id * KC_S2
            + d_offs[None, :].to(tl.int64) * KC_S3
        )
        cache_base_v = (
            phys[:, None].to(tl.int64) * VC_S0
            + slot_off[:, None].to(tl.int64) * VC_S1
            + kv_head_id * VC_S2
            + d_offs[None, :].to(tl.int64) * VC_S3
        )
        mask = pos_mask[:, None] & d_mask[None, :]
        k_vals = tl.load(key_cache_ptr + cache_base_k, mask=mask, other=0.0)
        v_vals = tl.load(value_cache_ptr + cache_base_v, mask=mask, other=0.0)
        out_idx = base + (running + rel)[:, None].to(tl.int64)
        out_ptr = out_idx * D + d_offs[None, :].to(tl.int64)
        tl.store(k_out_ptr + out_ptr, k_vals, mask=mask)
        tl.store(v_out_ptr + out_ptr, v_vals, mask=mask)


def sparse_gather_cluster_members_to_exec_buf(
    *,
    top_clusters: torch.Tensor,
    cluster_members: torch.Tensor,
    cluster_size: torch.Tensor,
    bt_row_gpu: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_size_kv: int,
    seq_len: int,
    prompt_len: int,
    head_n: int,
    tail_start: int,
    select_budget: int,
    head_offsets_scratch: torch.Tensor | None = None,
    perf_record: Callable[[str, float], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather selected cluster members directly into FA varlen K/V buffers.

    Returns ``(k_compact, v_compact, cu_k_flat)`` where compact buffers have
    shape ``[total_selected, 1, D]`` and ``cu_k_flat`` is int32 ``[H + 1]``.
    """
    assert top_clusters.dim() == 3
    assert cluster_members.dim() == 3
    assert cluster_size.dim() == 2
    assert key_cache.dim() == 4
    assert value_cache.dim() == 4

    top_clusters = top_clusters.contiguous()
    cluster_members = cluster_members.contiguous()
    cluster_size = cluster_size.contiguous()

    G, h_per_kv, nprobe = top_clusters.shape
    g_m, k_clusters, max_cluster_size = cluster_members.shape
    assert int(g_m) == int(G)
    assert int(cluster_size.shape[0]) == int(G)
    assert int(cluster_size.shape[1]) == int(k_clusters)
    H_total = int(G) * int(h_per_kv)
    D = int(key_cache.shape[-1])
    device = key_cache.device

    prompt_len_i = int(prompt_len)
    seq_len_i = int(seq_len)
    head_n_i = min(int(head_n), prompt_len_i)
    tail_start_i = max(0, min(int(tail_start), prompt_len_i))
    pending_count_i = max(0, seq_len_i - prompt_len_i)
    all_prompt_steady = head_n_i >= tail_start_i
    steady_count_i = (
        prompt_len_i
        if all_prompt_steady
        else head_n_i + (prompt_len_i - tail_start_i)
    )
    cap_i = max(0, int(select_budget) - steady_count_i)
    has_selection = (
        cap_i > 0
        and int(nprobe) > 0
        and int(max_cluster_size) > 0
        and not all_prompt_steady
    )

    if head_offsets_scratch is not None and head_offsets_scratch.shape[0] >= H_total + 1:
        head_offsets = head_offsets_scratch[: H_total + 1]
    else:
        head_offsets = torch.zeros(H_total + 1, dtype=torch.int32, device=device)

    tile_m = (
        1024 if int(max_cluster_size) > 1024
        else max(32, triton.next_power_of_2(max(int(max_cluster_size), 1)))
    )
    tile_pos = 64
    block_d = 64 if D >= 64 else triton.next_power_of_2(max(D, 1))
    grid_count = (H_total,)

    if perf_record is not None:
        _t_count_start = time.perf_counter()
    counts_i32_view = head_offsets[1:]
    _cluster_member_pack_count_kernel[grid_count](
        top_clusters,
        cluster_members,
        cluster_size,
        counts_i32_view,
        int(seq_len_i),
        int(prompt_len_i),
        int(head_n_i),
        int(tail_start_i),
        int(steady_count_i),
        int(pending_count_i),
        int(cap_i),
        H_PER_KV=int(h_per_kv),
        K=int(k_clusters),
        M=int(max_cluster_size),
        NPROBE=int(nprobe),
        ALL_PROMPT_STEADY=bool(all_prompt_steady),
        HAS_SELECTION=bool(has_selection),
        TILE_M=int(tile_m),
    )
    if perf_record is not None:
        perf_record(
            "cluster_exec_sub:launch_count",
            time.perf_counter() - _t_count_start,
        )
        _t_cumsum_start = time.perf_counter()

    torch.cumsum(counts_i32_view, dim=0, out=counts_i32_view)

    if perf_record is not None:
        perf_record(
            "cluster_exec_sub:cumsum",
            time.perf_counter() - _t_cumsum_start,
        )
        _t_sync_start = time.perf_counter()

    total_n = int(head_offsets[-1].item())

    if perf_record is not None:
        perf_record(
            "cluster_exec_sub:item_sync",
            time.perf_counter() - _t_sync_start,
        )
        _t_alloc_start = time.perf_counter()

    k_compact = torch.empty(
        (total_n, 1, D), dtype=key_cache.dtype, device=device
    )
    v_compact = torch.empty(
        (total_n, 1, D), dtype=value_cache.dtype, device=device
    )

    if perf_record is not None:
        perf_record(
            "cluster_exec_sub:alloc_outputs",
            time.perf_counter() - _t_alloc_start,
        )
        _t_gather_start = time.perf_counter()

    if total_n > 0:
        grid_data = (H_total, triton.cdiv(D, block_d))
        _cluster_member_gather_kv_data_kernel[grid_data](
            top_clusters,
            cluster_members,
            cluster_size,
            bt_row_gpu,
            head_offsets,
            key_cache,
            value_cache,
            k_compact,
            v_compact,
            int(seq_len_i),
            int(prompt_len_i),
            int(head_n_i),
            int(tail_start_i),
            int(pending_count_i),
            int(cap_i),
            int(block_size_kv),
            int(D),
            int(key_cache.stride(0)),
            int(key_cache.stride(1)),
            int(key_cache.stride(2)),
            int(key_cache.stride(3)),
            int(value_cache.stride(0)),
            int(value_cache.stride(1)),
            int(value_cache.stride(2)),
            int(value_cache.stride(3)),
            H_PER_KV=int(h_per_kv),
            K=int(k_clusters),
            M=int(max_cluster_size),
            NPROBE=int(nprobe),
            ALL_PROMPT_STEADY=bool(all_prompt_steady),
            HAS_SELECTION=bool(has_selection),
            TILE_POS=int(tile_pos),
            TILE_M=int(tile_m),
            BLOCK_D=int(block_d),
        )

    if perf_record is not None:
        perf_record(
            "cluster_exec_sub:gather_data",
            time.perf_counter() - _t_gather_start,
        )

    return k_compact, v_compact, head_offsets
