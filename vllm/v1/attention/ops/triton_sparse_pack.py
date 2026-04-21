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
    counts_ptr,  # [H] int64 output
    S: tl.int32,
    mask_stride_h: tl.int64,
    TILE_S: tl.constexpr,
):
    """Row-wise sum of a bool mask.

    Launches ``H`` programs, each sequentially walking its row in tiles
    of ``TILE_S`` bools and accumulating the count.  The reduction is
    equivalent to ``selected_mask.sum(dim=1).to(torch.int64)`` but
    writes directly into an int64 output and runs as a single kernel.
    """
    h = tl.program_id(axis=0)
    row_ptr = mask_ptr + h.to(tl.int64) * mask_stride_h

    acc = tl.zeros((), tl.int64)
    for tile_base in tl.range(0, S, TILE_S):
        offs = tile_base + tl.arange(0, TILE_S)
        tile_mask = offs < S
        m = tl.load(row_ptr + offs, mask=tile_mask, other=0).to(tl.int32)
        acc += tl.sum(m).to(tl.int64)

    tl.store(counts_ptr + h, acc)


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
    perf_record: Callable[[str, float], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    counts = torch.empty(H, dtype=torch.int64, device=device)
    _sparse_pack_count_kernel[grid](
        selected_mask,
        counts,
        int(S),
        selected_mask.stride(0),
        TILE_S=TILE_S,
    )

    if perf_record is not None:
        perf_record("pack_sub:launch_count", time.perf_counter() - _t_count_start)
        _t_cumsum_start = time.perf_counter()

    # head_offsets: int32 [H+1]; [0]=0, [1:] = cumsum(counts) (int32).
    # The consumer (``cu_mat``/``head_offsets``) expects int32 already.
    head_offsets = torch.empty(int(H) + 1, dtype=torch.int32, device=device)
    head_offsets[0] = 0
    torch.cumsum(counts.to(torch.int32), dim=0, out=head_offsets[1:])

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

    return flat_phys64, flat_slots64, kv_token_ids, counts, head_offsets
