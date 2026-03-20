# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Sequence

import torch


def logical_indices_to_physical_slots(
    block_ids: Sequence[int],
    block_size: int,
    logical_indices: Sequence[int],
    *,
    dtype: torch.dtype = torch.long,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Map per-request **logical** token positions to paged **physical** slots.

    This follows the same convention as ``ExampleConnector``: physical slot id is
    ``physical_block * block_size + offset_in_block`` where ``physical_block``
    is ``block_ids[token // block_size]`` and ``offset_in_block`` is
    ``token % block_size``.

    Args:
        block_ids: Ordered physical block table for the request (length >=
            ceil(seq_len / block_size)).
        block_size: KV block/page size.
        logical_indices: Token indices in ``[0, seq_len)`` (duplicates removed).

    Returns:
        1-D ``torch.long`` tensor of unique physical slots, sorted ascending.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not block_ids:
        return torch.empty(0, dtype=dtype, device=device)

    uniq = sorted({int(i) for i in logical_indices if i is not None})
    slots: list[int] = []
    for t in uniq:
        if t < 0:
            continue
        bi = t // block_size
        off = t % block_size
        if bi >= len(block_ids):
            continue
        physical_block = int(block_ids[bi])
        slots.append(physical_block * block_size + off)

    return torch.tensor(slots, dtype=dtype, device=device)
