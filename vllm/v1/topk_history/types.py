# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SparseSelectionResult:
    """Logical token indices (positions in the sequence, 0 .. seq_len - 1)."""

    indices: list[int]
    """Layer name -> same as ``indices`` when per-layer clustering is enabled."""

    per_layer_indices: dict[str, list[int]] = field(default_factory=dict)
