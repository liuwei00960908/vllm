from dataclasses import dataclass, field
from typing import ClassVar
import torch

@dataclass
class SparseManagerMetadata:
    temp_block_ids: torch.Tensor | None = None              # [max_temp_blocks]
    temp_block_kv_counts: torch.Tensor | None = None        # [1]
    temp_block_kv_owner: torch.Tensor | None = None         # [max_temp_blocks * cluster_block_size, 2]

    # currently unused
    steady_zone_head: torch.Tensor | None = None    # [Hkv, steady_zone_head_count]
    steady_zone_tail: torch.Tensor | None = None    # [Hkv, steady_zone_tail_count]

    INIT_CLUSTER_BLOCK_COUNT: ClassVar[int] = 64
    cluster_compact_block_ids: torch.Tensor | None = None   # [Hkv, C, max_cluster_block_count]
    cluster_temp_kv_pos: torch.Tensor | None = None         # [Hkv, C, cluster_block_size, 2], 0 = temp block id, 1 = offset
    cluster_total_kv_counts: torch.Tensor | None = None     # [Hkv, C]

    cluster_centers_T: torch.Tensor | None = None           # [Hkv, dim, C]
    mean: torch.Tensor | None = None                        # [Hkv, dim]
    in_cluster_token_count: int = 0                      

@dataclass
class RequestSparseClusterInfo:
    layers: dict[str, SparseManagerMetadata] = field(default_factory=dict)

import time
from collections import defaultdict, deque

import torch


class _Series:
    def __init__(self, maxlen=2000, warmup=10):
        self.maxlen = maxlen
        self.warmup = warmup
        self.wall_ms = deque(maxlen=maxlen)
        self.gpu_ms = deque(maxlen=maxlen)
        self.total_count = 0

    def add(self, wall_ms, gpu_ms):
        self.total_count += 1
        if self.total_count <= self.warmup:
            return
        self.wall_ms.append(wall_ms)
        if gpu_ms is not None:
            self.gpu_ms.append(gpu_ms)

    def _pct(self, xs, q):
        if not xs:
            return None
        ys = sorted(xs)
        idx = int((len(ys) - 1) * q)
        return ys[idx]

    def summary(self):
        def pack(xs):
            if not xs:
                return {
                    "n": 0,
                    "avg": None,
                    "p50": None,
                    "p95": None,
                    "max": None,
                }
            return {
                "n": len(xs),
                "avg": sum(xs) / len(xs),
                "p50": self._pct(xs, 0.50),
                "p95": self._pct(xs, 0.95),
                "max": max(xs),
            }

        return {
            "count_total": self.total_count,
            "wall_ms": pack(list(self.wall_ms)),
            "gpu_ms": pack(list(self.gpu_ms)),
        }


_ENABLED = True
_PRINT_EVERY = 1
_MAXLEN = 2000
_WARMUP = 10

_STATS = defaultdict(lambda: _Series(maxlen=_MAXLEN, warmup=_WARMUP))
_ACTIVE = {}


def kvprof_config(enabled=True, print_every=100, maxlen=2000, warmup=10):
    global _ENABLED, _PRINT_EVERY, _MAXLEN, _WARMUP, _STATS, _ACTIVE
    _ENABLED = enabled
    _PRINT_EVERY = print_every
    _MAXLEN = maxlen
    _WARMUP = warmup
    _STATS = defaultdict(lambda: _Series(maxlen=_MAXLEN, warmup=_WARMUP))
    _ACTIVE = {}


def kvprof_start(name: str):
    if not _ENABLED:
        return

    start_evt = None
    if torch.cuda.is_available():
        start_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

    _ACTIVE[name] = {
        "t0": time.perf_counter(),
        "start_evt": start_evt,
    }


def kvprof_end(name: str):
    if not _ENABLED:
        return
    state = _ACTIVE.pop(name, None)
    if state is None:
        return

    wall_ms = (time.perf_counter() - state["t0"]) * 1000.0

    gpu_ms = None
    if state["start_evt"] is not None:
        end_evt = torch.cuda.Event(enable_timing=True)
        end_evt.record()
        end_evt.synchronize()
        gpu_ms = state["start_evt"].elapsed_time(end_evt)

    _STATS[name].add(wall_ms, gpu_ms)

    if _STATS[name].total_count % _PRINT_EVERY == 0:
        print(f"[kvprof] {name}: {_STATS[name].summary()}", flush=True)


def kvprof_report():
    return {name: stat.summary() for name, stat in _STATS.items()}


def kvprof_print():
    for name, stat in _STATS.items():
        print(f"[kvprof][final] {name}: {stat.summary()}", flush=True)
