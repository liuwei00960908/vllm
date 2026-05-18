from dataclasses import dataclass, field
from typing import ClassVar
import torch


@dataclass
class SparseBlockTableBuffers:
    block_table: torch.Tensor | None = None
    bt_len: torch.Tensor | None = None
    seqused_k: torch.Tensor | None = None
    workspace_state: torch.Tensor | None = None
    workspace_row_free_base: torch.Tensor | None = None
    workspace_row_plan_base: torch.Tensor | None = None
    workspace_plan_row: torch.Tensor | None = None
    workspace_plan_src_tb_idx: torch.Tensor | None = None
    workspace_plan_src_tb_off: torch.Tensor | None = None

    @property
    def used_free_block_count(self) -> torch.Tensor:
        assert self.workspace_state is not None
        return self.workspace_state[1:2]

    def ensure_capacity(
        self,
        *,
        device: torch.device,
        rows: int,
        max_bt_len: int,
        plan_capacity: int,
    ) -> None:
        if self.block_table is None or self.block_table.shape != (rows, max_bt_len):
            self.block_table = torch.empty(
                (rows, max_bt_len), dtype=torch.int32, device=device
            )
        if self.bt_len is None or self.bt_len.shape != (rows,):
            self.bt_len = torch.empty((rows,), dtype=torch.int32, device=device)
        if self.seqused_k is None or self.seqused_k.shape != (rows,):
            self.seqused_k = torch.empty((rows,), dtype=torch.int32, device=device)
        if self.workspace_state is None or self.workspace_state.shape[0] < 3:
            self.workspace_state = torch.empty((3,), dtype=torch.int32, device=device)
        if (self.workspace_row_free_base is None
                or self.workspace_row_free_base.shape[0] < rows):
            self.workspace_row_free_base = torch.empty(
                (rows,), dtype=torch.int32, device=device
            )
        if (self.workspace_row_plan_base is None
                or self.workspace_row_plan_base.shape[0] < rows):
            self.workspace_row_plan_base = torch.empty(
                (rows,), dtype=torch.int32, device=device
            )
        if self.workspace_plan_row is None or self.workspace_plan_row.shape[0] < plan_capacity:
            self.workspace_plan_row = torch.empty(
                (plan_capacity,), dtype=torch.int32, device=device
            )
        if (self.workspace_plan_src_tb_idx is None
                or self.workspace_plan_src_tb_idx.shape[0] < plan_capacity):
            self.workspace_plan_src_tb_idx = torch.empty(
                (plan_capacity,), dtype=torch.int32, device=device
            )
        if (self.workspace_plan_src_tb_off is None
                or self.workspace_plan_src_tb_off.shape[0] < plan_capacity):
            self.workspace_plan_src_tb_off = torch.empty(
                (plan_capacity,), dtype=torch.int32, device=device
            )

    def build(
        self,
        *,
        top_clusters: torch.Tensor,
        cluster_compact_block_ids: torch.Tensor,
        cluster_temp_kv_pos: torch.Tensor,
        cluster_total_kv_counts: torch.Tensor,
        temp_block_ids: torch.Tensor,
        block_storage: torch.Tensor,
        free_block_ids: torch.Tensor,
        max_bt_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from vllm._custom_ops import build_sparse_block_table_out

        rows = top_clusters.shape[0] * top_clusters.shape[1]
        plan_capacity = max(
            rows * top_clusters.shape[2] * max(int(block_storage.shape[2]) - 1, 0), 1
        )
        self.ensure_capacity(
            device=top_clusters.device,
            rows=rows,
            max_bt_len=max_bt_len,
            plan_capacity=plan_capacity,
        )
        assert self.block_table is not None
        assert self.bt_len is not None
        assert self.seqused_k is not None
        assert self.workspace_state is not None
        assert self.workspace_row_free_base is not None
        assert self.workspace_row_plan_base is not None
        assert self.workspace_plan_row is not None
        assert self.workspace_plan_src_tb_idx is not None
        assert self.workspace_plan_src_tb_off is not None
        return build_sparse_block_table_out(
            top_clusters,
            cluster_compact_block_ids,
            cluster_temp_kv_pos,
            cluster_total_kv_counts,
            temp_block_ids,
            block_storage,
            free_block_ids,
            max_bt_len,
            self.block_table,
            self.bt_len,
            self.seqused_k,
            self.workspace_state,
            self.workspace_row_free_base,
            self.workspace_row_plan_base,
            self.workspace_plan_row,
            self.workspace_plan_src_tb_idx,
            self.workspace_plan_src_tb_off,
        )


@dataclass
class SparseAppendBuffers:
    used_free_block_count: torch.Tensor | None = None
    error_code: torch.Tensor | None = None

    def ensure_capacity(self, *, device: torch.device) -> None:
        if self.used_free_block_count is None or self.used_free_block_count.numel() != 1:
            self.used_free_block_count = torch.empty(
                (1,), dtype=torch.int32, device=device
            )
        if self.error_code is None or self.error_code.numel() != 1:
            self.error_code = torch.empty((1,), dtype=torch.int32, device=device)

    def append(
        self,
        *,
        block_storage: torch.Tensor,
        cluster_compact_block_ids: torch.Tensor,
        cluster_temp_kv_pos: torch.Tensor,
        cluster_total_kv_counts: torch.Tensor,
        temp_block_ids: torch.Tensor,
        temp_block_kv_counts: torch.Tensor,
        temp_block_kv_owner: torch.Tensor,
        free_block_ids: torch.Tensor,
        used_free_block_count: torch.Tensor | None,
        key: torch.Tensor,
        value: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        from vllm._custom_ops import append_kv_to_clusters_inplace

        self.ensure_capacity(device=key.device)
        assert self.error_code is not None
        counter = used_free_block_count
        if counter is None:
            assert self.used_free_block_count is not None
            counter = self.used_free_block_count
            counter.zero_()
        return append_kv_to_clusters_inplace(
            block_storage,
            cluster_compact_block_ids,
            cluster_temp_kv_pos,
            cluster_total_kv_counts,
            temp_block_ids,
            temp_block_kv_counts,
            temp_block_kv_owner,
            free_block_ids,
            counter,
            self.error_code,
            key,
            value,
            label,
        )


@dataclass
class SparseManagerMetadata:
    temp_block_ids: torch.Tensor | None = None              # [max_temp_blocks]
    temp_block_kv_counts: torch.Tensor | None = None        # [1]
    temp_block_kv_owner: torch.Tensor | None = None         # [max_temp_blocks * cluster_block_size, 2]

    # currently unused
    steady_zone_head: torch.Tensor | None = None    # [Hkv, steady_zone_head_count]
    steady_zone_tail: torch.Tensor | None = None    # [Hkv, steady_zone_tail_count]

    INIT_CLUSTER_BLOCK_COUNT: ClassVar[int] = 1024
    cluster_compact_block_ids: torch.Tensor | None = None   # [Hkv, C, max_cluster_block_count]
    cluster_temp_kv_pos: torch.Tensor | None = None         # [Hkv, C, cluster_block_size, 2], 0 = temp block id, 1 = offset
    cluster_total_kv_counts: torch.Tensor | None = None     # [Hkv, C]

    cluster_centers_T: torch.Tensor | None = None           # [Hkv, dim, C]
    mean: torch.Tensor | None = None                        # [Hkv, dim]
    in_cluster_token_count: int = 0
    block_table_buffers: SparseBlockTableBuffers | None = None
    append_buffers: SparseAppendBuffers | None = None
    cu_seqlens_q_buffer: torch.Tensor | None = None

@dataclass
class RequestSparseClusterInfo:
    layers: dict[str, SparseManagerMetadata] = field(default_factory=dict)


@dataclass
class SparseClusterBlockInfo:
    temp_block_ids: torch.Tensor | None = None
    reusable_block_ids: torch.Tensor | None = None
    allocated_block_ids: torch.Tensor | None = None
    used_count: torch.Tensor | int = 0
    cluster_block_size: int = 0
    num_cluster: int = 0
    num_segment: int = 0
    nprobe: int = 0
    num_kv_heads: int = 0

    temp_block_ids_gpu: torch.Tensor | None = None
    reusable_block_ids_gpu: torch.Tensor | None = None
    allocated_block_ids_gpu: torch.Tensor | None = None
    used_count_gpu: torch.Tensor | None = None

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
