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
        steady_start_block_ids: torch.Tensor,
        steady_end_block_ids: torch.Tensor,
        steady_state: torch.Tensor,
        max_bt_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from vllm._custom_ops import build_sparse_block_table_out

        rows = top_clusters.shape[0] * top_clusters.shape[1]
        block_size = int(block_storage.shape[2])
        steady_blocks = (
            int(steady_start_block_ids.shape[1])
            + int(steady_end_block_ids.shape[1])
        )
        plan_capacity = max(
            rows * (top_clusters.shape[2] + steady_blocks) * max(block_size - 1, 0),
            1,
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
            steady_start_block_ids,
            steady_end_block_ids,
            steady_state,
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
class SparseSteadyBuffers:
    evicted_key: torch.Tensor | None = None
    evicted_value: torch.Tensor | None = None
    evicted_count: torch.Tensor | None = None

    def ensure_capacity(
        self,
        *,
        capacity: int,
        hkv: int,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        shape = (max(1, capacity), hkv, dim)
        if (
            self.evicted_key is None
            or self.evicted_key.shape != shape
            or self.evicted_key.dtype != dtype
        ):
            self.evicted_key = torch.empty(shape, dtype=dtype, device=device)
        if (
            self.evicted_value is None
            or self.evicted_value.shape != shape
            or self.evicted_value.dtype != dtype
        ):
            self.evicted_value = torch.empty(shape, dtype=dtype, device=device)
        if self.evicted_count is None or self.evicted_count.numel() != 1:
            self.evicted_count = torch.empty((1,), dtype=torch.int32, device=device)

    def update(
        self,
        *,
        block_storage: torch.Tensor,
        steady_start_block_ids: torch.Tensor,
        steady_end_block_ids: torch.Tensor,
        steady_state: torch.Tensor,
        steady_start_capacity: int,
        steady_end_capacity: int,
        key: torch.Tensor,
        value: torch.Tensor,
        evicted_capacity: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from vllm._custom_ops import update_sparse_steady_kv_inplace

        self.ensure_capacity(
            capacity=evicted_capacity,
            hkv=key.shape[1],
            dim=key.shape[2],
            dtype=key.dtype,
            device=key.device,
        )
        assert self.evicted_key is not None
        assert self.evicted_value is not None
        assert self.evicted_count is not None
        return update_sparse_steady_kv_inplace(
            block_storage,
            steady_start_block_ids,
            steady_end_block_ids,
            steady_state,
            self.evicted_key,
            self.evicted_value,
            self.evicted_count,
            key,
            value,
            int(steady_start_capacity),
            int(steady_end_capacity),
        )


@dataclass
class SparseManagerMetadata:
    temp_block_ids: torch.Tensor | None = None              # [max_temp_blocks]
    temp_block_kv_counts: torch.Tensor | None = None        # [1]
    temp_block_kv_owner: torch.Tensor | None = None         # [max_temp_blocks * cluster_block_size, 2]

    steady_zone_head: torch.Tensor | None = None    # [Hkv, steady_zone_head_blocks]
    steady_zone_tail: torch.Tensor | None = None    # [Hkv, steady_zone_tail_blocks]
    # [total_seen, start_count, tail_count, tail_start]
    steady_state: torch.Tensor | None = None
    steady_start_capacity: int = 0
    steady_end_capacity: int = 0
    steady_total_token_count: int = 0
    steady_start_count: int = 0
    steady_end_count: int = 0
    steady_end_start: int = 0

    INIT_CLUSTER_BLOCK_COUNT: ClassVar[int] = 1024
    cluster_compact_block_ids: torch.Tensor | None = None   # [Hkv, C, max_cluster_block_count]
    cluster_temp_kv_pos: torch.Tensor | None = None         # [Hkv, C, cluster_block_size, 2], 0 = temp block id, 1 = offset
    cluster_total_kv_counts: torch.Tensor | None = None     # [Hkv, C]

    cluster_centers_T: torch.Tensor | None = None           # [Hkv, dim, C]
    mean: torch.Tensor | None = None                        # [Hkv, dim]
    in_cluster_token_count: int = 0
    block_table_buffers: SparseBlockTableBuffers | None = None
    append_buffers: SparseAppendBuffers | None = None
    steady_buffers: SparseSteadyBuffers | None = None
    cu_seqlens_q_buffer: torch.Tensor | None = None

    def reset_steady_state_from_python(self) -> None:
        assert self.steady_state is not None
        self.steady_state[0] = self.steady_total_token_count
        self.steady_state[1] = self.steady_start_count
        self.steady_state[2] = self.steady_end_count
        self.steady_state[3] = self.steady_end_start

    def count_steady_evictions(self, num_new_tokens: int) -> int:
        if num_new_tokens <= 0:
            return 0
        total = self.steady_total_token_count
        start_cap = self.steady_start_capacity
        end_cap = self.steady_end_capacity
        evicted = 0
        for _ in range(num_new_tokens):
            if total < start_cap:
                pass
            elif end_cap == 0:
                evicted += 1
            elif total < start_cap + end_cap:
                pass
            else:
                evicted += 1
            total += 1
        return evicted

    def advance_steady_python_state(self, num_new_tokens: int) -> int:
        evicted = self.count_steady_evictions(num_new_tokens)
        self.steady_total_token_count += num_new_tokens
        self.steady_start_count = min(
            self.steady_start_capacity,
            self.steady_total_token_count,
        )
        after_start = max(0, self.steady_total_token_count - self.steady_start_capacity)
        if self.steady_end_capacity > 0:
            self.steady_end_count = min(self.steady_end_capacity, after_start)
            if self.steady_end_count == self.steady_end_capacity:
                self.steady_end_start = (
                    self.steady_end_start + evicted
                ) % self.steady_end_capacity
        else:
            self.steady_end_count = 0
            self.steady_end_start = 0
        return evicted


@dataclass
class SparseManagerExtraInfo:
    req_id_list: list[str] = field(default_factory=list)
    layer_name: str = ""
    num_cluster: int = 0
    num_segment: int = 0
    nprobe: int = 0
    cluster_block_size: int = 0
    steady_start_capacity: int = 0
    steady_end_capacity: int = 0


@dataclass
class RequestSparseClusterInfo:
    layers: dict[str, SparseManagerMetadata] = field(default_factory=dict)


@dataclass
class SparseClusterBlockInfo:
    temp_block_ids: torch.Tensor | None = None
    reusable_block_ids: torch.Tensor | None = None
    allocated_block_ids: torch.Tensor | None = None
    steady_start_block_ids: torch.Tensor | None = None
    steady_end_block_ids: torch.Tensor | None = None
    used_count: torch.Tensor | int = 0
    cluster_block_size: int = 0
    steady_start_capacity: int = 0
    steady_end_capacity: int = 0
    num_cluster: int = 0
    num_segment: int = 0
    nprobe: int = 0
    num_kv_heads: int = 0

    temp_block_ids_gpu: torch.Tensor | None = None
    reusable_block_ids_gpu: torch.Tensor | None = None
    allocated_block_ids_gpu: torch.Tensor | None = None
    steady_start_block_ids_gpu: torch.Tensor | None = None
    steady_end_block_ids_gpu: torch.Tensor | None = None
    used_count_gpu: torch.Tensor | None = None
    packed_block_info_cpu: torch.Tensor | None = None
    packed_block_info_gpu: torch.Tensor | None = None
    used_count_ready_event: torch.Event | None = None
    window_seq: int = -1

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
