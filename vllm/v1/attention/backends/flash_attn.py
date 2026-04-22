# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention."""

import copy
import os
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np
import torch

from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_supports_fp8,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_supports_sinks,
        flash_attn_varlen_func,
        get_scheduler_metadata,
        reshape_and_cache_flash,
    )
from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
    get_current_vllm_config_or_none,
    get_layers_from_vllm_config,
)
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    get_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


# Gates all sparse-attention debug logs and the CUDA-event-based per-call timing
# used inside the FA forward (``[SparseDebug]`` / ``[SparsePerfFA]`` /
# ``[FullPerfFA]``).  Both the event.synchronize() calls and the log formatting
# are skipped when this is 0, which removes a substantial stall budget from the
# sparse decode hot path.  Default off.
_SPARSE_PERF_DEBUG: bool = int(os.getenv("VLLM_SPARSE_PERF_DEBUG", "0")) == 1
_SPARSE_DEBUG_ASSERT: bool = int(os.getenv("VLLM_SPARSE_DEBUG_ASSERT", "0")) == 1


def _sparse_tensor_range(
    name: str,
    tensor: torch.Tensor,
    upper: int,
) -> None:
    """Debug-only CUDA sync to turn sparse index corruption into ValueError."""
    if tensor.numel() == 0:
        return
    lo = int(tensor.min().item())
    hi = int(tensor.max().item())
    if lo < 0 or hi >= int(upper):
        raise ValueError(
            f"{name} out of range: min={lo} max={hi} upper={int(upper)}"
        )


def _sparse_assert_compact_gather_inputs(
    *,
    phys: torch.Tensor,
    slots: torch.Tensor,
    kv_heads: torch.Tensor,
    cu_k: torch.Tensor,
    key_cache: torch.Tensor,
    num_q_flat: int,
) -> None:
    """Debug-only validation before sparse compact-gather advanced indexing."""
    n = int(phys.numel())
    if int(slots.numel()) != n or int(kv_heads.numel()) != n:
        raise ValueError(
            "sparse compact gather metadata length mismatch: "
            f"phys={int(phys.numel())} slots={int(slots.numel())} "
            f"kv_heads={int(kv_heads.numel())}"
        )
    if int(cu_k.numel()) != int(num_q_flat) + 1:
        raise ValueError(
            "sparse compact gather cu_k length mismatch: "
            f"cu_k={int(cu_k.numel())} expected={int(num_q_flat) + 1}"
        )
    if int(cu_k[0].item()) != 0 or int(cu_k[-1].item()) != n:
        raise ValueError(
            "sparse compact gather cu_k endpoints mismatch: "
            f"first={int(cu_k[0].item())} last={int(cu_k[-1].item())} n={n}"
        )
    if cu_k.numel() > 1 and bool((cu_k[1:] < cu_k[:-1]).any().item()):
        raise ValueError("sparse compact gather cu_k is not monotonic")
    _sparse_tensor_range("sparse phys", phys, int(key_cache.shape[0]))
    _sparse_tensor_range("sparse slots", slots, int(key_cache.shape[1]))
    _sparse_tensor_range("sparse kv_head", kv_heads, int(key_cache.shape[2]))


class FlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        if (
            model_config
            and model_config.is_hybrid
            and (
                cache_config.mamba_ssm_cache_dtype == "float32"
                or cache_config.mamba_cache_dtype == "float32"
            )
        ):
            # NOTE(tdoublep): while in principle, FA supports
            # MultipleOf(16), these are the block sizes that do not
            # suffer from the NaN propagation problem described here:
            # https://github.com/Dao-AILab/flash-attention/issues/1974
            return [16, 32, 64]
        return [MultipleOf(16)]

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """FlashAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        fa_version = get_flash_attn_version()
        return fa_version is not None and fa_version >= 3

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, num_kv_heads, num_layers, 2, block_size, head_size)
            return (2, 4, 0, 1, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 8 == 0 and head_size <= 256

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        if kv_cache_dtype.startswith("fp8"):
            return flash_attn_supports_fp8()
        return kv_cache_dtype in ["auto", "float16", "bfloat16"]

    @classmethod
    def supports_sink(cls) -> bool:
        if not is_flash_attn_varlen_func_available():
            return False
        return flash_attn_supports_sinks()

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if has_sink and device_capability < DeviceCapability(9, 0):
            return "sink not supported on compute capability < 9.0"
        return None


@dataclass
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # For GQA DCP
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0

    causal: bool = True

    # Optional compact KV gather (token-sparse decode): paged cache indices.
    sparse_gather_phys: torch.Tensor | None = None
    sparse_gather_slots: torch.Tensor | None = None
    sparse_gather_cu_seqlens_k: torch.Tensor | None = None
    # Per query-head compact gather: len == num_heads, each (phys, slots, cu_k).
    sparse_q_head_gather: (
        tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...] | None
    ) = None
    # Flattened per query-head compact gather: phys/slots are flattened across
    # heads, with head_offsets marking each head segment and cu_k stored as
    # [num_heads, num_reqs + 1].
    sparse_q_head_gather_flat_phys: torch.Tensor | None = None
    sparse_q_head_gather_flat_slots: torch.Tensor | None = None
    sparse_q_head_gather_flat_cu_seqlens_k: torch.Tensor | None = None
    sparse_q_head_gather_head_offsets: torch.Tensor | None = None
    # Per query-head block-table sparse decode: [H, num_seqs, max_blocks], [H, num_seqs].
    sparse_per_head_block_table: torch.Tensor | None = None
    sparse_per_head_seq_lens: torch.Tensor | None = None
    # Upper bound on per-(head, req) selected token count for the compact
    # gather path.  Supplied by ``_build_sparse_runtime_q_head_gather`` via the
    # ``max_k`` key on the runtime gather dict, which lets FA skip a
    # ``.max().item()`` GPU sync when calling ``flash_attn_varlen_func``.
    sparse_q_head_gather_max_k: int | None = None
    # Pre-computed Tier-2 templates for the compact KV gather fast path.
    # Populated by ``_build_sparse_runtime_q_head_gather`` once per sparse
    # layer per step and consumed by
    # ``FlashAttentionImpl._forward_per_head_compact_kv_gather`` to avoid
    # rebuilding ~15 tiny tensor-op kernels on every layer forward.  All
    # fields are ``None`` when running full attention or when the builder
    # didn't run (fallback to per-call construction is retained in FA).
    sparse_q_head_gather_phys_int64: torch.Tensor | None = None
    sparse_q_head_gather_slots_int64: torch.Tensor | None = None
    sparse_q_head_gather_kv_token_ids: torch.Tensor | None = None
    sparse_q_head_gather_cu_k_flat: torch.Tensor | None = None
    sparse_q_head_gather_cu_q_flat: torch.Tensor | None = None
    sparse_q_head_gather_req_ids_flat: torch.Tensor | None = None
    sparse_q_head_gather_kv_pair_ids_flat: torch.Tensor | None = None
    sparse_q_head_gather_num_q_flat: int | None = None
    sparse_q_head_gather_num_q_heads: int | None = None
    sparse_q_head_gather_num_reqs: int | None = None

    # ── Retroinfer-style pre-gathered KV + estimation zone ──────────────────
    # Populated by the runner when running the new cluster-based sparse path
    # (Phase 4a builder + Phase 5 dispatcher).  Presence of
    # ``sparse_retroinfer_exec_buf_k`` is the sole branch switch that routes
    # ``FlashAttentionImpl.forward`` through
    # ``_forward_retroinfer_exec_buf`` instead of the legacy per-head
    # compact gather path.  All fields are ``None`` on layers / steps that
    # have not been converted to the retroinfer path, so the cluster rollout
    # can proceed layer-by-layer without forcing an all-or-nothing cutover.
    #
    # Shape contract (matching
    # ``GPUModelRunner._sparse_retroinfer_expand_and_gather_single_req``):
    #   exec_buf_k / exec_buf_v : [H_total, max_budget, D]   (cache dtype)
    #   valid_lengths           : [H_total]                   int32
    #   centres_es              : [H_total, es, D]            fp32
    #   value_sum_es            : [H_total, es, D]            fp32
    #   cluster_size_es         : [H_total, es]               fp32
    # ``H_total`` equals ``num_q_heads`` at num_reqs==1 (Phase 5 is decode
    # fast path only); the second FA pass over centres treats ``H_total`` as
    # an FA "batch" dimension so each query head gets its own per-row
    # ``valid_lengths`` – matching the legacy per-head compact gather's
    # per-head selection granularity.
    sparse_retroinfer_exec_buf_k: torch.Tensor | None = None
    sparse_retroinfer_exec_buf_v: torch.Tensor | None = None
    sparse_retroinfer_valid_lengths: torch.Tensor | None = None
    sparse_retroinfer_centres_es: torch.Tensor | None = None
    sparse_retroinfer_value_sum_es: torch.Tensor | None = None
    sparse_retroinfer_cluster_size_es: torch.Tensor | None = None


def _get_sliding_window_configs(
    vllm_config: VllmConfig,
) -> set[tuple[int, int] | None]:
    """Get the set of all sliding window configs used in the model."""
    sliding_window_configs: set[tuple[int, int] | None] = set()
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer in layers.values():
        assert isinstance(layer.impl, FlashAttentionImpl)
        sliding_window_configs.add(layer.impl.sliding_window)
    return sliding_window_configs


class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    # FA3:
    # Supports full cudagraphs for all cases.
    #
    # FA2:
    # For FA2, a graph is captured with max_query_len=1, (which is what we
    # capture by default for num_tokens <= max_num_seqs when there is no
    # spec-decode) then these graphs will not work for mixed prefill-decode
    # (unlike FA3). This is due to special max_query_len=1 packed-GQA handling
    # in FA2.
    # In summary if we are running with spec decodes the graphs would
    # work for mixed prefill-decode and uniform-decode. But for non-spec decodes
    # the graphs would not work for mixed prefill-decode; sorta the inverse
    # of UNIFORM_SINGLE_TOKEN_DECODE.
    # There's probably a better way to describe this using `AttentionCGSupport`
    # but for now just set it to `UNIFORM_BATCH` to get use to drop down
    # to FULL_AND_PIECEWISE.
    # TODO(luka, lucas): audit FA2 as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    _cudagraph_support = (
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() == 3
        else AttentionCGSupport.UNIFORM_BATCH
    )
    supports_update_block_table: bool = True

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.attention_config = vllm_config.attention_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.max_num_splits = 0  # No upper bound on the number of splits.
        self.aot_schedule = get_flash_attn_version() == 3

        try:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0

        self.cp_kv_cache_interleave_size = (
            self.parallel_config.cp_kv_cache_interleave_size
        )

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size

        if self.use_full_cuda_graph and self.aot_schedule:
            # FA3 scheduler_metadata size: 1 + round_up(batch_size, 4) * 4
            # The +1 is for the tile_count_semaphore (synchronization).
            # The 4 slots per batch element (num_prepare_batch_vectors) are:
            #   prepare_varlen + dynamic_split + sort_batches + head_swizzle
            # See: https://github.com/vllm-project/flash-attention/blob/5824e6e/hopper/flash_api.cpp#L664-L671  # noqa: E501
            max_batch_size = max(
                vllm_config.scheduler_config.max_num_seqs,
                self.max_cudagraph_size or 0,
            )
            self.scheduler_metadata = torch.zeros(
                1 + round_up(max_batch_size, 4) * 4,
                dtype=torch.int32,
                device=self.device,
            )
            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = (
                self.attention_config.flash_attn_max_num_splits_for_cuda_graph
            )

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: tuple[int, int] | None = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        """
        fast_build disables AOT scheduling, used when there will be few
        iterations i.e. spec-decode
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        # the overhead of the aot schedule is not worth it for spec-decode
        aot_schedule = self.aot_schedule and not fast_build

        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            # For the AOT scheduler we need the sliding window value to be
            # constant for all layers to. We have to populate this on the first
            # build() call so the layers are constructed (cannot populate)
            # in __init__.
            if aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(self.vllm_config)
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False
                    aot_schedule = False

        max_num_splits = 0  # 0 means use FA3's heuristics, not CG compatible
        if (
            self.use_full_cuda_graph
            and self.max_cudagraph_size is not None
            and num_actual_tokens <= self.max_cudagraph_size
        ):
            # NOTE(woosuk): Setting num_splits > 1 may increase the memory
            # usage, because the intermediate buffers of size [num_splits,
            # num_heads, num_tokens, head_size] are allocated. Therefore,
            # we only set num_splits when using cuda graphs.
            max_num_splits = self.max_num_splits

        if vllm_is_batch_invariant():
            max_num_splits = 1

        def schedule(
            batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal
        ):
            cache_dtype = self.cache_config.cache_dtype
            if cache_dtype.startswith("fp8"):
                qkv_dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                    cache_dtype
                )
            else:
                qkv_dtype = self.kv_cache_dtype
            if aot_schedule:
                return get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_seq_len,
                    num_heads_q=self.num_heads_q * self.dcp_world_size,
                    num_heads_kv=self.num_heads_kv,
                    headdim=self.headdim,
                    cache_seqlens=seqlens,
                    qkv_dtype=qkv_dtype,
                    cu_seqlens_q=cu_query_lens,
                    page_size=self.block_size,
                    causal=causal,
                    window_size=self.aot_sliding_window,
                    num_splits=max_num_splits,
                )
            return None

        use_cascade = common_prefix_len > 0
        max_dcp_context_kv_len = 0
        dcp_context_kv_lens = None

        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None

        if self.dcp_world_size > 1:
            query_kv_lens = query_start_loc[1:] - query_start_loc[:-1]
            dcp_context_kv_lens = seq_lens - query_kv_lens

            dcp_context_kv_lens = get_dcp_local_seq_lens(
                dcp_context_kv_lens,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )
            # After DCP distribution, the maximum number of tokens for any rank is
            # ceil(L / (N * I)) * I, where L is max_seq_len, N is dcp_world_size,
            # and I is cp_kv_cache_interleave_size.
            # This eliminates GPU->CPU sync while minimizing workspace over-allocation.
            num_partitions = self.dcp_world_size * self.cp_kv_cache_interleave_size
            max_dcp_context_kv_len = (
                (max_seq_len + num_partitions - 1) // num_partitions
            ) * self.cp_kv_cache_interleave_size

            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=dcp_context_kv_lens,
                max_seq_len=max_dcp_context_kv_len,
                causal=False,
            )
        elif use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            # Use GPU tensor directly - no CPU sync needed
            suffix_kv_lens = seq_lens[:num_reqs] - common_prefix_len
            prefix_scheduler_metadata = schedule(
                batch_size=1,
                cu_query_lens=cu_prefix_query_lens,
                max_query_len=num_actual_tokens,
                seqlens=prefix_kv_lens,
                max_seq_len=common_prefix_len,
                causal=False,
            )
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=suffix_kv_lens,
                max_seq_len=max_seq_len - common_prefix_len,
                causal=True,
            )
        else:
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=seq_lens,
                max_seq_len=max_seq_len,
                causal=causal,
            )
        # For FA3 + full cudagraph
        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            self.scheduler_metadata[:n] = scheduler_metadata
            # NOTE(woosuk): We should zero out the rest of the scheduler
            # metadata to guarantee the correctness. Otherwise, some thread
            # blocks may use the invalid scheduler metadata and overwrite the
            # output buffer.
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            max_dcp_context_kv_len=max_dcp_context_kv_len,
            dcp_context_kv_lens=dcp_context_kv_lens,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal,
        )
        return attn_metadata

    def update_block_table(
        self,
        metadata: FlashAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> FlashAttentionMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping
        return new_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return use_cascade_attention(*args, **kwargs)


class FlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version(
            requires_alibi=alibi_slopes is not None,
            head_size=head_size,
        )
        logger.info_once(
            "Using FlashAttention version %s",
            self.vllm_flash_attn_version,
            scope="local",
        )
        # Cache the batch invariant result for use in forward passes
        self.batch_invariant_enabled = vllm_is_batch_invariant()

        if is_quantized_kv_cache(self.kv_cache_dtype) and not flash_attn_supports_fp8():
            raise NotImplementedError(
                "FlashAttention does not support fp8 kv-cache on this device."
            )

        self.sinks = sinks
        if self.sinks is not None:
            assert flash_attn_supports_sinks(), (
                "Sinks are only supported in FlashAttention 3"
            )
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )

        self.supports_quant_query_input = True

        vllm_config = get_current_vllm_config_or_none()
        dcp_a2a = (
            vllm_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )
        self.dcp_combine = dcp_a2a_lse_reduce if dcp_a2a else cp_lse_ag_out_rs

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert output is not None, "Output tensor must be provided."
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(0)

        if self.kv_cache_dtype.startswith("fp8"):
            # queries are quantized in the attention layer
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            q_descale = layer._q_scale.expand(descale_shape)
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)

            if self.dcp_world_size > 1:
                self._forward_with_dcp(
                    query[:num_actual_tokens],
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    attn_metadata,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
                return output
            else:
                sliding_window_size = (
                    list(self.sliding_window)
                    if self.sliding_window is not None
                    else None
                )
                use_q_head_gather = (
                    (
                        attn_metadata.sparse_q_head_gather is not None
                        or attn_metadata.sparse_q_head_gather_flat_phys is not None
                    )
                    and self.alibi_slopes is None
                    and self.sinks is None
                    and sliding_window_size == [-1, -1]
                    and not self.kv_cache_dtype.startswith("fp8")
                )
                # Retroinfer-style cluster retrieval (Phase 5): the runner
                # attaches a pre-built ``exec_buf`` dict to the layer just
                # before invoking FA.  If present and compatible with the
                # current FA config (no alibi/sinks/sliding/fp8-kv), we run
                # the two-zone retrieval+estimation path.  The attribute is
                # cleared after consumption so a subsequent step that
                # doesn't populate it cleanly falls back to the legacy
                # sparse (or dense) path.
                runtime_retroinfer = getattr(
                    layer, "_vllm_sparse_runtime_retroinfer", None
                )
                use_retroinfer = (
                    runtime_retroinfer is not None
                    and self.alibi_slopes is None
                    and self.sinks is None
                    and sliding_window_size == [-1, -1]
                    and not self.kv_cache_dtype.startswith("fp8")
                )
                if use_retroinfer:
                    attn_metadata = replace(
                        attn_metadata,
                        sparse_retroinfer_exec_buf_k=(
                            runtime_retroinfer["exec_buf_k"]
                        ),
                        sparse_retroinfer_exec_buf_v=(
                            runtime_retroinfer["exec_buf_v"]
                        ),
                        sparse_retroinfer_valid_lengths=(
                            runtime_retroinfer["valid_lengths"]
                        ),
                        sparse_retroinfer_centres_es=(
                            runtime_retroinfer.get("centres_es")
                        ),
                        sparse_retroinfer_value_sum_es=(
                            runtime_retroinfer.get("value_sum_es")
                        ),
                        sparse_retroinfer_cluster_size_es=(
                            runtime_retroinfer.get("cluster_size_es")
                        ),
                        # Disable the legacy sparse paths for this step –
                        # the exec_buf has already folded in their work.
                        sparse_q_head_gather=None,
                        sparse_q_head_gather_flat_phys=None,
                        sparse_gather_phys=None,
                        sparse_per_head_block_table=None,
                    )
                    use_q_head_gather = False
                runtime_q_head_gather = (
                    None
                    if use_retroinfer
                    else getattr(
                        layer, "_vllm_sparse_runtime_q_head_gather", None
                    )
                )
                if runtime_q_head_gather is not None:
                    if isinstance(runtime_q_head_gather, dict):
                        attn_metadata = replace(
                            attn_metadata,
                            sparse_q_head_gather=None,
                            sparse_q_head_gather_flat_phys=runtime_q_head_gather[
                                "phys"
                            ],
                            sparse_q_head_gather_flat_slots=runtime_q_head_gather[
                                "slots"
                            ],
                            sparse_q_head_gather_flat_cu_seqlens_k=(
                                runtime_q_head_gather["cu"]
                            ),
                            sparse_q_head_gather_head_offsets=(
                                runtime_q_head_gather["head_offsets"]
                            ),
                            sparse_q_head_gather_max_k=(
                                runtime_q_head_gather.get("max_k")
                            ),
                            # Tier-2 precomputed fields (may be absent when
                            # the builder was produced by an older dict
                            # schema; FA falls back to on-the-fly build).
                            sparse_q_head_gather_phys_int64=(
                                runtime_q_head_gather.get("phys_int64")
                            ),
                            sparse_q_head_gather_slots_int64=(
                                runtime_q_head_gather.get("slots_int64")
                            ),
                            sparse_q_head_gather_kv_token_ids=(
                                runtime_q_head_gather.get("kv_token_ids")
                            ),
                            sparse_q_head_gather_cu_k_flat=(
                                runtime_q_head_gather.get("cu_k_flat")
                            ),
                            sparse_q_head_gather_cu_q_flat=(
                                runtime_q_head_gather.get("cu_q_flat")
                            ),
                            sparse_q_head_gather_req_ids_flat=(
                                runtime_q_head_gather.get("req_ids_flat")
                            ),
                            sparse_q_head_gather_kv_pair_ids_flat=(
                                runtime_q_head_gather.get("kv_pair_ids_flat")
                            ),
                            sparse_q_head_gather_num_q_flat=(
                                runtime_q_head_gather.get("num_q_flat")
                            ),
                            sparse_q_head_gather_num_q_heads=(
                                runtime_q_head_gather.get("num_q_heads")
                            ),
                            sparse_q_head_gather_num_reqs=(
                                runtime_q_head_gather.get("num_reqs")
                            ),
                            sparse_gather_phys=None,
                            sparse_gather_slots=None,
                            sparse_gather_cu_seqlens_k=None,
                        )
                    else:
                        attn_metadata = replace(
                            attn_metadata,
                            sparse_q_head_gather=runtime_q_head_gather,
                            sparse_q_head_gather_flat_phys=None,
                            sparse_q_head_gather_flat_slots=None,
                            sparse_q_head_gather_flat_cu_seqlens_k=None,
                            sparse_q_head_gather_head_offsets=None,
                            sparse_q_head_gather_max_k=None,
                            sparse_q_head_gather_phys_int64=None,
                            sparse_q_head_gather_slots_int64=None,
                            sparse_q_head_gather_kv_token_ids=None,
                            sparse_q_head_gather_cu_k_flat=None,
                            sparse_q_head_gather_cu_q_flat=None,
                            sparse_q_head_gather_req_ids_flat=None,
                            sparse_q_head_gather_kv_pair_ids_flat=None,
                            sparse_q_head_gather_num_q_flat=None,
                            sparse_q_head_gather_num_q_heads=None,
                            sparse_q_head_gather_num_reqs=None,
                            sparse_gather_phys=None,
                            sparse_gather_slots=None,
                            sparse_gather_cu_seqlens_k=None,
                        )
                    use_q_head_gather = True
                use_gather = (
                    attn_metadata.sparse_gather_phys is not None
                    and attn_metadata.sparse_gather_cu_seqlens_k is not None
                    and attn_metadata.sparse_q_head_gather is None
                    and attn_metadata.sparse_q_head_gather_flat_phys is None
                    and self.alibi_slopes is None
                    and self.sinks is None
                    and sliding_window_size == [-1, -1]
                    and not self.kv_cache_dtype.startswith("fp8")
                )
                use_per_head_bt = attn_metadata.sparse_per_head_block_table is not None
                if _SPARSE_PERF_DEBUG:
                    logger.info(
                        "[SparseDebug] FA sparse branch layer=%s num_tok=%d "
                        "use_q_head_gather=%s use_gather=%s use_per_head_bt=%s | "
                        "sparse_q_head_gather=%s sparse_q_head_gather_flat=%s sparse_gather_phys=%s sparse_per_head_bt=%s | "
                        "alibi=%s sinks=%s sliding=%s fp8_kv=%s",
                        getattr(layer, "__class__", type(layer)).__name__,
                        int(num_actual_tokens),
                        use_q_head_gather,
                        use_gather,
                        use_per_head_bt,
                        attn_metadata.sparse_q_head_gather is not None,
                        attn_metadata.sparse_q_head_gather_flat_phys is not None,
                        attn_metadata.sparse_gather_phys is not None,
                        attn_metadata.sparse_per_head_block_table is not None,
                        self.alibi_slopes is not None,
                        self.sinks is not None,
                        sliding_window_size,
                        bool(self.kv_cache_dtype.startswith("fp8")),
                    )
                if use_retroinfer:
                    self._forward_retroinfer_exec_buf(
                        query[:num_actual_tokens],
                        output[:num_actual_tokens],
                        attn_metadata,
                        q_descale=q_descale,
                        k_descale=k_descale,
                        v_descale=v_descale,
                    )
                    # Clear the per-layer runtime handle so the next step
                    # either re-populates it or cleanly falls back to the
                    # dense / legacy-sparse path.  Mirrors the clearing
                    # pattern used by the q_head_gather branch below.
                    layer._vllm_sparse_runtime_retroinfer = None
                elif use_q_head_gather:
                    self._forward_per_head_compact_kv_gather(
                        query[:num_actual_tokens],
                        key_cache,
                        value_cache,
                        output[:num_actual_tokens],
                        attn_metadata,
                        cu_seqlens_q,
                        max_seqlen_q,
                        q_descale,
                        k_descale,
                        v_descale,
                    )
                    if runtime_q_head_gather is not None:
                        layer._vllm_sparse_runtime_q_head_gather = None
                elif use_per_head_bt:
                    self._forward_per_head_block_table_sparse(
                        query[:num_actual_tokens],
                        key_cache,
                        value_cache,
                        output[:num_actual_tokens],
                        attn_metadata,
                        cu_seqlens_q,
                        max_seqlen_q,
                        max_seqlen_k,
                        scheduler_metadata,
                        q_descale,
                        k_descale,
                        v_descale,
                    )
                elif use_gather:
                    self._forward_compact_kv_gather(
                        query[:num_actual_tokens],
                        key_cache,
                        value_cache,
                        output[:num_actual_tokens],
                        attn_metadata,
                        cu_seqlens_q,
                        max_seqlen_q,
                        q_descale,
                        k_descale,
                        v_descale,
                    )
                else:
                    use_cuda_timing = _SPARSE_PERF_DEBUG and bool(query.is_cuda)
                    if use_cuda_timing:
                        total_start = torch.cuda.Event(enable_timing=True)
                        total_end = torch.cuda.Event(enable_timing=True)
                        fa_start = torch.cuda.Event(enable_timing=True)
                        fa_end = torch.cuda.Event(enable_timing=True)
                        total_start.record()
                        fa_start.record()
                    flash_attn_varlen_func(
                        q=query[:num_actual_tokens],
                        k=key_cache,
                        v=value_cache,
                        out=output[:num_actual_tokens],
                        cu_seqlens_q=cu_seqlens_q,
                        max_seqlen_q=max_seqlen_q,
                        seqused_k=seqused_k,
                        max_seqlen_k=max_seqlen_k,
                        softmax_scale=self.scale,
                        causal=attn_metadata.causal,
                        alibi_slopes=self.alibi_slopes,
                        window_size=sliding_window_size,
                        block_table=block_table,
                        softcap=self.logits_soft_cap,
                        scheduler_metadata=scheduler_metadata,
                        fa_version=self.vllm_flash_attn_version,
                        q_descale=q_descale,
                        k_descale=k_descale,
                        v_descale=v_descale,
                        num_splits=attn_metadata.max_num_splits,
                        s_aux=self.sinks,
                    )
                    if use_cuda_timing:
                        fa_end.record()
                        total_end.record()
                        total_end.synchronize()
                        fa_ms = float(fa_start.elapsed_time(fa_end))
                        total_ms = float(total_start.elapsed_time(total_end))
                        # Keep the full-attention comparison on the same
                        # decode-only footing as SparsePerfFA: one query token
                        # per sequence.
                        if int(max_seqlen_q) == 1:
                            logger.info(
                                "[FullPerfFA] paged_decode total_ms=%.3f fa_ms=%.3f "
                                "num_q_heads=%d num_tok=%d max_seqlen_q=%d max_seqlen_k=%d",
                                total_ms,
                                fa_ms,
                                int(query.shape[1]),
                                int(num_actual_tokens),
                                int(max_seqlen_q),
                                int(max_seqlen_k),
                            )
                return output

        # Cascade attention (rare case).
        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            max_num_splits=attn_metadata.max_num_splits,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            s_aux=self.sinks,
        )
        return output

    def _forward_compact_kv_gather(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        q_descale: torch.Tensor,
        k_descale: torch.Tensor,
        v_descale: torch.Tensor,
    ) -> None:
        """Gather selected K/V pages into a contiguous tensor and run varlen FA."""
        phys = attn_metadata.sparse_gather_phys.to(
            dtype=torch.int64, device=key_cache.device
        )
        slots = attn_metadata.sparse_gather_slots.to(
            dtype=torch.int64, device=key_cache.device
        )
        cu_k = attn_metadata.sparse_gather_cu_seqlens_k
        k_compact = key_cache[phys, slots]
        v_compact = value_cache[phys, slots]
        k_lens = cu_k[1:] - cu_k[:-1]
        max_seqlen_k = int(k_lens.max().item())
        win = (
            list(self.sliding_window)
            if self.sliding_window is not None
            else [-1, -1]
        )
        flash_attn_varlen_func(
            q=query,
            k=k_compact,
            v=v_compact,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_k,
            seqused_k=None,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=None,
            window_size=win,
            block_table=None,
            softcap=self.logits_soft_cap,
            scheduler_metadata=None,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=0,
            s_aux=None,
        )

    def _forward_per_head_compact_kv_gather(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        q_descale: torch.Tensor,
        k_descale: torch.Tensor,
        v_descale: torch.Tensor,
        ) -> None:
        """One compact-KV varlen FA call per query head (token-sparse)."""
        assert (
            attn_metadata.sparse_q_head_gather is not None
            or attn_metadata.sparse_q_head_gather_flat_phys is not None
        )
        flat_phys = attn_metadata.sparse_q_head_gather_flat_phys
        flat_slots = attn_metadata.sparse_q_head_gather_flat_slots
        flat_cu = attn_metadata.sparse_q_head_gather_flat_cu_seqlens_k
        head_offsets = attn_metadata.sparse_q_head_gather_head_offsets
        num_q_heads_gather = (
            int(head_offsets.numel() - 1)
            if head_offsets is not None
            else len(attn_metadata.sparse_q_head_gather)
        )
        if _SPARSE_PERF_DEBUG:
            logger.info(
                "[SparseDebug] _forward_per_head_compact_kv_gather ENTER "
                "num_q_heads_gather=%d q_shape0=%d max_seqlen_q=%d",
                num_q_heads_gather,
                int(query.shape[0]),
                int(max_seqlen_q),
            )
        total_ms = 0.0
        gather_ms = 0.0
        fa_ms = 0.0
        use_cuda_timing = _SPARSE_PERF_DEBUG and bool(query.is_cuda)
        win = (
            list(self.sliding_window)
            if self.sliding_window is not None
            else [-1, -1]
        )
        if use_cuda_timing:
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            total_start.record()
        # Pessimistic upper bound for FA's max_seqlen_k.  The runtime gather
        # builder attaches an ``int`` budget via ``sparse_q_head_gather_max_k``;
        # using that avoids the per-call ``.max().item()`` GPU→CPU sync.
        max_k_hint = attn_metadata.sparse_q_head_gather_max_k
        if (
            flat_phys is not None
            and flat_slots is not None
            and flat_cu is not None
            and head_offsets is not None
            and int(max_seqlen_q) == 1
            and int(query.shape[0]) == int(cu_seqlens_q.numel() - 1)
        ):
            if use_cuda_timing:
                gather_start = torch.cuda.Event(enable_timing=True)
                gather_end = torch.cuda.Event(enable_timing=True)
                fa_start = torch.cuda.Event(enable_timing=True)
                fa_end = torch.cuda.Event(enable_timing=True)
                gather_start.record()
            num_reqs = int(query.shape[0])
            # Tier-2 fast path: the sparse runtime gather builder has already
            # moved the int64 conversions, kv_token_ids repeat, cu_k_flat
            # cumsum and descale index templates into one-shot step-scoped
            # kernels, exposed via ``attn_metadata``.  When present, this
            # branch runs only the two advanced-indexing gathers + Q layout
            # copy + one FA launch (bitwise-equivalent to the legacy in-FA
            # rebuild below, but ~15 fewer tiny kernel launches per layer).
            phys64_pre = attn_metadata.sparse_q_head_gather_phys_int64
            slots64_pre = attn_metadata.sparse_q_head_gather_slots_int64
            kv_token_ids_pre = attn_metadata.sparse_q_head_gather_kv_token_ids
            cu_k_flat_pre = attn_metadata.sparse_q_head_gather_cu_k_flat
            cu_q_flat_pre = attn_metadata.sparse_q_head_gather_cu_q_flat
            req_ids_pre = attn_metadata.sparse_q_head_gather_req_ids_flat
            kv_pair_ids_pre = (
                attn_metadata.sparse_q_head_gather_kv_pair_ids_flat
            )
            have_pre = (
                phys64_pre is not None
                and slots64_pre is not None
                and kv_token_ids_pre is not None
                and cu_k_flat_pre is not None
                and cu_q_flat_pre is not None
                and req_ids_pre is not None
                and kv_pair_ids_pre is not None
            )
            if have_pre:
                phys64 = phys64_pre
                slots64 = slots64_pre
                kv_token_ids = kv_token_ids_pre
                cu_q_flat = cu_q_flat_pre
                cu_k_flat = cu_k_flat_pre
                req_ids = req_ids_pre
                kv_pair_ids = kv_pair_ids_pre
            else:
                # Legacy fallback: rebuild everything from ``flat_*`` tensors.
                # Kept so the FA path still works with builders that return
                # the pre-Tier-2 dict schema (older clients, partial rollout).
                head_ids = torch.arange(
                    num_q_heads_gather, dtype=torch.int64, device=query.device
                )
                kv_head_ids = head_ids // max(self.num_queries_per_kv, 1)
                head_token_counts = (
                    head_offsets[1:].to(dtype=torch.int64)
                    - head_offsets[:-1].to(dtype=torch.int64)
                )
                kv_token_ids = kv_head_ids.repeat_interleave(head_token_counts)
                phys64 = flat_phys.to(
                    dtype=torch.int64, device=key_cache.device
                )
                slots64 = flat_slots.to(
                    dtype=torch.int64, device=key_cache.device
                )
                kv_token_ids = kv_token_ids.to(device=key_cache.device)
                q_lens = torch.ones(
                    num_q_heads_gather * num_reqs,
                    dtype=cu_seqlens_q.dtype,
                    device=query.device,
                )
                cu_q_flat = torch.empty(
                    int(q_lens.numel()) + 1,
                    dtype=cu_seqlens_q.dtype,
                    device=query.device,
                )
                cu_q_flat[0] = 0
                cu_q_flat[1:] = torch.cumsum(q_lens, dim=0)
                k_lens_fallback = (flat_cu[:, 1:] - flat_cu[:, :-1]).reshape(-1)
                cu_k_flat = torch.empty(
                    int(k_lens_fallback.numel()) + 1,
                    dtype=flat_cu.dtype,
                    device=flat_cu.device,
                )
                cu_k_flat[0] = 0
                cu_k_flat[1:] = torch.cumsum(k_lens_fallback, dim=0)
                req_ids = torch.arange(
                    num_reqs, dtype=torch.int64, device=query.device
                ).repeat(num_q_heads_gather)
                kv_pair_ids = kv_head_ids.repeat_interleave(num_reqs)

            # The hot-path gather: these two advanced-indexing reads are the
            # only genuinely Tier-3 (per-layer, K/V-cache-dependent) work
            # that cannot be pre-computed.
            if _SPARSE_DEBUG_ASSERT:
                _sparse_assert_compact_gather_inputs(
                    phys=phys64,
                    slots=slots64,
                    kv_heads=kv_token_ids,
                    cu_k=cu_k_flat,
                    key_cache=key_cache,
                    num_q_flat=num_q_heads_gather * num_reqs,
                )
            k_slice = key_cache[phys64, slots64, kv_token_ids]
            v_slice = value_cache[phys64, slots64, kv_token_ids]
            k_compact = k_slice.unsqueeze(1)
            v_compact = v_slice.unsqueeze(1)

            q_flat = (
                query.transpose(0, 1)
                .contiguous()
                .view(num_q_heads_gather * num_reqs, 1, query.shape[-1])
            )
            out_flat = torch.empty_like(q_flat)

            # ``max_seqlen_k`` uses the static budget hint from the builder
            # when available, avoiding a per-call ``.max().item()`` sync.
            if max_k_hint is not None:
                max_seqlen_k = int(max_k_hint)
            else:
                k_lens_sync = (flat_cu[:, 1:] - flat_cu[:, :-1]).reshape(-1)
                max_seqlen_k = int(k_lens_sync.max().item())

            q_descale_flat = q_descale[req_ids, kv_pair_ids].view(-1, 1)
            k_descale_flat = k_descale[req_ids, kv_pair_ids].view(-1, 1)
            v_descale_flat = v_descale[req_ids, kv_pair_ids].view(-1, 1)
            if use_cuda_timing:
                gather_end.record()
                fa_start.record()
            flash_attn_varlen_func(
                q=q_flat,
                k=k_compact,
                v=v_compact,
                out=out_flat,
                cu_seqlens_q=cu_q_flat,
                max_seqlen_q=1,
                cu_seqlens_k=cu_k_flat,
                seqused_k=None,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=None,
                window_size=win,
                block_table=None,
                softcap=self.logits_soft_cap,
                scheduler_metadata=None,
                fa_version=self.vllm_flash_attn_version,
                q_descale=q_descale_flat,
                k_descale=k_descale_flat,
                v_descale=v_descale_flat,
                num_splits=0,
                s_aux=None,
            )
            output.copy_(
                out_flat.view(num_q_heads_gather, num_reqs, query.shape[-1])
                .transpose(0, 1)
                .contiguous()
            )
            if use_cuda_timing:
                fa_end.record()
                fa_end.synchronize()
                gather_ms += float(gather_start.elapsed_time(gather_end))
                fa_ms += float(fa_start.elapsed_time(fa_end))
        else:
            for qh in range(num_q_heads_gather):
                if flat_phys is not None:
                    assert flat_slots is not None
                    assert flat_cu is not None
                    assert head_offsets is not None
                    off0 = int(head_offsets[qh].item())
                    off1 = int(head_offsets[qh + 1].item())
                    phys = flat_phys[off0:off1]
                    slots = flat_slots[off0:off1]
                    cu_k = flat_cu[qh]
                else:
                    assert attn_metadata.sparse_q_head_gather is not None
                    phys, slots, cu_k = attn_metadata.sparse_q_head_gather[qh]
                if use_cuda_timing:
                    gather_start = torch.cuda.Event(enable_timing=True)
                    gather_end = torch.cuda.Event(enable_timing=True)
                    fa_start = torch.cuda.Event(enable_timing=True)
                    fa_end = torch.cuda.Event(enable_timing=True)
                    gather_start.record()
                phys64 = phys.to(dtype=torch.int64, device=key_cache.device)
                slots64 = slots.to(dtype=torch.int64, device=key_cache.device)
                kv_h = qh // max(self.num_queries_per_kv, 1)
                if _SPARSE_DEBUG_ASSERT:
                    if int(cu_k[0].item()) != 0 or int(cu_k[-1].item()) != int(
                        phys64.numel()
                    ):
                        raise ValueError(
                            "sparse compact gather per-head cu_k endpoints "
                            f"mismatch for qh={qh}: first={int(cu_k[0].item())} "
                            f"last={int(cu_k[-1].item())} n={int(phys64.numel())}"
                        )
                    if cu_k.numel() > 1 and bool(
                        (cu_k[1:] < cu_k[:-1]).any().item()
                    ):
                        raise ValueError(
                            f"sparse compact gather per-head cu_k is not "
                            f"monotonic for qh={qh}"
                        )
                    _sparse_tensor_range(
                        f"sparse phys qh={qh}", phys64, int(key_cache.shape[0])
                    )
                    _sparse_tensor_range(
                        f"sparse slots qh={qh}", slots64, int(key_cache.shape[1])
                    )
                    if kv_h < 0 or kv_h >= int(key_cache.shape[2]):
                        raise ValueError(
                            f"sparse kv_head out of range for qh={qh}: "
                            f"{kv_h} not in [0, {int(key_cache.shape[2])})"
                        )
                k_slice = key_cache[phys64, slots64, kv_h]
                v_slice = value_cache[phys64, slots64, kv_h]
                k_compact = k_slice.unsqueeze(1)
                v_compact = v_slice.unsqueeze(1)
                if max_k_hint is not None:
                    max_seqlen_k = int(max_k_hint)
                else:
                    k_lens = cu_k[1:] - cu_k[:-1]
                    max_seqlen_k = int(k_lens.max().item())
                if use_cuda_timing:
                    gather_end.record()
                    fa_start.record()
                flash_attn_varlen_func(
                    q=query[:, qh : qh + 1, :],
                    k=k_compact,
                    v=v_compact,
                    out=output[:, qh : qh + 1, :],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    cu_seqlens_k=cu_k,
                    seqused_k=None,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=None,
                    window_size=win,
                    block_table=None,
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=None,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=q_descale[:, kv_h : kv_h + 1],
                    k_descale=k_descale[:, kv_h : kv_h + 1],
                    v_descale=v_descale[:, kv_h : kv_h + 1],
                    num_splits=0,
                    s_aux=None,
                )
                if use_cuda_timing:
                    fa_end.record()
                    fa_end.synchronize()
                    gather_ms += float(gather_start.elapsed_time(gather_end))
                    fa_ms += float(fa_start.elapsed_time(fa_end))
        if use_cuda_timing:
            total_end.record()
            total_end.synchronize()
            total_ms = float(total_start.elapsed_time(total_end))
            logger.info(
                "[SparsePerfFA] compact_kv_gather total_ms=%.3f gather_ms=%.3f "
                "fa_ms=%.3f num_q_heads=%d num_tok=%d max_seqlen_q=%d",
                total_ms,
                gather_ms,
                fa_ms,
                num_q_heads_gather,
                int(query.shape[0]),
                int(max_seqlen_q),
            )

    def _forward_retroinfer_exec_buf(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: "FlashAttentionMetadata",
        *,
        q_descale: torch.Tensor,
        k_descale: torch.Tensor,
        v_descale: torch.Tensor,
    ) -> None:
        """Phase 5 — retroinfer-style two-zone attention.

        The runner has already:
          1. Picked ``retrieval`` + ``estimation`` cluster IDs via
             ``GPUModelRunner._sparse_online_select_clusters_batched``.
          2. Materialised the retrieval zone as per-head fixed-shape
             execution buffers
             (``exec_buf_k/v: [H, max_budget, D]`` + ``valid_lengths: [H]``)
             via ``_sparse_retroinfer_expand_and_gather_single_req``.
          3. Copied the estimation zone's centroids / cluster value-sums /
             cluster sizes into matching ``[H, es, D]`` / ``[H, es, D]`` /
             ``[H, es]`` tensors.

        This method runs **one** FA launch over the pre-gathered exec_buf
        (treating each query head as an FA "batch" so ``seqused_k`` can
        carry per-head valid lengths against a fixed ``max_budget`` K-cache
        stride), then evaluates the estimation zone's closed-form softmax
        and merges the two via LSE.  The output layout
        (``[num_tokens=1, H, D]``) matches what callers expect from the
        legacy per-head compact gather path, so the upstream attention
        layer sees no change.

        Decode-only fast path: ``query.shape[0] == 1``.  Prefill falls back
        to the legacy gather path via the retroinfer dispatch guard.
        """
        assert attn_metadata.sparse_retroinfer_exec_buf_k is not None
        assert attn_metadata.sparse_retroinfer_exec_buf_v is not None
        assert attn_metadata.sparse_retroinfer_valid_lengths is not None
        exec_buf_k = attn_metadata.sparse_retroinfer_exec_buf_k
        exec_buf_v = attn_metadata.sparse_retroinfer_exec_buf_v
        valid_lengths = attn_metadata.sparse_retroinfer_valid_lengths
        centres_es = attn_metadata.sparse_retroinfer_centres_es
        value_sum_es = attn_metadata.sparse_retroinfer_value_sum_es
        cluster_size_es = attn_metadata.sparse_retroinfer_cluster_size_es

        assert query.shape[0] == 1, (
            "retroinfer exec_buf path is decode-only (num_tokens==1); "
            "caller must dispatch prefill through the legacy gather path"
        )
        H = int(query.shape[1])
        D = int(query.shape[2])
        assert exec_buf_k.shape[0] == H and exec_buf_k.shape[2] == D
        max_budget = int(exec_buf_k.shape[1])
        device = query.device

        use_cuda_timing = _SPARSE_PERF_DEBUG and bool(query.is_cuda)
        if use_cuda_timing:
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            fa_start = torch.cuda.Event(enable_timing=True)
            fa_end = torch.cuda.Event(enable_timing=True)
            est_start = torch.cuda.Event(enable_timing=True)
            est_end = torch.cuda.Event(enable_timing=True)
            total_start.record()

        # ── retrieval zone: varlen FA over ``[H, max_budget, D]`` ──
        # Heads-as-batches with ``num_heads_q=1`` keeps FA's seqlen
        # dispatch working per head without needing a second varlen call.
        # Memory layout:
        #   q_flat : [H, 1, D]       (one query per "batch")
        #   k_flat : [H*max_budget, 1, D]
        #   v_flat : [H*max_budget, 1, D]
        # Packed K layout: batch ``b`` occupies rows ``[b*max_budget,
        # (b+1)*max_budget)`` in ``k_flat``.  Per-batch **valid** key count
        # is ``seqused_k[b]`` (``<= max_budget``); padding rows are never
        # read.  **Do not** pass ``cu_seqlens_k`` together with ``seqused_k``
        # – ``flash_attn_varlen_func`` asserts they are mutually exclusive;
        # the FA2 path substitutes an internal dummy ``cu_seqlens_k`` when
        # only ``seqused_k`` is supplied (see ``flash_attn_interface.py``).
        q_flat = (
            query.view(1, H, D).transpose(0, 1).contiguous().view(H, 1, D)
        )
        if _SPARSE_DEBUG_ASSERT:
            if exec_buf_v.shape != exec_buf_k.shape:
                raise ValueError(
                    "retroinfer exec buffers must have identical shape: "
                    f"k={tuple(exec_buf_k.shape)} v={tuple(exec_buf_v.shape)}"
                )
            _sparse_tensor_range(
                "retroinfer valid_lengths",
                valid_lengths.to(dtype=torch.int64),
                max_budget + 1,
            )

        # Varlen packing: FA's paged-KV contract only supports
        # block_size % 16 == 0, and with a single block per "batch" it was
        # also observed to silently corrupt outputs at non-trivial
        # max_budget values on FA2.  Compact the valid per-head prefixes
        # of ``exec_buf_k/v`` into a flat ``[sum(valid_lengths), 1, D]``
        # tensor and hand FA a plain varlen call with ``cu_seqlens_k`` –
        # the same pattern the legacy ``_forward_per_head_compact_kv_gather``
        # uses successfully.
        cu_q = torch.arange(0, H + 1, dtype=torch.int32, device=device)
        valid_i64 = valid_lengths.to(dtype=torch.int64)
        cu_k = torch.empty(H + 1, dtype=torch.int32, device=device)
        cu_k[0] = 0
        cu_k[1:] = torch.cumsum(valid_i64, dim=0).to(torch.int32)
        # Build advanced-index gather: for each batch h, copy
        # exec_buf[h, 0:valid[h]] to rows cu_k[h]:cu_k[h+1] of the flat
        # output.  ``batch_idx_flat[i]`` = which head row i came from;
        # ``row_in_batch_flat[i]`` = 0..valid[h]-1 within that head.
        total_k = int(cu_k[-1].item())
        if total_k == 0:
            # Degenerate: no K tokens for any head this step.  Return zero
            # output and a sentinel LSE so the estimation merge below
            # behaves sanely.  Also fire the FA-timing events so the
            # downstream ``fa_end.record()`` has a valid ``fa_start`` to
            # elapsed-time from.
            if use_cuda_timing:
                fa_start.record()
            out_ret_h = torch.zeros((H, D), dtype=query.dtype, device=device)
            lse_ret_h = torch.full(
                (H,), float("-inf"), dtype=torch.float32, device=device
            )
        else:
            batch_idx_flat = torch.repeat_interleave(
                torch.arange(H, dtype=torch.int64, device=device),
                valid_i64,
            )  # [total_k]
            row_in_batch_flat = (
                torch.arange(total_k, dtype=torch.int64, device=device)
                - cu_k[batch_idx_flat].to(torch.int64)
            )
            # Gather into packed varlen layout [total_k, 1, D].
            k_packed = exec_buf_k[batch_idx_flat, row_in_batch_flat].view(
                total_k, 1, D
            )
            v_packed = exec_buf_v[batch_idx_flat, row_in_batch_flat].view(
                total_k, 1, D
            )

            # Per-batch descale: FA expects ``[batch, num_heads]`` = ``[H, 1]``.
            # Upstream ``q_descale`` etc. are ``[num_reqs=1, num_kv_heads]``;
            # each q-head reads from its mapped kv-head.
            q_per_kv = max(self.num_queries_per_kv, 1)
            kv_head_ids = (
                torch.arange(H, dtype=torch.long, device=device) // q_per_kv
            )
            q_des_ret = q_descale[0, kv_head_ids].view(H, 1)
            k_des_ret = k_descale[0, kv_head_ids].view(H, 1)
            v_des_ret = v_descale[0, kv_head_ids].view(H, 1)

            # ``max_seqlen_k`` is the per-batch max of valid_lengths.  Using
            # max_budget as a safe upper bound avoids an extra sync.
            max_seqlen_k_varlen = int(max_budget)

            if use_cuda_timing:
                fa_start.record()
            # Varlen FA over the packed K/V.  ``causal=False`` is equivalent
            # here because every K position is <= the current query position
            # by construction (retrieval < prefill_len, steady ends at
            # prefill_len, pending is [prefill_len, current_len)).
            out_ret, lse_ret = flash_attn_varlen_func(
                q=q_flat,
                k=k_packed,
                v=v_packed,
                out=None,
                cu_seqlens_q=cu_q,
                max_seqlen_q=1,
                cu_seqlens_k=cu_k,
                seqused_k=None,
                max_seqlen_k=max_seqlen_k_varlen,
                softmax_scale=self.scale,
                causal=False,
                alibi_slopes=None,
                window_size=[-1, -1],
                block_table=None,
                softcap=self.logits_soft_cap,
                return_softmax_lse=True,
                fa_version=self.vllm_flash_attn_version,
                q_descale=q_des_ret,
                k_descale=k_des_ret,
                v_descale=v_des_ret,
                num_splits=0,
            )
            # FA reports ``softmax_lse`` with shape ``[num_heads_q, total_q]``
            # = ``[1, H]`` in our heads-as-batches layout.
            lse_ret_h = lse_ret.view(H).to(dtype=torch.float32)
            out_ret_h = out_ret.view(H, D)
        if use_cuda_timing:
            fa_end.record()

        es = int(centres_es.shape[1]) if centres_es is not None else 0
        if es == 0:
            # Retrieval-only: no estimation zone allocated this step.
            output.view(1, H, D).copy_(
                out_ret_h.view(1, H, D).to(dtype=output.dtype)
            )
            if use_cuda_timing:
                total_end.record()
                total_end.synchronize()
                total_ms = float(total_start.elapsed_time(total_end))
                fa_ms = float(fa_start.elapsed_time(fa_end))
                logger.info(
                    "[SparsePerfFA] retroinfer total_ms=%.3f fa_ms=%.3f "
                    "est_ms=0.000 H=%d max_budget=%d es=0",
                    total_ms,
                    fa_ms,
                    H,
                    max_budget,
                )
            return

        # ── estimation zone: closed-form softmax over cluster stats ──
        # Mirror of ``GPUModelRunner._sparse_retroinfer_estimation_attn``
        # but kept inline so FA has no reverse import on the runner.  Math
        # is kept in fp32 for numerical stability; FA's dtype is restored
        # at the final ``output.copy_`` cast.
        if use_cuda_timing:
            est_start.record()
        assert value_sum_es is not None and cluster_size_es is not None
        q_est = query.view(H, D).to(dtype=torch.float32)
        c_f = centres_es.to(dtype=torch.float32)
        vs_f = value_sum_es.to(dtype=torch.float32)
        sz_f = cluster_size_es.to(dtype=torch.float32).clamp_min(1.0)
        logits = torch.einsum("hd,hed->he", q_est, c_f) * float(self.scale)
        logits_max = logits.max(dim=-1, keepdim=True).values  # [H, 1]
        w_est = torch.exp(logits - logits_max)                # [H, es]
        num = torch.einsum("he,hed->hd", w_est, vs_f)         # [H, D]
        den = (w_est * sz_f).sum(dim=-1, keepdim=True)        # [H, 1]
        out_est = num / den.clamp_min(1e-12)                  # [H, D]
        # ``lse`` in FA's absolute reference frame (log of unshifted
        # softmax denominator) so we can merge with ``lse_ret_h`` directly.
        lse_est = (
            logits_max.squeeze(-1)
            + torch.log(den.squeeze(-1).clamp_min(1e-12))
        )                                                     # [H]

        # ── LSE merge (stable two-zone softmax average) ──
        la = lse_ret_h
        lb = lse_est
        m = torch.maximum(la, lb)
        w_a = torch.exp(la - m)
        w_b = torch.exp(lb - m)
        w_sum = w_a + w_b
        out_ret_f = out_ret_h.to(dtype=torch.float32)
        out_final = (
            (w_a.unsqueeze(-1) * out_ret_f + w_b.unsqueeze(-1) * out_est)
            / w_sum.unsqueeze(-1).clamp_min(1e-12)
        )
        output.view(1, H, D).copy_(
            out_final.view(1, H, D).to(dtype=output.dtype)
        )
        if use_cuda_timing:
            est_end.record()
            total_end.record()
            total_end.synchronize()
            total_ms = float(total_start.elapsed_time(total_end))
            fa_ms = float(fa_start.elapsed_time(fa_end))
            est_ms = float(est_start.elapsed_time(est_end))
            logger.info(
                "[SparsePerfFA] retroinfer total_ms=%.3f fa_ms=%.3f "
                "est_ms=%.3f H=%d max_budget=%d es=%d",
                total_ms,
                fa_ms,
                est_ms,
                H,
                max_budget,
                es,
            )

    def _forward_per_head_block_table_sparse(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        scheduler_metadata: torch.Tensor | None,
        q_descale: torch.Tensor,
        k_descale: torch.Tensor,
        v_descale: torch.Tensor,
    ) -> None:
        """Paged KV with a different block table / seq_len per query head."""
        assert attn_metadata.sparse_per_head_block_table is not None
        assert attn_metadata.sparse_per_head_seq_lens is not None
        H = int(attn_metadata.sparse_per_head_block_table.shape[0])
        win = (
            list(self.sliding_window)
            if self.sliding_window is not None
            else [-1, -1]
        )
        # Cache per-KV-head views to avoid repeated re-slicing/re-packing in the
        # hot per-query-head loop.
        kv_head_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for qh in range(H):
            bt_h = attn_metadata.sparse_per_head_block_table[qh]
            sl_h = attn_metadata.sparse_per_head_seq_lens[qh]
            msk = int(sl_h.max().item()) if sl_h.numel() > 0 else max_seqlen_k
            kv_h = qh // max(self.num_queries_per_kv, 1)
            # GQA: Q is sliced to one head; K/V must use the matching KV head only,
            # otherwise FA requires num_q_heads % num_kv_heads == 0 (fails for 1 vs N).
            kv_pair = kv_head_cache.get(kv_h)
            if kv_pair is None:
                # Conservative path: keep contiguous K/V once per KV head, then
                # reuse in all query heads mapped to this KV head.
                kv_pair = (
                    key_cache[..., kv_h : kv_h + 1, :].contiguous(),
                    value_cache[..., kv_h : kv_h + 1, :].contiguous(),
                )
                kv_head_cache[kv_h] = kv_pair
            k_h, v_h = kv_pair
            alibi_h = self.alibi_slopes
            if alibi_h is not None and alibi_h.ndim > 0:
                alibi_h = alibi_h[qh : qh + 1]
            sinks_h = self.sinks
            if sinks_h is not None and sinks_h.ndim > 0:
                sinks_h = sinks_h[qh : qh + 1]
            flash_attn_varlen_func(
                q=query[:, qh : qh + 1, :],
                k=k_h,
                v=v_h,
                out=output[:, qh : qh + 1, :],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=sl_h,
                max_seqlen_k=msk,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=alibi_h,
                window_size=win,
                block_table=bt_h,
                softcap=self.logits_soft_cap,
                scheduler_metadata=scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
                q_descale=q_descale[:, kv_h : kv_h + 1],
                k_descale=k_descale[:, kv_h : kv_h + 1],
                v_descale=v_descale[:, kv_h : kv_h + 1],
                num_splits=attn_metadata.max_num_splits,
                s_aux=sinks_h,
            )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return

        key_cache, value_cache = kv_cache.unbind(0)

        # Reshape the input keys and values and store them in the cache.
        # Skip this if sharing KV cache with an earlier attention layer.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens]
        # and value[:num_actual_tokens] because the reshape_and_cache_flash
        # op uses the slot_mapping's shape to determine the number of
        # actual tokens.
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def _forward_with_dcp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        q_descale: torch.Tensor | None = None,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        cu_seqlens_q = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        block_table = attn_metadata.block_table

        query = query.contiguous()
        query_across_dcp = get_dcp_group().all_gather(query, dim=1)
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        context_attn_out, context_lse = flash_attn_varlen_func(
            q=query_across_dcp,
            k=key_cache,
            v=value_cache,
            out=None,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=attn_metadata.dcp_context_kv_lens,
            max_seqlen_k=attn_metadata.max_dcp_context_kv_len,
            softmax_scale=self.scale,
            causal=False,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=attn_metadata.max_num_splits,
        )
        # FA returns LSE in shape [ H, B ] but DCP combine wants [ B, H ]
        context_attn_out_cor, context_lse_cor = self.dcp_combine(
            context_attn_out,
            context_lse.transpose(0, 1),
            get_dcp_group(),
            return_lse=True,
        )
        context_lse_cor = context_lse_cor.transpose(0, 1).contiguous()

        query_attn_out, query_lse = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=None,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_k=max_seqlen_q,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=attn_metadata.max_num_splits,
        )
        assert context_attn_out_cor.shape == query_attn_out.shape
        assert context_lse_cor.shape == query_lse.shape
        merge_attn_states(
            output,
            context_attn_out_cor,
            context_lse_cor,
            query_attn_out,
            query_lse,
        )

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        # For encoder attention, process FP8 quantization if needed
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        cu_seqlens_q = attn_metadata.query_start_loc
        cu_seqlens_k = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_query_len

        descale_shape = (
            cu_seqlens_q.shape[0] - 1,  # type: ignore[union-attr]
            self.num_kv_heads,
        )

        # Call flash attention directly on Q, K, V tensors
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,  # Encoder attention is bidirectional
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape),
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            num_splits=1 if self.batch_invariant_enabled else 0,
        )

        return output


def use_cascade_attention(
    common_prefix_len: int,
    query_lens: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    use_alibi: bool,
    use_sliding_window: bool,
    use_local_attention: bool,
    num_sms: int,
    dcp_world_size: int,
) -> bool:
    """Decide whether to use cascade attention.

    This function 1) checks whether cascade attention is supported with the
    given configuration, and 2) heuristically decides whether using cascade
    attention can improve performance.
    """
    # Too short common prefix. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 256 tokens. TODO: Tune this threshold.
    # NOTE(woosuk): This is the common case. We should return False as soon as
    # possible to avoid any unnecessary computation.
    if common_prefix_len < 256:
        return False
    # Cascade attention is currently not supported with these variants.
    if use_alibi or use_sliding_window or use_local_attention:
        return False
    # Too few queries. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 8 queries. TODO: Tune this threshold.
    num_reqs = len(query_lens)
    if num_reqs < 8:
        return False
    # disable cascade attention for DCP
    if dcp_world_size > 1:
        return False

    # Heuristics to decide whether using cascade attention is beneficial.
    # 1. When FlashDecoding is not used for normal attention, cascade attention
    #    is likely to be faster since it saves memory bandwidth.
    num_queries_per_kv = num_query_heads // num_kv_heads
    # The criteria for using FlashDecoding can be found in the following link:
    # https://github.com/vllm-project/flash-attention/blob/96266b1111111f3d11aabefaf3bacbab6a89d03c/csrc/flash_attn/flash_api.cpp#L535
    use_flash_decoding = (
        num_queries_per_kv > 1
        and not use_sliding_window
        and not use_alibi
        and np.all(query_lens == 1)
    )
    if not use_flash_decoding:
        # Use cascade attention.
        return True

    # 2. When FlashDecoding is used for normal attention, it is not clear
    #    whether cascade attention is beneficial, because FlashDecoding can
    #    launch more CTAs than cascade attention.
    #    We use a simple performance model to compare the two methods.
    #    NOTE(woosuk): The performance model is very rough and may not be
    #    accurate.
    num_tokens = num_reqs
    # NOTE(woosuk): These are default tile sizes. flash-attn might use
    # different tile sizes (e.g., 64 or 256) depending on the configuration.
    q_tile_size = 128
    kv_tile_size = 128
    num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

    cascade_ctas = num_query_heads * cdiv(num_tokens, q_tile_size)
    cascade_waves = cdiv(cascade_ctas, num_sms)
    cascade_time = cascade_waves * num_prefix_tiles

    flash_decoding_ctas = (
        num_reqs * num_kv_heads * cdiv(num_queries_per_kv, q_tile_size)
    )
    flash_decoding_ctas *= num_prefix_tiles
    flash_decoding_time = cdiv(flash_decoding_ctas, num_sms)

    # Use cascade attention if it is faster than FlashDecoding.
    return cascade_time < flash_decoding_time


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    max_num_splits: int,
    fa_version: int,
    prefix_scheduler_metadata: torch.Tensor | None = None,
    suffix_scheduler_metadata: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
) -> torch.Tensor:
    assert alibi_slopes is None, "Cascade attention does not support ALiBi."
    # TODO: Support sliding window.
    assert sliding_window == (-1, -1), (
        "Cascade attention does not support sliding window."
    )

    num_tokens = query.shape[0]
    block_size = key_cache.shape[-3]
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    assert num_common_kv_blocks > 0
    descale_shape = (cu_prefix_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process shared prefix.
    prefix_output, prefix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        seqused_k=prefix_kv_lens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=list(sliding_window),
        block_table=block_table[:1],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=prefix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape) if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape) if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape) if v_descale is not None else None,
        # s_aux is incorporated into prefix_lse inside the GPU kernel,
        # enabling its effect during the final attention merge.
        s_aux=s_aux,
        num_splits=1 if vllm_is_batch_invariant() else max_num_splits,
    )

    descale_shape = (cu_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process suffix per query.
    suffix_output, suffix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=suffix_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=list(sliding_window),
        block_table=block_table[:, num_common_kv_blocks:],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=suffix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape) if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape) if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape) if v_descale is not None else None,
        num_splits=1 if vllm_is_batch_invariant() else max_num_splits,
    )

    # Merge prefix and suffix outputs, and store the result in output.
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
