# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from dataclasses import dataclass, fields, replace
from math import prod
from typing import Literal

import torch
from typing_extensions import Self

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size

logger = init_logger(__name__)


@dataclass(frozen=True)
class KVCacheSpec:
    """
    A base class for specifying the KV cache format of one layer.
    """

    # number of tokens in a block
    block_size: int

    @property
    def page_size_bytes(self) -> int:
        """
        The size of a page with `block_size` tokens in bytes.

        Returns:
            The page size
        """
        raise NotImplementedError

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        """
        The maximum possible memory usage of this KV cache in bytes.

        Returns:
            The KV cache size in bytes
        """
        raise NotImplementedError

    def copy_with_new_block_size(self, block_size: int) -> Self:
        """
        Create a new KVCacheSpec from self but replacing the block size.
        """
        return replace(self, block_size=block_size)

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of KVCacheSpec objects into a single KVCacheSpec object.
        """
        assert all(spec == specs[0] for spec in specs[1:]), (
            "All layers in the same KV cache group must be the same."
        )
        return copy.deepcopy(specs[0])


@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    page_size_padded: int | None = None

    @property
    def page_size_bytes(self) -> int:
        real_page_size = self.real_page_size_bytes
        if self.page_size_padded is not None:
            assert self.page_size_padded >= real_page_size
            return self.page_size_padded
        return real_page_size

    @property
    def real_page_size_bytes(self) -> int:
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )


@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):
    """
    When hybrid allocator is disabled and the model contains both full
    attention layers and sliding window attention layers, sliding
    window attention are regarded as full attention in KV cache manager
    (blocks are allocated for all tokens), while computed as sliding window
    attention in model runner.
    In this case, we use FullAttentionSpec and record the sliding window size.
    """

    head_size_v: int = None  # type: ignore[assignment]

    sliding_window: int | None = None
    """
    Default to None for not using sliding window attention.
    """
    attention_chunk_size: int | None = None

    def __post_init__(self):
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
        # Note(hc): each dcp rank only need save
        # (max_model_len//dcp_world_size) tokens locally.
        if dcp_world_size * pcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size * pcp_world_size)
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes

    @classmethod
    def merge_window_sizes(cls, window_sizes: set[int]) -> int | None:
        if len(window_sizes) == 0:
            return None
        elif len(window_sizes) == 1:
            return window_sizes.pop()
        else:
            raise ValueError(
                "All attention layers in the same KV cache group must have the "
                "same window size."
            )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of FullAttentionSpec objects into a single
        FullAttentionSpec object.
        """
        assert all(isinstance(spec, FullAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be FullAttentionSpec."
        )

        sliding_window = set(
            spec.sliding_window for spec in specs if spec.sliding_window is not None
        )
        attention_chunk_size = set(
            spec.attention_chunk_size
            for spec in specs
            if spec.attention_chunk_size is not None
        )
        assert not any(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "MLAAttentionSpec should be merged in MLAAttentionSpec.merge"
        )
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            head_size_v=specs[0].head_size_v,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            sliding_window=cls.merge_window_sizes(sliding_window),
            attention_chunk_size=cls.merge_window_sizes(attention_chunk_size),
        )
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All attention layers in the same KV cache group must have "
                    "the same attention spec."
                )
        assert (merged_spec.sliding_window is not None) + (
            merged_spec.attention_chunk_size is not None
        ) <= 1, (
            "Model with both sliding window layers and chunked local attention "
            "layers is not supported."
        )
        return merged_spec

    @property
    def real_page_size_bytes(self) -> int:
        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size + self.head_size_v)
            * get_dtype_size(self.dtype)
        )


@dataclass(frozen=True, kw_only=True)
class MLAAttentionSpec(FullAttentionSpec):
    # TODO(Lucas/Chen): less hacky way to do this
    cache_dtype_str: str | None = None

    @property
    def real_page_size_bytes(self) -> int:
        if self.cache_dtype_str == "fp8_ds_mla":
            # See `vllm/v1/attention/backends/mla/flashmla_sparse.py`
            #  for details.
            return self.block_size * 656
        return (
            self.block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be MLAAttentionSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        assert len(cache_dtype_str_set) == 1, (
            "All attention layers in the same KV cache group must use the same "
            "quantization method."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            cache_dtype_str=cache_dtype_str_set.pop(),
        )


@dataclass(frozen=True, kw_only=True)
class ChunkedLocalAttentionSpec(AttentionSpec):
    attention_chunk_size: int

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        # During chunked prefill, we allocate KV cache for at most
        # `self.attention_chunk_size` computed tokens plus the newly scheduled
        # tokens. And we won't allocate KV cache for more than `max_model_len`
        # tokens.
        num_tokens = min(
            self.attention_chunk_size + max_num_batched_tokens, max_model_len
        )

        return cdiv(num_tokens, self.block_size) * self.page_size_bytes


@dataclass(frozen=True, kw_only=True)
class SlidingWindowSpec(AttentionSpec):
    sliding_window: int

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        assert vllm_config.parallel_config.decode_context_parallel_size == 1, (
            "DCP not support sliding window."
        )
        max_model_len = vllm_config.model_config.max_model_len
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        # During chunked prefill, we allocate KV cache for the last
        # `self.sliding_window-1` computed tokens plus the newly scheduled
        # tokens. And we won't allocate KV cache for more than `max_model_len`
        # tokens.
        num_tokens = min(
            self.sliding_window - 1 + max_num_batched_tokens, max_model_len
        )

        # +1 here because the sliding window may not start from the beginning
        # of the block. For example, if the block size is 4 and num_token
        # is 4, we need two blocks [XXCD] [EF] to store the sliding
        # window [CDEF] of 6 tokens.
        return (cdiv(num_tokens, self.block_size) + 1) * self.page_size_bytes


@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype]
    page_size_padded: int | None = None
    mamba_type: str = "mamba2"
    mamba_cache_mode: str = "none"
    num_speculative_blocks: int = 0

    @property
    def page_size_bytes(self) -> int:
        page_size = sum(
            prod(shape) * get_dtype_size(dtype)
            for (shape, dtype) in zip(self.shapes, self.dtypes)
        )
        if self.page_size_padded is not None:
            assert self.page_size_padded >= page_size
            return self.page_size_padded
        return page_size

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        if vllm_config.cache_config.mamba_cache_mode == "all":
            max_model_len = vllm_config.model_config.max_model_len
            return cdiv(max_model_len, self.block_size) * self.page_size_bytes
        elif vllm_config.cache_config.mamba_cache_mode == "align":
            return self.page_size_bytes * (2 + self.num_speculative_blocks)
        else:
            return self.page_size_bytes * (1 + self.num_speculative_blocks)


@dataclass(frozen=True)
class EncoderOnlyAttentionSpec(AttentionSpec):
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # Encoder-only layers do not need KV cache
        return 0


@dataclass(frozen=True)
class CrossAttentionSpec(AttentionSpec):
    """
    KV cache spec for cross-attention layers in encoder-decoder models.
    """

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # For cross-attention, we need to cache encoder states
        # Get encoder length (e.g., 1500 for Whisper).
        max_encoder_len = vllm_config.scheduler_config.max_num_encoder_input_tokens
        return cdiv(max_encoder_len, self.block_size) * self.page_size_bytes


@dataclass(frozen=True, kw_only=True)
class SparseAttentionSpec(FullAttentionSpec):
    """
    KV cache spec for RetroInfer-style dynamic sparse attention.

    Algorithm overview
    ------------------
    1. **Prefill** – after all prompt KV is computed, the block-level key
       features are clustered via Segment K-Means (``n_segment`` position
       segments, ``num_clusters`` centroids total).  Mean-centering is applied
       before clustering and restored afterwards.
    2. **Prefill TopK (optional)** – when ``refresh_topk_each_decode`` is False
       and ``prefill_topk_query_window`` > 0, the first post-indexing query
       pass can cache per-layer TopK for decode reuse.
    3. **Decode** – when ``refresh_topk_each_decode`` is True (default), each
       step uses the **latest** query vector with the current cluster index to
       re-run TopK (same path as prefill selection). When False, decode reuses
       the prefill TopK cache until a dynamic index update invalidates it.
       Every ``update_threshold_blocks`` new decode blocks the index is
       refreshed with a fresh segment K-Means on the accumulated decode KV.
    4. **Steady Zone** – the first ``static_pattern_start`` tokens (attention
       sinks) and the last ``static_pattern_end`` tokens (local window) are
       always loaded regardless of cluster scores.

    Attributes
    ----------
    num_clusters:
        Total number of K-Means cluster centroids across all segments.
    n_segment:
        Number of position-aligned segments the sequence is divided into
        before running per-segment K-Means (preserves temporal locality).
    nprobe:
        Retrieve zone size: number of top-scored clusters whose blocks are
        fully loaded into GPU memory each decode step.
    static_pattern_start:
        Number of leading tokens kept as attention sinks (steady zone head).
        Should be aligned to ``block_size`` for efficiency.
    static_pattern_end:
        Number of trailing tokens kept as the local window (steady zone tail).
        Should be aligned to ``block_size`` for efficiency.
    prefill_topk_query_window:
        Reserved for future multi-query prefill averaging.  When
        ``refresh_topk_each_decode`` is False and this is > 0, the manager
        warms a one-shot prefill TopK cache on the first post-indexing query
        update.  Set to 0 to skip that warmup (selection still runs each
        ``select()`` when the cache is unused).
    refresh_topk_each_decode:
        If True (default), every scheduler ``select()`` after a query update
        re-scores clusters with the current query (decode uses the same TopK
        path as prefill).  Steady-zone tokens
        (``static_pattern_start`` / ``static_pattern_end``) stay merged into the
        selection.  If False, reuse prefill TopK cache across decode for less
        CPU work.
    update_threshold_blocks:
        After this many new decode blocks accumulate, re-run segment K-Means
        on them and append new centroids to the index.
        Corresponds to ``THRESHOLD_LENGTH // block_size`` in RetroInfer.

    TODO(estimation-zone)
    ---------------------
    The Estimation Zone is **not yet implemented**.  The design is:
    - Select ``max_compute_cluster_num - nprobe`` additional clusters beyond
      the retrieve zone as the estimation zone.
    - Compute approximate attention for those clusters using only the stored
      ``_cluster_value_sum`` (centroid as K representative, value_sum/size as V).
    - Merge the estimation zone output with the retrieve zone flash-attention
      result via log-sum-exp (lse) combination, as in RetroInfer's
      ``weighted_flash_decoding``.
    To enable: set ``estimation_zone_size > 0`` and implement
    ``SparseKVManager._estimate_zone_attn()`` in the model runner side.
    """

    # ── Clustering ──────────────────────────────────────────────────────
    num_clusters: int = 32
    """Total K-Means centroids (distributed evenly across segments)."""
    n_segment: int = 8
    """Position segments for segment K-Means."""
    num_layer: int = 24
    """layer num"""
    num_query_heads: int = 0
    """Per-rank query head count used to size decode scratch buffers."""

    # ── Retrieve zone ───────────────────────────────────────────────────
    nprobe: int = 16
    """Number of top-scored clusters to fully load (retrieve zone)."""

    # TODO(estimation-zone): add ``estimation_zone_size: int = 0`` here
    # and ``max_compute_cluster_num = nprobe + estimation_zone_size``.

    # ── Steady zone ─────────────────────────────────────────────────────
    static_pattern_start: int = 0
    """Attention sink tokens always kept on GPU (leading tokens)."""
    static_pattern_end: int = 0
    """Local window tokens always kept on GPU (trailing tokens)."""

    # ── Prefill TopK caching (when refresh_topk_each_decode is False) ────
    prefill_topk_query_window: int = 8
    """
    When ``refresh_topk_each_decode`` is False and this is > 0, warm a
    one-shot TopK cache on the first query update after indexing.
    """

    refresh_topk_each_decode: bool = True
    """
    Re-run cluster TopK with the latest query every decode step (default).
    When False, reuse the prefill TopK cache until invalidation.
    """

    # ── Dynamic index update ─────────────────────────────────────────────
    update_threshold_blocks: int = 64
    """
    Re-run segment K-Means on accumulated decode blocks after this many
    new blocks.  Mirrors RetroInfer's THRESHOLD_LENGTH / block_size.
    """

    # ── Legacy / compat ─────────────────────────────────────────────────
    max_selected_blocks: int = 64
    """
    Hard cap on total selected blocks per decode step
    (steady zone + retrieve zone combined).
    """

    cluster_granularity: Literal["block", "token"] = "token"
    """
    ``block`` – cluster mean key features per KV block (RetroInfer default).
    ``token`` – one feature vector per prompt / history token; Top-K and
    cluster membership are per token.  Selected tokens are mapped to logical
    blocks for paging; FlashAttention still reads full KV slots inside each
    loaded block unless a future mask path is added.
    """

    max_selected_tokens: int | None = None
    """
    When ``cluster_granularity == "token"``, caps selected tokens per layer
    (steady + retrieve).  If ``None``, defaults to
    ``max_selected_blocks * block_size``.
    """

    update_threshold_tokens: int = 1024
    """
    In token mode, number of new decode token features buffered before
    ``rebalance`` runs a dynamic segment K-Means (mirrors
    ``update_threshold_blocks`` in block mode).
    """

    use_compact_kv_gather: bool = True
    """
    When ``cluster_granularity == "token"`` and this is True, decode attention
    gathers only selected keys/values into a contiguous tensor and runs
    FlashAttention varlen without a block table.  When False, the legacy
    sparse block-table path is used.
    """

    @property
    def n_sink_blocks(self) -> int:
        """Leading blocks permanently in the steady zone (attention sinks)."""
        return cdiv(self.static_pattern_start, self.block_size)

    @property
    def n_recent_blocks(self) -> int:
        """Trailing blocks permanently in the steady zone (local window)."""
        return cdiv(self.static_pattern_end, self.block_size)

    @property
    def effective_max_selected_tokens(self) -> int:
        """Per-layer token budget when ``cluster_granularity == "token"``."""
        if self.max_selected_tokens is not None:
            return int(self.max_selected_tokens)
        return int(self.max_selected_blocks * self.block_size)

    def sparse_selection_budget(self) -> int:
        """Cap passed to ``SparseKVManager.select`` (blocks or tokens)."""
        if self.cluster_granularity == "token":
            return self.effective_max_selected_tokens
        return int(self.max_selected_blocks)

    def max_blocks_for_sparse(self) -> int:
        """Upper bound on logical history blocks per layer after trimming / cache."""
        if self.cluster_granularity == "token":
            return max(
                int(self.max_selected_blocks),
                cdiv(self.effective_max_selected_tokens, self.block_size),
            )
        return int(self.max_selected_blocks)

    @classmethod
    def merge(cls, specs: list["SparseAttentionSpec"]) -> "SparseAttentionSpec":  # type: ignore[override]
        assert all(isinstance(spec, SparseAttentionSpec) for spec in specs), (
            "All layers in a sparse group must be SparseAttentionSpec."
        )
        base = FullAttentionSpec.merge(specs)  # type: ignore[arg-type]
        return cls(
            block_size=base.block_size,
            num_kv_heads=base.num_kv_heads,
            head_size=base.head_size,
            head_size_v=base.head_size_v,
            dtype=base.dtype,
            page_size_padded=base.page_size_padded,
            sliding_window=base.sliding_window,
            attention_chunk_size=base.attention_chunk_size,
            num_clusters=specs[0].num_clusters,
            n_segment=specs[0].n_segment,
            nprobe=specs[0].nprobe,
            num_layer=specs[0].num_layer,
            num_query_heads=specs[0].num_query_heads,
            static_pattern_start=specs[0].static_pattern_start,
            static_pattern_end=specs[0].static_pattern_end,
            prefill_topk_query_window=specs[0].prefill_topk_query_window,
            refresh_topk_each_decode=specs[0].refresh_topk_each_decode,
            update_threshold_blocks=specs[0].update_threshold_blocks,
            max_selected_blocks=specs[0].max_selected_blocks,
            cluster_granularity=specs[0].cluster_granularity,
            max_selected_tokens=specs[0].max_selected_tokens,
            update_threshold_tokens=specs[0].update_threshold_tokens,
            use_compact_kv_gather=specs[0].use_compact_kv_gather,
        )

    def max_memory_usage_bytes(self, vllm_config: "VllmConfig") -> int:
        # GPU resident at once: steady zone + retrieve zone + 1 new decode block.
        if self.cluster_granularity == "token":
            max_pages = cdiv(self.effective_max_selected_tokens, self.block_size)
            return (max_pages + 1) * self.page_size_bytes
        return (self.max_selected_blocks + 1) * self.page_size_bytes


@dataclass(frozen=True)
class SinkFullAttentionSpec(FullAttentionSpec):
    sink_len: int | None = None

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of FullAttentionSpec objects into a single
        FullAttentionSpec object.
        """
        assert all(isinstance(spec, FullAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be FullAttentionSpec."
        )

        sliding_window = set(
            spec.sliding_window for spec in specs if spec.sliding_window is not None
        )
        attention_chunk_size = set(
            spec.attention_chunk_size
            for spec in specs
            if spec.attention_chunk_size is not None
        )
        assert not any(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "MLAAttentionSpec should be merged in MLAAttentionSpec.merge"
        )
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            head_size_v=specs[0].head_size_v,
            sink_len=specs[0].sink_len,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            sliding_window=cls.merge_window_sizes(sliding_window),
            attention_chunk_size=cls.merge_window_sizes(attention_chunk_size),
        )
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All attention layers in the same KV cache group must have "
                    "the same attention spec."
                )
        assert (merged_spec.sliding_window is not None) + (
            merged_spec.attention_chunk_size is not None
        ) <= 1, (
            "Model with both sliding window layers and chunked local attention "
            "layers is not supported."
        )
        return merged_spec


@dataclass(frozen=True)
class UniformTypeKVCacheSpecs(KVCacheSpec):
    """
    A KV cache spec for multiple layers with the same type of attention. Here,
    same types means always need the same number of token slots. For example,
    sliding window attentions with different window sizes are not the same type
    and should not be merged into one UniformTypeKVCacheSpecs.
    """

    kv_cache_specs: dict[str, KVCacheSpec]

    @property
    def page_size_bytes(self) -> int:
        return sum(spec.page_size_bytes for spec in self.kv_cache_specs.values())

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_num_pages = max(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            for spec in self.kv_cache_specs.values()
        )
        return max_num_pages * self.page_size_bytes

    @classmethod
    def is_uniform_type(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
        """
        Whether all layers have the same type of KV cache spec.
        """
        block_sizes = set(spec.block_size for spec in kv_cache_specs.values())
        if len(block_sizes) > 1:
            # Different block sizes, not uniform.
            return False
        one_spec = next(iter(kv_cache_specs.values()))
        if isinstance(one_spec, SparseAttentionSpec):
            return all(
                isinstance(spec, SparseAttentionSpec)
                and spec.max_selected_blocks == one_spec.max_selected_blocks
                and spec.num_clusters == one_spec.num_clusters
                and spec.n_segment == one_spec.n_segment
                and spec.nprobe == one_spec.nprobe
                and spec.static_pattern_start == one_spec.static_pattern_start
                and spec.static_pattern_end == one_spec.static_pattern_end
                and spec.prefill_topk_query_window == one_spec.prefill_topk_query_window
                and spec.refresh_topk_each_decode
                == one_spec.refresh_topk_each_decode
                and spec.update_threshold_blocks == one_spec.update_threshold_blocks
                and spec.cluster_granularity == one_spec.cluster_granularity
                and spec.max_selected_tokens == one_spec.max_selected_tokens
                and spec.update_threshold_tokens == one_spec.update_threshold_tokens
                and spec.use_compact_kv_gather == one_spec.use_compact_kv_gather
                for spec in kv_cache_specs.values()
            )
        if isinstance(one_spec, FullAttentionSpec):
            return all(
                isinstance(spec, FullAttentionSpec) for spec in kv_cache_specs.values()
            )
        elif isinstance(one_spec, CrossAttentionSpec):
            return all(
                isinstance(spec, CrossAttentionSpec) for spec in kv_cache_specs.values()
            )
        elif isinstance(one_spec, SlidingWindowSpec):
            return all(
                isinstance(spec, SlidingWindowSpec)
                and spec.sliding_window == one_spec.sliding_window
                for spec in kv_cache_specs.values()
            )
        elif isinstance(one_spec, ChunkedLocalAttentionSpec):
            return all(
                isinstance(spec, ChunkedLocalAttentionSpec)
                and spec.attention_chunk_size == one_spec.attention_chunk_size
                for spec in kv_cache_specs.values()
            )
        elif isinstance(one_spec, MambaSpec):
            return all(
                isinstance(spec, MambaSpec)
                and spec.num_speculative_blocks == one_spec.num_speculative_blocks
                for spec in kv_cache_specs.values()
            )
        else:
            # NOTE(Chen): Please add new branches for new KV cache spec types.
            raise NotImplementedError(
                f"Unsupported KV cache spec type: {type(one_spec)}"
            )

    @classmethod
    def from_specs(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> Self | None:
        """
        Return a SameTypeKVCacheSpecs object if all layers have the same type
        of KV cache spec. Return None if not.
        """
        if cls.is_uniform_type(kv_cache_specs):
            block_size = next(iter(kv_cache_specs.values())).block_size
            return cls(block_size=block_size, kv_cache_specs=kv_cache_specs)
        else:
            return None


@dataclass
class KVCacheTensor:
    """
    A class for specifying how the workers should initialize the KV cache.
    """

    size: int  # size of the KV cache tensor in bytes
    shared_by: list[str]  # layer names that share the same KV cache tensor


@dataclass
class KVCacheGroupSpec:
    """
    Represents a group of model layers that share the same KV cache block table.
    These layers are regarded as one layer in the KV cache manager.
    """

    # The names of model layers in this group
    layer_names: list[str]
    # The KV cache spec of this manager layer
    kv_cache_spec: KVCacheSpec


@dataclass
class KVCacheConfig:
    """
    The KV cache configuration of a model.
    """

    num_blocks: int
    """The number of KV cache blocks"""
    kv_cache_tensors: list[KVCacheTensor]
    """How should model runner initialize the KV cache tensors for each layer"""
    kv_cache_groups: list[KVCacheGroupSpec]
    """
    The kv cache groups of the model.
    For models with only one type of attention, there is only one group that
    contains all layers.
    For models with multiple types of attention, there will be multiple groups,
    see `_get_kv_cache_config_uniform_page_size` for more details.
    """

    @property
    def has_mamba_layers(self) -> bool:
        return any(isinstance(g.kv_cache_spec, MambaSpec) for g in self.kv_cache_groups)

    @property
    def needs_kv_cache_zeroing(self) -> bool:
        return self.has_mamba_layers
