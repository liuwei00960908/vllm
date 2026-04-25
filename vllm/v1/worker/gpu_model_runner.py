# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import gc
import itertools
import os
import threading
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
from functools import reduce
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, cast

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from tqdm import tqdm

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphStat, CUDAGraphWrapper
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    set_current_vllm_config,
    update_config,
)
from vllm.config.cache import CacheConfig
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
    get_tp_group,
    graph_capture,
    is_global_first_rank,
    prepare_communication_buffer_for_model,
)
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    is_forward_context_available,
    set_forward_context,
)
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping, LoRAMappingType
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding,
    XDRotaryEmbedding,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsXDRoPE,
    is_mixture_of_experts,
    supports_eagle3,
    supports_mrope,
    supports_multimodal_pruning,
    supports_realtime,
    supports_transcription,
    supports_xdrope,
)
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling,
    is_pooling_model,
    is_text_generation_model,
)
from vllm.model_executor.offloader import (
    create_offloader,
    get_offloader,
    set_offloader,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.inputs import (
    BatchedTensorInputs,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.utils import group_and_batch_mm_kwargs
from vllm.platforms import current_platform
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.tracing import instrument
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
from vllm.utils.nvtx_pytorch_hooks import PytHooks
from vllm.utils.platform_utils import is_pin_memory_available, num_compute_units
from vllm.utils.torch_utils import (
    get_dtype_size,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.attention.backends.utils import (
    create_fast_prefill_custom_backend,
    get_dcp_local_seq_lens,
    reorder_batch_to_split_decodes_and_prefills,
)
from vllm.v1.attention.ops.triton_sparse_pack import (
    sparse_pack_cluster_members_single_req,
    sparse_pack_single_req,
)
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.core.sparse_kmeans_torch import (
    kmeans_features_from_kv_cache_torch,
    prefill_cluster_meta_from_features_device,
    prefill_cluster_meta_from_features_torch,
    prefill_cluster_meta_from_kv_cache_device,
    sparse_prefill_cluster_use_device_kmeans,
    value_sum_from_kv_cache_torch,
)
from vllm.v1.core.sparse_kv_cache_manager import (
    SparseKVManager,
    parse_sparse_kv_key,
    sparse_kv_unit_key,
    sparse_qh_unit_key,
)
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    SparseAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ECConnectorOutput,
    KVConnectorOutput,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    PoolerOutput,
    SamplerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates
from vllm.v1.sample.logits_processor import LogitsProcessors, build_logitsprocs
from vllm.v1.sample.logits_processor.interface import LogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer
from vllm.v1.spec_decode.medusa import MedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer_gpu import (
    NgramProposerGPU,
    copy_num_valid_draft_tokens,
    update_ngram_gpu_tensors_incremental,
    update_scheduler_for_invalid_drafts,
)
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import CpuGpuBuffer, record_function_or_nullcontext
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.cp_utils import (
    check_attention_cp_compatibility,
    get_total_cp_world_size,
)
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.ec_connector_model_runner_mixin import ECConnectorModelRunnerMixin
from vllm.v1.worker.gpu.pool.late_interaction_runner import LateInteractionRunner
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.ubatch_utils import (
    UBatchSlices,
    check_ubatch_thresholds,
    maybe_create_ubatch_slices,
    split_attn_metadata,
)
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.workspace import lock_workspace

from .utils import (
    AttentionGroup,
    KVBlockZeroer,
    add_kv_sharing_layers_to_kv_cache_groups,
    bind_kv_cache,
    prepare_kernel_block_sizes,
    sanity_check_mm_encoder_outputs,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.spec_decode.ngram_proposer import NgramProposer

logger = init_logger(__name__)
_SPARSE_DEBUG_ASSERT: bool = int(os.getenv("VLLM_SPARSE_DEBUG_ASSERT", "0")) == 1
_SPARSE_TOKEN_TOPK_TRACE: bool = (
    int(os.getenv("VLLM_SPARSE_TOKEN_TOPK_TRACE", "0")) == 1
)
_SPARSE_DECODE_STEP_TRACE: bool = (
    int(os.getenv("VLLM_SPARSE_DECODE_STEP_TRACE", "0")) == 1
)


def _sparse_debug_range(name: str, tensor: torch.Tensor, upper: int) -> None:
    """Debug-only sync to catch sparse index corruption before CUDA kernels."""
    if tensor.numel() == 0:
        return
    lo = int(tensor.min().item())
    hi = int(tensor.max().item())
    if lo < 0 or hi >= int(upper):
        raise ValueError(
            f"{name} out of range: min={lo} max={hi} upper={int(upper)}"
        )

# ---------------------------------------------------------------------------
# Sparse first-decode hard debug (NO env vars).
# When True: prints SparseFirstTok kv/sample lines (if sparse attention is on)
# and calls os._exit(0) after Nth valid new output token is appended in
# bookkeeping. Set to False before merge or normal serving.
# ---------------------------------------------------------------------------
_SPARSE_HARD_DEBUG_FIRST_NEW_TOKEN = False
# Stop after this many valid generated tokens for a request.
# 1 = first token, 2 = second token (useful for capturing "V" -> "Vo").
_SPARSE_HARD_DEBUG_STOP_AFTER_OUTPUT_N = 6

AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict


# Wrapper for ModelRunnerOutput to support overlapped execution.
class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        logprobs_tensors: LogprobsTensors | None,
        invalid_req_indices: list[int],
        async_output_copy_stream: torch.cuda.Stream,
        vocab_size: int,
    ):
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices

        # Event on the copy stream so we can synchronize the non-blocking copy.
        self.async_copy_ready_event = torch.Event()

        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        self._sampled_token_ids = sampled_token_ids
        self.vocab_size = vocab_size
        self._logprobs_tensors = logprobs_tensors

        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self.sampled_token_ids_cpu = self._sampled_token_ids.to(
                "cpu", non_blocking=True
            )
            self._logprobs_tensors_cpu = (
                self._logprobs_tensors.to_cpu_nonblocking()
                if self._logprobs_tensors
                else None
            )
            self.async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.

        This function blocks until the copy is finished.
        """
        max_gen_len = self.sampled_token_ids_cpu.shape[-1]
        self.async_copy_ready_event.synchronize()

        # Release the device tensors once the copy has completed.
        del self._logprobs_tensors
        del self._sampled_token_ids
        if max_gen_len == 1:
            valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()
            for i in self._invalid_req_indices:
                valid_sampled_token_ids[i].clear()
            logprobs_lists = None
            if self._logprobs_tensors_cpu is not None:
                logprobs_lists = self._logprobs_tensors_cpu.tolists()
        else:
            valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                self.sampled_token_ids_cpu,
                self.vocab_size,
                self._invalid_req_indices,
                logprobs_tensors=self._logprobs_tensors_cpu,
            )

        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        output.logprobs = logprobs_lists
        return output


def _copy_pooler_output_to_cpu(
    raw_pooler_output: PoolerOutput, finished_mask: list[bool]
) -> list[torch.Tensor | None]:
    num_reqs = len(finished_mask)

    if isinstance(raw_pooler_output, torch.Tensor):
        if raw_pooler_output.shape[0] != num_reqs:
            raise ValueError(
                "Pooler output batch size does not match finished mask size: "
                f"{raw_pooler_output.shape[0]} != {num_reqs}."
            )

        num_finished = sum(finished_mask)
        if num_finished == 0:
            return [None] * num_reqs
        if num_finished == num_reqs:
            return list(raw_pooler_output.to("cpu", non_blocking=True))

        # partial finished
        finished_indices = [i for i, include in enumerate(finished_mask) if include]
        index_tensor = torch.tensor(
            finished_indices, device=raw_pooler_output.device, dtype=torch.long
        )
        finished_outputs = raw_pooler_output.index_select(0, index_tensor).to(
            "cpu", non_blocking=True
        )
        partial_pooler_output: list[torch.Tensor | None] = [None] * num_reqs
        for i, out in zip(finished_indices, finished_outputs):
            partial_pooler_output[i] = out
        return partial_pooler_output

    assert isinstance(raw_pooler_output, list)
    if len(raw_pooler_output) != num_reqs:
        raise ValueError(
            "Pooler output batch size does not match finished mask size: "
            f"{len(raw_pooler_output)} != {num_reqs}."
        )

    pooler_output: list[torch.Tensor | None] = [None] * num_reqs
    for i, (out, include) in enumerate(zip(raw_pooler_output, finished_mask)):
        if include and out is not None:
            pooler_output[i] = out.to("cpu", non_blocking=True)
    return pooler_output


class AsyncGPUPoolingModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        raw_pooler_output: PoolerOutput,
        finished_mask: list[bool],
        async_output_copy_stream: torch.cuda.Stream,
    ):
        self._model_runner_output = model_runner_output

        # Event on the copy stream so we can synchronize the non-blocking copy.
        self.async_copy_ready_event = torch.Event()

        # Keep a reference to the device tensors to avoid them being
        # deallocated until we finish copying it to the host.
        self._raw_pooler_output = raw_pooler_output

        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self._model_runner_output.pooler_output = _copy_pooler_output_to_cpu(
                raw_pooler_output=self._raw_pooler_output,
                finished_mask=finished_mask,
            )
            self.async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.
        This function blocks until the copy is finished.
        """
        self.async_copy_ready_event.synchronize()

        # Release the device tensors once the copy has completed.
        del self._raw_pooler_output
        return self._model_runner_output


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None


class _SparseOnlineLayerState:
    """Per-(layer, kv_head) online index state for token-granularity sparse.

    ``all_block_features`` and ``block_to_cluster`` are stored in
    pre-allocated growth buffers.  Decode appends one row per step; the legacy
    ``torch.cat`` path allocates + copies the entire tensor every step, which
    costs O(decode_len) per step and dominates the decode critical path for
    long contexts.  Exposing ``block_to_cluster`` / ``all_block_features`` via
    properties that slice the underlying storage preserves the previous
    read-side contract (including in-place slice writes like
    ``state.block_to_cluster[-m:] = ...`` in dynamic kmeans updates).
    """

    __slots__ = (
        "cluster_centres",
        "cluster_size",
        "cluster_members",
        "mean_key",
        "decode_block_buffer",
        "_abf_storage",
        "_b2c_storage",
        "_len",
        # Retroinfer-style cluster-retrieval additions.
        #
        # ``value_sum``: per-cluster accumulated V rows, float32
        # ``[K, D]``.  Written once at prefill (and extended at each dynamic
        # k-means refresh, mirroring retroinfer's ``torch.cat`` pattern); used
        # by the estimation zone to approximate the non-retrieved clusters'
        # softmax-attention contribution.  Lives alongside ``cluster_centres``
        # / ``cluster_size`` and is kept consistent with them.
        #
        # ``_cs_storage`` / ``_cs_len``: CSR data array for the
        # cluster→token inverted index.  Stores **logical token positions**
        # within the request (int32), sorted by cluster id.  Resolution to
        # paged physical slots happens inside the gather kernel, so we avoid
        # holding a second copy of the vLLM block_table.
        #
        # ``cluster_offsets``: CSR row-pointer int32 ``[K+1]``.  Cluster ``c``
        # occupies ``cluster_slots[cluster_offsets[c]:cluster_offsets[c+1]]``.
        #
        # Note: tokens in the steady zone
        # (``static_pattern_start``/``static_pattern_end``) are intentionally
        # **excluded** from the CSR at prefill-time so the expand kernel does
        # not have to deduplicate against an always-on steady region.
        "value_sum",
        "_cs_storage",
        "_cs_len",
        "cluster_offsets",
    )

    def __init__(
        self,
        cluster_centres: torch.Tensor,
        cluster_size: torch.Tensor,
        block_to_cluster: torch.Tensor,
        all_block_features: torch.Tensor,
        mean_key: torch.Tensor,
        value_sum: torch.Tensor | None = None,
        cluster_members: torch.Tensor | None = None,
        cluster_slots: torch.Tensor | None = None,
        cluster_offsets: torch.Tensor | None = None,
        copy_all_block_features: bool = True,
    ) -> None:
        self.cluster_centres = cluster_centres
        self.cluster_size = cluster_size
        self.mean_key = mean_key
        self.decode_block_buffer: list[torch.Tensor] = []

        n = int(all_block_features.shape[0])
        head_size = int(all_block_features.shape[1]) if all_block_features.dim() >= 2 else 0
        # Init with exact capacity – prefill features can span the full prompt
        # length (thousands of tokens) and we are holding one buffer per
        # (layer, kv_head, req).  Over-allocating here would double the
        # persistent sparse-state footprint right when the prefill K-Means
        # scratch (e.g. ``torch.bmm`` in ``segment_kmeans_centered_torch``)
        # needs headroom, which triggered OOM in practice.  The first decode
        # append goes through ``_grow_if_needed`` which doubles the buffer;
        # by that point all prefill scratch tensors are already freed.
        cap = max(n, 1)
        self._abf_storage = torch.empty(
            (cap, head_size) if head_size > 0 else (cap,),
            dtype=all_block_features.dtype,
            device=all_block_features.device,
        )
        if n > 0 and copy_all_block_features:
            self._abf_storage[:n].copy_(all_block_features)
        self._b2c_storage = torch.empty(
            cap,
            dtype=block_to_cluster.dtype,
            device=block_to_cluster.device,
        )
        if n > 0:
            self._b2c_storage[:n].copy_(block_to_cluster)
        self._len = n

        # --- retroinfer-style CSR + value_sum init ---
        k_clusters = int(cluster_centres.shape[0]) if cluster_centres.dim() >= 2 else 0
        d_head = (
            int(cluster_centres.shape[1])
            if cluster_centres.dim() >= 2 and k_clusters > 0
            else head_size
        )
        dev = cluster_centres.device

        if cluster_members is not None:
            members = cluster_members
            if members.dtype != torch.int32:
                members = members.to(dtype=torch.int32)
            if members.device != dev:
                members = members.to(device=dev)
            self.cluster_members = members
        else:
            self.cluster_members = None

        if value_sum is not None:
            # Caller supplied a pre-built accumulator – trust shape/dtype.
            self.value_sum = value_sum
        else:
            # Empty placeholder; Phase 2 prefill path fills this in.
            self.value_sum = torch.zeros(
                (k_clusters, d_head), dtype=torch.float32, device=dev
            )

        if cluster_slots is not None:
            cs_n = int(cluster_slots.numel())
            cs_cap = max(cs_n, 1)
            self._cs_storage = torch.empty(
                cs_cap, dtype=torch.int32, device=dev
            )
            if cs_n > 0:
                src = cluster_slots
                if src.dtype != torch.int32:
                    src = src.to(dtype=torch.int32)
                self._cs_storage[:cs_n].copy_(src)
            self._cs_len = cs_n
        else:
            # Placeholder; Phase 2 prefill will rebuild via
            # ``rebuild_cluster_csr_from_labels``.
            self._cs_storage = torch.empty(1, dtype=torch.int32, device=dev)
            self._cs_len = 0

        if cluster_offsets is not None:
            off = cluster_offsets
            if off.dtype != torch.int32:
                off = off.to(dtype=torch.int32)
            self.cluster_offsets = off
        else:
            # Placeholder with a valid ``[0]`` sentinel so consumers can
            # always dereference ``cluster_offsets[0]`` without a shape check.
            self.cluster_offsets = torch.zeros(
                max(k_clusters + 1, 1), dtype=torch.int32, device=dev
            )

    @property
    def block_to_cluster(self) -> torch.Tensor:
        return self._b2c_storage[: self._len]

    @block_to_cluster.setter
    def block_to_cluster(self, value: torch.Tensor) -> None:
        """Replace ``block_to_cluster`` contents.

        Only used by ``_sparse_online_dynamic_update`` fallbacks that rebuild
        the entire label vector.  Preserves the pre-allocated capacity when
        ``value.numel() <= capacity``.
        """
        n = int(value.numel())
        if n > int(self._b2c_storage.shape[0]):
            self._b2c_storage = torch.empty(
                max(n, int(self._b2c_storage.shape[0]) * 2),
                dtype=value.dtype,
                device=value.device,
            )
        self._b2c_storage[:n].copy_(value)
        self._len = n

    @property
    def all_block_features(self) -> torch.Tensor:
        return self._abf_storage[: self._len]

    @all_block_features.setter
    def all_block_features(self, value: torch.Tensor) -> None:
        """Replace ``all_block_features`` contents (full rewrite path)."""
        n = int(value.shape[0])
        head_size = int(value.shape[1]) if value.dim() >= 2 else 0
        storage_head_size = (
            int(self._abf_storage.shape[1]) if self._abf_storage.dim() >= 2 else 0
        )
        needs_realloc = (
            n > int(self._abf_storage.shape[0])
            or head_size != storage_head_size
            or value.dtype != self._abf_storage.dtype
            or value.device != self._abf_storage.device
        )
        if needs_realloc:
            cap = max(n, int(self._abf_storage.shape[0]) * 2)
            self._abf_storage = torch.empty(
                (cap, head_size) if head_size > 0 else (cap,),
                dtype=value.dtype,
                device=value.device,
            )
        self._abf_storage[:n].copy_(value)
        self._len = n

    # --- retroinfer-style CSR helpers ----------------------------------

    @property
    def cluster_slots(self) -> torch.Tensor:
        """Read-side view of the current CSR data array.

        Shape ``[cluster_offsets[-1]]`` int32, i.e. the number of clustered
        (non-steady) tokens registered into the inverted index so far.
        """
        return self._cs_storage[: self._cs_len]

    def rebuild_cluster_csr_from_labels(
        self,
        labels: torch.Tensor,
        positions: torch.Tensor,
        num_clusters: int,
    ) -> None:
        """Rebuild the CSR (``cluster_slots`` / ``cluster_offsets``) in-place.

        Args:
            labels: ``[M]`` int (any int dtype); cluster id per entry.
            positions: ``[M]`` int; the logical-token position to store.
            num_clusters: total cluster count ``K`` (including empty clusters).

        Runs on ``labels.device``.  Uses ``argsort`` on the cluster ids so the
        CSR data is laid out grouped-by-cluster, with ``torch.bincount`` +
        ``cumsum`` producing the offsets – this mirrors retroinfer's offline
        index layout but stays fully on-device and avoids any per-cluster
        Python loop.
        """
        m = int(labels.numel())
        dev = labels.device
        if m == 0:
            self._cs_storage = torch.empty(1, dtype=torch.int32, device=dev)
            self._cs_len = 0
            self.cluster_offsets = torch.zeros(
                max(int(num_clusters) + 1, 1), dtype=torch.int32, device=dev
            )
            return

        # Stable sort by cluster id so slot order within a cluster is
        # deterministic (helps reproducibility of downstream debug traces
        # and keeps ``exec_buf`` ordering predictable across runs).
        lab64 = labels if labels.dtype == torch.int64 else labels.to(torch.int64)
        order = torch.argsort(lab64, stable=True)
        sorted_positions = positions.to(dtype=torch.int32)[order].contiguous()

        # Grow storage if needed; reuse the existing allocator-friendly
        # ``*2`` doubling so repeated dynamic refreshes converge on a stable
        # capacity.
        cap = int(self._cs_storage.shape[0])
        if m > cap:
            new_cap = cap if cap > 0 else 128
            while new_cap < m:
                new_cap *= 2
            self._cs_storage = torch.empty(
                new_cap, dtype=torch.int32, device=dev
            )
        self._cs_storage[:m].copy_(sorted_positions)
        self._cs_len = m

        # CSR offsets: bincount of labels + cumsum, prepended with 0.
        # ``torch.cumsum`` on int32 is not supported on all backends, so we
        # go through int64 scratch and cast back to int32 – the ``K`` axis
        # here is tiny (hundreds to low thousands) so the conversion cost is
        # noise compared to the attention passes downstream.
        counts64 = torch.bincount(lab64, minlength=int(num_clusters))
        offs = torch.empty(
            int(num_clusters) + 1, dtype=torch.int32, device=dev
        )
        offs[0] = 0
        offs[1:].copy_(torch.cumsum(counts64, dim=0).to(torch.int32))
        self.cluster_offsets = offs

    def accumulate_value_sum(
        self,
        values: torch.Tensor,
        labels: torch.Tensor,
        num_clusters: int,
    ) -> None:
        """Scatter-add ``values`` into ``self.value_sum`` bucketed by ``labels``.

        Args:
            values: ``[M, D]`` float (any float dtype); V rows to aggregate.
            labels: ``[M]`` int; cluster id per row.
            num_clusters: target cluster count ``K``.

        Always stores accumulation in float32 for stability (mirrors
        retroinfer which materialises ``value_sum`` in the model dtype only
        lazily; we keep it fp32 to tolerate K-Means refreshes).
        """
        if values.numel() == 0:
            d = int(self.value_sum.shape[-1]) if self.value_sum.dim() == 2 else 0
            self.value_sum = torch.zeros(
                (int(num_clusters), d),
                dtype=torch.float32,
                device=values.device,
            )
            return
        d = int(values.shape[-1])
        vs = torch.zeros(
            (int(num_clusters), d),
            dtype=torch.float32,
            device=values.device,
        )
        lab64 = labels if labels.dtype == torch.int64 else labels.to(torch.int64)
        vs.index_add_(0, lab64, values.to(dtype=torch.float32))
        self.value_sum = vs

    def _grow_if_needed(self, extra: int) -> None:
        needed = self._len + extra
        cap = int(self._abf_storage.shape[0])
        if needed <= cap:
            return
        new_cap = cap if cap > 0 else 128
        while new_cap < needed:
            new_cap *= 2
        head_size = (
            int(self._abf_storage.shape[1]) if self._abf_storage.dim() >= 2 else 0
        )
        new_abf = torch.empty(
            (new_cap, head_size) if head_size > 0 else (new_cap,),
            dtype=self._abf_storage.dtype,
            device=self._abf_storage.device,
        )
        new_abf[: self._len].copy_(self._abf_storage[: self._len])
        self._abf_storage = new_abf
        new_b2c = torch.empty(
            new_cap,
            dtype=self._b2c_storage.dtype,
            device=self._b2c_storage.device,
        )
        new_b2c[: self._len].copy_(self._b2c_storage[: self._len])
        self._b2c_storage = new_b2c

    def append_decode_feature(
        self, feat: torch.Tensor, cluster_id: torch.Tensor
    ) -> None:
        """Append one decode-step feature with its cluster id.

        ``feat`` is a ``[head_size]`` tensor already on the correct device.
        ``cluster_id`` is a 0-dim int64 tensor on the correct device.
        """
        self._grow_if_needed(1)
        self._abf_storage[self._len].copy_(feat)
        self._b2c_storage[self._len].copy_(cluster_id)
        self._len += 1

    @staticmethod
    def bulk_append(
        states: "list[_SparseOnlineLayerState]",
        feats_stack: torch.Tensor,
        clusters_stack: torch.Tensor,
        perf_recorder: "object | None" = None,
    ) -> None:
        """Append one decode-step feature to each of ``states`` in batch.

        ``feats_stack`` is a ``[U, D]`` tensor, ``clusters_stack`` is ``[U]``
        int64; entry ``i`` is appended to ``states[i]``.  Compared to calling
        :meth:`append_decode_feature` in a Python loop this collapses
        ``2 * U`` small ``copy_`` kernel launches into two fused
        ``torch._foreach_copy_`` launches (when available), eliminating the
        per-(layer, kv_head) launch overhead that dominates
        ``_update_sparse_online_index``.

        If ``perf_recorder`` is a :class:`GPUModelRunner`-like object with a
        ``_sparse_perf_record`` method and ``_sparse_perf_stats_enabled`` set,
        sub-stage timings are recorded for diagnostics (no-op otherwise).
        """
        u = len(states)
        if u == 0:
            return
        assert feats_stack.shape[0] == u, (feats_stack.shape, u)
        assert clusters_stack.shape[0] == u, (clusters_stack.shape, u)

        _stats_on = bool(
            perf_recorder is not None
            and getattr(perf_recorder, "_sparse_perf_stats_enabled", False)
        )
        _record = (
            getattr(perf_recorder, "_sparse_perf_record", None)
            if _stats_on else None
        )

        _t = time.perf_counter() if _stats_on else None
        # First ensure every storage has room for one more entry.  This is a
        # pure-Python loop but is a no-op on the fast path (storage was grown
        # geometrically on a previous append).  Any rare realloc happens here
        # and not on the hot tensor-level path below.
        for s in states:
            s._grow_if_needed(1)
        if _record is not None and _t is not None:
            _record(
                "_update_sparse_online_index:ba_grow",
                time.perf_counter() - _t,
            )

        _t = time.perf_counter() if _stats_on else None
        abf_dsts: list[torch.Tensor] = [
            s._abf_storage[s._len] for s in states
        ]
        b2c_dsts: list[torch.Tensor] = [
            s._b2c_storage[s._len] for s in states
        ]
        abf_srcs: list[torch.Tensor] = list(feats_stack.unbind(0))
        b2c_srcs: list[torch.Tensor] = list(clusters_stack.unbind(0))
        if _record is not None and _t is not None:
            _record(
                "_update_sparse_online_index:ba_views",
                time.perf_counter() - _t,
            )

        _t = time.perf_counter() if _stats_on else None
        # ``torch._foreach_copy_`` is a multi-tensor fused copy available since
        # PyTorch 2.0.  Fall back to a plain per-tensor ``copy_`` loop when
        # unavailable (e.g. a stripped-down test build); correctness is
        # identical, only the fusion speedup is lost.
        foreach_copy = getattr(torch, "_foreach_copy_", None)
        if foreach_copy is not None:
            foreach_copy(abf_dsts, abf_srcs)
            foreach_copy(b2c_dsts, b2c_srcs)
        else:
            for dst, src in zip(abf_dsts, abf_srcs, strict=True):
                dst.copy_(src)
            for dst, src in zip(b2c_dsts, b2c_srcs, strict=True):
                dst.copy_(src)
        if _record is not None and _t is not None:
            _record(
                "_update_sparse_online_index:ba_foreach",
                time.perf_counter() - _t,
            )

        for s in states:
            s._len += 1


class GPUModelRunner(
    LoRAModelRunnerMixin, KVConnectorModelRunnerMixin, ECConnectorModelRunnerMixin
):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.offload_config = vllm_config.offload_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype

        self.kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            cache_config.cache_dtype, self.model_config
        )

        self.is_pooling_model = model_config.runner_type == "pooling"
        self.enable_prompt_embeds = model_config.enable_prompt_embeds
        self.is_multimodal_raw_input_only_model = (
            model_config.is_multimodal_raw_input_only_model
        )
        # These will be overridden in load_model()
        self.is_multimodal_pruning_enabled = False
        self.requires_sequential_video_encoding = False
        # Set to True after init_routed_experts_capturer() completes.
        # Prevents routed experts code from running during profiling/dummy run.
        self.routed_experts_initialized = False
        self.max_model_len = model_config.max_model_len

        # Always set to false after the first forward pass
        self.calculate_kv_scales = self.cache_config.calculate_kv_scales
        self.dcp_world_size = self.parallel_config.decode_context_parallel_size
        self.dcp_rank = 0 if self.dcp_world_size <= 1 else get_dcp_group().rank_in_group
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping micro-batches
        # https://github.com/vllm-project/vllm/issues/18019
        self.broadcast_pp_output = (
            self.parallel_config.distributed_executor_backend == "external_launcher"
            and len(get_pp_group().ranks) > 1
        )

        # Model-related.
        self.num_query_heads = model_config.get_num_attention_heads(parallel_config)
        self.inputs_embeds_size = model_config.get_inputs_embeds_size()
        self.attention_chunk_size = model_config.attention_chunk_size
        # Only relevant for models using ALiBi (e.g, MPT)
        self.use_alibi = model_config.uses_alibi

        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn
        self.is_mm_prefix_lm = self.model_config.is_mm_prefix_lm

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        self.uses_xdrope_dim = model_config.uses_xdrope_dim
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            model_config
        )

        if self.model_config.is_encoder_decoder:
            # Maximum length of the encoder input, only for encoder-decoder
            # models.
            self.max_encoder_len = scheduler_config.max_num_encoder_input_tokens
        else:
            self.max_encoder_len = 0

        # Async scheduling
        self.use_async_scheduling = self.scheduler_config.async_scheduling

        # Sampler
        self.sampler = Sampler(logprobs_mode=self.model_config.logprobs_mode)

        self.eplb_state: EplbState | None = None
        # NOTE(yongji): flag to temporarily disable EPLB during scaling up/down
        self.eep_eplb_suppressed = False
        """
        State of the expert parallelism load balancer.

        Will be lazily initialized when the model is loaded.
        """

        # Lazy initializations
        # self.model: nn.Module  # Set after load_model
        # Initialize in initialize_kv_cache
        self.kv_caches: list[torch.Tensor] = []
        # Initialize in initialize_kv_cache_tensors
        self.cross_layers_kv_cache: torch.Tensor | None = None
        self.cross_layers_attn_backend: type[AttentionBackend] | None = None
        # indexes: [kv_cache_group_id][attn_group]
        self.attn_groups: list[list[AttentionGroup]] = []
        # self.kv_cache_config: KVCacheConfig

        # ── Sparse KV attention state (RetroInfer-style) ──────────────────────
        # Set to True once a SparseAttentionSpec KV-cache group is detected.
        self._has_sparse_attn: bool = False
        # Captured query tensors from the current forward pass, keyed by layer
        # name.  Populated by forward pre-hooks registered in
        # _setup_sparse_attention(); cleared at the start of each step.
        self._sparse_q_captures: dict[str, torch.Tensor] = {}
        # Hook handles so they can be removed if needed.
        self._sparse_q_hooks: list[torch.utils.hooks.RemovableHandle] = []
        # Emit sparse branch probe logs only when explicitly enabled.
        self._sparse_probe_info_enabled: bool = (
            int(os.getenv("VLLM_SPARSE_PROBE_INFO", "0")) == 1
        )
        # Sparse performance stats switch (independent from debug/probe logs).
        self._sparse_perf_stats_enabled: bool = (
            int(os.getenv("VLLM_SPARSE_PERF_STATS", "0")) == 1
        )
        # Fused Triton pack kernel for the sparse decode ``num_reqs == 1``
        # fast path.  Bitwise-equivalent to the legacy nonzero + divmod +
        # index_select + sum chain, but cuts the per-layer kernel launch
        # count from ~5 down to 2 (count + pack).  Measured steady-state
        # (VLLM_SPARSE_PERF_WARMUP_SKIP=20 on a 20-sparse-layer Qwen-
        # style decode): forward_ms 54.59 → 52.07 (-2.52 ms/step), with
        # the bulk of the win coming from the num_reqs==1 finalize
        # reusing the Triton-produced int32 ``head_offsets`` (skipping
        # a redundant ``empty + zero + cumsum`` chain).  Defaults to
        # on; set ``VLLM_SPARSE_TRITON_PACK=0`` to fall back to the
        # legacy path for bisection.
        self._sparse_triton_pack_enabled: bool = (
            int(os.getenv("VLLM_SPARSE_TRITON_PACK", "1")) == 1
        )
        # ── Retroinfer-style cluster retrieval (Phase 5) ────────────────
        # The new path replaces the per-token top-k selector with a
        # cluster-level selector (``_sparse_online_select_clusters_batched``)
        # + pre-gathered per-head exec buffers
        # (``_sparse_retroinfer_expand_and_gather_single_req``).  FA consumes
        # these via the ``_forward_retroinfer_exec_buf`` dispatch.  Two
        # env vars govern rollout so we can bisect regressions without a
        # code-level revert:
        #
        # * ``VLLM_SPARSE_LEGACY_TOKEN_TOPK=1`` — force the old per-token
        #   topk + compact-kv gather path for every sparse layer.  Useful
        #   for regression bisection during Phase 5 rollout.  Defaults off.
        #
        # * ``VLLM_SPARSE_ESTIMATION_BUDGET=K`` — how many additional
        #   clusters to allocate to the estimation zone (on top of
        #   ``spec.nprobe`` for the retrieval zone).  ``K=0`` degenerates
        #   to a retrieval-only path (matches the legacy semantic up to
        #   cluster-vs-token granularity).  Defaults to 0 so an initial
        #   rollout step can be validated against the previous reference
        #   before enabling the estimation zone.
        self._sparse_legacy_token_topk: bool = (
            int(os.getenv("VLLM_SPARSE_LEGACY_TOKEN_TOPK", "0")) == 1
        )
        self._sparse_estimation_budget: int = max(
            0, int(os.getenv("VLLM_SPARSE_ESTIMATION_BUDGET", "0"))
        )
        # Per-layer scratch: when the Triton pack fast path runs in the
        # per-req loop, it already computes the int32 exclusive-prefix
        # ``head_offsets`` needed by finalize.  Pass it across via this
        # attribute to avoid a redundant ``empty + cumsum`` pair in
        # the num_reqs==1 finalize.  Cleared after each use.
        self._sparse_triton_head_offsets_cache: torch.Tensor | None = None
        # Phase-B: Triton pack also emits the per-entry ``kv_token_ids``
        # (int64, one kv_head id per selected entry) in the same
        # kernel.  Stash it here so the finalize can reuse directly,
        # skipping ``torch.repeat_interleave(kv_head_ids, counts)``.
        self._sparse_triton_kv_token_ids_cache: torch.Tensor | None = None
        self._decode_perf_stats_enabled: bool = bool(
            int(os.getenv("VLLM_DECODE_PERF_STATS", "0"))
        )
        self._e2e_perf_trace_enabled: bool = envs.VLLM_E2E_PERF_TRACE
        # Wall-clock start of the current step, captured at ``execute_model``
        # entry so ``sample_tokens`` can close an end-to-end window covering
        # both halves of the two-phase decode RPC (execute_model +
        # sample_tokens).  The ``avg_total_ms`` field inside ``[DecodePerf]``
        # only covers ``execute_model``; ``[DecodePerfE2E]`` logged from
        # ``sample_tokens`` covers the whole step.  ``None`` when stats are
        # disabled or when no step is in flight.
        self._decode_perf_step_t0: float | None = None
        self._decode_perf_step_exec_ms: float = 0.0
        # Wall-clock at ``execute_model`` EXIT (after the [DecodePerf] log
        # emits).  ``sample_tokens`` diffs against ``time.perf_counter()`` at
        # entry to expose the "engine-side gap" between the two RPC halves
        # (grammar apply, PP dispatch, etc.) that would otherwise silently
        # roll into ``avg_sample_tokens_ms``.
        self._decode_perf_step_exec_exit_t: float | None = None
        self._sparse_perf_log_interval: int = max(
            1, int(os.getenv("VLLM_SPARSE_PERF_LOG_INTERVAL", "50"))
        )
        # Post-warmup step counter — used for ``window_steps`` modulo
        # logging.  Only ticks once the warmup-skip gate below has
        # cleared.
        self._sparse_perf_steps: int = 0
        # Total steps seen (including warmup).  Records emitted while
        # ``total < warmup_skip`` are dropped so that steady-state
        # timings aren't polluted by CUDA / allocator / GPU-clock
        # warmup (observed: early decode steps sit ~10% higher than
        # steady-state).  Defaults to 0 for backwards-compat; set
        # ``VLLM_SPARSE_PERF_WARMUP_SKIP=N`` to drop the first ``N``
        # post-prefill decode flush cycles before aggregating.
        self._sparse_perf_total_steps: int = 0
        self._sparse_perf_warmup_skip: int = max(
            0, int(os.getenv("VLLM_SPARSE_PERF_WARMUP_SKIP", "0"))
        )
        self._sparse_perf_accum_ms: dict[str, float] = defaultdict(float)
        self._sparse_perf_accum_calls: dict[str, int] = defaultdict(int)
        # Optional sparse decode token debug logs.
        self._sparse_debug_decode_tokens: bool = bool(
            int(os.getenv("VLLM_SPARSE_DEBUG_DECODE_TOKENS", "0"))
        )
        # Latest scheduler-selected logical blocks per request. This is used to
        # print the KV blocks/tokens that are actually fed to sparse attention.
        self._sparse_debug_selected_logical_blocks: dict[str, list[int]] = {}
        # Merged (union) logical indices aligned with ``req_state.block_ids`` history.
        self._sparse_merged_logical: dict[str, list[int]] = {}
        # Per-layer logical selections for building layer-specific sparse block tables.
        self._sparse_by_layer_logical: dict[str, dict[str, list[int]]] = {}
        self._sparse_chrono_phys: dict[str, list[int]] = {}
        self._sparse_layer_spec_by_name: dict[str, SparseAttentionSpec] = {}
        self._sparse_layer_gid_by_name: dict[str, int] = {}
        # Per-layer static context lazily populated on first sparse forward.
        # Holds ``num_heads``, ``num_kv_heads``, ``head_size`` and the cached
        # ``kv_to_qh_tensor`` mapping so ``_build_sparse_runtime_q_head_gather``
        # does not rebuild them every step.
        self._sparse_layer_ctx: dict[str, dict[str, object]] = {}
        self._sparse_decode_trace_qhead_ms: float = 0.0
        self._sparse_decode_trace_qhead_calls: int = 0
        self._sparse_decode_trace_qhead_none: int = 0
        self._sparse_decode_trace_retro_ms: float = 0.0
        self._sparse_decode_trace_retro_calls: int = 0
        self._sparse_online_index: dict[str, dict[str, _SparseOnlineLayerState]] = {}
        # GPU-resident handoff: per-(req, unit_key) precomputed ``value_sum``
        # ``[K, D]`` fp32 tensors written during ``_collect_sparse_features``
        # (next to the K-Means call, while V rows are still hot in the paged
        # cache's L2) and consumed one line later in
        # ``_update_sparse_online_index`` to avoid a redundant D2H→numpy→H2D
        # roundtrip.  Intentionally not a function parameter: the two callers
        # are adjacent in ``sample_tokens`` and bouncing it through a tuple
        # would bloat the already-wide return signature of
        # ``_collect_sparse_features``.  Cleared at the end of each update.
        self._sparse_pending_value_sum_gpu: dict[str, dict[str, torch.Tensor]] = {}
        # Snapshot of per-request output length BEFORE this step appends sampled
        # tokens. Used by sparse feature collection to classify prefill/decode
        # boundary correctly.
        self._sparse_output_tokens_before_step: dict[str, int] = {}
        # Prefill-features-emitted guard: ``is_prefill_done`` can re-fire on
        # subsequent steps (async placeholder / first-decode boundary – see
        # comment near ``_collect_sparse_features``).  Re-sending the full
        # ``[N_prompt_tokens, D]`` feature dict forces the scheduler to redo
        # ``SparseKVManager.indexing()`` which rebuilds per-token Python state
        # at ~7-8s per prompt.  Track emission here and gate so the payload
        # goes out exactly once per request.
        self._sparse_prefill_emitted: set[str] = set()
        # Track requests whose first-step commit was deferred under async
        # scheduling so the deferral does not repeat indefinitely.
        self._sparse_deferred_once: set[str] = set()
        # One-shot diagnostics for logical/physical length mismatch.
        self._sparse_mismatch_logged: set[tuple[str, str, int]] = set()
        self._sparse_debug_decode_tokens_max: int = int(
            os.getenv("VLLM_SPARSE_DEBUG_DECODE_TOKENS_MAX", "256")
        )
        # One-shot KV + logits diagnostics for the first decode step only
        # (reduces log volume vs full per-layer compact_gather tracing).
        self._sparse_debug_first_token: bool = bool(
            int(os.getenv("VLLM_SPARSE_DEBUG_FIRST_TOKEN", "0"))
        )
        self._sparse_first_token_sample_logged: set[str] = set()
        # <= 0 means "no cap" (print all selected blocks).
        self._sparse_debug_max_blocks: int = int(
            os.getenv("VLLM_SPARSE_DEBUG_MAX_BLOCKS", "0")
        )
        self._sparse_debug_tokenizer = None
        if (
            self._sparse_debug_decode_tokens
            or self._sparse_debug_first_token
            or _SPARSE_HARD_DEBUG_FIRST_NEW_TOKEN
        ):
            try:
                self._sparse_debug_tokenizer = cached_tokenizer_from_config(
                    self.model_config
                )
            except Exception as e:
                logger.warning(
                    "Failed to initialize tokenizer for sparse decode debug logs: %s",
                    e,
                )
        if self._sparse_debug_decode_tokens:
            logger.info(
                "Sparse decode debug logging enabled: "
                "VLLM_SPARSE_DEBUG_DECODE_TOKENS=1 "
                "VLLM_SPARSE_DEBUG_DECODE_TOKENS_MAX=%d "
                "VLLM_SPARSE_DEBUG_MAX_BLOCKS=%d",
                self._sparse_debug_decode_tokens_max,
                self._sparse_debug_max_blocks,
            )
        if self._sparse_debug_first_token:
            logger.info(
                "Sparse first-token debug enabled: VLLM_SPARSE_DEBUG_FIRST_TOKEN=1",
            )
        if _SPARSE_HARD_DEBUG_FIRST_NEW_TOKEN:
            logger.critical(
                "_SPARSE_HARD_DEBUG_FIRST_NEW_TOKEN=True: process will "
                "os._exit(0) after valid output token count reaches %d.",
                int(_SPARSE_HARD_DEBUG_STOP_AFTER_OUTPUT_N),
            )
        if self._sparse_perf_stats_enabled:
            logger.info(
                "Sparse perf stats enabled: VLLM_SPARSE_PERF_STATS=1 "
                "VLLM_SPARSE_PERF_LOG_INTERVAL=%d "
                "VLLM_SPARSE_PERF_WARMUP_SKIP=%d",
                self._sparse_perf_log_interval,
                self._sparse_perf_warmup_skip,
            )
        if self._decode_perf_stats_enabled:
            logger.info("Decode perf stats enabled: VLLM_DECODE_PERF_STATS=1")
        if self._e2e_perf_trace_enabled:
            logger.info("E2E perf trace enabled: VLLM_E2E_PERF_TRACE=1")
        # Async sparse prefill D2H copy (independent switch).
        self._sparse_async_d2h_enabled: bool = (
            int(os.getenv("VLLM_SPARSE_ASYNC_D2H", "0")) == 1
        )
        self._sparse_d2h_stream: torch.cuda.Stream | None = None
        if self._sparse_async_d2h_enabled and self.device.type == "cuda":
            self._sparse_d2h_stream = torch.cuda.Stream(device=self.device)
            logger.info("Sparse async D2H enabled: VLLM_SPARSE_ASYNC_D2H=1")

        # mm_hash ->  encoder_output
        self.encoder_cache: dict[str, torch.Tensor] = {}
        self.late_interaction_runner = LateInteractionRunner()

        self.use_aux_hidden_state_outputs = False
        # Set up speculative decoding.
        # NOTE(Jiayi): currently we put the entire draft model on
        # the last PP rank. This is not ideal if there are many
        # layers in the draft model.
        if self.speculative_config and get_pp_group().is_last_rank:
            self.drafter: (
                NgramProposer  # noqa: F823
                | NgramProposerGPU
                | SuffixDecodingProposer
                | EagleProposer
                | DraftModelProposer
                | MedusaProposer
                | ExtractHiddenStatesProposer
            )
            if self.speculative_config.method == "ngram":
                from vllm.v1.spec_decode.ngram_proposer import NgramProposer

                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.uses_draft_model():
                self.drafter = DraftModelProposer(
                    vllm_config=self.vllm_config,
                    device=self.device,
                    runner=self,
                )
            elif self.speculative_config.use_ngram_gpu():
                self.drafter = NgramProposerGPU(self.vllm_config, self.device, self)
                self.num_tokens_no_spec_gpu = torch.zeros(
                    self.max_num_reqs, dtype=torch.int32, device=device
                )
                self.token_ids_gpu_tensor = torch.zeros(
                    self.max_num_reqs,
                    self.max_model_len,
                    dtype=torch.int32,
                    device=device,
                )
                self._ngram_pinned_idx_buf = torch.zeros(
                    self.max_num_reqs, dtype=torch.long, pin_memory=True
                )
                self._ngram_pinned_val_buf = torch.zeros(
                    self.max_num_reqs, dtype=torch.int32, pin_memory=True
                )
            elif self.speculative_config.method == "suffix":
                self.drafter = SuffixDecodingProposer(self.vllm_config)
            elif self.speculative_config.use_eagle():
                self.drafter = EagleProposer(self.vllm_config, self.device, self)
                if self.speculative_config.method == "eagle3":
                    self.use_aux_hidden_state_outputs = (
                        self.drafter.eagle3_use_aux_hidden_state
                    )
            elif self.speculative_config.method == "medusa":
                self.drafter = MedusaProposer(
                    vllm_config=self.vllm_config, device=self.device
                )
            elif self.speculative_config.method == "extract_hidden_states":
                self.drafter = ExtractHiddenStatesProposer(
                    vllm_config=self.vllm_config, device=self.device
                )
                self.use_aux_hidden_state_outputs = True
            else:
                raise ValueError(
                    "Unknown speculative decoding method: "
                    f"{self.speculative_config.method}"
                )
            self.rejection_sampler = RejectionSampler(self.sampler)

        self.num_spec_tokens = 0
        if self.speculative_config:
            self.num_spec_tokens = self.speculative_config.num_speculative_tokens
            draft_config = self.speculative_config.draft_model_config
            if draft_config is not None and draft_config.max_model_len is not None:
                self.effective_drafter_max_model_len = draft_config.max_model_len
            else:
                self.effective_drafter_max_model_len = self.max_model_len

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}
        # NOTE(rob): num_prompt_logprobs only includes reqs
        # that are currently in the prefill phase.
        self.num_prompt_logprobs: dict[str, int] = {}
        self.comm_stream = torch.cuda.Stream()

        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        logits_processors = model_config.logits_processors
        custom_logitsprocs: Sequence[str | type[LogitsProcessor]] = (
            tuple(logits_processors) if logits_processors is not None else ()
        )
        placeholder_block_size = (
            self.cache_config.block_size or CacheConfig.DEFAULT_BLOCK_SIZE
        )
        self._init_block_sizes = [placeholder_block_size]
        self._init_kernel_block_sizes = [placeholder_block_size]
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            # We need to use the encoder length for encoder-decoder
            # because of KV cache for cross-attention.
            max_model_len=max(self.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[placeholder_block_size],
            kernel_block_sizes=[placeholder_block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config,
                self.device,
                self.pin_memory,
                self.is_pooling_model,
                custom_logitsprocs,
            ),
            # We currently don't know whether a particular custom logits processor
            # uses output token ids so we set this conservatively.
            logitsprocs_need_output_token_ids=bool(custom_logitsprocs),
            is_pooling_model=self.is_pooling_model,
            cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
        )

        # Separate cuda stream for overlapping transfer of sampled token ids from
        # GPU to CPU when async scheduling is enabled.
        self.async_output_copy_stream: torch.cuda.Stream | None = None
        # cuda event to synchronize use of reused CPU tensors between steps
        # when async scheduling is enabled.
        self.prepare_inputs_event: torch.Event | None = None
        if self.use_async_scheduling:
            self.async_output_copy_stream = torch.cuda.Stream()
            self.prepare_inputs_event = torch.Event()

        # self.cudagraph_batch_sizes sorts in ascending order.
        if (
            self.compilation_config.cudagraph_capture_sizes
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            self.cudagraph_batch_sizes = sorted(
                self.compilation_config.cudagraph_capture_sizes
            )
        else:
            self.cudagraph_batch_sizes = []

        # Cache the device properties.
        self._init_device_properties()

        # Encoder timing registry for observability
        self.encoder_timing_registry: dict[str, EncoderTimingStats] = {}
        self._encoder_timing_lock = threading.Lock()

        # Persistent buffers for CUDA graphs.
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        self.positions = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
        self.query_start_loc = self._make_buffer(
            self.max_num_reqs + 1, dtype=torch.int32
        )
        self.seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        self.encoder_seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        if self.dcp_world_size > 1:
            self.dcp_local_seq_lens = self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            )
        # Because inputs_embeds may be bfloat16 and we don't need a numpy
        # version of this tensor, avoid a RuntimeError by not creating a
        # numpy buffer.
        self.inputs_embeds = self._make_buffer(
            self.max_num_tokens, self.inputs_embeds_size, dtype=self.dtype, numpy=False
        )
        self.is_token_ids = self._make_buffer(self.max_num_tokens, dtype=torch.bool)
        self.discard_request_mask = self._make_buffer(
            self.max_num_reqs, dtype=torch.bool
        )
        self.num_decode_draft_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        self.num_accepted_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int64
        )

        # Only relevant for multimodal models
        if self.supports_mm_inputs:
            # Double buffer to avoid race condition: previous iteration's async
            # copy may still be reading from CPU while current iteration writes.
            self.is_mm_embed_buffers = [
                self._make_buffer(self.max_num_tokens, dtype=torch.bool),
                self._make_buffer(self.max_num_tokens, dtype=torch.bool),
            ]
            self.is_mm_embed_idx = 0

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = self._make_buffer(
                (3, self.max_num_tokens + 1), dtype=torch.int64
            )

        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        if self.uses_xdrope_dim > 0:
            # Similar to mrope but use assigned dimension number for RoPE, 4 as default.
            self.xdrope_positions = self._make_buffer(
                (self.uses_xdrope_dim, self.max_num_tokens + 1), dtype=torch.int64
            )

        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: IntermediateTensors | None = None

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(
            max(self.max_num_reqs + 1, self.max_model_len, self.max_num_tokens),
            dtype=np.int64,
        )

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}
        self.kv_sharing_fast_prefill_eligible_layers: set[str] = set()

        self.kv_sharing_fast_prefill_logits_indices = None
        if self.cache_config.kv_sharing_fast_prefill:
            self.kv_sharing_fast_prefill_logits_indices = torch.zeros(
                self.max_num_tokens, dtype=torch.int32, device=self.device
            )

        self.uniform_decode_query_len = 1 + self.num_spec_tokens

        # When spec decode is active, the mamba backend classifies requests
        # with query_len <= reorder_batch_threshold as "decodes". Prefill
        # chunks that fall under this threshold get processed via the decode
        # path, which stores intermediate states at sequential slots. We must
        # set num_accepted_tokens to the chunk's query_len for those requests
        # so the next iteration reads from the correct final-state slot.
        # Prefills that went through the actual prefill path should keep the
        # default value of 1 (the prefill path stores state at slot 0 only).
        self.needs_prefill_as_decode_slots: bool = False
        self.prefill_as_decode_num_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )

        # Cudagraph dispatcher for runtime cudagraph dispatching.
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        self.mm_budget = (
            MultiModalBudget(self.vllm_config, self.mm_registry)
            if self.supports_mm_inputs
            else None
        )

        self.reorder_batch_threshold: int | None = None

        # Attention layers that are only in the KVCacheConfig of the runner
        # (e.g., KV sharing, encoder-only attention), but not in the
        # KVCacheConfig of the scheduler.
        self.runner_only_attn_layers: set[str] = set()

        # Cached outputs.
        self._draft_token_ids: list[list[int]] | torch.Tensor | None = None
        # N-gram GPU path: async D2H buffer/event for per-request valid draft counts.
        self._num_valid_draft_tokens: torch.Tensor | None = None
        self._num_valid_draft_tokens_cpu: torch.Tensor | None = None
        self._num_valid_draft_tokens_event: torch.cuda.Event | None = None
        self._num_valid_draft_tokens_copy_stream: torch.cuda.Stream | None = None
        if (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        ):
            self._num_valid_draft_tokens_cpu = torch.empty(
                self.max_num_reqs, dtype=torch.int32, pin_memory=self.pin_memory
            )
            self._num_valid_draft_tokens_event = torch.cuda.Event()
            self._num_valid_draft_tokens_copy_stream = torch.cuda.Stream()

        self._draft_token_req_ids: list[str] | None = None
        self.transfer_event = torch.Event()
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_num_reqs, 1),
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.pin_memory,
        )

        # Pre-allocated tensor for copying valid sampled token counts to CPU,
        # with dedicated stream for overlapping and event for coordination.
        self.valid_sampled_token_count_event: torch.Event | None = None
        self.valid_sampled_token_count_copy_stream: torch.cuda.Stream | None = None
        # We also copy the drafted tokens to the CPU asynchronously,
        # in case we need them for structured outputs.
        self.draft_token_ids_event: torch.Event | None = None
        self.draft_token_ids_copy_stream: torch.cuda.Stream | None = None
        self.valid_sampled_token_count_cpu: torch.Tensor | None = None
        self.draft_token_ids_cpu: torch.Tensor | None = None
        self.num_accepted_tokens_event: torch.Event | None = None
        if self.num_spec_tokens:
            self.draft_token_ids_event = torch.Event()
            self.num_accepted_tokens_event = torch.Event()
            self.draft_token_ids_copy_stream = torch.cuda.Stream()
            self.draft_token_ids_cpu = torch.empty(
                (self.max_num_reqs, self.num_spec_tokens),
                dtype=torch.int64,
                device="cpu",
                pin_memory=self.pin_memory,
            )
            if self.use_async_scheduling:
                self.valid_sampled_token_count_event = torch.Event()
                self.valid_sampled_token_count_copy_stream = torch.cuda.Stream()
                self.valid_sampled_token_count_cpu = torch.empty(
                    self.max_num_reqs,
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=self.pin_memory,
                )

        # Model weight offloader
        # Make sure this is called before any get_offloader call
        set_offloader(create_offloader(self.offload_config))

        # Ephemeral state transferred between execute_model() and sample_tokens().
        self.execute_model_state: ExecuteModelState | None = None
        self.kv_connector_output: KVConnectorOutput | None = None
        self.mamba_state_idx: dict[str, int] = {}
        self._mamba_copy_bufs: mamba_utils.MambaCopyBuffers | None = None
        self.layerwise_nvtx_hooks_registered = False

    def update_max_model_len(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len
        if self.speculative_config:
            draft_config = self.speculative_config.draft_model_config
            if draft_config is None or draft_config.max_model_len is None:
                self.effective_drafter_max_model_len = self.max_model_len

    def reset_mm_cache(self) -> None:
        """
        Clear the multi-modal cache that was used during profiling,
        but no longer needed during inference.
        """
        if self.mm_budget:
            self.mm_budget.reset_cache()
        self.late_interaction_runner.clear()

    def reset_encoder_cache(self) -> None:
        """Clear the GPU-side encoder cache storing vision embeddings.

        This should be called when model weights are updated to ensure
        stale embeddings computed with old weights are not reused.
        """
        self.encoder_cache.clear()
        self.late_interaction_runner.clear()

    @torch.inference_mode()
    def init_fp8_kv_scales(self) -> None:
        """
        Re-initialize the KV cache and FP8 scales after waking from sleep.
        1. Zero out the KV cache tensors to remove garbage data from re-allocation.
        2. Reset Attention layer scaling factors (_k_scale, _v_scale) to 1.0.
          If these are left at 0.0 (default after wake_up), all KV cache values
          become effectively zero, causing gibberish output.
        """
        if not self.cache_config.cache_dtype.startswith("fp8"):
            return

        kv_caches = getattr(self, "kv_caches", [])
        for cache_tensor in kv_caches:
            if cache_tensor is not None:
                cache_tensor.zero_()

        k_attr_names = ("_k_scale", "k_scale")
        v_attr_names = ("_v_scale", "v_scale")

        attn_layers = self.compilation_config.static_forward_context
        for name, module in attn_layers.items():
            if isinstance(module, (Attention, MLAAttention)):
                # TODO: Generally, scale is 1.0 if user uses on-the-fly fp8
                # kvcache quant. However, to get better accuracy, compression
                # frameworks like llm-compressors allow users to tune the
                # scale. We may need to restore the specific calibrated scales
                # here in the future.
                k_scale_val, v_scale_val = 1.0, 1.0

                # Processing K Scale
                for attr in k_attr_names:
                    if hasattr(module, attr):
                        param = getattr(module, attr)
                        if isinstance(param, torch.Tensor):
                            param.fill_(k_scale_val)

                # Processing V Scale
                for attr in v_attr_names:
                    if hasattr(module, attr):
                        param = getattr(module, attr)
                        if isinstance(param, torch.Tensor):
                            param.fill_(v_scale_val)

    def _get_positions(self, num_tokens: Any):
        if isinstance(num_tokens, int):
            if self.uses_mrope:
                return self.mrope_positions.gpu[:, :num_tokens]
            if self.uses_xdrope_dim > 0:
                return self.xdrope_positions.gpu[:, :num_tokens]
            return self.positions.gpu[:num_tokens]
        else:
            if self.uses_mrope:
                return self.mrope_positions.gpu[:, num_tokens]
            if self.uses_xdrope_dim > 0:
                return self.xdrope_positions.gpu[:, num_tokens]
            return self.positions.gpu[num_tokens]

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype, numpy: bool = True
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            with_numpy=numpy,
        )

    def _get_mamba_copy_bufs(self) -> mamba_utils.MambaCopyBuffers:
        if self._mamba_copy_bufs is None:
            self._mamba_copy_bufs = mamba_utils.MambaCopyBuffers.create(
                self.max_num_reqs,
                self.kv_cache_config,
                self.model.get_mamba_state_copy_func(),
                self._make_buffer,
            )
        return self._mamba_copy_bufs

    def _init_model_kwargs(self):
        model_kwargs = dict[str, Any]()

        if not self.is_pooling_model:
            return model_kwargs

        num_reqs = self.input_batch.num_reqs
        pooling_params = self.input_batch.get_pooling_params()

        token_type_id_requests = dict[int, Any]()
        for i, param in enumerate(pooling_params):
            if (
                param.extra_kwargs is not None
                and (token_types := param.extra_kwargs.get("compressed_token_type_ids"))
                is not None
            ):
                token_type_id_requests[i] = token_types

        if len(token_type_id_requests) == 0:
            return model_kwargs

        seq_lens = self.seq_lens.gpu[:num_reqs]
        token_type_ids = []

        for i in range(num_reqs):
            pos = token_type_id_requests.get(i, seq_lens[i])
            ids = (torch.arange(seq_lens[i]) >= pos).int()
            token_type_ids.append(ids)

        model_kwargs["token_type_ids"] = torch.concat(token_type_ids).to(
            device=self.device
        )
        return model_kwargs

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """
        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.
        """
        # Attention free models have zero kv_cache_groups, however models
        # like Mamba are also attention free but use the kv_cache for
        # keeping its internal state. This is why we check the number
        # of kv_cache groups instead of solely checking
        # for self.model_config.is_attention_free.
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return

        if self.reorder_batch_threshold is not None:
            reorder_batch_to_split_decodes_and_prefills(
                self.input_batch,
                scheduler_output,
                decode_threshold=self.reorder_batch_threshold,
            )

    def _init_kv_zero_meta(self) -> None:
        """One-time precomputation for _zero_block_ids.

        Delegates to KVBlockZeroer.init_meta with the runner's state.
        Called from gpu_worker.py outside the CuMem pool context.
        """
        self._kv_block_zeroer = KVBlockZeroer(self.device, self.pin_memory)
        self._kv_block_zeroer.init_meta(
            attn_groups_iter=self._kv_cache_spec_attn_group_iterator(),
            kernel_block_sizes=self._kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            runner_only_attn_layers=self.runner_only_attn_layers,
            static_forward_context=(self.compilation_config.static_forward_context),
        )

    def _zero_block_ids(self, block_ids: list[int]) -> None:
        """Zero the KV cache memory for the given block IDs."""
        if hasattr(self, "_kv_block_zeroer"):
            self._kv_block_zeroer.zero_block_ids(block_ids)

    # Note: used for model runner override.
    def _init_device_properties(self) -> None:
        """Initialize attributes from torch.cuda.get_device_properties"""

        self.num_sms = num_compute_units(self.device.index)

    # Note: used for model runner override.
    def _sync_device(self) -> None:
        torch.accelerator.synchronize()

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.num_prompt_logprobs.pop(req_id, None)
            self._sparse_deferred_once.discard(req_id)
            self._sparse_prefill_emitted.discard(req_id)
            self._sparse_online_index.pop(req_id, None)
        self.late_interaction_runner.on_requests_finished(
            scheduler_output.finished_req_ids
        )
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # Zero GPU memory for freshly allocated cache blocks to prevent
        # stale NaN/data from corrupting attention or SSM computation.
        if scheduler_output.new_block_ids_to_zero:
            self._zero_block_ids(scheduler_output.new_block_ids_to_zero)

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        # NOTE(zhuohan): cached_req_ids and resumed_req_ids are usually disjoint,
        # so `(scheduled_req_ids - resumed_req_ids) == scheduled_req_ids` holds
        # apart from the forced-preemption case in reset_prefix_cache. And in
        # that case we include the resumed_req_ids in the unscheduled set so
        # that they get cleared from the persistent batch before being re-scheduled
        # in the normal resumed request path.
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        is_ngram_gpu = (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        )
        if is_ngram_gpu:
            ngram_gpu_new_reqs: list[CachedRequestState] = []

        reqs_to_add: list[CachedRequestState] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            if req_id in self.requests:
                # For streaming case only.
                req_state = self._update_streaming_request(req_id, new_req_data)
                reqs_to_add.append(req_state)
                continue

            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if (
                sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED
            ):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if self.is_pooling_model:
                assert pooling_params is not None
                task = pooling_params.task
                assert task is not None, "You did not set `task` in the API"

                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            self.requests[req_id] = req_state
            self.late_interaction_runner.register_request(req_id, pooling_params)

            if sampling_params and sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = (
                    self.input_batch.vocab_size
                    if sampling_params.prompt_logprobs == -1
                    else sampling_params.prompt_logprobs
                )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self._init_mrope_positions(req_state)

            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            if self.uses_xdrope_dim > 0:
                self._init_xdrope_positions(req_state)

            reqs_to_add.append(req_state)
            # Track new requests for ngram_gpu full tensor copy
            if is_ngram_gpu:
                ngram_gpu_new_reqs.append(req_state)

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        sparse_selected_map = req_data.sparse_selected_block_indices or {}
        sparse_retrieve_map = req_data.sparse_retrieve_block_indices or {}
        sparse_by_layer_map = (
            req_data.sparse_selected_block_indices_by_layer or {}
        )
        sparse_chrono_phys_map = req_data.sparse_chrono_phys_block_ids or {}
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        token_sparse_async_real_tokens = False
        if self.use_async_scheduling and self._has_sparse_attn and hasattr(
            self, "kv_cache_config"
        ):
            for group in self.kv_cache_config.kv_cache_groups:
                spec = group.kv_cache_spec
                if (
                    isinstance(spec, SparseAttentionSpec)
                    and spec.cluster_granularity == "token"
                ):
                    token_sparse_async_real_tokens = True
                    break

        # Save scheduler-allocated spec lengths before trimming so
        # prev_num_draft_len keeps the optimistic count for rejection correction.
        original_num_spec_per_req: dict[str, int] = {}
        if (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        ):
            for req_id, toks in scheduled_spec_tokens.items():
                original_num_spec_per_req[req_id] = len(toks)
            update_scheduler_for_invalid_drafts(
                self._num_valid_draft_tokens_event,
                self._num_valid_draft_tokens_cpu,
                scheduler_output,
                self.input_batch.req_id_to_index,
            )

        # Wait until valid_sampled_tokens_count is copied to cpu,
        # then use it to update actual num_computed_tokens of each request.
        valid_sampled_token_count = self._get_valid_sampled_token_count()

        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            num_output_tokens = req_data.num_output_tokens[i]
            req_index = self.input_batch.req_id_to_index.get(req_id)

            if req_state.prev_num_draft_len and self.use_async_scheduling:
                # prev_num_draft_len is used in async scheduling mode with
                # spec decode. it indicates if need to update num_computed_tokens
                # of the request. for example:
                # first step: num_computed_tokens = 0, spec_tokens = [],
                # prev_num_draft_len = 0.
                # second step: num_computed_tokens = 100(prompt length),
                # spec_tokens = [a,b], prev_num_draft_len = 0.
                # third step: num_computed_tokens = 100 + 2, spec_tokens = [c,d],
                # prev_num_draft_len = 2.
                # num_computed_tokens in first step and second step doesn't contain
                # the spec tokens length, but in third step it contains the
                # spec tokens length. we only need to update num_computed_tokens
                # when prev_num_draft_len > 0.
                if req_index is None:
                    req_state.prev_num_draft_len = 0
                else:
                    assert self.input_batch.prev_req_id_to_index is not None
                    prev_req_index = self.input_batch.prev_req_id_to_index[req_id]
                    num_accepted = valid_sampled_token_count[prev_req_index] - 1
                    num_rejected = req_state.prev_num_draft_len - num_accepted
                    num_computed_tokens -= num_rejected
                    if (
                        token_sparse_async_real_tokens
                        and num_accepted > 0
                        and self.input_batch.prev_sampled_token_ids is not None
                    ):
                        accepted_tid = int(
                            self.input_batch.prev_sampled_token_ids[
                                prev_req_index, 0
                            ].item()
                        )
                        req_state.output_token_ids.extend(
                            [accepted_tid] * num_accepted
                        )
                    else:
                        req_state.output_token_ids.extend([-1] * num_accepted)

                    if is_ngram_gpu and num_accepted > 0 and req_index is not None:
                        self.input_batch.num_tokens_no_spec[req_index] += num_accepted

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens
            new_token_ids_len = -1

            _worker_had_output_before_trunc = len(req_state.output_token_ids) > 0
            if not is_last_rank:
                if not req_data.new_token_ids:
                    # Async scheduled PP: Sampled tokens propagated via GPU broadcast.
                    new_token_ids: list[int] = []
                else:
                    # Non-async scheduling with PP: The scheduler sends
                    # sampled token ids back because there's no direct communication
                    # between the first-stage worker and the last-stage worker.
                    new_token_ids = req_data.new_token_ids[i]
                    # Add the sampled token(s) from the previous step (if any).
                    # This doesn't include "unverified" tokens like spec tokens.
                    num_new_tokens = (
                        num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                    )
                    if num_new_tokens == 1:
                        # Avoid slicing list in most common case.
                        req_state.output_token_ids.append(new_token_ids[-1])
                    elif num_new_tokens > 0:
                        req_state.output_token_ids.extend(
                            new_token_ids[-num_new_tokens:]
                        )
                new_token_ids_len = len(new_token_ids)
            elif num_output_tokens < len(req_state.output_token_ids):
                # Some output tokens were discarded due to a sync-KV-load
                # failure. Align the cached state.
                del req_state.output_token_ids[num_output_tokens:]
                if req_index is not None:
                    end_idx = (
                        self.input_batch.num_prompt_tokens[req_index]
                        + num_output_tokens
                    )
                    self.input_batch.num_tokens_no_spec[req_index] = end_idx

            selected_logical_blocks = sparse_selected_map.get(req_id)
            retrieve_logical_blocks = sparse_retrieve_map.get(req_id)
            selected_count = (
                0 if selected_logical_blocks is None else len(selected_logical_blocks)
            )
            retrieve_count = (
                0 if retrieve_logical_blocks is None else len(retrieve_logical_blocks)
            )
            incoming_row_lens = (
                [] if new_block_ids is None else [len(ids) for ids in new_block_ids]
            )
            prev_row_lens = [len(ids) for ids in req_state.block_ids]
            if self._has_sparse_attn and num_output_tokens > 0:
                by_layer = sparse_by_layer_map.get(req_id)
                chrono_by_req = sparse_chrono_phys_map.get(req_id)
                if by_layer is None or chrono_by_req is None:
                    logger.debug(
                        "[SparseRC:bridge_worker] req_id=%s "
                        "missing by_layer=%s chrono=%s "
                        "selected_len=%d retrieve_len=%d num_output_tokens=%d "
                        "incoming_row_lens=%s prev_row_lens=%s "
                        "cached_prev by_layer_keys=%d chrono_len=%d",
                        req_id,
                        by_layer is None,
                        chrono_by_req is None,
                        selected_count,
                        retrieve_count,
                        num_output_tokens,
                        incoming_row_lens,
                        prev_row_lens,
                        len(self._sparse_by_layer_logical.get(req_id, {})),
                        len(self._sparse_chrono_phys.get(req_id, [])),
                    )
            sparse_row_collapse_guard = (
                self._has_sparse_attn
                and num_output_tokens > 0
                and new_block_ids is not None
                and len(prev_row_lens) > 0
                and len(incoming_row_lens) > 0
                and max(prev_row_lens) > 1
                and max(incoming_row_lens) <= 1
                and selected_count == 0
                and retrieve_count == 0
            )
            sparse_should_replace_row = (
                self._has_sparse_attn
                and num_output_tokens > 0
                and not sparse_row_collapse_guard
            )

            # Update the block IDs.
            # For sparse decode requests we usually REPLACE the entire block
            # table row (not append) because the sparse manager rebuilds
            # req_to_blocks from scratch each step (Bug 1 fix).  But when
            # sparse selection unexpectedly goes empty while the request had
            # long history, replacing with a single decode block would collapse
            # seq_lens to one block and permanently poison subsequent steps.
            # In async scheduling the scheduler's num_output_tokens may
            # lag one step behind the worker's output_token_ids (the worker
            # already appended the sampled token in _bookkeeping_sync but
            # the scheduler hasn't confirmed it yet).  Clearing sparse
            # metadata based solely on num_output_tokens==0 would destroy
            # the prefill-stage sparse selection that was just computed,
            # causing compact gather to fall back to an empty token set
            # for the remainder of the request.
            # _worker_had_output_before_trunc captures the pre-truncation
            # state so the guard survives output_token_ids alignment.
            is_sparse_decode = (
                self._has_sparse_attn
                and (num_output_tokens > 0 or _worker_had_output_before_trunc)
            )
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    if is_sparse_decode and sparse_should_replace_row:
                        # Sparse decode: replace block IDs in-place so
                        # that any subsequent resumption uses the correct
                        # (sparse) block set.
                        for gid, new_ids_for_group in enumerate(new_block_ids):
                            req_state.block_ids[gid].clear()
                            req_state.block_ids[gid].extend(new_ids_for_group)
                        if (
                            self._sparse_probe_info_enabled
                            or self._sparse_debug_decode_tokens
                        ):
                            new_ids_lens = [len(ids) for ids in new_block_ids]
                            row_lens_after = [len(ids) for ids in req_state.block_ids]
                            logger.info(
                                "[SparseProbe] block_ids_update req_id=%s "
                                "selected_logical_blocks=%d retrieve_logical_blocks=%d "
                                "new_ids_lens=%s row_lens_after=%s",
                                req_id,
                                0
                                if selected_logical_blocks is None
                                else len(selected_logical_blocks),
                                0
                                if retrieve_logical_blocks is None
                                else len(retrieve_logical_blocks),
                                new_ids_lens,
                                row_lens_after,
                            )
                        if retrieve_logical_blocks is not None:
                            self._debug_log_sparse_selected_kv_tokens(
                                req_id=req_id,
                                req_state=req_state,
                                selected_logical_blocks=retrieve_logical_blocks,
                                zone_name="retrieve",
                            )
                        if selected_logical_blocks is not None:
                            self._sparse_debug_selected_logical_blocks[req_id] = list(
                                selected_logical_blocks
                            )
                            self._sparse_merged_logical[req_id] = list(
                                selected_logical_blocks
                            )
                            by_l = sparse_by_layer_map.get(req_id)
                            if by_l is not None:
                                self._sparse_by_layer_logical[req_id] = {
                                    k: list(v) for k, v in by_l.items()
                                }
                            else:
                                self._sparse_by_layer_logical.pop(req_id, None)
                            self._debug_log_sparse_selected_kv_tokens(
                                req_id=req_id,
                                req_state=req_state,
                                selected_logical_blocks=selected_logical_blocks,
                                zone_name="merged",
                            )
                        else:
                            if (
                                is_sparse_decode
                                and (
                                    self._sparse_probe_info_enabled
                                    or self._sparse_debug_decode_tokens
                                )
                            ):
                                logger.warning(
                                    "[SparseRC] clear_stale_sparse_meta "
                                    "req_id=%s kind=logical merged_prev=%d by_layer_prev=%d",
                                    req_id,
                                    len(self._sparse_merged_logical.get(req_id, [])),
                                    len(self._sparse_by_layer_logical.get(req_id, {})),
                                )
                            self._sparse_merged_logical.pop(req_id, None)
                            self._sparse_by_layer_logical.pop(req_id, None)
                        chrono = sparse_chrono_phys_map.get(req_id)
                        if chrono is not None:
                            self._sparse_chrono_phys[req_id] = list(chrono)
                        else:
                            if (
                                is_sparse_decode
                                and (
                                    self._sparse_probe_info_enabled
                                    or self._sparse_debug_decode_tokens
                                )
                            ):
                                logger.warning(
                                    "[SparseRC] clear_stale_sparse_meta "
                                    "req_id=%s kind=chrono chrono_prev_len=%d",
                                    req_id,
                                    len(self._sparse_chrono_phys.get(req_id, [])),
                                )
                            self._sparse_chrono_phys.pop(req_id, None)
                    elif is_sparse_decode:
                        # Guard against sparse-row collapse. Keep historical
                        # row shape and append truly new blocks only.
                        for block_ids, new_ids in zip(
                            req_state.block_ids, new_block_ids
                        ):
                            for bid in new_ids:
                                if not block_ids or block_ids[-1] != bid:
                                    block_ids.append(bid)
                        if (
                            self._sparse_probe_info_enabled
                            or self._sparse_debug_decode_tokens
                        ):
                            row_lens_after = [len(ids) for ids in req_state.block_ids]
                            logger.warning(
                                "[SparseProbe] block_ids_update_guard req_id=%s "
                                "selected_logical_blocks=%d retrieve_logical_blocks=%d "
                                "incoming_row_lens=%s prev_row_lens=%s row_lens_after=%s",
                                req_id,
                                selected_count,
                                retrieve_count,
                                incoming_row_lens,
                                prev_row_lens,
                                row_lens_after,
                            )
                    else:
                        # Standard: append new blocks to existing IDs.
                        for block_ids, new_ids in zip(
                            req_state.block_ids, new_block_ids
                        ):
                            block_ids.extend(new_ids)
            else:
                assert req_index is None
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.

                if self.use_async_scheduling and num_output_tokens > 0:
                    # We must recover the output token ids for resumed requests in the
                    # async scheduling case, so that correct input_ids are obtained.
                    resumed_token_ids = req_data.all_token_ids[req_id]
                    req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]

                reqs_to_add.append(req_state)
                # Track resumed requests for ngram_gpu full tensor copy
                if is_ngram_gpu:
                    ngram_gpu_new_reqs.append(req_state)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                if is_sparse_decode and sparse_should_replace_row:
                    # Sparse decode: rebuild the entire row (not append)
                    # because the sparse manager provides the full new block
                    # list (selected history + decode block).
                    self.input_batch.block_table.add_row(
                        new_block_ids, req_index
                    )
                elif is_sparse_decode:
                    self.input_batch.block_table.append_row(
                        new_block_ids, req_index
                    )
                else:
                    self.input_batch.block_table.append_row(
                        new_block_ids, req_index
                    )

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_token_index:end_token_index
                ] = new_token_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)
            # Restore scheduler-side draft count after ngram trimming.
            if original_num_spec_per_req:
                orig = original_num_spec_per_req.get(req_id, 0)
                if orig != req_state.prev_num_draft_len:
                    req_state.prev_num_draft_len = orig

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)
            self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

        # Incrementally update ngram_gpu tensors after batch is stable
        if is_ngram_gpu:
            update_ngram_gpu_tensors_incremental(
                self.input_batch,
                self.token_ids_gpu_tensor,
                self.num_tokens_no_spec_gpu,
                ngram_gpu_new_reqs,
                self.device,
                _pinned_idx_buf=self._ngram_pinned_idx_buf,
                _pinned_val_buf=self._ngram_pinned_val_buf,
            )

    def _update_states_after_model_execute(
        self, output_token_ids: torch.Tensor, scheduler_output: "SchedulerOutput"
    ) -> None:
        """Update the cached states after model execution.

        This is used for MTP/EAGLE for hybrid models, as in linear attention,
        only the last token's state is kept. In MTP/EAGLE, for draft tokens
        the state are kept util we decide how many tokens are accepted for
        each sequence, and a shifting is done during the next iteration
        based on the number of accepted tokens.
        """
        if not self.speculative_config or not self.model_config.is_hybrid:
            return

        # Find the number of accepted tokens for each sequence.
        num_reqs = output_token_ids.size(0)
        self.num_accepted_tokens.gpu[:num_reqs] = (
            (
                torch.cat(
                    [
                        output_token_ids,
                        torch.full(
                            (num_reqs, 1),
                            -1,
                            device=output_token_ids.device,
                        ),
                    ],
                    dim=1,
                )
                == -1
            )
            .int()
            .argmax(-1)
        )
        spec_decode_active = bool(scheduler_output.scheduled_spec_decode_tokens)
        if self.needs_prefill_as_decode_slots and spec_decode_active:
            mamba_utils.update_accepted_tokens_for_prefill_as_decode(
                self.input_batch,
                self.prefill_as_decode_num_tokens,
                self.num_accepted_tokens.gpu,
                scheduler_output,
                self.reorder_batch_threshold,
                num_reqs,
            )

        if self.cache_config.mamba_cache_mode == "align":
            for i, num_tokens in enumerate(
                self.num_accepted_tokens.gpu[:num_reqs].cpu().numpy()
            ):
                self.input_batch.num_accepted_tokens_cpu[i] = num_tokens
            mamba_utils.postprocess_mamba(
                scheduler_output,
                self.kv_cache_config,
                self.input_batch,
                self.requests,
                self.mamba_state_idx,
                self.compilation_config.static_forward_context,
                self.model.get_mamba_state_copy_func(),
                self._get_mamba_copy_bufs(),
            )
        else:
            self.input_batch.num_accepted_tokens_cpu_tensor[:num_reqs].copy_(
                self.num_accepted_tokens.gpu[:num_reqs], non_blocking=True
            )
            assert self.num_accepted_tokens_event is not None
            self.num_accepted_tokens_event.record()

    def _update_streaming_request(
        self, req_id: str, new_req_data: NewRequestData
    ) -> CachedRequestState:
        """Updates streaming session request from `scheduled_new_reqs`.

        Removes the request from InputBatch (if present), updates the cached
        state, and prepares it for re-addition to the batch.

        NOTE: prompt_token_ids includes intermediate output tokens - tokens
        previously generated but now are input context (part of the prompt).
        """
        self.input_batch.remove_request(req_id)
        req_state = self.requests[req_id]

        req_state.prompt_token_ids = new_req_data.prompt_token_ids
        req_state.mm_features = new_req_data.mm_features
        req_state.prompt_embeds = new_req_data.prompt_embeds
        req_state.sampling_params = new_req_data.sampling_params
        req_state.pooling_params = new_req_data.pooling_params
        self.late_interaction_runner.register_request(req_id, req_state.pooling_params)
        req_state.block_ids = new_req_data.block_ids
        req_state.num_computed_tokens = new_req_data.num_computed_tokens
        req_state.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            req_state.prompt_token_ids, req_state.prompt_embeds
        )

        # Clear `output_token_ids` as previous output tokens are now part of
        # `prompt_token_ids`.
        req_state.output_token_ids.clear()

        if self.uses_mrope:
            self._init_mrope_positions(req_state)

        return req_state

    def _init_mrope_positions(self, req_state: CachedRequestState):
        model = self.get_model()
        assert supports_mrope(model), "M-RoPE support is not implemented."
        assert req_state.prompt_token_ids is not None, (
            "M-RoPE requires prompt_token_ids to be available."
        )
        mrope_model = cast(SupportsMRoPE, model)

        req_state.mrope_positions, req_state.mrope_position_delta = (
            mrope_model.get_mrope_input_positions(
                req_state.prompt_token_ids,
                req_state.mm_features,
            )
        )

    def _init_xdrope_positions(self, req_state: CachedRequestState):
        model = self.get_model()
        xdrope_model = cast(SupportsXDRoPE, model)
        assert req_state.prompt_token_ids is not None, (
            "XD-RoPE requires prompt_token_ids to be available."
        )
        assert supports_xdrope(model), "XD-RoPE support is not implemented."

        req_state.xdrope_positions = xdrope_model.get_xdrope_input_positions(
            req_state.prompt_token_ids,
            req_state.mm_features,
        )

    def _extract_mm_kwargs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> BatchedTensorInputs:
        if not scheduler_output or not self.is_multimodal_raw_input_only_model:
            return {}

        mm_kwargs = list[tuple[str, MultiModalKwargsItem]]()
        for req in scheduler_output.scheduled_new_reqs:
            for feature in req.mm_features:
                if feature.data is not None:
                    mm_kwargs.append((feature.modality, feature.data))

        # Input all modalities at once
        mm_kwargs_combined: BatchedTensorInputs = {}
        for _, _, mm_kwargs_batch in group_and_batch_mm_kwargs(
            mm_kwargs,
            device=self.device,
            pin_memory=self.pin_memory,
        ):
            mm_kwargs_combined.update(mm_kwargs_batch)

        return mm_kwargs_combined

    def _dummy_mm_kwargs(self, num_seqs: int) -> BatchedTensorInputs:
        if not self.is_multimodal_raw_input_only_model:
            return {}

        mm_budget = self.mm_budget
        assert mm_budget is not None

        if not mm_budget.mm_max_toks_per_item:
            return {}  # No tower modalities (embed-only mode)

        dummy_modality = mm_budget.get_modality_with_max_tokens()
        return self._get_mm_dummy_batch(dummy_modality, num_seqs)

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def _prepare_input_ids(
        self,
        scheduler_output: "SchedulerOutput",
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids."""

        if self.input_batch.prev_sampled_token_ids is None:
            # Normal scheduling case
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
            return

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the GPU from prev_sampled_token_ids.
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        assert prev_req_id_to_index is not None
        sample_flattened_indices: list[int] = []
        spec_flattened_indices: list[int] = []
        prev_common_req_indices: list[int] = []
        prev_draft_token_indices: list[int] = []
        indices_match = True
        max_flattened_index = -1
        total_num_spec_tokens = 0
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        for req_id, cur_index in self.input_batch.req_id_to_index.items():
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                # We need to compute the flattened input_ids index of the
                # last token in each common request.
                draft_len = len(scheduled_spec_tokens.get(req_id, ()))
                total_num_spec_tokens += draft_len
                flattened_index = cu_num_tokens[cur_index].item() - 1
                # example: cu_num_tokens = [2, 5, 8], draft_tokens = [1, 2, 2]
                # sample_flattened_indices = [0, 2, 5]
                # spec_flattened_indices = [1,   3, 4,    6, 7]
                sample_flattened_indices.append(flattened_index - draft_len)
                spec_flattened_indices.extend(
                    range(flattened_index - draft_len + 1, flattened_index + 1)
                )
                start = prev_index * self.num_spec_tokens
                # prev_draft_token_indices is used to find which draft_tokens_id
                # should be copied to input_ids
                # example: prev draft_tokens_id [[1,2], [3,4], [5, 6]]
                # flatten draft_tokens_id [1,2,3,4,5,6]
                # draft_len of each request [1, 2, 1]
                # then prev_draft_token_indices is [0,   2, 3,   4]
                prev_draft_token_indices.extend(range(start, start + draft_len))
                indices_match &= prev_index == flattened_index
                max_flattened_index = max(max_flattened_index, flattened_index)
                if (
                    (self._sparse_debug_first_token
                     or _SPARSE_HARD_DEBUG_FIRST_NEW_TOKEN)
                    and self.input_batch.prev_sampled_token_ids is not None
                ):
                    try:
                        prev_tok = int(
                            self.input_batch.prev_sampled_token_ids[
                                prev_index, 0
                            ].item()
                        )
                    except Exception:
                        prev_tok = -1
                    logger.info(
                        "[SparseStep:consume] req_id=%s cur_idx=%d prev_idx=%d "
                        "flattened_idx=%d draft_len=%d prev_tok=%d",
                        req_id,
                        int(cur_index),
                        int(prev_index),
                        int(flattened_index),
                        int(draft_len),
                        prev_tok,
                    )
        num_common_tokens = len(sample_flattened_indices)
        total_without_spec = total_num_scheduled_tokens - total_num_spec_tokens
        if num_common_tokens < total_without_spec:
            # If not all requests are decodes from the last iteration,
            # We need to copy the input_ids_cpu to the GPU first.
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
        if num_common_tokens == 0:
            # No requests in common with the previous iteration
            # So input_ids.cpu will have all the input ids.
            return
        if indices_match and max_flattened_index == (num_common_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            self.input_ids.gpu[:num_common_tokens].copy_(
                self.input_batch.prev_sampled_token_ids[:num_common_tokens, 0],
                non_blocking=True,
            )
            if self.enable_prompt_embeds:
                self.is_token_ids.gpu[:num_common_tokens] = True
            return
        # Upload the index tensors asynchronously so the scatter can be non-blocking.
        sampled_tokens_index_tensor = torch.tensor(
            sample_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_common_req_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0
            ],
        )

        # Scatter the draft tokens after the sampled tokens are scattered.
        if self._draft_token_ids is None or not spec_flattened_indices:
            return

        assert isinstance(self._draft_token_ids, torch.Tensor)
        draft_tokens_index_tensor = torch.tensor(
            spec_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_draft_token_indices_tensor = torch.tensor(
            prev_draft_token_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)

        # because input_ids dtype is torch.int32,
        # so convert draft_token_ids to torch.int32 here.
        draft_token_ids = self._draft_token_ids.to(dtype=torch.int32)

        self.input_ids.gpu.scatter_(
            dim=0,
            index=draft_tokens_index_tensor,
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],
        )

    def _get_encoder_seq_lens(
        self,
        num_scheduled_tokens: dict[str, int],
        kv_cache_spec: KVCacheSpec,
        num_reqs: int,
        for_cudagraph_capture: bool = False,
    ) -> tuple[torch.Tensor | None, np.ndarray | None]:
        if not isinstance(kv_cache_spec, CrossAttentionSpec):
            return None, None

        # Zero out buffer for padding requests that are not actually scheduled (CGs)
        self.encoder_seq_lens.np[:num_reqs] = 0

        # Build encoder_seq_lens array mapping request indices to
        # encoder lengths for inputs scheduled in this batch
        for req_id in num_scheduled_tokens:
            req_index = self.input_batch.req_id_to_index[req_id]
            req_state = self.requests[req_id]
            if req_state.mm_features is None:
                self.encoder_seq_lens.np[req_index] = 0
                continue

            # Get the total number of encoder input tokens for running encoder requests
            # whether encoding is finished or not so that cross-attention knows how
            # many encoder tokens to attend to.
            encoder_input_tokens = sum(
                feature.mm_position.length for feature in req_state.mm_features
            )
            self.encoder_seq_lens.np[req_index] = encoder_input_tokens
        if for_cudagraph_capture:
            # During CUDA graph capture, we need to use realistic encoder lengths
            # so that max_seqlen_k is captured with the correct value.
            max_encoder_len = getattr(
                self.model_config.hf_config,
                "max_source_positions",
                self.max_encoder_len,
            )
            self.encoder_seq_lens.np[:num_reqs] = max_encoder_len

        self.encoder_seq_lens.copy_to_gpu(num_reqs)
        encoder_seq_lens = self.encoder_seq_lens.gpu[:num_reqs]
        encoder_seq_lens_cpu = self.encoder_seq_lens.np[:num_reqs]

        return encoder_seq_lens, encoder_seq_lens_cpu

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        SpecDecodeMetadata | None,
    ]:
        """
        :return: tuple[
            logits_indices, spec_decode_metadata,
        ]
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)

        # Get positions.
        positions_np = self.positions.np[:total_num_scheduled_tokens]
        np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            arange,
            out=positions_np,
        )

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # Calculate XD-RoPE positions.
        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        if self.uses_xdrope_dim > 0:
            self._calc_xdrope_positions(scheduler_output)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (
            positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        )
        token_indices_tensor = torch.from_numpy(token_indices)

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids,
                0,
                token_indices_tensor,
                out=self.is_token_ids.cpu[:total_num_scheduled_tokens],
            )

        # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
        # the InputBatch, we need to fill in the prompt embeds into the expected
        # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
        if self.input_batch.req_prompt_embeds:
            output_idx = 0
            for req_idx in range(num_reqs):
                num_sched = num_scheduled_tokens[req_idx]

                # Skip if this request doesn't have embeddings
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue

                # Skip if no tokens scheduled
                if num_sched <= 0:
                    output_idx += num_sched
                    continue

                req_embeds = self.input_batch.req_prompt_embeds[req_idx]
                start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

                # Skip if trying to read beyond available embeddings
                if start_pos >= req_embeds.shape[0]:
                    output_idx += num_sched
                    continue

                # Copy available embeddings
                end_pos = start_pos + num_sched
                actual_end = min(end_pos, req_embeds.shape[0])
                actual_num_sched = actual_end - start_pos

                if actual_num_sched > 0:
                    self.inputs_embeds.cpu[
                        output_idx : output_idx + actual_num_sched
                    ].copy_(req_embeds[start_pos:actual_end])

                output_idx += num_sched

        self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)

        # NOTE: _override_sparse_slot_mapping is intentionally DISABLED.
        # compute_slot_mapping above uses the *original* (non-sparse)
        # block_table, which already maps each decode token to the correct
        # physical slot.  The sparse block-table rewrite
        # (_build_sparse_layer_block_table_tensor) only produces a separate
        # tensor for the attention kernel and never mutates
        # input_batch.block_table.  The old override mis-computed
        # fill_offset = num_output_tokens % block_size instead of the
        # correct (prompt_len + num_output_tokens - 1) % block_size,
        # corrupting newly-written KV cache entries and causing garbled
        # output.
        # if self._has_sparse_attn:
        #     self._override_sparse_slot_mapping(num_reqs, cu_num_tokens)

        self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)

        # Prepare the attention metadata.
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        # Note: pad query_start_loc to be non-decreasing, as kernels
        # like FlashAttention requires that
        self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()
        query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]

        self.seq_lens.np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
        )
        # Fill unused with 0 for full cuda graph mode.
        self.seq_lens.np[num_reqs:].fill(0)
        self.seq_lens.copy_to_gpu()

        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)

        # Record which requests should not be sampled,
        # so that we could clear the sampled tokens before returning
        self.discard_request_mask.np[:num_reqs] = (
            self.seq_lens.np[:num_reqs] < num_tokens_np
        )
        self.discard_request_mask.copy_to_gpu(num_reqs)

        # Copy the tensors to the GPU.
        self._prepare_input_ids(
            scheduler_output,
            total_num_scheduled_tokens,
            cu_num_tokens,
        )

        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        elif self.uses_xdrope_dim > 0:
            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        else:
            # Common case (1D positions)
            self.positions.copy_to_gpu(total_num_scheduled_tokens)

        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if self._has_sparse_attn and (
            self._sparse_probe_info_enabled or self._sparse_debug_decode_tokens
        ):
            pp_enabled = bool(get_pp_group().world_size > 1)
            logger.info(
                "[SparseRC] runtime_mode use_async_scheduling=%s use_pp=%s "
                "use_spec_decode=%s num_reqs=%d total_num_scheduled_tokens=%d",
                self.use_async_scheduling,
                pp_enabled,
                use_spec_decode,
                num_reqs,
                total_num_scheduled_tokens,
            )
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = query_start_loc[1:] - 1
            spec_decode_metadata = None
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
                if (
                    self.input_batch.num_computed_tokens_cpu[req_idx]
                    >= self.input_batch.num_prompt_tokens[req_idx]
                ):
                    num_decode_draft_tokens[req_idx] = len(draft_token_ids)
            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens
            )
            logits_indices = spec_decode_metadata.logits_indices
            num_sampled_tokens = num_draft_tokens + 1
            # For DECODE only cuda graph of some attention backends (e.g., GDN).
            self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()

        # Hot-Swap lora model
        if self.lora_config:
            assert (
                np.sum(num_sampled_tokens)
                <= self.vllm_config.scheduler_config.max_num_batched_tokens
            )
            self.set_active_loras(
                self.input_batch, num_scheduled_tokens, num_sampled_tokens
            )

        return (
            logits_indices,
            spec_decode_metadata,
        )

    def _build_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices: UBatchSlices | None = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
        slot_mappings: dict[int, torch.Tensor] | None = None,
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
        """
        :return: tuple[attn_metadata, spec_decode_common_attn_metadata]
        """
        # Attention metadata is not needed for attention free models
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return {}, None

        num_tokens_padded = num_tokens_padded or num_tokens
        num_reqs_padded = num_reqs_padded or num_reqs
        assert num_reqs_padded is not None and num_tokens_padded is not None

        attn_metadata: PerLayerAttnMetadata = {}
        if ubatch_slices is not None:
            attn_metadata = [dict() for _ in range(len(ubatch_slices))]

        if for_cudagraph_capture:
            # For some attention backends (e.g. FA) with sliding window models we need
            # to make sure the backend see a max_seq_len that is larger to the sliding
            # window size when capturing to make sure the correct kernel is selected.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = self.seq_lens.np[:num_reqs].max().item()

        if use_spec_decode:
            if self.num_accepted_tokens_event is not None:
                self.num_accepted_tokens_event.synchronize()
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs]
            )
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

        kv_cache_groups = self.kv_cache_config.kv_cache_groups

        def _get_block_table(kv_cache_gid: int):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                blk_table_tensor = torch.zeros(
                    (num_reqs_padded, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_table[kv_cache_gid]
                blk_table_tensor = blk_table.get_device_tensor(num_reqs_padded)

            # Fill unused with -1. Needed for reshape_and_cache in full cuda
            # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
            blk_table_tensor[num_reqs:num_reqs_padded].fill_(-1)
            return blk_table_tensor

        assert slot_mappings is not None
        block_table_gid_0 = _get_block_table(0)
        slot_mapping_gid_0 = slot_mappings[0]

        if self.routed_experts_initialized:
            attn_gid = self.routed_experts_attn_gid
            slot_mapping_attn = slot_mappings[attn_gid]
            self.slot_mapping = slot_mapping_attn[:num_tokens].cpu().numpy()
        cm_base = CommonAttentionMetadata(
            query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
            seq_lens=self.seq_lens.gpu[:num_reqs_padded],
            _seq_lens_cpu=self.seq_lens.cpu[:num_reqs_padded],
            _num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[
                :num_reqs_padded
            ],
            num_reqs=num_reqs_padded,
            num_actual_tokens=num_tokens_padded,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=block_table_gid_0,
            slot_mapping=slot_mapping_gid_0,
            causal=True,
        )

        if self.dcp_world_size > 1:
            self.dcp_local_seq_lens.cpu[:num_reqs] = get_dcp_local_seq_lens(
                self.seq_lens.cpu[:num_reqs],
                self.dcp_world_size,
                self.dcp_rank,
                self.parallel_config.cp_kv_cache_interleave_size,
            )
            self.dcp_local_seq_lens.cpu[num_reqs:].fill_(0)
            self.dcp_local_seq_lens.copy_to_gpu(num_reqs_padded)

            cm_base.dcp_local_seq_lens = self.dcp_local_seq_lens.gpu[:num_reqs_padded]
            cm_base.dcp_local_seq_lens_cpu = self.dcp_local_seq_lens.cpu[
                :num_reqs_padded
            ]

        if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
            cm_base.num_logits_indices = logits_indices.size(0)
            cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
                logits_indices
            )

        # Cache attention metadata builds across hybrid KV-cache groups
        # The only thing that changes between different hybrid KV-cache groups when the
        # same metadata builder and KVCacheSpec is the same is the block table, so we
        # can cache the attention metadata builds and just update the block table using
        # `builder.update_block_table` if the builder supports it.
        cached_attn_metadata: dict[
            tuple[KVCacheSpec, type[AttentionMetadataBuilder]], AttentionMetadata
        ] = {}
        perf_sparse_enabled = (
            self._sparse_perf_stats_enabled and self._has_sparse_attn
        )

        def _build_attn_group_metadata(
            kv_cache_gid: int,
            attn_gid: int,
            common_attn_metadata: CommonAttentionMetadata,
            ubid: int | None = None,
            *,
            skip_block_table_cache: bool = False,
            layer_names_override: list[str] | None = None,
            sparse_per_head_block_table: torch.Tensor | None = None,
            sparse_per_head_seq_lens: torch.Tensor | None = None,
            disable_token_sparse_boundary: bool = False,
        ) -> None:
            attn_group = self.attn_groups[kv_cache_gid][attn_gid]
            builder = attn_group.get_metadata_builder(ubid or 0)
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = kv_cache_spec.kv_cache_specs[attn_group.layer_names[0]]
            cache_key = (kv_cache_spec, type(builder))

            cascade_attn_prefix_len = (
                cascade_attn_prefix_lens[kv_cache_gid][attn_gid]
                if cascade_attn_prefix_lens
                else 0
            )

            if isinstance(builder, Mamba2AttentionMetadataBuilder):
                self.needs_prefill_as_decode_slots = True
            extra_attn_metadata_args = {}
            if use_spec_decode and isinstance(
                builder, (Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder)
            ):
                assert ubid is None, "UBatching not supported with GDN yet"
                extra_attn_metadata_args = dict(
                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                    num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
                        :num_reqs_padded
                    ],
                )

            _t_builder = time.perf_counter() if perf_sparse_enabled else None
            if for_cudagraph_capture:
                attn_metadata_i = builder.build_for_cudagraph_capture(
                    common_attn_metadata
                )
            elif (
                not skip_block_table_cache
                and cache_key in cached_attn_metadata
                and builder.supports_update_block_table
            ):
                attn_metadata_i = builder.update_block_table(
                    cached_attn_metadata[cache_key],
                    common_attn_metadata.block_table_tensor,
                    common_attn_metadata.slot_mapping,
                )
            else:
                attn_metadata_i = builder.build(
                    common_prefix_len=cascade_attn_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                    **extra_attn_metadata_args,
                )
                if builder.supports_update_block_table and not skip_block_table_cache:
                    cached_attn_metadata[cache_key] = attn_metadata_i
            if _t_builder is not None:
                self._sparse_perf_record(
                    "_build_attention_metadata:builder_build_or_update",
                    time.perf_counter() - _t_builder,
                )

            if (
                sparse_per_head_block_table is not None
                and sparse_per_head_seq_lens is not None
                and isinstance(attn_metadata_i, FlashAttentionMetadata)
            ):
                attn_metadata_i = replace(
                    attn_metadata_i,
                    sparse_per_head_block_table=sparse_per_head_block_table,
                    sparse_per_head_seq_lens=sparse_per_head_seq_lens,
                )

            if ubid is None:
                assert isinstance(attn_metadata, dict)
                attn_metadata_dict = attn_metadata
            else:
                assert isinstance(attn_metadata, list)
                attn_metadata_dict = attn_metadata[ubid]

            layers_fill = layer_names_override or attn_group.layer_names
            for layer_name in layers_fill:
                attn_metadata_dict[layer_name] = attn_metadata_i

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        spec_decode_common_attn_metadata = None
        for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):
            cm = copy(cm_base)  # shallow copy

            # Basically only the encoder seq_lens, block_table and slot_mapping change
            # for each kv_cache_group.
            _t_enc = time.perf_counter() if perf_sparse_enabled else None
            cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = self._get_encoder_seq_lens(
                num_scheduled_tokens or {},
                kv_cache_group.kv_cache_spec,
                num_reqs_padded,
                for_cudagraph_capture=for_cudagraph_capture,
            )
            if kv_cache_gid > 0:
                cm.block_table_tensor = _get_block_table(kv_cache_gid)
                cm.slot_mapping = slot_mappings[kv_cache_gid]
            if _t_enc is not None:
                self._sparse_perf_record(
                    "_build_attention_metadata:get_encoder_seq_lens",
                    time.perf_counter() - _t_enc,
                )

            is_sparse_group = isinstance(
                kv_cache_group.kv_cache_spec, SparseAttentionSpec
            )
            boundary_disable_sparse = False
            boundary_req_ids: list[str] = []
            if is_sparse_group:
                _t_boundary = time.perf_counter() if perf_sparse_enabled else None
                for req_idx in range(num_reqs):
                    rid = self.input_batch.req_ids[req_idx]
                    seq_i = int(cm._seq_lens_cpu[req_idx].item())
                    p_i = int(self.input_batch.num_prompt_tokens[req_idx])
                    rs = self.requests.get(rid)
                    out_n = 0 if rs is None else len(rs.output_token_ids)
                    # Async placeholder boundary: first decode forward can have
                    # seq_len == prompt_len + 1 while no output token is yet
                    # committed on worker side. Treat this as boundary as well
                    # so token-sparse patching is disabled for that transition
                    # step and cannot affect the first sampled token.
                    if out_n == 0 and seq_i <= (p_i + 1):
                        boundary_req_ids.append(rid)
                boundary_disable_sparse = len(boundary_req_ids) > 0
                if boundary_disable_sparse and (
                    self._sparse_probe_info_enabled
                    or self._sparse_debug_decode_tokens
                    or _SPARSE_HARD_DEBUG_FIRST_NEW_TOKEN
                ):
                    logger.info(
                        "[SparseRC] boundary_disable_sparse gid=%d reqs=%d "
                        "req_id_head=%s",
                        kv_cache_gid,
                        len(boundary_req_ids),
                        boundary_req_ids[:2],
                    )
                if _t_boundary is not None:
                    self._sparse_perf_record(
                        "_build_attention_metadata:sparse_boundary_scan",
                        time.perf_counter() - _t_boundary,
                    )
            # Token-level compact gather uses sparse_q_head_gather metadata
            # (phys, slot, cu_seqlens_k) to index KV directly with
            # block_table=None.  The underlying block_table and seq_lens
            # must therefore stay *unmodified* so that when compact gather
            # is unavailable (e.g. first decode step) FA falls back to
            # standard paged attention with the full, correct metadata.
            # Skipping _override_sparse_seq_lens here also eliminates the
            # async-scheduling state drift that caused repeated-token output.
            _group_is_tok_compact = (
                is_sparse_group
                and isinstance(kv_cache_group.kv_cache_spec, SparseAttentionSpec)
                and kv_cache_group.kv_cache_spec.cluster_granularity == "token"
                and kv_cache_group.kv_cache_spec.use_compact_kv_gather
            )
            if is_sparse_group and not boundary_disable_sparse and not _group_is_tok_compact:
                _t0_sparse_seq = (
                    time.perf_counter() if self._sparse_perf_stats_enabled else None
                )
                cm_union = self._override_sparse_seq_lens(
                    cm,
                    kv_cache_gid,
                    num_reqs,
                    num_reqs_padded,
                    kv_cache_group.kv_cache_spec.block_size,
                )
                if _t0_sparse_seq is not None:
                    self._sparse_perf_record(
                        "_override_sparse_seq_lens",
                        time.perf_counter() - _t0_sparse_seq,
                    )
            else:
                cm_union = cm

            if self.speculative_config and spec_decode_common_attn_metadata is None:
                if isinstance(self.drafter, EagleProposer):
                    if self.drafter.kv_cache_gid == kv_cache_gid:
                        spec_decode_common_attn_metadata = cm_union
                else:
                    spec_decode_common_attn_metadata = cm_union

            for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
                attn_group_i = self.attn_groups[kv_cache_gid][attn_gid]
                if is_sparse_group:
                    spec_sp = kv_cache_group.kv_cache_spec
                    assert isinstance(spec_sp, SparseAttentionSpec)
                    is_tok_compact = (
                        spec_sp.cluster_granularity == "token"
                        and spec_sp.use_compact_kv_gather
                    )
                    for layer_name in attn_group_i.layer_names:
                        attn_mod_sh = (
                            self.compilation_config.static_forward_context.get(
                                layer_name
                            )
                        )
                        n_heads_sh = (
                            int(attn_mod_sh.num_heads)
                            if attn_mod_sh is not None
                            else 1
                        )
                        bsz = spec_sp.block_size
                        sp_bt: torch.Tensor | None = None
                        sp_sl: torch.Tensor | None = None
                        decode_override: np.ndarray | None = None
                        if boundary_disable_sparse:
                            cm_layer = copy(cm_union)
                            cm_layer.block_table_tensor = (
                                cm_union.block_table_tensor.clone()
                            )
                        elif is_tok_compact:
                            # Token-level compact gather uses
                            # sparse_q_head_gather metadata (phys, slot,
                            # cu_seqlens_k) to index KV directly – it
                            # passes block_table=None to FA.  Therefore
                            # the block_table and seq_lens in
                            # CommonAttentionMetadata must remain
                            # *unmodified* so that when compact gather is
                            # unavailable (e.g. first decode step) FA
                            # falls back to standard paged attention with
                            # the full, correct block_table.
                            cm_layer = copy(cm_union)
                            cm_layer.block_table_tensor = (
                                cm_union.block_table_tensor.clone()
                            )
                        else:
                            _t_per_head = (
                                time.perf_counter() if perf_sparse_enabled else None
                            )
                            per_bt, per_sl = (
                                self._build_sparse_per_head_block_table_and_lens(
                                    cm_union.block_table_tensor,
                                    cm_union,
                                    kv_cache_gid,
                                    layer_name,
                                    n_heads_sh,
                                    num_reqs,
                                    num_reqs_padded,
                                    bsz,
                                )
                            )
                            if _t_per_head is not None:
                                self._sparse_perf_record(
                                    "_build_attention_metadata:per_head_block_table",
                                    time.perf_counter() - _t_per_head,
                                )
                            if per_bt is not None and per_sl is not None:
                                cm_layer = copy(cm)
                                cm_layer.block_table_tensor = per_bt[0]
                                cm_layer.seq_lens = per_sl[0]
                                cm_layer._seq_lens_cpu = torch.tensor(
                                    per_sl[0].cpu().numpy(), dtype=torch.int32
                                )
                                if num_reqs > 0:
                                    cm_layer.max_seq_len = int(
                                        per_sl[0, :num_reqs].max().item()
                                    )
                                sp_bt, sp_sl = per_bt, per_sl
                            else:
                                cm_layer = copy(cm)
                                cm_layer.block_table_tensor = (
                                    cm_union.block_table_tensor.clone()
                                )
                                _t0_sparse_seq = (
                                    time.perf_counter()
                                    if self._sparse_perf_stats_enabled
                                    else None
                                )
                                cm_layer = self._override_sparse_seq_lens(
                                    cm_layer,
                                    kv_cache_gid,
                                    num_reqs,
                                    num_reqs_padded,
                                    bsz,
                                    decode_num_blocks_override=None,
                                )
                                if _t0_sparse_seq is not None:
                                    self._sparse_perf_record(
                                        "_override_sparse_seq_lens",
                                        time.perf_counter() - _t0_sparse_seq,
                                    )
                        if (
                            (self._sparse_probe_info_enabled
                             or self._sparse_debug_decode_tokens)
                            and num_reqs > 0
                        ):
                            req0 = self.input_batch.req_ids[0]
                            blk0 = int(
                                self.input_batch.block_table[
                                    kv_cache_gid
                                ].num_blocks_per_row[0]
                            )
                            seq0 = int(cm_layer._seq_lens_cpu[0].item())
                            ov0 = (
                                -1
                                if decode_override is None
                                else int(decode_override[0])
                            )
                            logger.info(
                                "[SparseProbe] seq_lens_bridge req_id=%s "
                                "layer=%s is_tok_compact=%s "
                                "decode_num_blocks_override=%d "
                                "num_blocks_per_row=%d seq_len_cpu=%d block_size=%d",
                                req0,
                                layer_name,
                                is_tok_compact,
                                ov0,
                                blk0,
                                seq0,
                                int(bsz),
                            )
                            logger.info(
                                "[SparseRC] layer_bridge req_id=%s layer=%s gid=%d "
                                "union_seq_len=%d layer_seq_len=%d "
                                "union_blocks=%d layer_override_blocks=%d",
                                req0,
                                layer_name,
                                kv_cache_gid,
                                int(cm_union._seq_lens_cpu[0].item()),
                                seq0,
                                blk0,
                                ov0,
                            )
                        if ubatch_slices is not None:
                            for ubid, _cm in enumerate(
                                split_attn_metadata(ubatch_slices, cm_layer)
                            ):
                                _build_attn_group_metadata(
                                    kv_cache_gid,
                                    attn_gid,
                                    _cm,
                                    ubid,
                                    skip_block_table_cache=True,
                                    layer_names_override=[layer_name],
                                    sparse_per_head_block_table=sp_bt,
                                    sparse_per_head_seq_lens=sp_sl,
                                    disable_token_sparse_boundary=boundary_disable_sparse,
                                )
                        else:
                            _build_attn_group_metadata(
                                kv_cache_gid,
                                attn_gid,
                                cm_layer,
                                None,
                                skip_block_table_cache=True,
                                layer_names_override=[layer_name],
                                sparse_per_head_block_table=sp_bt,
                                sparse_per_head_seq_lens=sp_sl,
                                disable_token_sparse_boundary=boundary_disable_sparse,
                            )
                elif ubatch_slices is not None:
                    for ubid, _cm in enumerate(
                        split_attn_metadata(ubatch_slices, cm_union)
                    ):
                        _build_attn_group_metadata(
                            kv_cache_gid,
                            attn_gid,
                            _cm,
                            ubid,
                            disable_token_sparse_boundary=boundary_disable_sparse,
                        )

                else:
                    _build_attn_group_metadata(
                        kv_cache_gid,
                        attn_gid,
                        cm_union,
                        disable_token_sparse_boundary=boundary_disable_sparse,
                    )

        if self.is_mm_prefix_lm:
            req_doc_ranges = {}
            for req_id in self.input_batch.req_ids:
                image_doc_ranges = []
                req_state = self.requests[req_id]
                for mm_feature in req_state.mm_features:
                    pos_info = mm_feature.mm_position
                    img_doc_range = pos_info.extract_embeds_range()
                    image_doc_ranges.extend(img_doc_range)
                req_idx = self.input_batch.req_id_to_index[req_id]
                req_doc_ranges[req_idx] = image_doc_ranges

            if isinstance(attn_metadata, list):
                for ub_metadata in attn_metadata:
                    for _metadata in ub_metadata.values():
                        _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]
            else:
                for _metadata in attn_metadata.values():
                    _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]

        if spec_decode_common_attn_metadata is not None and (
            num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
        ):
            # Currently the drafter still only uses piecewise cudagraphs (and modifies
            # the attention metadata in directly), and therefore does not want to use
            # padded attention metadata.
            spec_decode_common_attn_metadata = (
                spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)
            )

        return attn_metadata, spec_decode_common_attn_metadata

    def _compute_cascade_attn_prefix_lens(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        num_common_prefix_blocks: list[int],
    ) -> list[list[int]] | None:
        """
        :return: Optional[cascade_attn_prefix_lens]
            cascade_attn_prefix_lens is 2D: ``[kv_cache_group_id][attn_group_idx]``,
            None if we should not use cascade attention
        """

        use_cascade_attn = False
        num_kv_cache_groups = len(self.kv_cache_config.kv_cache_groups)
        cascade_attn_prefix_lens: list[list[int]] = [
            [] for _ in range(num_kv_cache_groups)
        ]

        for kv_cache_gid in range(num_kv_cache_groups):
            for attn_group in self.attn_groups[kv_cache_gid]:
                if isinstance(attn_group.kv_cache_spec, EncoderOnlyAttentionSpec):
                    cascade_attn_prefix_len = 0
                else:
                    # 0 if cascade attention should not be used
                    cascade_attn_prefix_len = self._compute_cascade_attn_prefix_len(
                        num_scheduled_tokens,
                        num_computed_tokens,
                        num_common_prefix_blocks[kv_cache_gid],
                        attn_group.kv_cache_spec,
                        attn_group.get_metadata_builder(),
                    )
                cascade_attn_prefix_lens[kv_cache_gid].append(cascade_attn_prefix_len)
                use_cascade_attn |= cascade_attn_prefix_len > 0

        return cascade_attn_prefix_lens if use_cascade_attn else None

    def _compute_cascade_attn_prefix_len(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        num_common_prefix_blocks: int,
        kv_cache_spec: KVCacheSpec,
        attn_metadata_builder: AttentionMetadataBuilder,
    ) -> int:
        """Compute the length of the common prefix for cascade attention.

        NOTE(woosuk): The common prefix length returned by this function
        represents the length used specifically for cascade attention, not the
        actual number of tokens shared between requests. When cascade attention
        is disabled (use_cascade=False), this function returns 0 even if
        requests share common tokens. Additionally, the common prefix length is
        truncated to a multiple of the block size and may be further truncated
        due to implementation details explained below.

        Args:
            num_scheduled_tokens: Number of tokens scheduled per request.
            num_common_prefix_blocks: Number of shared KV cache blocks.

        Returns:
            int: Length of common prefix in tokens.
        """

        common_prefix_len = num_common_prefix_blocks * kv_cache_spec.block_size
        if common_prefix_len == 0:
            # Common case.
            return 0

        # NOTE(woosuk): Cascade attention uses two attention kernels: one
        # for the common prefix and the other for the rest. For the first
        # kernel, we concatenate all the query tokens (possibly from
        # different requests) and treat them as if they are from the same
        # request. Then, we use bi-directional attention to process the
        # common prefix in the KV cache. Importantly, this means that the
        # first kernel does not do any masking.

        # Consider the following example:
        # Request 1's input query: [D, E, X]
        # Request 1's kv cache: [A, B, C, D, E, X]
        # Request 1's num_computed_tokens: 3 (i.e., [A, B, C])
        # Request 2's input query: [E, Y]
        # Request 2's kv cache: [A, B, C, D, E, Y]
        # Request 2's num_computed_tokens: 4 (i.e., [A, B, C, D])

        # If we use [A, B, C, D, E] as the common prefix, then the
        # first kernel will compute the bi-directional attention between
        # input query [D, E, X, E, Y] and common prefix [A, B, C, D, E].
        # However, this is wrong because D in Request 1 should not attend to
        # E in the common prefix (i.e., we need masking).
        # To avoid this, [A, B, C, D] should be the common prefix.
        # That is, the common prefix should be capped by the minimum
        # num_computed_tokens among the requests, and plus one to include
        # the first token of the query.

        # In practice, we use [A, B, C] as the common prefix, instead of
        # [A, B, C, D] (i.e., the common prefix is capped by the minimum
        # num_computed_tokens, without plus one).
        # This is because of an implementation detail: We want to always
        # use two kernels for cascade attention. Let's imagine:
        # Request 3's input query: [D]
        # Request 3's kv cache: [A, B, C, D]
        # Request 3's num_computed_tokens: 3 (i.e., [A, B, C])
        # If we use [A, B, C, D] as the common prefix for Request 1-3,
        # then Request 3 will be processed only by the first kernel,
        # and the second kernel will get an empty input. While this is not
        # a fundamental problem, our current implementation does not support
        # this case.
        common_prefix_len = min(common_prefix_len, num_computed_tokens.min())
        # common_prefix_len should be a multiple of the block size.
        common_prefix_len = (
            common_prefix_len // kv_cache_spec.block_size * kv_cache_spec.block_size
        )
        use_sliding_window = isinstance(kv_cache_spec, SlidingWindowSpec) or (
            isinstance(kv_cache_spec, FullAttentionSpec)
            and kv_cache_spec.sliding_window is not None
        )
        use_local_attention = isinstance(kv_cache_spec, ChunkedLocalAttentionSpec) or (
            isinstance(kv_cache_spec, FullAttentionSpec)
            and kv_cache_spec.attention_chunk_size is not None
        )
        assert isinstance(kv_cache_spec, AttentionSpec)
        use_cascade = attn_metadata_builder.use_cascade_attention(
            common_prefix_len=common_prefix_len,
            query_lens=num_scheduled_tokens,
            num_query_heads=self.num_query_heads,
            num_kv_heads=kv_cache_spec.num_kv_heads,
            use_alibi=self.use_alibi,
            use_sliding_window=use_sliding_window,
            use_local_attention=use_local_attention,
            num_sms=self.num_sms,
            dcp_world_size=self.dcp_world_size,
        )
        return common_prefix_len if use_cascade else 0

    def _calc_mrope_positions(self, scheduler_output: "SchedulerOutput"):
        mrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.mrope_positions is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
                req.prompt_token_ids, req.prompt_embeds
            )

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's mrope_positions are pre-computed
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.mrope_positions.cpu[:, dst_start:dst_end] = req.mrope_positions[
                    :, src_start:src_end
                ]
                mrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's mrope_positions on-the-fly
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + completion_part_len

                assert req.mrope_position_delta is not None
                MRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.mrope_positions.np,
                    out_offset=dst_start,
                    mrope_position_delta=req.mrope_position_delta,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                mrope_pos_ptr += completion_part_len

    def _calc_xdrope_positions(self, scheduler_output: "SchedulerOutput"):
        xdrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.xdrope_positions is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
                req.prompt_token_ids, req.prompt_embeds
            )

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's xdrope_positions are pre-computed
                dst_start = xdrope_pos_ptr
                dst_end = xdrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.xdrope_positions.cpu[:, dst_start:dst_end] = req.xdrope_positions[
                    :, src_start:src_end
                ]
                xdrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's xdrope_positions on-the-fly
                dst_start = xdrope_pos_ptr
                dst_end = xdrope_pos_ptr + completion_part_len

                XDRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.xdrope_positions.np,
                    out_offset=dst_start,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                xdrope_pos_ptr += completion_part_len

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1

        # Step 1. cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # arange: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(
            num_sampled_tokens, cumsum_dtype=np.int32
        )
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
        )
        # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        # arange: [0, 1, 2, 0, 1, 0]
        cu_num_draft_tokens, arange = self._get_cumsum_and_arange(
            num_draft_tokens, cumsum_dtype=np.int32
        )
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
        )
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> GPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True
        )
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens).to(
            self.device, non_blocking=True
        )
        logits_indices = torch.from_numpy(logits_indices).to(
            self.device, non_blocking=True
        )
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True
        )
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True
        )

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

    def _prepare_kv_sharing_fast_prefill(
        self,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor:
        assert self.kv_sharing_fast_prefill_logits_indices is not None
        num_logits = logits_indices.shape[0]
        assert num_logits > 0
        self.kv_sharing_fast_prefill_logits_indices[:num_logits].copy_(logits_indices)
        # There might have leftover indices in logits_indices[num_logits:]
        # from previous iterations, whose values may be greater than the
        # batch size in the current iteration. To ensure indices are always
        # valid, we fill the padded indices with the last index.
        self.kv_sharing_fast_prefill_logits_indices[num_logits:].fill_(
            logits_indices[-1].item()
        )
        # Dispatch for the decoder portion of the model.
        _, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_logits, invalid_modes={CUDAGraphMode.FULL}
        )
        num_logits_padded = batch_desc.num_tokens
        logits_indices_padded = self.kv_sharing_fast_prefill_logits_indices[
            :num_logits_padded
        ]
        return logits_indices_padded

    def _batch_mm_inputs_from_scheduler(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[
        list[str],
        list[tuple[str, MultiModalKwargsItem]],
        list[tuple[str, PlaceholderRange]],
    ]:
        """Batch multimodal inputs from scheduled encoder inputs.

        Args:
            scheduler_output: The scheduler output containing scheduled encoder
                inputs.

        Returns:
            A tuple of (mm_hashes, mm_kwargs, mm_lora_refs) where:
            - mm_hashes: List of multimodal hashes for each item
            - mm_kwargs: List of multimodal kwargs for each item
            - mm_lora_refs: List of (req_id, placeholder_range) for each item
        """
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return [], [], []

        mm_hashes = list[str]()
        mm_kwargs = list[tuple[str, MultiModalKwargsItem]]()
        # Multimodal LoRA reference info to map each multimodal item
        # back to its request & position
        mm_lora_refs = list[tuple[str, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_feature = req_state.mm_features[mm_input_id]
                if mm_feature.data is None:
                    continue

                mm_hashes.append(mm_feature.identifier)
                mm_kwargs.append((mm_feature.modality, mm_feature.data))
                mm_lora_refs.append((req_id, mm_feature.mm_position))

        return mm_hashes, mm_kwargs, mm_lora_refs

    def _execute_mm_encoder(
        self, scheduler_output: "SchedulerOutput"
    ) -> list[torch.Tensor]:
        mm_hashes, mm_kwargs, mm_lora_refs = self._batch_mm_inputs_from_scheduler(
            scheduler_output
        )

        if not mm_kwargs:
            return []

        should_time = bool(
            self.observability_config
            and self.observability_config.enable_mm_processor_stats
            and scheduler_output.scheduled_encoder_inputs
        )

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        model = cast(SupportsMultiModal, self.model)

        if self.lora_config and self.lora_manager.supports_tower_connector_lora():
            # Build LoRA mappings independently for encoder inputs
            # (encoder batch structure is different from main batch)
            prompt_lora_mapping = []
            token_lora_mapping = []
            lora_requests = set()
            encoder_token_counts = []

            for req_id, pos_info in mm_lora_refs:
                req_idx = self.input_batch.req_id_to_index[req_id]
                lora_id = int(self.input_batch.request_lora_mapping[req_idx])

                # Prefer pos_info.get_num_embeds to count precise MM embedding tokens.
                num_tokens = self.model.get_num_mm_encoder_tokens(  # type: ignore[attr-defined]
                    pos_info.get_num_embeds()
                )
                prompt_lora_mapping.append(lora_id)
                token_lora_mapping.extend([lora_id] * num_tokens)
                encoder_token_counts.append(num_tokens)

                if lora_id > 0:
                    lora_request = self.input_batch.lora_id_to_lora_request.get(lora_id)
                    if lora_request is not None:
                        lora_requests.add(lora_request)

            # Set tower adapter mapping
            tower_mapping = LoRAMapping(
                tuple(token_lora_mapping),
                tuple(prompt_lora_mapping),
                is_prefill=True,
                type=LoRAMappingType.TOWER,
            )
            self.lora_manager.set_active_adapters(lora_requests, tower_mapping)

            if hasattr(self.model, "get_num_mm_connector_tokens"):
                post_op_counts = [
                    self.model.get_num_mm_connector_tokens(num_tokens)  # type: ignore[attr-defined]
                    for num_tokens in encoder_token_counts
                ]

                connector_token_mapping = np.repeat(
                    np.array(prompt_lora_mapping, dtype=np.int32),
                    np.array(post_op_counts, dtype=np.int32),
                )
                connector_mapping = LoRAMapping(
                    index_mapping=tuple(connector_token_mapping.tolist()),
                    prompt_mapping=tuple(prompt_lora_mapping),
                    is_prefill=True,
                    type=LoRAMappingType.CONNECTOR,
                )

                self.lora_manager.set_active_adapters(
                    lora_requests,
                    connector_mapping,
                )

        encoder_outputs: list[torch.Tensor] = []
        # Track the current index in mm_kwargs/mm_lora_refs to map groups to request IDs
        current_item_idx = 0
        for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
            mm_kwargs,
            device=self.device,
            pin_memory=self.pin_memory,
        ):
            batch_outputs: MultiModalEmbeddings

            # EVS and dynamic res video related change.
            # (ekhvedchenia): Temporary hack to limit peak memory usage when
            # processing multimodal data. This solves the issue with scheduler
            # putting too many video samples into a single batch. Scheduler
            # uses pruned vision tokens count to compare it versus compute
            # budget which is incorrect (Either input media size or non-pruned
            # output vision tokens count should be considered)
            # dynamic res video for nemotron temporarily uses this hack via
            # requires_sequential_video_encoding
            # because it doesn't yet support video batching.
            # TODO(ywang96): Fix memory profiling to take EVS into account and
            # remove this hack.
            if (
                (
                    self.is_multimodal_pruning_enabled
                    or self.requires_sequential_video_encoding
                )
                and modality == "video"
                and num_items > 1
            ):
                batch_outputs_lst = list[torch.Tensor]()
                for video_idx in range(num_items):
                    video_mm_kwargs_item = mm_kwargs[current_item_idx + video_idx]
                    with self.timed_encoder_operation(
                        should_time, mm_lora_refs, current_item_idx + video_idx, 1
                    ):
                        _, _, micro_batch_mm_inputs = next(
                            group_and_batch_mm_kwargs(
                                [video_mm_kwargs_item],
                                device=self.device,
                                pin_memory=self.pin_memory,
                            )
                        )

                        micro_batch_outputs = model.embed_multimodal(
                            **micro_batch_mm_inputs
                        )

                        batch_outputs_lst.extend(micro_batch_outputs)

                batch_outputs = batch_outputs_lst
            else:
                # Run the encoder.
                # `batch_outputs` is either of the following:
                # 1. A tensor of shape (num_items, feature_size, hidden_size)
                # in case feature_size is fixed across all multimodal items.
                # 2. A list or tuple (length: num_items) of tensors,
                # each of shape (feature_size, hidden_size) in case the feature
                # size is dynamic depending on the input multimodal items.

                with self.timed_encoder_operation(
                    should_time, mm_lora_refs, current_item_idx, num_items
                ):
                    batch_outputs = model.embed_multimodal(**mm_kwargs_batch)

            sanity_check_mm_encoder_outputs(batch_outputs, expected_num_items=num_items)
            encoder_outputs.extend(batch_outputs)

            current_item_idx += num_items

        # Cache the encoder outputs by mm_hash
        for mm_hash, output in zip(mm_hashes, encoder_outputs):
            self.encoder_cache[mm_hash] = output
            logger.debug("Finish execute for mm hash %s", mm_hash)
            self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)

        return encoder_outputs

    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
        shift_computed_tokens: int = 0,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        # Swap to the other buffer to avoid race condition with previous
        # iteration's async copy that may still be reading from CPU.
        self.is_mm_embed_idx = 1 - self.is_mm_embed_idx
        is_mm_embed_buf = self.is_mm_embed_buffers[self.is_mm_embed_idx]

        mm_embeds = list[torch.Tensor]()
        is_mm_embed = is_mm_embed_buf.cpu
        is_mm_embed[:total_num_scheduled_tokens] = False

        req_start_idx = 0
        should_sync_mrope_positions = False
        should_sync_xdrope_positions = False

        for req_id in self.input_batch.req_ids:
            mm_embeds_req: list[torch.Tensor] = []

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens + shift_computed_tokens

            for mm_feature in req_state.mm_features:
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                # The encoder output is needed if the two ranges overlap:
                # [num_computed_tokens,
                #  num_computed_tokens + num_scheduled_tokens) and
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens,
                )
                assert start_idx < end_idx
                curr_embeds_start, curr_embeds_end = (
                    pos_info.get_embeds_indices_in_range(start_idx, end_idx)
                )
                # If there are no embeddings in the current range, we skip
                # gathering the embeddings.
                if curr_embeds_start == curr_embeds_end:
                    continue

                mm_hash = mm_feature.identifier
                encoder_output = self.encoder_cache.get(mm_hash, None)
                assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]
                    mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
                else:
                    mm_embeds_item = encoder_output[start_idx:end_idx]

                req_start_pos = req_start_idx + start_pos - num_computed_tokens
                # OR mask for overlapping mm_features (use_audio_in_video)
                if is_embed is None:
                    is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = (
                        True
                    )
                else:
                    is_mm_embed[
                        req_start_pos + start_idx : req_start_pos + end_idx
                    ] |= is_embed
                mm_embeds_req.append(mm_embeds_item)

            if self.is_multimodal_pruning_enabled and self.uses_mrope:
                assert req_state.mrope_positions is not None
                should_sync_mrope_positions = True
                mm_embeds_req, new_mrope_positions, new_delta = (
                    self.model.recompute_mrope_positions(
                        input_ids=req_state.prompt_token_ids,
                        multimodal_embeddings=mm_embeds_req,
                        mrope_positions=req_state.mrope_positions,
                        num_computed_tokens=req_state.num_computed_tokens,
                    )
                )
                req_state.mrope_positions.copy_(new_mrope_positions)
                req_state.mrope_position_delta = new_delta

            mm_embeds.extend(mm_embeds_req)
            req_start_idx += num_scheduled_tokens

        is_mm_embed = is_mm_embed_buf.copy_to_gpu(total_num_scheduled_tokens)

        if should_sync_mrope_positions:
            self._calc_mrope_positions(scheduler_output)
            self.mrope_positions.copy_to_gpu(total_num_scheduled_tokens)

        if should_sync_xdrope_positions:
            self._calc_xdrope_positions(scheduler_output)
            self.xdrope_positions.copy_to_gpu(total_num_scheduled_tokens)

        return mm_embeds, is_mm_embed

    def get_model(self) -> nn.Module:
        if not hasattr(self, "model"):
            raise ValueError("Cannot get model before model has been initialized")
        if isinstance(self.model, (CUDAGraphWrapper, UBatchWrapper)):
            # get raw model out of the cudagraph wrapper.
            return self.model.unwrap()
        return self.model

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        model = self.get_model()
        supported_tasks = list[GenerationTask]()

        if is_text_generation_model(model):
            supported_tasks.append("generate")

        if supports_transcription(model):
            if model.supports_transcription_only:
                return ["transcription"]

            supported_tasks.append("transcription")

        if supports_realtime(model):
            supported_tasks.append("realtime")

        return supported_tasks

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        model = self.get_model()
        if not is_pooling_model(model):
            return []

        supported_tasks = list(model.pooler.get_supported_tasks())

        if "score" in supported_tasks:
            num_labels = getattr(self.model_config.hf_config, "num_labels", 0)
            if num_labels != 1:
                supported_tasks.remove("score")
                logger.debug_once("Score API is only enabled for num_labels == 1.")

        return supported_tasks

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)

    def sync_and_slice_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors | None,
        sync_self: bool,
    ) -> IntermediateTensors:
        assert self.intermediate_tensors is not None

        tp = self.vllm_config.parallel_config.tensor_parallel_size
        is_rs = is_residual_scattered_for_sp(self.vllm_config, num_tokens)

        # When sequence parallelism is enabled, the "residual" tensor is sharded
        # across tensor parallel ranks, so each rank only needs its own slice.
        if sync_self:
            assert intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                is_scattered = k == "residual" and is_rs
                copy_len = num_tokens // tp if is_scattered else num_tokens
                self.intermediate_tensors[k][:copy_len].copy_(
                    v[:copy_len], non_blocking=True
                )

        return IntermediateTensors(
            {
                k: v[: num_tokens // tp]
                if k == "residual" and is_rs
                else v[:num_tokens]
                for k, v in self.intermediate_tensors.items()
            }
        )

    def eplb_step(self, is_dummy: bool = False, is_profile: bool = False) -> None:
        """
        Step for the EPLB (Expert Parallelism Load Balancing) state.
        """
        if not self.parallel_config.enable_eplb or self.eep_eplb_suppressed:
            return

        assert self.eplb_state is not None
        model = self.get_model()
        assert is_mixture_of_experts(model)
        self.eplb_state.step(
            is_dummy,
            is_profile,
            log_stats=self.parallel_config.eplb_config.log_balancedness,
        )

    def setup_eplb_from_mapping(
        self,
        expanded_physical_to_logical: torch.Tensor,
        old_num_physical_experts: int,
    ) -> None:
        model = self.get_model()
        assert is_mixture_of_experts(model)

        self.eplb_state = EplbState.from_mapping(
            model=model,
            model_config=self.model_config,
            device=self.device,
            parallel_config=self.parallel_config,
            expanded_physical_to_logical=expanded_physical_to_logical,
            num_valid_physical_experts=old_num_physical_experts,
        )

    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        kv_connector_output: KVConnectorOutput | None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        num_reqs = self.input_batch.num_reqs
        assert num_reqs == len(self.input_batch.pooling_params), (
            "Either all or none of the requests in a batch must be pooling request"
        )

        hidden_states = hidden_states[:num_scheduled_tokens]
        seq_lens_cpu = self.seq_lens.cpu[:num_reqs]

        pooling_metadata = self.input_batch.get_pooling_metadata()
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens_np,
            seq_lens_cpu,
            device=hidden_states.device,
            query_start_loc_gpu=self.query_start_loc.gpu[: num_reqs + 1],
        )

        model = cast(VllmModelForPooling, self.model)
        raw_pooler_output: PoolerOutput = model.pooler(
            hidden_states=hidden_states, pooling_metadata=pooling_metadata
        )

        finished_mask = [
            seq_len == prompt_len
            for seq_len, prompt_len in zip(seq_lens_cpu, pooling_metadata.prompt_lens)
        ]
        raw_pooler_output = self.late_interaction_runner.postprocess_pooler_output(
            raw_pooler_output=raw_pooler_output,
            pooling_params=pooling_metadata.pooling_params,
            req_ids=self.input_batch.req_ids,
            finished_mask=finished_mask,
        )

        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids.copy(),
            req_id_to_index=self.input_batch.req_id_to_index.copy(),
            kv_connector_output=kv_connector_output,
        )

        if raw_pooler_output is None or not any(finished_mask):
            model_runner_output.pooler_output = [None] * num_reqs
            return model_runner_output

        if self.use_async_scheduling:
            return AsyncGPUPoolingModelRunnerOutput(
                model_runner_output=model_runner_output,
                raw_pooler_output=raw_pooler_output,
                finished_mask=finished_mask,
                async_output_copy_stream=self.async_output_copy_stream,
            )

        model_runner_output.pooler_output = _copy_pooler_output_to_cpu(
            raw_pooler_output=raw_pooler_output,
            finished_mask=finished_mask,
        )
        self._sync_device()

        return model_runner_output

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        # Pad tokens to multiple of tensor_parallel_size when
        # enabled collective fusion for SP
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if self.compilation_config.pass_config.enable_sp and tp_size > 1:
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens

    def _prepare_mm_inputs(
        self, num_tokens: int
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.model.requires_raw_input_tokens:
            input_ids = self.input_ids.gpu[:num_tokens]
        else:
            input_ids = None

        inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
        return input_ids, inputs_embeds

    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        num_input_tokens: int,  # Padded
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        IntermediateTensors | None,
        dict[str, Any],
        ECConnectorOutput | None,
    ]:
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        is_first_rank = get_pp_group().is_first_rank
        is_encoder_decoder = self.model_config.is_encoder_decoder

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        ec_connector_output = None

        if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:
            # Run the multimodal encoder if any.
            with self.maybe_get_ec_connector_output(
                scheduler_output,
                encoder_cache=self.encoder_cache,
            ) as ec_connector_output:
                self._execute_mm_encoder(scheduler_output)
                mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)

            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            inputs_embeds_scheduled = self.model.embed_input_ids(
                self.input_ids.gpu[:num_scheduled_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(inputs_embeds_scheduled)

            input_ids, inputs_embeds = self._prepare_mm_inputs(num_input_tokens)
            model_kwargs = {
                **self._init_model_kwargs(),
                **self._extract_mm_kwargs(scheduler_output),
            }
        elif self.enable_prompt_embeds and is_first_rank:
            # Get the input embeddings for the tokens that are not input embeds,
            # then put them into the appropriate positions.
            # TODO(qthequartermasterman): Since even when prompt embeds are
            # enabled, (a) not all requests will use prompt embeds, and (b)
            # after the initial prompt is processed, the rest of the generated
            # tokens will be token ids, it is not desirable to have the
            # embedding layer outside of the CUDA graph all the time. The v0
            # engine avoids this by "double compiling" the CUDA graph, once
            # with input_ids and again with inputs_embeds, for all num_tokens.
            # If a batch only has token ids, then including the embedding layer
            # in the CUDA graph will be more performant (like in the else case
            # below).
            token_ids_idx = (
                self.is_token_ids.gpu[:num_scheduled_tokens]
                .nonzero(as_tuple=False)
                .squeeze(1)
            )
            # Some tokens ids may need to become embeds
            if token_ids_idx.numel() > 0:
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.embed_input_ids(input_ids=token_ids)
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = self._init_model_kwargs()
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = self._init_model_kwargs()

        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        elif self.uses_xdrope_dim > 0:
            positions = self.xdrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions.gpu[:num_input_tokens]

        if is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True
            )

        if is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
            # Run the encoder, just like we do with other multimodal inputs.
            # For an encoder-decoder model, our processing here is a bit
            # simpler, because the outputs are just passed to the decoder.
            # We are not doing any prompt replacement. We also will only
            # ever have a single encoder input.
            encoder_outputs = self._execute_mm_encoder(scheduler_output)
            model_kwargs.update({"encoder_outputs": encoder_outputs})

        return (
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            ec_connector_output,
        )

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:
        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        # Update output token ids with tokens sampled in last step
        # if async scheduling and required by current sampling params.
        self.input_batch.update_async_output_token_ids()
        if spec_decode_metadata is None:
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        # Update spec_token_ids with real draft tokens from pre step only when
        # output_token_ids is needed (penalties or bad_words are in use).
        if self.use_async_scheduling and self._draft_token_req_ids is not None:
            draft_token_ids_cpu, _ = self._get_draft_token_ids_cpu()
            self.input_batch.update_async_spec_token_ids(draft_token_ids_cpu)

        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
        return sampler_output

    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> tuple[
        dict[str, int],
        LogprobsLists | None,
        list[list[int]],
        dict[str, LogprobsTensors | None],
        list[str],
        dict[str, int],
        list[int],
    ]:
        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        num_reqs = self.input_batch.num_reqs
        discard_sampled_tokens_req_indices = np.nonzero(
            self.discard_request_mask.np[:num_reqs]
        )[0]
        for i in discard_sampled_tokens_req_indices:
            gen = self.input_batch.generators.get(int(i))
            if gen is not None:
                gen.set_offset(gen.get_offset() - 4)

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        logprobs_tensors = sampler_output.logprobs_tensors
        invalid_req_indices = []
        logprobs_lists = None
        force_real_sampled_ids = False
        if not self.use_async_scheduling:
            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
                # Mask out the sampled tokens that should not be sampled.
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[int(i)].clear()

                if logprobs_tensors is not None:
                    logprobs_lists = logprobs_tensors.tolists()
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                    discard_sampled_tokens_req_indices,
                    logprobs_tensors=logprobs_tensors,
                )
        else:
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)
            force_real_sampled_ids = False
            if self._has_sparse_attn and hasattr(self, "kv_cache_config"):
                for group in self.kv_cache_config.kv_cache_groups:
                    spec = group.kv_cache_spec
                    if (
                        isinstance(spec, SparseAttentionSpec)
                        and spec.cluster_granularity == "token"
                    ):
                        force_real_sampled_ids = True
                        break
            if self._sparse_probe_info_enabled or self._sparse_debug_decode_tokens:
                logger.info(
                    "[SparseRC] async_sampled_ids mode=%s has_sparse_attn=%s",
                    "real" if force_real_sampled_ids else "placeholder",
                    self._has_sparse_attn,
                )

            # Cache the sampled tokens on the GPU and avoid CPU sync.
            # These will be copied into input_ids in the next step
            # when preparing inputs.
            # With spec decoding, this is done in propose_draft_token_ids().
            if self.input_batch.prev_sampled_token_ids is None:
                assert sampled_token_ids.shape[-1] == 1
                self.input_batch.prev_sampled_token_ids = sampled_token_ids
            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids)
                if i not in invalid_req_indices_set
            }

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        req_ids = self.input_batch.req_ids
        self._sparse_output_tokens_before_step.clear()
        for req_idx in range(num_sampled_tokens):
            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            prev_output_n = len(req_state.output_token_ids)
            prompt_n = int(self.input_batch.num_prompt_tokens[req_idx])
            nspec_before = int(self.input_batch.num_tokens_no_spec[req_idx])
            if self.use_async_scheduling:
                if req_idx in invalid_req_indices_set:
                    sampled_ids = None
                elif force_real_sampled_ids:
                    sampled_ids = [int(sampled_token_ids[req_idx, 0].item())]
                else:
                    sampled_ids = [-1]
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]

            num_sampled_ids: int = len(sampled_ids) if sampled_ids else 0

            if not sampled_ids:
                continue

            self._sparse_output_tokens_before_step[req_id] = prev_output_n

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + num_sampled_ids
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}"
            )

            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx

            req_state.output_token_ids.extend(sampled_ids)
            out_n_after = len(req_state.output_token_ids)
            self._sparse_log_sample_step(
                req_idx=req_idx,
                req_id=req_id,
                prev_output_n=prev_output_n,
                sampled_ids=sampled_ids,
                logits=logits,
            )
            if (
                self._sparse_debug_first_token or _SPARSE_HARD_DEBUG_FIRST_NEW_TOKEN
            ) and prev_output_n <= 1:
                self._sparse_log_first_sampled_token(
                    req_idx=req_idx,
                    req_id=req_id,
                    prev_output_n=prev_output_n,
                    sampled_ids=sampled_ids,
                    logits=logits,
                    spec_decode_metadata=spec_decode_metadata,
                )

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        return (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    @contextmanager
    def synchronize_input_prep(self):
        if self.prepare_inputs_event is None:
            yield
            return

        # Ensure prior step has finished with reused CPU tensors.
        # This is required in the async scheduling case because
        # the CPU->GPU transfer happens async.
        self.prepare_inputs_event.synchronize()
        try:
            yield
        finally:
            self.prepare_inputs_event.record()

    def _model_forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> Any:
        """Helper method to call the model forward pass.

        This method can be overridden by subclasses for model execution.
        Motivation: We can inspect only this method versus
        the whole execute_model, which has additional logic.

        Args:
            input_ids: Input token IDs
            positions: Token positions
            intermediate_tensors: Tensors from previous pipeline stages
            inputs_embeds: Input embeddings (alternative to input_ids)
            **model_kwargs: Additional model arguments

        Returns:
            Model output tensor
        """
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    @staticmethod
    def _is_uniform_decode(
        max_num_scheduled_tokens: int,
        uniform_decode_query_len: int,
        num_tokens: int,
        num_reqs: int,
        force_uniform_decode: bool | None = None,
    ) -> bool:
        """
        Checks if it's a decode batch with same amount scheduled tokens
        across all requests.
        """
        return (
            (
                (max_num_scheduled_tokens == uniform_decode_query_len)
                and (num_tokens == max_num_scheduled_tokens * num_reqs)
            )
            if force_uniform_decode is None
            else force_uniform_decode
        )

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = True,
        force_eager: bool = False,
        # For cudagraph capture TODO(lucas): Refactor how we capture cudagraphs (will
        # be improved in model runner v2)
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
    ) -> tuple[
        CUDAGraphMode,
        BatchDescriptor,
        bool,
        torch.Tensor | None,
        CUDAGraphStat | None,
    ]:
        uniform_decode = self._is_uniform_decode(
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            uniform_decode_query_len=self.uniform_decode_query_len,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            force_uniform_decode=force_uniform_decode,
        )
        # Encoder-decoder models only support CG for decoder_step > 0 (no enc_output
        # is present). Also, chunked-prefill is disabled, so batch are uniform.
        has_encoder_output = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        # Compute LoRA state for cudagraph dispatch
        num_active_loras = (
            force_num_active_loras
            if force_num_active_loras is not None
            else len(self.input_batch.lora_id_to_lora_request)
        )
        has_lora = num_active_loras > 0 if force_has_lora is None else force_has_lora

        num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)

        def dispatch_cudagraph(num_tokens, disable_full=False, valid_modes=None):
            return self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens,
                has_lora=has_lora,
                uniform_decode=uniform_decode,
                num_active_loras=num_active_loras,
                valid_modes={CUDAGraphMode.NONE} if force_eager else valid_modes,
                invalid_modes={CUDAGraphMode.FULL} if disable_full else None,
            )

        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded, disable_full=use_cascade_attn or has_encoder_output
        )
        num_tokens_padded = batch_descriptor.num_tokens
        if self.compilation_config.pass_config.enable_sp:
            assert (
                batch_descriptor.num_tokens
                % self.vllm_config.parallel_config.tensor_parallel_size
                == 0
            ), (
                "Sequence parallelism requires num_tokens to be "
                "a multiple of tensor parallel size"
            )

        # Extra coordination when running data-parallel since we need to coordinate
        # across ranks
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=self.parallel_config,
                    allow_microbatching=allow_microbatching,
                    num_tokens_padded=num_tokens_padded,
                    uniform_decode=uniform_decode,
                    num_scheduled_tokens_per_request=num_scheduled_tokens_np,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )

            # Extract DP-synced values
            if num_tokens_across_dp is not None:
                dp_rank = self.parallel_config.data_parallel_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # Re-dispatch with DP padding so we have the correct batch_descriptor
                cudagraph_mode, batch_descriptor = dispatch_cudagraph(
                    num_tokens_padded,
                    valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
                )
                # Assert to make sure the agreed upon token count is correct otherwise
                # num_tokens_across_dp will no-longer be valid
                assert batch_descriptor.num_tokens == num_tokens_padded

        cudagraph_stats = None
        if self.vllm_config.observability_config.cudagraph_metrics:
            cudagraph_stats = CUDAGraphStat(
                num_unpadded_tokens=num_tokens,
                num_padded_tokens=batch_descriptor.num_tokens,
                num_paddings=batch_descriptor.num_tokens - num_tokens,
                runtime_mode=str(cudagraph_mode),
            )

        return (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        )

    def _register_layerwise_nvtx_hooks(self) -> None:
        """
        Register layerwise NVTX hooks if --enable-layerwise-nvtx-tracing is enabled
        to trace detailed information of each layer or module in the model.
        """

        if (
            self.vllm_config.observability_config.enable_layerwise_nvtx_tracing
            and not self.layerwise_nvtx_hooks_registered
        ):
            if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                logger.debug_once(
                    "layerwise NVTX tracing is not supported when CUDA graph is "
                    "turned off; you may observe part or all of the model "
                    "missing NVTX markers"
                )

            # In STOCK_TORCH_COMPILE mode, after registering hooks here,
            # the __call__ function of nn.module will be recompiled with
            # fullgraph=True. Since nvtx.range_push/pop are not traceable
            # by torch dynamo, we can't register hook functions here
            # because hook functions will also be traced by torch dynamo.
            if (
                self.vllm_config.compilation_config.mode
                == CompilationMode.STOCK_TORCH_COMPILE
            ):
                logger.debug_once(
                    "layerwise NVTX tracing is not supported when "
                    "CompilationMode is STOCK_TORCH_COMPILE, skipping "
                    "function hooks registration"
                )
            else:
                pyt_hooks = PytHooks()
                pyt_hooks.register_hooks(self.model, self.model.__class__.__name__)
                self.layerwise_nvtx_hooks_registered = True

    def _get_slot_mappings(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_tokens_unpadded: int,
        ubatch_slices: "UBatchSlices | None" = None,
    ) -> tuple[
        dict[int, torch.Tensor] | None,
        dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
    ]:
        """
        Build slot mappings in both formats needed by the system.

        Args:
            num_tokens_padded: Total number of tokens (padded)
            num_reqs_padded: Total number of requests (padded)
            num_tokens_unpadded: Actual number of tokens (unpadded)
            ubatch_slices: Optional ubatch slicing info for DBO

        Returns:
            A tuple of:
            - slot_mappings_by_gid: dict[int, torch.Tensor] for attention metadata
            - slot_mappings_by_layer: dict[str, torch.Tensor] or list for ForwardContext
        """
        if not (
            hasattr(self, "kv_cache_config")
            and self.kv_cache_config is not None
            and len(self.kv_cache_config.kv_cache_groups) > 0
        ):
            return None, None

        def _get_slot_mapping(kv_cache_gid: int):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = self.kv_cache_config.kv_cache_groups[
                kv_cache_gid
            ].kv_cache_spec
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                slot_mapping = torch.zeros(
                    (num_tokens_padded,),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_table[kv_cache_gid]
                slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded]

            # Fill unused with -1. Needed for reshape_and_cache in full cuda
            # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
            slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)

            return slot_mapping

        slot_mappings_by_gid = {
            gid: _get_slot_mapping(gid)
            for gid, _ in enumerate(self.kv_cache_config.kv_cache_groups)
        }

        slot_mappings_by_layer: dict[str, torch.Tensor] = {}
        for gid, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            slot_mapping = slot_mappings_by_gid[gid]
            for layer_name in kv_cache_group.layer_names:
                slot_mappings_by_layer[layer_name] = slot_mapping

        if ubatch_slices is not None:
            result: list[dict[str, torch.Tensor]] = []
            for ubatch in ubatch_slices:
                sliced_mappings: dict[str, torch.Tensor] = {}
                for layer_name, slot_mapping in slot_mappings_by_layer.items():
                    sliced_mappings[layer_name] = slot_mapping[ubatch.token_slice]
                result.append(sliced_mappings)
            return slot_mappings_by_gid, result

        return slot_mappings_by_gid, slot_mappings_by_layer

    def _e2e_trace_batch_info(
        self, scheduler_output: "SchedulerOutput"
    ) -> tuple[str, int, int, int, int, list[str], list[str]]:
        prefill_req_ids = [req.req_id for req in scheduler_output.scheduled_new_reqs]
        prefill_tokens = sum(
            scheduler_output.num_scheduled_tokens.get(req_id, 0)
            for req_id in prefill_req_ids
        )
        decode_req_ids: list[str] = []
        decode_tokens = 0

        cached = scheduler_output.scheduled_cached_reqs
        for req_id, num_output_tokens in zip(
            cached.req_ids, cached.num_output_tokens, strict=True
        ):
            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_output_tokens == 0:
                prefill_req_ids.append(req_id)
                prefill_tokens += num_tokens
            else:
                decode_req_ids.append(req_id)
                decode_tokens += num_tokens

        if prefill_req_ids and decode_req_ids:
            phase = "mixed"
        elif prefill_req_ids:
            phase = "prefill"
        elif decode_req_ids:
            phase = "decode"
        else:
            phase = "empty"
        return (
            phase,
            len(prefill_req_ids),
            len(decode_req_ids),
            prefill_tokens,
            decode_tokens,
            prefill_req_ids,
            decode_req_ids,
        )

    def _log_e2e_perf_trace(
        self,
        label: str,
        phase: str,
        prefill_reqs: int,
        decode_reqs: int,
        prefill_tokens: int,
        decode_tokens: int,
        prefill_req_ids: list[str],
        decode_req_ids: list[str],
        marks: list[tuple[str, float]],
    ) -> None:
        if len(marks) < 2:
            return

        start = marks[0][1]
        prev = start
        parts: list[str] = []
        for name, timestamp in marks[1:]:
            parts.append(f"{name}_ms={(timestamp - prev) * 1000.0:.3f}")
            prev = timestamp

        logger.info(
            "[E2EPerf][GPUModelRunner] label=%s phase=%s total_ms=%.3f "
            "prefill_reqs=%d decode_reqs=%d prefill_tokens=%d "
            "decode_tokens=%d prefill_ids=%s decode_ids=%s %s",
            label,
            phase,
            (marks[-1][1] - start) * 1000.0,
            prefill_reqs,
            decode_reqs,
            prefill_tokens,
            decode_tokens,
            prefill_req_ids,
            decode_req_ids,
            " ".join(parts),
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
        _decode_perf_t0 = (
            time.perf_counter() if self._decode_perf_stats_enabled else None
        )
        _e2e_trace_marks: list[tuple[str, float]] | None = None
        _e2e_trace_phase = "unknown"
        _e2e_trace_prefill_reqs = 0
        _e2e_trace_decode_reqs = 0
        _e2e_trace_prefill_tokens = 0
        _e2e_trace_decode_tokens = 0
        _e2e_trace_prefill_ids: list[str] = []
        _e2e_trace_decode_ids: list[str] = []
        if self._e2e_perf_trace_enabled:
            _e2e_trace_marks = [("start", time.perf_counter())]
            (
                _e2e_trace_phase,
                _e2e_trace_prefill_reqs,
                _e2e_trace_decode_reqs,
                _e2e_trace_prefill_tokens,
                _e2e_trace_decode_tokens,
                _e2e_trace_prefill_ids,
                _e2e_trace_decode_ids,
            ) = self._e2e_trace_batch_info(scheduler_output)
        if self._decode_perf_stats_enabled and _decode_perf_t0 is not None:
            # Seed the step-level end-to-end window.  ``sample_tokens`` reads
            # this on exit to compute ``[DecodePerfE2E] total_ms`` spanning
            # both RPC halves; cleared there so we do not attribute a stale
            # timestamp to the next step on PP/async paths.
            self._decode_perf_step_t0 = _decode_perf_t0
            self._decode_perf_step_exec_ms = 0.0
        if _SPARSE_DECODE_STEP_TRACE and self._has_sparse_attn:
            self._sparse_decode_trace_qhead_ms = 0.0
            self._sparse_decode_trace_qhead_calls = 0
            self._sparse_decode_trace_qhead_none = 0
            self._sparse_decode_trace_retro_ms = 0.0
            self._sparse_decode_trace_retro_calls = 0
        _decode_perf_preprocess_ms = 0.0
        _decode_perf_update_states_ms = 0.0
        _decode_perf_prepare_inputs_ms = 0.0
        _decode_perf_batch_plan_ms = 0.0
        _decode_perf_slot_mapping_ms = 0.0
        _decode_perf_attn_metadata_ms = 0.0
        _decode_perf_model_preprocess_ms = 0.0
        _decode_perf_forward_ms = 0.0
        _decode_perf_postprocess_ms = 0.0
        _decode_perf_compute_logits_ms = 0.0
        _decode_perf_num_reqs = 0
        _decode_perf_num_tokens = 0
        _decode_perf_max_scheduled = 0
        _decode_perf_decode_only = False
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        if self.routed_experts_initialized:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.clear_buffer()  # noqa
            else:
                logger.error("RoutedExpertsCapturer not initialized.")

        # If ngram_gpu is used, we need to copy the scheduler_output to avoid
        # the modification has influence on the scheduler_output in engine core process.
        # The replace is much faster than deepcopy.
        if (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        ):
            num_scheduled_tokens_copy = scheduler_output.num_scheduled_tokens.copy()
            spec_decode_tokens_copy = (
                scheduler_output.scheduled_spec_decode_tokens.copy()
            )
            scheduler_output = replace(
                scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens_copy,
                scheduled_spec_decode_tokens=spec_decode_tokens_copy,
            )

        if has_kv_transfer_group():
            kv_connector_metadata = scheduler_output.kv_connector_metadata
            assert kv_connector_metadata is not None
            get_kv_transfer_group().handle_preemptions(kv_connector_metadata)

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        _decode_perf_num_tokens = int(num_scheduled_tokens)
        _t_decode_preprocess = (
            time.perf_counter() if self._decode_perf_stats_enabled else None
        )
        with (
            record_function_or_nullcontext("gpu_model_runner: preprocess"),
            self.synchronize_input_prep(),
        ):
            # Update persistent batch states.
            _t_decode_update_states = (
                time.perf_counter() if self._decode_perf_stats_enabled else None
            )
            self._update_states(scheduler_output)
            if _e2e_trace_marks is not None:
                _e2e_trace_marks.append(("update_states", time.perf_counter()))
            if _t_decode_update_states is not None:
                _decode_perf_update_states_ms = (
                    time.perf_counter() - _t_decode_update_states
                ) * 1000.0

            if has_ec_transfer() and not get_ec_transfer().is_consumer:
                with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
                ) as ec_connector_output:
                    self._execute_mm_encoder(scheduler_output)
                    return make_empty_encoder_model_runner_output(scheduler_output)

            if not num_scheduled_tokens:
                if (
                    self.parallel_config.distributed_executor_backend
                    == "external_launcher"
                    and self.parallel_config.data_parallel_size > 1
                ):
                    # this is a corner case when both external launcher
                    # and DP are enabled, num_scheduled_tokens could be
                    # 0, and has_unfinished_requests in the outer loop
                    # returns True. before returning early here we call
                    # dummy run to ensure coordinate_batch_across_dp
                    # is called into to avoid out of sync issues.
                    self._dummy_run(1)
                if not has_kv_transfer_group():
                    # Return empty ModelRunnerOutput if no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs"
                )

            num_reqs = self.input_batch.num_reqs
            _decode_perf_num_reqs = int(num_reqs)
            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
            _decode_perf_max_scheduled = max_num_scheduled_tokens
            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
            _decode_perf_decode_only = (
                max_num_scheduled_tokens == 1
                and num_tokens_unpadded == num_reqs
                and len(scheduler_output.scheduled_encoder_inputs) == 0
            )

            _t_decode_prepare_inputs = (
                time.perf_counter()
                if (
                    self._decode_perf_stats_enabled
                    and _decode_perf_decode_only
                )
                else None
            )
            logits_indices, spec_decode_metadata = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )
            if _e2e_trace_marks is not None:
                _e2e_trace_marks.append(("prepare_inputs", time.perf_counter()))
            if _t_decode_prepare_inputs is not None:
                _decode_perf_prepare_inputs_ms = (
                    time.perf_counter() - _t_decode_prepare_inputs
                ) * 1000.0

            _t_decode_batch_plan = (
                time.perf_counter()
                if (
                    self._decode_perf_stats_enabled
                    and _decode_perf_decode_only
                )
                else None
            )
            cascade_attn_prefix_lens = None
            # Disable cascade attention when using microbatching (DBO)
            if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:
                # Pre-compute cascade attention prefix lengths
                cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                    num_scheduled_tokens_np,
                    self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    scheduler_output.num_common_prefix_blocks,
                )

            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
            )

            logger.debug(
                "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                "should_ubatch: %s, num_tokens_across_dp: %s",
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
            )

            num_tokens_padded = batch_desc.num_tokens
            num_reqs_padded = (
                batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
            )
            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                should_ubatch,
                num_scheduled_tokens_np,
                num_tokens_padded,
                num_reqs_padded,
                self.parallel_config.num_ubatches,
            )

            logger.debug(
                "ubatch_slices: %s, ubatch_slices_padded: %s",
                ubatch_slices,
                ubatch_slices_padded,
            )
            if _t_decode_batch_plan is not None:
                _decode_perf_batch_plan_ms = (
                    time.perf_counter() - _t_decode_batch_plan
                ) * 1000.0
            if _e2e_trace_marks is not None:
                _e2e_trace_marks.append(("batch_plan", time.perf_counter()))

            # True if any attention backend handles KV cache update separately
            # from forward() (i.e., forward_includes_kv_cache_update=False). When true,
            # slot_mappings must use padded dimensions to match the key/value tensors.
            has_separate_kv_update = not all(
                all(
                    g.backend.forward_includes_kv_cache_update
                    for g in self.attn_groups[id]
                )
                for id, spec in enumerate(self.kv_cache_config.kv_cache_groups)
                if not isinstance(spec.kv_cache_spec, EncoderOnlyAttentionSpec)
            )
            pad_attn = cudagraph_mode == CUDAGraphMode.FULL

            if self.cache_config.mamba_cache_mode == "align":
                mamba_utils.preprocess_mamba(
                    scheduler_output,
                    self.kv_cache_config,
                    self.cache_config,
                    self.mamba_state_idx,
                    self.input_batch,
                    self.requests,
                    self.compilation_config.static_forward_context,
                    self.model.get_mamba_state_copy_func(),
                    self._get_mamba_copy_bufs(),
                )

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

            _t_decode_slot_mapping = (
                time.perf_counter()
                if (
                    self._decode_perf_stats_enabled
                    and _decode_perf_decode_only
                )
                else None
            )
            slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
                num_tokens_padded=num_tokens_padded
                if pad_attn or has_separate_kv_update
                else num_tokens_unpadded,
                num_reqs_padded=(
                    num_reqs_padded if pad_attn or has_separate_kv_update else num_reqs
                ),
                num_tokens_unpadded=num_tokens_unpadded,
                ubatch_slices=ubatch_slices_padded,
            )
            if _t_decode_slot_mapping is not None:
                _decode_perf_slot_mapping_ms = (
                    time.perf_counter() - _t_decode_slot_mapping
                ) * 1000.0
            if _e2e_trace_marks is not None:
                _e2e_trace_marks.append(("slot_mapping", time.perf_counter()))

            _t0_attn_meta = (
                time.perf_counter()
                if (
                    self._sparse_perf_stats_enabled
                    or (
                        self._decode_perf_stats_enabled
                        and _decode_perf_decode_only
                    )
                )
                else None
            )
            attn_metadata, spec_decode_common_attn_metadata = (
                self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded if pad_attn else None,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded if pad_attn else None,
                    max_query_len=max_num_scheduled_tokens,
                    ubatch_slices=ubatch_slices_attn,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                    slot_mappings=slot_mappings_by_group,
                )
            )
            if _t0_attn_meta is not None:
                _attn_metadata_elapsed = time.perf_counter() - _t0_attn_meta
                if (
                    self._decode_perf_stats_enabled
                    and _decode_perf_decode_only
                ):
                    _decode_perf_attn_metadata_ms = (
                        _attn_metadata_elapsed * 1000.0
                    )
            if _t0_attn_meta is not None and self._sparse_perf_stats_enabled:
                self._sparse_perf_record(
                    "_build_attention_metadata",
                    _attn_metadata_elapsed,
                )
            if _e2e_trace_marks is not None:
                _e2e_trace_marks.append(("attn_metadata", time.perf_counter()))

            _t_decode_model_preprocess = (
                time.perf_counter()
                if (
                    self._decode_perf_stats_enabled
                    and _decode_perf_decode_only
                )
                else None
            )
            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output, num_tokens_padded, intermediate_tensors
            )
            if _t_decode_model_preprocess is not None:
                _decode_perf_model_preprocess_ms = (
                    time.perf_counter() - _t_decode_model_preprocess
                ) * 1000.0
            if _e2e_trace_marks is not None:
                _e2e_trace_marks.append(("model_preprocess", time.perf_counter()))

        if self._decode_perf_stats_enabled and _t_decode_preprocess is not None:
            _decode_perf_preprocess_ms = (
                time.perf_counter() - _t_decode_preprocess
            ) * 1000.0
        if _e2e_trace_marks is not None:
            _e2e_trace_marks.append(("preprocess_total", time.perf_counter()))

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:
            cudagraph_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False

        # Encoder-decoder models can only compile the pure decode steps where no
        # encoder inputs are present. Use eager for the first pass.
        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        # When spec decode is enabled, defer connector finalization
        # (wait_for_save + clear metadata) until after draft model runs.
        defer_kv_connector_finalize = self.speculative_config is not None
        _t_decode_forward = (
            time.perf_counter()
            if self._decode_perf_stats_enabled and _decode_perf_decode_only
            else None
        )
        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                ubatch_slices=ubatch_slices_padded,
                slot_mapping=slot_mappings,
                skip_compiled=has_encoder_input,
            ),
            record_function_or_nullcontext("gpu_model_runner: forward"),
            self.maybe_get_kv_connector_output(
                scheduler_output,
                defer_finalize=defer_kv_connector_finalize,
            ) as kv_connector_output,
        ):
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        if _t_decode_forward is not None:
            _decode_perf_forward_ms = (
                time.perf_counter() - _t_decode_forward
            ) * 1000.0
        if _e2e_trace_marks is not None:
            _e2e_trace_marks.append(("forward", time.perf_counter()))

        _t_decode_postprocess = (
            time.perf_counter()
            if self._decode_perf_stats_enabled and _decode_perf_decode_only
            else None
        )
        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
            if self.use_aux_hidden_state_outputs:
                # True when EAGLE 3 is used.
                hidden_states, aux_hidden_states = model_output
            else:
                # Common case.
                hidden_states = model_output
                aux_hidden_states = None

            if not self.broadcast_pp_output:
                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    return hidden_states

                if self.is_pooling_model:
                    # Return the pooling output.
                    return self._pool(
                        hidden_states,
                        num_scheduled_tokens,
                        num_scheduled_tokens_np,
                        kv_connector_output,
                    )

                _t_decode_compute_logits = (
                    time.perf_counter()
                    if (
                        self._decode_perf_stats_enabled
                        and _decode_perf_decode_only
                    )
                    else None
                )
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
                if _t_decode_compute_logits is not None:
                    _decode_perf_compute_logits_ms += (
                        time.perf_counter() - _t_decode_compute_logits
                    ) * 1000.0
            else:
                # Rare case.
                assert not self.is_pooling_model

                sample_hidden_states = hidden_states[logits_indices]
                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(
                            self.vllm_config, num_tokens_padded
                        )
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                    logits = None
                else:
                    _t_decode_compute_logits = (
                        time.perf_counter()
                        if (
                            self._decode_perf_stats_enabled
                            and _decode_perf_decode_only
                        )
                        else None
                    )
                    logits = self.model.compute_logits(sample_hidden_states)
                    if _t_decode_compute_logits is not None:
                        _decode_perf_compute_logits_ms += (
                            time.perf_counter() - _t_decode_compute_logits
                        ) * 1000.0

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()

                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

        if _t_decode_postprocess is not None:
            _decode_perf_postprocess_ms = (
                time.perf_counter() - _t_decode_postprocess
            ) * 1000.0
        if _e2e_trace_marks is not None:
            _e2e_trace_marks.append(("postprocess", time.perf_counter()))

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        )
        self.kv_connector_output = kv_connector_output
        if _e2e_trace_marks is not None:
            _e2e_trace_marks.append(("state_ready", time.perf_counter()))
            self._log_e2e_perf_trace(
                "execute_model",
                _e2e_trace_phase,
                _e2e_trace_prefill_reqs,
                _e2e_trace_decode_reqs,
                _e2e_trace_prefill_tokens,
                _e2e_trace_decode_tokens,
                _e2e_trace_prefill_ids,
                _e2e_trace_decode_ids,
                _e2e_trace_marks,
            )
        if self._decode_perf_stats_enabled and not _decode_perf_decode_only:
            # Non-decode-only step (e.g. prefill / mixed).  We do not want
            # ``[DecodePerfE2E]`` to fire from ``sample_tokens`` in that case
            # – drop the step-level window so the next call is a fresh
            # start.
            self._decode_perf_step_t0 = None
            self._decode_perf_step_exec_ms = 0.0
            self._decode_perf_step_exec_exit_t = None
        if (
            self._decode_perf_stats_enabled
            and _decode_perf_decode_only
            and _decode_perf_t0 is not None
        ):
            total_ms = (time.perf_counter() - _decode_perf_t0) * 1000.0
            # Remember for end-to-end log emitted by ``sample_tokens``.  Safe
            # to write outside the ``decode_only`` branch: we only read it in
            # ``sample_tokens`` under the same guard.
            self._decode_perf_step_exec_ms = total_ms
            known_ms = (
                _decode_perf_preprocess_ms
                + _decode_perf_forward_ms
                + _decode_perf_postprocess_ms
            )
            other_ms = max(0.0, total_ms - known_ms)
            _t_decode_perf_log = (
                time.perf_counter()
                if self._sparse_perf_stats_enabled and self._has_sparse_attn
                else None
            )
            logger.info(
                "[DecodePerf] mode=%s total_ms=%.3f preprocess_ms=%.3f "
                "update_states_ms=%.3f prepare_inputs_ms=%.3f "
                "batch_plan_ms=%.3f slot_mapping_ms=%.3f "
                "attn_metadata_ms=%.3f model_preprocess_ms=%.3f "
                "forward_ms=%.3f postprocess_ms=%.3f "
                "compute_logits_ms=%.3f "
                "other_ms=%.3f num_reqs=%d num_tokens=%d max_scheduled=%d "
                "has_kv_connector=%d cudagraph=%s",
                "sparse" if self._has_sparse_attn else "full",
                total_ms,
                _decode_perf_preprocess_ms,
                _decode_perf_update_states_ms,
                _decode_perf_prepare_inputs_ms,
                _decode_perf_batch_plan_ms,
                _decode_perf_slot_mapping_ms,
                _decode_perf_attn_metadata_ms,
                _decode_perf_model_preprocess_ms,
                _decode_perf_forward_ms,
                _decode_perf_postprocess_ms,
                _decode_perf_compute_logits_ms,
                other_ms,
                _decode_perf_num_reqs,
                _decode_perf_num_tokens,
                _decode_perf_max_scheduled,
                int(has_kv_transfer_group()),
                cudagraph_mode.name,
            )
            if _SPARSE_DECODE_STEP_TRACE and self._has_sparse_attn:
                trace_req_ids = list(req_ids[:_decode_perf_num_reqs])
                out_tokens_before: list[int] = []
                sparse_units: list[int] = []
                for rid in trace_req_ids:
                    req_state = self.requests.get(rid)
                    out_tokens_before.append(
                        0 if req_state is None
                        else len(req_state.output_token_ids)
                    )
                    sparse_units.append(
                        len(self._sparse_online_index.get(rid, {}))
                    )
                logger.info(
                    "[SparseDecodeStep] req_ids=%s out_tokens_before=%s "
                    "sparse_units=%s total_ms=%.3f preprocess_ms=%.3f "
                    "attn_metadata_ms=%.3f forward_ms=%.3f "
                    "postprocess_ms=%.3f other_ms=%.3f "
                    "qhead_ms=%.3f qhead_calls=%d qhead_none=%d "
                    "retro_ms=%.3f retro_calls=%d cudagraph=%s",
                    trace_req_ids,
                    out_tokens_before,
                    sparse_units,
                    total_ms,
                    _decode_perf_preprocess_ms,
                    _decode_perf_attn_metadata_ms,
                    _decode_perf_forward_ms,
                    _decode_perf_postprocess_ms,
                    other_ms,
                    self._sparse_decode_trace_qhead_ms,
                    self._sparse_decode_trace_qhead_calls,
                    self._sparse_decode_trace_qhead_none,
                    self._sparse_decode_trace_retro_ms,
                    self._sparse_decode_trace_retro_calls,
                    cudagraph_mode.name,
                )
            if _t_decode_perf_log is not None:
                # Direct measurement of the per-step ``[DecodePerf]``
                # logger.info cost.  This lands in the ``sample_tokens_ms``
                # share of ``[DecodePerfE2E]`` because ``step_exec_ms`` is
                # captured BEFORE the logger.info fires (so sample_ms absorbs
                # it).  If this key reports >> 1 ms on sparse and ~0 ms on
                # full, we have our smoking gun for the 70 ms gap between
                # ``sample_tokens:total`` and ``avg_sample_tokens_ms``.
                self._sparse_perf_record(
                    "execute_model:decode_perf_log",
                    time.perf_counter() - _t_decode_perf_log,
                )
        # Stamp exit time AFTER the log so ``sample_tokens`` can isolate the
        # engine-side gap (apply_grammar_bitmask, PP, IPC, ...) – this
        # quantity is normally 0 for single-process TP=1, and any non-trivial
        # value is a red flag for instrumentation overhead leaking in.
        if self._decode_perf_stats_enabled and _decode_perf_decode_only:
            self._decode_perf_step_exec_exit_t = time.perf_counter()
        return None

    @torch.inference_mode
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        _sp_perf_enabled = (
            self._sparse_perf_stats_enabled and self._has_sparse_attn
        )
        _st_t0 = time.perf_counter() if _sp_perf_enabled else None
        _e2e_sample_marks: list[tuple[str, float]] | None = None
        _e2e_sample_phase = "unknown"
        _e2e_sample_prefill_reqs = 0
        _e2e_sample_decode_reqs = 0
        _e2e_sample_prefill_tokens = 0
        _e2e_sample_decode_tokens = 0
        _e2e_sample_prefill_ids: list[str] = []
        _e2e_sample_decode_ids: list[str] = []
        if self._e2e_perf_trace_enabled and self.execute_model_state is not None:
            _e2e_sample_marks = [("start", time.perf_counter())]
            (
                _e2e_sample_phase,
                _e2e_sample_prefill_reqs,
                _e2e_sample_decode_reqs,
                _e2e_sample_prefill_tokens,
                _e2e_sample_decode_tokens,
                _e2e_sample_prefill_ids,
                _e2e_sample_decode_ids,
            ) = self._e2e_trace_batch_info(self.execute_model_state.scheduler_output)
        if _st_t0 is not None and self._decode_perf_step_exec_exit_t is not None:
            self._sparse_perf_record(
                "sample_tokens:entry_gap_after_execute_model",
                _st_t0 - self._decode_perf_step_exec_exit_t,
            )
            # One-shot: clear so a non-decode execute_model cannot leak into
            # the next sample_tokens window.
            self._decode_perf_step_exec_exit_t = None
        if (
            _st_t0 is not None
            and self._decode_perf_step_t0 is not None
            and self._decode_perf_step_exec_ms > 0.0
        ):
            # Full gap from ``[DecodePerf]`` exec_ms measurement point to
            # here.  This INCLUDES the ``logger.info`` call and any Python
            # tail work between line 5104 (exec_ms stamp) and line 5148
            # (exec_exit_t).  If this is ~ ``entry_gap_after_execute_model``
            # then logger is cheap; if it's orders of magnitude bigger, the
            # 70 ms gap between ``sample_tokens:total`` and
            # ``avg_sample_tokens_ms`` is logger-info time absorbed into
            # sample_ms.
            exec_ms_stamp = (
                self._decode_perf_step_t0
                + self._decode_perf_step_exec_ms / 1000.0
            )
            gap = _st_t0 - exec_ms_stamp
            if gap > 0.0:
                self._sparse_perf_record(
                    "sample_tokens:entry_gap_from_exec_ms_stamp",
                    gap,
                )
        if self.execute_model_state is None:
            # These short-circuit branches do not correspond to a completed
            # decode step (PP pass-through / empty KV-transfer).  Drop the
            # pending e2e timer so the next real step starts from a fresh
            # execute_model entry.
            self._decode_perf_step_t0 = None
            self._decode_perf_step_exec_ms = 0.0
            self._decode_perf_step_exec_exit_t = None
            kv_connector_output = self.kv_connector_output
            self.kv_connector_output = None
            # receive sampled token ids from the last PP rank.
            if self.use_async_scheduling and get_pp_group().world_size > 1:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
            if not kv_connector_output:
                return None  # type: ignore[return-value]

            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        _t_pre_bookkeep = time.perf_counter() if _sp_perf_enabled else None
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )
        if _e2e_sample_marks is not None:
            _e2e_sample_marks.append(("grammar", time.perf_counter()))

        with record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)
        if _e2e_sample_marks is not None:
            _e2e_sample_marks.append(("sample", time.perf_counter()))

        self._update_states_after_model_execute(
            sampler_output.sampled_token_ids, scheduler_output
        )
        if _e2e_sample_marks is not None:
            _e2e_sample_marks.append(("update_states_after", time.perf_counter()))
        if self.use_async_scheduling:
            pp = get_pp_group()
            # For torchrun external_launcher PP mode with broadcast_pp_output=True,
            # PP outputs have been broadcasted to all ranks at logits computation.
            # Therefore, here is no need to send sampled token ids again in this case.
            if not self.broadcast_pp_output and pp.world_size > 1 and pp.is_last_rank:
                self._pp_broadcast_prev_sampled_token_ids(
                    sampler_output.sampled_token_ids
                )

        self._draft_token_ids = None
        self._draft_token_req_ids = None
        self.input_batch.prev_sampled_token_ids = None

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("gpu_model_runner: draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                    slot_mappings,
                )
                self._copy_draft_token_ids_to_cpu(scheduler_output)

        spec_config = self.speculative_config
        propose_drafts_after_bookkeeping = False
        if spec_config is not None:
            input_fits_in_drafter = spec_decode_common_attn_metadata is not None and (
                spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens
                <= self.effective_drafter_max_model_len
            )
            use_gpu_toks = (
                spec_config.use_eagle()
                or spec_config.uses_draft_model()
                or spec_config.uses_extract_hidden_states()
            ) and not spec_config.disable_padded_drafter_batch
            if use_gpu_toks:
                # EAGLE/DraftModel speculative decoding can use the GPU sampled tokens
                # as inputs, and does not need to wait for bookkeeping to finish.
                assert isinstance(
                    self.drafter,
                    EagleProposer | DraftModelProposer | ExtractHiddenStatesProposer,
                )
                sampled_token_ids = sampler_output.sampled_token_ids
                if input_fits_in_drafter:
                    propose_draft_token_ids(sampled_token_ids)
                elif self.valid_sampled_token_count_event is not None:
                    assert spec_decode_common_attn_metadata is not None
                    next_token_ids, valid_sampled_tokens_count = (
                        self.drafter.prepare_next_token_ids_padded(
                            spec_decode_common_attn_metadata,
                            sampled_token_ids,
                            self.requests,
                            self.input_batch,
                            self.discard_request_mask.gpu,
                        )
                    )
                    self._copy_valid_sampled_token_count(
                        next_token_ids, valid_sampled_tokens_count
                    )
                    self._draft_token_ids = torch.zeros(
                        1, device=self.device, dtype=torch.int32
                    ).expand(len(self.input_batch.req_ids), self.num_spec_tokens)
                    self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)
            elif (
                spec_config.use_ngram_gpu()
                and not spec_config.disable_padded_drafter_batch
            ):
                assert isinstance(self.drafter, NgramProposerGPU)
                sampled_token_ids = sampler_output.sampled_token_ids
                if input_fits_in_drafter:
                    propose_draft_token_ids(sampled_token_ids)
                elif self.valid_sampled_token_count_event is not None:
                    assert spec_decode_common_attn_metadata is not None
                    next_token_ids, valid_sampled_tokens_count, _ = (
                        self.drafter.update_token_ids_ngram(
                            sampled_token_ids,
                            self.input_batch,
                            self.token_ids_gpu_tensor,
                            self.num_tokens_no_spec_gpu,
                            self.discard_request_mask.gpu,
                        )
                    )
                    self._copy_valid_sampled_token_count(
                        next_token_ids, valid_sampled_tokens_count
                    )
                    # Since we couldn't run the drafter,
                    # just use zeros for the draft tokens.
                    self._draft_token_ids = torch.zeros(
                        1, device=self.device, dtype=torch.int32
                    ).expand(len(self.input_batch.req_ids), self.num_spec_tokens)
                    self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)
            else:
                propose_drafts_after_bookkeeping = input_fits_in_drafter

        if _t_pre_bookkeep is not None:
            # Covers grammar bitmask apply + _sample + update_states_after +
            # any GPU-side speculative-decode prep that runs before
            # bookkeeping.  Bookkeeping below is the first spot that syncs
            # sampled_token_ids back to CPU, so this window is pure-GPU
            # launch / lightweight Python.
            self._sparse_perf_record(
                "sample_tokens:pre_bookkeep",
                time.perf_counter() - _t_pre_bookkeep,
            )

        _t_bookkeep = time.perf_counter() if _sp_perf_enabled else None
        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                scheduler_output.total_num_scheduled_tokens,
                spec_decode_metadata,
            )
        if _e2e_sample_marks is not None:
            _e2e_sample_marks.append(("bookkeeping_sync", time.perf_counter()))
        if _t_bookkeep is not None:
            # ``_bookkeeping_sync`` is the major GPU→CPU sync point (pulls
            # sampled token ids, logprobs, num_nans, etc.).  If the forward
            # pipeline still has pending attention-kernel work, this is where
            # the CPU blocks – that's why it's the prime suspect for the
            # sparse/full ``sample_tokens_ms`` gap.
            self._sparse_perf_record(
                "sample_tokens:bookkeeping_sync",
                time.perf_counter() - _t_bookkeep,
            )

        _t_post_bookkeep = time.perf_counter() if _sp_perf_enabled else None
        if propose_drafts_after_bookkeeping:
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            propose_draft_token_ids(valid_sampled_token_ids)

        # Finalize KV connector (wait_for_save + clear metadata) after
        # draft model runs. Deferred from target model forward to allow
        # draft model to also save its KV cache.
        if spec_config is not None:
            self.finalize_kv_connector()
        if _e2e_sample_marks is not None:
            _e2e_sample_marks.append(("draft_kv_finalize", time.perf_counter()))

        with record_function_or_nullcontext("gpu_model_runner: eplb"):
            self.eplb_step()

        # self.kv_connector_output may be modified during drafting
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        with record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
            if self.routed_experts_initialized:
                capturer = RoutedExpertsCapturer.get_instance()
                if capturer is not None:
                    capturer.save_captured_experts(indices=self.slot_mapping)  # noqa
                else:
                    logger.error("RoutedExpertsCapturer not initialized.")

            if _t_post_bookkeep is not None:
                # Window covers propose_drafts_after_bookkeeping +
                # finalize_kv_connector + eplb_step + RoutedExperts capture
                # – i.e. everything that happens on the CPU path between
                # the bookkeeping sync and the sparse feature collection.
                self._sparse_perf_record(
                    "sample_tokens:post_bookkeep_pre_sparse",
                    time.perf_counter() - _t_post_bookkeep,
                )

            # ── Sparse KV attention features ──────────────────────────────
            _t0_sparse_collect = (
                time.perf_counter() if self._sparse_perf_stats_enabled else None
            )
            (
                sparse_block_features,
                sparse_query_vectors,
                sparse_new_block_features,
                sparse_prefill_cluster_meta,
                sparse_new_block_features_gpu,
                sparse_prefill_block_features_gpu,
                sparse_prefill_cluster_meta_gpu,
            ) = self._collect_sparse_features(
                scheduler_output, self.input_batch.num_reqs
            )
            self._update_sparse_online_index(
                sparse_block_features,
                sparse_prefill_cluster_meta,
                sparse_new_block_features,
                sparse_new_block_features_gpu,
                sparse_prefill_block_features_gpu,
                sparse_prefill_cluster_meta_gpu,
            )
            if _e2e_sample_marks is not None:
                _e2e_sample_marks.append(("sparse_features", time.perf_counter()))
            if _t0_sparse_collect is not None:
                # Keep ``_collect_sparse_features`` key bitwise-compatible with
                # earlier logs (covers both ``_collect_sparse_features`` and
                # ``_update_sparse_online_index``).  We also split out the
                # pure _collect portion below so decode-only analyses stop
                # being contaminated by ``_update_sparse_online_index``.
                self._sparse_perf_record(
                    "_collect_sparse_features",
                    time.perf_counter() - _t0_sparse_collect,
                )
            # Clear per-step Q captures to free GPU memory references.
            self._sparse_q_captures.clear()
            _t0_perf_flush = (
                time.perf_counter() if self._sparse_perf_stats_enabled else None
            )
            self._sparse_perf_flush_if_needed()
            if _t0_perf_flush is not None:
                # Window logger.info cost at VLLM_SPARSE_PERF_LOG_INTERVAL=1
                # dominates sample_tokens runtime – isolate it so it does not
                # get attributed to genuine sparse work.  Note the timer only
                # fires on steps where the window actually flushes; at larger
                # intervals it is near-zero on the non-flush steps.
                self._sparse_perf_record(
                    "sample_tokens:perf_log_flush",
                    time.perf_counter() - _t0_perf_flush,
                )
            # ──────────────────────────────────────────────────────────────

            _t_tail = time.perf_counter() if _sp_perf_enabled else None
            output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output
                if self.supports_mm_inputs
                else None,
                num_nans_in_logits=num_nans_in_logits,
                cudagraph_stats=cudagraph_stats,
                sparse_block_features=sparse_block_features,
                sparse_query_vectors=sparse_query_vectors,
                sparse_new_block_features=sparse_new_block_features,
                sparse_prefill_cluster_meta=sparse_prefill_cluster_meta,
            )
            if _t_tail is not None:
                # ModelRunnerOutput construction – should be near-zero even
                # when sparse_* dict payloads are attached (they are already
                # realised tensors/dicts by this point).  Non-trivial values
                # here point at hidden cost in dataclass __post_init__ or
                # attribute serialization.
                self._sparse_perf_record(
                    "sample_tokens:output_build",
                    time.perf_counter() - _t_tail,
                )
            if _e2e_sample_marks is not None:
                _e2e_sample_marks.append(("output_build", time.perf_counter()))

        if not self.use_async_scheduling:
            if _sp_perf_enabled and _st_t0 is not None:
                self._sparse_perf_record(
                    "sample_tokens:total",
                    time.perf_counter() - _st_t0,
                )
            self._decode_perf_flush_e2e(st_t0=_st_t0)
            if _e2e_sample_marks is not None:
                _e2e_sample_marks.append(("return", time.perf_counter()))
                self._log_e2e_perf_trace(
                    "sample_tokens",
                    _e2e_sample_phase,
                    _e2e_sample_prefill_reqs,
                    _e2e_sample_decode_reqs,
                    _e2e_sample_prefill_tokens,
                    _e2e_sample_decode_tokens,
                    _e2e_sample_prefill_ids,
                    _e2e_sample_decode_ids,
                    _e2e_sample_marks,
                )
            return output

        with record_function_or_nullcontext(
            "gpu_model_runner: AsyncGPUModelRunnerOutput"
        ):
            async_output = AsyncGPUModelRunnerOutput(
                model_runner_output=output,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs_tensors=sampler_output.logprobs_tensors,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self.async_output_copy_stream,
                vocab_size=self.input_batch.vocab_size,
            )
        with record_function_or_nullcontext(
            "gpu_model_runner: set_async_sampled_token_ids"
        ):
            # Save ref of sampled_token_ids CPU tensor if the batch contains
            # any requests with sampling params that require output ids.
            self.input_batch.set_async_sampled_token_ids(
                async_output.sampled_token_ids_cpu,
                async_output.async_copy_ready_event,
            )

        if _sp_perf_enabled and _st_t0 is not None:
            self._sparse_perf_record(
                "sample_tokens:total",
                time.perf_counter() - _st_t0,
            )
        self._decode_perf_flush_e2e(st_t0=_st_t0)
        if _e2e_sample_marks is not None:
            _e2e_sample_marks.append(("return", time.perf_counter()))
            self._log_e2e_perf_trace(
                "sample_tokens",
                _e2e_sample_phase,
                _e2e_sample_prefill_reqs,
                _e2e_sample_decode_reqs,
                _e2e_sample_prefill_tokens,
                _e2e_sample_decode_tokens,
                _e2e_sample_prefill_ids,
                _e2e_sample_decode_ids,
                _e2e_sample_marks,
            )
        return async_output

    def _pp_broadcast_prev_sampled_token_ids(
        self, sampled_token_ids: torch.Tensor
    ) -> None:
        """Broadcast sampled token ids (GPU) from last PP stage"""
        pp = get_pp_group()
        assert pp.is_last_rank
        # `prev_sampled_token_ids` is expected to have shape [num_reqs, 1].
        assert sampled_token_ids.dim() == 2 and sampled_token_ids.shape[-1] == 1, (
            "PP+async expects sampled_token_ids to have shape [num_reqs, 1]"
        )
        torch.distributed.broadcast(
            sampled_token_ids, src=pp.rank, group=pp.device_group
        )

    def _pp_receive_prev_sampled_token_ids_to_input_batch(self) -> None:
        """Receive sampled token ids broadcast from last PP stage"""
        pp = get_pp_group()
        assert not pp.is_last_rank
        num_reqs = self.input_batch.num_reqs
        # `prev_sampled_token_ids` is expected to have shape [num_reqs, 1].
        recv = torch.empty((num_reqs, 1), dtype=torch.int32, device=self.device)
        torch.distributed.broadcast(recv, src=pp.last_rank, group=pp.device_group)
        self.input_batch.prev_sampled_token_ids = recv

        # construct `prev_req_id_to_index` here so `_prepare_input_ids`
        # can map req_id -> previous batch row
        discard_req_indices = np.nonzero(self.discard_request_mask.np[:num_reqs])[0]
        discard_req_indices_set = set(discard_req_indices)
        force_real_sampled_ids = False
        if self._has_sparse_attn and hasattr(self, "kv_cache_config"):
            for group in self.kv_cache_config.kv_cache_groups:
                spec = group.kv_cache_spec
                if (
                    isinstance(spec, SparseAttentionSpec)
                    and spec.cluster_granularity == "token"
                ):
                    force_real_sampled_ids = True
                    break
        prev_req_id_to_index: dict[str, int] = {}
        for i, req_id in enumerate(self.input_batch.req_ids):
            if i in discard_req_indices_set:
                continue
            prev_req_id_to_index[req_id] = i
            # PP+async scheduling: advance per-request local cached output length by
            # appending a placeholder token id. For token-sparse mode, keep
            # the real sampled token id to avoid polluting token traces with -1.
            if (req_state := self.requests.get(req_id)) is not None:
                if force_real_sampled_ids:
                    req_state.output_token_ids.append(int(recv[i, 0].item()))
                else:
                    req_state.output_token_ids.append(-1)
        self.input_batch.prev_req_id_to_index = prev_req_id_to_index

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        if not self.num_spec_tokens or not self._draft_token_req_ids:
            return None
        draft_token_ids, req_ids = self._get_draft_token_ids_cpu()
        return DraftTokenIds(req_ids, draft_token_ids)

    def _copy_draft_token_ids_to_cpu(
        self, scheduler_output: "SchedulerOutput", zeros_only: bool = False
    ) -> None:
        # Check if we need to copy draft tokens to CPU. In async scheduling,
        # we only copy when needed for structured output, penalties or bad_words.
        if self.use_async_scheduling and not (
            scheduler_output.has_structured_output_requests
            or self.input_batch.sampling_metadata.output_token_ids
        ):
            return
        # We must also set the corresponding request ids.
        self._draft_token_req_ids = self.input_batch.req_ids.copy()

        draft_token_ids: torch.Tensor = self._draft_token_ids
        if not torch.is_tensor(draft_token_ids):
            return
        assert self.draft_token_ids_event is not None
        assert self.draft_token_ids_copy_stream is not None
        assert self.draft_token_ids_cpu is not None
        default_stream = torch.cuda.current_stream()
        num_reqs = draft_token_ids.shape[0]
        with torch.cuda.stream(self.draft_token_ids_copy_stream):
            if not zeros_only:
                # Trigger async copy of draft token ids to cpu.
                self.draft_token_ids_copy_stream.wait_stream(default_stream)
                self.draft_token_ids_cpu[:num_reqs].copy_(
                    draft_token_ids, non_blocking=True
                )
            else:
                # No copy needed, just zero-out cpu tensor.
                self.draft_token_ids_cpu[:num_reqs] = 0
            self.draft_token_ids_event.record()

    def _get_draft_token_ids_cpu(self) -> tuple[list[list[int]], list[str]]:
        if isinstance(self._draft_token_ids, list):
            return self._draft_token_ids, self.input_batch.req_ids
        req_ids = self._draft_token_req_ids
        if req_ids is None:
            return [], []
        assert self.draft_token_ids_event is not None
        assert self.draft_token_ids_cpu is not None
        self.draft_token_ids_event.synchronize()
        return self.draft_token_ids_cpu[: len(req_ids)].tolist(), req_ids

    def _copy_valid_sampled_token_count(
        self, next_token_ids: torch.Tensor, valid_sampled_tokens_count: torch.Tensor
    ) -> None:
        if self.valid_sampled_token_count_event is None:
            return

        default_stream = torch.cuda.current_stream()
        # Initialize a new stream to overlap the copy operation with
        # prepare_input of draft model.
        with torch.cuda.stream(self.valid_sampled_token_count_copy_stream):
            self.valid_sampled_token_count_copy_stream.wait_stream(default_stream)  # type: ignore
            counts = valid_sampled_tokens_count
            counts_cpu = self.valid_sampled_token_count_cpu
            assert counts_cpu is not None
            counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)
            self.valid_sampled_token_count_event.record()

        self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)

    def _get_valid_sampled_token_count(self) -> list[int]:
        # Wait until valid_sampled_tokens_count is copied to cpu,
        prev_sampled_token_ids = self.input_batch.prev_sampled_token_ids
        sampled_count_event = self.valid_sampled_token_count_event
        if sampled_count_event is None or prev_sampled_token_ids is None:
            return []

        counts_cpu = self.valid_sampled_token_count_cpu
        assert counts_cpu is not None
        sampled_count_event.synchronize()
        return counts_cpu[: prev_sampled_token_ids.shape[0]].tolist()

    def propose_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        common_attn_metadata: CommonAttentionMetadata,
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
    ) -> list[list[int]] | torch.Tensor:
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        spec_config = self.speculative_config
        assert spec_config is not None
        if spec_config.method == "ngram":
            from vllm.v1.spec_decode.ngram_proposer import NgramProposer

            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, NgramProposer)
            draft_token_ids = self.drafter.propose(
                sampled_token_ids,
                self.input_batch.num_tokens_no_spec,
                self.input_batch.token_ids_cpu,
                slot_mappings=slot_mappings,
            )
        elif spec_config.use_ngram_gpu():
            assert isinstance(self.drafter, NgramProposerGPU)
            (
                next_token_ids,
                valid_sampled_tokens_count,
                valid_sampled_token_ids_gpu,
            ) = self.drafter.update_token_ids_ngram(
                sampled_token_ids,
                self.input_batch,
                self.token_ids_gpu_tensor,
                self.num_tokens_no_spec_gpu,
                self.discard_request_mask.gpu,
            )
            self._copy_valid_sampled_token_count(
                next_token_ids, valid_sampled_tokens_count
            )

            batch_size = next_token_ids.shape[0]

            draft_token_ids, num_valid_draft_tokens = self.drafter.propose(
                self.num_tokens_no_spec_gpu[:batch_size],
                self.token_ids_gpu_tensor[:batch_size],
                valid_sampled_token_ids_gpu,
                valid_sampled_tokens_count,
            )

            # Cache valid draft counts for scheduler-side trimming.
            self._num_valid_draft_tokens = num_valid_draft_tokens

            # Async D2H copy on a dedicated stream.
            copy_num_valid_draft_tokens(
                self._num_valid_draft_tokens_cpu,
                self._num_valid_draft_tokens_copy_stream,
                self._num_valid_draft_tokens_event,
                self._num_valid_draft_tokens,
                self.input_batch.num_reqs,
            )
        elif spec_config.method == "suffix":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, SuffixDecodingProposer)
            draft_token_ids = self.drafter.propose(
                self.input_batch, sampled_token_ids, slot_mappings=slot_mappings
            )
        elif spec_config.method == "medusa":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, MedusaProposer)

            if sample_hidden_states.shape[0] == len(sampled_token_ids):
                # The input to the target model does not include draft tokens.
                hidden_states = sample_hidden_states
            else:
                indices = []
                offset = 0
                assert spec_decode_metadata is not None, (
                    "No spec decode metadata for medusa"
                )
                for num_draft, tokens in zip(
                    spec_decode_metadata.num_draft_tokens, sampled_token_ids
                ):
                    indices.append(offset + len(tokens) - 1)
                    offset += num_draft + 1
                indices = torch.tensor(indices, device=self.device)
                hidden_states = sample_hidden_states[indices]

            draft_token_ids = self.drafter.propose(
                target_hidden_states=hidden_states,
                sampling_metadata=sampling_metadata,
                slot_mappings=slot_mappings,
            )
        elif spec_config.uses_extract_hidden_states():
            assert isinstance(self.drafter, ExtractHiddenStatesProposer)
            assert isinstance(sampled_token_ids, torch.Tensor), (
                "sampled_token_ids should be a torch.Tensor for "
                "extract_hidden_states method."
            )
            if not self.use_aux_hidden_state_outputs or aux_hidden_states is None:
                raise ValueError(
                    "aux_hidden_states are required when using `extract_hidden_states`"
                )
            target_hidden_states = [h[:num_scheduled_tokens] for h in aux_hidden_states]

            draft_token_ids = self.drafter.propose(
                sampled_token_ids=sampled_token_ids,
                target_hidden_states=target_hidden_states,
                common_attn_metadata=common_attn_metadata,
                slot_mappings=slot_mappings,
            )
            next_token_ids, valid_sampled_tokens_count = (
                self.drafter.prepare_next_token_ids_padded(
                    common_attn_metadata,
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    self.discard_request_mask.gpu,
                )
            )
            self._copy_valid_sampled_token_count(
                next_token_ids, valid_sampled_tokens_count
            )

        elif spec_config.use_eagle() or spec_config.uses_draft_model():
            assert isinstance(self.drafter, EagleProposer | DraftModelProposer)

            if spec_config.disable_padded_drafter_batch:
                # When padded-batch is disabled, the sampled_token_ids should be
                # the cpu-side list[list[int]] of valid sampled tokens for each
                # request, with invalid requests having empty lists.
                assert isinstance(sampled_token_ids, list), (
                    "sampled_token_ids should be a python list when"
                    "padded-batch is disabled."
                )
                next_token_ids = self.drafter.prepare_next_token_ids_cpu(
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    scheduler_output.num_scheduled_tokens,
                )
            else:
                # When using padded-batch, the sampled_token_ids should be
                # the gpu tensor of sampled tokens for each request, of shape
                # (num_reqs, num_spec_tokens + 1) with rejected tokens having
                # value -1.
                assert isinstance(sampled_token_ids, torch.Tensor), (
                    "sampled_token_ids should be a torch.Tensor when"
                    "padded-batch is enabled."
                )
                next_token_ids, valid_sampled_tokens_count = (
                    self.drafter.prepare_next_token_ids_padded(
                        common_attn_metadata,
                        sampled_token_ids,
                        self.requests,
                        self.input_batch,
                        self.discard_request_mask.gpu,
                    )
                )
                self._copy_valid_sampled_token_count(
                    next_token_ids, valid_sampled_tokens_count
                )

            num_rejected_tokens_gpu = None
            if spec_decode_metadata is None:
                token_indices_to_sample = None
                # input_ids can be None for multimodal models.
                target_token_ids = self.input_ids.gpu[:num_scheduled_tokens]
                target_positions = self._get_positions(num_scheduled_tokens)
                if self.use_aux_hidden_state_outputs:
                    assert aux_hidden_states is not None
                    target_hidden_states = torch.cat(
                        [h[:num_scheduled_tokens] for h in aux_hidden_states], dim=-1
                    )
                else:
                    target_hidden_states = hidden_states[:num_scheduled_tokens]
            else:
                if spec_config.disable_padded_drafter_batch:
                    token_indices_to_sample = None
                    common_attn_metadata, token_indices = self.drafter.prepare_inputs(
                        common_attn_metadata,
                        sampled_token_ids,
                        spec_decode_metadata.num_draft_tokens,
                    )
                    target_token_ids = self.input_ids.gpu[token_indices]
                    target_positions = self._get_positions(token_indices)
                    if self.use_aux_hidden_state_outputs:
                        assert aux_hidden_states is not None
                        target_hidden_states = torch.cat(
                            [h[token_indices] for h in aux_hidden_states], dim=-1
                        )
                    else:
                        target_hidden_states = hidden_states[token_indices]
                else:
                    (
                        common_attn_metadata,
                        token_indices_to_sample,
                        num_rejected_tokens_gpu,
                    ) = self.drafter.prepare_inputs_padded(
                        common_attn_metadata,
                        spec_decode_metadata,
                        valid_sampled_tokens_count,
                    )
                    total_num_tokens = common_attn_metadata.num_actual_tokens
                    # When padding the batch, token_indices is just a range
                    target_token_ids = self.input_ids.gpu[:total_num_tokens]
                    target_positions = self._get_positions(total_num_tokens)
                    if self.use_aux_hidden_state_outputs:
                        assert aux_hidden_states is not None
                        target_hidden_states = torch.cat(
                            [h[:total_num_tokens] for h in aux_hidden_states], dim=-1
                        )
                    else:
                        target_hidden_states = hidden_states[:total_num_tokens]

            if self.supports_mm_inputs and self.drafter.supports_mm_inputs:
                mm_embed_inputs = self._gather_mm_embeddings(
                    scheduler_output,
                    shift_computed_tokens=1,
                )
            else:
                mm_embed_inputs = None

            draft_token_ids = self.drafter.propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                token_indices_to_sample=token_indices_to_sample,
                sampling_metadata=sampling_metadata,
                common_attn_metadata=common_attn_metadata,
                mm_embed_inputs=mm_embed_inputs,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                slot_mappings=slot_mappings,
            )

        return draft_token_ids

    def update_config(self, overrides: dict[str, Any]) -> None:
        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            assert config_name in allowed_config_names, (
                f"Config `{config_name}` not supported. "
                f"Allowed configs: {allowed_config_names}"
            )
            config = getattr(self, config_name)
            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)

    @instrument(span_name="Loading (GPU)")
    def load_model(self, load_dummy_weights: bool = False) -> None:
        """
        Args:
            load_dummy_weights: load dummy weights instead of real weights.
        """
        logger.info_once(
            "Starting to load model %s...",
            self.model_config.model,
            scope="global",
        )

        if self.parallel_config.enable_eplb:
            self.eplb_state = EplbState(self.parallel_config, self.device)
            eplb_models = 0

        try:
            with DeviceMemoryProfiler() as m:
                time_before_load = time.perf_counter()
                if load_dummy_weights:
                    self.load_config.load_format = "dummy"
                model_loader = get_model_loader(self.load_config)
                self.model = model_loader.load_model(
                    vllm_config=self.vllm_config, model_config=self.model_config
                )
                if self.lora_config:
                    self.model = self.load_lora_model(
                        self.model, self.vllm_config, self.device
                    )
                if hasattr(self, "drafter"):
                    logger.info_once("Loading drafter model...")
                    self.drafter.load_model(self.model)
                    if (
                        hasattr(self.drafter, "model")
                        and is_mixture_of_experts(self.drafter.model)
                        and self.parallel_config.enable_eplb
                    ):
                        assert not self.parallel_config.enable_elastic_ep, (
                            "Elastic EP is not supported with drafter model."
                        )
                        spec_config = self.vllm_config.speculative_config
                        assert spec_config is not None
                        assert spec_config.draft_model_config is not None
                        logger.info_once(
                            "EPLB is enabled for drafter model %s.",
                            spec_config.draft_model_config.model,
                        )
                        if self.eplb_state is None:
                            self.eplb_state = EplbState(
                                self.parallel_config, self.device
                            )
                        self.eplb_state.add_model(
                            self.drafter.model,
                            spec_config.draft_model_config,
                        )
                        eplb_models += 1

                if self.use_aux_hidden_state_outputs:
                    if not supports_eagle3(self.get_model()):
                        raise RuntimeError(
                            "Model does not support EAGLE3 interface but "
                            "aux_hidden_state_outputs was requested"
                        )

                    # Try to get auxiliary layers from speculative config,
                    # otherwise use model's default layers
                    aux_layers = self._get_eagle3_aux_layers_from_config()
                    if aux_layers:
                        logger.info(
                            "Using auxiliary layers from speculative config: %s",
                            aux_layers,
                        )
                    else:
                        aux_layers = (
                            self.model.get_eagle3_default_aux_hidden_state_layers()
                        )

                    self.model.set_aux_hidden_state_layers(aux_layers)
                time_after_load = time.perf_counter()
            self.model_memory_usage = m.consumed_memory
        except torch.cuda.OutOfMemoryError as e:
            msg = (
                "Failed to load model - not enough GPU memory. "
                "Try lowering --gpu-memory-utilization to free memory for weights, "
                "increasing --tensor-parallel-size, or using --quantization. "
                "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
                "for more tips."
            )
            combined_msg = f"{msg} (original error: {e})"
            logger.error(combined_msg)
            raise e
        logger.info_once(
            "Model loading took %s GiB memory and %.6f seconds",
            format_gib(self.model_memory_usage),
            time_after_load - time_before_load,
            scope="local",
        )
        if not load_dummy_weights:
            prepare_communication_buffer_for_model(self.model)
            if (drafter := getattr(self, "drafter", None)) and (
                drafter_model := getattr(drafter, "model", None)
            ):
                prepare_communication_buffer_for_model(drafter_model)
        mm_config = self.model_config.multimodal_config
        self.is_multimodal_pruning_enabled = (
            supports_multimodal_pruning(self.get_model())
            and mm_config is not None
            and mm_config.is_multimodal_pruning_enabled()
        )
        self.requires_sequential_video_encoding = hasattr(
            self.get_model(), "requires_sequential_video_encoding"
        )  # Temporary hack for dynamic res video w/o support for bs>1 yet

        if (
            is_mixture_of_experts(self.model)
            and self.parallel_config.enable_eplb
            and not load_dummy_weights
        ):
            logger.info_once("EPLB is enabled for model %s.", self.model_config.model)
            assert self.eplb_state is not None
            self.eplb_state.add_model(
                self.model,
                self.model_config,
            )
            if self.eplb_state.is_async:
                self.eplb_state.start_async_loop()

        if (
            self.vllm_config.compilation_config.mode
            == CompilationMode.STOCK_TORCH_COMPILE
        ):
            backend = self.vllm_config.compilation_config.init_backend(self.vllm_config)
            compilation_counter.stock_torch_compile_count += 1
            self.model.compile(fullgraph=True, backend=backend)
            return
        # for other compilation modes, cudagraph behavior is controlled by
        # CudagraphWrapper and CudagraphDispatcher of vllm.

        # wrap the model with full cudagraph wrapper if needed.
        cudagraph_mode = self.compilation_config.cudagraph_mode
        assert cudagraph_mode is not None
        if (
            cudagraph_mode.has_full_cudagraphs()
            and not self.parallel_config.use_ubatching
        ):
            self.model = CUDAGraphWrapper(
                self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
            )
        elif self.parallel_config.use_ubatching:
            if cudagraph_mode.has_full_cudagraphs():
                self.model = UBatchWrapper(
                    self.model, self.vllm_config, CUDAGraphMode.FULL, self.device
                )
            else:
                self.model = UBatchWrapper(
                    self.model, self.vllm_config, CUDAGraphMode.NONE, self.device
                )

        get_offloader().post_init()

    def _get_eagle3_aux_layers_from_config(self) -> tuple[int, ...] | None:
        """Extract Eagle3 auxiliary layer indices from speculative config.

        These indices specify which hidden states from the base model should
        be used as auxiliary inputs for the Eagle3 drafter model during
        speculative decoding.

        Returns:
            Tuple of layer indices if found in draft model config,
            None otherwise.
        """
        if not (self.speculative_config and self.speculative_config.draft_model_config):
            return None

        hf_config = self.speculative_config.draft_model_config.hf_config
        if not hasattr(hf_config, "eagle_aux_hidden_state_layer_ids"):
            return None

        layer_ids = hf_config.eagle_aux_hidden_state_layer_ids
        if layer_ids and isinstance(layer_ids, (list, tuple)):
            return tuple(layer_ids)

        return None

    def reload_weights(
        self,
        weights_iterator: Iterable[tuple[str, torch.Tensor]] | None = None,
        weights_path: str | None = None,
        is_checkpoint_format: bool = True,
    ) -> None:
        """
        Reload weights from a weights iterator or from disk

        :param weights_iterator: weights to load into model
        :param weights_path: path to load weights from if weights_iterator is not
            provided. Use path of original model if neither is provided.
        :param is_checkpoint_format: set to False if weights have already been processed
            into kernel format (repacking, renaming, etc.)
        """
        # TODO(@kylesayrs): generalize to all runners and loaders
        # argument validation
        if weights_iterator is None and not is_checkpoint_format:
            logger.warning(
                "Reloading from disk means that weights will be in checkpoint format. "
                "Please use `is_checkpoint_format=True` "
                "to avoid weight reloading errors"
            )

        model = self.get_model()
        weights_to_load = {name for name, _ in model.named_parameters()}
        counter_before_reloading = time.perf_counter()

        # load weights from disk if none are provided
        if weights_iterator is None:
            model_loader = get_model_loader(self.load_config)
            if not hasattr(model_loader, "get_all_weights"):
                raise NotImplementedError(
                    f"Model reloading with `{self.load_config.load_format}` format"
                )

            if weights_path is not None:
                self.model_config.model = weights_path
            weights_iterator = model_loader.get_all_weights(self.model_config, model)
            weights_iterator = cast(
                Iterable[tuple[str, torch.Tensor]], weights_iterator
            )

        # begin loading weights
        logger.info_once("Reloading weights inplace...", scope="local")
        load_device = (
            self.vllm_config.load_config.device or self.vllm_config.device_config.device
        )
        with torch.device(load_device):
            if is_checkpoint_format:
                # load weights from checkpoint/ original model format
                initialize_layerwise_reload(model)
                loaded_weights = model.load_weights(weights_iterator)
                finalize_layerwise_reload(model, self.model_config)

            else:
                # load weights from kernel format
                logger.warning_once(
                    "Reloading with `is_checkpoint_format=True` requires that "
                    "weights be in kernel format and already sharded",
                    scope="local",
                )
                loaded_weights = set()
                for name, loaded_weight in weights_iterator:
                    param = model.get_parameter(name)  # TODO: buffers?
                    param.copy_(loaded_weight)
                    loaded_weights.add(name)

        # logging and validation
        counter_after_reloading = time.perf_counter()
        diff_seconds = counter_after_reloading - counter_before_reloading
        logger.info_once(
            "Reloading and processing weights took %.2f seconds",
            diff_seconds,
            scope="local",
        )
        if self.model_config.quantization is None and loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                logger.warning(
                    "Following weights were not loaded from checkpoint: %s",
                    weights_not_loaded,
                )

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, LogprobsTensors | None]:
        num_prompt_logprobs_dict = self.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            num_tokens = num_scheduled_tokens.get(req_id)
            if num_tokens is None:
                # This can happen if the request was preempted in prefill stage.
                continue

            # Get metadata for this request.
            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                # Prompt logprobs is incompatible with prompt embeddings
                continue

            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True
            )

            # Set up target LogprobsTensors object.
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1
                )
                in_progress_dict[req_id] = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[offset : offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks, _ = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids
            )

            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True
            )
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True
            )

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    def _get_nans_in_logits(
        self,
        logits: torch.Tensor | None,
    ) -> dict[str, int]:
        try:
            if logits is None:
                return {req_id: 0 for req_id in self.input_batch.req_ids}

            num_nans_in_logits = {}
            num_nans_for_index = logits.isnan().sum(dim=-1).cpu().numpy()
            for req_id in self.input_batch.req_ids:
                req_index = self.input_batch.req_id_to_index[req_id]
                num_nans_in_logits[req_id] = (
                    int(num_nans_for_index[req_index])
                    if num_nans_for_index is not None and req_index < logits.shape[0]
                    else 0
                )
            return num_nans_in_logits
        except IndexError:
            return {}

    @contextmanager
    def maybe_randomize_inputs(
        self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None
    ):
        """
        Randomize input_ids if VLLM_RANDOMIZE_DP_DUMMY_INPUTS is set.
        This is to help balance expert-selection
         - during profile_run
         - during DP rank dummy run
        """

        dp_size = self.vllm_config.parallel_config.data_parallel_size
        randomize_inputs = envs.VLLM_RANDOMIZE_DP_DUMMY_INPUTS and dp_size > 1
        if not randomize_inputs:
            yield
        elif input_ids is not None:

            @functools.cache
            def rand_input_ids() -> torch.Tensor:
                return torch.randint_like(
                    self.input_ids.gpu,
                    low=0,
                    high=self.model_config.get_vocab_size(),
                )

            logger.debug_once("Randomizing dummy input_ids for DP Rank")
            input_ids.copy_(rand_input_ids()[: input_ids.size(0)], non_blocking=True)
            yield
            input_ids.fill_(0)
        else:

            @functools.cache
            def rand_inputs_embeds() -> torch.Tensor:
                return torch.randn_like(
                    self.inputs_embeds.gpu,
                )

            assert inputs_embeds is not None
            logger.debug_once("Randomizing dummy inputs_embeds for DP Rank")
            inputs_embeds.copy_(
                rand_inputs_embeds()[: inputs_embeds.size(0)], non_blocking=True
            )
            yield
            inputs_embeds.fill_(0)

    def _get_mm_dummy_batch(
        self,
        modality: str,
        max_items_per_batch: int,
    ) -> BatchedTensorInputs:
        """Dummy data for profiling and precompiling multimodal models."""
        assert self.mm_budget is not None

        # Don't use `max_items_per_batch` here to avoid redundant computation
        dummy_mm_inputs = self.mm_registry.get_dummy_mm_inputs(
            self.model_config,
            mm_counts={modality: 1},
            cache=self.mm_budget.cache,
        )
        dummy_mm_item = dummy_mm_inputs["mm_kwargs"][modality][0]

        # We use the cache so that the item is saved to the cache,
        # but not read from the cache
        assert dummy_mm_item is not None, "Item should not already be cached"

        return next(
            mm_kwargs_batch
            for _, _, mm_kwargs_batch in group_and_batch_mm_kwargs(
                [(modality, dummy_mm_item)] * max_items_per_batch,
                device=self.device,
                pin_memory=self.pin_memory,
            )
        )

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
        profile_seq_lens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - if not set will determine the cudagraph mode based on using
                    the self.cudagraph_dispatcher.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                    needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            create_mixed_batch: If True, create a mixed batch with both decode
                (1 token) and prefill (multiple tokens) requests.
            remove_lora: If False, dummy LoRAs are not destroyed after the run
            num_active_loras: Number of distinct active LoRAs to capture for.
                LoRA is activated when num_active_loras > 0.
            profile_seq_lens: If provided, use this value for seq_lens instead
                of max_query_len. Used to profile attention workspace that
                scales with context length.
        """
        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # The current dummy run only covers LM execution, so we can skip it.
            # mm encoder dummy run may need to add in the future.
            return torch.tensor([]), torch.tensor([])

        assert (
            cudagraph_runtime_mode is None
            or cudagraph_runtime_mode.is_valid_runtime_mode()
        )

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.max_num_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            assert not uniform_decode
            # Create mixed batch:
            # first half decode tokens, second half one prefill
            num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        num_tokens_unpadded = int(num_scheduled_tokens.sum())

        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

        _cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = (
            self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens,
                max_num_scheduled_tokens=max_query_len,
                use_cascade_attn=False,
                allow_microbatching=allow_microbatching,
                force_eager=is_profile
                or (cudagraph_runtime_mode == CUDAGraphMode.NONE),
                # `force_uniform_decode` is used for cudagraph capture; because for
                # capturing mixed prefill-decode batches, we sometimes use
                # num_tokens == num_reqs which looks like a uniform decode batch to the
                # dispatcher; but we actually want to capture a piecewise cudagraph
                force_uniform_decode=uniform_decode,
                # `force_has_lora` is used for cudagraph capture; because LoRA is
                # activated later in the context manager, but we need to know the
                # LoRA state when determining the batch descriptor for capture
                force_has_lora=num_active_loras > 0,
                # `force_num_active_loras` is used for cudagraph capture; because we
                # need to capture graphs for specific num_active_loras counts
                force_num_active_loras=num_active_loras,
            )
        )

        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cudagraph_mode
        else:
            assert cudagraph_runtime_mode == _cudagraph_mode, (
                f"Cudagraph runtime mode mismatch in dummy_run. "
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
            )

        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = (
            batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        )
        ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
            should_ubatch,
            num_scheduled_tokens,
            num_tokens_padded,
            num_reqs_padded,
            self.vllm_config.parallel_config.num_ubatches,
        )
        logger.debug(
            "ubatch_slices: %s, ubatch_slices_padded: %s",
            ubatch_slices,
            ubatch_slices_padded,
        )

        attn_metadata: PerLayerAttnMetadata | None = None

        slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
            num_tokens_padded=num_tokens,
            num_reqs_padded=num_reqs_padded,
            num_tokens_unpadded=num_tokens_unpadded,
            ubatch_slices=ubatch_slices_padded,
        )

        # _dummy_run shares pinned CPU buffers (seq_lens, query_start_loc,
        # etc.) with execute_model.  It must participate in the same event
        # protocol so that back-to-back dummy/real steps don't overwrite
        # pinned memory while a prior non_blocking H2D DMA is still reading.
        with self.synchronize_input_prep():
            # If force_attention is True, we always capture attention.
            # Otherwise, it only happens for cudagraph_runtime_mode=FULL.
            if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
                if profile_seq_lens is not None:
                    seq_lens = profile_seq_lens  # type: ignore[assignment]
                elif create_mixed_batch:
                    # In the mixed batch mode (used for FI warmup), we use
                    # shorter sequence lengths to run faster.
                    # TODO(luka) better system for describing dummy batches
                    seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]  # type: ignore[assignment]
                else:
                    seq_lens = max_query_len  # type: ignore[assignment]
                self.seq_lens.np[:num_reqs] = seq_lens
                self.seq_lens.np[num_reqs:] = 0
                self.seq_lens.copy_to_gpu()

                cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
                self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
                self.query_start_loc.copy_to_gpu()

                pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
                _t0_attn_meta = (
                    time.perf_counter() if self._sparse_perf_stats_enabled else None
                )
                attn_metadata, _ = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded if pad_attn else None,
                    num_reqs=num_reqs_padded,
                    max_query_len=max_query_len,
                    ubatch_slices=(ubatch_slices_padded if pad_attn else ubatch_slices),
                    for_cudagraph_capture=is_graph_capturing,
                    slot_mappings=slot_mappings_by_group,
                    use_spec_decode=self.speculative_config is not None,
                )
                if _t0_attn_meta is not None:
                    self._sparse_perf_record(
                        "_build_attention_metadata",
                        time.perf_counter() - _t0_attn_meta,
                    )

        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            remove_lora,
            num_active_loras,
        ):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            model_kwargs = self._init_model_kwargs()
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)

                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
                model_kwargs = self._init_model_kwargs()
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions.gpu[:num_tokens_padded]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device,
                        )
                    )

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens_padded, None, False
                )

            if ubatch_slices_padded is not None:
                # Adjust values to reflect a single ubatch.
                # TODO(sage,lucas): this is cruft that should be addressed in
                #  the padding refactor.
                num_tokens_padded = ubatch_slices_padded[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_padded

            with (
                self.maybe_randomize_inputs(input_ids, inputs_embeds),
                set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                    ubatch_slices=ubatch_slices_padded,
                    slot_mapping=slot_mappings,
                ),
            ):
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            if self.speculative_config and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
                or self.speculative_config.uses_extract_hidden_states()
            ):
                assert isinstance(
                    self.drafter,
                    EagleProposer | DraftModelProposer | ExtractHiddenStatesProposer,
                )
                assert self.speculative_config is not None
                # Eagle currently only supports PIECEWISE cudagraphs.
                # Therefore only use cudagraphs if the main model uses PIECEWISE
                # NOTE(lucas): this is a hack, need to clean up.
                use_cudagraphs = (
                    (
                        is_graph_capturing
                        and cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                    )
                    or (
                        not is_graph_capturing
                        and cudagraph_runtime_mode != CUDAGraphMode.NONE
                    )
                ) and not self.speculative_config.enforce_eager

                # Note(gnovack) - We need to disable cudagraphs for one of the two
                # lora cases when cudagraph_specialize_lora is enabled. This is a
                # short term mitigation for issue mentioned in
                # https://github.com/vllm-project/vllm/issues/28334
                if (
                    self.compilation_config.cudagraph_specialize_lora
                    and num_active_loras > 0
                ):
                    use_cudagraphs = False

                self.drafter.dummy_run(
                    num_tokens,
                    use_cudagraphs=use_cudagraphs,
                    is_graph_capturing=is_graph_capturing,
                    slot_mappings=slot_mappings,
                )

        # We register layerwise NVTX hooks here after the first dynamo tracing is
        # done to avoid nvtx operations in hook functions being traced by
        # torch dynamo and causing graph breaks.
        # Note that for DYNAMO_ONCE and VLLM_COMPILE mode,
        # compiled model's dynamo tracing is only done once and the compiled model's
        # __call__ function is replaced by calling the compiled function.
        # So it's safe to register hooks here. Hooks will be registered to
        # both compiled and uncompiled models but they will never
        # be called on the compiled model execution path.
        self._register_layerwise_nvtx_hooks()

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        logit_indices_device = torch.from_numpy(logit_indices).to(
            self.device, non_blocking=True
        )
        return hidden_states, hidden_states[logit_indices_device]

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # The dummy hidden states may contain special values,
        # like `inf` or `nan`.
        # To avoid breaking the sampler, we use a random tensor here instead.

        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # MM Encoder only model no need to run sampler.
            return torch.tensor([])

        hidden_states = torch.rand_like(hidden_states)

        logits = self.model.compute_logits(hidden_states)
        num_reqs = logits.size(0)

        dummy_tensors = lambda v: torch.full((num_reqs,), v, device=self.device)

        dummy_metadata = SamplingMetadata(
            temperature=dummy_tensors(0.5),
            all_greedy=False,
            all_random=False,
            top_p=dummy_tensors(0.9),
            top_k=dummy_tensors(logits.size(1) - 1),
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=dummy_tensors(0.1),
            presence_penalties=dummy_tensors(0.1),
            repetition_penalties=dummy_tensors(0.1),
            output_token_ids=[[] for _ in range(num_reqs)],
            spec_token_ids=[[] for _ in range(num_reqs)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )
        try:
            sampler_output = self.sampler(
                logits=logits, sampling_metadata=dummy_metadata
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up sampler with "
                    f"{num_reqs} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            else:
                raise e
        if self.speculative_config:
            draft_token_ids = [[0] for _ in range(num_reqs)]
            dummy_spec_decode_metadata = SpecDecodeMetadata.make_dummy(
                draft_token_ids, self.device
            )

            num_tokens = sum(len(ids) for ids in draft_token_ids)
            # draft_probs = torch.randn(
            #     num_tokens, logits.shape[-1], device=self.device,
            #     dtype=logits.dtype)
            draft_probs = None
            logits = torch.randn(
                num_tokens + num_reqs,
                logits.shape[-1],
                device=self.device,
                dtype=logits.dtype,
            )
            self.rejection_sampler(
                dummy_spec_decode_metadata,
                draft_probs,
                logits,
                dummy_metadata,
            )
        return sampler_output

    def _dummy_pooler_run_task(
        self,
        hidden_states: torch.Tensor,
        task: PoolingTask,
    ) -> PoolerOutput:
        num_tokens = hidden_states.shape[0]
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_np = np.full(num_reqs, min_tokens_per_req)
        num_scheduled_tokens_np[-1] += num_tokens % num_reqs
        assert np.sum(num_scheduled_tokens_np) == num_tokens
        assert len(num_scheduled_tokens_np) == num_reqs

        req_num_tokens = num_tokens // num_reqs

        dummy_prompt_lens = torch.from_numpy(num_scheduled_tokens_np)
        dummy_token_ids = torch.zeros(
            (num_reqs, req_num_tokens), dtype=torch.int32, device=self.device
        )

        model = cast(VllmModelForPooling, self.get_model())
        dummy_pooling_params = PoolingParams(task=task)
        dummy_pooling_params.verify(self.model_config)
        to_update = model.pooler.get_pooling_updates(task)
        to_update.apply(dummy_pooling_params)

        dummy_metadata = PoolingMetadata(
            prompt_lens=dummy_prompt_lens,
            prompt_token_ids=dummy_token_ids,
            pooling_params=[dummy_pooling_params] * num_reqs,
            pooling_states=[PoolingStates() for i in range(num_reqs)],
        )

        dummy_metadata.build_pooling_cursor(
            num_scheduled_tokens_np,
            seq_lens_cpu=dummy_prompt_lens,
            device=hidden_states.device,
        )

        try:
            return model.pooler(
                hidden_states=hidden_states, pooling_metadata=dummy_metadata
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up pooler "
                    f"({task=}) with {num_reqs} dummy requests. Please try "
                    "lowering `max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            else:
                raise e

    @torch.inference_mode()
    def _dummy_pooler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> PoolerOutput:
        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # MM Encoder only model not need to run pooler.
            return torch.tensor([])

        # Find the task that has the largest output for subsequent steps
        supported_pooling_tasks = self.get_supported_pooling_tasks()

        if not supported_pooling_tasks:
            raise RuntimeError(
                f"Model {self.model_config.model} does not support "
                "any pooling tasks. See "
                "https://docs.vllm.ai/en/latest/models/pooling_models.html "
                "to learn more."
            )

        output_size = dict[PoolingTask, float]()
        for task in supported_pooling_tasks:
            # Run a full batch with each task to ensure none of them OOMs
            output = self._dummy_pooler_run_task(hidden_states, task)
            output_size[task] = sum(o.nbytes for o in output if o is not None)
            del output  # Allow GC

        max_task = max(output_size.items(), key=lambda x: x[1])[0]
        return self._dummy_pooler_run_task(hidden_states, max_task)

    def profile_run(self) -> None:
        # Profile with multimodal encoder & encoder cache.
        if self.supports_mm_inputs:
            mm_config = self.model_config.multimodal_config
            if mm_config is not None and mm_config.skip_mm_profiling:
                logger.info(
                    "Skipping memory profiling for multimodal encoder and "
                    "encoder cache."
                )
            else:
                mm_budget = self.mm_budget
                assert mm_budget is not None

                if (encoder_budget := mm_budget.get_encoder_budget()) > 0:
                    if not mm_budget.mm_max_toks_per_item:
                        # All modality limits are 0 — embedding-only mode.
                        # Budget is non-zero for embedding storage, but
                        # there's no encoder to profile.
                        logger.info(
                            "Skipping encoder profiling for embedding-only "
                            "mode (all modality limits=0 with "
                            "enable_mm_embeds=True).",
                        )
                    else:
                        # NOTE: Currently model is profiled with a single
                        # non-text modality with the max possible input
                        # tokens even when it supports multiple.
                        dummy_modality = mm_budget.get_modality_with_max_tokens()
                        max_mm_items_per_batch = mm_budget.mm_max_items_per_batch[
                            dummy_modality
                        ]

                        logger.info_once(
                            "Encoder cache will be initialized with a "
                            "budget of %s tokens, and profiled with "
                            "%s %s items of the maximum feature size.",
                            encoder_budget,
                            max_mm_items_per_batch,
                            dummy_modality,
                            scope="local",
                        )

                        # Create dummy batch of multimodal inputs.
                        batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                            dummy_modality,
                            max_mm_items_per_batch,
                        )

                        # Run multimodal encoder.
                        dummy_encoder_outputs = self.model.embed_multimodal(
                            **batched_dummy_mm_inputs
                        )

                        sanity_check_mm_encoder_outputs(
                            dummy_encoder_outputs,
                            expected_num_items=max_mm_items_per_batch,
                        )
                        for i, output in enumerate(dummy_encoder_outputs):
                            self.encoder_cache[f"tmp_{i}"] = output

        # Add `is_profile` here to pre-allocate communication buffers
        hidden_states, last_hidden_states = self._dummy_run(
            self.max_num_tokens, is_profile=True
        )
        if get_pp_group().is_last_rank:
            if self.is_pooling_model:
                output = self._dummy_pooler_run(hidden_states)
            else:
                output = self._dummy_sampler_run(last_hidden_states)
        else:
            output = None
        self._sync_device()
        del hidden_states, output
        self.encoder_cache.clear()
        gc.collect()

    def _init_minimal_kv_cache_for_profiling(self) -> None:
        from vllm.v1.core.kv_cache_utils import (
            get_kv_cache_config_from_groups,
            get_kv_cache_groups,
        )

        kv_cache_spec = self.get_kv_cache_spec()
        kv_cache_groups = get_kv_cache_groups(self.vllm_config, kv_cache_spec)
        min_blocks = self.compilation_config.max_cudagraph_capture_size or 1

        # Temporarily change num_gpu_blocks_override to allocate a minimal KV cache
        saved_override = self.cache_config.num_gpu_blocks_override
        self.cache_config.num_gpu_blocks_override = min_blocks
        minimal_config = get_kv_cache_config_from_groups(
            self.vllm_config, kv_cache_groups, available_memory=0
        )
        self.cache_config.num_gpu_blocks_override = saved_override

        self.initialize_kv_cache(minimal_config)
        self.cache_config.num_gpu_blocks = minimal_config.num_blocks

        logger.debug("Initialized minimal KV cache for CUDA graph profiling")

    @staticmethod
    @contextmanager
    def _freeze_gc():
        gc.collect()
        should_freeze = not envs.VLLM_ENABLE_CUDAGRAPH_GC
        if should_freeze:
            gc.freeze()
        try:
            yield
        finally:
            if should_freeze:
                gc.unfreeze()
                gc.collect()

    def _cleanup_profiling_kv_cache(self) -> None:
        torch.accelerator.synchronize()
        if hasattr(self, "kv_caches") and self.kv_caches:
            for i in range(len(self.kv_caches)):
                self.kv_caches[i] = None  # type: ignore
            self.kv_caches.clear()
        if hasattr(self, "cross_layers_kv_cache"):
            self.cross_layers_kv_cache = None
            self.cross_layers_attn_backend = None
        if hasattr(self, "attn_groups"):
            self.attn_groups.clear()
        if hasattr(self, "kv_cache_config"):
            delattr(self, "kv_cache_config")
        self.cache_config.num_gpu_blocks = None

        for layer in self.compilation_config.static_forward_context.values():
            if hasattr(layer, "kv_cache"):
                layer.kv_cache = []

        gc.collect()
        torch.accelerator.empty_cache()

        logger.debug("Cleaned up profiling KV cache and CUDA graphs")

    @torch.inference_mode()
    def profile_cudagraph_memory(self) -> int:
        with set_current_vllm_config(self.vllm_config):
            self._init_minimal_kv_cache_for_profiling()

        saved_num_cudagraph_captured = compilation_counter.num_cudagraph_captured

        capture_descs = self.cudagraph_dispatcher.get_capture_descs()

        total_graphs = sum(len(descs) for _, descs in capture_descs)
        if total_graphs == 0:
            logger.debug("No CUDA graphs will be captured, skipping profiling")
            self._cleanup_profiling_kv_cache()
            return 0

        logger.info(
            "Profiling CUDA graph memory: %s",
            ", ".join(
                f"{mode.name}={len(descs)} (largest={descs[0].num_tokens})"
                for mode, descs in capture_descs
                if descs
            ),
        )

        # Use a temporary pool for profiling to avoid fragmentation in the main pool.
        profiling_pool = current_platform.graph_pool_handle()
        original_pools: dict[int, Any] = {}
        for instance in list(CUDAGraphWrapper._all_instances):
            original_pools[id(instance)] = instance.graph_pool
            instance.graph_pool = profiling_pool

        set_cudagraph_capturing_enabled(True)
        with self._freeze_gc(), graph_capture(device=self.device):
            shared_memory_estimate = {}
            per_graph_estimate = {}
            torch.accelerator.synchronize()
            torch.accelerator.empty_cache()

            for mode, descs in capture_descs:
                profile_descs = descs[:2]
                mem_samples: list[int] = []

                for i, desc in enumerate(profile_descs):
                    mem_before = torch.cuda.mem_get_info()[0]
                    self._warmup_and_capture(
                        desc,
                        cudagraph_runtime_mode=mode,
                        profile_seq_lens=(
                            min(
                                self.max_model_len,
                                self.max_num_tokens // desc.num_tokens,
                            )
                            if mode == CUDAGraphMode.FULL and i == 0
                            else None
                        ),
                    )
                    torch.accelerator.synchronize()
                    free_after = torch.cuda.mem_get_info()[0]
                    mem_samples.append(mem_before - free_after)

                first_capture = mem_samples[0]
                # Use at least 1 MiB per graph for driver overhead
                per_graph = max(mem_samples[1] if len(mem_samples) > 1 else 0, 1 << 20)

                shared_memory_estimate[mode] = first_capture
                per_graph_estimate[mode] = per_graph * (len(descs) - 1)

                logger.debug(
                    "Estimated %s CUDA graph memory: "
                    "%.2f MiB first-capture + (%d-1) × %.2f MiB per-graph",
                    mode.name,
                    first_capture / (1 << 20),
                    len(descs),
                    per_graph / (1 << 20),
                )

        set_cudagraph_capturing_enabled(False)
        CUDAGraphWrapper.clear_all_graphs()
        for instance in list(CUDAGraphWrapper._all_instances):
            if id(instance) in original_pools:
                instance.graph_pool = original_pools[id(instance)]
        for key_set in self.cudagraph_dispatcher.cudagraph_keys.values():
            key_set.clear()
        self.cudagraph_dispatcher.keys_initialized = False
        self.maybe_remove_all_loras(self.lora_config)
        self._cleanup_profiling_kv_cache()
        compilation_counter.num_cudagraph_captured = saved_num_cudagraph_captured

        # FULL and PIECEWISE graphs share the global pool at runtime and are
        # never replayed concurrently, so the pool overlays their memory.
        # Take the max to avoid double-counting the overlap.
        total_estimate = max(shared_memory_estimate.values()) + sum(
            per_graph_estimate.values()
        )
        logger.info(
            "Estimated CUDA graph memory: %.2f GiB total",
            total_estimate / (1 << 30),
        )

        return int(total_estimate)

    @instrument(span_name="Capture model")
    def capture_model(self) -> int:
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        set_cudagraph_capturing_enabled(True)
        with self._freeze_gc(), graph_capture(device=self.device):
            torch.accelerator.synchronize()
            torch.accelerator.empty_cache()
            start_free_gpu_memory = torch.cuda.mem_get_info()[0]

            for (
                runtime_mode,
                batch_descs,
            ) in self.cudagraph_dispatcher.get_capture_descs():
                self._capture_cudagraphs(
                    batch_descriptors=batch_descs,
                    cudagraph_runtime_mode=runtime_mode,
                )
                torch.accelerator.synchronize()

            torch.accelerator.synchronize()
            end_free_gpu_memory = torch.cuda.mem_get_info()[0]

        # Disable cudagraph capturing globally, so any unexpected cudagraph
        # capturing will be detected and raise an error after here.
        # Note: We don't put it into graph_capture context manager because
        # we may do lazy capturing in future that still allows capturing
        # after here.
        set_cudagraph_capturing_enabled(False)

        torch.accelerator.synchronize()
        torch.accelerator.empty_cache()

        # Lock workspace to prevent resizing during execution.
        # Max workspace sizes should have been captured during warmup/profiling.
        lock_workspace()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info_once(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
            scope="local",
        )
        return cuda_graph_size

    def _warmup_and_capture(
        self,
        desc: BatchDescriptor,
        cudagraph_runtime_mode: CUDAGraphMode,
        profile_seq_lens: int | None = None,
        allow_microbatching: bool = False,
        num_warmups: int | None = None,
    ):
        if num_warmups is None:
            num_warmups = self.compilation_config.cudagraph_num_of_warmups
        force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL
        for _ in range(num_warmups):
            self._dummy_run(
                desc.num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                force_attention=force_attention,
                uniform_decode=desc.uniform,
                allow_microbatching=allow_microbatching,
                skip_eplb=True,
                remove_lora=False,
                num_active_loras=desc.num_active_loras,
            )
        self._dummy_run(
            desc.num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            uniform_decode=desc.uniform,
            allow_microbatching=allow_microbatching,
            skip_eplb=True,
            remove_lora=False,
            num_active_loras=desc.num_active_loras,
            is_graph_capturing=True,
            profile_seq_lens=profile_seq_lens,
        )

    def _capture_cudagraphs(
        self,
        batch_descriptors: list[BatchDescriptor],
        cudagraph_runtime_mode: CUDAGraphMode,
    ):
        assert (
            cudagraph_runtime_mode != CUDAGraphMode.NONE
            and cudagraph_runtime_mode.is_valid_runtime_mode()
        ), f"Invalid cudagraph runtime mode: {cudagraph_runtime_mode}"

        if not batch_descriptors:
            return

        uniform_decode = batch_descriptors[0].uniform

        # Only rank 0 should print progress bar during capture
        if is_global_first_rank():
            batch_descriptors = tqdm(
                batch_descriptors,
                disable=not self.load_config.use_tqdm_on_load,
                desc="Capturing CUDA graphs ({}, {})".format(
                    "decode" if uniform_decode else "mixed prefill-decode",
                    cudagraph_runtime_mode.name,
                ),
            )

        # We skip EPLB here since we don't want to record dummy metrics
        for batch_desc in batch_descriptors:
            # We currently only capture ubatched graphs when its a FULL
            # cudagraph, a uniform decode batch, and the number of tokens
            # is above the threshold. Otherwise we just capture a non-ubatched
            # version of the graph
            allow_microbatching = (
                self.parallel_config.use_ubatching
                and cudagraph_runtime_mode == CUDAGraphMode.FULL
                and uniform_decode
                and check_ubatch_thresholds(
                    config=self.vllm_config.parallel_config,
                    num_tokens=batch_desc.num_tokens,
                    uniform_decode=uniform_decode,
                )
            )
            self._warmup_and_capture(
                batch_desc,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                allow_microbatching=allow_microbatching,
            )
            torch.accelerator.synchronize()
        self.maybe_remove_all_loras(self.lora_config)

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_groups) == 0, "Attention backends are already initialized"

        class AttentionGroupKey(NamedTuple):
            attn_backend: type[AttentionBackend]
            kv_cache_spec: KVCacheSpec

        def get_attn_backends_for_group(
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> tuple[dict[AttentionGroupKey, list[str]], set[type[AttentionBackend]]]:
            layer_type = cast(type[Any], AttentionLayerBase)
            layers = get_layers_from_vllm_config(
                self.vllm_config, layer_type, kv_cache_group_spec.layer_names
            )
            attn_backends = {}
            attn_backend_layers = defaultdict(list)
            # Dedupe based on full class name; this is a bit safer than
            # using the class itself as the key because when we create dynamic
            # attention backend subclasses (e.g. ChunkedLocalAttention) unless
            # they are cached correctly, there will be different objects per
            # layer.
            for layer_name in kv_cache_group_spec.layer_names:
                attn_backend = layers[layer_name].get_attn_backend()

                if layer_name in self.kv_sharing_fast_prefill_eligible_layers:
                    attn_backend = create_fast_prefill_custom_backend(
                        "FastPrefill",
                        attn_backend,  # type: ignore[arg-type]
                    )

                full_cls_name = attn_backend.full_cls_name()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
                key = (full_cls_name, layer_kv_cache_spec)
                attn_backends[key] = AttentionGroupKey(
                    attn_backend, layer_kv_cache_spec
                )
                attn_backend_layers[key].append(layer_name)
            return (
                {attn_backends[k]: v for k, v in attn_backend_layers.items()},
                set(group_key.attn_backend for group_key in attn_backends.values()),
            )

        def create_attn_groups(
            attn_backends_map: dict[AttentionGroupKey, list[str]],
            kv_cache_group_id: int,
        ) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for (attn_backend, kv_cache_spec), layer_names in attn_backends_map.items():
                attn_group = AttentionGroup(
                    attn_backend,
                    layer_names,
                    kv_cache_spec,
                    kv_cache_group_id,
                )

                attn_groups.append(attn_group)
            return attn_groups

        attention_backend_maps = []
        attention_backend_list = []
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            attn_backends = get_attn_backends_for_group(kv_cache_group_spec)
            attention_backend_maps.append(attn_backends[0])
            attention_backend_list.append(attn_backends[1])

        # Resolve cudagraph_mode before actually initialize metadata_builders
        self._check_and_update_cudagraph_mode(
            attention_backend_list, kv_cache_config.kv_cache_groups
        )

        # Check if attention backend supports PCP&DCP and related features.
        check_attention_cp_compatibility(self.vllm_config)

        for i, attn_backend_map in enumerate(attention_backend_maps):
            self.attn_groups.append(create_attn_groups(attn_backend_map, i))

    def initialize_metadata_builders(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """
        Create the metadata builders for all KV cache groups and attn groups.
        """
        for kv_cache_group_id in range(len(kv_cache_config.kv_cache_groups)):
            for attn_group in self.attn_groups[kv_cache_group_id]:
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_sizes[kv_cache_group_id]
                    if kv_cache_group_id < len(kernel_block_sizes)
                    else None,
                    num_metadata_builders=1
                    if not self.parallel_config.use_ubatching
                    else self.parallel_config.num_ubatches,
                )
        # Calculate reorder batch threshold (if needed)
        # Note (tdoublep): do this *after* constructing builders,
        # because some of them change the threshold at init time.
        self.calculate_reorder_batch_threshold()

        # Initialize drafter attention backend
        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
        ):
            assert isinstance(self.drafter, EagleProposer | DraftModelProposer)
            self.drafter.initialize_attn_backend(kv_cache_config, kernel_block_sizes)

    def _check_and_update_cudagraph_mode(
        self,
        attention_backends: list[set[type[AttentionBackend]]],
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> None:
        """
        Resolve the cudagraph_mode when there are multiple attention
        groups with potential conflicting CUDA graph support.
        Then initialize the cudagraph_dispatcher based on the resolved
        cudagraph_mode.
        """
        min_cg_support = AttentionCGSupport.ALWAYS
        min_cg_backend_name = None

        for attn_backend_set, kv_cache_group in zip(
            attention_backends, kv_cache_groups
        ):
            for attn_backend in attn_backend_set:
                builder_cls = attn_backend.get_builder_cls()

                cg_support = builder_cls.get_cudagraph_support(
                    self.vllm_config, kv_cache_group.kv_cache_spec
                )
                if cg_support.value < min_cg_support.value:
                    min_cg_support = cg_support
                    min_cg_backend_name = attn_backend.__name__
        # Flexible resolve the cudagraph mode
        cudagraph_mode = self.compilation_config.cudagraph_mode
        assert cudagraph_mode is not None
        # check cudagraph for mixed batch is supported
        if (
            cudagraph_mode.mixed_mode() == CUDAGraphMode.FULL
            and min_cg_support != AttentionCGSupport.ALWAYS
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                f"with {min_cg_backend_name} backend (support: "
                f"{min_cg_support})"
            )
            if min_cg_support == AttentionCGSupport.NEVER:
                # if not supported any full cudagraphs, just raise it.
                msg += (
                    "; please try cudagraph_mode=PIECEWISE, and "
                    "make sure compilation mode is VLLM_COMPILE"
                )
                raise ValueError(msg)

            # attempt to resolve the full cudagraph related mode
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=FULL_AND_PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL_AND_PIECEWISE
                )
            else:
                msg += "; setting cudagraph_mode=FULL_DECODE_ONLY"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL_DECODE_ONLY
                )
            logger.warning(msg)

        # check that if we are doing decode full-cudagraphs it is supported
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and min_cg_support == AttentionCGSupport.NEVER
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                f"with {min_cg_backend_name} backend (support: "
                f"{min_cg_support})"
            )
            if self.compilation_config.mode == CompilationMode.VLLM_COMPILE and (
                self.compilation_config.splitting_ops_contain_attention()
                or self.compilation_config.use_inductor_graph_partition
            ):
                msg += (
                    "; setting cudagraph_mode=PIECEWISE because "
                    "attention is compiled piecewise"
                )
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.PIECEWISE
                )
            else:
                msg += (
                    "; setting cudagraph_mode=NONE because "
                    "attention is not compiled piecewise"
                )
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.NONE
                )
            logger.warning(msg)

        # check that if we are doing spec-decode + decode full-cudagraphs it is
        # supported
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and self.uniform_decode_query_len > 1
            and min_cg_support.value < AttentionCGSupport.UNIFORM_BATCH.value
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported"
                f" with spec-decode for attention backend "
                f"{min_cg_backend_name} (support: {min_cg_support})"
            )
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.PIECEWISE
                )
            else:
                msg += "; setting cudagraph_mode=NONE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.NONE
                )
            logger.warning(msg)

        # double check that we can support full cudagraph if they are requested
        # even after automatic downgrades
        if (
            cudagraph_mode.has_full_cudagraphs()
            and min_cg_support == AttentionCGSupport.NEVER
        ):
            raise ValueError(
                f"CUDAGraphMode.{cudagraph_mode.name} is not "
                f"supported with {min_cg_backend_name} backend ("
                f"support:{min_cg_support}) "
                "; please try cudagraph_mode=PIECEWISE, "
                "and make sure compilation mode is VLLM_COMPILE"
            )

        # if we have dedicated decode cudagraphs, and spec-decode is enabled,
        # we need to adjust the cudagraph sizes to be a multiple of the uniform
        # decode query length to avoid: https://github.com/vllm-project/vllm/issues/28207
        # temp-fix: https://github.com/vllm-project/vllm/issues/28207#issuecomment-3504004536
        # Will be removed in the near future when we have separate cudagraph capture
        # sizes for decode and mixed prefill-decode.
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and cudagraph_mode.separate_routine()
            and self.uniform_decode_query_len > 1
        ):
            self.compilation_config.adjust_cudagraph_sizes_for_spec_decode(
                self.uniform_decode_query_len, self.parallel_config.tensor_parallel_size
            )

        # If the model has Mamba layers and cudagraph mode includes FULL
        # decode, cap cudagraph capture sizes to the number of available
        # Mamba cache blocks. Each decode request needs one conv_state
        # cache line, so capture batch sizes cannot exceed num_blocks.
        # Only FULL decode graphs are affected because PIECEWISE captures
        # run GDN/Mamba ops eagerly (prefill path, no causal_conv1d_update).
        # See: https://github.com/vllm-project/vllm/issues/34094
        if cudagraph_mode.has_full_cudagraphs():
            has_mamba = any(
                isinstance(g.kv_cache_spec, MambaSpec) for g in kv_cache_groups
            )
            if has_mamba and self.kv_cache_config is not None:
                self.compilation_config.adjust_cudagraph_sizes_for_mamba_cache(
                    self.kv_cache_config.num_blocks
                )

        # Trigger cudagraph dispatching keys initialization after
        # resolved cudagraph mode.
        self.compilation_config.cudagraph_mode = cudagraph_mode
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            cudagraph_mode, self.uniform_decode_query_len
        )

        # Initialize drafter's cudagraph dispatcher if using spec decode.
        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_extract_hidden_states()
        ):
            assert isinstance(self.drafter, EagleProposer | ExtractHiddenStatesProposer)
            self.drafter.initialize_cudagraph_keys(cudagraph_mode)

    def calculate_reorder_batch_threshold(self) -> None:
        """
        Choose the minimum reorder batch threshold from all attention groups.
        Backends should be able to support lower threshold then what they request
        just may have a performance penalty due to that backend treating decodes
        as prefills.
        """
        min_none_high = lambda a, b: a if b is None else b if a is None else min(a, b)

        reorder_batch_thresholds: list[int | None] = [
            group.get_metadata_builder().reorder_batch_threshold
            for group in self._attn_group_iterator()
        ]
        # If there are no attention groups (attention-free model) or no backend
        # reports a threshold, leave reordering disabled.
        if len(reorder_batch_thresholds) == 0:
            self.reorder_batch_threshold = None
            return
        self.reorder_batch_threshold = reduce(min_none_high, reorder_batch_thresholds)  # type: ignore[assignment]

    def may_reinitialize_input_batch(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        what it was originally created with. This happens when the final
        block size (determined after model loading) differs from the
        placeholder used during __init__, or when there are multiple
        KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        """
        block_sizes = []
        max_num_blocks = []
        max_model_len = max(self.max_model_len, self.max_encoder_len)
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            if isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            block_size = kv_cache_group.kv_cache_spec.block_size
            block_sizes.append(block_size)
            max_num_blocks_per_req = cdiv(
                max_model_len, block_size * get_total_cp_world_size()
            )
            if isinstance(kv_cache_group.kv_cache_spec, MambaSpec):
                max_num_blocks_per_req = (
                    max_num_blocks_per_req
                    if self.cache_config.enable_prefix_caching
                    else 1
                ) + kv_cache_group.kv_cache_spec.num_speculative_blocks
            max_num_blocks.append(max_num_blocks_per_req)

        if (
            block_sizes != self._init_block_sizes
            or kernel_block_sizes != self._init_kernel_block_sizes
        ):
            assert self.offload_config.uva.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details."
            )
            self._init_block_sizes = block_sizes
            self._init_kernel_block_sizes = kernel_block_sizes
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                kernel_block_sizes=kernel_block_sizes,
                max_num_blocks_per_req=max_num_blocks,
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self.input_batch.logitsprocs,
                logitsprocs_need_output_token_ids=self.input_batch.logitsprocs_need_output_token_ids,
                is_pooling_model=self.is_pooling_model,
            )

        assert self._init_block_sizes == block_sizes, (
            f"InputBatch block_sizes {self._init_block_sizes} != "
            f"kv_cache block_sizes {block_sizes}"
        )
        assert self._init_kernel_block_sizes == kernel_block_sizes, (
            f"InputBatch kernel_block_sizes {self._init_kernel_block_sizes} "
            f"!= kv_cache kernel_block_sizes {kernel_block_sizes}"
        )

    def _allocate_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig
    ) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            tensor = torch.zeros(
                kv_cache_tensor.size, dtype=torch.int8, device=self.device
            )
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys()), (
            "Some layers are not correctly initialized"
        )
        return kv_cache_raw_tensors

    def _attn_group_iterator(self) -> Iterator[AttentionGroup]:
        return itertools.chain.from_iterable(self.attn_groups)

    def _kv_cache_spec_attn_group_iterator(self) -> Iterator[AttentionGroup]:
        if not self.kv_cache_config.kv_cache_groups:
            return
        for attn_groups in self.attn_groups:
            yield from attn_groups

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
                correct size but uninitialized shape.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_caches: dict[str, torch.Tensor] = {}
        has_attn, has_mamba = False, False
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            if group.kv_cache_group_id == len(kernel_block_sizes):
                # There may be a last group for layers without kv cache.
                continue
            kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):
                    has_attn = True
                    num_blocks_per_kv_block = (
                        kv_cache_spec.block_size // kernel_block_size
                    )
                    kernel_num_blocks = num_blocks * num_blocks_per_kv_block

                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        kernel_num_blocks,
                        kernel_block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        cache_dtype_str=self.cache_config.cache_dtype,
                    )
                    dtype = kv_cache_spec.dtype
                    try:
                        kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
                        assert len(kv_cache_stride_order) == len(kv_cache_shape)
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(range(len(kv_cache_shape)))
                    # The allocation respects the backend-defined stride order
                    # to ensure the semantic remains consistent for each
                    # backend. We first obtain the generic kv cache shape and
                    # then permute it according to the stride order which could
                    # result in a non-contiguous tensor.
                    kv_cache_shape = tuple(
                        kv_cache_shape[i] for i in kv_cache_stride_order
                    )
                    # Maintain original KV shape view.
                    inv_order = [
                        kv_cache_stride_order.index(i)
                        for i in range(len(kv_cache_stride_order))
                    ]
                    kv_caches[layer_name] = (
                        kv_cache_raw_tensors[layer_name]
                        .view(dtype)
                        .view(kv_cache_shape)
                        .permute(*inv_order)
                    )
                elif isinstance(kv_cache_spec, MambaSpec):
                    has_mamba = True
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    state_tensors = []
                    storage_offset_bytes = 0
                    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                        dtype_size = get_dtype_size(dtype)
                        num_element_per_page = (
                            kv_cache_spec.page_size_bytes // dtype_size
                        )
                        target_shape = (num_blocks, *shape)
                        stride = torch.empty(target_shape).stride()
                        target_stride = (num_element_per_page, *stride[1:])
                        assert storage_offset_bytes % dtype_size == 0
                        tensor = torch.as_strided(
                            raw_tensor.view(dtype),
                            size=target_shape,
                            stride=target_stride,
                            storage_offset=storage_offset_bytes // dtype_size,
                        )
                        state_tensors.append(tensor)
                        storage_offset_bytes += stride[0] * dtype_size

                    kv_caches[layer_name] = state_tensors
                else:
                    raise NotImplementedError

        if has_attn and has_mamba:
            self._update_hybrid_attention_mamba_layout(kv_caches)

        return kv_caches

    def _update_hybrid_attention_mamba_layout(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> None:
        """
        Update the layout of attention layers from (2, num_blocks, ...) to
        (num_blocks, 2, ...).

        Args:
            kv_caches: The KV cache buffer of each layer.
        """

        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                kv_cache = kv_caches[layer_name]
                if isinstance(kv_cache_spec, AttentionSpec) and kv_cache.shape[0] == 2:
                    assert kv_cache.shape[1] != 2, (
                        "Fail to determine whether the layout is "
                        "(2, num_blocks, ...) or (num_blocks, 2, ...) for "
                        f"a tensor of shape {kv_cache.shape}"
                    )
                    hidden_size = kv_cache.shape[2:].numel()
                    kv_cache.as_strided_(
                        size=kv_cache.shape,
                        stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),
                    )

    def initialize_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
            kernel_block_sizes: The kernel block sizes for each KV cache group.

        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """

        # Try creating KV caches optimized for kv-connector transfers
        cache_dtype = self.cache_config.cache_dtype
        if self.use_uniform_kv_cache(self.attn_groups, cache_dtype):
            kv_caches, cross_layers_kv_cache, attn_backend = (
                self.allocate_uniform_kv_caches(
                    kv_cache_config,
                    self.attn_groups,
                    cache_dtype,
                    self.device,
                    kernel_block_sizes,
                )
            )
            self.cross_layers_kv_cache = cross_layers_kv_cache
            self.cross_layers_attn_backend = attn_backend
        else:
            # Fallback to the general case
            # Initialize the memory buffer for KV cache
            kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)

            # Change the memory buffer to the desired shape
            kv_caches = self._reshape_kv_cache_tensors(
                kv_cache_config, kv_cache_raw_tensors, kernel_block_sizes
            )

        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        num_attn_module = (
            2 if self.model_config.hf_config.model_type == "longcat_flash" else 1
        )
        bind_kv_cache(
            kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
            num_attn_module,
        )
        return kv_caches

    def maybe_add_kv_sharing_layers_to_kv_cache_groups(
        self, kv_cache_config: KVCacheConfig
    ) -> None:
        """
        Add layers that re-use KV cache to KV cache group of its target layer.
        Mapping of KV cache tensors happens in `initialize_kv_cache_tensors()`
        """
        if not self.shared_kv_cache_layers:
            # No cross-layer KV sharing, return
            return

        add_kv_sharing_layers_to_kv_cache_groups(
            self.shared_kv_cache_layers,
            kv_cache_config.kv_cache_groups,
            self.runner_only_attn_layers,
        )

        if self.cache_config.kv_sharing_fast_prefill:
            # In You Only Cache Once (https://arxiv.org/abs/2405.05254) or other
            # similar KV sharing setups, only the layers that generate KV caches
            # are involved in the prefill phase, enabling prefill to early exit.
            attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
            for layer_name in reversed(attn_layers):
                if layer_name in self.shared_kv_cache_layers:
                    self.kv_sharing_fast_prefill_eligible_layers.add(layer_name)
                else:
                    break

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self._mamba_copy_bufs = None
        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        self.initialize_attn_backend(kv_cache_config)
        # The kernel block size for all KV cache groups. For example, if
        # kv_cache_manager uses block_size 256 for a given group, but the attention
        # backends for that group only supports block_size 64, we will return
        # kernel_block_size 64 and split the 256-token-block to 4 blocks with 64
        # tokens each.
        kernel_block_sizes = prepare_kernel_block_sizes(
            kv_cache_config, self.attn_groups
        )
        self._kernel_block_sizes = kernel_block_sizes

        # create metadata builders
        self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)

        # Reinitialize need to after initialize_attn_backend
        self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)
        kv_caches = self.initialize_kv_cache_tensors(
            kv_cache_config, kernel_block_sizes
        )

        if (
            self.speculative_config
            and self.speculative_config.uses_extract_hidden_states()
        ):
            assert isinstance(self.drafter, ExtractHiddenStatesProposer)
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        if has_kv_transfer_group():
            kv_transfer_group = get_kv_transfer_group()
            if self.cross_layers_kv_cache is not None:
                assert self.cross_layers_attn_backend is not None
                kv_transfer_group.register_cross_layers_kv_cache(
                    self.cross_layers_kv_cache, self.cross_layers_attn_backend
                )
            else:
                kv_transfer_group.register_kv_caches(kv_caches)
            kv_transfer_group.set_host_xfer_buffer_ops(copy_kv_blocks)

        # Register sparse Q-capture hooks if any sparse attention groups exist.
        self._setup_sparse_attention()

    def _get_attention_kv_cache_gid(self) -> int:
        """Find the KV cache group index for attention layers."""
        for gid, group in enumerate(self.kv_cache_config.kv_cache_groups):
            if isinstance(group.kv_cache_spec, AttentionSpec):
                return gid
        return 0

    # ── Sparse KV attention helpers ───────────────────────────────────────────

    def _setup_sparse_attention(self) -> None:
        """Register forward pre-hooks on sparse attention layers to capture Q.

        Called once at the end of ``initialize_kv_cache``.  For each attention
        layer that belongs to a ``SparseAttentionSpec`` KV-cache group, a
        ``register_forward_pre_hook`` is attached.  The hook stores the raw
        query tensor so that ``_collect_sparse_features`` can later read it
        without an extra GPU→CPU copy.

        The hooks write into ``self._sparse_q_captures`` unconditionally on
        every forward pass.  This means the writes are captured by CUDA graphs
        in the usual way (they are part of the static graph).  We only *read*
        from ``_sparse_q_captures`` when sparse features are needed.
        """
        if not hasattr(self, "kv_cache_config"):
            return
        sparse_group_count = 0
        hooked_layers = 0
        for gid, group in enumerate(self.kv_cache_config.kv_cache_groups):
            if not isinstance(group.kv_cache_spec, SparseAttentionSpec):
                continue
            sparse_group_count += 1
            self._has_sparse_attn = True
            for layer_name in group.layer_names:
                self._sparse_layer_spec_by_name[layer_name] = group.kv_cache_spec
                self._sparse_layer_gid_by_name[layer_name] = gid
                attn_mod = self.compilation_config.static_forward_context.get(
                    layer_name
                )
                if attn_mod is None:
                    continue

                def _make_hook(name: str):
                    def _hook(
                        module: torch.nn.Module, args: tuple
                    ) -> None:
                        # args[0] is *query*; shape is either
                        # [num_tokens, num_q_heads * head_size] (2-D) or
                        # [num_tokens, num_q_heads, head_size] (3-D).
                        if args:
                            q = args[0]
                            self._sparse_q_captures[name] = q
                            spec = self._sparse_layer_spec_by_name.get(name)
                            if (
                                spec is None
                                or spec.cluster_granularity != "token"
                                or not spec.use_compact_kv_gather
                                or self.dcp_world_size > 1
                            ):
                                setattr(
                                    module,
                                    "_vllm_sparse_runtime_q_head_gather",
                                    None,
                                )
                                setattr(
                                    module,
                                    "_vllm_sparse_runtime_retroinfer",
                                    None,
                                )
                                return
                            try:
                                if q.is_cuda and torch.cuda.is_current_stream_capturing():
                                    setattr(
                                        module,
                                        "_vllm_sparse_runtime_q_head_gather",
                                        None,
                                    )
                                    setattr(
                                        module,
                                        "_vllm_sparse_runtime_retroinfer",
                                        None,
                                    )
                                    return
                            except Exception:
                                pass
                            # Retroinfer fast path is the default under
                            # ``cluster_granularity == "token"`` once the
                            # Phase 1–6 code is in.  ``VLLM_SPARSE_LEGACY_TOKEN_TOPK=1``
                            # forces the legacy per-token topk + compact
                            # gather path for regression bisection.  The
                            # retroinfer builder returns ``None`` when its
                            # own fast-path checks don't hold (num_reqs!=1,
                            # non-GQA-regular layout, prefill step, etc.),
                            # in which case we cleanly fall back to the
                            # legacy builder without running retroinfer.
                            retro: dict | None = None
                            if not self._sparse_legacy_token_topk:
                                _t_retro = (
                                    time.perf_counter()
                                    if _SPARSE_DECODE_STEP_TRACE
                                    else None
                                )
                                retro = self._build_sparse_runtime_retroinfer(
                                    layer_name=name,
                                    query=q,
                                    spec=spec,
                                )
                                if _t_retro is not None:
                                    self._sparse_decode_trace_retro_ms += (
                                        time.perf_counter() - _t_retro
                                    ) * 1000.0
                                    self._sparse_decode_trace_retro_calls += 1
                            if retro is not None:
                                setattr(
                                    module,
                                    "_vllm_sparse_runtime_retroinfer",
                                    retro,
                                )
                                # Suppress the legacy runtime handle for
                                # this step so FA's dispatch selects the
                                # retroinfer exec_buf path unambiguously.
                                setattr(
                                    module,
                                    "_vllm_sparse_runtime_q_head_gather",
                                    None,
                                )
                            else:
                                setattr(
                                    module,
                                    "_vllm_sparse_runtime_retroinfer",
                                    None,
                                )
                                _t_qh = (
                                    time.perf_counter()
                                    if _SPARSE_DECODE_STEP_TRACE
                                    else None
                                )
                                q_head_gather = (
                                    self._build_sparse_runtime_q_head_gather(
                                        layer_name=name,
                                        query=q,
                                        spec=spec,
                                    )
                                )
                                if _t_qh is not None:
                                    self._sparse_decode_trace_qhead_ms += (
                                        time.perf_counter() - _t_qh
                                    ) * 1000.0
                                    self._sparse_decode_trace_qhead_calls += 1
                                    if q_head_gather is None:
                                        self._sparse_decode_trace_qhead_none += 1
                                setattr(
                                    module,
                                    "_vllm_sparse_runtime_q_head_gather",
                                    q_head_gather,
                                )

                    return _hook

                handle = attn_mod.register_forward_pre_hook(
                    _make_hook(layer_name)
                )
                self._sparse_q_hooks.append(handle)
                hooked_layers += 1
        if self._sparse_probe_info_enabled:
            logger.info(
                "[SparseProbe] setup_sparse_attention has_sparse_attn=%s "
                "sparse_groups=%d hooked_layers=%d",
                self._has_sparse_attn,
                sparse_group_count,
                hooked_layers,
            )

    @staticmethod
    def _sparse_online_qh_to_kv_index(
        qh_idx: int, num_q: int, num_kv: int
    ) -> int:
        if num_kv <= 1:
            return 0
        q_per_kv = max(1, num_q // num_kv)
        return min(qh_idx // q_per_kv, num_kv - 1)

    def _sparse_online_select_tokens(
        self,
        state: _SparseOnlineLayerState,
        q: torch.Tensor,
        total_tokens: int,
        spec: SparseAttentionSpec,
        budget_override: int | None = None,
    ) -> torch.Tensor:
        """Return ``[num_q_heads, total_tokens]`` bool selection mask.

        Rewrite notes (behaviour must remain bitwise-equivalent to the legacy
        implementation):

        * Removed three GPU→CPU sync points:

          - the ``(combined_counts <= budget).all().item()`` early-return,
          - the ``steady_mask.sum().item()`` steady count,
          - the ``nonsteady_retr_mask.any().item()`` guard.

          The early-return is equivalent to the top-k path with ``k = cap``:
          when ``combined_count_h <= budget`` we have ``nonsteady_retr_h <=
          cap``, and ``torch.topk(masked_scores, k=cap)`` returns all valid
          non-steady retrieve tokens (the remaining ``-inf`` padding is
          filtered by ``isfinite``), so ``steady | selected_nonsteady`` equals
          ``combined_mask`` in that case.  The empty-retrieve guard is
          similarly subsumed by the ``isfinite`` filter.

        * ``steady_count`` is now computed from ``spec`` + ``total_tokens``
          with plain Python int math (no tensor ``.sum()`` involved).

        * ``labels`` is taken as a direct view of ``state.block_to_cluster``
          (already int64 on the correct device) instead of an extra ``.to()``.
        """
        _t0 = time.perf_counter() if self._sparse_perf_stats_enabled else None
        try:
            if q.dim() == 1:
                q = q.unsqueeze(0)
            device = q.device
            H = int(q.shape[0])
            N = int(total_tokens)
            budget = (
                int(spec.sparse_selection_budget())
                if budget_override is None
                else int(budget_override)
            )
            if budget <= 0:
                return torch.zeros((H, N), dtype=torch.bool, device=device)

            head_n = min(int(spec.static_pattern_start), N)
            tail_start = max(0, N - int(spec.static_pattern_end))
            if head_n >= tail_start:
                # Head and tail overlap: every token in [0, N) is steady.
                steady_count = N
            else:
                steady_count = head_n + (N - tail_start)
            steady_mask = torch.zeros(N, dtype=torch.bool, device=device)
            if head_n > 0:
                steady_mask[:head_n] = True
            if tail_start < N:
                steady_mask[tail_start:] = True
            steady_row = steady_mask.unsqueeze(0)

            indexed_tokens = min(N, int(state.block_to_cluster.shape[0]))

            if (state.cluster_centres.numel() == 0
                    or state.block_to_cluster.numel() == 0):
                fallback_start = max(0, N - budget)
                fallback_mask = torch.zeros(N, dtype=torch.bool, device=device)
                fallback_mask[fallback_start:N] = True
                return fallback_mask.unsqueeze(0) | steady_row

            scores_c = torch.matmul(
                q.to(dtype=torch.float32), state.cluster_centres.transpose(0, 1)
            ) / float(max(int(q.shape[-1]), 1)) ** 0.5
            nprobe = min(int(spec.nprobe), int(state.cluster_centres.shape[0]))
            if nprobe <= 0:
                return steady_row.expand(H, -1).contiguous()
            top_clusters = torch.topk(scores_c, k=nprobe, dim=-1).indices.to(
                dtype=torch.int64
            )
            labels = state.block_to_cluster[:indexed_tokens]
            if labels.dtype != torch.int64:
                labels = labels.to(dtype=torch.int64)
            if labels.numel() == 0 or indexed_tokens <= 0:
                return steady_row.expand(H, -1).contiguous()
            cluster_mask = torch.zeros(
                (H, int(state.cluster_centres.shape[0])),
                dtype=torch.bool,
                device=device,
            )
            cluster_mask.scatter_(1, top_clusters, True)
            retr_mask_indexed = cluster_mask[:, labels]  # [H, indexed_tokens]
            retr_mask = torch.zeros((H, N), dtype=torch.bool, device=device)
            retr_mask[:, :indexed_tokens] = retr_mask_indexed

            cap = max(0, budget - steady_count)
            if cap <= 0:
                return steady_row.expand(H, -1).contiguous()

            nonsteady_retr_mask = retr_mask & ~steady_row

            token_scores_indexed = scores_c.gather(
                1, labels.unsqueeze(0).expand(H, indexed_tokens)
            )
            token_scores = torch.full(
                (H, N),
                float("-inf"),
                dtype=token_scores_indexed.dtype,
                device=device,
            )
            token_scores[:, :indexed_tokens] = token_scores_indexed
            masked_scores = token_scores.masked_fill(
                ~nonsteady_retr_mask, float("-inf")
            )
            k = min(cap, N)
            if k <= 0:
                return steady_row.expand(H, -1).contiguous()
            top_vals, top_idx = torch.topk(masked_scores, k=k, dim=1)
            valid = torch.isfinite(top_vals)
            selected_nonsteady = torch.zeros_like(nonsteady_retr_mask)
            selected_nonsteady.scatter_(1, top_idx, valid)
            return steady_row | selected_nonsteady
        finally:
            if _t0 is not None:
                self._sparse_perf_record(
                    "_sparse_online_select_tokens",
                    time.perf_counter() - _t0,
                )

    def _sparse_online_select_tokens_batched(
        self,
        states: "list[_SparseOnlineLayerState]",
        q_list: "list[torch.Tensor]",
        total_tokens: int,
        spec: "SparseAttentionSpec",
        budget_override: int | None = None,
    ) -> "list[torch.Tensor]":
        """Batched variant of ``_sparse_online_select_tokens``.

        All ``(state, q)`` pairs must share the same request (``total_tokens``)
        and ``spec``.  The implementation stacks them along a new batch
        dimension to replace ``G`` copies of the kernel soup with a single
        set of 3D kernels, cutting launch overhead by ~G×.

        Pre-conditions for the fast path (validated inside):

        * ``len(states) == len(q_list)`` and ``G >= 1``.
        * All ``q`` tensors are 2-D with identical ``(H, D)``.
        * All ``state.cluster_centres`` are non-empty with identical ``(K, D)``.
        * All ``state.block_to_cluster`` are non-empty with identical length.

        On pre-condition miss we fall back to the per-state method so behaviour
        stays bitwise-equivalent.  The fast path produces results that match
        ``_sparse_online_select_tokens`` within floating-point rounding on
        ``scores_c`` (same ``matmul`` math, just batched as ``bmm``).

        Returns a list of ``[H, total_tokens]`` bool masks in input order.
        """
        G = len(states)
        if G == 0:
            return []
        if G == 1:
            return [
                self._sparse_online_select_tokens(
                    state=states[0],
                    q=q_list[0],
                    total_tokens=total_tokens,
                    spec=spec,
                    budget_override=budget_override,
                )
            ]

        _t0 = time.perf_counter() if self._sparse_perf_stats_enabled else None
        try:
            # Normalise q to 2-D and check homogeneity; bail on any mismatch.
            q0 = q_list[0]
            if q0.dim() == 1:
                q0 = q0.unsqueeze(0)
            H = int(q0.shape[0])
            D = int(q0.shape[-1])
            device = q0.device
            N = int(total_tokens)
            budget = (
                int(spec.sparse_selection_budget())
                if budget_override is None
                else int(budget_override)
            )

            def _fallback() -> "list[torch.Tensor]":
                return [
                    self._sparse_online_select_tokens(
                        state=s,
                        q=q,
                        total_tokens=N,
                        spec=spec,
                        budget_override=budget,
                    )
                    for s, q in zip(states, q_list, strict=True)
                ]

            if budget <= 0:
                return [
                    torch.zeros((H, N), dtype=torch.bool, device=device)
                    for _ in range(G)
                ]

            q_norm: list[torch.Tensor] = []
            for q in q_list:
                if q.dim() == 1:
                    q = q.unsqueeze(0)
                if int(q.shape[0]) != H or int(q.shape[-1]) != D:
                    return _fallback()
                q_norm.append(q)

            K0 = int(states[0].cluster_centres.shape[0]) \
                if states[0].cluster_centres.numel() > 0 else 0
            N_idx0 = int(states[0].block_to_cluster.shape[0]) \
                if states[0].block_to_cluster.numel() > 0 else 0
            if K0 == 0 or N_idx0 == 0:
                return _fallback()
            for s in states[1:]:
                if (
                    s.cluster_centres.numel() == 0
                    or s.block_to_cluster.numel() == 0
                    or int(s.cluster_centres.shape[0]) != K0
                    or int(s.block_to_cluster.shape[0]) != N_idx0
                    or int(s.cluster_centres.shape[-1]) != D
                ):
                    return _fallback()

            indexed_tokens = min(N, N_idx0)
            if indexed_tokens <= 0:
                return _fallback()

            # ── Steady mask (shared across all G) ─────────────────────────
            head_n = min(int(spec.static_pattern_start), N)
            tail_start = max(0, N - int(spec.static_pattern_end))
            if head_n >= tail_start:
                steady_count = N
            else:
                steady_count = head_n + (N - tail_start)
            steady_mask = torch.zeros(N, dtype=torch.bool, device=device)
            if head_n > 0:
                steady_mask[:head_n] = True
            if tail_start < N:
                steady_mask[tail_start:] = True
            steady_row = steady_mask.view(1, 1, N)  # [1,1,N] broadcastable

            cap = max(0, budget - steady_count)
            nprobe = min(int(spec.nprobe), K0)
            if nprobe <= 0 or cap <= 0:
                full_steady = steady_row.expand(G, H, N).contiguous()
                return list(full_steady.unbind(0))

            # ── Stacked inputs ────────────────────────────────────────────
            q_stack = torch.stack(q_norm, dim=0).to(dtype=torch.float32)
            # [G, H, D]
            centres = torch.stack(
                [s.cluster_centres for s in states], dim=0
            )  # [G, K, D]
            labels = torch.stack(
                [s.block_to_cluster[:indexed_tokens] for s in states], dim=0
            )  # [G, N_idx]
            if labels.dtype != torch.int64:
                labels = labels.to(dtype=torch.int64)

            # ── Cluster scores & top-nprobe ───────────────────────────────
            scores_c = torch.bmm(
                q_stack, centres.transpose(1, 2)
            ) / float(max(D, 1)) ** 0.5  # [G, H, K]
            top_clusters = torch.topk(
                scores_c, k=nprobe, dim=-1
            ).indices.to(dtype=torch.int64)  # [G, H, nprobe]

            cluster_mask = torch.zeros(
                (G, H, K0), dtype=torch.bool, device=device
            )
            cluster_mask.scatter_(2, top_clusters, True)

            # retr_mask[g, h, n] = cluster_mask[g, h, labels[g, n]] for n<idx
            labels_exp = labels.unsqueeze(1).expand(G, H, indexed_tokens)
            retr_mask_indexed = cluster_mask.gather(2, labels_exp)

            retr_mask = torch.zeros(
                (G, H, N), dtype=torch.bool, device=device
            )
            retr_mask[:, :, :indexed_tokens] = retr_mask_indexed

            nonsteady_retr_mask = retr_mask & ~steady_row

            # Per-token retrieval scores (only defined for n<indexed_tokens)
            token_scores_indexed = scores_c.gather(2, labels_exp)
            # [G, H, N_idx]
            token_scores = torch.full(
                (G, H, N),
                float("-inf"),
                dtype=token_scores_indexed.dtype,
                device=device,
            )
            token_scores[:, :, :indexed_tokens] = token_scores_indexed
            masked_scores = token_scores.masked_fill(
                ~nonsteady_retr_mask, float("-inf")
            )

            k = min(cap, N)
            top_vals, top_idx = torch.topk(masked_scores, k=k, dim=-1)
            valid = torch.isfinite(top_vals)
            selected_nonsteady = torch.zeros_like(nonsteady_retr_mask)
            selected_nonsteady.scatter_(2, top_idx, valid)

            out = selected_nonsteady | steady_row  # [G, H, N]
            return list(out.unbind(0))
        finally:
            if _t0 is not None:
                self._sparse_perf_record(
                    "_sparse_online_select_tokens_batched",
                    time.perf_counter() - _t0,
                )

    def _sparse_online_select_tokens_from_clusters_batched(
        self,
        states: "list[_SparseOnlineLayerState]",
        q_list: "list[torch.Tensor]",
        total_tokens: int,
        spec: "SparseAttentionSpec",
        budget_override: int | None = None,
    ) -> "list[torch.Tensor] | None":
        """Select prompt tokens by expanding Triton cluster member lists.

        This fast path is only used by the legacy TOPK=1 compact-gather mode.
        It scores clusters directly as ``Q dot centroid`` and expands the
        selected dense ``clusters`` rows returned by ``segment_k_means_paged``.
        It intentionally does not read ``block_to_cluster`` labels or the CSR
        inverted index.  If the dense member layout is unavailable or stale
        (for example after a dynamic cluster refresh), callers fall back to the
        label-based selector.
        """
        G = len(states)
        if G == 0:
            return []
        if len(q_list) != G:
            return None
        _t0 = time.perf_counter() if self._sparse_perf_stats_enabled else None
        try:
            q0 = q_list[0]
            if q0.dim() == 1:
                q0 = q0.unsqueeze(0)
            H = int(q0.shape[0])
            D = int(q0.shape[-1])
            device = q0.device
            N = int(total_tokens)
            budget = (
                int(spec.sparse_selection_budget())
                if budget_override is None
                else int(budget_override)
            )
            if budget <= 0:
                return [
                    torch.zeros((H, N), dtype=torch.bool, device=device)
                    for _ in range(G)
                ]
            if N <= 0:
                return [
                    torch.zeros((H, 0), dtype=torch.bool, device=device)
                    for _ in range(G)
                ]

            members0 = states[0].cluster_members
            K0 = (
                int(states[0].cluster_centres.shape[0])
                if states[0].cluster_centres.numel() > 0
                else 0
            )
            if (
                K0 == 0
                or members0 is None
                or members0.dim() != 2
                or int(members0.shape[0]) != K0
            ):
                return None
            M0 = int(members0.shape[1])
            if M0 <= 0:
                return None

            q_norm: list[torch.Tensor] = []
            for q in q_list:
                if q.dim() == 1:
                    q = q.unsqueeze(0)
                if int(q.shape[0]) != H or int(q.shape[-1]) != D:
                    return None
                q_norm.append(q)
            for state in states:
                members = state.cluster_members
                if (
                    state.cluster_centres.numel() == 0
                    or int(state.cluster_centres.shape[0]) != K0
                    or int(state.cluster_centres.shape[-1]) != D
                    or int(state.cluster_size.numel()) != K0
                    or members is None
                    or members.dim() != 2
                    or int(members.shape[0]) != K0
                    or int(members.shape[1]) != M0
                ):
                    return None

            head_n = min(int(spec.static_pattern_start), N)
            tail_start = max(0, N - int(spec.static_pattern_end))
            if head_n >= tail_start:
                steady_count = N
            else:
                steady_count = head_n + (N - tail_start)
            steady_mask = torch.zeros(N, dtype=torch.bool, device=device)
            if head_n > 0:
                steady_mask[:head_n] = True
            if tail_start < N:
                steady_mask[tail_start:] = True
            steady_row = steady_mask.unsqueeze(0)

            cap = max(0, budget - steady_count)
            nprobe = min(int(spec.nprobe), K0)
            if nprobe <= 0 or cap <= 0:
                return [
                    steady_row.expand(H, -1).contiguous()
                    for _ in range(G)
                ]

            q_stack = torch.stack(q_norm, dim=0).to(dtype=torch.float32)
            centres = torch.stack(
                [state.cluster_centres for state in states], dim=0
            )
            scores_c = torch.bmm(
                q_stack, centres.transpose(1, 2)
            ) / float(max(D, 1)) ** 0.5
            top_clusters = torch.topk(
                scores_c, k=nprobe, dim=-1
            ).indices.to(dtype=torch.int64)

            slot_idx = torch.arange(M0, dtype=torch.int64, device=device).view(
                1, 1, M0
            )
            out: list[torch.Tensor] = []
            for g, state in enumerate(states):
                cids = top_clusters[g].reshape(-1)
                members = state.cluster_members
                assert members is not None
                member_rows = members.index_select(0, cids).view(
                    H, nprobe, M0
                )
                size_rows = state.cluster_size.index_select(0, cids).view(
                    H, nprobe
                ).to(dtype=torch.int64)

                valid = slot_idx < size_rows.unsqueeze(-1)
                valid &= member_rows >= head_n
                valid &= member_rows < tail_start
                valid &= member_rows < N

                valid_flat = valid.reshape(H, -1)
                if cap < int(valid_flat.shape[1]):
                    rank = valid_flat.to(torch.int32).cumsum(dim=1)
                    valid_flat &= rank <= cap
                pos_flat = member_rows.reshape(H, -1).clamp(0, N - 1)
                selected_i = torch.zeros(
                    (H, N), dtype=torch.int32, device=device
                )
                selected_i.scatter_add_(
                    1,
                    pos_flat.to(dtype=torch.int64),
                    valid_flat.to(dtype=torch.int32),
                )
                out.append((selected_i > 0) | steady_row)
            return out
        finally:
            if _t0 is not None:
                self._sparse_perf_record(
                    "_sparse_online_select_tokens_from_clusters_batched",
                    time.perf_counter() - _t0,
                )

    def _sparse_online_pack_tokens_from_clusters_batched(
        self,
        states: "list[_SparseOnlineLayerState]",
        q_list: "list[torch.Tensor]",
        *,
        seq_len: int,
        prompt_len: int,
        spec: "SparseAttentionSpec",
        select_budget: int,
        bt_row_gpu: torch.Tensor,
        kv_head_ids: torch.Tensor,
        head_offsets_scratch: torch.Tensor | None,
    ) -> (
        "tuple[torch.Tensor, torch.Tensor, torch.Tensor, "
        "torch.Tensor | None, torch.Tensor] | None"
    ):
        """Cluster-member selector fused with compact-KV pack.

        This is the TOPK=1 hot path that avoids both the label reverse lookup
        and the dense ``[H, N]`` selected-mask scan.  It is deliberately gated
        to homogeneous GQA layouts by the caller; any stale/missing dense
        cluster member tensor returns ``None`` and falls back to the mask path.
        """
        G = len(states)
        if G == 0 or len(q_list) != G:
            return None
        _t0 = time.perf_counter() if self._sparse_perf_stats_enabled else None
        try:
            q0 = q_list[0]
            if q0.dim() == 1:
                q0 = q0.unsqueeze(0)
            H = int(q0.shape[0])
            D = int(q0.shape[-1])
            device = q0.device
            if int(kv_head_ids.numel()) != G * H:
                return None

            members0 = states[0].cluster_members
            K0 = (
                int(states[0].cluster_centres.shape[0])
                if states[0].cluster_centres.numel() > 0
                else 0
            )
            if (
                K0 <= 0
                or members0 is None
                or members0.dim() != 2
                or int(members0.shape[0]) != K0
            ):
                return None
            M0 = int(members0.shape[1])
            if M0 <= 0:
                return None

            q_norm: list[torch.Tensor] = []
            for q in q_list:
                if q.dim() == 1:
                    q = q.unsqueeze(0)
                if int(q.shape[0]) != H or int(q.shape[-1]) != D:
                    return None
                q_norm.append(q)

            for state in states:
                members = state.cluster_members
                if (
                    state.cluster_centres.numel() == 0
                    or int(state.cluster_centres.shape[0]) != K0
                    or int(state.cluster_centres.shape[-1]) != D
                    or int(state.cluster_size.numel()) != K0
                    or members is None
                    or members.dim() != 2
                    or int(members.shape[0]) != K0
                    or int(members.shape[1]) != M0
                ):
                    return None

            N = int(prompt_len)
            head_n = min(int(spec.static_pattern_start), N)
            tail_start = max(0, N - int(spec.static_pattern_end))
            all_prompt_steady = head_n >= tail_start
            steady_count = (
                N if all_prompt_steady else head_n + (N - tail_start)
            )
            cap = max(0, int(select_budget) - steady_count)
            nprobe = min(int(spec.nprobe), K0)

            if cap > 0 and nprobe > 0 and not all_prompt_steady:
                q_stack = torch.stack(q_norm, dim=0).to(dtype=torch.float32)
                centres = torch.stack(
                    [state.cluster_centres for state in states], dim=0
                )
                scores_c = torch.bmm(
                    q_stack, centres.transpose(1, 2)
                ) / float(max(D, 1)) ** 0.5
                top_clusters = torch.topk(
                    scores_c, k=nprobe, dim=-1
                ).indices.to(dtype=torch.int64)
            else:
                top_clusters = torch.empty(
                    (G, H, 1), dtype=torch.int64, device=device
                )

            members_g = torch.stack(
                [state.cluster_members for state in states], dim=0
            )
            sizes_g = torch.stack(
                [state.cluster_size for state in states], dim=0
            )
            return sparse_pack_cluster_members_single_req(
                top_clusters,
                members_g,
                sizes_g,
                bt_row_gpu,
                kv_head_ids,
                int(spec.block_size),
                seq_len=int(seq_len),
                prompt_len=int(prompt_len),
                head_n=int(head_n),
                tail_start=int(tail_start),
                select_budget=int(select_budget),
                head_offsets_scratch=head_offsets_scratch,
                perf_record=(
                    self._sparse_perf_record
                    if self._sparse_perf_stats_enabled and self._has_sparse_attn
                    else None
                ),
            )
        finally:
            if _t0 is not None:
                self._sparse_perf_record(
                    "_sparse_online_pack_tokens_from_clusters_batched",
                    time.perf_counter() - _t0,
                )

    def _sparse_online_cluster_exec_runtime_batched(
        self,
        states: "list[_SparseOnlineLayerState]",
        q_list: "list[torch.Tensor]",
        *,
        seq_len: int,
        prompt_len: int,
        spec: "SparseAttentionSpec",
        select_budget: int,
        bt_row_gpu: torch.Tensor,
    ) -> "dict[str, Any] | None":
        """Build cluster-direct runtime for FA-side K/V gather.

        The runner only scores Q against centroids and carries the selected
        cluster member tables forward.  K/V cache reads happen inside the FA
        backend after the current-step KV cache update.
        """
        G = len(states)
        if G == 0 or len(q_list) != G:
            return None
        _t0 = time.perf_counter() if self._sparse_perf_stats_enabled else None
        try:
            q0 = q_list[0]
            if q0.dim() == 1:
                q0 = q0.unsqueeze(0)
            H = int(q0.shape[0])
            D = int(q0.shape[-1])
            device = q0.device

            members0 = states[0].cluster_members
            K0 = (
                int(states[0].cluster_centres.shape[0])
                if states[0].cluster_centres.numel() > 0
                else 0
            )
            if (
                K0 <= 0
                or members0 is None
                or members0.dim() != 2
                or int(members0.shape[0]) != K0
            ):
                return None
            M0 = int(members0.shape[1])
            if M0 <= 0:
                return None

            q_norm: list[torch.Tensor] = []
            for q in q_list:
                if q.dim() == 1:
                    q = q.unsqueeze(0)
                if int(q.shape[0]) != H or int(q.shape[-1]) != D:
                    return None
                q_norm.append(q)

            for state in states:
                members = state.cluster_members
                if (
                    state.cluster_centres.numel() == 0
                    or int(state.cluster_centres.shape[0]) != K0
                    or int(state.cluster_centres.shape[-1]) != D
                    or int(state.cluster_size.numel()) != K0
                    or members is None
                    or members.dim() != 2
                    or int(members.shape[0]) != K0
                    or int(members.shape[1]) != M0
                ):
                    return None

            N = int(prompt_len)
            head_n = min(int(spec.static_pattern_start), N)
            tail_start = max(0, N - int(spec.static_pattern_end))
            all_prompt_steady = head_n >= tail_start
            steady_count = (
                N if all_prompt_steady else head_n + (N - tail_start)
            )
            cap = max(0, int(select_budget) - steady_count)
            nprobe = min(int(spec.nprobe), K0)

            if cap > 0 and nprobe > 0 and not all_prompt_steady:
                q_stack = torch.stack(q_norm, dim=0).to(dtype=torch.float32)
                centres = torch.stack(
                    [state.cluster_centres for state in states], dim=0
                )
                scores_c = torch.bmm(
                    q_stack, centres.transpose(1, 2)
                ) / float(max(D, 1)) ** 0.5
                top_clusters = torch.topk(
                    scores_c, k=nprobe, dim=-1
                ).indices.to(dtype=torch.int64)
            else:
                top_clusters = torch.empty(
                    (G, H, 1), dtype=torch.int64, device=device
                )

            return {
                "cluster_top_clusters": top_clusters,
                "cluster_members": torch.stack(
                    [state.cluster_members for state in states], dim=0
                ),
                "cluster_size": torch.stack(
                    [state.cluster_size for state in states], dim=0
                ),
                "cluster_block_table": bt_row_gpu,
                "cluster_seq_len": int(seq_len),
                "cluster_prompt_len": int(prompt_len),
                "cluster_head_n": int(head_n),
                "cluster_tail_start": int(tail_start),
                "cluster_select_budget": int(select_budget),
                "cluster_block_size": int(spec.block_size),
            }
        finally:
            if _t0 is not None:
                self._sparse_perf_record(
                    "_sparse_online_cluster_exec_runtime_batched",
                    time.perf_counter() - _t0,
                )

    def _sparse_online_select_clusters_batched(
        self,
        states: "list[_SparseOnlineLayerState]",
        q_list: "list[torch.Tensor]",
        spec: "SparseAttentionSpec",
        *,
        estimation_budget: int = 0,
    ) -> "tuple[torch.Tensor, torch.Tensor] | None":
        """Retroinfer-style cluster-level top-k selector.

        Scores ``Q`` against cluster **centroids only** (never per-token), then
        partitions the top ranks into a retrieval zone (exact FA over the
        expanded tokens) and an estimation zone (approximate FA over centroid
        summaries).  This replaces the legacy
        ``_sparse_online_select_tokens[_batched]`` which produced an
        ``[H, N]`` boolean token mask – under retroinfer we never build a
        per-token mask, so the whole mask → nonzero → pack → gather chain
        downstream collapses to "cluster id → CSR expand → paged gather"
        (Phase 4 consumers).

        Args:
            states: per-kv-head online states for one request, length ``G``.
            q_list: matching per-kv-head Q projections ``[H_per_kv, D]``.
            spec: shared sparse-attention spec (we read ``nprobe`` and the
                centroid dim from here).
            estimation_budget: number of clusters to reserve for the
                estimation zone, ranked immediately below the retrieval
                zone.  ``0`` = estimation disabled (Phase 6 wires the
                real default; Phase 9 plumbs it through config).

        Returns:
            ``(retrieval_cids, estimation_cids)`` int32 tensors of shape
            ``[G, H, nprobe_eff]`` and ``[G, H, es_eff]`` respectively, or
            ``None`` if the fast path's homogeneity checks fail (caller
            should fall back to the legacy per-token selector).  The two
            returned ranges are disjoint by construction (``topk`` slice
            at ``[:nprobe]`` vs ``[nprobe:nprobe+es]``).

        Notes:
            * All cluster ids stay on ``q_list[0].device``.
            * Cluster ids are int32 – typical ``K`` is well below 2³¹.
            * Returns empty trailing dim (still int32, shape ``[G, H, 0]``)
              when ``nprobe_eff`` or ``es_eff`` clips to zero, so downstream
              callers do not need to special-case ``None`` vs empty.
        """
        G = len(states)
        if G == 0:
            return None
        _t0 = time.perf_counter() if self._sparse_perf_stats_enabled else None
        try:
            q0 = q_list[0]
            if q0.dim() == 1:
                q0 = q0.unsqueeze(0)
            H = int(q0.shape[0])
            D = int(q0.shape[-1])
            device = q0.device

            # Homogeneity checks – mirror _sparse_online_select_tokens_batched.
            # If any pair mismatches we bail with ``None`` and let the
            # caller route to the legacy token-level selector.  This keeps
            # the cluster path a strict fast path; we never silently cross
            # into slower heterogeneous territory.
            K0 = (
                int(states[0].cluster_centres.shape[0])
                if states[0].cluster_centres.numel() > 0
                else 0
            )
            if K0 == 0:
                return None
            for s in states[1:]:
                if (
                    s.cluster_centres.numel() == 0
                    or int(s.cluster_centres.shape[0]) != K0
                    or int(s.cluster_centres.shape[-1]) != D
                ):
                    return None
            q_norm: list[torch.Tensor] = []
            for q in q_list:
                if q.dim() == 1:
                    q = q.unsqueeze(0)
                if int(q.shape[0]) != H or int(q.shape[-1]) != D:
                    return None
                q_norm.append(q)

            nprobe_eff = max(0, min(int(spec.nprobe), K0))
            es_eff = max(0, min(int(estimation_budget), K0 - nprobe_eff))
            total_k = nprobe_eff + es_eff
            if total_k == 0:
                # Degenerate – caller can synthesise empty outputs itself.
                empty = torch.empty(
                    (G, H, 0), dtype=torch.int32, device=device
                )
                return empty, empty

            # Stacked scoring path.  Using ``bmm`` (one kernel) instead of
            # a per-state ``matmul`` loop collapses G launches to 1 and is
            # a significant win at num_kv_heads=8~32.
            q_stack = torch.stack(q_norm, dim=0).to(dtype=torch.float32)
            # [G, H, D]
            centres = torch.stack(
                [s.cluster_centres for s in states], dim=0
            )  # [G, K, D]
            scores_c = torch.bmm(
                q_stack, centres.transpose(1, 2)
            ) / float(max(D, 1)) ** 0.5  # [G, H, K]

            # Single top-(nprobe+es) call – retrieval vs estimation split
            # is just an index slice on the result; topk's own partial
            # sort within the top-k values guarantees retrieval gets the
            # strictly-higher scores.
            top_idx = torch.topk(
                scores_c, k=total_k, dim=-1
            ).indices.to(dtype=torch.int32)  # [G, H, total_k]

            retrieval_cids = top_idx[..., :nprobe_eff].contiguous()
            estimation_cids = (
                top_idx[..., nprobe_eff:total_k].contiguous()
                if es_eff > 0
                else torch.empty(
                    (G, H, 0), dtype=torch.int32, device=device
                )
            )
            return retrieval_cids, estimation_cids
        finally:
            if _t0 is not None:
                self._sparse_perf_record(
                    "_sparse_online_select_clusters_batched",
                    time.perf_counter() - _t0,
                )

    def _sparse_retroinfer_expand_and_gather_single_req(
        self,
        *,
        states: "list[_SparseOnlineLayerState]",
        retrieval_cids: torch.Tensor,
        estimation_cids: torch.Tensor,
        block_table_row: torch.Tensor,
        kv_cache: "torch.Tensor | tuple[torch.Tensor, torch.Tensor]",
        block_size: int,
        prefill_len: int,
        current_len: int,
        static_pattern_start: int,
        static_pattern_end: int,
        max_budget: int,
    ) -> "dict[str, torch.Tensor]":
        """Phase 4a — retroinfer-style cluster-expand + paged-KV gather.

        Produces the fixed-shape ``exec_buf_k/v`` / ``valid_lengths`` inputs
        that Phase 5's FA pass will consume in place of the legacy
        ``phys``/``slots``/``head_offsets`` per-head compact gather, **plus**
        the estimation-zone centroid / value_sum / cluster_size tensors that
        Phase 6's second FA pass + LSE merge will consume.

        This is the unfused pytorch baseline.  Correctness here matters more
        than speed: it is the reference the Phase 4b Triton
        ``cluster_expand_and_gather`` kernel will be diffed against.  All of
        the slow parts (per-head Python loop, per-cluster ``torch.cat``)
        collapse to a single fused kernel in 4b.

        Assumptions / scope:

        * Single request (``num_reqs == 1``).  The select-clusters fast path
          already gates on GQA-regular + uniform K, so per-group states
          share ``K``, ``D``, and ``cluster_offsets``/``cluster_slots``
          layout shapes (though the contents differ per kv head).
        * GQA-regular layout – q heads within a kv group are contiguous.
          ``retrieval_cids``/``estimation_cids`` are laid out
          ``[G = num_kv_heads, H_per_kv, ...]`` matching Phase 3's output.
        * Paged KV cache shape ``[num_blocks, block_size, num_kv_heads, D]``
          (the dominant vLLM layout).  A 3D fallback is accepted and
          treated as ``num_kv_heads == 1``.
        * ``max_budget`` is the caller-sized capacity of the output buffer.
          It must cover the worst-case retrieval expansion plus steady +
          pending zones (Phase 8 CUDA-Graph wiring derives it from
          ``nprobe * max_cluster_size + static + pending_cap`` so the shape
          is replay-stable).  Excess retrieval tokens are truncated; FA
          never reads past ``valid_lengths[h]`` regardless.

        Output layout:

        * ``exec_buf_k`` / ``exec_buf_v``: ``[H_total, max_budget, D]`` in
          the KV cache's native dtype.  ``H_total = G * H_per_kv`` in q-head
          index order so downstream FA's ``[num_heads, seqlen, D]`` path is
          drop-in.
        * ``valid_lengths``: ``[H_total]`` int32 – FA's ``cache_seqlens`` /
          ``seqused_k`` argument.
        * ``centres_es`` / ``value_sum_es`` / ``cluster_size_es``:
          per-head gathered estimation-zone tensors.  ``cluster_size_es``
          is fp32 so Phase 6 can apply the ``log(cluster_size)`` reweight
          with a single ``torch.log`` op instead of a cast + log chain.

        Position composition (what ends up in ``exec_buf_*[h, :valid_lengths[h]]``):

          1. **Retrieval positions** (cluster-expanded) – ``sum_k sizes[h,k]``
             entries from ``cluster_slots`` via the ``cids → CSR`` lookup.
          2. **Steady zone** – prefill prefix ``[0, static_pattern_start)``
             and suffix ``[N_prefill - static_pattern_end, N_prefill)``.
             Always included; excluded from CSR by Phase 2a so no dup.
          3. **Pending zone** – decoded-so-far tokens
             ``[N_prefill, current_len)``.  Not yet re-clustered (Phase 7
             handles the refresh); always included as a linear tail.

        The ordering mirrors retroinfer so FA's attention weights match
        their reference – retrieval first, steady / pending appended.
        """
        device = retrieval_cids.device
        if retrieval_cids.dim() != 3:
            raise ValueError(
                "retrieval_cids must be [G, H_per_kv, nprobe], got "
                f"{tuple(retrieval_cids.shape)}"
            )
        G, H_per_kv, nprobe = (
            int(retrieval_cids.shape[0]),
            int(retrieval_cids.shape[1]),
            int(retrieval_cids.shape[2]),
        )
        es = int(estimation_cids.shape[-1]) if estimation_cids.numel() else 0
        H_total = G * H_per_kv

        if isinstance(kv_cache, tuple):
            k_cache, v_cache = kv_cache
        else:
            # Legacy stacked ``[2, num_blocks, block_size, H, D]``.
            k_cache = kv_cache[0]
            v_cache = kv_cache[1]
        if k_cache.dim() == 3:
            # Squeezed ``[num_blocks, block_size, D]`` → treat as one kv head.
            k_cache = k_cache.unsqueeze(2)
            v_cache = v_cache.unsqueeze(2)
        head_dim = int(k_cache.shape[-1])
        cache_dtype = k_cache.dtype

        # ── Fixed-shape output buffers ────────────────────────────────────
        # Zero init is safe padding: FA's ``cache_seqlens`` mechanism caps
        # reads at ``valid_lengths[h]``, so the unused tail is never
        # touched.  Keeping pad = 0 (instead of a sentinel) makes Phase 8's
        # CUDA-graph capture less picky about buffer aliasing across steps.
        # Deferred-gather contract: the builder NO LONGER pre-reads K/V
        # from the paged cache here – that was unsafe because the Attention
        # pre-hook runs BEFORE ``unified_kv_cache_update`` writes the
        # current decode step's K/V slot, so any read of the last pending
        # position returned stale bytes.  We build only the per-head flat
        # index arrays here and let ``_forward_retroinfer_exec_buf`` do the
        # actual ``k_cache[...]`` read at FA time, after the cache update.
        valid_lengths = torch.zeros(
            H_total, dtype=torch.int32, device=device
        )
        # Accumulate per-head flat index arrays on CPU first (one Python
        # list per head) and concat at the end – H_total * one int64
        # cumulative concat is cheaper than H_total repeated in-place
        # writes into a pre-sized tensor.
        per_head_phys: list[torch.Tensor] = []
        per_head_slots: list[torch.Tensor] = []
        per_head_kv_token_ids: list[torch.Tensor] = []

        # ── Shared steady + pending position vector ───────────────────────
        # Built once per request and re-used for every head to avoid
        # ``3 * H_total`` tiny ``arange`` launches.  ``torch.cat`` of up to
        # 3 pieces keeps the result contiguous so the downstream gather can
        # do a single ``index_select`` per head.
        head_n = min(int(static_pattern_start), int(prefill_len))
        tail_start = max(
            0, int(prefill_len) - int(static_pattern_end)
        )
        # Order: pending first, then tail, then head.  If the caller-supplied
        # ``max_budget`` ever turns out to be too small, the ``[:max_budget]``
        # truncation below drops from the end; this order preserves the
        # newest output tokens (pending) and the local window (tail) first,
        # and only sacrifices the attention sink (head) on extreme overflow.
        parts: list[torch.Tensor] = []
        if current_len > prefill_len:
            parts.append(torch.arange(
                prefill_len, current_len,
                dtype=torch.int32, device=device,
            ))
        if tail_start < prefill_len:
            parts.append(torch.arange(
                tail_start, prefill_len,
                dtype=torch.int32, device=device,
            ))
        if head_n > 0:
            parts.append(torch.arange(
                0, head_n, dtype=torch.int32, device=device
            ))
        steady_pending = (
            torch.cat(parts) if parts
            else torch.empty(0, dtype=torch.int32, device=device)
        )
        steady_pending_cnt = int(steady_pending.numel())

        # ── Per-head retrieval expansion (Python loop – Phase 4b fuses) ──
        # We do one ``cluster_offsets.index_select`` at the G scope to
        # amortise the K→H_per_kv broadcast, but the inner per-cluster
        # ``cluster_slots[s:e]`` slicing still forces a
        # ``.item()`` per (head, cluster) pair.  For num_kv_heads=8 and
        # nprobe=32 that is 256 syncs per layer which is unacceptable
        # long-term; 4b replaces this with a single fused kernel.
        for g in range(G):
            state = states[g]
            offsets_g = state.cluster_offsets.to(torch.int64)
            slots_g = state.cluster_slots

            # [H_per_kv * nprobe] → starts/ends in one gather.
            cids_flat = retrieval_cids[g].reshape(-1).to(torch.int64)
            starts_flat = offsets_g.index_select(0, cids_flat)
            ends_flat = offsets_g.index_select(0, cids_flat + 1)
            sizes_flat = (ends_flat - starts_flat)
            starts_hn = starts_flat.view(H_per_kv, nprobe)
            sizes_hn = sizes_flat.view(H_per_kv, nprobe)

            # One D2H sync per kv group rather than per (head, cluster).
            # ``sizes_cpu`` / ``starts_cpu`` drive the subsequent Python
            # slice plan without any further GPU roundtrip.
            starts_cpu = starts_hn.to(
                device="cpu", dtype=torch.int64
            ).numpy()
            sizes_cpu = sizes_hn.to(
                device="cpu", dtype=torch.int64
            ).numpy()

            for h in range(H_per_kv):
                h_total = g * H_per_kv + h
                retr_slices: list[torch.Tensor] = []
                retr_cnt = 0
                # Truncate if retrieval alone would overrun the budget
                # minus steady/pending reserve – steady/pending must
                # always fit (they are always-on).
                retr_budget = max(0, max_budget - steady_pending_cnt)
                for k_idx in range(nprobe):
                    n = int(sizes_cpu[h, k_idx])
                    if n <= 0:
                        continue
                    if retr_cnt + n > retr_budget:
                        n = retr_budget - retr_cnt
                        if n <= 0:
                            break
                    s = int(starts_cpu[h, k_idx])
                    retr_slices.append(slots_g[s : s + n])
                    retr_cnt += n
                if retr_slices:
                    retr_positions = torch.cat(retr_slices)
                else:
                    retr_positions = slots_g.new_empty(0)

                # Final position list: retrieval ⊕ steady ⊕ pending.
                if steady_pending_cnt > 0 and retr_cnt > 0:
                    all_positions = torch.cat(
                        [retr_positions, steady_pending]
                    )
                elif steady_pending_cnt > 0:
                    all_positions = steady_pending
                else:
                    all_positions = retr_positions
                total_count = int(all_positions.numel())
                if total_count > max_budget:
                    all_positions = all_positions[:max_budget]
                    total_count = max_budget
                valid_lengths[h_total] = total_count
                if total_count == 0:
                    continue

                # Logical position → physical (block, slot).
                # block_table_row holds int32 phys_block ids; we cast
                # to int64 once to match the advanced-indexing expected
                # dtype and keep the gather on the fast path.
                all64 = all_positions.to(torch.int64)
                if _SPARSE_DEBUG_ASSERT:
                    _sparse_debug_range(
                        "retroinfer logical positions",
                        all64,
                        int(current_len),
                    )
                    _sparse_debug_range(
                        "retroinfer block_idx",
                        all64 // int(block_size),
                        int(block_table_row.numel()),
                    )
                phys_blocks = block_table_row.to(torch.int64) \
                    .index_select(0, all64 // int(block_size))
                slots = all64 % int(block_size)
                if _SPARSE_DEBUG_ASSERT:
                    _sparse_debug_range(
                        "retroinfer physical block ids",
                        phys_blocks,
                        int(k_cache.shape[0]),
                    )
                    _sparse_debug_range(
                        "retroinfer slots",
                        slots,
                        int(k_cache.shape[1]),
                    )

                # Record per-head indices for deferred FA-time gather.
                # ``kv_ids_h`` broadcasts scalar kv-head ``g`` to one entry
                # per position so downstream ``k_cache[phys, slots, kv_ids]``
                # can gather in a single advanced-index kernel (same pattern
                # as the legacy compact gather).
                kv_ids_h = torch.full_like(phys_blocks, int(g))
                per_head_phys.append(phys_blocks)
                per_head_slots.append(slots)
                per_head_kv_token_ids.append(kv_ids_h)
                if _SPARSE_DEBUG_ASSERT:
                    # Sanity: all_positions must be strictly below
                    # current_len; a stale pending range would silently
                    # over-index into uninitialised KV slots.
                    max_pos = int(all_positions.max().item())
                    if max_pos >= int(current_len):
                        raise ValueError(
                            "[SparseDebug] retroinfer position overflow: "
                            f"max={max_pos} current_len={int(current_len)} "
                            f"h_total={h_total}"
                        )

        # ── Estimation zone gather (cheap: K × D per head) ───────────────
        if es > 0:
            centres_es = torch.empty(
                (H_total, es, head_dim),
                dtype=torch.float32, device=device,
            )
            value_sum_es = torch.empty(
                (H_total, es, head_dim),
                dtype=torch.float32, device=device,
            )
            cluster_size_es = torch.empty(
                (H_total, es), dtype=torch.float32, device=device,
            )
            for g in range(G):
                state = states[g]
                est_g = estimation_cids[g].to(torch.int64)  # [H_per_kv, es]
                flat_idx = est_g.reshape(-1)
                # state.cluster_centres: [K, D] fp32
                # state.value_sum: [K, D] fp32
                # state.cluster_size: [K] int32
                c_flat = state.cluster_centres.to(torch.float32) \
                    .index_select(0, flat_idx)
                vs_flat = state.value_sum.to(torch.float32) \
                    .index_select(0, flat_idx)
                sz_flat = state.cluster_size.to(torch.float32) \
                    .index_select(0, flat_idx)
                centres_es[
                    g * H_per_kv : (g + 1) * H_per_kv
                ] = c_flat.view(H_per_kv, es, head_dim)
                value_sum_es[
                    g * H_per_kv : (g + 1) * H_per_kv
                ] = vs_flat.view(H_per_kv, es, head_dim)
                cluster_size_es[
                    g * H_per_kv : (g + 1) * H_per_kv
                ] = sz_flat.view(H_per_kv, es)
        else:
            centres_es = torch.empty(
                (H_total, 0, head_dim),
                dtype=torch.float32, device=device,
            )
            value_sum_es = torch.empty(
                (H_total, 0, head_dim),
                dtype=torch.float32, device=device,
            )
            cluster_size_es = torch.empty(
                (H_total, 0), dtype=torch.float32, device=device,
            )

        # Build flat per-head index arrays for the FA-time gather.  When
        # every head is empty (e.g. zero retrieval + zero steady + zero
        # pending, which shouldn't happen in practice but keeps the
        # downstream kernel launches defensible) we still emit empty
        # tensors with the right dtype/device so the consumer doesn't
        # have to special-case ``None``.
        if per_head_phys:
            flat_phys = torch.cat(per_head_phys).to(torch.int64)
            flat_slots = torch.cat(per_head_slots).to(torch.int64)
            flat_kv_token_ids = torch.cat(per_head_kv_token_ids).to(torch.int64)
        else:
            flat_phys = torch.empty(0, dtype=torch.int64, device=device)
            flat_slots = torch.empty(0, dtype=torch.int64, device=device)
            flat_kv_token_ids = torch.empty(
                0, dtype=torch.int64, device=device
            )
        cu_k = torch.empty(H_total + 1, dtype=torch.int32, device=device)
        cu_k[0] = 0
        cu_k[1:] = torch.cumsum(valid_lengths.to(torch.int64), dim=0).to(
            torch.int32
        )

        return {
            "valid_lengths": valid_lengths,
            "flat_phys": flat_phys,
            "flat_slots": flat_slots,
            "kv_token_ids": flat_kv_token_ids,
            "cu_k": cu_k,
            "centres_es": centres_es,
            "value_sum_es": value_sum_es,
            "cluster_size_es": cluster_size_es,
        }

    @staticmethod
    def _sparse_retroinfer_estimation_attn(
        q: torch.Tensor,
        centres_es: torch.Tensor,
        value_sum_es: torch.Tensor,
        cluster_size_es: torch.Tensor,
        scale: float,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Phase 6 — estimation-zone softmax attention, closed-form.

        Approximates the softmax contribution of the **non-retrieved**
        clusters by treating each cluster as a single super-token with:

        * logit = ``Q · centroid / sqrt(D)``      (one dot-product per cluster)
        * unnormalised value = ``value_sum[c]``   (sum of V across the
          cluster, precomputed at prefill / dynamic refresh)
        * occupancy weight   = ``cluster_size[c]`` (how many real tokens
          this super-token stands in for)

        The closed-form softmax aggregate is then::

            numerator   = Σ_c  exp(logit_c) · value_sum[c]
            denominator = Σ_c  exp(logit_c) · cluster_size[c]
            out = numerator / denominator
            lse = logsumexp(logit_c + log(cluster_size[c]))

        Mirrors retroinfer's fused estimation kernel (their
        ``batch_gemm_softmax`` + estimation epilogue) but as plain torch
        ops – fine for Phase 4a's baseline; Phase 4b fuses this with the
        cluster-expand kernel so the ``Q @ centres`` GEMM stays on-chip.

        ``lse`` is returned in the **same absolute reference frame** as
        FA's own ``softmax_lse`` output (log of the unshifted softmax
        denominator), so the caller's
        :meth:`_sparse_retroinfer_lse_merge` can combine it with the
        retrieval zone's LSE without per-zone re-normalisation.

        Args:
            q: ``[H, D]`` float query vectors (one per query head).
            centres_es: ``[H, es, D]`` per-head estimation-zone centroids.
            value_sum_es: ``[H, es, D]`` matching ``value_sum`` gather.
            cluster_size_es: ``[H, es]`` float cluster occupancies.
            scale: ``1 / sqrt(D)`` multiplier applied to the logits.

        Returns:
            ``(out_est [H, D] fp32, lse_est [H] fp32)``.  Shape carries
            H_total so the caller can feed it straight into the LSE merge
            alongside FA's per-head output.
        """
        if centres_es.shape[1] == 0:
            # No estimation clusters allocated – caller's LSE merge
            # treats ``lse = -inf`` as a zero-weight branch, so the
            # retrieval-only path falls through unchanged.
            H = int(q.shape[0])
            D = int(q.shape[-1])
            device, dtype = q.device, torch.float32
            return (
                torch.zeros((H, D), dtype=dtype, device=device),
                torch.full(
                    (H,), float("-inf"), dtype=dtype, device=device
                ),
            )
        q_f = q.to(dtype=torch.float32)
        c_f = centres_es.to(dtype=torch.float32)
        vs_f = value_sum_es.to(dtype=torch.float32)
        sz_f = cluster_size_es.to(dtype=torch.float32).clamp_min(1.0)

        # Logits ``[H, es]``.  ``einsum`` here is a single GEMM per head
        # (batched via the leading H dim); ``bmm`` would work too but
        # einsum keeps the index semantics self-documenting.
        logits = torch.einsum("hd,hed->he", q_f, c_f) * float(scale)
        log_sizes = torch.log(sz_f)

        # Stable closed-form softmax aggregate:
        # Subtract a shared scalar max so ``exp`` doesn't overflow.  We
        # use the ``logit`` max (not ``logit + log_size``) because the
        # numerator exponent is pure ``logit`` – matching max keeps
        # ``w_c = exp(logit - max)`` in ``(0, 1]`` with full precision on
        # the dominant cluster.
        logits_max = logits.max(dim=-1, keepdim=True).values  # [H, 1]
        w = torch.exp(logits - logits_max)  # [H, es]
        num = torch.einsum("he,hed->hd", w, vs_f)  # [H, D]
        den = (w * sz_f).sum(dim=-1, keepdim=True)  # [H, 1]
        out = num / den.clamp_min(1e-12)  # [H, D]

        # ``lse`` in absolute units: logsumexp(logit + log_size)
        #   = max_shift + log( Σ_c exp(logit - max_shift) · size_c )
        #   = logits_max + log(den)
        # which is exactly the form FA reports.
        lse = (
            logits_max.squeeze(-1)
            + torch.log(den.squeeze(-1).clamp_min(1e-12))
        )
        return out, lse

    @staticmethod
    def _sparse_retroinfer_lse_merge(
        out_a: torch.Tensor,
        lse_a: torch.Tensor,
        out_b: torch.Tensor,
        lse_b: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Merge two softmax zones via their outputs + per-row LSE.

        Both branches are assumed to report LSE in the **same absolute
        reference frame** (FA's convention – log of unshifted softmax
        denominator).  The merged output is the softmax-weighted average::

            w_a = exp(lse_a - m),  w_b = exp(lse_b - m),  m = max(lse_a, lse_b)
            out = (w_a · out_a + w_b · out_b) / (w_a + w_b)
            lse = m + log(w_a + w_b)

        Treats ``lse = -inf`` as a zero-weight branch (e.g. empty
        estimation zone), in which case ``out`` falls through to the
        non-inf side and ``lse`` equals that side's LSE.  No NaNs even
        when both branches are ``-inf`` (an all-steady-empty-est
        pathological case) – caller should not reach this with a
        populated retrieval zone.

        Args:
            out_a, out_b: ``[H, D]`` per-row softmax outputs.
            lse_a, lse_b: ``[H]`` matching log-sum-exp.

        Returns:
            ``(out [H, D], lse [H])`` in the same dtype as ``out_a``.
        """
        out_a_f = out_a.to(dtype=torch.float32)
        out_b_f = out_b.to(dtype=torch.float32)
        la = lse_a.to(dtype=torch.float32)
        lb = lse_b.to(dtype=torch.float32)
        m = torch.maximum(la, lb)
        # ``exp(-inf - m) = 0`` so zero-branch contributions drop cleanly.
        w_a = torch.exp(la - m)
        w_b = torch.exp(lb - m)
        w_sum = w_a + w_b
        out = (
            (w_a.unsqueeze(-1) * out_a_f
             + w_b.unsqueeze(-1) * out_b_f)
            / w_sum.unsqueeze(-1).clamp_min(1e-12)
        )
        lse = m + torch.log(w_sum.clamp_min(1e-12))
        return out.to(dtype=out_a.dtype), lse

    def _build_sparse_runtime_retroinfer(
        self,
        *,
        layer_name: str,
        query: torch.Tensor,
        spec: "SparseAttentionSpec",
    ) -> "dict[str, torch.Tensor] | None":
        """Phase 5 — retroinfer-style runtime builder for FA.

        Composes the Phase 3 cluster-level selector with the Phase 4a
        cluster-expand + paged-KV gather to produce the dict consumed by
        :meth:`FlashAttentionImpl._forward_retroinfer_exec_buf`:

            {"exec_buf_k":  [H, max_budget, D]  (cache dtype),
             "exec_buf_v":  [H, max_budget, D]  (cache dtype),
             "valid_lengths": [H]               int32,
             "centres_es":    [H, es, D]        fp32,
             "value_sum_es":  [H, es, D]        fp32,
             "cluster_size_es":[H, es]          fp32}

        Returns ``None`` when the retroinfer fast path can't run for this
        layer/step – in that case the caller falls back to the legacy
        ``_build_sparse_runtime_q_head_gather`` path (guarded by the
        ``VLLM_SPARSE_LEGACY_TOKEN_TOPK`` bisection switch at the
        hook-level dispatcher).  Reasons we bail:

        * ``num_reqs != 1`` – Phase 5 is the decode fast path only.  The
          per-request for-loop variant lives in Phase 8 (CUDA graph wiring),
          which also wants a single-req shape contract.
        * ``spec`` isn't GQA-regular (exotic head-group layouts) – the
          shared layer ctx cache flags this; the legacy builder already
          handles the slow path correctly.
        * Prefill / mixed prefill-decode – the retroinfer path needs a
          populated ``value_sum`` + CSR (Phase 2b prefill) and a tok_end
          of 1; prefill falls through to dense FA via the usual mechanism.
        * ``cluster_centres`` / ``value_sum`` haven't been materialised yet
          (pre-first-decode warmup window, or spec reconfigured mid-run).

        All ``None`` returns are benign: the hook leaves
        ``_vllm_sparse_runtime_retroinfer`` unset and FA dispatches through
        the legacy path with identical (slower) semantics.
        """
        _t0 = time.perf_counter() if self._sparse_perf_stats_enabled else None
        # One-shot bail diagnostics: count which gate trips each layer/step
        # and log once per unique (gate, layer) pair so we can tell whether
        # the retroinfer path is silently falling back to the legacy
        # builder (e.g. under async scheduling).  Guarded on the debug env
        # var so production runs pay nothing.
        def _bail(gate: str) -> None:
            if not _SPARSE_DEBUG_ASSERT:
                return
            key = (layer_name, gate)
            if not hasattr(self, "_retroinfer_bail_seen"):
                self._retroinfer_bail_seen: set = set()
                self._retroinfer_bail_count: dict = {}
            self._retroinfer_bail_count[gate] = (
                self._retroinfer_bail_count.get(gate, 0) + 1
            )
            if key not in self._retroinfer_bail_seen:
                self._retroinfer_bail_seen.add(key)
                logger.info(
                    "[SparseDebug] retroinfer bail gate=%s layer=%s "
                    "cumulative_count_for_gate=%d",
                    gate,
                    layer_name,
                    self._retroinfer_bail_count[gate],
                )

        try:
            kv_cache_gid = self._sparse_layer_gid_by_name.get(layer_name)
            if kv_cache_gid is None:
                _bail("kv_cache_gid_none")
                return None
            num_reqs = int(self.input_batch.num_reqs)
            if num_reqs != 1:
                _bail(f"num_reqs={num_reqs}")
                return None
            # Decode fast path requires exactly 1 query token for the
            # sole active request.  ``query_start_loc`` encodes the
            # cumulative token offsets so the delta is the per-req query
            # token count.
            tok_end = int(self.query_start_loc.np[1])
            if tok_end != 1:
                _bail(f"tok_end={tok_end}")
                return None

            rid = self.input_batch.req_ids[0]
            req_states = self._sparse_online_index.get(rid)
            if not req_states:
                _bail("req_states_empty")
                return None
            req_state = self.requests.get(rid)
            if req_state is None:
                _bail("req_state_none")
                return None
            seq_len = int(self.seq_lens.np[0])
            p_count = int(self.input_batch.num_prompt_tokens[0])
            if seq_len <= p_count:
                _bail(f"seq_len<=p_count seq_len={seq_len} p_count={p_count}")
                return None
            out_before_step = self._sparse_output_tokens_before_step.get(
                rid, len(req_state.output_token_ids)
            )
            if out_before_step <= 0:
                _bail(f"out_before_step={out_before_step}")
                return None

            ctx = self._sparse_layer_ctx.get(layer_name)
            if ctx is None:
                # Ctx gets warmed by the legacy builder on its first call
                # for a given layer.  Returning ``None`` here lets the
                # hook dispatcher fall back to the legacy path for a
                # single step; subsequent steps hit the retroinfer path
                # once ctx is cached.  Self-healing and one-shot only.
                _bail("ctx_none")
                return None
            if not bool(ctx.get("gqa_regular", False)):
                _bail("not_gqa_regular")
                return None
            num_heads = int(ctx["num_heads"])
            head_size = int(ctx["head_size"])
            num_kv_heads = int(ctx["num_kv_heads"])
            if num_kv_heads <= 0:
                _bail(f"num_kv_heads={num_kv_heads}")
                return None
            npk = max(1, num_heads // num_kv_heads)
            unit_keys_in_kv_order = ctx.get("unit_keys_in_kv_order", ())
            if len(unit_keys_in_kv_order) != num_kv_heads:
                _bail(
                    "unit_keys_mismatch "
                    f"len={len(unit_keys_in_kv_order)} "
                    f"num_kv_heads={num_kv_heads}"
                )
                return None

            # Zero-kernel q split (GQA-regular layout).  Same pattern as
            # ``_build_sparse_runtime_q_head_gather``'s fast path so that
            # the two builders share layout invariants and the cluster
            # selector sees an identical q slice.
            q_flat = (
                query.view(-1, num_heads, head_size)
                if query.dim() != 3 else query
            )
            q_tok = q_flat[tok_end - 1].to(dtype=torch.float32)
            q_list_views = [
                q_tok[k * npk : (k + 1) * npk]
                for k in range(num_kv_heads)
            ]

            batched_states: list[_SparseOnlineLayerState] = []
            for uk in unit_keys_in_kv_order:
                st = req_states.get(uk)
                if st is None:
                    _bail(f"state_missing_unit_key={uk}")
                    return None
                # Retroinfer requires both the centroid table *and* the
                # per-cluster value_sum.  Empty centroids indicate the
                # online state hasn't clustered yet (rare – only at the
                # very first decode step after a reconfigure).  Missing
                # ``value_sum`` is a harder failure: Phase 2b fills it
                # during prefill collection, so a non-zero centroid
                # table with a zero-numel ``value_sum`` signals an
                # out-of-sync state and we must bail.
                if st.cluster_centres.numel() == 0:
                    _bail(f"cluster_centres_empty unit_key={uk}")
                    return None
                if st.value_sum.numel() == 0:
                    _bail(f"value_sum_empty unit_key={uk}")
                    return None
                batched_states.append(st)

            sel = self._sparse_online_select_clusters_batched(
                states=batched_states,
                q_list=q_list_views,
                spec=spec,
                estimation_budget=self._sparse_estimation_budget,
            )
            if sel is None:
                _bail("cluster_selector_returned_none")
                return None
            retrieval_cids, estimation_cids = sel

            # ``kv_caches`` is a per-group list keyed by layer group id;
            # the sparse layer context already resolved ``kv_cache_gid``
            # above so this lookup is a direct slot access.
            kv_cache = self.kv_caches[kv_cache_gid]
            blk_tbl = self.input_batch.block_table[kv_cache_gid]
            bt_row_gpu = blk_tbl.block_table.gpu[0]
            # ``sparse_selection_budget`` is the legacy per-head retrieval
            # cap.  The retroinfer exec buffer also has to hold the always-on
            # steady zone (head + tail) and the pending zone (output tokens
            # generated so far); when ``static_pattern_start +
            # static_pattern_end`` already exceeds the retrieval cap those
            # positions would otherwise fall off the end of ``steady_pending``
            # inside the gather and silently corrupt decode.  Mirror the
            # legacy ``max_k_hint = max(budget, steady + pending)`` invariant.
            head_n = min(int(spec.static_pattern_start), int(p_count))
            tail_start = max(
                0, int(p_count) - int(spec.static_pattern_end)
            )
            steady_count = (
                int(p_count) if head_n >= tail_start
                else head_n + (int(p_count) - tail_start)
            )
            pending_count = max(0, int(seq_len) - int(p_count))
            max_budget = max(
                int(spec.sparse_selection_budget()),
                steady_count + pending_count,
            )

            out = self._sparse_retroinfer_expand_and_gather_single_req(
                states=batched_states,
                retrieval_cids=retrieval_cids,
                estimation_cids=estimation_cids,
                block_table_row=bt_row_gpu,
                kv_cache=kv_cache,
                block_size=int(spec.block_size),
                prefill_len=p_count,
                current_len=seq_len,
                static_pattern_start=int(spec.static_pattern_start),
                static_pattern_end=int(spec.static_pattern_end),
                max_budget=max_budget,
            )
            if _SPARSE_DEBUG_ASSERT:
                if not hasattr(self, "_retroinfer_success_seen"):
                    self._retroinfer_success_seen: set = set()
                    self._retroinfer_success_count: int = 0
                self._retroinfer_success_count += 1
                if layer_name not in self._retroinfer_success_seen:
                    self._retroinfer_success_seen.add(layer_name)
                    logger.info(
                        "[SparseDebug] retroinfer SUCCESS layer=%s "
                        "cumulative_global_success=%d",
                        layer_name,
                        self._retroinfer_success_count,
                    )
            return out
        finally:
            if _t0 is not None:
                self._sparse_perf_record(
                    "_build_sparse_runtime_retroinfer",
                    time.perf_counter() - _t0,
                )

    def _build_sparse_runtime_q_head_gather(
        self,
        *,
        layer_name: str,
        query: torch.Tensor,
        spec: SparseAttentionSpec,
    ) -> dict[str, torch.Tensor] | None:
        """Build the per-query-head compact gather metadata for FA.

        Rewrite notes (bitwise-equivalent to the legacy per-head numpy path):

        * Removed the ``selected_mask.detach().cpu().numpy()`` D2H and the
          matching ``torch.as_tensor(..., device=query.device)`` H2D at the end.
          All packing (block index, slot, physical block id, per-head cu) is
          performed on GPU via ``torch.nonzero`` + ``index_select`` +
          ``cumsum``.  ``torch.nonzero`` introduces a single implicit sync per
          request which was already present in the legacy path (via
          ``.cpu().numpy()``), so total sync count is unchanged.

        * Per-head static mapping (``kv_to_qh_tensor`` etc.) is cached on
          ``self._sparse_layer_ctx`` – it is fully determined by
          ``num_heads`` / ``num_kv_heads`` and does not change across steps.

        * Adds ``max_k`` to the returned dict (= ``spec.sparse_selection_budget()``),
          used by ``FlashAttentionImpl._forward_per_head_compact_kv_gather``
          to skip a ``.max().item()`` sync when calling FA varlen.
        """
        _t0 = time.perf_counter() if self._sparse_perf_stats_enabled else None
        perf_enabled = self._sparse_perf_stats_enabled and self._has_sparse_attn
        perf_local_ms: dict[str, float] = defaultdict(float)

        def _perf_add(name: str, start_t: float | None) -> None:
            if start_t is None:
                return
            perf_local_ms[name] += (time.perf_counter() - start_t) * 1000.0

        try:
            kv_cache_gid = self._sparse_layer_gid_by_name.get(layer_name)
            if kv_cache_gid is None:
                return None

            ctx = self._sparse_layer_ctx.get(layer_name)
            if ctx is None:
                attn_mod = self.compilation_config.static_forward_context.get(
                    layer_name
                )
                if attn_mod is None:
                    return None
                _t_qh_map = time.perf_counter() if perf_enabled else None
                num_heads_i = int(attn_mod.num_heads)
                num_kv_heads_i = int(getattr(attn_mod, "num_kv_heads", 1))
                head_size_i = int(attn_mod.head_size)
                qh_to_kv = [
                    self._sparse_online_qh_to_kv_index(
                        qh_idx, num_heads_i, num_kv_heads_i
                    )
                    for qh_idx in range(num_heads_i)
                ]
                kv_to_qh: dict[int, list[int]] = defaultdict(list)
                for qh_idx, kv_idx in enumerate(qh_to_kv):
                    kv_to_qh[kv_idx].append(qh_idx)
                kv_to_qh_tensor = {
                    kv_idx: torch.tensor(
                        qh_indices, dtype=torch.int64, device=query.device
                    )
                    for kv_idx, qh_indices in kv_to_qh.items()
                }
                # Tier-0 statics: only depend on model shape.  Cached on the
                # layer context so the per-layer forward never rebuilds them
                # (previously ``torch.arange`` + ``//`` were re-run every call
                # inside ``_forward_per_head_compact_kv_gather``).
                num_queries_per_kv_i = max(1, num_heads_i // max(1, num_kv_heads_i))
                head_ids_int64 = torch.arange(
                    num_heads_i, dtype=torch.int64, device=query.device
                )
                kv_head_ids_int64 = head_ids_int64 // num_queries_per_kv_i
                # GQA-regular layout check: when ``num_heads == num_kv *
                # num_queries_per_kv`` and kv_idx ``k`` owns the contiguous
                # head range ``[k*npk, (k+1)*npk)``, we can replace the G
                # per-kv ``index_select`` + G per-kv scatter with a single
                # ``torch.cat`` at the end (and the per-kv q slice becomes
                # a zero-kernel view).  This pattern holds for every
                # mainstream GQA / MQA model (Llama, Qwen, Mistral, ...).
                # We cache the flag + ordered unit-keys so the hot path
                # does at most one dict lookup per kv_head.
                gqa_regular = (
                    num_heads_i == num_kv_heads_i * num_queries_per_kv_i
                ) and all(
                    kv_to_qh.get(k)
                    == list(
                        range(
                            k * num_queries_per_kv_i,
                            (k + 1) * num_queries_per_kv_i,
                        )
                    )
                    for k in range(num_kv_heads_i)
                )
                unit_keys_in_kv_order = tuple(
                    sparse_kv_unit_key(layer_name, k)
                    for k in range(num_kv_heads_i)
                )
                # Phase-B static tensors for the num_reqs==1 decode
                # finalize.  At num_reqs==1, ``cu_q_flat`` collapses to
                # ``arange(num_heads + 1, int32)`` and ``req_ids_flat``
                # is a zero-filled int64 vector of length ``num_heads``.
                # Both are purely layer-static (only depend on
                # ``num_heads``) and identical across layers that
                # share the same (num_heads, num_kv_heads) shape, so
                # we build them once here and the hot path just
                # returns the cached tensor.  Eliminates 2 tiny kernel
                # launches per sparse layer (``arange`` + ``zeros``).
                cu_q_flat_nreqs1_int32 = torch.arange(
                    num_heads_i + 1,
                    dtype=torch.int32,
                    device=query.device,
                )
                req_ids_flat_nreqs1_zero = torch.zeros(
                    num_heads_i,
                    dtype=torch.int64,
                    device=query.device,
                )
                # Phase-C: persistent scratch for Triton pack's
                # head_offsets (int32 [H+1]).  Zero-initialized once;
                # the ``[0]`` stays at zero forever because subsequent
                # calls only touch ``[1:]`` (count kernel writes, then
                # in-place cumsum).  Eliminates 2 tiny kernel launches
                # per sparse layer (``[0] = 0`` scalar write and the
                # ``.to(int32)`` cast), ~70 µs saved per pack call
                # (~2 ms/decode step at 28 sparse layers).
                pack_head_offsets_scratch = torch.zeros(
                    num_heads_i + 1,
                    dtype=torch.int32,
                    device=query.device,
                )
                ctx = {
                    "num_heads": num_heads_i,
                    "num_kv_heads": num_kv_heads_i,
                    "num_queries_per_kv": num_queries_per_kv_i,
                    "head_size": head_size_i,
                    "kv_to_qh_tensor": kv_to_qh_tensor,
                    "head_ids_int64": head_ids_int64,
                    "kv_head_ids_int64": kv_head_ids_int64,
                    "gqa_regular": gqa_regular,
                    "unit_keys_in_kv_order": unit_keys_in_kv_order,
                    "cu_q_flat_nreqs1_int32": cu_q_flat_nreqs1_int32,
                    "req_ids_flat_nreqs1_zero": req_ids_flat_nreqs1_zero,
                    "pack_head_offsets_scratch": pack_head_offsets_scratch,
                }
                self._sparse_layer_ctx[layer_name] = ctx
                _perf_add(
                    "_build_sparse_runtime_q_head_gather:qhead_map", _t_qh_map
                )
            num_heads = int(ctx["num_heads"])
            head_size = int(ctx["head_size"])
            kv_to_qh_tensor = ctx["kv_to_qh_tensor"]
            # Backfill Tier-0 statics for any layer ctx that predates this
            # cache (e.g. warm restart where the dict was built by an older
            # version of this method).  Idempotent and cheap.
            num_queries_per_kv = int(
                ctx.get(
                    "num_queries_per_kv",
                    max(1, num_heads // max(1, int(ctx["num_kv_heads"]))),
                )
            )
            head_ids_int64 = ctx.get("head_ids_int64")
            if head_ids_int64 is None:
                head_ids_int64 = torch.arange(
                    num_heads, dtype=torch.int64, device=query.device
                )
                ctx["head_ids_int64"] = head_ids_int64
            kv_head_ids_int64 = ctx.get("kv_head_ids_int64")
            if kv_head_ids_int64 is None:
                kv_head_ids_int64 = head_ids_int64 // num_queries_per_kv
                ctx["kv_head_ids_int64"] = kv_head_ids_int64
            ctx.setdefault("num_queries_per_kv", num_queries_per_kv)
            # Backfill GQA-layout hints for older ctx dicts.
            if "gqa_regular" not in ctx:
                num_kv_heads_bf = int(ctx["num_kv_heads"])
                ctx["gqa_regular"] = (
                    num_heads == num_kv_heads_bf * num_queries_per_kv
                ) and all(
                    len(kv_to_qh_tensor.get(k, torch.empty(0))) == num_queries_per_kv
                    and int(kv_to_qh_tensor[k][0].item())
                    == k * num_queries_per_kv
                    and int(kv_to_qh_tensor[k][-1].item())
                    == (k + 1) * num_queries_per_kv - 1
                    for k in range(num_kv_heads_bf)
                )
            if "unit_keys_in_kv_order" not in ctx:
                ctx["unit_keys_in_kv_order"] = tuple(
                    sparse_kv_unit_key(layer_name, k)
                    for k in range(int(ctx["num_kv_heads"]))
                )
            # Phase-B backfill: if the ctx was built by an older build
            # that predates these cached statics, lazy-init on first use.
            if "cu_q_flat_nreqs1_int32" not in ctx:
                ctx["cu_q_flat_nreqs1_int32"] = torch.arange(
                    num_heads + 1,
                    dtype=torch.int32,
                    device=query.device,
                )
            # Phase-C backfill: persistent head_offsets scratch for
            # the Triton pack.  Zero-init once; kept at [0]=0 forever.
            if "pack_head_offsets_scratch" not in ctx:
                ctx["pack_head_offsets_scratch"] = torch.zeros(
                    num_heads + 1,
                    dtype=torch.int32,
                    device=query.device,
                )
            if "req_ids_flat_nreqs1_zero" not in ctx:
                ctx["req_ids_flat_nreqs1_zero"] = torch.zeros(
                    num_heads,
                    dtype=torch.int64,
                    device=query.device,
                )

            num_reqs = int(self.input_batch.num_reqs)
            if num_reqs <= 0:
                return None

            q_flat = (
                query.view(-1, num_heads, head_size)
                if query.dim() != 3 else query
            )

            forward_additional_kwargs = (
                get_forward_context().additional_kwargs
                if is_forward_context_available()
                else {}
            )
            publish_lmcache_rows = bool(
                forward_additional_kwargs.get(
                    "lmcache_collect_sparse_per_head_token_indices"
                )
            )
            lmcache_rows_by_req: dict[str, list[list[int]]] = {}
            blk_tbl = self.input_batch.block_table[kv_cache_gid]
            bsz = int(spec.block_size)
            budget = int(spec.sparse_selection_budget())

            # Per-req GPU accumulators.  ``nonzero`` returns row-major order
            # (head first, then token), so for each req the appended
            # ``head_ids`` is already head-major and within each head sorted by
            # token.  A stable argsort by (head, req) at the end yields the
            # final head-major flat layout.
            per_req_head_ids: list[torch.Tensor] = []
            per_req_phys: list[torch.Tensor] = []
            per_req_slots: list[torch.Tensor] = []
            per_req_counts: list[torch.Tensor] = []
            max_k_hint = budget

            for req_idx in range(num_reqs):
                _t_req = time.perf_counter() if perf_enabled else None
                rid = self.input_batch.req_ids[req_idx]
                req_states = self._sparse_online_index.get(rid)
                if not req_states:
                    return None
                req_state = self.requests.get(rid)
                if req_state is None:
                    return None
                seq_len = int(self.seq_lens.np[req_idx])
                p_count = int(self.input_batch.num_prompt_tokens[req_idx])
                if seq_len <= p_count:
                    return None
                prompt_len = min(p_count, seq_len)
                pending_count = max(0, seq_len - prompt_len)
                head_n = min(int(spec.static_pattern_start), prompt_len)
                tail_start = max(
                    0, prompt_len - int(spec.static_pattern_end)
                )
                steady_count = (
                    prompt_len
                    if head_n >= tail_start
                    else head_n + (prompt_len - tail_start)
                )
                select_budget = max(0, budget - pending_count)
                max_k_hint = max(max_k_hint, steady_count + pending_count)
                out_before_step = self._sparse_output_tokens_before_step.get(
                    rid, len(req_state.output_token_ids)
                )
                if out_before_step <= 0:
                    return None
                tok_end = int(self.query_start_loc.np[req_idx + 1])
                if tok_end <= 0:
                    return None

                q_tok = q_flat[tok_end - 1].to(dtype=torch.float32)
                # GPU block-table row is needed by both the legacy
                # selected-mask pack and the direct cluster-member pack.
                bt_row_gpu = blk_tbl.block_table.gpu[req_idx]
                _t_select = time.perf_counter() if perf_enabled else None
                # Collect per-kv_head inputs first so we can hand the whole
                # layer's kv_heads to the batched selector in one call.  This
                # replaces G × (cluster-score, topk, scatter, gather, topk)
                # micro-kernels with 3-D variants, shaving the bulk of the
                # launch overhead that ``_sparse_online_select_tokens``
                # otherwise accumulates on decode.
                #
                # Fast path (GQA-regular layout): ``q_tok`` already lays out
                # q-heads in kv-head-major order, so each kv-group's q slice
                # is a zero-kernel view (``q_tok[k*npk:(k+1)*npk]``) instead
                # of an ``index_select`` kernel.  The per-kv scatter back
                # into ``selected_mask`` also collapses to a single
                # ``torch.cat`` that *is* the final mask, dropping the
                # initial ``zeros`` allocation.  Saves ~8 kernels / layer
                # (G index_selects + G scatters + zeros alloc).  The
                # fallback path below is retained for exotic layouts where
                # kv-heads own non-contiguous q-head ranges.
                gqa_regular = bool(ctx.get("gqa_regular", False))
                unit_keys_in_kv_order = ctx.get("unit_keys_in_kv_order", ())
                select_source = "labels"
                direct_cluster_runtime = None
                direct_cluster_pack = None
                selected_mask: torch.Tensor | None = None
                if gqa_regular and len(unit_keys_in_kv_order) == int(
                    ctx["num_kv_heads"]
                ):
                    num_kv_heads = int(ctx["num_kv_heads"])
                    npk = num_queries_per_kv
                    batched_states = []
                    missing_state = False
                    for unit_key in unit_keys_in_kv_order:
                        state = req_states.get(unit_key)
                        if state is None:
                            missing_state = True
                            break
                        batched_states.append(state)
                    if missing_state:
                        return None
                    # Zero-kernel q split: slices share q_tok's storage.
                    q_list_views = [
                        q_tok[k * npk : (k + 1) * npk]
                        for k in range(num_kv_heads)
                    ]
                    if (
                        self._sparse_legacy_token_topk
                        and num_reqs == 1
                        and not publish_lmcache_rows
                    ):
                        direct_cluster_runtime = (
                            self._sparse_online_cluster_exec_runtime_batched(
                                states=batched_states,
                                q_list=q_list_views,
                                seq_len=seq_len,
                                prompt_len=prompt_len,
                                spec=spec,
                                select_budget=select_budget,
                                bt_row_gpu=bt_row_gpu,
                            )
                        )
                    if direct_cluster_runtime is not None:
                        select_source = "clusters_exec"
                    else:
                        group_masks = (
                            self._sparse_online_select_tokens_from_clusters_batched(
                                states=batched_states,
                                q_list=q_list_views,
                                total_tokens=prompt_len,
                                spec=spec,
                                budget_override=select_budget,
                            )
                            if self._sparse_legacy_token_topk
                            else None
                        )
                        if group_masks is not None:
                            select_source = "clusters"
                        if group_masks is None:
                            group_masks = self._sparse_online_select_tokens_batched(
                                states=batched_states,
                                q_list=q_list_views,
                                total_tokens=prompt_len,
                                spec=spec,
                                budget_override=select_budget,
                            )
                        # Concatenation in kv-head order == head-order under
                        # GQA-regular layout, so this is the final mask with
                        # no scatter.  ``cat`` guarantees a contiguous tensor
                        # that downstream ``nonzero`` / indexing expects.
                        selected_mask = torch.cat(group_masks, dim=0)
                else:
                    selected_mask = torch.zeros(
                        (num_heads, seq_len),
                        dtype=torch.bool,
                        device=query.device,
                    )
                    batched_states = []
                    batched_q: list[torch.Tensor] = []
                    batched_qh_indices: list[torch.Tensor] = []
                    for kv_idx, qh_indices_t in kv_to_qh_tensor.items():
                        unit_key = sparse_kv_unit_key(layer_name, kv_idx)
                        state = req_states.get(unit_key)
                        if state is None:
                            return None
                        q_group = q_tok.index_select(0, qh_indices_t)
                        batched_states.append(state)
                        batched_q.append(q_group)
                        batched_qh_indices.append(qh_indices_t)
                    group_masks = (
                        self._sparse_online_select_tokens_from_clusters_batched(
                            states=batched_states,
                            q_list=batched_q,
                            total_tokens=prompt_len,
                            spec=spec,
                            budget_override=select_budget,
                        )
                        if self._sparse_legacy_token_topk
                        else None
                    )
                    if group_masks is not None:
                        select_source = "clusters"
                    if group_masks is None:
                        group_masks = self._sparse_online_select_tokens_batched(
                            states=batched_states,
                            q_list=batched_q,
                            total_tokens=prompt_len,
                            spec=spec,
                            budget_override=select_budget,
                        )
                    for qh_indices_t, group_mask in zip(
                        batched_qh_indices, group_masks, strict=True
                    ):
                        selected_mask[qh_indices_t] = group_mask
                _perf_add(
                    "_build_sparse_runtime_q_head_gather:select_tokens",
                    _t_select,
                )
                if direct_cluster_runtime is not None:
                    if _SPARSE_TOKEN_TOPK_TRACE:
                        logger.info(
                            "[SparseTokenTopK] layer=%s req_id=%s "
                            "seq_len=%d prompt_len=%d pending=%d "
                            "budget=%d select_budget=%d steady=%d "
                            "select_source=%s pack_triton=%s",
                            layer_name,
                            rid,
                            int(seq_len),
                            int(prompt_len),
                            int(pending_count),
                            int(budget),
                            int(select_budget),
                            int(steady_count),
                            select_source,
                            True,
                        )
                    _perf_add(
                        "_build_sparse_runtime_q_head_gather:per_req_total",
                        _t_req,
                    )
                    direct_cluster_runtime.update(
                        {
                            "phys": None,
                            "slots": None,
                            "cu": None,
                            "head_offsets": None,
                            "max_k": max_k_hint,
                            "cu_q_flat": ctx["cu_q_flat_nreqs1_int32"],
                            "req_ids_flat": ctx["req_ids_flat_nreqs1_zero"],
                            "kv_pair_ids_flat": kv_head_ids_int64,
                            "num_q_flat": num_heads,
                            "num_q_heads": num_heads,
                            "num_reqs": num_reqs,
                            "cluster_head_offsets_scratch": ctx[
                                "pack_head_offsets_scratch"
                            ],
                        }
                    )
                    return direct_cluster_runtime

                # Post-prompt tokens are always included, but they must be
                # reserved from the sparse budget before prompt retrieval.
                # Otherwise actual per-head K length can exceed the
                # ``max_k`` hint passed to FA, which corrupts decode outputs.
                if direct_cluster_pack is None:
                    assert selected_mask is not None
                    if int(selected_mask.shape[1]) != seq_len:
                        selected_prompt = selected_mask
                        selected_mask = torch.zeros(
                            (num_heads, seq_len),
                            dtype=torch.bool,
                            device=query.device,
                        )
                        selected_mask[:, :prompt_len] = selected_prompt
                    selected_mask[:, prompt_len:seq_len] = True

                _t_pack = time.perf_counter() if perf_enabled else None
                # Skip the per-layer H2D block-table copy: the block-table
                # buffer keeps a committed GPU mirror (``block_table.gpu``,
                # int32) that the core attention path already consumes.
                # The row view ``bt_gpu[req_idx]`` is zero-kernel.
                bt_row_gpu = blk_tbl.block_table.gpu[req_idx]

                # Phase-A fused pack: for the decode-dominant
                # ``num_reqs == 1`` case, a single Triton kernel pair
                # replaces ``nonzero`` + ``//`` + ``-*`` +
                # ``index_select`` + ``sum(dim=1)`` (5 ops → 2 ops).
                # Bitwise-equivalent, same implicit sync count (the
                # total-count sync matches ``nonzero``'s size sync).
                # ``head_ids_r`` is never consumed in the
                # ``num_reqs == 1`` finalize path, so it is not
                # returned; ``per_req_head_ids`` stores a zero-length
                # placeholder to keep list lengths consistent for any
                # multi-layer accounting elsewhere.
                use_triton = direct_cluster_pack is not None or (
                    self._sparse_triton_pack_enabled
                    and not _SPARSE_DEBUG_ASSERT
                    and num_reqs == 1
                    and selected_mask is not None
                    and selected_mask.is_cuda
                )
                if direct_cluster_pack is not None:
                    (
                        phys_r,
                        slots_r,
                        triton_kv_token_ids,
                        counts_r,
                        triton_head_offsets,
                    ) = direct_cluster_pack
                    self._sparse_triton_head_offsets_cache = (
                        triton_head_offsets
                    )
                    self._sparse_triton_kv_token_ids_cache = (
                        triton_kv_token_ids
                    )
                    head_ids_r = torch.empty(
                        0, dtype=torch.int64, device=query.device
                    )
                elif use_triton:
                    assert selected_mask is not None
                    # Phase-B: Triton emits int64 phys/slots +
                    # per-entry kv_token_ids in one kernel, folding
                    # the three legacy finalize ops (two
                    # ``.to(int64)`` casts plus
                    # ``torch.repeat_interleave``) into the pack step.
                    # The returned ``phys_r`` / ``slots_r`` are int64
                    # (the FA fast path consumes them unchanged via
                    # ``phys_int64`` / ``slots_int64``).
                    (
                        phys_r,
                        slots_r,
                        triton_kv_token_ids,
                        counts_r,
                        triton_head_offsets,
                    ) = sparse_pack_single_req(
                        selected_mask,
                        bt_row_gpu,
                        kv_head_ids_int64,
                        int(bsz),
                        # Phase-C: pass the layer-persistent int32 [H+1]
                        # scratch buffer so the pack reuses it instead
                        # of paying ``torch.zeros + cumsum`` kernels
                        # every call.  The ``[0]=0`` invariant is
                        # preserved because the pack only writes
                        # ``[1:]``.
                        head_offsets_scratch=ctx["pack_head_offsets_scratch"],
                        # Phase-C: forward the perf recorder so the
                        # ``pack_sub:*`` breakdown (launch_count, cumsum,
                        # item_sync, alloc_outputs, launch_data) is
                        # captured only when ``VLLM_SPARSE_PERF_DEBUG``
                        # is on.  The conditional keeps the hot path
                        # overhead-free when stats are disabled.
                        perf_record=(
                            self._sparse_perf_record if perf_enabled else None
                        ),
                    )
                    # Stash for num_reqs==1 finalize fast-path reuse.
                    self._sparse_triton_head_offsets_cache = (
                        triton_head_offsets
                    )
                    self._sparse_triton_kv_token_ids_cache = (
                        triton_kv_token_ids
                    )
                    head_ids_r = torch.empty(
                        0, dtype=torch.int64, device=query.device
                    )
                else:
                    assert selected_mask is not None
                    # Legacy path (num_reqs > 1 or explicit fallback):
                    # ``nonzero`` produces [N, 2] row-major (head first,
                    # token second), matching one implicit GPU sync per
                    # req as the legacy ``.cpu().numpy()`` had.
                    self._sparse_triton_head_offsets_cache = None
                    self._sparse_triton_kv_token_ids_cache = None
                    nz = torch.nonzero(selected_mask, as_tuple=False)
                    head_ids_r = nz[:, 0]
                    tok_ids_r = nz[:, 1]
                    block_idx = tok_ids_r // int(bsz)
                    slots_r = (
                        tok_ids_r - block_idx * int(bsz)
                    ).to(torch.int32)
                    if _SPARSE_DEBUG_ASSERT:
                        _sparse_debug_range(
                            "sparse token ids",
                            tok_ids_r,
                            int(seq_len),
                        )
                        _sparse_debug_range(
                            "sparse block_idx",
                            block_idx,
                            int(bt_row_gpu.numel()),
                        )
                    phys_r = bt_row_gpu.index_select(0, block_idx)
                    if _SPARSE_DEBUG_ASSERT:
                        _sparse_debug_range(
                            "sparse physical block ids",
                            phys_r.to(torch.int64),
                            int(self.kv_caches[kv_cache_gid][0].shape[0])
                            if isinstance(self.kv_caches[kv_cache_gid], tuple)
                            else int(self.kv_caches[kv_cache_gid].shape[1]),
                        )
                    counts_r = selected_mask.sum(dim=1).to(torch.int64)

                if _SPARSE_TOKEN_TOPK_TRACE:
                    counts_for_trace = (
                        counts_r
                        if counts_r is not None
                        else triton_head_offsets[1:].to(torch.int64)
                    )
                    counts_trace = counts_for_trace.to(torch.float32)
                    unique_phys = int(torch.unique(phys_r).numel()) \
                        if phys_r.numel() else 0
                    logger.info(
                        "[SparseTokenTopK] layer=%s req_id=%s "
                        "seq_len=%d prompt_len=%d pending=%d "
                        "budget=%d select_budget=%d steady=%d "
                        "heads=%d selected_min=%.0f selected_avg=%.1f "
                        "selected_max=%.0f selected_total=%.0f "
                        "unique_phys_blocks=%d select_source=%s "
                        "pack_triton=%s",
                        layer_name,
                        rid,
                        int(seq_len),
                        int(prompt_len),
                        int(pending_count),
                        int(budget),
                        int(select_budget),
                        int(steady_count),
                        int(counts_for_trace.numel()),
                        float(counts_trace.min().item())
                        if counts_trace.numel() else 0.0,
                        float(counts_trace.mean().item())
                        if counts_trace.numel() else 0.0,
                        float(counts_trace.max().item())
                        if counts_trace.numel() else 0.0,
                        float(counts_trace.sum().item())
                        if counts_trace.numel() else 0.0,
                        unique_phys,
                        select_source,
                        bool(use_triton),
                    )

                per_req_head_ids.append(head_ids_r)
                per_req_phys.append(phys_r)
                per_req_slots.append(slots_r)
                per_req_counts.append(counts_r)
                _perf_add(
                    "_build_sparse_runtime_q_head_gather:pack_per_req_gpu",
                    _t_pack,
                )

                if publish_lmcache_rows:
                    selected_mask_np = selected_mask.detach().cpu().numpy()
                    token_ids_np = np.arange(seq_len, dtype=np.int64)
                    lmcache_rows_by_req[rid] = [
                        token_ids_np[selected_mask_np[qh]].tolist()
                        for qh in range(num_heads)
                    ]

                _perf_add(
                    "_build_sparse_runtime_q_head_gather:per_req_total",
                    _t_req,
                )

            if not per_req_head_ids:
                return None

            _t_finalize = time.perf_counter() if perf_enabled else None
            # Fast path: decode has ``num_reqs == 1`` in practice.  The
            # generic multi-req path below launches ~18 tiny kernels
            # (``cat`` x3, H2D ``tensor``, ``arange`` + ``repeat_interleave``
            # for ``all_req_ids``, ``mul`` + ``argsort`` + ``index_select`` x2
            # for reordering, ``stack`` + ``cumsum`` x2 for ``cu_mat`` /
            # ``total_per_head``, and another ``empty`` + ``cumsum`` +
            # ``repeat_interleave`` for the Tier-2 ``cu_k_flat`` /
            # ``kv_pair_ids_flat``).  Almost all are no-ops or identity
            # transforms when there is a single request:
            #   * ``cat`` of a 1-element list == the single tensor
            #   * ``nonzero`` already yields head-major order, so the
            #     stable argsort produces an identity permutation
            #   * ``k_lens_flat == counts_r`` and ``cu_k_flat == head_offsets``
            #   * ``kv_pair_ids_flat == kv_head_ids_int64`` (repeat by 1)
            # Bypassing them keeps bitwise-equivalent outputs while saving
            # ~5-8 ms / decode step on the reference model.  The generic
            # path is retained for multi-req prefill-style calls.
            if num_reqs == 1:
                head_ids_r = per_req_head_ids[0]
                flat_phys = per_req_phys[0].contiguous()
                flat_slots = per_req_slots[0].contiguous()
                counts_r = per_req_counts[0]

                # Reuse the already-computed prefix from Triton pack when
                # available (Phase-A fast path): ``head_offsets`` is an
                # int32 [H+1] tensor with ``head_offsets[0] == 0`` by
                # construction, so ``head_offsets[1:]`` is a zero-cost
                # view that equals ``counts_r.to(int32)``.  Skipping the
                # redundant ``empty + zero-init + cumsum`` pair saves two
                # tiny kernel launches per sparse layer.
                triton_head_offsets = getattr(
                    self, "_sparse_triton_head_offsets_cache", None
                )
                triton_kv_token_ids = getattr(
                    self, "_sparse_triton_kv_token_ids_cache", None
                )
                if triton_head_offsets is not None:
                    head_offsets = triton_head_offsets
                    # Zero-copy view (contiguous slice from element 1).
                    counts_r_i32 = head_offsets[1:]
                    self._sparse_triton_head_offsets_cache = None
                else:
                    counts_r_i32 = counts_r.to(torch.int32)
                    head_offsets = torch.empty(
                        num_heads + 1,
                        dtype=torch.int32,
                        device=query.device,
                    )
                    head_offsets[0] = 0
                    torch.cumsum(counts_r_i32, dim=0, out=head_offsets[1:])

                # ``cu_mat[:, 0] == 0`` and ``cu_mat[:, 1] == counts_r``.
                # Build via a single ``stack`` -> one kernel vs
                # ``zeros + cumsum + stack`` in the generic path.
                cu_mat = torch.stack(
                    (torch.zeros_like(counts_r_i32), counts_r_i32), dim=1
                )

                # Phase-B: when the Triton pack ran, ``flat_phys`` /
                # ``flat_slots`` are already int64 and ``kv_token_ids``
                # is already built (all three in one fused kernel), so
                # the ``.to(int64)`` / ``repeat_interleave`` triplet is
                # skipped.  Falls back to the explicit ops when the
                # Triton path is disabled or bypassed (num_reqs > 1,
                # CPU input).
                #
                # Phase-C: when the Triton path runs, ``counts_r`` is
                # returned as ``None`` (not built separately — only
                # ``head_offsets[1:]`` is produced).  ``total_per_head``
                # is therefore materialized lazily inside the fallback
                # branch from the int32 view, which is the only place
                # it is read.
                if triton_kv_token_ids is not None:
                    flat_phys_int64 = flat_phys
                    flat_slots_int64 = flat_slots
                    kv_token_ids = triton_kv_token_ids
                    self._sparse_triton_kv_token_ids_cache = None
                else:
                    flat_phys_int64 = flat_phys.to(torch.int64)
                    flat_slots_int64 = flat_slots.to(torch.int64)
                    total_per_head = (
                        counts_r
                        if counts_r is not None
                        else counts_r_i32.to(torch.int64)
                    )
                    kv_token_ids = torch.repeat_interleave(
                        kv_head_ids_int64, total_per_head.to(torch.int64)
                    )

                # cu_k_flat == head_offsets when num_reqs == 1 (they both
                # encode the same per-head cumulative counts, just with
                # different flat/2D shapes).  Share the tensor.
                cu_k_flat = head_offsets
                num_q_flat = num_heads
                # Phase-B: ``cu_q_flat`` and ``req_ids_flat`` are
                # purely layer-static at num_reqs==1 (they only depend
                # on ``num_heads``).  Use the cached tensors built
                # once in the ctx init to skip per-layer ``arange`` +
                # ``zeros`` allocations.
                cu_q_flat = ctx["cu_q_flat_nreqs1_int32"]
                req_ids_flat = ctx["req_ids_flat_nreqs1_zero"]
                # kv_pair_ids_flat == kv_head_ids_int64 when num_reqs == 1.
                kv_pair_ids_flat = kv_head_ids_int64
            else:
                all_head_ids = torch.cat(per_req_head_ids)
                all_phys = torch.cat(per_req_phys)
                all_slots = torch.cat(per_req_slots)
                # Build a ``req_id`` companion tensor on GPU without a loop
                # by repeating req_idx for the length of each per_req entry.
                lens = torch.tensor(
                    [int(h.shape[0]) for h in per_req_head_ids],
                    dtype=torch.int64,
                    device=query.device,
                )
                req_range = torch.arange(
                    num_reqs, dtype=torch.int64, device=query.device
                )
                all_req_ids = torch.repeat_interleave(req_range, lens)

                # Stable argsort by (head, req) groups entries head-major
                # with reqs in ascending order.  Within each (head, req)
                # group the original nonzero order (ascending token id) is
                # preserved.
                sort_key = all_head_ids * int(num_reqs) + all_req_ids
                order = torch.argsort(sort_key, stable=True)
                flat_phys = all_phys.index_select(0, order).contiguous()
                flat_slots = all_slots.index_select(0, order).contiguous()

                counts_2d = torch.stack(per_req_counts, dim=1)  # [H, num_reqs]
                cu_mat = torch.zeros(
                    (num_heads, num_reqs + 1),
                    dtype=torch.int32,
                    device=query.device,
                )
                cu_mat[:, 1:] = counts_2d.cumsum(dim=1).to(torch.int32)
                total_per_head = counts_2d.sum(dim=1)
                head_offsets = torch.zeros(
                    num_heads + 1, dtype=torch.int32, device=query.device
                )
                head_offsets[1:] = total_per_head.cumsum(0).to(torch.int32)

                # Tier-2 pre-computations: everything the FA per-head
                # compact gather fast path used to rebuild on every layer
                # call.  Moving them here turns the FA hot path into just
                # two advanced-indexing gathers + Q layout copy + one FA
                # launch.  Measured (20 sparse layers, 1 req, 28 q-heads)
                # per-layer ``gather_ms`` overhead was 0.57 ms (~20 tiny
                # kernels); pre-building here amortizes the non-data ones
                # to a single build call per step.
                flat_phys_int64 = flat_phys.to(torch.int64)
                flat_slots_int64 = flat_slots.to(torch.int64)
                # ``kv_token_ids[i]`` = kv_head index for the i-th entry
                # of the flattened phys/slots arrays.  Derived from the
                # layer-static ``kv_head_ids_int64`` and the dynamic
                # ``total_per_head`` counts.
                kv_token_ids = torch.repeat_interleave(
                    kv_head_ids_int64, total_per_head.to(torch.int64)
                )
                # 1D flat ``cu_seqlens_k`` that FA wants: indexed by
                # (head * num_reqs + req).  Replaces the FA-side
                # ``(cu_mat[:, 1:] - cu_mat[:, :-1]).reshape(-1)`` +
                # ``cumsum`` pair.
                k_lens_flat = (cu_mat[:, 1:] - cu_mat[:, :-1]).reshape(-1)
                cu_k_flat = torch.empty(
                    int(k_lens_flat.numel()) + 1,
                    dtype=cu_mat.dtype,
                    device=query.device,
                )
                cu_k_flat[0] = 0
                cu_k_flat[1:] = torch.cumsum(k_lens_flat, dim=0)
                # Step-shared shape-only tensors: ``max_seqlen_q == 1`` in
                # the decode fast path and the compact-gather flattening
                # has one (head, req) query per slot, so ``q_lens`` is
                # all-ones and ``cu_q_flat`` is just an arange.
                num_q_flat = num_heads * num_reqs
                cu_q_flat = torch.arange(
                    num_q_flat + 1,
                    dtype=torch.int32,
                    device=query.device,
                )
                # Descale flat index: ``req_ids`` / ``kv_pair_ids`` are
                # the advanced-indexing selectors FA uses to flatten
                # ``{q,k,v}_descale`` from [num_reqs, num_kv_heads] to
                # [num_q_flat, 1].  Both depend only on
                # (num_reqs, num_heads, num_queries_per_kv).
                req_ids_flat = torch.arange(
                    num_reqs, dtype=torch.int64, device=query.device
                ).repeat(num_heads)
                kv_pair_ids_flat = kv_head_ids_int64.repeat_interleave(
                    num_reqs
                )
            _perf_add(
                "_build_sparse_runtime_q_head_gather:finalize_cat",
                _t_finalize,
            )

            if perf_enabled and perf_local_ms:
                # Respect the same warmup-skip gate as ``_sparse_perf_record``
                # so this hot-path accumulator doesn't pollute the
                # steady-state window with early-step JIT / allocator /
                # GPU-clock ramp noise.
                if (
                    self._sparse_perf_total_steps
                    >= self._sparse_perf_warmup_skip
                ):
                    for k, ms in perf_local_ms.items():
                        self._sparse_perf_accum_ms[k] += ms
                        self._sparse_perf_accum_calls[k] += 1
            if publish_lmcache_rows and lmcache_rows_by_req:
                by_req = forward_additional_kwargs.setdefault(
                    "lmcache_per_head_token_indices_by_layer_by_req_id", {}
                )
                for rid, rows in lmcache_rows_by_req.items():
                    by_layer = by_req.setdefault(rid, {})
                    by_layer[layer_name] = rows
            return {
                "phys": flat_phys,
                "slots": flat_slots,
                "cu": cu_mat,
                "head_offsets": head_offsets,
                "max_k": max_k_hint,
                # Tier-2 pre-computed tensors consumed by the FA
                # ``_forward_per_head_compact_kv_gather`` fast path.  Their
                # presence lets FA skip ~15 tiny kernel launches per sparse
                # layer in decode (measured on our reference 20-sparse-layer
                # model, ~7 ms/step of savings).
                "phys_int64": flat_phys_int64,
                "slots_int64": flat_slots_int64,
                "kv_token_ids": kv_token_ids,
                "cu_k_flat": cu_k_flat,
                "cu_q_flat": cu_q_flat,
                "req_ids_flat": req_ids_flat,
                "kv_pair_ids_flat": kv_pair_ids_flat,
                "num_q_flat": num_q_flat,
                "num_q_heads": num_heads,
                "num_reqs": num_reqs,
            }
        finally:
            if _t0 is not None:
                self._sparse_perf_record(
                    "_build_sparse_runtime_q_head_gather",
                    time.perf_counter() - _t0,
                )

    def _sparse_online_dynamic_update(
        self,
        state: _SparseOnlineLayerState,
        *,
        append_buffered_features: bool = False,
    ) -> None:
        _t0 = time.perf_counter() if self._sparse_perf_stats_enabled else None
        try:
            if not state.decode_block_buffer:
                return
            feat = torch.stack(state.decode_block_buffer, dim=0).to(
                dtype=torch.float32
            )
            m = int(feat.shape[0])
            k_new = max(1, m // 16)
            k_new = (k_new // max(1, 32)) * 32
            k_new = max(k_new, 1)
            k_new = min(k_new, m)
            # Keep decode-side dynamic refresh on the original torch path.
            # The Triton K-Means rollout is intentionally limited to prefill
            # indexing first; dynamic refresh fires on the decode critical
            # path and should be enabled only after separate latency testing.
            raw = prefill_cluster_meta_from_features_torch(
                feat, num_clusters=k_new, n_segment=1
            )
            centres_new = raw["cluster_centres"]
            labels_new = raw["block_to_cluster"]
            sizes_new = raw["cluster_size"]
            n_existing = int(state.cluster_centres.shape[0])
            state.cluster_centres = torch.cat(
                [state.cluster_centres, centres_new.to(dtype=torch.float32)], dim=0
            )
            state.cluster_size = torch.cat(
                [
                    state.cluster_size,
                    sizes_new.to(
                        dtype=torch.int32, device=state.cluster_size.device
                    ),
                ],
                dim=0,
            )
            labels_new = labels_new.to(
                dtype=torch.int64, device=state.block_to_cluster.device
            )
            new_slot = labels_new + n_existing
            if append_buffered_features:
                state._grow_if_needed(m)
                start = state._len
                state._abf_storage[start : start + m].copy_(
                    feat.to(dtype=state._abf_storage.dtype)
                )
                state._b2c_storage[start : start + m].copy_(new_slot)
                state._len += m
            else:
                state.block_to_cluster[-m:] = new_slot
            state.decode_block_buffer.clear()
        finally:
            if _t0 is not None:
                self._sparse_perf_record(
                    "_sparse_online_dynamic_update",
                    time.perf_counter() - _t0,
                )

    def _update_sparse_online_index(
        self,
        sparse_block_features: dict[str, dict[str, np.ndarray]] | None,
        sparse_prefill_cluster_meta: dict[str, dict[str, dict[str, np.ndarray]]] | None,
        sparse_new_block_features: dict[str, dict[str, np.ndarray]] | None,
        sparse_new_block_features_gpu: dict[str, dict[str, torch.Tensor]] | None = None,
        sparse_prefill_block_features_gpu: (
            dict[str, dict[str, torch.Tensor]] | None
        ) = None,
        sparse_prefill_cluster_meta_gpu: (
            dict[str, dict[str, dict[str, torch.Tensor]]] | None
        ) = None,
    ) -> None:
        _t0 = time.perf_counter() if self._sparse_perf_stats_enabled else None
        try:
            def _init_prefill_unit(
                req_id: str,
                unit_key: str,
                feat_t: torch.Tensor,
                meta: dict[str, Any] | None,
            ) -> None:
                parsed = parse_sparse_kv_key(unit_key)
                if parsed is None:
                    return
                layer_name, _ = parsed
                spec = self._sparse_layer_spec_by_name.get(layer_name)
                if spec is None or spec.cluster_granularity != "token":
                    return
                if feat_t.device != self.device:
                    feat_t = feat_t.to(device=self.device)
                clusters_t = None
                compact_legacy = bool(
                    self._sparse_legacy_token_topk
                    and spec.use_compact_kv_gather
                )
                if meta is None:
                    raw = prefill_cluster_meta_from_features_device(
                        feat_t,
                        num_clusters=spec.num_clusters,
                        n_segment=spec.n_segment,
                    )
                    centres_t = raw["cluster_centres"].to(
                        dtype=torch.float32, device=self.device
                    )
                    labels_t = raw["block_to_cluster"].to(
                        dtype=torch.int64, device=self.device
                    )
                    sizes_t = raw["cluster_size"].to(
                        dtype=torch.int32, device=self.device
                    )
                    if compact_legacy:
                        mean_t = torch.empty(
                            0, dtype=torch.float32, device=self.device
                        )
                    else:
                        mean_t = raw["mean_key"].to(
                            dtype=torch.float32, device=self.device
                        )
                    if "clusters" in raw:
                        clusters_t = raw["clusters"].to(
                            dtype=torch.int32, device=self.device
                        )
                else:
                    centres_t = torch.as_tensor(
                        meta["cluster_centres"],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    labels_t = torch.as_tensor(
                        meta["block_to_cluster"],
                        dtype=torch.int64,
                        device=self.device,
                    )
                    sizes_t = torch.as_tensor(
                        meta["cluster_size"],
                        dtype=torch.int32,
                        device=self.device,
                    )
                    mean_src = meta.get("mean_key")
                    if compact_legacy or mean_src is None:
                        mean_t = torch.empty(
                            0, dtype=torch.float32, device=self.device
                        )
                    else:
                        mean_t = torch.as_tensor(
                            mean_src,
                            dtype=torch.float32,
                            device=self.device,
                        )
                    clusters_src = meta.get("clusters")
                    if clusters_src is not None:
                        clusters_t = torch.as_tensor(
                            clusters_src,
                            dtype=torch.int32,
                            device=self.device,
                        )
                pending_vs_map = self._sparse_pending_value_sum_gpu.get(req_id)
                vs_tensor = None
                if pending_vs_map is not None:
                    vs_tensor = pending_vs_map.get(unit_key)
                state_new = _SparseOnlineLayerState(
                    cluster_centres=centres_t,
                    cluster_size=sizes_t,
                    block_to_cluster=labels_t,
                    all_block_features=feat_t,
                    mean_key=mean_t,
                    value_sum=vs_tensor,
                    cluster_members=clusters_t,
                    copy_all_block_features=False,
                )
                n_tokens = int(labels_t.numel())
                k_clusters = int(centres_t.shape[0]) \
                    if centres_t.dim() >= 2 else 0
                head_n = min(int(spec.static_pattern_start), n_tokens)
                tail_start = max(0, n_tokens - int(spec.static_pattern_end))
                if (
                    not compact_legacy
                    and head_n < tail_start
                    and k_clusters > 0
                ):
                    positions = torch.arange(
                        head_n,
                        tail_start,
                        dtype=torch.int32,
                        device=labels_t.device,
                    )
                    state_new.rebuild_cluster_csr_from_labels(
                        labels=labels_t[head_n:tail_start],
                        positions=positions,
                        num_clusters=k_clusters,
                    )
                req_states = self._sparse_online_index.setdefault(req_id, {})
                req_states[unit_key] = state_new

            _t_prefill_init = (
                time.perf_counter()
                if self._sparse_perf_stats_enabled
                and (sparse_block_features or sparse_prefill_block_features_gpu)
                else None
            )
            if sparse_block_features:
                for req_id, unit_map in sparse_block_features.items():
                    meta_map = (
                        {} if sparse_prefill_cluster_meta is None else
                        sparse_prefill_cluster_meta.get(req_id, {})
                    )
                    for unit_key, feat_np in unit_map.items():
                        feat_t = torch.as_tensor(
                            feat_np, dtype=torch.float32, device=self.device
                        )
                        _init_prefill_unit(
                            req_id, unit_key, feat_t, meta_map.get(unit_key)
                        )
            if sparse_prefill_block_features_gpu:
                for req_id, unit_map in sparse_prefill_block_features_gpu.items():
                    meta_map = (
                        {} if sparse_prefill_cluster_meta_gpu is None else
                        sparse_prefill_cluster_meta_gpu.get(req_id, {})
                    )
                    for unit_key, feat_t in unit_map.items():
                        _init_prefill_unit(
                            req_id, unit_key, feat_t, meta_map.get(unit_key)
                        )
            if _t_prefill_init is not None:
                # Isolates the prefill-only state-materialisation block so
                # we can compare it cleanly against the batched decode path
                # (pass1/bucket/stack_bmm/bulk_append/bookkeep).  Fires only
                # on steps that observed a prefill completion, meaning the
                # per-call avg divided by #prefill steps gives the true
                # prefill-attributable cost of ``_update_sparse_online_index``.
                self._sparse_perf_record(
                    "_update_sparse_online_index:prefill_init",
                    time.perf_counter() - _t_prefill_init,
                )
            # Prefer the GPU-resident dict produced by
            # ``_collect_sparse_features``: it avoids the ``torch.as_tensor``
            # H2D sync per (req, unit_key) pair and keeps the new feature on
            # the same CUDA stream as the rest of the decode work.  The numpy
            # dict is kept around for backwards compatibility (and used as a
            # fallback when ``*_gpu`` is ``None``).
            iter_src: list[
                tuple[
                    str,
                    dict[str, torch.Tensor] | dict[str, np.ndarray],
                    bool,
                ]
            ]
            if sparse_new_block_features_gpu:
                iter_src = [
                    (rid, umap, True)
                    for rid, umap in sparse_new_block_features_gpu.items()
                ]
            elif sparse_new_block_features:
                iter_src = [
                    (rid, umap, False)
                    for rid, umap in sparse_new_block_features.items()
                ]
            else:
                iter_src = []

            # ------------------------------------------------------------
            # Pass 1: gather every (state, feat_t, spec) unit for this step.
            # Collecting once lets Pass 2 / Pass 3 batch across all
            # (layer, kv_head) pairs instead of launching small kernels in a
            # Python loop (which used to dominate the decode critical path –
            # 28 layers x 4 kv_heads x ~3 kernels / unit ~= 300 launches).
            # ------------------------------------------------------------
            _t_sub = (
                time.perf_counter()
                if self._sparse_perf_stats_enabled else None
            )
            pass1_states: list[_SparseOnlineLayerState] = []
            pass1_feats: list[torch.Tensor] = []
            pass1_specs: list[SparseAttentionSpec] = []
            pass1_defer_append: list[bool] = []
            for req_id, unit_map, is_gpu in iter_src:
                req_states = self._sparse_online_index.get(req_id)
                if not req_states:
                    continue
                for unit_key, feat in unit_map.items():
                    parsed = parse_sparse_kv_key(unit_key)
                    if parsed is None:
                        continue
                    layer_name, _ = parsed
                    spec = self._sparse_layer_spec_by_name.get(layer_name)
                    if spec is None or spec.cluster_granularity != "token":
                        continue
                    state = req_states.get(unit_key)
                    if state is None:
                        continue
                    defer_append = bool(
                        self._sparse_legacy_token_topk
                        and spec.use_compact_kv_gather
                    )
                    if is_gpu:
                        feat_t = feat
                        if not defer_append and feat_t.dtype != torch.float32:
                            feat_t = feat_t.to(dtype=torch.float32)
                    else:
                        feat_t = torch.as_tensor(
                            feat, dtype=torch.float32, device=self.device
                        )
                    pass1_states.append(state)
                    pass1_feats.append(feat_t)
                    pass1_specs.append(spec)
                    pass1_defer_append.append(defer_append)

            if _t_sub is not None:
                self._sparse_perf_record(
                    "_update_sparse_online_index:pass1_gather",
                    time.perf_counter() - _t_sub,
                )

            if not pass1_states:
                return

            device = pass1_feats[0].device
            immediate_idxs = [
                i for i, defer in enumerate(pass1_defer_append) if not defer
            ]

            # ------------------------------------------------------------
            # Pass 2 + Pass 3: bucket units by (K, D) and, for each bucket,
            # fuse the nearest-cluster argmax (single ``bmm``) with the
            # multi-tensor append (single ``torch._foreach_copy_``).
            #
            # Why bucket by D: ``torch.stack`` requires identical shapes, and
            # different layers can legally have different ``head_size`` (e.g.
            # MLA / MQA layers coexisting with GQA).  Within a bucket all
            # tensors share shape so we get a single ``bmm`` instead of one
            # small matmul per unit.  In the common homogeneous case there is
            # exactly one bucket, collapsing up to ~L*KH launches into O(1).
            # ------------------------------------------------------------
            _t_sub = (
                time.perf_counter()
                if self._sparse_perf_stats_enabled else None
            )
            buckets: dict[
                tuple[int, int], list[int]
            ] = {}
            for i in immediate_idxs:
                s = pass1_states[i]
                k = int(s.cluster_centres.shape[0])
                d = (
                    int(s.mean_key.shape[0])
                    if s.mean_key.dim() >= 1 else 0
                )
                buckets.setdefault((k, d), []).append(i)
            if _t_sub is not None:
                self._sparse_perf_record(
                    "_update_sparse_online_index:bucket",
                    time.perf_counter() - _t_sub,
                )

            for (k, d), idxs in buckets.items():
                _t_sub = (
                    time.perf_counter()
                    if self._sparse_perf_stats_enabled else None
                )
                bucket_states = [pass1_states[i] for i in idxs]
                bucket_feats = [pass1_feats[i] for i in idxs]
                feats_g = torch.stack(bucket_feats, dim=0)  # [G, D]
                g = feats_g.shape[0]

                if k == 0 or d == 0:
                    # Degenerate state – ``nearest`` is forced to 0 (matching
                    # the legacy scalar ``torch.zeros(())`` path) and we skip
                    # the bmm entirely.
                    nearest_g = torch.zeros(
                        g, dtype=torch.int64, device=device
                    )
                    if _t_sub is not None:
                        self._sparse_perf_record(
                            "_update_sparse_online_index:stack_bmm",
                            time.perf_counter() - _t_sub,
                        )
                else:
                    means_g = torch.stack(
                        [s.mean_key for s in bucket_states], dim=0
                    )  # [G, D]
                    centres_g = torch.stack(
                        [s.cluster_centres for s in bucket_states], dim=0
                    )  # [G, K, D]
                    centered = (feats_g - means_g).unsqueeze(-1)  # [G, D, 1]
                    centred_c = centres_g - means_g.unsqueeze(1)  # [G, K, D]
                    dots = torch.bmm(centred_c, centered).squeeze(-1)  # [G, K]
                    nearest_g = dots.argmax(dim=-1).to(dtype=torch.int64)
                    if _t_sub is not None:
                        self._sparse_perf_record(
                            "_update_sparse_online_index:stack_bmm",
                            time.perf_counter() - _t_sub,
                        )

                _t_sub = (
                    time.perf_counter()
                    if self._sparse_perf_stats_enabled else None
                )
                _SparseOnlineLayerState.bulk_append(
                    bucket_states, feats_g, nearest_g, perf_recorder=self
                )
                if _t_sub is not None:
                    self._sparse_perf_record(
                        "_update_sparse_online_index:bulk_append",
                        time.perf_counter() - _t_sub,
                    )

            # ------------------------------------------------------------
            # Pass 4: per-state bookkeeping that stays in Python (it's
            # trivially cheap – only list ``append`` and an ``int`` compare,
            # no GPU work).  Dynamic K-Means updates are rare (fire every
            # ``update_threshold_tokens`` decode steps) so we keep the
            # existing path intact instead of trying to batch them.
            # ------------------------------------------------------------
            _t_sub = (
                time.perf_counter()
                if self._sparse_perf_stats_enabled else None
            )
            for i, (state, feat_t, spec) in enumerate(
                zip(pass1_states, pass1_feats, pass1_specs, strict=True)
            ):
                # Legacy token compact gather includes generated tokens as
                # pending tail, so per-step nearest-cluster labels are not
                # consumed before the periodic dynamic update.  Buffer only
                # the K tensor and append clustered rows in that update.
                state.decode_block_buffer.append(feat_t)
                if (
                    len(state.decode_block_buffer)
                    >= int(spec.update_threshold_tokens)
                ):
                    self._sparse_online_dynamic_update(
                        state,
                        append_buffered_features=pass1_defer_append[i],
                    )
            if _t_sub is not None:
                self._sparse_perf_record(
                    "_update_sparse_online_index:bookkeep",
                    time.perf_counter() - _t_sub,
                )
        finally:
            # Phase 2b: release the GPU-resident value_sum handoff so the
            # prefill V tensors (``[H, K, D]`` per req) stop pinning memory
            # after they've been stitched into their ``_SparseOnlineLayerState``
            # counterparts.  Safe to clear unconditionally even when the
            # dict was never populated this step.
            if self._sparse_pending_value_sum_gpu:
                self._sparse_pending_value_sum_gpu.clear()
            if _t0 is not None:
                self._sparse_perf_record(
                    "_update_sparse_online_index",
                    time.perf_counter() - _t0,
                )

    def _decode_perf_flush_e2e(self, st_t0: float | None = None) -> None:
        """Emit the step-level end-to-end ``[DecodePerfE2E]`` log.

        Covers both ``execute_model`` and ``sample_tokens`` halves of the
        decode RPC, so callers can diff against ``[DecodePerf] total_ms``
        (which only covers ``execute_model``) to isolate work that happens
        between the two halves – notably the sparse feature collect and
        online index update executed inside ``sample_tokens``.
        """
        if (
            not self._decode_perf_stats_enabled
            or self._decode_perf_step_t0 is None
        ):
            return
        now_s = time.perf_counter()
        total_ms = (now_s - self._decode_perf_step_t0) * 1000.0
        exec_ms = self._decode_perf_step_exec_ms
        sample_ms = max(0.0, total_ms - exec_ms)
        # Debug variant: measure sample_tokens body directly from the entry
        # timestamp (st_t0) passed in by the caller, bypassing the indirection
        # via step_exec_ms.  If ``sample_body_ms`` is materially smaller than
        # ``sample_tokens_ms`` (the historical field), something between the
        # execute_model exec_ms stamp and sample_tokens entry is being
        # silently absorbed into the e2e sample_ms – that is our 70–80 ms
        # mystery gap.
        sample_body_ms = (
            (now_s - st_t0) * 1000.0 if st_t0 is not None else -1.0
        )
        pre_entry_ms = (
            (total_ms - sample_body_ms - exec_ms)
            if st_t0 is not None
            else -1.0
        )
        logger.info(
            "[DecodePerfE2E] mode=%s total_ms=%.3f execute_model_ms=%.3f "
            "sample_tokens_ms=%.3f sample_body_ms=%.3f pre_entry_ms=%.3f",
            "sparse" if self._has_sparse_attn else "full",
            total_ms,
            exec_ms,
            sample_ms,
            sample_body_ms,
            pre_entry_ms,
        )
        self._decode_perf_step_t0 = None
        self._decode_perf_step_exec_ms = 0.0

    def _sparse_perf_record(self, key: str, elapsed_s: float) -> None:
        if not self._sparse_perf_stats_enabled or not self._has_sparse_attn:
            return
        # Drop records emitted during the warmup window.  Checking the
        # counter here (rather than only gating at flush) keeps the
        # accumulators from growing unboundedly across a warmup window
        # that happens to straddle a log-interval boundary.
        if self._sparse_perf_total_steps < self._sparse_perf_warmup_skip:
            return
        self._sparse_perf_accum_ms[key] += float(elapsed_s) * 1000.0
        self._sparse_perf_accum_calls[key] += 1

    def _sparse_perf_flush_if_needed(self) -> None:
        if not self._sparse_perf_stats_enabled or not self._has_sparse_attn:
            return
        self._sparse_perf_total_steps += 1
        if self._sparse_perf_total_steps <= self._sparse_perf_warmup_skip:
            # Still within the warmup window.  Any records that slipped
            # through before the gate took effect (first step) are
            # cleared so the first post-warmup window starts clean.
            if self._sparse_perf_total_steps == self._sparse_perf_warmup_skip:
                self._sparse_perf_accum_ms.clear()
                self._sparse_perf_accum_calls.clear()
                logger.info(
                    "[SparsePerf] warmup complete after %d steps; "
                    "steady-state aggregation starts now",
                    self._sparse_perf_total_steps,
                )
            return
        self._sparse_perf_steps += 1
        if self._sparse_perf_steps % self._sparse_perf_log_interval != 0:
            return
        if not self._sparse_perf_accum_ms:
            logger.info(
                "[SparsePerf] steps=%d no sparse perf samples collected",
                self._sparse_perf_steps,
            )
            return

        items = sorted(
            self._sparse_perf_accum_ms.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        parts: list[str] = []
        for key, total_ms in items:
            calls = max(1, self._sparse_perf_accum_calls.get(key, 0))
            avg_ms = total_ms / calls
            parts.append(
                f"{key}:total_ms={total_ms:.2f},calls={calls},avg_ms={avg_ms:.3f}"
            )
        logger.info(
            "[SparsePerf] window_steps=%d %s",
            self._sparse_perf_log_interval,
            " | ".join(parts),
        )
        self._sparse_perf_accum_ms.clear()
        self._sparse_perf_accum_calls.clear()

    def _sparse_async_copy_to_cpu_pinned(self, t: torch.Tensor) -> torch.Tensor:
        """Enqueue async D2H copy into pinned CPU tensor on sparse copy stream."""
        assert self._sparse_d2h_stream is not None
        src = t.detach()
        dst = torch.empty_like(src, device="cpu", pin_memory=True)
        with torch.cuda.stream(self._sparse_d2h_stream):
            self._sparse_d2h_stream.wait_stream(torch.cuda.current_stream())
            dst.copy_(src, non_blocking=True)
        return dst

    def _sparse_log_first_sampled_token(
        self,
        *,
        req_idx: int,
        req_id: str,
        prev_output_n: int,
        sampled_ids: list[int],
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> None:
        if not self._has_sparse_attn:
            return
        if (
            not self._sparse_debug_first_token
            and not _SPARSE_HARD_DEBUG_FIRST_NEW_TOKEN
        ):
            return
        if req_id in self._sparse_first_token_sample_logged and prev_output_n == 0:
            return
        if not sampled_ids:
            return
        tok = int(sampled_ids[0])
        if tok < 0:
            return
        top_txt = ""
        top_pairs: list[tuple[int, float]] = []
        if logits is not None and logits.dim() >= 2 and req_idx < logits.size(0):
            row = logits[req_idx]
            k = min(8, row.numel())
            vals, idx = torch.topk(row, k=k)
            top_pairs = [
                (int(idx[i].item()), float(vals[i].item())) for i in range(k)
            ]
            if self._sparse_debug_tokenizer is not None:
                try:
                    parts = []
                    for tid, val in top_pairs[:5]:
                        t = self._sparse_debug_tokenizer.decode([tid])
                        parts.append(f"{tid}:{val:.3f}:{t!r}")
                    top_txt = " ".join(parts)
                except Exception:
                    top_txt = str(top_pairs[:5])
        sampled_txt = ""
        if self._sparse_debug_tokenizer is not None:
            try:
                sampled_txt = self._sparse_debug_tokenizer.decode([tok])
            except Exception:
                sampled_txt = ""
        spec_hint = (
            "spec_decode"
            if spec_decode_metadata is not None
            else "no_spec_decode"
        )
        logger.info(
            "[SparseFirstTok:sample] req_id=%s req_idx=%d tok_id=%d tok=%r "
            "top8=%s top5_detok=%s out_before=%d pp_rank=%s/%s %s",
            req_id,
            req_idx,
            tok,
            sampled_txt,
            top_pairs,
            top_txt,
            int(prev_output_n),
            get_pp_group().rank_in_group,
            get_pp_group().world_size,
            spec_hint,
        )
        if prev_output_n == 0:
            self._sparse_first_token_sample_logged.add(req_id)

    def _sparse_log_sample_step(
        self,
        *,
        req_idx: int,
        req_id: str,
        prev_output_n: int,
        sampled_ids: list[int],
        logits: torch.Tensor | None,
    ) -> None:
        """Lightweight per-step sampled-token log for early decode debugging."""
        if not self._has_sparse_attn:
            return
        if not (
            self._sparse_debug_first_token
            or _SPARSE_HARD_DEBUG_FIRST_NEW_TOKEN
        ):
            return
        if prev_output_n >= int(_SPARSE_HARD_DEBUG_STOP_AFTER_OUTPUT_N):
            return
        if not sampled_ids:
            return
        tok = int(sampled_ids[0])
        if tok < 0:
            return
        tok_txt = ""
        if self._sparse_debug_tokenizer is not None:
            try:
                tok_txt = self._sparse_debug_tokenizer.decode([tok])
            except Exception:
                tok_txt = ""
        top_txt = ""
        if logits is not None and logits.dim() >= 2 and req_idx < logits.size(0):
            row = logits[req_idx]
            k = min(3, row.numel())
            vals, idx = torch.topk(row, k=k)
            if self._sparse_debug_tokenizer is not None:
                try:
                    parts = []
                    for i in range(k):
                        tid = int(idx[i].item())
                        tv = float(vals[i].item())
                        tt = self._sparse_debug_tokenizer.decode([tid])
                        parts.append(f"{tid}:{tv:.3f}:{tt!r}")
                    top_txt = " ".join(parts)
                except Exception:
                    top_txt = str(
                        [
                            (int(idx[i].item()), float(vals[i].item()))
                            for i in range(k)
                        ]
                    )
        logger.info(
            "[SparseStep:sample] req_id=%s step=%d tok_id=%d tok=%r top3=%s",
            req_id,
            int(prev_output_n + 1),
            tok,
            tok_txt,
            top_txt,
        )

    def _sparse_store_prefill_block_features(
        self,
        req_id: str,
        unit_key: str,
        k_feat: torch.Tensor,
        spec: SparseAttentionSpec,
        sparse_block_features: dict[str, dict[str, np.ndarray]],
        sparse_prefill_cluster_meta: dict[str, dict[str, dict[str, np.ndarray]]],
        pending_async_feat: list[tuple[str, str, torch.Tensor]],
        pending_async_meta: list[tuple[str, str, dict[str, torch.Tensor]]],
        precomputed_meta: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Copy prefill K features to CPU and optionally run K-Means on device."""
        perf_enabled = self._sparse_perf_stats_enabled and self._has_sparse_attn
        use_async_d2h = (
            self._sparse_async_d2h_enabled
            and self._sparse_d2h_stream is not None
            and k_feat.is_cuda
        )
        raw = precomputed_meta
        if raw is None and sparse_prefill_cluster_use_device_kmeans(k_feat):
            _t_kmeans = time.perf_counter() if perf_enabled else None
            raw = prefill_cluster_meta_from_features_device(
                k_feat,
                num_clusters=spec.num_clusters,
                n_segment=spec.n_segment,
            )
            if _t_kmeans is not None:
                self._sparse_perf_record(
                    "collect:prefill_device_kmeans",
                    time.perf_counter() - _t_kmeans,
                )
        if raw is not None:
            if use_async_d2h:
                _t_meta_copy = time.perf_counter() if perf_enabled else None
                pending_async_meta.append(
                    (
                        req_id,
                        unit_key,
                        {
                            "cluster_centres": self._sparse_async_copy_to_cpu_pinned(
                                raw["cluster_centres"]
                            ),
                            "block_to_cluster": self._sparse_async_copy_to_cpu_pinned(
                                raw["block_to_cluster"]
                            ),
                            "cluster_size": self._sparse_async_copy_to_cpu_pinned(
                                raw["cluster_size"]
                            ),
                            "mean_key": self._sparse_async_copy_to_cpu_pinned(
                                raw["mean_key"]
                            ),
                        },
                    )
                )
                if _t_meta_copy is not None:
                    self._sparse_perf_record(
                        "collect:prefill_meta_d2h_enqueue",
                        time.perf_counter() - _t_meta_copy,
                    )
            else:
                _t_meta_copy = time.perf_counter() if perf_enabled else None
                sparse_prefill_cluster_meta.setdefault(req_id, {})[unit_key] = {
                    "cluster_centres": raw["cluster_centres"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False),
                    "block_to_cluster": raw["block_to_cluster"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int32, copy=False),
                    "cluster_size": raw["cluster_size"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int32, copy=False),
                    "mean_key": raw["mean_key"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False),
                }
                if _t_meta_copy is not None:
                    self._sparse_perf_record(
                        "collect:prefill_meta_cpu_numpy",
                        time.perf_counter() - _t_meta_copy,
                    )
        if use_async_d2h:
            _t_feat_copy = time.perf_counter() if perf_enabled else None
            pending_async_feat.append(
                (req_id, unit_key, self._sparse_async_copy_to_cpu_pinned(k_feat))
            )
            if _t_feat_copy is not None:
                self._sparse_perf_record(
                    "collect:prefill_feat_d2h_enqueue",
                    time.perf_counter() - _t_feat_copy,
                )
        else:
            _t_feat_copy = time.perf_counter() if perf_enabled else None
            sparse_block_features.setdefault(req_id, {})[unit_key] = (
                k_feat.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            if _t_feat_copy is not None:
                self._sparse_perf_record(
                    "collect:prefill_feat_cpu_numpy",
                    time.perf_counter() - _t_feat_copy,
                )

    def _sparse_store_prefill_kv_heads_block_features(
        self,
        req_id: str,
        layer_name: str,
        k_heads: torch.Tensor,
        spec: SparseAttentionSpec,
        sparse_block_features: dict[str, dict[str, np.ndarray]],
        sparse_prefill_cluster_meta: dict[str, dict[str, dict[str, np.ndarray]]],
        pending_async_feat: list[tuple[str, str, torch.Tensor]],
        pending_async_meta: list[tuple[str, str, dict[str, torch.Tensor]]],
        precomputed_meta: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Prefill sparse features for all KV heads of one layer in one batched K-Means.

        ``k_heads`` is ``[num_kv, N, D]`` (token or block rows).  Per-head outputs match
        ``_sparse_store_prefill_block_features`` for each ``layer##kv{h}`` key.
        """
        if k_heads.dim() != 3:
            raise ValueError(
                "batched sparse prefill expects k_heads [H, N, D], "
                f"got shape {tuple(k_heads.shape)}"
            )
        perf_enabled = self._sparse_perf_stats_enabled and self._has_sparse_attn
        use_async_d2h = (
            self._sparse_async_d2h_enabled
            and self._sparse_d2h_stream is not None
            and k_heads.is_cuda
        )
        num_kv = int(k_heads.shape[0])

        raw_b = precomputed_meta
        if raw_b is None and sparse_prefill_cluster_use_device_kmeans(k_heads):
            _t_kmeans = time.perf_counter() if perf_enabled else None
            raw_b = prefill_cluster_meta_from_features_device(
                k_heads,
                num_clusters=spec.num_clusters,
                n_segment=spec.n_segment,
            )
            if _t_kmeans is not None:
                self._sparse_perf_record(
                    "collect:prefill_device_kmeans",
                    time.perf_counter() - _t_kmeans,
                )
        if raw_b is not None:
            for kv_h in range(num_kv):
                unit_key = sparse_kv_unit_key(layer_name, kv_h)
                raw = {
                    "cluster_centres": raw_b["cluster_centres"][kv_h],
                    "block_to_cluster": raw_b["block_to_cluster"][kv_h],
                    "cluster_size": raw_b["cluster_size"][kv_h],
                    "mean_key": raw_b["mean_key"][kv_h],
                }
                if use_async_d2h:
                    _t_meta_copy = time.perf_counter() if perf_enabled else None
                    pending_async_meta.append(
                        (
                            req_id,
                            unit_key,
                            {
                                "cluster_centres": self._sparse_async_copy_to_cpu_pinned(
                                    raw["cluster_centres"]
                                ),
                                "block_to_cluster": self._sparse_async_copy_to_cpu_pinned(
                                    raw["block_to_cluster"]
                                ),
                                "cluster_size": self._sparse_async_copy_to_cpu_pinned(
                                    raw["cluster_size"]
                                ),
                                "mean_key": self._sparse_async_copy_to_cpu_pinned(
                                    raw["mean_key"]
                                ),
                            },
                        )
                    )
                    if _t_meta_copy is not None:
                        self._sparse_perf_record(
                            "collect:prefill_meta_d2h_enqueue",
                            time.perf_counter() - _t_meta_copy,
                        )
                else:
                    _t_meta_copy = time.perf_counter() if perf_enabled else None
                    sparse_prefill_cluster_meta.setdefault(req_id, {})[unit_key] = {
                        "cluster_centres": raw["cluster_centres"]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False),
                        "block_to_cluster": raw["block_to_cluster"]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int32, copy=False),
                        "cluster_size": raw["cluster_size"]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int32, copy=False),
                        "mean_key": raw["mean_key"]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False),
                    }
                    if _t_meta_copy is not None:
                        self._sparse_perf_record(
                            "collect:prefill_meta_cpu_numpy",
                            time.perf_counter() - _t_meta_copy,
                        )

        for kv_h in range(num_kv):
            unit_key = sparse_kv_unit_key(layer_name, kv_h)
            k_feat = k_heads[kv_h]
            if use_async_d2h:
                _t_feat_copy = time.perf_counter() if perf_enabled else None
                pending_async_feat.append(
                    (req_id, unit_key, self._sparse_async_copy_to_cpu_pinned(k_feat))
                )
                if _t_feat_copy is not None:
                    self._sparse_perf_record(
                        "collect:prefill_feat_d2h_enqueue",
                        time.perf_counter() - _t_feat_copy,
                    )
            else:
                _t_feat_copy = time.perf_counter() if perf_enabled else None
                sparse_block_features.setdefault(req_id, {})[unit_key] = (
                    k_feat.detach().cpu().numpy().astype(np.float32, copy=False)
                )
                if _t_feat_copy is not None:
                    self._sparse_perf_record(
                        "collect:prefill_feat_cpu_numpy",
                        time.perf_counter() - _t_feat_copy,
                    )

    def _collect_sparse_features(
        self,
        scheduler_output: "SchedulerOutput",
        num_reqs: int,
    ) -> tuple[
        "dict[str, dict[str, np.ndarray]] | None",
        "dict[str, dict[str, np.ndarray]] | None",
        "dict[str, dict[str, np.ndarray]] | None",
        "dict[str, dict[str, dict[str, np.ndarray]]] | None",
        "dict[str, dict[str, torch.Tensor]] | None",
        "dict[str, dict[str, torch.Tensor]] | None",
        "dict[str, dict[str, dict[str, torch.Tensor]]] | None",
    ]:
        """Extract block-level K features and per-request Q vectors.

        Called at the end of ``sample_tokens`` before building
        ``ModelRunnerOutput``.

        Feature convention
        ------------------
        * ``feature_dim = head_size`` – one vector per KV head / query head
          (no averaging across heads).
        * Keys use ``layer##kv{i}`` for K/V block features (clustering) and
          ``layer##qh{j}`` for query vectors and TopK selection.  ``SparseKVManager``
          runs K-Means per KV head and TopK per query head; the scheduler keeps
          the **union** of logical blocks for allocation while decode attention
          uses **per-query-head** block tables / compact gathers.

        Returns
        -------
        sparse_block_features
            ``req_id → layer##kv{i} → [num_blocks, head_size]`` CPU float32;
            emitted only when prefill completes this step.
        sparse_query_vectors
            ``req_id → layer##qh{j} → [head_size]`` CPU float32; every scheduled
            request (prefill completion or decode).
        sparse_new_block_features
            ``req_id → layer##kv{i} → [head_size]`` CPU float32; decode steps –
            K vector of the last slot per KV head.
        sparse_prefill_cluster_meta
            Optional GPU-computed K-Means metadata per ``layer##kv{i}`` (CPU numpy).
        sparse_new_block_features_gpu
            GPU-resident mirror of decode ``sparse_new_block_features`` for the
            in-process online index update.
        sparse_prefill_block_features_gpu / sparse_prefill_cluster_meta_gpu
            GPU-resident prefill handoff for token compact legacy TopK. These
            are internal to the runner and are not serialized to the scheduler.
        """
        perf_enabled = self._sparse_perf_stats_enabled and self._has_sparse_attn
        perf_local_ms: dict[str, float] = defaultdict(float)

        def _perf_add(name: str, start_t: float | None) -> None:
            if start_t is None:
                return
            perf_local_ms[name] += (time.perf_counter() - start_t) * 1000.0

        if not self._has_sparse_attn:
            return None, None, None, None, None, None, None

        sparse_groups = [
            (gid, grp)
            for gid, grp in enumerate(self.kv_cache_config.kv_cache_groups)
            if isinstance(grp.kv_cache_spec, SparseAttentionSpec)
        ]
        if not sparse_groups:
            return None, None, None, None, None, None, None

        sparse_block_features: dict[str, dict[str, np.ndarray]] = {}
        sparse_query_vectors: dict[str, dict[str, np.ndarray]] = {}
        sparse_new_block_features: dict[str, dict[str, np.ndarray]] = {}
        sparse_prefill_cluster_meta: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        sparse_prefill_block_features_gpu: dict[str, dict[str, torch.Tensor]] = {}
        sparse_prefill_cluster_meta_gpu: (
            dict[str, dict[str, dict[str, torch.Tensor]]]
        ) = {}
        # GPU-resident mirror of ``sparse_new_block_features`` for the in-worker
        # ``_update_sparse_online_index`` consumer.  Avoids the GPU→CPU→GPU
        # round-trip that otherwise happens per (req, layer, kv_head) pair on
        # every decode step.  The numpy dict above is still produced because
        # it is serialized into ``ModelRunnerOutput`` and consumed by the
        # scheduler across process boundaries.
        sparse_new_block_features_gpu: dict[str, dict[str, torch.Tensor]] = {}
        pending_async_feat: list[tuple[str, str, torch.Tensor]] = []
        pending_async_meta: list[tuple[str, str, dict[str, torch.Tensor]]] = []

        def _store_prefill_gpu_handoff(
            req_id: str,
            layer_name: str,
            k_heads: torch.Tensor,
            raw_b: dict[str, torch.Tensor],
        ) -> None:
            if k_heads.dim() != 3:
                raise ValueError(
                    "token prefill GPU handoff expects k_heads [H, N, D], "
                    f"got shape {tuple(k_heads.shape)}"
                )
            num_kv_local = int(k_heads.shape[0])
            feat_map = sparse_prefill_block_features_gpu.setdefault(req_id, {})
            meta_map = sparse_prefill_cluster_meta_gpu.setdefault(req_id, {})
            for kv_h in range(num_kv_local):
                unit_key = sparse_kv_unit_key(layer_name, kv_h)
                feat_map[unit_key] = k_heads[kv_h]
                meta = {
                    "cluster_centres": raw_b["cluster_centres"][kv_h],
                    "block_to_cluster": raw_b["block_to_cluster"][kv_h],
                    "cluster_size": raw_b["cluster_size"][kv_h],
                }
                if "clusters" in raw_b:
                    meta["clusters"] = raw_b["clusters"][kv_h]
                meta_map[unit_key] = meta
            if _SPARSE_TOKEN_TOPK_TRACE:
                centres = raw_b["cluster_centres"]
                cluster_count = (
                    int(centres.shape[1])
                    if centres.dim() == 3 else int(centres.shape[0])
                )
                logger.info(
                    "[SparseTokenTopK] prefill_gpu_handoff req_id=%s "
                    "layer=%s kv_heads=%d tokens=%d clusters=%d "
                    "cluster_members=%s",
                    req_id,
                    layer_name,
                    num_kv_local,
                    int(k_heads.shape[1]) if k_heads.dim() >= 2 else 0,
                    cluster_count,
                    "clusters" in raw_b,
                )

        # Diagnostic: full per-req body wall-clock (covers q_extract_total,
        # k_extract_total, and the Python bookkeeping between them).  Running
        # outer - (q_extract_total + k_extract_total) tells us how much time
        # is spent in per-req Python glue that is currently not sub-timed.
        _t_per_req_loop = time.perf_counter() if perf_enabled else None
        for req_idx in range(num_reqs):
            req_id = self.input_batch.req_ids[req_idx]
            if req_id not in scheduler_output.num_scheduled_tokens:
                continue  # not scheduled this step
            num_scheduled = scheduler_output.num_scheduled_tokens[req_id]
            if num_scheduled == 0:
                continue

            req_state = self.requests.get(req_id)
            if req_state is None:
                continue
            # IMPORTANT: classify phase using output length BEFORE this step
            # appends sampled ids. `_collect_sparse_features` runs after
            # `_bookkeeping_sync`, so reading current `output_token_ids` would
            # incorrectly treat the first decode step as already-decoded and
            # skip prefill indexing.
            num_output_before = self._sparse_output_tokens_before_step.get(
                req_id, len(req_state.output_token_ids)
            )

            num_prompt_tokens = int(
                self.input_batch.num_prompt_tokens[req_idx]
            )
            seq_len_after = int(self.seq_lens.np[req_idx])

            # Boundary case: first decode output is still pending (async/placeholders),
            # but seq_len may already exceed prompt length in this forward.
            # We still must emit prefill indexing features (prompt range only)
            # and query vectors so scheduler can produce sparse selections for
            # the next step.
            is_prefill_done = (
                num_output_before == 0
                and seq_len_after >= num_prompt_tokens
            )
            is_first_decode_boundary = (
                num_output_before == 0
                and seq_len_after > num_prompt_tokens
            )
            # Idempotence guard: once the prefill features were sent to the
            # scheduler (and consumed into the online index), the boundary case
            # above can fire a second time on the first decode step.  Re-sending
            # the [N_prompt_tokens, D] payload makes SparseKVManager.indexing()
            # rebuild per-token Python state again (~7-8s for N=24640).  Clear
            # the flag only when the request is finished (_update_states).
            if is_prefill_done and req_id in self._sparse_prefill_emitted:
                is_prefill_done = False
            # Treat decode as committed only when output existed before this
            # step. Boundary forwards (including async placeholder transitions)
            # should not emit decode new-block features.
            is_decode_committed = num_output_before > 0
            is_decode = is_decode_committed or is_first_decode_boundary

            if not is_prefill_done and not is_decode:
                # Mid-prefill chunk – nothing to emit yet.
                continue

            # Token index range for this request in the flat batch tensor.
            tok_start = int(self.query_start_loc.np[req_idx])
            tok_end = int(self.query_start_loc.np[req_idx + 1])
            last_tok_idx = tok_end - 1  # last token of this request

            # ── Q vectors: one row per query head (layer##qh{j})
            _t_q_extract = time.perf_counter() if perf_enabled else None
            q_per_unit: dict[str, np.ndarray] = {}
            for _, grp in sparse_groups:
                spec_q = grp.kv_cache_spec
                if (
                    isinstance(spec_q, SparseAttentionSpec)
                    and spec_q.cluster_granularity == "token"
                    and spec_q.use_compact_kv_gather
                ):
                    # Token compact gather scores Q against centroids in the
                    # runner pre-hook.  Emitting CPU Q vectors would only feed
                    # the scheduler-side selector that this path bypasses.
                    continue
                for layer_name in grp.layer_names:
                    q_raw = self._sparse_q_captures.get(layer_name)
                    if q_raw is None or last_tok_idx >= q_raw.shape[0]:
                        continue
                    attn_mod = (
                        self.compilation_config.static_forward_context.get(
                            layer_name
                        )
                    )
                    if attn_mod is None or attn_mod.num_heads <= 0:
                        continue
                    q_tok = q_raw[last_tok_idx].float()
                    head_size = int(attn_mod.head_size)
                    num_q = int(attn_mod.num_heads)
                    if q_tok.dim() == 2:
                        q_heads = q_tok
                    else:
                        q_heads = q_tok.view(num_q, head_size)
                    _t_q_cpu = time.perf_counter() if perf_enabled else None
                    q_heads_np = q_heads.cpu().numpy()
                    _perf_add("collect:q_cpu_numpy", _t_q_cpu)
                    for qh in range(num_q):
                        q_per_unit[sparse_qh_unit_key(layer_name, qh)] = (
                            q_heads_np[qh]
                        )
            _perf_add("collect:q_extract_total", _t_q_extract)

            if q_per_unit:
                sparse_query_vectors[req_id] = q_per_unit

            # ── K block features from KV cache (per layer, no cross-layer mean)
            _t_k_extract = time.perf_counter() if perf_enabled else None
            for gid, grp in sparse_groups:
                block_table = self.input_batch.block_table[gid]
                num_blocks = int(block_table.num_blocks_per_row[req_idx])
                if num_blocks == 0:
                    continue

                block_ids_np = block_table.block_table.np[
                    req_idx, :num_blocks
                ]
                block_ids_t: torch.Tensor | None = None

                def _get_block_ids_t() -> torch.Tensor:
                    nonlocal block_ids_t
                    if block_ids_t is None:
                        _t_block_ids = (
                            time.perf_counter() if perf_enabled else None
                        )
                        block_ids_t = torch.from_numpy(block_ids_np).to(
                            self.device
                        )
                        _perf_add("collect:block_ids_to_device", _t_block_ids)
                    assert block_ids_t is not None
                    return block_ids_t

                spec = grp.kv_cache_spec
                token_sparse = isinstance(
                    spec, SparseAttentionSpec
                ) and spec.cluster_granularity == "token"
                block_size = int(block_table.block_size)

                for layer_name in grp.layer_names:
                    attn_mod = (
                        self.compilation_config.static_forward_context.get(
                            layer_name
                        )
                    )
                    if (
                        attn_mod is None
                        or not attn_mod.kv_cache
                        or attn_mod.kv_cache[0] is None
                    ):
                        continue
                    kv = attn_mod.kv_cache[0]
                    # Fast-path: for a pure decode step in token-sparse mode we
                    # only need the newly-written K slot (single row), not the
                    # full block gather.  Skipping the full gather saves a
                    # per-layer copy proportional to ``num_blocks * block_size``
                    # and removes the dominant D2H in decode critical path.
                    decode_only_token_sparse = (
                        token_sparse
                        and is_decode_committed
                        and not is_prefill_done
                    )
                    k_blocks: torch.Tensor | None = None

                    def _read_k_blocks() -> torch.Tensor:
                        nonlocal k_blocks
                        if k_blocks is None:
                            _t_k_cache_read = (
                                time.perf_counter() if perf_enabled else None
                            )
                            k_blocks = kv[0][_get_block_ids_t()]
                            _perf_add("collect:k_cache_read", _t_k_cache_read)
                        return k_blocks

                    num_kv = int(attn_mod.num_kv_heads)
                    if token_sparse:
                        if is_prefill_done:
                            # Keep prefill index aligned to prompt tokens only.
                            valid_len = num_prompt_tokens
                            if valid_len > 0:
                                _t_prefill_store = (
                                    time.perf_counter() if perf_enabled else None
                                )
                                raw_b = None
                                prefill_gpu_handoff_allowed = bool(
                                    self._sparse_legacy_token_topk
                                    and spec.use_compact_kv_gather
                                )
                                if sparse_prefill_cluster_use_device_kmeans(kv[0]):
                                    _t_kmeans = (
                                        time.perf_counter()
                                        if perf_enabled
                                        else None
                                    )
                                    raw_b = prefill_cluster_meta_from_kv_cache_device(
                                        kv[0],
                                        _get_block_ids_t(),
                                        valid_len,
                                        value_cache=kv[1],
                                        num_clusters=spec.num_clusters,
                                        n_segment=spec.n_segment,
                                        is_centered=False,
                                        return_features=not (
                                            prefill_gpu_handoff_allowed
                                        ),
                                    )
                                    if _t_kmeans is not None:
                                        self._sparse_perf_record(
                                            "collect:prefill_device_kmeans",
                                            time.perf_counter() - _t_kmeans,
                                        )
                                    k_heads = raw_b["features"]
                                    # --- Phase 2b: per-cluster V accumulator ---
                                    # Compute ``value_sum`` on the same stream
                                    # while K-Means output ``block_to_cluster``
                                    # is still hot.  Uses ``kv[1]`` (V side of
                                    # the paged cache) with the same block_ids
                                    # gather; no extra D2H.  Only fires on the
                                    # device-kmeans path – CPU fallback leaves
                                    # ``value_sum`` as zeros (estimation zone
                                    # degrades to uniform, retrieval zone is
                                    # unaffected).
                                    #
                                    # Gate on the estimation budget: when the
                                    # runtime has no estimation zone configured
                                    # (``VLLM_SPARSE_ESTIMATION_BUDGET=0`` – the
                                    # default), the retroinfer path never reads
                                    # ``value_sum``, and computing it here
                                    # triggered a large transient FP32 upcast of
                                    # the V-cache slice in prefill (observed:
                                    # 770 MiB OOM in a 23.7 GiB card at long
                                    # prompts).  Skipping it keeps the prefill
                                    # peak memory equivalent to the
                                    # pre-Phase 2b path; the state constructor
                                    # still allocates a tiny ``[K, D]`` zeros
                                    # placeholder so FA's fallback path is
                                    # type-stable.
                                    head_n_cv = min(
                                        int(spec.static_pattern_start),
                                        valid_len,
                                    )
                                    tail_start_cv = max(
                                        0,
                                        valid_len
                                        - int(spec.static_pattern_end),
                                    )
                                    need_value_sum = (
                                        head_n_cv < tail_start_cv
                                        and int(
                                            self._sparse_estimation_budget
                                        ) > 0
                                    )
                                    if need_value_sum:
                                        _t_vs = (
                                            time.perf_counter()
                                            if perf_enabled
                                            else None
                                        )
                                        actual_k = int(
                                            raw_b["cluster_centres"].shape[1]
                                        )
                                        vs_hkd = value_sum_from_kv_cache_torch(
                                            kv[1],
                                            _get_block_ids_t(),
                                            valid_len,
                                            labels=raw_b[
                                                "block_to_cluster"
                                            ],
                                            num_clusters=actual_k,
                                            middle_start=head_n_cv,
                                            middle_end=tail_start_cv,
                                        )
                                        if vs_hkd.dim() == 2:
                                            vs_hkd = vs_hkd.unsqueeze(0)
                                        req_vs_map = (
                                            self
                                            ._sparse_pending_value_sum_gpu
                                            .setdefault(req_id, {})
                                        )
                                        for kv_h in range(num_kv):
                                            req_vs_map[
                                                sparse_kv_unit_key(
                                                    layer_name, kv_h
                                                )
                                            ] = vs_hkd[kv_h]
                                        if _t_vs is not None:
                                            self._sparse_perf_record(
                                                "collect:prefill_value_sum",
                                                time.perf_counter() - _t_vs,
                                            )
                                else:
                                    k_heads = kmeans_features_from_kv_cache_torch(
                                        kv[0],
                                        _get_block_ids_t(),
                                        valid_len,
                                        is_centered=False,
                                    )
                                use_gpu_prefill_handoff = (
                                    prefill_gpu_handoff_allowed
                                    and raw_b is not None
                                    and k_heads.is_cuda
                                )
                                if use_gpu_prefill_handoff:
                                    assert raw_b is not None
                                    _t_handoff = (
                                        time.perf_counter()
                                        if perf_enabled else None
                                    )
                                    _store_prefill_gpu_handoff(
                                        req_id, layer_name, k_heads, raw_b
                                    )
                                    _perf_add(
                                        "collect:prefill_gpu_handoff",
                                        _t_handoff,
                                    )
                                elif num_kv == 1:
                                    raw = None if raw_b is None else {
                                        "cluster_centres": raw_b[
                                            "cluster_centres"
                                        ][0],
                                        "block_to_cluster": raw_b[
                                            "block_to_cluster"
                                        ][0],
                                        "cluster_size": raw_b["cluster_size"][0],
                                        "mean_key": raw_b["mean_key"][0],
                                    }
                                    self._sparse_store_prefill_block_features(
                                        req_id,
                                        sparse_kv_unit_key(layer_name, 0),
                                        k_heads[0],
                                        spec,
                                        sparse_block_features,
                                        sparse_prefill_cluster_meta,
                                        pending_async_feat,
                                        pending_async_meta,
                                        precomputed_meta=raw,
                                    )
                                else:
                                    self._sparse_store_prefill_kv_heads_block_features(
                                        req_id,
                                        layer_name,
                                        k_heads,
                                        spec,
                                        sparse_block_features,
                                        sparse_prefill_cluster_meta,
                                        pending_async_feat,
                                        pending_async_meta,
                                        precomputed_meta=raw_b,
                                    )
                                _perf_add(
                                    "collect:prefill_store_features",
                                    _t_prefill_store,
                                )
                        if is_decode_committed:
                            last_g = seq_len_after - 1
                            slot = int(last_g % block_size)
                            defer_compact_update = bool(
                                self._sparse_legacy_token_topk
                                and spec.use_compact_kv_gather
                            )
                            if decode_only_token_sparse:
                                # Direct scalar read: no block_ids_t.to(device)
                                # needed, no full-gather allocation.
                                last_block_id = int(block_ids_np[-1])
                                _t_k_cache_read = (
                                    time.perf_counter() if perf_enabled else None
                                )
                                k_row_gpu = kv[0][last_block_id, slot]
                                _perf_add(
                                    "collect:k_cache_read", _t_k_cache_read
                                )
                            else:
                                k_row_gpu = _read_k_blocks()[-1][slot]
                            if not defer_compact_update:
                                k_row_gpu = k_row_gpu.float()
                            emit_decode_feature_to_scheduler = (
                                not bool(spec.use_compact_kv_gather)
                            )
                            k_row_np = None
                            if emit_decode_feature_to_scheduler:
                                _t_decode_cpu = (
                                    time.perf_counter() if perf_enabled else None
                                )
                                k_row_np = k_row_gpu.cpu().numpy()
                                _perf_add(
                                    "collect:k_decode_cpu_numpy", _t_decode_cpu
                                )
                            for kv_h in range(num_kv):
                                unit_key = sparse_kv_unit_key(layer_name, kv_h)
                                if num_kv == 1:
                                    k_row_one_gpu = (
                                        k_row_gpu[0]
                                        if k_row_gpu.dim() == 2
                                        else k_row_gpu
                                    )
                                    if k_row_np is not None:
                                        k_row_one_np = (
                                            k_row_np[0]
                                            if k_row_gpu.dim() == 2
                                            else k_row_np
                                        )
                                        sparse_new_block_features.setdefault(
                                            req_id, {}
                                        )[unit_key] = k_row_one_np
                                    sparse_new_block_features_gpu.setdefault(
                                        req_id, {}
                                    )[unit_key] = k_row_one_gpu
                                else:
                                    if k_row_np is not None:
                                        sparse_new_block_features.setdefault(
                                            req_id, {}
                                        )[unit_key] = k_row_np[kv_h]
                                    sparse_new_block_features_gpu.setdefault(
                                        req_id, {}
                                    )[unit_key] = k_row_gpu[kv_h]
                    else:
                        k_blk: torch.Tensor | None = None
                        raw_b = None
                        if is_prefill_done:
                            if sparse_prefill_cluster_use_device_kmeans(kv[0]):
                                _t_kmeans = (
                                    time.perf_counter()
                                    if perf_enabled
                                    else None
                                )
                                raw_b = prefill_cluster_meta_from_kv_cache_device(
                                    kv[0],
                                    _get_block_ids_t(),
                                    num_prompt_tokens,
                                    num_clusters=spec.num_clusters,
                                    n_segment=spec.n_segment,
                                    is_centered=True,
                                )
                                if _t_kmeans is not None:
                                    self._sparse_perf_record(
                                        "collect:prefill_device_kmeans",
                                        time.perf_counter() - _t_kmeans,
                                    )
                                k_heads = raw_b["features"]
                            else:
                                _t_block_mean = (
                                    time.perf_counter()
                                    if perf_enabled
                                    else None
                                )
                                k_heads = kmeans_features_from_kv_cache_torch(
                                    kv[0],
                                    _get_block_ids_t(),
                                    num_prompt_tokens,
                                    is_centered=True,
                                )
                                _perf_add("collect:k_block_mean", _t_block_mean)
                            k_blk = (
                                k_heads[0]
                                if num_kv == 1
                                else k_heads.transpose(0, 1).contiguous()
                            )
                        elif is_decode_committed:
                            _t_block_mean = (
                                time.perf_counter() if perf_enabled else None
                            )
                            k_blk = _read_k_blocks().mean(dim=1).float()
                            _perf_add("collect:k_block_mean", _t_block_mean)
                        if is_prefill_done:
                            _t_prefill_store = (
                                time.perf_counter() if perf_enabled else None
                            )
                            assert k_blk is not None
                            if num_kv == 1:
                                assert num_kv == 1, (
                                    "KV cache layout: expected num_kv_heads==1 "
                                    "when block mean has rank 2"
                                )
                                raw = None if raw_b is None else {
                                    "cluster_centres": raw_b["cluster_centres"][0],
                                    "block_to_cluster": raw_b["block_to_cluster"][0],
                                    "cluster_size": raw_b["cluster_size"][0],
                                    "mean_key": raw_b["mean_key"][0],
                                }
                                self._sparse_store_prefill_block_features(
                                    req_id,
                                    sparse_kv_unit_key(layer_name, 0),
                                    k_blk,
                                    spec,
                                    sparse_block_features,
                                    sparse_prefill_cluster_meta,
                                    pending_async_feat,
                                    pending_async_meta,
                                    precomputed_meta=raw,
                                )
                            else:
                                self._sparse_store_prefill_kv_heads_block_features(
                                    req_id,
                                    layer_name,
                                    k_heads,
                                    spec,
                                    sparse_block_features,
                                    sparse_prefill_cluster_meta,
                                    pending_async_feat,
                                    pending_async_meta,
                                    precomputed_meta=raw_b,
                                )
                            _perf_add(
                                "collect:prefill_store_features",
                                _t_prefill_store,
                            )
                        if is_decode_committed:
                            assert k_blk is not None
                            if k_blk.dim() == 2:
                                assert num_kv == 1
                                _t_decode_cpu = (
                                    time.perf_counter() if perf_enabled else None
                                )
                                sparse_new_block_features.setdefault(req_id, {})[
                                    sparse_kv_unit_key(layer_name, 0)
                                ] = k_blk[-1].cpu().numpy()
                                _perf_add("collect:k_decode_cpu_numpy", _t_decode_cpu)
                            else:
                                _t_decode_cpu = (
                                    time.perf_counter() if perf_enabled else None
                                )
                                k_last_np = k_blk[-1].cpu().numpy()
                                _perf_add("collect:k_decode_cpu_numpy", _t_decode_cpu)
                                for kv_h in range(num_kv):
                                    sparse_new_block_features.setdefault(req_id, {})[
                                        sparse_kv_unit_key(layer_name, kv_h)
                                    ] = k_last_np[kv_h]
            _perf_add("collect:k_extract_total", _t_k_extract)
            if is_prefill_done:
                # Latch so a repeat boundary fire on the next step cannot
                # push the same prefill payload through again.
                self._sparse_prefill_emitted.add(req_id)
        _perf_add("collect:per_req_loop_total", _t_per_req_loop)

        _t_post_loop = time.perf_counter() if perf_enabled else None
        if pending_async_feat or pending_async_meta:
            _t_wait = time.perf_counter() if perf_enabled else None
            assert self._sparse_d2h_stream is not None
            self._sparse_d2h_stream.synchronize()
            _perf_add("collect:async_d2h_wait", _t_wait)

            _t_finalize = time.perf_counter() if perf_enabled else None
            for req_id, unit_key, feat_cpu in pending_async_feat:
                sparse_block_features.setdefault(req_id, {})[unit_key] = (
                    feat_cpu.numpy().astype(np.float32, copy=False)
                )
            for req_id, unit_key, meta_cpu in pending_async_meta:
                sparse_prefill_cluster_meta.setdefault(req_id, {})[unit_key] = {
                    "cluster_centres": meta_cpu["cluster_centres"]
                    .numpy()
                    .astype(np.float32, copy=False),
                    "block_to_cluster": meta_cpu["block_to_cluster"]
                    .numpy()
                    .astype(np.int32, copy=False),
                    "cluster_size": meta_cpu["cluster_size"]
                    .numpy()
                    .astype(np.int32, copy=False),
                    "mean_key": meta_cpu["mean_key"]
                    .numpy()
                    .astype(np.float32, copy=False),
                }
            _perf_add("collect:async_d2h_finalize_numpy", _t_finalize)
        _perf_add("collect:post_loop_total", _t_post_loop)

        if self._sparse_probe_info_enabled:
            logger.info(
                "[SparseProbe] collect_sparse_features scheduled_reqs=%d "
                "prefill_feature_reqs=%d prefill_gpu_feature_reqs=%d "
                "query_reqs=%d decode_new_block_reqs=%d",
                len(scheduler_output.num_scheduled_tokens),
                len(sparse_block_features),
                len(sparse_prefill_block_features_gpu),
                len(sparse_query_vectors),
                len(sparse_new_block_features),
            )

        if perf_enabled and perf_local_ms:
            # Same warmup-skip gate as ``_sparse_perf_record`` to keep
            # the aggregated window clean of early-step ramp noise.
            if (
                self._sparse_perf_total_steps
                >= self._sparse_perf_warmup_skip
            ):
                for k, ms in perf_local_ms.items():
                    self._sparse_perf_accum_ms[k] += ms
                    self._sparse_perf_accum_calls[k] += 1

        return (
            sparse_block_features or None,
            sparse_query_vectors or None,
            sparse_new_block_features or None,
            sparse_prefill_cluster_meta or None,
            sparse_new_block_features_gpu or None,
            sparse_prefill_block_features_gpu or None,
            sparse_prefill_cluster_meta_gpu or None,
        )

    def _sparse_layer_decode_phys_row(
        self,
        req_id: str,
        kv_cache_gid: int,
        sparse_row_key: str,
    ) -> list[int] | None:
        """Physical KV block ids (history then decode slot) for one sparse row.

        ``sparse_row_key`` is ``layer##qh{j}`` (per query head).  When absent
        from the per-head map, falls back to merged union order (legacy).

        Uses the block_table (logical-index → physical-block-id mapping) to
        resolve physical addresses rather than zipping with bid_row, which
        can differ in length from the logical selection list.
        """
        rs = self.requests.get(req_id)
        if rs is None or len(rs.output_token_ids) == 0:
            return None
        merged = self._sparse_merged_logical.get(req_id)
        if not merged:
            return None
        bid_row = rs.block_ids[kv_cache_gid]
        if len(bid_row) < 2:
            return None
        decode_id = int(bid_row[-1])

        req_idx = self.input_batch.req_id_to_index.get(req_id)
        if req_idx is None:
            return None
        blk_table = self.input_batch.block_table[kv_cache_gid]
        bt_np = blk_table.block_table.np[req_idx]
        num_blocks = int(blk_table.num_blocks_per_row[req_idx])

        logical_to_phys: dict[int, int] = {}
        for log_idx in merged:
            li = int(log_idx)
            if 0 <= li < num_blocks:
                logical_to_phys[li] = int(bt_np[li])

        by_l = self._sparse_by_layer_logical.get(req_id)
        if by_l and sparse_row_key in by_l:
            logical_order = [int(x) for x in by_l[sparse_row_key]]
        else:
            logical_order = [int(x) for x in merged]
        row: list[int] = []
        for lg in logical_order:
            p = logical_to_phys.get(lg)
            if p is not None:
                row.append(p)
        row.append(decode_id)
        return row

    def _sparse_decode_num_blocks_np(
        self,
        kv_cache_gid: int,
        sparse_row_key: str,
        num_reqs: int,
        num_reqs_padded: int,
    ) -> np.ndarray:
        """Per-request sparse table width (in blocks) for one sparse row key."""
        out = np.full(num_reqs_padded, -1, dtype=np.int32)
        for req_idx in range(num_reqs):
            req_id = self.input_batch.req_ids[req_idx]
            row = self._sparse_layer_decode_phys_row(
                req_id, kv_cache_gid, sparse_row_key
            )
            if row is not None:
                out[req_idx] = int(len(row))
        return out

    def _build_sparse_layer_block_table_tensor(
        self,
        union_bt: torch.Tensor,
        kv_cache_gid: int,
        sparse_row_key: str,
        num_reqs: int,
        num_reqs_padded: int,
    ) -> torch.Tensor:
        """Shrink each sparse-decode row for ``sparse_row_key`` (``layer##qh{j}``)."""
        out = union_bt.clone()
        pad_id = -1
        for req_idx in range(num_reqs):
            req_id = self.input_batch.req_ids[req_idx]
            row = self._sparse_layer_decode_phys_row(
                req_id, kv_cache_gid, sparse_row_key
            )
            if row is None:
                continue
            out[req_idx].fill_(pad_id)
            row_t = torch.tensor(row, dtype=torch.int32, device=out.device)
            out[req_idx, : row_t.shape[0]] = row_t
        out[num_reqs:num_reqs_padded].fill_(pad_id)
        return out

    def _build_sparse_per_head_block_table_and_lens(
        self,
        union_bt: torch.Tensor,
        cm: "CommonAttentionMetadata",
        kv_cache_gid: int,
        layer_name: str,
        num_heads: int,
        num_reqs: int,
        num_reqs_padded: int,
        block_size: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Stack per-query-head sparse block tables ``[H, S, max_blocks]`` and seq lens."""
        _, max_blocks = union_bt.shape
        device = union_bt.device
        stacked = torch.full(
            (num_heads, num_reqs_padded, max_blocks),
            -1,
            dtype=torch.int32,
            device=device,
        )
        row_lens = np.full((num_heads, num_reqs_padded), -1, dtype=np.int32)
        any_sparse = False
        for qh in range(num_heads):
            sk = sparse_qh_unit_key(layer_name, qh)
            for req_idx in range(num_reqs):
                req_id = self.input_batch.req_ids[req_idx]
                row = self._sparse_layer_decode_phys_row(req_id, kv_cache_gid, sk)
                if row is None:
                    stacked[qh, req_idx] = union_bt[req_idx]
                else:
                    any_sparse = True
                    row_len = len(row)
                    row_lens[qh, req_idx] = row_len
                    stacked[qh, req_idx, :row_len] = torch.as_tensor(
                        row, dtype=torch.int32, device=device
                    )
        if not any_sparse:
            return None, None
        seq_lens_np = cm._seq_lens_cpu.numpy().copy()
        per_head_lens = np.zeros((num_heads, num_reqs_padded), dtype=np.int32)
        blk_table = self.input_batch.block_table[kv_cache_gid]
        for qh in range(num_heads):
            for req_idx in range(num_reqs):
                req_id = self.input_batch.req_ids[req_idx]
                req_state = self.requests.get(req_id)
                if req_state is None:
                    per_head_lens[qh, req_idx] = int(seq_lens_np[req_idx])
                    continue
                if len(req_state.output_token_ids) == 0:
                    per_head_lens[qh, req_idx] = int(seq_lens_np[req_idx])
                    continue
                old_seq_len = int(seq_lens_np[req_idx])
                num_blocks = int(blk_table.num_blocks_per_row[req_idx])
                ob = int(row_lens[qh, req_idx])
                if ob >= 0:
                    num_blocks = ob
                sparse_cap = num_blocks * block_size
                per_head_lens[qh, req_idx] = min(old_seq_len, sparse_cap)
        seq_t = torch.tensor(per_head_lens, dtype=torch.int32, device=device)
        return stacked, seq_t

    def _override_sparse_seq_lens(
        self,
        cm: "CommonAttentionMetadata",
        kv_cache_gid: int,
        num_reqs: int,
        num_reqs_padded: int,
        block_size: int,
        decode_num_blocks_override: np.ndarray | None = None,
    ) -> "CommonAttentionMetadata":
        """Bug 2 fix: replace seq_lens with sparse block-table length.

        For sparse decode requests the block table only contains the selected
        history blocks plus the single decode block.  The standard
        ``seq_lens`` value (full context length) would make the attention
        kernel attempt to read KV from blocks that don't exist in the sparse
        table, producing garbage outputs.

        This helper builds a corrected ``seq_lens`` GPU tensor where each
        decode request's length is capped to
        ``num_blocks_in_sparse_table * block_size``.
        Prefill requests are left unchanged (their block tables still cover
        the full prompt).
        """
        blk_table = self.input_batch.block_table[kv_cache_gid]
        seq_lens_np = cm._seq_lens_cpu.numpy().copy()

        for req_idx in range(num_reqs):
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests.get(req_id)
            if req_state is None:
                continue
            num_output_tokens = len(req_state.output_token_ids)
            if num_output_tokens == 0:
                continue  # prefill – leave seq_len as-is
            old_seq_len = int(seq_lens_np[req_idx])
            # Decode: cap to actual number of sparse blocks * block_size.
            num_blocks = int(blk_table.num_blocks_per_row[req_idx])
            if decode_num_blocks_override is not None and req_idx < len(
                decode_num_blocks_override
            ):
                ob = int(decode_num_blocks_override[req_idx])
                if ob >= 0:
                    num_blocks = ob
            sparse_cap_seq_len = num_blocks * block_size
            # Keep seq_len within sparse-table capacity, but never expose
            # unwritten decode-block tail beyond the currently materialized
            # sequence length.
            new_seq_len = min(old_seq_len, sparse_cap_seq_len)
            seq_lens_np[req_idx] = new_seq_len
            if self._sparse_probe_info_enabled or self._sparse_debug_decode_tokens:
                override_blocks = -1
                if decode_num_blocks_override is not None and req_idx < len(
                    decode_num_blocks_override
                ):
                    override_blocks = int(decode_num_blocks_override[req_idx])
                logger.info(
                    "[SparseRC] seq_lens req_id=%s gid=%d req_idx=%d "
                    "old_seq_len=%d new_seq_len=%d sparse_cap_seq_len=%d "
                    "num_blocks=%d block_size=%d decode_override=%d "
                    "num_output_tokens=%d",
                    req_id,
                    kv_cache_gid,
                    req_idx,
                    old_seq_len,
                    new_seq_len,
                    sparse_cap_seq_len,
                    num_blocks,
                    block_size,
                    override_blocks,
                    num_output_tokens,
                )
            # Diagnostic: sparse capacity can be larger than materialized
            # tokens while the decode block is not yet full. This is expected
            # as long as we keep applied seq_len equal to old_seq_len.
            tail_slack = sparse_cap_seq_len - old_seq_len
            if 0 < tail_slack < block_size:
                if new_seq_len != old_seq_len:
                    logger.warning_once(
                        "Sparse decode tail exposure detected: req_id=%s gid=%d "
                        "old_seq_len=%d sparse_cap_seq_len=%d applied_seq_len=%d "
                        "block_size=%d num_blocks=%d num_output_tokens=%d",
                        req_id,
                        kv_cache_gid,
                        old_seq_len,
                        sparse_cap_seq_len,
                        new_seq_len,
                        block_size,
                        num_blocks,
                        num_output_tokens,
                    )
                else:
                    logger.debug_once(
                        "Sparse decode tail slack observed and safely hidden "
                        "(applied_seq_len == old_seq_len)."
                    )
            self._debug_log_sparse_decode_tokens(
                req_id=req_id,
                req_idx=req_idx,
                seq_len=new_seq_len,
            )
            self._debug_log_sparse_forward_kv_tokens(
                req_id=req_id,
                req_idx=req_idx,
                seq_len=new_seq_len,
                kv_cache_gid=kv_cache_gid,
                block_size=block_size,
            )

        sparse_seq_lens_gpu = torch.tensor(
            seq_lens_np[:num_reqs_padded],
            dtype=torch.int32,
            device=self.device,
        )
        new_cm = copy(cm)
        new_cm.seq_lens = sparse_seq_lens_gpu
        new_cm._seq_lens_cpu = torch.tensor(
            seq_lens_np[:num_reqs_padded], dtype=torch.int32
        )
        if num_reqs > 0:
            new_cm.max_seq_len = int(seq_lens_np[:num_reqs].max())
        return new_cm

    def _debug_log_sparse_decode_tokens(
        self, req_id: str, req_idx: int, seq_len: int
    ) -> None:
        """Optional per-step sparse decode token trace for debugging."""
        if not self._sparse_debug_decode_tokens:
            return
        if seq_len <= 0:
            return
        token_ids = self.input_batch.token_ids_cpu[req_idx, :seq_len].tolist()
        max_show = max(1, self._sparse_debug_decode_tokens_max)
        shown_token_ids = token_ids[-max_show:]

        decoded_text = None
        token_pieces = None
        if self._sparse_debug_tokenizer is not None:
            try:
                decoded_text = self._sparse_debug_tokenizer.decode(
                    shown_token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except Exception:
                decoded_text = None
            try:
                token_pieces = self._sparse_debug_tokenizer.convert_ids_to_tokens(
                    shown_token_ids,
                    skip_special_tokens=False,
                )
            except Exception:
                token_pieces = None

        logger.info(
            "Sparse decode token trace: req_id=%s seq_len=%d shown=%d "
            "token_ids=%s token_pieces=%r decoded_tail=%r",
            req_id,
            seq_len,
            len(shown_token_ids),
            shown_token_ids,
            token_pieces,
            decoded_text,
        )
        tail_has_neg1 = any(tid == -1 for tid in shown_token_ids)
        first_neg1_rel = next(
            (idx for idx, tid in enumerate(shown_token_ids) if tid == -1),
            -1,
        )
        logger.info(
            "[SparseRC] token_tail req_id=%s req_idx=%d seq_len=%d shown=%d "
            "tail_has_neg1=%s first_neg1_rel=%d tail=%s",
            req_id,
            req_idx,
            seq_len,
            len(shown_token_ids),
            tail_has_neg1,
            first_neg1_rel,
            shown_token_ids,
        )

    def _debug_log_sparse_selected_kv_tokens(
        self,
        req_id: str,
        req_state: "CachedRequestState",
        selected_logical_blocks: list[int],
        zone_name: str = "merged",
    ) -> None:
        """Log original tokens that sparse decode KV actually uses."""
        if not self._sparse_debug_decode_tokens:
            return
        if req_state.prompt_token_ids is None:
            logger.debug(
                "Sparse %s KV trace skipped (no prompt token IDs): req_id=%s",
                zone_name,
                req_id,
            )
            return
        if not selected_logical_blocks:
            logger.debug(
                "Sparse %s KV trace: req_id=%s selected_blocks=[]",
                zone_name,
                req_id,
            )
            return

        all_token_ids = req_state.prompt_token_ids + req_state.output_token_ids
        # Debug mapping is for sparse-selected logical blocks; use sparse
        # KV group block_size from config (fall back to first block table size).
        block_size = None
        if hasattr(self, "kv_cache_config"):
            for grp in self.kv_cache_config.kv_cache_groups:
                if isinstance(grp.kv_cache_spec, SparseAttentionSpec):
                    block_size = int(grp.kv_cache_spec.block_size)
                    break
        if block_size is None:
            block_size = int(self.input_batch.block_table[0].block_size)
        max_blocks = self._sparse_debug_max_blocks
        shown_blocks = (
            selected_logical_blocks
            if max_blocks <= 0
            else selected_logical_blocks[: max(1, max_blocks)]
        )
        block_lines: list[str] = []

        for logical_block_idx in shown_blocks:
            start = int(logical_block_idx) * block_size
            end = min(start + block_size, len(all_token_ids))
            if start >= len(all_token_ids):
                continue
            block_token_ids = all_token_ids[start:end]
            block_text = None
            block_pieces = None
            if self._sparse_debug_tokenizer is not None and block_token_ids:
                try:
                    block_text = self._sparse_debug_tokenizer.decode(
                        block_token_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                except Exception:
                    block_text = None
                try:
                    block_pieces = self._sparse_debug_tokenizer.convert_ids_to_tokens(
                        block_token_ids,
                        skip_special_tokens=False,
                    )
                except Exception:
                    block_pieces = None
            block_lines.append(
                f"  - block={int(logical_block_idx)} span=[{start},{end}) "
                f"len={len(block_token_ids)} ids={block_token_ids} "
                f"pieces={block_pieces!r} text={block_text!r}"
            )

        logger.debug(
            "Sparse %s KV trace:\n"
            "req_id=%s\n"
            "selected_blocks(total=%d shown=%d, max=%d)=%s\n"
            "%s",
            zone_name,
            req_id,
            len(selected_logical_blocks),
            len(shown_blocks),
            max_blocks,
            shown_blocks,
            "\n".join(block_lines) if block_lines else "  - (no mappable blocks)",
        )

    def _debug_log_sparse_forward_kv_tokens(
        self,
        req_id: str,
        req_idx: int,
        seq_len: int,
        kv_cache_gid: int,
        block_size: int,
    ) -> None:
        """Log KV tokens that are actually consumed by sparse attention forward."""
        if not self._sparse_debug_decode_tokens:
            return
        if seq_len <= 0:
            return
        req_state = self.requests.get(req_id)
        if req_state is None or req_state.prompt_token_ids is None:
            logger.info(
                "Sparse forward KV trace skipped: req_id=%s reason=%s",
                req_id,
                "missing_request_state_or_prompt_ids",
            )
            return

        blk_table = self.input_batch.block_table[kv_cache_gid]
        num_blocks = int(blk_table.num_blocks_per_row[req_idx])
        block_ids = blk_table.block_table.np[req_idx, :num_blocks].tolist()
        used_blocks = min(num_blocks, cdiv(seq_len, block_size))
        used_block_ids = block_ids[:used_blocks]

        all_token_ids = req_state.prompt_token_ids + req_state.output_token_ids
        selected_logical_blocks = self._sparse_debug_selected_logical_blocks.get(
            req_id, []
        )

        max_blocks = self._sparse_debug_max_blocks
        shown_used_blocks = (
            list(range(used_blocks))
            if max_blocks <= 0
            else list(range(min(used_blocks, max(1, max_blocks))))
        )

        block_lines: list[str] = []
        for table_pos in shown_used_blocks:
            physical_block_id = int(used_block_ids[table_pos])
            block_token_ids: list[int] = []
            source = "unknown"

            if table_pos < len(selected_logical_blocks):
                logical_block_idx = int(selected_logical_blocks[table_pos])
                start = logical_block_idx * block_size
                end = min(start + block_size, len(all_token_ids))
                if start < len(all_token_ids):
                    block_token_ids = all_token_ids[start:end]
                source = f"logical_block={logical_block_idx}"
            else:
                # The extra block after selected history is the active decode block.
                # We only map the currently materialized tail tokens.
                compact_start = table_pos * block_size
                compact_end = min((table_pos + 1) * block_size, seq_len)
                n_tokens = max(0, compact_end - compact_start)
                if n_tokens > 0:
                    block_token_ids = all_token_ids[-n_tokens:]
                source = "decode_block_tail(best_effort)"

            block_text = None
            block_pieces = None
            if self._sparse_debug_tokenizer is not None and block_token_ids:
                try:
                    block_text = self._sparse_debug_tokenizer.decode(
                        block_token_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                except Exception:
                    block_text = None
                try:
                    block_pieces = self._sparse_debug_tokenizer.convert_ids_to_tokens(
                        block_token_ids,
                        skip_special_tokens=False,
                    )
                except Exception:
                    block_pieces = None

            block_lines.append(
                f"  - table_pos={table_pos} block_id={physical_block_id} "
                f"source={source} len={len(block_token_ids)} ids={block_token_ids} "
                f"pieces={block_pieces!r} text={block_text!r}"
            )

        logger.info(
            "Sparse forward KV trace:\n"
            "req_id=%s gid=%d seq_len=%d block_size=%d\n"
            "used_blocks(total=%d shown=%d, max=%d) block_ids=%s\n"
            "%s",
            req_id,
            kv_cache_gid,
            seq_len,
            block_size,
            used_blocks,
            len(shown_used_blocks),
            max_blocks,
            used_block_ids,
            "\n".join(block_lines) if block_lines else "  - (no mapped blocks)",
        )

    def _override_sparse_slot_mapping(
        self, num_reqs: int, cu_num_tokens: np.ndarray
    ) -> None:
        """Bug 3 fix: write sparse-decode tokens into the decode block slot.

        Standard ``compute_slot_mapping`` uses the full-context sequence
        position to index the block table:
          ``slot = block_table[req, position // block_size] * B + position % B``

        For sparse requests the block table only has ``k+1`` entries
        (k selected history blocks + 1 decode block), so
        ``position // block_size`` is typically far out of range and
        resolves to a null / stale block ID.

        For each sparse decode request we override the slot to:
          ``decode_block_id * block_size + fill_offset``
        where ``decode_block_id`` is the last (newest) block in the sparse
        table and ``fill_offset`` is the token's position within that block.
        ``fill_offset`` is derived from the number of already generated decode
        tokens modulo ``block_size``.

        Args:
            num_reqs: Number of active requests in the batch.
            cu_num_tokens: Cumulative scheduled-token counts per request,
                shape ``[num_reqs]`` (``cu_num_tokens[i]`` = end index of
                request i's tokens in the flat batch; start index of
                request 0 is 0, start of request i > 0 is
                ``cu_num_tokens[i-1]``).
        """
        sparse_debug_enabled = self._sparse_debug_decode_tokens

        if not hasattr(self, "kv_cache_config"):
            # TODO(sparse-debug): remove after locating sparse slot_mapping issues.
            if sparse_debug_enabled:
                logger.info(
                    "[SparseDebug] _override_sparse_slot_mapping "
                    "SKIP no kv_cache_config"
                )
            return

        sparse_gids = [
            gid
            for gid, grp in enumerate(self.kv_cache_config.kv_cache_groups)
            if isinstance(grp.kv_cache_spec, SparseAttentionSpec)
        ]
        if not sparse_gids:
            # TODO(sparse-debug): remove after locating sparse slot_mapping issues.
            if sparse_debug_enabled:
                logger.info(
                    "[SparseDebug] _override_sparse_slot_mapping "
                    "SKIP empty sparse_gids"
                )
            return

        # TODO(sparse-debug): remove after locating sparse slot_mapping issues.
        if sparse_debug_enabled:
            logger.info(
                "[SparseDebug] _override_sparse_slot_mapping ENTER num_reqs=%d "
                "sparse_gids=%s cu_num_tokens_tail=%s",
                int(num_reqs),
                sparse_gids,
                (
                    int(cu_num_tokens[num_reqs - 1])
                    if num_reqs > 0 and cu_num_tokens.size > 0
                    else -1
                ),
            )

        # For each sparse group, override the slot_mapping of decode reqs.
        for gid in sparse_gids:
            blk_table = self.input_batch.block_table[gid]
            block_size = blk_table.block_size

            for req_idx in range(num_reqs):
                req_id = self.input_batch.req_ids[req_idx]
                req_state = self.requests.get(req_id)

                # Token range for this request in the flat batch.
                tok_start = int(cu_num_tokens[req_idx - 1]) if req_idx > 0 else 0
                tok_end = int(cu_num_tokens[req_idx])
                num_sched = tok_end - tok_start

                # TODO(sparse-debug): remove after locating sparse slot_mapping issues.
                _out_n = (
                    len(req_state.output_token_ids) if req_state is not None else -1
                )
                if sparse_debug_enabled:
                    logger.info(
                        "[SparseDebug] _override_sparse_slot_mapping "
                        "gid=%d req_idx=%d req_id=%s has_req_state=%s "
                        "num_sched=%d tok_start=%d tok_end=%d "
                        "num_output_tokens=%d num_prompt_tokens=%s",
                        gid,
                        req_idx,
                        req_id,
                        req_state is not None,
                        num_sched,
                        tok_start,
                        tok_end,
                        _out_n,
                        (
                            int(self.input_batch.num_prompt_tokens[req_idx])
                            if req_idx < len(self.input_batch.num_prompt_tokens)
                            else None
                        ),
                    )

                if req_state is None or num_sched == 0:
                    continue

                num_output_tokens = len(req_state.output_token_ids)
                if num_output_tokens == 0:
                    # Prefill: leave slot_mapping unchanged.
                    continue

                # Decode: redirect to the last (active decode) block.
                # Sparse decode reuses this block until it is full, so writes
                # must start from the current in-block offset.
                num_blocks = int(blk_table.num_blocks_per_row[req_idx])
                if num_blocks == 0:
                    continue
                decode_block_id = int(
                    blk_table.block_table.np[req_idx, num_blocks - 1]
                )
                base_slot = decode_block_id * block_size
                fill_offset = int(num_output_tokens % block_size)
                # Override token slots for this request. For common decode
                # (num_sched=1), this writes one slot. For multi-token decode,
                # write contiguous offsets within the decode block.
                if num_sched == 1:
                    blk_table.slot_mapping.np[tok_start] = (
                        base_slot + min(fill_offset, block_size - 1)
                    )
                else:
                    fill_offsets = fill_offset + np.arange(
                        num_sched, dtype=np.int64
                    )
                    if fill_offset + num_sched > block_size:
                        logger.warning_once(
                            "Sparse decode token burst exceeds decode-block "
                            "capacity: req_id=%s gid=%d num_sched=%d "
                            "fill_offset=%d block_size=%d decode_block_id=%d",
                            req_id,
                            gid,
                            num_sched,
                            fill_offset,
                            block_size,
                            decode_block_id,
                        )
                    fill_offsets = np.minimum(fill_offsets, block_size - 1)
                    blk_table.slot_mapping.np[tok_start:tok_end] = (
                        base_slot + fill_offsets
                    )
                if self._sparse_probe_info_enabled or self._sparse_debug_decode_tokens:
                    mapped = blk_table.slot_mapping.np[tok_start:tok_end].tolist()
                    logger.info(
                        "[SparseRC] slot_map req_id=%s gid=%d req_idx=%d "
                        "tok_start=%d tok_end=%d num_sched=%d "
                        "decode_block_id=%d base_slot=%d fill_offset=%d mapped=%s",
                        req_id,
                        gid,
                        req_idx,
                        tok_start,
                        tok_end,
                        num_sched,
                        decode_block_id,
                        base_slot,
                        fill_offset,
                        mapped,
                    )
                if num_sched > 1:
                    logger.warning_once(
                        "Sparse decode multi-token slot mapping applied: "
                        "req_id=%s gid=%d num_sched=%d decode_block_id=%d "
                        "base_slot=%d tok_range=[%d,%d)",
                        req_id,
                        gid,
                        num_sched,
                        decode_block_id,
                        base_slot,
                        tok_start,
                        tok_end,
                    )

    # ── End sparse KV attention helpers ──────────────────────────────────────

    def init_routed_experts_capturer(self):
        logger.info(
            "Initializing routed experts capturer, enable_return_routed_experts: %s",
            self.model_config.enable_return_routed_experts,
        )
        routed_experts_capturer = RoutedExpertsCapturer.create()
        self.routed_experts_attn_gid = self._get_attention_kv_cache_gid()
        min_block_size = min(
            [
                group.kv_cache_spec.block_size
                for group in self.kv_cache_config.kv_cache_groups
            ]
        )
        num_groups = len(self.kv_cache_config.kv_cache_groups)
        self.max_num_kv_tokens = (
            self.kv_cache_config.num_blocks // num_groups
        ) * min_block_size
        dcp_size = self.vllm_config.parallel_config.decode_context_parallel_size
        pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size
        if pcp_size * dcp_size > 1:
            self.max_num_kv_tokens *= pcp_size * dcp_size

        routed_experts_capturer.init_buffer(
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            max_num_kv_tokens=self.max_num_kv_tokens,
            vllm_config=self.vllm_config,
        )
        self._bind_routed_experts_capturer(routed_experts_capturer)
        self.routed_experts_initialized = True

    def _bind_routed_experts_capturer(self, capturer: RoutedExpertsCapturer) -> None:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from vllm.model_executor.layers.fused_moe.router.base_router import (
            BaseRouter,
        )

        for module in self.compilation_config.static_forward_context.values():
            if isinstance(module, FusedMoE) and isinstance(module.router, BaseRouter):
                layer_id = module.layer_id

                def _capture_fn(topk_ids, _layer_id=layer_id, _capturer=capturer):
                    _capturer.capture(_layer_id, topk_ids)

                module.router.set_capture_fn(_capture_fn)

    def may_add_encoder_only_layers_to_kv_cache_config(self) -> None:
        """
        Add encoder-only layers to the KV cache config.
        """
        block_size = self.vllm_config.cache_config.block_size
        encoder_only_attn_specs: dict[AttentionSpec, list[str]] = defaultdict(list)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for layer_name, attn_module in attn_layers.items():
            if attn_module.attn_type == AttentionType.ENCODER_ONLY:
                attn_spec: AttentionSpec = EncoderOnlyAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                )
                encoder_only_attn_specs[attn_spec].append(layer_name)
                self.runner_only_attn_layers.add(layer_name)
        if len(encoder_only_attn_specs) > 0:
            assert len(encoder_only_attn_specs) == 1, (
                "Only support one encoder-only attention spec now"
            )
            spec, layer_names = encoder_only_attn_specs.popitem()
            self.kv_cache_config.kv_cache_groups.append(
                KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)
            )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        if has_ec_transfer() and not get_ec_transfer().is_consumer:
            return {}
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
        for layer_name, attn_module in attn_layers.items():
            if isinstance(attn_module, Attention) and (
                kv_tgt_layer := attn_module.kv_sharing_target_layer_name
            ):
                # The layer doesn't need its own KV cache and will use that of
                # the target layer. We skip creating a KVCacheSpec for it, so
                # that KV cache management logic will act as this layer does
                # not exist, and doesn't allocate KV cache for the layer. This
                # enables the memory saving of cross-layer kv sharing, allowing
                # a given amount of memory to accommodate longer context lengths
                # or enable more requests to be processed simultaneously.
                self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                continue
            # Skip modules that don't need KV cache (eg encoder-only attention)
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                kv_cache_spec[layer_name] = spec

        return kv_cache_spec

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        # This is a short term mitigation for issue mentioned in
        # https://github.com/vllm-project/vllm/issues/22754.
        # `tolist` would trigger a cuda wise stream sync, which
        # would block other copy ops from other cuda streams.
        # A cuda event sync would avoid such a situation. Since
        # this is in the critical path of every single model
        # forward loop, this has caused perf issue for a disagg
        # setup.
        pinned = self.sampled_token_ids_pinned_cpu[: sampled_token_ids.shape[0]]
        pinned.copy_(sampled_token_ids, non_blocking=True)
        self.transfer_event.record()
        self.transfer_event.synchronize()
        return pinned.tolist()

    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:
        """
        Get encoder timing stats for all requests and clear the registry.

        Returns:
            Dictionary mapping request_id to stats dict.
        """
        with self._encoder_timing_lock:
            stats = {
                req_id: stats_obj.to_dict()
                for req_id, stats_obj in self.encoder_timing_registry.items()
            }
            self.encoder_timing_registry.clear()
            return stats

    @contextmanager
    def timed_encoder_operation(
        self,
        should_time: bool,
        group_lora_refs: list[tuple[str, Any]],
        current_item_idx: int,
        num_items: int,
    ):
        """
        Context manager to time encoder forward operations.

        Args:
            should_time: Whether timing is enabled
            group_lora_refs: Full list of (request_id, pos_info) tuples
            current_item_idx: Starting index for this group
            num_items: Number of items in this group
        """
        if not should_time:
            yield
            return

        group_refs = group_lora_refs[current_item_idx : current_item_idx + num_items]
        group_request_ids = {req_id for req_id, _ in group_refs}

        torch.accelerator.synchronize()
        start_time = time.perf_counter()

        try:
            yield
        finally:
            torch.accelerator.synchronize()
            elapsed = time.perf_counter() - start_time

            per_request_time = elapsed / max(len(group_request_ids), 1)

            with self._encoder_timing_lock:
                for req_id in group_request_ids:
                    if req_id not in self.encoder_timing_registry:
                        self.encoder_timing_registry[req_id] = EncoderTimingStats()

                    stats = self.encoder_timing_registry[req_id]
                    stats.encoder_forward_secs += per_request_time
                    stats.num_encoder_calls += 1


@dataclass
class EncoderTimingStats:
    """Per-request timing statistics for encoder forward pass."""

    encoder_forward_secs: float = 0.0
    """Time spent in vision encoder forward pass (seconds)."""

    num_encoder_calls: int = 0
    """Number of times encoder was called for this request."""

    def to_dict(self) -> dict[str, float | int]:
        return {
            "encoder_forward_secs": self.encoder_forward_secs,
            "num_encoder_calls": self.num_encoder_calls,
        }
