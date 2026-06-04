# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse FlashAttention backend and implementation."""

import atexit
from collections import Counter
from dataclasses import dataclass, field, fields
import os
from typing import ClassVar

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    is_v1_kv_transfer_group,
)
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
    flash_attn_varlen_func,
)
from vllm.v1.core.sparse_kmeans_torch import (
    prefill_cluster_meta_from_features_torch_batched,
)
from vllm.v1.core.sparse_kv_utils import (
    SparseAppendBuffers,
    SparseBlockTableBuffers,
    SparseClusterBlockInfo,
    SparseCompactLoadSpec,
    SparseCompactStoreSpec,
    SparseManagerMetadata,
)

logger = init_logger(__name__)

_sparse_union_cluster_hist: Counter[int] = Counter()


def _print_sparse_union_cluster_hist() -> None:
    if not _sparse_union_cluster_hist:
        return
    total = sum(_sparse_union_cluster_hist.values())
    weighted = sum(k * v for k, v in _sparse_union_cluster_hist.items())
    items = ",".join(
        f"{k}:{_sparse_union_cluster_hist[k]}"
        for k in sorted(_sparse_union_cluster_hist)
    )
    print(
        "SPARSE_UNION_CLUSTER_HIST "
        f"total={total} mean={weighted / total:.3f} hist={items}",
        flush=True,
    )


atexit.register(_print_sparse_union_cluster_hist)


@dataclass
class SparseManagerExtraInfo:
    req_id_list: list[str] | None = None
    layer_name: str = ""
    num_cluster: int = 0
    num_segment: int = 1
    nprobe: int = 0
    gqa_topk_mode: str = "head_union"
    cluster_block_size: int = 0
    steady_start_capacity: int = 0
    steady_end_capacity: int = 0


@dataclass
class SparseFlashAttentionMetadata(FlashAttentionMetadata):
    cluster_allocated_block_info: (
        list[tuple[SparseClusterBlockInfo | None, ...]] | None
    ) = None
    sparse_manager_metadata: list[SparseManagerMetadata] | None = None
    extra_sparse_manager_info: SparseManagerExtraInfo = field(
        default_factory=SparseManagerExtraInfo
    )


class SparseFlashAttentionBackend(FlashAttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return FlashAttentionBackend.get_supported_kernel_block_sizes()

    @staticmethod
    def get_name() -> str:
        return "SPARSE_FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SparseFlashAttentionImpl"]:
        return SparseFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["SparseFlashAttentionMetadataBuilder"]:
        return SparseFlashAttentionMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return FlashAttentionBackend.supports_compute_capability(capability)

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
        reason = super().supports_combination(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            device_capability,
        )
        if reason is not None:
            return reason
        vllm_config = get_current_vllm_config_or_none()
        if (
            vllm_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
        ):
            return "sparse attention does not support decode context parallelism"
        return None


class SparseFlashAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        if (
            vllm_config.scheduler_config.max_num_seqs != 1
            or getattr(kv_cache_spec, "n_segment", 1) != 1
        ):
            return AttentionCGSupport.NEVER
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: "CommonAttentionMetadata",
        fast_build: bool = False,
    ) -> SparseFlashAttentionMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build=fast_build,
        )
        return SparseFlashAttentionMetadata(
            **{
                field.name: getattr(metadata, field.name)
                for field in fields(FlashAttentionMetadata)
            }
        )


class SparseFlashAttentionImpl(FlashAttentionImpl):
    @staticmethod
    def _ensure_cluster_compact_capacity(
        smm: SparseManagerMetadata,
        hkv: int,
        num_clusters: int,
        required_blocks: int,
        device: torch.device,
    ) -> None:
        required_blocks = max(1, int(required_blocks))
        if (
            smm.cluster_compact_block_ids is not None
            and smm.cluster_compact_block_ids.shape[0] == hkv
            and smm.cluster_compact_block_ids.shape[1] == num_clusters
            and smm.cluster_compact_block_ids.shape[2] >= required_blocks
        ):
            return
        old = smm.cluster_compact_block_ids
        old_cap = 0 if old is None else old.shape[2]
        new_cap = max(required_blocks, max(old_cap * 2, SparseManagerMetadata.INIT_CLUSTER_BLOCK_COUNT))
        new_tensor = torch.full(
            (hkv, num_clusters, new_cap),
            fill_value=-1,
            dtype=torch.int32,
            device=device,
        )
        if old is not None:
            new_tensor[:, :, : old.shape[2]].copy_(old)
        smm.cluster_compact_block_ids = new_tensor

    @staticmethod
    def _supports_sparse_compact_kv_transfer() -> bool:
        if not has_kv_transfer_group():
            return False
        connector = get_kv_transfer_group()
        return is_v1_kv_transfer_group(connector) and getattr(
            connector,
            "supports_sparse_compact_kv_transfer",
            False,
        )

    def _append_pending_compact_store_specs(
        self,
        *,
        smm: SparseManagerMetadata,
        prev_full_block_counts: torch.Tensor,
        new_full_block_counts: torch.Tensor,
    ) -> None:
        if not self._supports_sparse_compact_kv_transfer():
            return
        assert smm.cluster_compact_block_ids is not None
        if prev_full_block_counts.numel() == 0:
            return

        block_deltas = new_full_block_counts - prev_full_block_counts
        max_new_blocks = smm.cluster_compact_block_ids.shape[2]
        if max_new_blocks <= 0:
            return

        block_offsets = torch.arange(
            max_new_blocks,
            device=block_deltas.device,
            dtype=block_deltas.dtype,
        ).view(1, 1, -1)
        new_block_mask = block_offsets < block_deltas.unsqueeze(-1)
        new_block_coords = new_block_mask.nonzero(as_tuple=False)
        if new_block_coords.numel() == 0:
            return

        block_indices = (
            prev_full_block_counts.unsqueeze(-1) + block_offsets
        )[new_block_coords[:, 0], new_block_coords[:, 1], new_block_coords[:, 2]]
        src_block_ids = smm.cluster_compact_block_ids[
            new_block_coords[:, 0],
            new_block_coords[:, 1],
            block_indices.to(dtype=torch.long),
        ]
        spec_rows = torch.stack(
            (
                new_block_coords[:, 0].to(dtype=torch.int32),
                new_block_coords[:, 1].to(dtype=torch.int32),
                block_indices.to(dtype=torch.int32),
                src_block_ids.to(dtype=torch.int32),
            ),
            dim=1,
        ).to(device="cpu")
        smm.pending_compact_store_specs.extend(
            SparseCompactStoreSpec(
                kv_head_idx=kv_head_idx,
                cluster_idx=cluster_idx,
                block_idx=block_idx,
                src_block_id=src_block_id,
            )
            for kv_head_idx, cluster_idx, block_idx, src_block_id in spec_rows.tolist()
        )

    @staticmethod
    def _max_cluster_full_blocks_upper_bound(
        smm: SparseManagerMetadata,
        cluster_block_size: int,
        incoming_tokens: int = 0,
    ) -> int:
        clustered_token_count = smm.clustered_token_count
        if incoming_tokens > 0:
            clustered_token_count += smm.count_steady_evictions(incoming_tokens)
        return clustered_token_count // cluster_block_size

    def _select_grouped_top_clusters(
        self,
        *,
        query: torch.Tensor,
        attn_metadata: SparseFlashAttentionMetadata,
        smm: SparseManagerMetadata,
        hkv: int,
    ) -> torch.Tensor:
        assert smm.mean is not None
        assert smm.cluster_centers_T is not None
        assert smm.block_table_buffers is not None
        cluster_centers_t = smm.cluster_centers_T
        nprobe = min(
            attn_metadata.extra_sparse_manager_info.nprobe,
            cluster_centers_t.shape[2],
        )
        gqa_topk_mode = attn_metadata.extra_sparse_manager_info.gqa_topk_mode
        if gqa_topk_mode == "group_avg":
            return smm.block_table_buffers.group_avg_top_clusters_by_kv_group(
                query=query,
                cluster_centers_T=cluster_centers_t,
                mean=smm.mean,
                cluster_center_count=smm.cluster_center_count,
                nprobe=nprobe,
                hkv=hkv,
            )

        hq = query.shape[1]
        q_per_kv = hq // hkv
        top_clusters = smm.block_table_buffers.select_top_clusters(
            query=query,
            cluster_centers_T=cluster_centers_t,
            mean=smm.mean,
            cluster_center_count=smm.cluster_center_count,
            nprobe=nprobe,
        )
        union_nprobe = min(cluster_centers_t.shape[2], nprobe * q_per_kv)
        grouped_top_clusters = (
            smm.block_table_buffers.union_top_clusters_by_kv_group(
                top_clusters=top_clusters,
                hkv=hkv,
                num_clusters=cluster_centers_t.shape[2],
                union_nprobe=union_nprobe,
            )
        )
        if os.environ.get("TEST_SPARSE_UNION_STATS", "0") != "0":
            counts = (
                (grouped_top_clusters >= 0)
                .sum(dim=-1)
                .detach()
                .to(device="cpu")
                .reshape(-1)
                .tolist()
            )
            _sparse_union_cluster_hist.update(int(count) for count in counts)
        return grouped_top_clusters

    def prepare_layer_for_load(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor | None,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        attn_metadata: SparseFlashAttentionMetadata,
    ) -> None:
        del layer, value
        if query is None or key is None:
            return
        sparse_manager_metadata = attn_metadata.sparse_manager_metadata
        cluster_allocated_block_info = attn_metadata.cluster_allocated_block_info
        if (
            sparse_manager_metadata is None
            or not sparse_manager_metadata
            or cluster_allocated_block_info is None
            or not cluster_allocated_block_info
        ):
            return
        smm = sparse_manager_metadata[0]
        cluster_info = cluster_allocated_block_info[0][0]
        if cluster_info is None or smm.mean is None:
            return

        num_actual_tokens = attn_metadata.num_actual_tokens
        query_actual = query[:num_actual_tokens]
        if attn_metadata.max_query_len == attn_metadata.max_seq_len:
            smm.active_top_clusters = None
            smm.active_grouped_top_clusters = None
            smm.pending_compact_load_specs.clear()
            return

        hkv = key.shape[1]
        if smm.block_table_buffers is None:
            smm.block_table_buffers = SparseBlockTableBuffers()
        grouped_top_clusters = self._select_grouped_top_clusters(
            query=query_actual,
            attn_metadata=attn_metadata,
            smm=smm,
            hkv=hkv,
        )
        smm.active_grouped_top_clusters = grouped_top_clusters

        if not self._supports_sparse_compact_kv_transfer():
            smm.pending_compact_load_specs.clear()
            return

        assert smm.cluster_total_kv_counts is not None
        cluster_block_size = int(attn_metadata.extra_sparse_manager_info.cluster_block_size)
        full_block_counts = torch.div(
            smm.cluster_total_kv_counts,
            cluster_block_size,
            rounding_mode="floor",
        )
        max_full_blocks = self._max_cluster_full_blocks_upper_bound(
            smm,
            cluster_block_size,
        )
        self._ensure_cluster_compact_capacity(
            smm,
            hkv,
            int(smm.cluster_total_kv_counts.shape[1]),
            max_full_blocks,
            query.device,
        )
        assert smm.cluster_compact_block_ids is not None

        new_block_map = {
            (spec.kv_head_idx, spec.cluster_idx, spec.block_idx): spec.src_block_id
            for spec in smm.pending_compact_store_specs
        }
        load_specs: list[SparseCompactLoadSpec] = []
        pending_block_updates: list[tuple[int, int, int, int]] = []
        free_block_ids = cluster_info.allocated_block_ids_gpu
        assert free_block_ids is not None
        grouped_top_clusters_cpu = (
            grouped_top_clusters.permute(1, 0, 2)
            .contiguous()
            .reshape(hkv, -1)
            .to(device="cpu")
        )
        full_block_counts_cpu = full_block_counts.to(device="cpu")
        free_block_ids_cpu = free_block_ids.to(device="cpu")
        dst_cursor = len(smm.pending_compact_store_specs)
        for kv_head_idx in range(hkv):
            selected = {
                cluster_idx
                for cluster_idx in grouped_top_clusters_cpu[kv_head_idx].tolist()
                if cluster_idx >= 0
            }
            for cluster_idx in selected:
                full_count = int(full_block_counts_cpu[kv_head_idx, cluster_idx])
                if full_count <= 0:
                    continue
                if full_count > smm.cluster_compact_block_ids.shape[2]:
                    self._ensure_cluster_compact_capacity(
                        smm,
                        hkv,
                        int(smm.cluster_total_kv_counts.shape[1]),
                        full_count,
                        query.device,
                    )
                for block_idx in range(full_count):
                    block_key = (kv_head_idx, cluster_idx, block_idx)
                    block_id = new_block_map.get(block_key)
                    if block_id is None:
                        if dst_cursor >= free_block_ids_cpu.numel():
                            raise RuntimeError(
                                "Sparse compact scratch pool is undersized for selected clusters."
                            )
                        block_id = int(free_block_ids_cpu[dst_cursor])
                        load_specs.append(
                            SparseCompactLoadSpec(
                                kv_head_idx=kv_head_idx,
                                cluster_idx=cluster_idx,
                                block_idx=block_idx,
                                dst_block_id=block_id,
                            )
                        )
                        dst_cursor += 1
                    pending_block_updates.append(
                        (kv_head_idx, cluster_idx, block_idx, block_id)
                    )
        if pending_block_updates:
            update_rows = torch.tensor(
                pending_block_updates,
                dtype=torch.int64,
                device=query.device,
            )
            smm.cluster_compact_block_ids[
                update_rows[:, 0],
                update_rows[:, 1],
                update_rows[:, 2],
            ] = update_rows[:, 3].to(dtype=torch.int32)
        smm.pending_compact_load_specs = load_specs

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SparseFlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens
        query_actual = query[:num_actual_tokens]
        key_actual = key[:num_actual_tokens]
        value_actual = value[:num_actual_tokens]
        output_actual = output[:num_actual_tokens]
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        if self.dcp_world_size > 1:
            raise NotImplementedError(
                "SparseFlashAttentionImpl does not support decode context "
                "parallelism."
            )

        key_cache, value_cache = kv_cache.unbind(0)
        if self.kv_cache_dtype.startswith("fp8"):
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        if attn_metadata.use_cascade:
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

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

        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        sparse_manager_metadata = attn_metadata.sparse_manager_metadata
        cluster_allocated_block_info = attn_metadata.cluster_allocated_block_info
        use_sparse_cluster = (
            sparse_manager_metadata is not None
            and len(sparse_manager_metadata) > 0
            and sparse_manager_metadata[0].mean is not None
            and cluster_allocated_block_info is not None
            and len(cluster_allocated_block_info) > 0
        )
        assert use_sparse_cluster, (
            "SparseFlashAttentionImpl requires sparse cluster metadata."
        )
        cluster_storage = kv_cache.reshape(
            kv_cache.shape[0], kv_cache.shape[1], -1, kv_cache.shape[4]
        )
        req_index = 0
        smm = sparse_manager_metadata[0]
        cluster_allocated_block_info = cluster_allocated_block_info[req_index][0]

        is_prefill = max_seqlen_q == max_seqlen_k
        if is_prefill:
            flash_attn_varlen_func(
                q=query_actual,
                k=key_actual,
                v=value_actual,
                out=output_actual,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                cu_seqlens_k=cu_seqlens_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=sliding_window_size,
                block_table=None,
                softcap=self.logits_soft_cap,
                scheduler_metadata=scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                num_splits=attn_metadata.max_num_splits,
                s_aux=self.sinks,
            )
            return output

        assert smm.mean is not None
        assert smm.cluster_centers_T is not None
        cluster_block_size = cluster_storage.shape[2]
        num_queries = num_actual_tokens
        hq = query.shape[1]
        hkv = key.shape[1]
        assert hq % hkv == 0, "query heads must be divisible by kv heads"
        q_per_kv = hq // hkv
        max_blocks = block_table.shape[-1]
        nprobe = attn_metadata.extra_sparse_manager_info.nprobe

        cluster_centers_t = smm.cluster_centers_T
        nprobe = min(nprobe, cluster_centers_t.shape[2])
        assert smm.cluster_center_count is not None
        if smm.block_table_buffers is None:
            smm.block_table_buffers = SparseBlockTableBuffers()
        assert smm.block_table_buffers is not None
        gqa_topk_mode = attn_metadata.extra_sparse_manager_info.gqa_topk_mode
        free_block_ids = cluster_allocated_block_info.reusable_block_ids_gpu
        assert free_block_ids is not None
        assert smm.steady_zone_head is not None
        assert smm.steady_zone_tail is not None
        assert smm.steady_state is not None
        steady_blocks = smm.steady_zone_head.shape[1] + smm.steady_zone_tail.shape[1]
        use_grouped_sparse_fa = os.environ.get("TEST_SPARSE_GROUPED_FA", "1") != "0"
        if not use_grouped_sparse_fa:
            top_clusters = smm.block_table_buffers.select_top_clusters(
                query=query_actual,
                cluster_centers_T=cluster_centers_t,
                mean=smm.mean,
                cluster_center_count=smm.cluster_center_count,
                nprobe=nprobe,
            )
            required_free_blocks = num_queries * hq * (nprobe + steady_blocks)
            assert free_block_ids.numel() >= required_free_blocks, (
                "Sparse reusable scratch pool is undersized: "
                f"have {free_block_ids.numel()}, need at least "
                f"{required_free_blocks}"
            )
            block_table, _, seqused_k = smm.block_table_buffers.build(
                top_clusters=top_clusters,
                cluster_compact_block_ids=smm.cluster_compact_block_ids,
                cluster_temp_kv_pos=smm.cluster_temp_kv_pos,
                cluster_total_kv_counts=smm.cluster_total_kv_counts,
                temp_block_ids=smm.temp_block_ids,
                block_storage=cluster_storage,
                free_block_ids=free_block_ids,
                steady_start_block_ids=smm.steady_zone_head,
                steady_end_block_ids=smm.steady_zone_tail,
                steady_state=smm.steady_state,
                max_bt_len=max_blocks,
            )

            num_rows = num_queries * hq
            device = query.device
            q_flat = query_actual.reshape(num_rows, 1, -1)
            out_flat = output_actual.reshape(num_rows, 1, -1)
            seqused_k_flat = seqused_k.reshape(num_rows)
            block_table_flat = block_table.reshape(num_rows, -1)
            if (
                smm.cu_seqlens_q_buffer is None
                or smm.cu_seqlens_q_buffer.shape[0] < num_rows + 1
                or smm.cu_seqlens_q_buffer.dtype != cu_seqlens_q.dtype
                or smm.cu_seqlens_q_step != 1
            ):
                smm.cu_seqlens_q_buffer = torch.arange(
                    num_rows + 1,
                    device=device,
                    dtype=cu_seqlens_q.dtype,
                )
                smm.cu_seqlens_q_step = 1
            cu_seqlens_q_batch = smm.cu_seqlens_q_buffer[: num_rows + 1]

            flash_attn_varlen_func(
                q=q_flat,
                k=cluster_storage[0, :, :, None, :],
                v=cluster_storage[1, :, :, None, :],
                out=out_flat,
                cu_seqlens_q=cu_seqlens_q_batch,
                max_seqlen_q=1,
                seqused_k=seqused_k_flat,
                max_seqlen_k=max_blocks * cluster_block_size,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=sliding_window_size,
                block_table=block_table_flat,
                softcap=self.logits_soft_cap,
                scheduler_metadata=None,
                fa_version=self.vllm_flash_attn_version,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                num_splits=attn_metadata.max_num_splits,
                s_aux=self.sinks,
            )
            return output

        grouped_top_clusters = smm.active_grouped_top_clusters
        if grouped_top_clusters is None:
            grouped_top_clusters = self._select_grouped_top_clusters(
                query=query_actual,
                attn_metadata=attn_metadata,
                smm=smm,
                hkv=hkv,
            )
        required_free_blocks = (
            num_queries * hkv * (grouped_top_clusters.shape[2] + steady_blocks)
        )
        assert free_block_ids.numel() >= required_free_blocks, (
            "Sparse reusable scratch pool is undersized: "
            f"have {free_block_ids.numel()}, need at least "
            f"{required_free_blocks}"
        )
        block_table, _, seqused_k = smm.block_table_buffers.build(
            top_clusters=grouped_top_clusters,
            cluster_compact_block_ids=smm.cluster_compact_block_ids,
            cluster_temp_kv_pos=smm.cluster_temp_kv_pos,
            cluster_total_kv_counts=smm.cluster_total_kv_counts,
            temp_block_ids=smm.temp_block_ids,
            block_storage=cluster_storage,
            free_block_ids=free_block_ids,
            steady_start_block_ids=smm.steady_zone_head,
            steady_end_block_ids=smm.steady_zone_tail,
            steady_state=smm.steady_state,
            max_bt_len=max_blocks,
        )

        num_rows = num_queries * hkv
        device = query.device
        q_flat = query_actual.reshape(num_queries, hkv, q_per_kv, -1).reshape(
            num_rows, q_per_kv, -1
        )
        out_flat = output_actual.reshape(num_queries, hkv, q_per_kv, -1).reshape(
            num_rows, q_per_kv, -1
        )
        seqused_k_flat = seqused_k.reshape(num_rows)
        block_table_flat = block_table.reshape(num_rows, -1)
        if (
            smm.cu_seqlens_q_buffer is None
            or smm.cu_seqlens_q_buffer.shape[0] < num_rows + 1
            or smm.cu_seqlens_q_buffer.dtype != cu_seqlens_q.dtype
            or smm.cu_seqlens_q_step != 1
        ):
            smm.cu_seqlens_q_buffer = torch.arange(
                num_rows + 1,
                device=device,
                dtype=cu_seqlens_q.dtype,
            )
            smm.cu_seqlens_q_step = 1
        cu_seqlens_q_batch = smm.cu_seqlens_q_buffer[: num_rows + 1]

        flash_attn_varlen_func(
            q=q_flat,
            k=cluster_storage[0, :, :, None, :],
            v=cluster_storage[1, :, :, None, :],
            out=out_flat,
            cu_seqlens_q=cu_seqlens_q_batch,
            max_seqlen_q=1,
            seqused_k=seqused_k_flat,
            max_seqlen_k=max_blocks * cluster_block_size,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            block_table=block_table_flat,
            softcap=self.logits_soft_cap,
            scheduler_metadata=None,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=attn_metadata.max_num_splits,
            s_aux=self.sinks,
        )
        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        attn_metadata = self.get_attn_metadata_for_update()
        self.clear_attn_metadata_for_update()
        use_sparse_cluster = False
        cluster_allocated_block_info = (
            None if attn_metadata is None else attn_metadata.cluster_allocated_block_info
        )
        if (
            cluster_allocated_block_info is not None
            and len(cluster_allocated_block_info) != 0
        ):
            assert len(cluster_allocated_block_info) == 1
            req_index = 0
            assert len(cluster_allocated_block_info[req_index]) == 1
            assert attn_metadata.extra_sparse_manager_info.num_segment == 1
            use_sparse_cluster = (
                cluster_allocated_block_info[req_index][0] is not None
            )

        if use_sparse_cluster:
            free_blocks_info = cluster_allocated_block_info[req_index][0]
            smm = attn_metadata.sparse_manager_metadata[req_index]
            num_actual_tokens = attn_metadata.num_actual_tokens
            key_actual = key[:num_actual_tokens]
            value_actual = value[:num_actual_tokens]
            is_prefill = smm.mean is None
            hkv = key_actual.shape[1]
            num_clusters = attn_metadata.extra_sparse_manager_info.num_cluster
            dim = key_actual.shape[-1]
            cluster_storage = kv_cache.reshape(
                kv_cache.shape[0], kv_cache.shape[1], -1, kv_cache.shape[4]
            )
            cluster_block_size = cluster_storage.shape[-2]
            if is_prefill:
                device = key.device
                smm.cluster_centers_T = torch.zeros(
                    size=(hkv, dim, num_clusters),
                    dtype=key.dtype,
                    device=device,
                )
                smm.mean = torch.zeros(
                    size=(hkv, dim), dtype=key.dtype, device=device
                )
                smm.cluster_center_count = torch.zeros(
                    (1,), dtype=torch.int32, device=device
                )
                smm.cluster_compact_block_ids = torch.full(
                    fill_value=-1,
                    size=(
                        hkv,
                        num_clusters,
                        SparseManagerMetadata.INIT_CLUSTER_BLOCK_COUNT,
                    ),
                    dtype=torch.int32,
                    device=device,
                )
                smm.cluster_temp_kv_pos = torch.zeros(
                    size=(hkv, num_clusters, cluster_block_size, 2),
                    dtype=torch.int32,
                    device=device,
                )
                smm.cluster_total_kv_counts = torch.zeros(
                    size=(hkv, num_clusters), dtype=torch.int32, device=device
                )
                smm.clustered_token_count = 0
            if smm.steady_state is None:
                smm.steady_state = torch.zeros(
                    (4,), dtype=torch.int32, device=key.device
                )
            smm.active_top_clusters = None
            smm.active_grouped_top_clusters = None
            smm.pending_compact_load_specs.clear()
            smm.pending_compact_store_specs.clear()
            smm.steady_start_capacity = int(
                attn_metadata.extra_sparse_manager_info.steady_start_capacity
            )
            smm.steady_end_capacity = int(
                attn_metadata.extra_sparse_manager_info.steady_end_capacity
            )
            assert smm.steady_zone_head is not None
            assert smm.steady_zone_tail is not None
            assert smm.steady_state is not None
            if smm.append_buffers is None:
                smm.append_buffers = SparseAppendBuffers()
            assert smm.append_buffers is not None
            assert smm.cluster_total_kv_counts is not None
            compact_transfer_enabled = self._supports_sparse_compact_kv_transfer()
            prev_full_block_counts = torch.div(
                smm.cluster_total_kv_counts,
                cluster_block_size,
                rounding_mode="floor",
            )
            if compact_transfer_enabled:
                prev_full_block_counts = prev_full_block_counts.clone()
            self._ensure_cluster_compact_capacity(
                smm,
                hkv,
                num_clusters,
                self._max_cluster_full_blocks_upper_bound(
                    smm,
                    cluster_block_size,
                    num_actual_tokens,
                ),
                key.device,
            )
            if is_prefill:
                if smm.steady_buffers is None:
                    from vllm.v1.core.sparse_kv_utils import SparseSteadyBuffers

                    smm.steady_buffers = SparseSteadyBuffers()
                assert smm.steady_buffers is not None
                evicted_key, evicted_value, _ = smm.steady_buffers.update(
                    block_storage=cluster_storage,
                    steady_start_block_ids=smm.steady_zone_head,
                    steady_end_block_ids=smm.steady_zone_tail,
                    steady_state=smm.steady_state,
                    steady_start_capacity=smm.steady_start_capacity,
                    steady_end_capacity=smm.steady_end_capacity,
                    key=key_actual,
                    value=value_actual,
                    evicted_capacity=max(num_actual_tokens, 1),
                )
                evicted_count_i = smm.advance_steady_python_state(num_actual_tokens)
                smm.clustered_token_count += evicted_count_i
                if evicted_count_i == 0:
                    assert free_blocks_info.used_count_gpu is not None
                    free_blocks_info.used_count_gpu.zero_()
                    return
                cluster_key = evicted_key[:evicted_count_i]
                cluster_value = evicted_value[:evicted_count_i]
                res = prefill_cluster_meta_from_features_torch_batched(
                    cluster_key.transpose(0, 1),
                    num_clusters,
                    attn_metadata.extra_sparse_manager_info.num_segment,
                    granularity="token",
                )
                labels = (
                    res["token_to_cluster"]
                    .transpose_(-1, -2)
                    .to(dtype=torch.int32)
                    .contiguous()
                )

                valid_len = res["cluster_centres"].shape[1]
                smm.cluster_centers_T[:, :, :valid_len] = res[
                    "cluster_centres"
                ].transpose(1, 2)
                smm.mean = res["mean_key"].to(dtype=key.dtype)
                smm.cluster_center_count.fill_(valid_len)
                smm.append_buffers.append_with_labels(
                    block_storage=cluster_storage,
                    cluster_compact_block_ids=smm.cluster_compact_block_ids,
                    cluster_temp_kv_pos=smm.cluster_temp_kv_pos,
                    cluster_total_kv_counts=smm.cluster_total_kv_counts,
                    temp_block_ids=smm.temp_block_ids,
                    temp_block_kv_counts=smm.temp_block_kv_counts,
                    temp_block_kv_owner=smm.temp_block_kv_owner,
                    free_block_ids=free_blocks_info.allocated_block_ids_gpu,
                    used_free_block_count=free_blocks_info.used_count_gpu,
                    key=cluster_key,
                    value=cluster_value,
                    label=labels,
                )
            else:
                smm.append_buffers.append_with_steady(
                    block_storage=cluster_storage,
                    cluster_compact_block_ids=smm.cluster_compact_block_ids,
                    cluster_temp_kv_pos=smm.cluster_temp_kv_pos,
                    cluster_total_kv_counts=smm.cluster_total_kv_counts,
                    temp_block_ids=smm.temp_block_ids,
                    temp_block_kv_counts=smm.temp_block_kv_counts,
                    temp_block_kv_owner=smm.temp_block_kv_owner,
                    free_block_ids=free_blocks_info.allocated_block_ids_gpu,
                    used_free_block_count=free_blocks_info.used_count_gpu,
                    steady_start_block_ids=smm.steady_zone_head,
                    steady_end_block_ids=smm.steady_zone_tail,
                    steady_state=smm.steady_state,
                    steady_start_capacity=smm.steady_start_capacity,
                    steady_end_capacity=smm.steady_end_capacity,
                    key=key_actual,
                    value=value_actual,
                    cluster_centers_T=smm.cluster_centers_T,
                    mean=smm.mean,
                    cluster_center_count=smm.cluster_center_count,
                )
                smm.clustered_token_count += smm.advance_steady_python_state(
                    num_actual_tokens
                )
            if compact_transfer_enabled:
                assert smm.cluster_compact_block_ids is not None
                new_full_block_counts = torch.div(
                    smm.cluster_total_kv_counts,
                    cluster_block_size,
                    rounding_mode="floor",
                )
                num_clusters_i = int(smm.cluster_total_kv_counts.shape[1])
                self._ensure_cluster_compact_capacity(
                    smm,
                    hkv,
                    num_clusters_i,
                    self._max_cluster_full_blocks_upper_bound(
                        smm,
                        cluster_block_size,
                    ),
                    key.device,
                )
                self._append_pending_compact_store_specs(
                    smm=smm,
                    prev_full_block_counts=prev_full_block_counts,
                    new_full_block_counts=new_full_block_counts,
                )
            return

        return super().do_kv_cache_update(
            layer,
            key,
            value,
            kv_cache,
            slot_mapping,
        )
