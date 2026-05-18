# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse FlashAttention backend and implementation."""

from dataclasses import dataclass, field, fields
from typing import ClassVar

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
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
    SparseManagerMetadata,
)

logger = init_logger(__name__)


@dataclass
class SparseFlashAttentionMetadata(FlashAttentionMetadata):
    cluster_allocated_block_info: (
        list[tuple[SparseClusterBlockInfo | None, ...]] | None
    ) = None
    sparse_manager_metadata: list[SparseManagerMetadata] | None = None
    extra_sparse_manager_info: dict[str, object] = field(default_factory=dict)


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
        return AttentionCGSupport.NEVER

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
        use_sparse_cluster = (
            sparse_manager_metadata is not None
            and len(sparse_manager_metadata) > 0
            and sparse_manager_metadata[0].mean is not None
        )
        if use_sparse_cluster:
            cluster_storage = kv_cache.reshape(
                kv_cache.shape[0], kv_cache.shape[1], -1, kv_cache.shape[4]
            )
            req_index = 0
            smm = sparse_manager_metadata[0]
            cluster_allocated_block_info = attn_metadata.cluster_allocated_block_info[
                req_index
            ][0]

            is_prefill = smm.in_cluster_token_count == query.shape[0]
            if is_prefill:
                flash_attn_varlen_func(
                    q=query[:num_actual_tokens],
                    k=key,
                    v=value,
                    out=output[:num_actual_tokens],
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

            hkv = smm.mean.shape[0]
            cluster_block_size = cluster_storage.shape[2]
            num_queries = num_actual_tokens
            hq = query.shape[1]
            head_group_size = hq // hkv
            max_blocks = block_table.shape[-1]
            nprobe = attn_metadata.extra_sparse_manager_info["nprobe"]
            dim = query.shape[-1]

            if smm.in_cluster_token_count < smm.cluster_centers_T.shape[2]:
                cluster_centers_t = smm.cluster_centers_T[
                    :, :, :smm.in_cluster_token_count
                ]
            else:
                cluster_centers_t = smm.cluster_centers_T
            nprobe = min(nprobe, cluster_centers_t.shape[2])
            query_centered = (
                query.reshape(num_queries, hkv, head_group_size, dim)
                - smm.mean[None, :, None]
            )
            score = torch.matmul(query_centered, cluster_centers_t[None])
            score = score.reshape(num_queries, hq, -1)
            top_clusters = torch.topk(score, k=nprobe, dim=-1).indices.to(
                dtype=torch.int32
            )
            free_block_ids = cluster_allocated_block_info.reusable_block_ids_gpu
            assert free_block_ids is not None
            required_free_blocks = num_queries * hq * nprobe
            assert free_block_ids.numel() >= required_free_blocks, (
                "Sparse reusable scratch pool is undersized: "
                f"have {free_block_ids.numel()}, need at least "
                f"{required_free_blocks}"
            )
            if smm.block_table_buffers is None:
                smm.block_table_buffers = SparseBlockTableBuffers()
            assert smm.block_table_buffers is not None
            block_table, _, seqused_k = smm.block_table_buffers.build(
                top_clusters=top_clusters,
                cluster_compact_block_ids=smm.cluster_compact_block_ids,
                cluster_temp_kv_pos=smm.cluster_temp_kv_pos,
                cluster_total_kv_counts=smm.cluster_total_kv_counts,
                temp_block_ids=smm.temp_block_ids,
                block_storage=cluster_storage,
                free_block_ids=free_block_ids,
                max_bt_len=max_blocks,
            )

            num_rows = num_queries * hq
            device = query.device
            q_flat = query.reshape(num_rows, 1, -1)
            out_flat = output.reshape(num_rows, 1, -1)
            seqused_k_flat = seqused_k.reshape(num_rows)
            block_table_flat = block_table.reshape(num_rows, -1)
            if (
                smm.cu_seqlens_q_buffer is None
                or smm.cu_seqlens_q_buffer.shape[0] < num_rows + 1
                or smm.cu_seqlens_q_buffer.dtype != cu_seqlens_q.dtype
            ):
                smm.cu_seqlens_q_buffer = torch.arange(
                    num_rows + 1,
                    device=device,
                    dtype=cu_seqlens_q.dtype,
                )
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
        else:
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
        if attn_metadata is not None and len(attn_metadata.cluster_allocated_block_info) != 0:
            assert len(attn_metadata.cluster_allocated_block_info) == 1
            req_index = 0
            assert len(attn_metadata.cluster_allocated_block_info[req_index]) == 1
            assert attn_metadata.extra_sparse_manager_info["num_segment"] == 1
            use_sparse_cluster = (
                attn_metadata.cluster_allocated_block_info[req_index][0] is not None
            )

        if use_sparse_cluster:
            free_blocks_info = attn_metadata.cluster_allocated_block_info[req_index][0]
            smm = attn_metadata.sparse_manager_metadata[req_index]
            is_prefill = smm.mean is None
            hkv = key.shape[1]
            num_clusters = attn_metadata.extra_sparse_manager_info["num_cluster"]
            dim = key.shape[-1]
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
                smm.in_cluster_token_count = 0

                res = prefill_cluster_meta_from_features_torch_batched(
                    key.transpose(0, 1),
                    num_clusters,
                    attn_metadata.extra_sparse_manager_info["num_segment"],
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
            else:
                assert key.shape[0] == 1
                if smm.in_cluster_token_count < smm.cluster_centers_T.shape[-1]:
                    labels = torch.full(
                        fill_value=smm.in_cluster_token_count,
                        size=key.shape[:2],
                        dtype=torch.int32,
                        device=key.device,
                    )
                    smm.cluster_centers_T[:, :, smm.in_cluster_token_count] = key[0]
                else:
                    score = torch.matmul(
                        (key - smm.mean[None])[:, :, None],
                        smm.cluster_centers_T[None],
                    )
                    labels = score.argmax(dim=-1).to(dtype=torch.int32)[:, :, 0]

            if smm.append_buffers is None:
                smm.append_buffers = SparseAppendBuffers()
            assert smm.append_buffers is not None
            used_free_blocks = smm.append_buffers.append(
                block_storage=cluster_storage,
                cluster_compact_block_ids=smm.cluster_compact_block_ids,
                cluster_temp_kv_pos=smm.cluster_temp_kv_pos,
                cluster_total_kv_counts=smm.cluster_total_kv_counts,
                temp_block_ids=smm.temp_block_ids,
                temp_block_kv_counts=smm.temp_block_kv_counts,
                temp_block_kv_owner=smm.temp_block_kv_owner,
                free_block_ids=free_blocks_info.allocated_block_ids_gpu,
                used_free_block_count=free_blocks_info.used_count_gpu,
                key=key.contiguous(),
                value=value.contiguous(),
                label=labels,
            )
            smm.in_cluster_token_count += key.shape[0]
            return

        return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
