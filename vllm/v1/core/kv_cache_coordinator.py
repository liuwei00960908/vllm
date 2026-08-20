# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import NamedTuple

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool, DSASharedLogicalBlockPool
from vllm.v1.core.dsa_shared_pool import (
    DSASharedBlockLayout,
    DSASharedBlockOwner,
    DSASharedBundleAllocator,
)
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager,
    SingleTypeKVCacheManager,
    get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    SlidingWindowSpec,
    dsa_shared_pool_enabled,
    dsa_two_groups_enabled,
)
from vllm.v1.request import Request

logger = init_logger(__name__)

# Throttle for the [KVSTARVE] admission-failure diagnostic (seconds).
_last_starve_log = [0.0]


def _get_dsa_pool_log_interval() -> float:
    """Interval (seconds) for the DSA shared pool usage log; 0 disables.

    Fork semantics: vllm-dsa-two-groups@4575d8a12 kv_cache_coordinator.py
    reads VLLM_ASCEND_DSA_POOL_LOG_INTERVAL directly (kept as a raw env read
    here; registering it in vllm's envs.py is unnecessary for one knob).
    """
    raw = os.getenv("VLLM_ASCEND_DSA_POOL_LOG_INTERVAL", "5")
    try:
        return max(float(raw), 0.0)
    except ValueError:
        logger.warning(
            "Invalid VLLM_ASCEND_DSA_POOL_LOG_INTERVAL=%r; using 5 seconds.", raw
        )
        return 5.0


def _validate_prefix_cache_retention_interval(
    retention_interval: int | None,
    scheduler_block_size: int,
    kv_cache_config: KVCacheConfig,
) -> None:
    if retention_interval is None:
        return

    # Retention only sparsifies sliding-window checkpoints for now; every other
    # manager (full attention, Mamba, chunked-local) caches densely and
    # ignores it to be conservative.
    # TODO: Support Mamba/linear attention.
    if not any(
        isinstance(g.kv_cache_spec, SlidingWindowSpec)
        for g in kv_cache_config.kv_cache_groups
    ):
        raise ValueError(
            "VLLM_PREFIX_CACHE_RETENTION_INTERVAL is set but this model has "
            "no sliding-window KV cache group, so retention has no effect. "
            "Unset it (the feature only applies to sliding-window attention)."
        )

    if retention_interval < 0 or retention_interval % scheduler_block_size != 0:
        raise ValueError(
            f"VLLM_PREFIX_CACHE_RETENTION_INTERVAL ({retention_interval}) "
            "must be non-negative and a multiple of scheduler_block_size "
            f"({scheduler_block_size})."
        )


class KVCacheCoordinator(ABC):
    """
    Coordinate the KV cache of different KV cache groups.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_num_batched_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        # The scheduling granularity (LCM of all group block sizes), must be a multiple
        # of the hash_block_size and the block size of each group.
        assert scheduler_block_size % hash_block_size == 0 and all(
            scheduler_block_size % g.kv_cache_spec.block_size == 0
            for g in kv_cache_config.kv_cache_groups
        )
        self.scheduler_block_size = scheduler_block_size

        # DSA two-groups per-group mode: every KV cache group is backed by its
        # OWN BlockPool (sized from num_blocks_per_group) instead of sharing
        # one pool, so latent/indexer block lifetimes are independent.
        # Fork semantics: vllm-dsa-two-groups@4575d8a12 kv_cache_coordinator.py
        # :87-108, :145-168, :177 (shared-bundle variant deferred to Step 5).
        group_page_sizes = {
            g.kv_cache_spec.page_size_bytes for g in kv_cache_config.kv_cache_groups
        }
        # DSA shared-bundle variant: two logical pools over one physical
        # bundle pool (Step 5). Fork semantics: vllm-dsa-two-groups@4575d8a12
        # kv_cache_coordinator.py:90-101.
        self.use_dsa_shared_block_pool = (
            dsa_shared_pool_enabled()
            and dsa_two_groups_enabled()
            and len(kv_cache_config.kv_cache_groups) == 2
            and len(group_page_sizes) > 1
        )
        self.use_per_group_block_pools = (
            dsa_two_groups_enabled()
            and not self.use_dsa_shared_block_pool
            and len(kv_cache_config.kv_cache_groups) > 1
            and len(group_page_sizes) > 1
        )
        assert not (
            (self.use_per_group_block_pools or self.use_dsa_shared_block_pool)
            and enable_caching
        ), (
            "DSA two-group mode does not support prefix caching yet; "
            "launch with --no-enable-prefix-caching."
        )
        self._dsa_pool_last_log = 0.0
        self._dsa_pool_log_interval = _get_dsa_pool_log_interval()
        self._dsa_last_starve: tuple[str, int, int, list[int]] | None = None
        if self.use_dsa_shared_block_pool:
            latent_group = max(
                kv_cache_config.kv_cache_groups,
                key=lambda g: g.kv_cache_spec.page_size_bytes,
            )
            indexer_group = min(
                kv_cache_config.kv_cache_groups,
                key=lambda g: g.kv_cache_spec.page_size_bytes,
            )
            layout = DSASharedBlockLayout(
                latent_page_size_bytes=latent_group.kv_cache_spec.page_size_bytes,
                indexer_page_size_bytes=indexer_group.kv_cache_spec.page_size_bytes,
                capacity_bundles=kv_cache_config.num_blocks,
            )
            self.dsa_shared_allocator = DSASharedBundleAllocator(layout)
            self.dsa_shared_num_layer_pairs = len(latent_group.layer_names)
            self.block_pools = []
            for group in kv_cache_config.kv_cache_groups:
                owner = (
                    DSASharedBlockOwner.LATENT
                    if group.kv_cache_spec.page_size_bytes
                    == latent_group.kv_cache_spec.page_size_bytes
                    else DSASharedBlockOwner.INDEXER
                )
                self.block_pools.append(
                    DSASharedLogicalBlockPool(self.dsa_shared_allocator, owner)
                )
            logger.info(
                "DSA shared KV bundle pool: %d bundles, latent=%d blocks/bundle, "
                "indexer=%d blocks/bundle.",
                kv_cache_config.num_blocks,
                layout.latent_blocks_per_bundle,
                layout.indexer_blocks_per_bundle,
            )
        elif self.use_per_group_block_pools:
            num_pools = len(kv_cache_config.kv_cache_groups)
            pool_sizes = (
                kv_cache_config.num_blocks_per_group
                if kv_cache_config.num_blocks_per_group is not None
                else [kv_cache_config.num_blocks] * num_pools
            )
            logger.info("Per-group KV block pools: %s blocks.", pool_sizes)
        else:
            num_pools = 1
            pool_sizes = [kv_cache_config.num_blocks]
        if not self.use_dsa_shared_block_pool:
            # Real BlockPools (single shared pool or per-group pools); the
            # DSA shared variant builds its logical pools above instead.
            self.block_pools = [
                BlockPool(
                    num_gpu_blocks=pool_sizes[i],
                    enable_caching=enable_caching,
                    hash_block_size=hash_block_size,
                    enable_kv_cache_events=enable_kv_cache_events,
                    metrics_collector=metrics_collector,
                )
                for i in range(num_pools)
            ]
        # Kept for compatibility with single-pool consumers (e.g. metrics).
        self.block_pool = self.block_pools[0]

        # KV cache group indices that get the EAGLE last-block drop.
        self.eagle_group_ids: set[int] = {
            i for i, g in enumerate(kv_cache_config.kv_cache_groups) if g.is_eagle_group
        }
        # Conservatively fall back to flag all groups when no group is flagged.
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))

        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
                # DSA per-group and shared modes: group i allocates from its
                # own (logical) pool at index i; single-pool mode: everyone
                # shares pool 0. Mirrors dsa_block_pool_index() from
                # dsa_shared_pool.py.
                block_pool=self.block_pools[
                    i
                    if (
                        self.use_per_group_block_pools
                        or self.use_dsa_shared_block_pool
                    )
                    else 0
                ],
                enable_caching=enable_caching,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
                scheduler_block_size=self.scheduler_block_size,
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )

        # A positive retention interval must be a multiple of the base hit granularity
        # (``scheduler_block_size``) to land on real cache-hit boundaries.
        # 0 = keep only the latest replay boundary; None = dense;
        self.retention_interval = envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL
        _validate_prefix_cache_retention_interval(
            self.retention_interval, self.scheduler_block_size, kv_cache_config
        )

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.
            num_encoder_tokens: The number of encoder tokens for allocating
                blocks for cross-attention.
            total_computed_tokens: Include both local and external tokens.
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
            apply_admission_cap: If True, apply the recycling-aware
                per-request admission cap (SWA / chunked-local). Set only by
                the full-sequence admission gate; per-step allocation must
                leave it False so the predictor matches `allocate_new_blocks`.

        Returns:
            The number of blocks to allocate.
        """
        num_blocks_to_allocate = 0
        for i, manager in enumerate(self.single_type_managers):
            if isinstance(manager, CrossAttentionManager):
                # For cross-attention, we issue a single static allocation
                # of blocks based on the number of encoder input tokens.
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id,
                    num_encoder_tokens,
                    [],
                    0,
                    num_encoder_tokens,
                    apply_admission_cap=apply_admission_cap,
                )
            else:
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id,
                    num_tokens,
                    new_computed_blocks[i],
                    total_computed_tokens,
                    num_tokens_main_model,
                    apply_admission_cap=apply_admission_cap,
                )
        return num_blocks_to_allocate

    def get_num_blocks_to_allocate_per_group(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> list[int]:
        """Per-group variant of get_num_blocks_to_allocate (one entry per KV
        cache group / single-type manager). Fork semantics:
        vllm-dsa-two-groups@4575d8a12 kv_cache_coordinator.py:253-293."""
        num_blocks_to_allocate: list[int] = []
        for i, manager in enumerate(self.single_type_managers):
            if isinstance(manager, CrossAttentionManager):
                num_blocks_to_allocate.append(
                    manager.get_num_blocks_to_allocate(
                        request_id,
                        num_encoder_tokens,
                        [],
                        0,
                        num_encoder_tokens,
                        apply_admission_cap=apply_admission_cap,
                    )
                )
            else:
                num_blocks_to_allocate.append(
                    manager.get_num_blocks_to_allocate(
                        request_id,
                        num_tokens,
                        new_computed_blocks[i],
                        total_computed_tokens,
                        num_tokens_main_model,
                        apply_admission_cap=apply_admission_cap,
                    )
                )
        return num_blocks_to_allocate

    def has_enough_free_blocks(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
        reserved_blocks: int = 0,
    ) -> bool:
        """Whether the allocation can be satisfied.

        With per-group block pools, EVERY group's demand must fit its own
        pool; with a single shared pool, the summed demand must fit that
        pool (identical to the official admission comparison). Fork
        semantics: vllm-dsa-two-groups@4575d8a12 kv_cache_coordinator.py
        :295-339 ([KVSTARVE] diagnostic kept).
        """
        if self.use_per_group_block_pools:
            assert reserved_blocks == 0, (
                "DSA per-group mode with async-connector block reservation is "
                "not supported in this replay slice (reserved_blocks must be 0)."
            )
            per_group = self.get_num_blocks_to_allocate_per_group(
                request_id,
                num_tokens,
                new_computed_blocks,
                num_encoder_tokens,
                total_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
            )
            frees = [
                m.block_pool.get_num_free_blocks() for m in self.single_type_managers
            ]
            ok = all(needed <= free for needed, free in zip(per_group, frees))
            if not ok and (time.monotonic() - _last_starve_log[0]) > 1.0:
                # [KVSTARVE] one-per-second snapshot of WHICH group blocked an
                # admission, with each group's demand vs free blocks.
                _last_starve_log[0] = time.monotonic()
                parts = []
                for i, (needed, free, m) in enumerate(
                    zip(per_group, frees, self.single_type_managers)
                ):
                    spec = getattr(m, "kv_cache_spec", None)
                    page = getattr(spec, "page_size_bytes", "?")
                    flag = " <<BLOCKED" if needed > free else ""
                    parts.append(
                        f"g{i}({type(m).__name__},page={page}): "
                        f"need={needed} free={free}{flag}"
                    )
                logger.info(
                    "[KVSTARVE] req=%s not admitted | %s", request_id, "  ".join(parts)
                )
            return ok
        if self.use_dsa_shared_block_pool:
            # Shared bundle: demands from both logical pools are converted to
            # bundles and compared against the single physical free count.
            # Fork semantics: vllm-dsa-two-groups@4575d8a12
            # kv_cache_coordinator.py:340-363.
            per_group = self.get_num_blocks_to_allocate_per_group(
                request_id,
                num_tokens,
                new_computed_blocks,
                num_encoder_tokens,
                total_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
            )
            needed_bundles = sum(
                manager.block_pool.get_num_bundles_to_allocate(needed)
                for manager, needed in zip(self.single_type_managers, per_group)
            )
            ok = needed_bundles <= self.dsa_shared_allocator.free_bundle_count
            if not ok:
                self._dsa_last_starve = (
                    request_id,
                    needed_bundles,
                    self.dsa_shared_allocator.free_bundle_count,
                    per_group,
                )
                if (time.monotonic() - _last_starve_log[0]) > 1.0:
                    _last_starve_log[0] = time.monotonic()
                    logger.info(
                        "[KVSTARVE] req=%s not admitted | need_bundles=%d free=%d "
                        "per_group_blocks=%s",
                        request_id,
                        needed_bundles,
                        self.dsa_shared_allocator.free_bundle_count,
                        per_group,
                    )
            return ok
        # Single-pool fallback: mathematically identical to the official
        # `summed demand <= free - reserved` admission comparison.
        num_blocks_to_allocate = self.get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            num_encoder_tokens,
            total_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=apply_admission_cap,
        )
        available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
        return num_blocks_to_allocate <= available_blocks

    def maybe_log_dsa_shared_pool_usage(
        self,
        requests: dict[str, Request],
        running_count: int,
        waiting_count: int,
        max_running_count: int,
    ) -> None:
        """Periodic usage log for the DSA shared bundle pool.

        Transcribed (reduced) from fork kv_cache_coordinator.py:366-505: the
        prefill/post-prefill per-request block walk is omitted (depends on
        manager.req_to_blocks internals); owner bundle split, fragmentation,
        and the last starve snapshot are kept. Returns immediately when the
        shared pool is not active, so the scheduler-side call is free for
        every other mode.
        """
        if not self.use_dsa_shared_block_pool:
            return
        if self._dsa_pool_log_interval <= 0:
            return
        now = time.monotonic()
        if now - self._dsa_pool_last_log < self._dsa_pool_log_interval:
            return
        self._dsa_pool_last_log = now

        allocator = self.dsa_shared_allocator
        layout = allocator.layout
        capacity = layout.capacity_bundles
        free = allocator.free_bundle_count
        used = capacity - free
        owner_counts = allocator.owner_bundle_counts()
        latent_owner_bundles = owner_counts.get(DSASharedBlockOwner.LATENT, 0)
        indexer_owner_bundles = owner_counts.get(DSASharedBlockOwner.INDEXER, 0)
        bundle_bytes = (
            layout.bundle_page_size_bytes * self.dsa_shared_num_layer_pairs
        )
        used_gib = used * bundle_bytes / 2**30
        total_gib = capacity * bundle_bytes / 2**30

        starve_req = "-"
        starve_need = 0
        starve_free = free
        starve_per_group: list[int] = []
        if self._dsa_last_starve is not None:
            starve_req, starve_need, starve_free, starve_per_group = (
                self._dsa_last_starve
            )
        logger.info(
            "DSA shared pool usage: running=%d waiting=%d used_bundles=%d "
            "(latent=%d indexer=%d) free_bundles=%d free_ranges=%d "
            "largest_free_range=%d used=%.2f/%.2f GiB | last_starve=%s "
            "need=%d free=%d per_group=%s",
            running_count,
            waiting_count,
            used,
            latent_owner_bundles,
            indexer_owner_bundles,
            free,
            allocator.free_range_count,
            allocator.largest_free_range,
            used_gib,
            total_gib,
            starve_req,
            starve_need,
            starve_free,
            starve_per_group,
        )

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Add the new computed blocks to the request. Optionally allocate new
            blocks for external computed tokens (if any).

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
        """
        for i, manager in enumerate(self.single_type_managers):
            manager.allocate_new_computed_blocks(
                request_id,
                new_computed_blocks[i],
                num_local_computed_tokens,
                num_external_computed_tokens,
            )

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
        num_encoder_tokens: int = 0,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
            num_encoder_tokens: The number of encoder tokens for allocating
                blocks for cross-attention.

        Returns:
            The new allocated blocks.
        """
        return tuple(
            manager.allocate_new_blocks(
                request_id,
                num_encoder_tokens
                if isinstance(manager, CrossAttentionManager)
                else num_tokens,
                num_tokens_main_model,
            )
            for manager in self.single_type_managers
        )

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_computed_tokens: The total number of tokens
                that need to be cached
                (including tokens that are already cached).
        """
        for manager in self.single_type_managers:
            manager.cache_blocks(
                request,
                num_computed_tokens,
                retention_interval=self.retention_interval,
            )

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        for manager in self.single_type_managers:
            manager.free(request_id)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """
        Get the number of common prefix blocks for all requests with allocated
        KV cache for each kv cache group.

        Args:
            running_request_id: The request ID of any running request, used to
                identify the common prefix blocks.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache group.
        """
        return [
            manager.get_num_common_prefix_blocks(running_request_id)
            for manager in self.single_type_managers
        ]

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """
        Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            total_computed_tokens: The total number of computed tokens, including
                local computed tokens and external computed tokens.
        """
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(request_id, total_computed_tokens)

    def remove_saved_decode_window_blocks(
        self, request_id: str, committed_end: int
    ) -> int:
        """DSA shrink replay (B1c): forward committed receipts to the
        DSALatentManager (the only manager that implements the release);
        other managers contribute 0. Provenance: fork
        kv_cache_coordinator.py:646-657."""
        freed = 0
        for manager in self.single_type_managers:
            release = getattr(
                manager, "remove_saved_decode_window_blocks", None
            )
            if callable(release):
                freed += release(request_id, committed_end)
        return freed

    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        """
        Get the blocks for the request.
        """
        return tuple(
            manager.req_to_blocks.get(request_id) or []
            for manager in self.single_type_managers
        )

    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        pass

    def new_step_starts(self) -> None:
        """Called when a new step is started."""
        for manager in self.single_type_managers:
            manager.new_step_starts()


class KVCacheCoordinatorNoPrefixCache(KVCacheCoordinator):
    """
    KV cache coordinator to use if prefix caching is disabled or unsupported.
    In contrast to UnitaryKVCacheCoordinator and HybridKVCacheCoordinator,
    supports arbitrary numbers of KV cache groups (including 0 groups).
    Does not implement any features related to prefix caching.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_num_batched_tokens: int,
        use_eagle: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            max_num_batched_tokens,
            use_eagle,
            False,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        self.num_single_type_manager = len(self.single_type_managers)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        return [0] * self.num_single_type_manager

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(self.num_single_type_manager)
        )
        return blocks, 0


class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for models with only one KV cache group. This is the
    case for models with only one KV cache type, e.g., all attention layers use
    full attention or all attention layers use sliding window attention.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_num_batched_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            max_num_batched_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.block_size = self.kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        if dcp_world_size > 1:
            self.block_size *= dcp_world_size
        if pcp_world_size > 1:
            self.block_size *= pcp_world_size
        # For models using only Mamba, block_size is set to max_model_len when
        # prefix caching is disabled, and hash_block_size validation is skipped.
        assert not enable_caching or (hash_block_size == self.block_size), (
            "UnitaryKVCacheCoordinator assumes hash_block_size == block_size"
        )
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "UnitaryKVCacheCoordinator assumes only one kv cache group"
        )
        # Single group; useless but just set ``use_eagle`` for consistency regardless.
        self.single_type_managers[0].use_eagle = 0 in self.eagle_group_ids

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        hit_blocks = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            drop_eagle_block=0 in self.eagle_group_ids,
            alignment_tokens=self.block_size,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
        )
        return hit_blocks, len(hit_blocks[0]) * self.block_size


class SpecGroup(NamedTuple):
    """KV cache groups that share one spec, batched together for a single
    cache-hit lookup.

    ``use_eagle`` is True iff any member group is an EAGLE/MTP group. Members
    sharing a spec are cached and looked up jointly, so the EAGLE last-block drop
    is necessarily decided for the whole spec group.
    """

    spec: KVCacheSpec
    group_ids: list[int]
    manager_cls: type[SingleTypeKVCacheManager]
    use_eagle: bool


class HybridKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for hybrid models with multiple KV cache types, and
    thus multiple kv cache groups.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_num_batched_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            max_num_batched_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        # hash_block_size: the block size used to compute block hashes.
        # The actual block size usually equals hash_block_size, but in cases where
        # different KV cache groups have different block sizes, the actual block size
        # can be a multiple of hash_block_size.
        self.hash_block_size = hash_block_size
        assert all(
            g.kv_cache_spec.block_size % hash_block_size == 0
            for g in kv_cache_config.kv_cache_groups
        ), "block_size must be divisible by hash_block_size"
        assert dcp_world_size == 1, "DCP not support hybrid attn now."
        assert pcp_world_size == 1, "PCP not support hybrid attn now."
        self.verify_and_split_kv_cache_groups()

    def verify_and_split_kv_cache_groups(self) -> None:
        """
        Groups KV cache groups by their spec type for efficient batch processing
        during cache hit lookup.
        """
        self.attention_groups: list[SpecGroup] = []
        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            manager_cls = self.single_type_managers[i].__class__
            spec = g.kv_cache_spec
            use_eagle = i in self.eagle_group_ids

            # Try to find an existing group with the same spec
            for idx, group in enumerate(self.attention_groups):
                if group.spec == spec:
                    assert manager_cls is group.manager_cls, (
                        "Expected same manager class for identical KV cache specs."
                    )
                    group.group_ids.append(i)
                    if use_eagle and not group.use_eagle:
                        self.attention_groups[idx] = group._replace(use_eagle=True)
                    break
            else:
                self.attention_groups.append(
                    SpecGroup(spec, [i], manager_cls, use_eagle)
                )

        assert len(self.attention_groups) > 1, (
            "HybridKVCacheCoordinator requires at least two attention groups."
        )

        # Put full attention first: its efficient left-to-right scan provides
        # a tighter initial bound, reducing work for subsequent groups.
        self.attention_groups.sort(
            key=lambda g: not isinstance(g.spec, FullAttentionSpec)
        )

        # Propagate the eagle bit to each manager (default to ``use_eagle=False``).
        for group in self.attention_groups:
            if group.use_eagle:
                for gid in group.group_ids:
                    self.single_type_managers[gid].use_eagle = True

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        # Cache hits in this coordinator are always a multiple of
        # ``scheduler_block_size`` tokens (see ``find_longest_cache_hit``).
        # Within an aligned region, SWA groups may only consult a subset of blocks
        # per ``scheduler_block_size``-segment so the unused blocks also stay
        # out of the prefix-cache hash map.
        aligned_num_computed_tokens = (
            num_computed_tokens // self.scheduler_block_size * self.scheduler_block_size
        )
        for manager in self.single_type_managers:
            num_tokens_to_cache = aligned_num_computed_tokens
            # EAGLE groups match one block past each aligned boundary and drop
            # it, so make that lookahead block eligible to be cached.
            if manager.use_eagle and aligned_num_computed_tokens > 0:
                num_tokens_to_cache = min(
                    num_computed_tokens,
                    aligned_num_computed_tokens + manager.block_size,
                )
            # The manager already knows the fine hit granularity
            # (``scheduler_block_size``); retention is passed separately so it
            # can keep both the coarse segment tails and the fine replay
            # boundary (which needs the fine value).
            manager.cache_blocks(
                request,
                num_tokens_to_cache,
                retention_interval=self.retention_interval,
            )

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        Find the longest cache hit using an iterative fixed-point algorithm.

        Each attention type either accepts the current candidate length or
        reduces it. If any type reduces the length, restart checks over all
        types. This converges because length monotonically decreases and is
        bounded below by 0.

        Args:
            block_hashes: The block hashes of the request.
            max_cache_hit_length: The maximum length of the cache hit.

        Returns:
            A tuple containing:
                - A tuple of the cache hit blocks for each single type manager.
                - The number of tokens of the longest cache hit.
        """

        def _get_block_hashes(kv_cache_spec: KVCacheSpec) -> BlockHashList:
            if kv_cache_spec.block_size == self.hash_block_size:
                return block_hashes
            return BlockHashListWithBlockSize(
                block_hashes, self.hash_block_size, kv_cache_spec.block_size
            )

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups

        # Simple hybrid (1 full attn + 1 other): one iteration suffices.
        # Full attn is always first if it exists.
        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0].spec, FullAttentionSpec
        )

        # Attention-group indices whose EAGLE drop is verified at the current
        # ``curr_hit_length``. Each eagle group applies the drop at most once
        # per candidate length (see issue #32802).
        eagle_verified: set[int] = set()

        while True:
            curr_hit_length = hit_length

            for idx, (spec, group_ids, manager_cls, use_eagle) in enumerate(
                self.attention_groups
            ):
                cached_blocks = hit_blocks_by_group[group_ids[0]]
                if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                    # Full attention is downward-closed: we only need to look
                    # up cached blocks once; on subsequent iterations just trim
                    # to the (reduced) current hit length.
                    curr_hit_length = (
                        curr_hit_length // spec.block_size * spec.block_size
                    )
                    continue

                drop_eagle_block = use_eagle and idx not in eagle_verified

                _max_length = curr_hit_length
                if drop_eagle_block:
                    # Eagle needs to match one more block and then pop the last.
                    _max_length = min(
                        curr_hit_length + spec.block_size, max_cache_hit_length
                    )
                hit_blocks = manager_cls.find_longest_cache_hit(
                    block_hashes=_get_block_hashes(spec),
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
                    drop_eagle_block=drop_eagle_block,
                    alignment_tokens=self.scheduler_block_size,
                )
                _new_hit_length = len(hit_blocks[0]) * spec.block_size
                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        # Truncate full attention blocks to final hit_length (if present)
        first_group = self.attention_groups[0]
        if isinstance(first_group.spec, FullAttentionSpec):
            num_blocks = hit_length // first_group.spec.block_size
            for group_id in first_group.group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]

        return tuple(
            blocks if blocks is not None else [] for blocks in hit_blocks_by_group
        ), hit_length


def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    max_num_batched_tokens: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    scheduler_block_size: int,
    hash_block_size: int,
    metrics_collector: KVCacheMetricsCollector | None = None,
) -> KVCacheCoordinator:
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(
            kv_cache_config,
            max_model_len,
            max_num_batched_tokens,
            use_eagle,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(
            kv_cache_config,
            max_model_len,
            max_num_batched_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
    return HybridKVCacheCoordinator(
        kv_cache_config,
        max_model_len,
        max_num_batched_tokens,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        metrics_collector=metrics_collector,
    )
