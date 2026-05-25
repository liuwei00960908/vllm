# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import (
    BlockStored,
    KVCacheEvent,
    KVConnectorKVEvents,
    KVEventAggregator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class LMCacheKVEvents(KVConnectorKVEvents):
    """
    Concrete implementation of KVConnectorKVEvents using KVEventAggregator.
    """

    def __init__(self, num_workers: int) -> None:
        self._aggregator = KVEventAggregator(num_workers)

    def add_events(self, events: list[KVCacheEvent]) -> None:
        self._aggregator.add_events(events)

    def aggregate(self) -> "LMCacheKVEvents":
        """
        Aggregate KV events and retain only common events.
        """
        common_events = self._aggregator.get_common_events()
        self._aggregator.clear_events()
        self._aggregator.add_events(common_events)
        self._aggregator.reset_workers()
        return self

    def increment_workers(self, count: int = 1) -> None:
        self._aggregator.increment_workers(count)

    def get_all_events(self) -> list[KVCacheEvent]:
        return self._aggregator.get_all_events()

    def get_number_of_workers(self) -> int:
        return self._aggregator.get_number_of_workers()

    def clear_events(self) -> None:
        self._aggregator.clear_events()
        self._aggregator.reset_workers()

    def __repr__(self) -> str:
        return f"<LMCacheKVEvents events={self.get_all_events()}>"


class LMCacheConnectorV1(KVConnectorBase_V1):
    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        """
        LMCache requires PIECEWISE CUDA graph mode when layerwise
        operations are enabled. The wait_for_layer_load and save_kv_layer
        methods perform actual async synchronization that cannot be
        captured in CUDA graphs.
        """
        return extra_config.get("use_layerwise", False)

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        assert vllm_config.kv_transfer_config is not None
        use_native = vllm_config.kv_transfer_config.get_from_extra_config(
            "use_native", False
        )
        if use_native:
            logger.info("Initializing native LMCache connector")
            # lazy import
            from vllm.distributed.kv_transfer.kv_connector.v1 import lmcache_integration

            _adapter = lmcache_integration.vllm_v1_adapter

            cls = _adapter.LMCacheConnectorV1Impl
        else:
            logger.info("Initializing latest dev LMCache connector")
            # lazy import
            from lmcache.integration.vllm.vllm_v1_adapter import (
                LMCacheConnectorV1Impl as LMCacheConnectorLatestImpl,
            )

            cls = LMCacheConnectorLatestImpl

        self._lmcache_engine = cls(vllm_config, role, self)

        self._kv_cache_events: LMCacheKVEvents | None = None

    # ==============================
    # Worker-side methods
    # ==============================
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args:
            kv_caches: dictionary of layer names, kv cache
        """
        if hasattr(self._lmcache_engine, "register_kv_caches"):
            self._lmcache_engine.register_kv_caches(kv_caches)
        else:
            logger.warning(
                "LMCache engine does not support register_kv_caches, "
                "please check and use the latest version"
            )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        self._lmcache_engine.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str, selected_tokens: list[torch.Tensor], cluster_meta: list, token_start_index: list[torch.Tensor], cluster_start_index: list[torch.Tensor], retrieve_budget: int) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        self._lmcache_engine.wait_for_layer_load(layer_name, selected_tokens, cluster_meta, token_start_index, cluster_start_index, retrieve_budget)

    def get_prefill_saved(self) -> set[str] | None:
        fn = getattr(self._lmcache_engine, "get_prefill_saved", None)
        return fn() if fn is not None else None

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """
        Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        self._lmcache_engine.save_kv_layer(
            layer_name, kv_layer, attn_metadata, **kwargs
        )

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        self._lmcache_engine.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        return self._lmcache_engine.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.
        """
        method = getattr(self._lmcache_engine, "get_block_ids_with_load_errors", None)
        if callable(method):
            return method()

        # Fallback for older versions that don't support this method
        return set()

    def get_kv_connector_kv_cache_events(self) -> LMCacheKVEvents | None:
        """
        Get the KV connector kv cache events collected during the last interval.
        """

        events = self._lmcache_engine.get_kv_events()  # type: ignore [attr-defined]
        if not events:
            return None

        blocks: list[BlockStored] = [
            BlockStored(
                block_hashes=e.block_hashes,
                parent_block_hash=e.parent_block_hash,
                token_ids=e.token_ids,
                lora_id=e.lora_id,
                block_size=e.block_size,
                medium=e.medium,
                lora_name=getattr(e, "lora_name", None),
            )
            for e in events
        ]

        lmcache_kv_events = LMCacheKVEvents(num_workers=1)
        lmcache_kv_events.add_events(blocks)
        return lmcache_kv_events

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        return self._lmcache_engine.get_num_new_matched_tokens(
            request, num_computed_tokens
        ), False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        """
        self._lmcache_engine.update_state_after_alloc(request, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = self._lmcache_engine.build_connector_meta(scheduler_output)
        #self._maybe_inject_offloaded_loads(meta, scheduler_output)
        return meta

    def _maybe_inject_offloaded_loads(
        self,
        meta: KVConnectorMetadata,
        scheduler_output: SchedulerOutput,
    ) -> None:
        cached = scheduler_output.scheduled_cached_reqs
        offloaded = cached.sparse_offloaded_req_ids
        scratch_map = cached.sparse_scratch_block_ids
        if not offloaded or not scratch_map:
            return
        requests = getattr(meta, "requests", None)
        if requests is None:
            return
        block_size = getattr(self._lmcache_engine, "_block_size", None)
        trackers = getattr(self._lmcache_engine, "_request_trackers", None)
        if block_size is None or trackers is None:
            return
        try:
            from lmcache.integration.vllm.vllm_v1_adapter import (
                LoadSpec, ReqMeta,
            )
        except ImportError:
            from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.vllm_v1_adapter import (  # noqa: E501
                LoadSpec, ReqMeta,
            )

        existing = {r.req_id: r for r in requests if hasattr(r, "req_id")}
        for req_id in offloaded:
            scratch_ids = scratch_map.get(req_id) or []
            if not scratch_ids:
                continue
            tracker = trackers.get(req_id)
            if tracker is None:
                continue
            token_ids = list(tracker.token_ids)
            tracker_prompt_len = int(getattr(tracker, "prompt_len", len(token_ids)))
            scratch_cap = len(scratch_ids) * block_size
            import logging
            _log = logging.getLogger(__name__)
            _log.warning(
                "[sparse-offload-inject] req=%s tracker.prompt_len=%d "
                "len(tracker.token_ids)=%d scratch_cap=%d",
                req_id, tracker_prompt_len, len(token_ids), scratch_cap,
            )
            # Cover all prompt positions if possible; never exceed what
            # tracker actually has (else token_ids/slot_mapping length mismatch).
            n_load = min(tracker_prompt_len, len(token_ids))
            if n_load <= 0:
                continue
            # LMCache requires len(slot_mapping) == len(tokens) == n_load.
            # With rank-indexed scatter, only the first `scratch_cap` entries
            # are written to; pad the rest with 0 (never accessed because
            # num_selected per layer <= scratch_cap).
            block_ids_t = torch.tensor(scratch_ids, dtype=torch.long)
            offsets = torch.arange(0, block_size, dtype=torch.long)
            real_slots = (
                offsets.reshape(1, block_size)
                + block_ids_t.reshape(-1, 1) * block_size
            ).flatten()
            slot_mapping = torch.zeros(n_load, dtype=torch.long)
            copy_n = min(n_load, scratch_cap)
            slot_mapping[:copy_n] = real_slots[:copy_n]
            load_spec = LoadSpec(
                vllm_cached_tokens=0,
                lmcache_cached_tokens=n_load,
                can_load=True,
            )
            r = existing.get(req_id)
            if r is not None:
                r.token_ids = token_ids[:n_load]
                r.slot_mapping = slot_mapping
                r.load_spec = load_spec
            else:
                requests.append(
                    ReqMeta(
                        req_id=req_id,
                        token_ids=token_ids[:n_load],
                        slot_mapping=slot_mapping,
                        load_spec=load_spec,
                    )
                )
            import logging
            _log = logging.getLogger(__name__)
            _log.warning(
                "[sparse-offload-load] req=%s n_load=%d block_size=%d "
                "n_scratch_blocks=%d scratch_ids[0]=%d scratch_ids[-1]=%d "
                "slot_mapping.shape=%s slot_mapping[:8]=%s "
                "slot_mapping[-8:]=%s slot_mapping.max=%d slot_mapping.min=%d "
                "load_spec=%s",
                req_id,
                n_load,
                block_size,
                len(scratch_ids),
                int(scratch_ids[0]),
                int(scratch_ids[-1]),
                tuple(slot_mapping.shape),
                slot_mapping[:8].tolist(),
                slot_mapping[-8:].tolist(),
                int(slot_mapping.max()),
                int(slot_mapping.min()),
                load_spec,
            )

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        # Get the KV events
        kv_cache_events = connector_output.kv_cache_events
        if not kv_cache_events or not isinstance(kv_cache_events, LMCacheKVEvents):
            return

        if self._kv_cache_events is None:
            self._kv_cache_events = kv_cache_events
        else:
            self._kv_cache_events.add_events(kv_cache_events.get_all_events())
            self._kv_cache_events.increment_workers(
                kv_cache_events.get_number_of_workers()
            )
        return

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        return self._lmcache_engine.request_finished(request, block_ids)

    def take_events(self) -> Iterable["KVCacheEvent"]:
        """
        Take the KV cache events from the connector.

        Yields:
            New KV cache events since the last call.
        """
        if self._kv_cache_events is not None:
            self._kv_cache_events.aggregate()
            kv_cache_events = self._kv_cache_events.get_all_events()
            yield from kv_cache_events
            self._kv_cache_events.clear_events()
            self._kv_cache_events = None
