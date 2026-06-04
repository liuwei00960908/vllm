# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from collections.abc import Callable
from functools import wraps

from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    is_v1_kv_transfer_group,
)


def maybe_transfer_kv_layer(func: Callable) -> Callable:
    """Decorator that handles KV layer transfer prior and after execution of
    an attention layer, if enabled. Otherwise, the wrapper is a no-op.

    On entry: waits for the KV layer from the connector.
    On exit: saves the KV layer to the connector.
    """
    # Import at runtime to avoid circular dependency
    from vllm.model_executor.layers.attention.attention import get_attention_context

    # Inspect the signature ONCE when the decorator is applied.
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    # Find the index of 'layer_name' parameter.
    try:
        layer_name_index = param_names.index("layer_name")
    except ValueError as e:
        raise TypeError(
            f"Function {func.__name__} must have a 'layer_name' parameter"
        ) from e
    query_index = param_names.index("query") if "query" in param_names else None
    key_index = param_names.index("key") if "key" in param_names else None
    value_index = param_names.index("value") if "value" in param_names else None

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
            return func(*args, **kwargs)

        layer_name: str = args[layer_name_index]

        # Extract attention context (metadata, layer, kv_cache, layer_slot_mapping)
        attn_metadata, attn_layer, kv_cache, _ = get_attention_context(layer_name)
        connector = get_kv_transfer_group()
        if attn_metadata is None or not connector.has_connector_metadata():
            return func(*args, **kwargs)

        if (
            query_index is not None
            and key_index is not None
            and value_index is not None
            and hasattr(attn_layer, "impl")
            and hasattr(attn_layer.impl, "prepare_layer_for_load")
        ):
            query = args[query_index] if len(args) > query_index else kwargs.get("query")
            key = args[key_index] if len(args) > key_index else kwargs.get("key")
            value = args[value_index] if len(args) > value_index else kwargs.get("value")
            attn_layer.impl.prepare_layer_for_load(
                attn_layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
            )

        # Wait for KV layer on entry
        connector.wait_for_layer_load(
            layer_name,
            kv_layer=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Execute the function
        result = func(*args, **kwargs)

        # Save KV cache layer on exit
        connector.save_kv_layer(layer_name, kv_cache, attn_metadata)

        return result

    return wrapper
