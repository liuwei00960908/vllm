# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the decode-window receipt pipeline (replay B1c).

Covers: KVConnectorOutput.completed_decode_window_saves default/emptiness,
KVOutputAggregator per-request max merging across workers, and the
MultiConnector sub-connector aggregation (duck-typed).
"""

from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.outputs import KVConnectorOutput


def _output(**overrides) -> KVConnectorOutput:
    base = KVConnectorOutput()
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _runner_output(kv_output: KVConnectorOutput | None):
    return SimpleNamespace(kv_connector_output=kv_output)


def test_default_is_empty_dict_and_counts_as_empty():
    out = KVConnectorOutput()
    assert out.completed_decode_window_saves == {}
    assert out.is_empty()
    out.completed_decode_window_saves = {"r1": 256}
    assert not out.is_empty()


def test_aggregator_merges_per_request_max():
    agg = KVOutputAggregator(expected_finished_count=8)
    outputs = [
        _runner_output(_output(completed_decode_window_saves={"r1": 512})),
        _runner_output(_output(completed_decode_window_saves={"r1": 768, "r2": 256})),
        _runner_output(_output()),  # worker without receipts
    ] + [_runner_output(None)] * 5  # remaining TP ranks
    merged = agg.aggregate(outputs, output_rank=0)
    assert merged.kv_connector_output.completed_decode_window_saves == {
        "r1": 768,
        "r2": 256,
    }


def test_aggregator_all_empty_keeps_empty():
    agg = KVOutputAggregator(expected_finished_count=2)
    outputs = [_runner_output(_output()), _runner_output(_output())]
    merged = agg.aggregate(outputs, output_rank=0)
    assert merged.kv_connector_output.completed_decode_window_saves == {}


def test_multi_connector_aggregates_duck_typed():
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )

    class _WithSaves:
        def get_completed_decode_window_saves(self):
            return {"r1": 256, "r2": 1024}

    class _NewerSaves:
        def get_completed_decode_window_saves(self):
            return {"r1": 512}

    class _Plain:  # official-style connector without the method
        pass

    multi = MultiConnector.__new__(MultiConnector)
    multi._connectors = [_WithSaves(), _NewerSaves(), _Plain()]
    assert multi.get_completed_decode_window_saves() == {
        "r1": 512,
        "r2": 1024,
    }


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
