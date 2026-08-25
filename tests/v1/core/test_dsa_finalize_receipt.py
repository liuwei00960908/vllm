# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for finalize-time decode-window receipt draining (M1).

Covers: KVConnectorModelRunnerMixin.finalize_kv_connector() returning the
completed decode-window saves drained after wait_for_save (duck-typed),
clear-on-error guarantee, and the per-request max-merge semantics used by
the spec-decode call sites.

Provenance: fork kv_connector_model_runner_mixin.py:81-101 and
fork gpu_model_runner.py:4043-4059.
"""

from typing import Any

import pytest

import vllm.v1.worker.kv_connector_model_runner_mixin as mixin_module
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin,
)


class _RecordingConnector:
    """Duck-typed connector recording the finalize call sequence."""

    def __init__(
        self,
        saves: dict[str, int] | None = None,
        saves_attr: bool = True,
        fail_wait: bool = False,
    ) -> None:
        self.saves = saves or {}
        self.saves_attr = saves_attr
        self.fail_wait = fail_wait
        self.calls: list[str] = []

    def wait_for_save(self) -> None:
        self.calls.append("wait_for_save")
        if self.fail_wait:
            raise RuntimeError("simulated wait_for_save failure")

    def clear_connector_metadata(self) -> None:
        self.calls.append("clear_connector_metadata")

    def __getattr__(self, name: str) -> Any:
        if name == "get_completed_decode_window_saves" and self.saves_attr:
            self.calls.append("get_completed_decode_window_saves")
            return lambda: self.saves
        raise AttributeError(name)


@pytest.fixture
def patch_connector(monkeypatch: pytest.MonkeyPatch):
    def _install(connector: _RecordingConnector | None) -> None:
        monkeypatch.setattr(
            mixin_module, "has_kv_transfer_group", lambda: connector is not None
        )
        monkeypatch.setattr(
            mixin_module, "get_kv_transfer_group", lambda: connector
        )

    return _install


def test_finalize_returns_saves_after_wait(patch_connector):
    connector = _RecordingConnector(saves={"r1": 512, "r2": 256})
    patch_connector(connector)

    saves = KVConnectorModelRunnerMixin.finalize_kv_connector()

    assert saves == {"r1": 512, "r2": 256}
    assert connector.calls == [
        "wait_for_save",
        "get_completed_decode_window_saves",
        "clear_connector_metadata",
    ]


def test_finalize_duck_types_official_connectors(patch_connector):
    connector = _RecordingConnector(saves_attr=False)
    patch_connector(connector)

    saves = KVConnectorModelRunnerMixin.finalize_kv_connector()

    assert saves == {}
    assert connector.calls == ["wait_for_save", "clear_connector_metadata"]


def test_finalize_without_connector_returns_empty(patch_connector):
    patch_connector(None)

    assert KVConnectorModelRunnerMixin.finalize_kv_connector() == {}


def test_finalize_clears_metadata_even_when_wait_fails(patch_connector):
    connector = _RecordingConnector(fail_wait=True)
    patch_connector(connector)

    with pytest.raises(RuntimeError, match="simulated wait_for_save failure"):
        KVConnectorModelRunnerMixin.finalize_kv_connector()

    assert connector.calls == ["wait_for_save", "clear_connector_metadata"]


def test_merge_keeps_max_committed_end_per_request():
    # Mirrors the spec-decode call sites (core: self.kv_connector_output,
    # ascend: the local kv_connector_output captured in sample_tokens).
    output = KVConnectorOutput(
        completed_decode_window_saves={"r1": 768, "r2": 256},
    )
    drained = {"r1": 512, "r2": 1024, "r3": 128}

    for req_id, window_end in drained.items():
        output.completed_decode_window_saves[req_id] = max(
            output.completed_decode_window_saves.get(req_id, 0), window_end
        )

    assert output.completed_decode_window_saves == {
        "r1": 768,  # keeps the more advanced existing frontier
        "r2": 1024,  # advances
        "r3": 128,  # newly reported
    }


def test_merge_creates_output_when_missing():
    output: KVConnectorOutput | None = None
    drained = {"r1": 256}

    if drained:
        if output is None:
            output = KVConnectorOutput()
        for req_id, window_end in drained.items():
            output.completed_decode_window_saves[req_id] = max(
                output.completed_decode_window_saves.get(req_id, 0), window_end
            )

    assert output is not None
    assert output.completed_decode_window_saves == {"r1": 256}


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
