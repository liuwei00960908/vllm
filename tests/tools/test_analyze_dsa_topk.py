"""Tests for the standalone DSA top-k log analyzer."""

from pathlib import Path

import pytest

from tools.analyze_dsa_topk import (
    LogFormatError,
    calculate_mtp_samples,
    calculate_overlap_samples,
    infer_base_seq_lengths,
    load_input_directory,
    parse_log_rows,
    reconstruct_union_topk,
)


def _line(rank, layer, row, request_id, positions):
    values = ", ".join(map(str, positions))
    return (
        f"(Worker_TP{rank} pid=1) INFO [sfa_v1.py:1] [DSA-TOPK] "
        f"layer=model.layers.{layer}.self_attn.attn row={row} "
        f"req={request_id} n_valid={len(positions)} pos_head=[{values}]\n"
    )


def _write_log(path: Path, *, omit_last=False):
    lines = []
    for step in range(3):
        for layer in range(2):
            lines.append(_line(1, layer, 0, "req-a", [999, 998]))
            lines.append(_line(0, layer, 0, "req-a", [step, step + 1]))
            lines.append(_line(0, layer, 1, "req-a", [step + 1, step + 2]))
            lines.append(_line(0, layer, 2, "?", [7, 7]))
    if omit_last:
        lines = lines[:-4]
    path.write_text("".join(lines), encoding="utf-8")


def test_reconstructs_tp0_mtp_union_and_ignores_padding(tmp_path):
    log = tmp_path / "log.txt"
    _write_log(log)
    rows = parse_log_rows(log, tp_rank=0, topk=2, num_layers=2)
    records = reconstruct_union_topk(rows, num_layers=2)

    assert len(records) == 6
    assert records[0].positions == frozenset({0, 1, 2})
    assert records[0].row_count == 2
    assert records[-1].step == 2
    assert infer_base_seq_lengths(records) == {"req-a": 3}

    mtp = calculate_mtp_samples(records, topk=2)
    assert mtp[0] == [0.75, 0.75, 0.75]
    overlap = calculate_overlap_samples(records, max_previous_steps=2)
    assert overlap[(0, 1)] == pytest.approx([2 / 3, 2 / 3])
    assert overlap[(0, 2)] == pytest.approx([2 / 3])


def test_incomplete_layer_cycle_is_fatal(tmp_path):
    log = tmp_path / "log.txt"
    _write_log(log, omit_last=True)
    rows = parse_log_rows(log, tp_rank=0, topk=2, num_layers=2)
    with pytest.raises(LogFormatError, match=r"missing layers \[1\]"):
        reconstruct_union_topk(rows, num_layers=2)


def test_non_topk_named_row_is_fatal(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text(_line(0, 0, 0, "req-a", [1]), encoding="utf-8")
    with pytest.raises(LogFormatError, match="expected configured topk=2"):
        parse_log_rows(log, tp_rank=0, topk=2, num_layers=1)


def test_parse_error_prints_previous_current_and_next_lines(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text(
        "before context\n"
        "(Worker_TP0 pid=1) INFO [sfa_v1.py:1] [DSA-TOPK] "
        "layer=model.layers.0.self_attn.attn row=0 req=req-a "
        "n_valid=2 pos_head=[1, broken]\n"
        "after context\n",
        encoding="utf-8",
    )

    with pytest.raises(LogFormatError) as caught:
        parse_log_rows(log, tp_rank=0, topk=2, num_layers=1)

    message = str(caught.value)
    assert f"log context: {log}" in message
    assert "  1: before context" in message
    assert "> 2:" in message
    assert "broken" in message
    assert "  3: after context" in message


def test_ignores_graph_padding_before_parsing_positions(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text(
        _line(0, 0, 0, "req-a", [1, 2])
        + "(Worker_TP0 pid=1) INFO [sfa_v1.py:1] [DSA-TOPK] "
        "layer=model.layers.0.self_attn.attn row=1 req=? "
        "n_valid=1026 pos_head=[garbage INFO [another log without closure\n",
        encoding="utf-8",
    )

    rows = parse_log_rows(log, tp_rank=0, topk=2, num_layers=1)

    assert len(rows) == 1
    assert rows[0].request_id == "req-a"
    assert rows[0].positions == (1, 2)


def test_ignores_real_rank_copy_embedded_in_padding_line(tmp_path):
    log = tmp_path / "log.txt"
    real = _line(0, 0, 0, "req-a", [1, 2])
    embedded = _line(2, 0, 0, "req-a", [1, 2]).rstrip()
    log.write_text(
        real + "(Worker_TP0 pid=1) INFO [sfa_v1.py:1] [DSA-TOPK] "
        "layer=model.layers.0.self_attn.attn row=1 req=? "
        "n_valid=1026 pos_head=[garbage " + embedded + "\n",
        encoding="utf-8",
    )

    rows = parse_log_rows(log, tp_rank=0, topk=2, num_layers=1)

    assert len(rows) == 1
    assert rows[0].request_id == "req-a"
    assert rows[0].positions == (1, 2)


def test_rejects_rank_with_interleaved_worker_copy(tmp_path):
    log = tmp_path / "log.txt"
    complete = _line(2, 0, 0, "req-a", [1, 2]).rstrip()
    log.write_text(
        "(Worker_TP0 pid=1) INFO [sfa_v1.py:1] [DSA-TOPK] "
        "layer=model.layers.0.self_attn.attn row=0 req=req-a "
        "n_valid=2 pos_head=[1, broken " + complete + "\n",
        encoding="utf-8",
    )

    with pytest.raises(LogFormatError, match="no \\[DSA-TOPK\\] records"):
        parse_log_rows(log, tp_rank=0, topk=2, num_layers=1)


def test_loads_one_request_per_text_file(tmp_path):
    first = tmp_path / "01.txt"
    second = tmp_path / "02.txt"
    _write_log(first)
    second.write_text(
        first.read_text(encoding="utf-8").replace("req-a", "req-b"),
        encoding="utf-8",
    )

    rows, records = load_input_directory(tmp_path, tp_rank=0, topk=2, num_layers=2)

    assert len(rows) == 24
    assert {item.request_id for item in records} == {"req-a", "req-b"}
    assert max(item.step for item in records) == 2


def test_rejects_multiple_requests_in_one_file(tmp_path):
    log = tmp_path / "01.txt"
    _write_log(log)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(_line(0, 0, 0, "req-b", [1, 2]))
        stream.write(_line(0, 1, 0, "req-b", [1, 2]))

    with pytest.raises(LogFormatError, match="exactly one"):
        load_input_directory(tmp_path, tp_rank=0, topk=2, num_layers=2)


def test_auto_rank_merges_partial_clean_layers_across_ranks(tmp_path):
    log = tmp_path / "01.txt"
    log.write_text(
        _line(0, 0, 0, "req-a", [1, 2]) + _line(1, 1, 0, "req-a", [2, 3]),
        encoding="utf-8",
    )

    rows, records = load_input_directory(tmp_path, tp_rank=None, topk=2, num_layers=2)

    assert len(rows) == 2
    assert len(records) == 2
    assert [item.layer_index for item in records] == [0, 1]


def test_skips_incomplete_file_and_keeps_valid_file(tmp_path, capsys):
    bad = tmp_path / "01.txt"
    good = tmp_path / "02.txt"
    bad.write_text(
        _line(0, 0, 0, "req-bad", [1, 2]),
        encoding="utf-8",
    )
    _write_log(good)

    rows, records = load_input_directory(tmp_path, tp_rank=0, topk=2, num_layers=2)

    assert len(rows) == 12
    assert {item.request_id for item in records} == {"req-a"}
    output = capsys.readouterr().out
    assert f"SKIP {bad}:" in output
    assert "Skipped 1 of 2 input log files" in output
