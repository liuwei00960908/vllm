# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze logs emitted by VLLM_E2E_PERF_TRACE=1.

Example:
    python tools/analyze_e2e_perf_trace.py vllm.log --top 10
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
KV_RE = re.compile(rf"([A-Za-z0-9_]+)=({NUMBER_RE})")
ENGINE_RE = re.compile(
    rf"\[E2EPerf\]\[(EngineLoop|EngineCore)\].*?total_ms=({NUMBER_RE})(.*)"
)
GPU_RE = re.compile(
    rf"\[E2EPerf\]\[GPUModelRunner\]\s+label=(\S+)\s+phase=(\S+)"
    rf".*?total_ms=({NUMBER_RE})(.*)"
)
REQ_HEADER_RE = re.compile(
    r"request_id=(\S+)\s+internal_request_id=(\S+)\s+finish_reason=(\S+)"
)
REQ_TOKENS_RE = re.compile(
    r"prompt_tokens=(\d+)\s+cached_tokens=(\d+)\s+generated_tokens=(\d+)"
    r"\s+prefill_output_tokens=(\d+)\s+decode_output_tokens=(\d+)"
)
REQ_LAT_RE = re.compile(
    rf"ttft_ms=({NUMBER_RE})\s+e2e_ms=({NUMBER_RE})\s+queue_ms=({NUMBER_RE})"
    rf"\s+prefill_ms=({NUMBER_RE})\s+decode_total_ms=({NUMBER_RE})"
    rf"\s+inference_ms=({NUMBER_RE})"
)
REQ_OVERHEAD_RE = re.compile(
    rf"ttft_server_overhead_ms=({NUMBER_RE})\s+"
    rf"e2e_server_overhead_ms=({NUMBER_RE})"
)
REQ_DECODE_STEP_RE = re.compile(
    rf"step=(\d+)\s+generated_token=(\S+)\s+new_tokens=(\d+)"
    rf"\s+latency_ms=({NUMBER_RE})"
)


@dataclass
class TraceRecord:
    source: str
    label: str
    phase: str = ""
    total_ms: float = 0.0
    fields: dict[str, float] = field(default_factory=dict)
    line_no: int = 0


@dataclass
class RequestRecord:
    request_id: str = ""
    internal_request_id: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    cached_tokens: int = 0
    generated_tokens: int = 0
    prefill_output_tokens: int = 0
    decode_output_tokens: int = 0
    ttft_ms: float = 0.0
    e2e_ms: float = 0.0
    queue_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_total_ms: float = 0.0
    inference_ms: float = 0.0
    ttft_server_overhead_ms: float = 0.0
    e2e_server_overhead_ms: float = 0.0
    decode_steps: list[float] = field(default_factory=list)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def fmt_ms(value: float) -> str:
    return f"{value:9.3f}"


def parse_kv_fields(text: str) -> dict[str, float]:
    return {key: float(value) for key, value in KV_RE.findall(text)}


def parse_log(path: Path) -> tuple[list[TraceRecord], list[RequestRecord]]:
    traces: list[TraceRecord] = []
    requests: list[RequestRecord] = []
    current_request: RequestRecord | None = None
    in_timing_report = False

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            if match := ENGINE_RE.search(line):
                source, total_ms, rest = match.groups()
                traces.append(
                    TraceRecord(
                        source=source,
                        label=source,
                        total_ms=float(total_ms),
                        fields=parse_kv_fields(rest),
                        line_no=line_no,
                    )
                )
                continue

            if match := GPU_RE.search(line):
                label, phase, total_ms, rest = match.groups()
                traces.append(
                    TraceRecord(
                        source="GPUModelRunner",
                        label=label,
                        phase=phase,
                        total_ms=float(total_ms),
                        fields=parse_kv_fields(rest),
                        line_no=line_no,
                    )
                )
                continue

            if "Request timing report" in line:
                if current_request is not None:
                    requests.append(current_request)
                current_request = RequestRecord()
                in_timing_report = True
                continue

            if not in_timing_report or current_request is None:
                continue

            if match := REQ_HEADER_RE.search(line):
                (
                    current_request.request_id,
                    current_request.internal_request_id,
                    current_request.finish_reason,
                ) = match.groups()
                continue

            if match := REQ_TOKENS_RE.search(line):
                (
                    current_request.prompt_tokens,
                    current_request.cached_tokens,
                    current_request.generated_tokens,
                    current_request.prefill_output_tokens,
                    current_request.decode_output_tokens,
                ) = (int(value) for value in match.groups())
                continue

            if match := REQ_LAT_RE.search(line):
                (
                    current_request.ttft_ms,
                    current_request.e2e_ms,
                    current_request.queue_ms,
                    current_request.prefill_ms,
                    current_request.decode_total_ms,
                    current_request.inference_ms,
                ) = (float(value) for value in match.groups())
                continue

            if match := REQ_OVERHEAD_RE.search(line):
                (
                    current_request.ttft_server_overhead_ms,
                    current_request.e2e_server_overhead_ms,
                ) = (float(value) for value in match.groups())
                continue

            if match := REQ_DECODE_STEP_RE.search(line):
                current_request.decode_steps.append(float(match.group(4)))
                continue

            if line.strip() == "" and current_request.request_id:
                requests.append(current_request)
                current_request = None
                in_timing_report = False

    if current_request is not None and current_request.request_id:
        requests.append(current_request)

    return traces, requests


def print_distribution(name: str, values: list[float]) -> None:
    if not values:
        print(f"{name}: no samples")
        return
    print(
        f"{name:32s} count={len(values):6d} "
        f"mean={fmt_ms(mean(values))} p50={fmt_ms(percentile(values, 50))} "
        f"p90={fmt_ms(percentile(values, 90))} "
        f"p99={fmt_ms(percentile(values, 99))} max={fmt_ms(max(values))}"
    )


def summarize_traces(traces: list[TraceRecord], top: int) -> None:
    print("\n== Step Trace Summary ==")
    groups: dict[tuple[str, str, str], list[TraceRecord]] = defaultdict(list)
    for record in traces:
        groups[(record.source, record.label, record.phase)].append(record)

    for key in sorted(groups):
        source, label, phase = key
        title = "/".join(part for part in (source, label, phase) if part)
        records = groups[key]
        print_distribution(title, [record.total_ms for record in records])

        field_values: dict[str, list[float]] = defaultdict(list)
        for record in records:
            for field, value in record.fields.items():
                if field.endswith("_ms"):
                    field_values[field].append(value)

        ranked = sorted(
            (
                (mean(values), field, len(values))
                for field, values in field_values.items()
            ),
            reverse=True,
        )
        if ranked:
            print("  avg stage cost:")
            for avg, field, count in ranked[:top]:
                print(f"    {field:32s} mean={fmt_ms(avg)} count={count}")


def summarize_requests(requests: list[RequestRecord], top: int) -> None:
    print("\n== Request Timing Summary ==")
    if not requests:
        print("No request timing reports found.")
        return

    print_distribution("ttft_ms", [req.ttft_ms for req in requests])
    print_distribution("e2e_ms", [req.e2e_ms for req in requests])
    print_distribution("queue_ms", [req.queue_ms for req in requests])
    print_distribution("prefill_ms", [req.prefill_ms for req in requests])
    print_distribution("decode_total_ms", [req.decode_total_ms for req in requests])
    print_distribution(
        "ttft_server_overhead_ms",
        [req.ttft_server_overhead_ms for req in requests],
    )
    all_decode_steps = [
        latency for req in requests for latency in req.decode_steps
    ]
    print_distribution("decode_step_ms", all_decode_steps)

    print("\nTop slow requests by e2e_ms:")
    for req in sorted(requests, key=lambda item: item.e2e_ms, reverse=True)[:top]:
        print(
            f"  e2e={fmt_ms(req.e2e_ms)} ttft={fmt_ms(req.ttft_ms)} "
            f"prefill={fmt_ms(req.prefill_ms)} decode={fmt_ms(req.decode_total_ms)} "
            f"queue={fmt_ms(req.queue_ms)} prompt={req.prompt_tokens} "
            f"gen={req.generated_tokens} id={req.request_id}"
        )

    print("\nTop slow requests by ttft_ms:")
    for req in sorted(requests, key=lambda item: item.ttft_ms, reverse=True)[:top]:
        print(
            f"  ttft={fmt_ms(req.ttft_ms)} prefill={fmt_ms(req.prefill_ms)} "
            f"queue={fmt_ms(req.queue_ms)} overhead="
            f"{fmt_ms(req.ttft_server_overhead_ms)} prompt={req.prompt_tokens} "
            f"id={req.request_id}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize vLLM logs emitted by VLLM_E2E_PERF_TRACE=1."
    )
    parser.add_argument("log_file", type=Path)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    traces, requests = parse_log(args.log_file)
    print(f"Parsed {len(traces)} step traces and {len(requests)} request reports.")
    summarize_traces(traces, args.top)
    summarize_requests(requests, args.top)


if __name__ == "__main__":
    main()
