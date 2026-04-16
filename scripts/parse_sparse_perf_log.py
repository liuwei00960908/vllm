#!/usr/bin/env python3
"""Summarize sparse performance logs from vLLM runs.

Parses lines like:
  [SparsePerf] window_steps=20 foo:total_ms=12.34,calls=4,avg_ms=3.085 | ...
  sparse dynamic update: req xxx layer=yyy - added 32 clusters (total 96 ...)

Usage:
  python scripts/parse_sparse_perf_log.py path/to/log.txt
  python scripts/parse_sparse_perf_log.py path/to/log.txt --json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


SPARSE_PERF_RE = re.compile(r"\[SparsePerf\]\s+window_steps=(\d+)\s+(.*)")
ENTRY_RE = re.compile(
    r"(?P<key>[^:|]+):total_ms=(?P<total_ms>\d+(?:\.\d+)?),"
    r"calls=(?P<calls>\d+),avg_ms=(?P<avg_ms>\d+(?:\.\d+)?)"
)
NO_SAMPLE_RE = re.compile(
    r"\[SparsePerf\]\s+steps=(?P<steps>\d+)\s+no sparse perf samples collected"
)
UPDATE_RE = re.compile(
    r"sparse dynamic update: req (?P<req>\S+) layer=(?P<layer>\S+).*?added "
    r"(?P<added>\d+) clusters.*?\(total (?P<total>\d+)"
)


@dataclass
class PerfAgg:
    total_ms: float = 0.0
    calls: int = 0
    windows_seen: int = 0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0


def parse_log(path: Path) -> dict:
    perf_by_key: dict[str, PerfAgg] = defaultdict(PerfAgg)
    dynamic_update_by_layer: dict[str, dict[str, int]] = defaultdict(
        lambda: {"events": 0, "added_clusters": 0, "last_total_clusters": 0}
    )
    window_steps: list[int] = []
    empty_windows = 0
    perf_lines = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            m = SPARSE_PERF_RE.search(line)
            if m:
                perf_lines += 1
                window_steps.append(int(m.group(1)))
                payload = m.group(2)
                for entry in payload.split(" | "):
                    entry = entry.strip()
                    em = ENTRY_RE.fullmatch(entry)
                    if em is None:
                        continue
                    key = em.group("key").strip()
                    agg = perf_by_key[key]
                    agg.total_ms += float(em.group("total_ms"))
                    agg.calls += int(em.group("calls"))
                    agg.windows_seen += 1
                continue

            m = NO_SAMPLE_RE.search(line)
            if m:
                empty_windows += 1
                continue

            m = UPDATE_RE.search(line)
            if m:
                layer = m.group("layer")
                stats = dynamic_update_by_layer[layer]
                stats["events"] += 1
                stats["added_clusters"] += int(m.group("added"))
                stats["last_total_clusters"] = int(m.group("total"))
                continue

    perf_summary = []
    for key, agg in sorted(
        perf_by_key.items(), key=lambda kv: kv[1].total_ms, reverse=True
    ):
        perf_summary.append(
            {
                "key": key,
                "total_ms": round(agg.total_ms, 3),
                "calls": agg.calls,
                "avg_ms": round(agg.avg_ms, 4),
                "windows_seen": agg.windows_seen,
            }
        )

    dynamic_summary = []
    for layer, stats in sorted(
        dynamic_update_by_layer.items(),
        key=lambda kv: kv[1]["added_clusters"],
        reverse=True,
    ):
        dynamic_summary.append({"layer": layer, **stats})

    return {
        "log_path": str(path),
        "perf_window_lines": perf_lines,
        "empty_windows": empty_windows,
        "window_steps_values": sorted(set(window_steps)),
        "perf_summary": perf_summary,
        "dynamic_update_summary": dynamic_summary,
    }


def render_text(summary: dict, top_n: int) -> str:
    lines: list[str] = []
    lines.append(f"log: {summary['log_path']}")
    lines.append(f"sparse perf windows: {summary['perf_window_lines']}")
    lines.append(f"empty windows: {summary['empty_windows']}")
    if summary["window_steps_values"]:
        vals = ", ".join(str(v) for v in summary["window_steps_values"])
        lines.append(f"window_steps values: {vals}")

    lines.append("")
    lines.append("Top sparse perf entries:")
    perf_rows = summary["perf_summary"][:top_n]
    if not perf_rows:
        lines.append("  (none)")
    else:
        for row in perf_rows:
            lines.append(
                "  {key}: total_ms={total_ms:.3f} calls={calls} "
                "avg_ms={avg_ms:.4f} windows={windows_seen}".format(**row)
            )

    lines.append("")
    lines.append("Dynamic update summary:")
    dyn_rows = summary["dynamic_update_summary"]
    if not dyn_rows:
        lines.append("  (none)")
    else:
        for row in dyn_rows[:top_n]:
            lines.append(
                "  {layer}: events={events} added_clusters={added_clusters} "
                "last_total_clusters={last_total_clusters}".format(**row)
            )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    summary = parse_log(args.logfile)
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    print(render_text(summary, top_n=args.top))


if __name__ == "__main__":
    main()
