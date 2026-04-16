#!/usr/bin/env python3
"""Summarize sparse performance logs from vLLM runs.

Parses lines like:
  [SparsePerf] window_steps=20 foo:total_ms=12.34,calls=4,avg_ms=3.085 | ...
  [SparsePerfFA] compact_kv_gather total_ms=7.240 gather_ms=3.926 fa_ms=2.168 ...
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
from dataclasses import dataclass
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
SPARSE_PERF_FA_RE = re.compile(
    r"\[SparsePerfFA\]\s+(?P<key>\S+)\s+total_ms=(?P<total_ms>\d+(?:\.\d+)?)\s+"
    r"gather_ms=(?P<gather_ms>\d+(?:\.\d+)?)\s+fa_ms=(?P<fa_ms>\d+(?:\.\d+)?)"
    r"(?:\s+num_q_heads=(?P<num_q_heads>\d+))?"
    r"(?:\s+num_tok=(?P<num_tok>\d+))?"
    r"(?:\s+max_seqlen_q=(?P<max_seqlen_q>\d+))?"
)


@dataclass
class PerfAgg:
    total_ms: float = 0.0
    calls: int = 0
    windows_seen: int = 0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0


@dataclass
class FaAgg:
    total_ms: float = 0.0
    gather_ms: float = 0.0
    fa_ms: float = 0.0
    calls: int = 0
    num_q_heads_sum: int = 0
    num_tok_sum: int = 0
    max_seqlen_q_max: int = 0

    @property
    def avg_total_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0

    @property
    def avg_gather_ms(self) -> float:
        return self.gather_ms / self.calls if self.calls else 0.0

    @property
    def avg_fa_ms(self) -> float:
        return self.fa_ms / self.calls if self.calls else 0.0

    @property
    def gather_share(self) -> float:
        return self.gather_ms / self.total_ms if self.total_ms else 0.0

    @property
    def fa_share(self) -> float:
        return self.fa_ms / self.total_ms if self.total_ms else 0.0


def parse_log(path: Path) -> dict:
    perf_by_key: dict[str, PerfAgg] = defaultdict(PerfAgg)
    fa_by_key: dict[str, FaAgg] = defaultdict(FaAgg)
    dynamic_update_by_layer: dict[str, dict[str, int]] = defaultdict(
        lambda: {"events": 0, "added_clusters": 0, "last_total_clusters": 0}
    )
    window_steps: list[int] = []
    empty_windows = 0
    perf_lines = 0
    perf_fa_lines = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
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

            m = SPARSE_PERF_FA_RE.search(line)
            if m:
                perf_fa_lines += 1
                key = m.group("key")
                agg = fa_by_key[key]
                agg.total_ms += float(m.group("total_ms"))
                agg.gather_ms += float(m.group("gather_ms"))
                agg.fa_ms += float(m.group("fa_ms"))
                agg.calls += 1
                if m.group("num_q_heads") is not None:
                    agg.num_q_heads_sum += int(m.group("num_q_heads"))
                if m.group("num_tok") is not None:
                    agg.num_tok_sum += int(m.group("num_tok"))
                if m.group("max_seqlen_q") is not None:
                    agg.max_seqlen_q_max = max(
                        agg.max_seqlen_q_max, int(m.group("max_seqlen_q"))
                    )
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

    fa_summary = []
    for key, agg in sorted(
        fa_by_key.items(), key=lambda kv: kv[1].total_ms, reverse=True
    ):
        fa_summary.append(
            {
                "key": key,
                "total_ms": round(agg.total_ms, 3),
                "gather_ms": round(agg.gather_ms, 3),
                "fa_ms": round(agg.fa_ms, 3),
                "calls": agg.calls,
                "avg_total_ms": round(agg.avg_total_ms, 4),
                "avg_gather_ms": round(agg.avg_gather_ms, 4),
                "avg_fa_ms": round(agg.avg_fa_ms, 4),
                "gather_share": round(agg.gather_share, 4),
                "fa_share": round(agg.fa_share, 4),
                "avg_num_q_heads": round(
                    agg.num_q_heads_sum / agg.calls, 3
                ) if agg.calls else 0.0,
                "avg_num_tok": round(
                    agg.num_tok_sum / agg.calls, 3
                ) if agg.calls else 0.0,
                "max_seqlen_q_max": agg.max_seqlen_q_max,
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
        "perf_fa_lines": perf_fa_lines,
        "empty_windows": empty_windows,
        "window_steps_values": sorted(set(window_steps)),
        "perf_summary": perf_summary,
        "fa_summary": fa_summary,
        "dynamic_update_summary": dynamic_summary,
    }


def render_text(summary: dict, top_n: int) -> str:
    lines: list[str] = []
    lines.append(f"log: {summary['log_path']}")
    lines.append(f"sparse perf windows: {summary['perf_window_lines']}")
    lines.append(f"sparse perf fa lines: {summary['perf_fa_lines']}")
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
    lines.append("Top SparsePerfFA entries:")
    fa_rows = summary["fa_summary"][:top_n]
    if not fa_rows:
        lines.append("  (none)")
    else:
        for row in fa_rows:
            lines.append(
                "  {key}: total_ms={total_ms:.3f} gather_ms={gather_ms:.3f} "
                "fa_ms={fa_ms:.3f} calls={calls} avg_total_ms={avg_total_ms:.4f} "
                "avg_gather_ms={avg_gather_ms:.4f} avg_fa_ms={avg_fa_ms:.4f} "
                "gather_share={gather_share:.2%} fa_share={fa_share:.2%} "
                "avg_num_q_heads={avg_num_q_heads} avg_num_tok={avg_num_tok} "
                "max_seqlen_q_max={max_seqlen_q_max}".format(**row)
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
