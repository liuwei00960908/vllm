#!/usr/bin/env python3
"""Summarize sparse performance logs from vLLM runs.

Parses lines like:
  [SparsePerf] window_steps=20 foo:total_ms=12.34,calls=4,avg_ms=3.085 | ...
  [SparsePerfFA] compact_kv_gather total_ms=7.240 gather_ms=3.926 fa_ms=2.168 ...
  [DecodePerf] mode=sparse total_ms=10.0 preprocess_ms=...
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
    r"(?P<key>.+?):total_ms=(?P<total_ms>\d+(?:\.\d+)?),"
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
FULL_PERF_FA_RE = re.compile(
    r"\[FullPerfFA\]\s+(?P<key>\S+)\s+total_ms=(?P<total_ms>\d+(?:\.\d+)?)\s+"
    r"fa_ms=(?P<fa_ms>\d+(?:\.\d+)?)"
    r"(?:\s+num_q_heads=(?P<num_q_heads>\d+))?"
    r"(?:\s+num_tok=(?P<num_tok>\d+))?"
    r"(?:\s+max_seqlen_q=(?P<max_seqlen_q>\d+))?"
    r"(?:\s+max_seqlen_k=(?P<max_seqlen_k>\d+))?"
)
DECODE_PERF_RE = re.compile(
    r"\[DecodePerf\]\s+mode=(?P<mode>\S+)\s+"
    r"total_ms=(?P<total_ms>\d+(?:\.\d+)?)\s+"
    r"preprocess_ms=(?P<preprocess_ms>\d+(?:\.\d+)?)\s+"
    r"attn_metadata_ms=(?P<attn_metadata_ms>\d+(?:\.\d+)?)\s+"
    r"forward_ms=(?P<forward_ms>\d+(?:\.\d+)?)\s+"
    r"postprocess_ms=(?P<postprocess_ms>\d+(?:\.\d+)?)\s+"
    r"other_ms=(?P<other_ms>\d+(?:\.\d+)?)\s+"
    r"num_reqs=(?P<num_reqs>\d+)\s+"
    r"num_tokens=(?P<num_tokens>\d+)\s+"
    r"max_scheduled=(?P<max_scheduled>\d+)\s+"
    r"has_kv_connector=(?P<has_kv_connector>\d+)\s+"
    r"cudagraph=(?P<cudagraph>\S+)"
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


@dataclass
class DecodePerfAgg:
    total_ms: float = 0.0
    preprocess_ms: float = 0.0
    attn_metadata_ms: float = 0.0
    forward_ms: float = 0.0
    postprocess_ms: float = 0.0
    other_ms: float = 0.0
    calls: int = 0
    num_reqs_sum: int = 0
    num_tokens_sum: int = 0
    max_scheduled_max: int = 0
    has_kv_connector_sum: int = 0
    cudagraph_counts: dict[str, int] | None = None

    def add(self, m: re.Match[str]) -> None:
        self.total_ms += float(m.group("total_ms"))
        self.preprocess_ms += float(m.group("preprocess_ms"))
        self.attn_metadata_ms += float(m.group("attn_metadata_ms"))
        self.forward_ms += float(m.group("forward_ms"))
        self.postprocess_ms += float(m.group("postprocess_ms"))
        self.other_ms += float(m.group("other_ms"))
        self.calls += 1
        self.num_reqs_sum += int(m.group("num_reqs"))
        self.num_tokens_sum += int(m.group("num_tokens"))
        self.max_scheduled_max = max(
            self.max_scheduled_max, int(m.group("max_scheduled"))
        )
        self.has_kv_connector_sum += int(m.group("has_kv_connector"))
        if self.cudagraph_counts is None:
            self.cudagraph_counts = {}
        cudagraph = m.group("cudagraph")
        self.cudagraph_counts[cudagraph] = self.cudagraph_counts.get(cudagraph, 0) + 1

    def avg(self, value: float) -> float:
        return value / self.calls if self.calls else 0.0


def parse_log(path: Path) -> dict:
    perf_by_key: dict[str, PerfAgg] = defaultdict(PerfAgg)
    fa_by_key: dict[str, FaAgg] = defaultdict(FaAgg)
    full_fa_by_key: dict[str, FaAgg] = defaultdict(FaAgg)
    decode_perf_by_mode: dict[str, DecodePerfAgg] = defaultdict(DecodePerfAgg)
    dynamic_update_by_layer: dict[str, dict[str, int]] = defaultdict(
        lambda: {"events": 0, "added_clusters": 0, "last_total_clusters": 0}
    )
    window_steps: list[int] = []
    empty_windows = 0
    perf_lines = 0
    perf_fa_lines = 0
    full_perf_fa_lines = 0
    decode_perf_lines = 0

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

            m = FULL_PERF_FA_RE.search(line)
            if m:
                full_perf_fa_lines += 1
                key = m.group("key")
                agg = full_fa_by_key[key]
                agg.total_ms += float(m.group("total_ms"))
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

            m = DECODE_PERF_RE.search(line)
            if m:
                decode_perf_lines += 1
                decode_perf_by_mode[m.group("mode")].add(m)
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

    full_fa_summary = []
    for key, agg in sorted(
        full_fa_by_key.items(), key=lambda kv: kv[1].total_ms, reverse=True
    ):
        full_fa_summary.append(
            {
                "key": key,
                "total_ms": round(agg.total_ms, 3),
                "fa_ms": round(agg.fa_ms, 3),
                "calls": agg.calls,
                "avg_total_ms": round(agg.avg_total_ms, 4),
                "avg_fa_ms": round(agg.avg_fa_ms, 4),
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

    decode_perf_summary = []
    for mode, agg in sorted(
        decode_perf_by_mode.items(), key=lambda kv: kv[0]
    ):
        decode_perf_summary.append(
            {
                "mode": mode,
                "total_ms": round(agg.total_ms, 3),
                "calls": agg.calls,
                "avg_total_ms": round(agg.avg(agg.total_ms), 4),
                "avg_preprocess_ms": round(agg.avg(agg.preprocess_ms), 4),
                "avg_attn_metadata_ms": round(agg.avg(agg.attn_metadata_ms), 4),
                "avg_forward_ms": round(agg.avg(agg.forward_ms), 4),
                "avg_postprocess_ms": round(agg.avg(agg.postprocess_ms), 4),
                "avg_other_ms": round(agg.avg(agg.other_ms), 4),
                "preprocess_share": round(
                    agg.preprocess_ms / agg.total_ms, 4
                ) if agg.total_ms else 0.0,
                "attn_metadata_share": round(
                    agg.attn_metadata_ms / agg.total_ms, 4
                ) if agg.total_ms else 0.0,
                "forward_share": round(
                    agg.forward_ms / agg.total_ms, 4
                ) if agg.total_ms else 0.0,
                "postprocess_share": round(
                    agg.postprocess_ms / agg.total_ms, 4
                ) if agg.total_ms else 0.0,
                "other_share": round(
                    agg.other_ms / agg.total_ms, 4
                ) if agg.total_ms else 0.0,
                "avg_num_reqs": round(
                    agg.num_reqs_sum / agg.calls, 3
                ) if agg.calls else 0.0,
                "avg_num_tokens": round(
                    agg.num_tokens_sum / agg.calls, 3
                ) if agg.calls else 0.0,
                "max_scheduled_max": agg.max_scheduled_max,
                "kv_connector_calls": agg.has_kv_connector_sum,
                "cudagraph_counts": agg.cudagraph_counts or {},
            }
        )
    decode_perf_comparison = None
    decode_by_mode = {row["mode"]: row for row in decode_perf_summary}
    if "sparse" in decode_by_mode and "full" in decode_by_mode:
        sparse = decode_by_mode["sparse"]
        full = decode_by_mode["full"]

        def ratio(key: str) -> float:
            denom = float(full.get(key, 0.0))
            return round(float(sparse.get(key, 0.0)) / denom, 4) if denom else 0.0

        decode_perf_comparison = {
            "sparse_vs_full_avg_total_ratio": ratio("avg_total_ms"),
            "sparse_vs_full_avg_preprocess_ratio": ratio("avg_preprocess_ms"),
            "sparse_vs_full_avg_attn_metadata_ratio": ratio(
                "avg_attn_metadata_ms"
            ),
            "sparse_vs_full_avg_forward_ratio": ratio("avg_forward_ms"),
            "sparse_vs_full_avg_postprocess_ratio": ratio(
                "avg_postprocess_ms"
            ),
        }

    return {
        "log_path": str(path),
        "perf_window_lines": perf_lines,
        "perf_fa_lines": perf_fa_lines,
        "full_perf_fa_lines": full_perf_fa_lines,
        "decode_perf_lines": decode_perf_lines,
        "empty_windows": empty_windows,
        "window_steps_values": sorted(set(window_steps)),
        "perf_summary": perf_summary,
        "fa_summary": fa_summary,
        "full_fa_summary": full_fa_summary,
        "decode_perf_summary": decode_perf_summary,
        "decode_perf_comparison": decode_perf_comparison,
        "dynamic_update_summary": dynamic_summary,
    }


def render_text(summary: dict, top_n: int) -> str:
    lines: list[str] = []
    lines.append(f"log: {summary['log_path']}")
    lines.append(f"sparse perf windows: {summary['perf_window_lines']}")
    lines.append(f"sparse perf fa lines: {summary['perf_fa_lines']}")
    lines.append(f"full perf fa lines: {summary['full_perf_fa_lines']}")
    lines.append(f"decode perf lines: {summary['decode_perf_lines']}")
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

    lines.append("")
    lines.append("DecodePerf summary:")
    decode_rows = summary["decode_perf_summary"]
    if not decode_rows:
        lines.append("  (none)")
    else:
        for row in decode_rows:
            lines.append(
                "  {mode}: total_ms={total_ms:.3f} calls={calls} "
                "avg_total_ms={avg_total_ms:.4f} "
                "avg_preprocess_ms={avg_preprocess_ms:.4f} "
                "avg_attn_metadata_ms={avg_attn_metadata_ms:.4f} "
                "avg_forward_ms={avg_forward_ms:.4f} "
                "avg_postprocess_ms={avg_postprocess_ms:.4f} "
                "avg_other_ms={avg_other_ms:.4f} "
                "forward_share={forward_share:.2%} "
                "avg_num_reqs={avg_num_reqs} avg_num_tokens={avg_num_tokens} "
                "kv_connector_calls={kv_connector_calls} "
                "cudagraph_counts={cudagraph_counts}".format(**row)
            )
    if summary.get("decode_perf_comparison"):
        cmp_row = summary["decode_perf_comparison"]
        lines.append(
            "  sparse/full ratios: avg_total={sparse_vs_full_avg_total_ratio:.4f} "
            "preprocess={sparse_vs_full_avg_preprocess_ratio:.4f} "
            "attn_metadata={sparse_vs_full_avg_attn_metadata_ratio:.4f} "
            "forward={sparse_vs_full_avg_forward_ratio:.4f} "
            "postprocess={sparse_vs_full_avg_postprocess_ratio:.4f}".format(
                **cmp_row
            )
        )

    lines.append("")
    lines.append("Top FullPerfFA entries:")
    full_fa_rows = summary["full_fa_summary"][:top_n]
    if not full_fa_rows:
        lines.append("  (none)")
    else:
        for row in full_fa_rows:
            lines.append(
                "  {key}: total_ms={total_ms:.3f} fa_ms={fa_ms:.3f} calls={calls} "
                "avg_total_ms={avg_total_ms:.4f} avg_fa_ms={avg_fa_ms:.4f} "
                "fa_share={fa_share:.2%} avg_num_q_heads={avg_num_q_heads} "
                "avg_num_tok={avg_num_tok} max_seqlen_q_max={max_seqlen_q_max}"
                .format(**row)
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
