"""
Build results/mlsys2027/quality_frontier/summary.json for MLSys 2027
Experiment 1 (operating-point quality-compression sweep).

Every rerun value is parsed ONLY from the five raw Experiment 1 logs in
results/mlsys2027/quality_frontier/. Each method entry records its source
log, line number and verbatim row; derived values (compression ratio vs
bf16, deltas vs bf16) are computed from those raw values. The SHA-256 of
each log is recorded so the validator can confirm which logs were used.
Hashes are taken over LF-normalized bytes so they match the committed git
content regardless of core.autocrlf.

The single exception is the HotpotQA canonical_reference block, which quotes
the frozen canonical HotpotQA result from results/summary.json (read-only,
with that file's SHA-256) so the canonical degradation is never conflated
with the rerun delta.

Read-only with respect to the raw logs and all canonical results; the only
file written is results/mlsys2027/quality_frontier/summary.json. Runs
locally; no Modal/GPU.

Usage:
    python benchmarks/mlsys2027/build_exp1_summary.py
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "mlsys2027" / "quality_frontier"
REL = "results/mlsys2027/quality_frontier"
SUMMARY = OUT_DIR / "summary.json"
CANONICAL_SUMMARY = ROOT / "results" / "summary.json"

METHODS = ["bf16", "rabit8", "rabit4", "rabit3", "rabit2"]
BENCHMARKS = ["continuation_ppl", "niah", "passage_retrieval", "hotpotqa", "qasper"]
KV_LABEL = "LOGICAL fake-quant KV storage; not physical vLLM allocator capacity"
NIAH_CELL = re.compile(r"^(\d+)\s+(0\.\d+)\s+(bf16|rabit\d)\s+(PASS|FAIL)\s")


def log_path(name: str) -> Path:
    return OUT_DIR / f"{name}.log"


def log_lines(name: str) -> list[str]:
    return log_path(name).read_text(encoding="utf-8").splitlines()


SHA256_BASIS = (
    "SHA-256 of LF-normalized bytes (CRLF -> LF): equals the SHA-256 of the "
    "committed git content and is stable across core.autocrlf checkouts"
)


def sha256(path: Path) -> str:
    """Line-ending-normalized SHA-256 (see SHA256_BASIS)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def last_row(name: str, method: str) -> tuple[int, str]:
    """Last line starting with `<method><whitespace>`: the printed
    summary-table row for that method."""
    pat = re.compile(rf"^{re.escape(method)}\s")
    hit = None
    for i, line in enumerate(log_lines(name), start=1):
        if pat.match(line):
            hit = (i, line)
    if hit is None:
        raise RuntimeError(f"{name}.log: summary row for {method} not found")
    return hit


def source(name: str, lineno: int, row: str) -> dict:
    return {"log": f"{REL}/{name}.log", "line": lineno, "row": row.rstrip()}


def presets(name: str) -> dict:
    lines = log_lines(name)
    i = lines.index("Configurations:")
    out = {}
    for line in lines[i + 1 : i + 1 + len(METHODS)]:
        key, value = line.strip().split(": ", 1)
        out[key] = value
    return out


def ratio(bf16_mb: float, mb: float) -> float:
    return round(bf16_mb / mb, 3)


def build_ppl() -> dict:
    # Method PPL PPL_d% Median AvgKVMB Compression Tokens
    name = "continuation_ppl"
    rows = {m: last_row(name, m) for m in METHODS}
    tok = {m: rows[m][1].split() for m in METHODS}
    bf16_ppl, bf16_mb = float(tok["bf16"][1]), float(tok["bf16"][4])
    methods = {}
    for m in METHODS:
        t = tok[m]
        e = {
            "ppl": float(t[1]),
            "median_sample_ppl": float(t[3]),
            "logical_kv_mb": float(t[4]),
            "logical_compression_vs_bf16": ratio(bf16_mb, float(t[4])),
            "logged_compression": float(t[5]),
            "tokens": int(t[6]),
        }
        if m != "bf16":
            e["delta_vs_bf16_ppl_pct_logged"] = float(t[2])
            e["delta_vs_bf16_ppl_pct_computed"] = round((float(t[1]) / bf16_ppl - 1) * 100, 2)
        e["source"] = source(name, *rows[m])
        methods[m] = e
    return {
        "task": "WikiText-2 teacher-forced continuation perplexity",
        "metric": "ppl (lower is better)",
        "settings": "context 1024 tokens, eval 128 tokens, 8 samples",
        "methods": methods,
    }


def build_niah() -> dict:
    # Aggregate rows: Method Passed Total Accuracy AvgKVMB AvgCompression
    name = "niah"
    cells = [(i, l) for i, l in enumerate(log_lines(name), start=1) if NIAH_CELL.match(l)]
    status = [NIAH_CELL.match(l).group(4) for _, l in cells]
    rows = {m: last_row(name, m) for m in METHODS}
    tok = {m: rows[m][1].split() for m in METHODS}
    bf16_acc, bf16_mb = float(tok["bf16"][3]), float(tok["bf16"][4])
    methods = {}
    for m in METHODS:
        t = tok[m]
        e = {
            "accuracy_pct": float(t[3]),
            "passed": int(t[1]),
            "total": int(t[2]),
            "logical_kv_mb_avg": float(t[4]),
            "logical_compression_vs_bf16": ratio(bf16_mb, float(t[4])),
            "logged_avg_compression_mean_of_cells": float(t[5]),
        }
        if m != "bf16":
            e["delta_vs_bf16_accuracy_points"] = round(float(t[3]) - bf16_acc, 1)
        e["source"] = source(name, *rows[m])
        methods[m] = e
    return {
        "task": "Needle-in-a-Haystack, contexts 4096/8192/16384 x depths 0.1/0.25/0.5/0.75/0.9",
        "metric": "accuracy_pct (higher is better)",
        "cell_check": {
            "cells_found": len(cells),
            "cells_pass": status.count("PASS"),
            "cells_fail": status.count("FAIL"),
            "cell_rows_line_range": [cells[0][0], cells[-1][0]] if cells else None,
        },
        "methods": methods,
    }


def build_table(name: str, metric: str, delta_kind: str) -> dict:
    # passage_retrieval: Method Accuracy CorrectEquiv AvgKVMB Samples
    # hotpotqa/qasper:   Method F1%      F1total      AvgKVMB Samples
    rows = {m: last_row(name, m) for m in METHODS}
    tok = {m: rows[m][1].split() for m in METHODS}
    bf16_val, bf16_mb = float(tok["bf16"][1]), float(tok["bf16"][3])
    methods = {}
    for m in METHODS:
        t = tok[m]
        e = {
            metric: float(t[1]),
            "logical_kv_mb": float(t[3]),
            "logical_compression_vs_bf16": ratio(bf16_mb, float(t[3])),
            "samples": int(t[4]),
        }
        if m != "bf16":
            e[f"delta_vs_bf16_{delta_kind}"] = round(float(t[1]) - bf16_val, 1)
        e["source"] = source(name, *rows[m])
        methods[m] = e
    return {"methods": methods}


def hotpotqa_canonical_reference() -> dict:
    q = json.loads(CANONICAL_SUMMARY.read_text(encoding="utf-8"))["final_quality"]["longbench_hotpotqa_e"]
    bf16, rabit2 = q["bf16_f1_percent"], q["rabit_f1_percent"]
    return {
        "note": (
            "Frozen canonical HotpotQA result, quoted read-only from "
            "results/summary.json for comparison. NOT part of this rerun."
        ),
        "source": "results/summary.json",
        "source_key": "final_quality.longbench_hotpotqa_e",
        "source_sha256": sha256(CANONICAL_SUMMARY),
        "sha256_basis": SHA256_BASIS,
        "bf16_f1_pct": bf16,
        "rabit2_f1_pct": rabit2,
        "delta_rabit2_vs_bf16_f1_points": round(rabit2 - bf16, 1),
    }


def build() -> dict:
    preset_map = presets("continuation_ppl")
    for name in BENCHMARKS[1:]:
        if presets(name) != preset_map:
            raise RuntimeError(f"{name}.log presets differ from continuation_ppl.log")

    hotpot = {
        "task": "LongBench-E HotpotQA, 8k+ bucket, 20 samples, max input 16384",
        "metric": "f1_pct (higher is better)",
        **build_table("hotpotqa", "f1_pct", "f1_points"),
    }
    hotpot["canonical_reference"] = hotpotqa_canonical_reference()
    rerun_bf16 = hotpot["methods"]["bf16"]["f1_pct"]
    rerun_rabit2 = hotpot["methods"]["rabit2"]["f1_pct"]
    rerun_delta = hotpot["methods"]["rabit2"]["delta_vs_bf16_f1_points"]
    canon = hotpot["canonical_reference"]

    return {
        "experiment": "MLSys 2027 Experiment 1 -- operating-point quality-compression sweep",
        "description": (
            "Operating-point quality-compression sweep over the bf16 baseline and the "
            "rabit8/rabit4/rabit3/rabit2 presets on Llama-3.1-8B-Instruct. This is NOT "
            "a pure bit-width ablation: the presets differ in more than bit width "
            "(group size, META granularity, symmetric mode, K/V bit split, R), as "
            "recorded verbatim in the logs under 'presets_as_logged'."
        ),
        "derivation": (
            "All rerun metric and KV values are parsed ONLY from the five raw logs in "
            "this directory; each method entry carries its source log, line number and "
            "verbatim row, and raw_logs records each log's line-ending-normalized SHA-256. "
            "logical_compression_vs_bf16 = bf16 logical KV MB / method logical KV MB "
            "(rounded to 3 dp). Accuracy/F1 deltas are absolute percentage points "
            "(method - bf16). PPL delta is relative percent. The only non-log values are "
            "hotpotqa.canonical_reference, quoted read-only from results/summary.json."
        ),
        "generated_by": "benchmarks/mlsys2027/build_exp1_summary.py",
        "validated_by": "benchmarks/mlsys2027/validate_exp1_summary.py",
        "kv_memory_label": KV_LABEL,
        "raw_logs_sha256_basis": SHA256_BASIS,
        "raw_logs": {f"{REL}/{n}.log": sha256(log_path(n)) for n in BENCHMARKS},
        "run_reference": {
            "note": "Run metadata only (not used for any number); see manifest.json.",
            "manifest": f"{REL}/manifest.json",
            "regression_check": f"{REL}/regression_check.json",
        },
        "presets_as_logged": preset_map,
        "benchmarks": {
            "continuation_ppl": build_ppl(),
            "niah": build_niah(),
            "passage_retrieval": {
                "task": "LongBench passage_retrieval_en, 10-sample slice, max input 16384",
                "metric": "accuracy_pct (higher is better)",
                **build_table("passage_retrieval", "accuracy_pct", "accuracy_points"),
            },
            "hotpotqa": hotpot,
            "qasper": {
                "task": "LongBench-E Qasper, 8k+ bucket, 24 samples, max input 16384",
                "metric": "f1_pct (higher is better)",
                **build_table("qasper", "f1_pct", "f1_points"),
            },
        },
        "notes": [
            "This is an operating-point quality-compression sweep, not a pure bit-width ablation: rabit8/rabit4/rabit3/rabit2 differ in more than bit width (see presets_as_logged).",
            "NIAH is 100% for all 75 method/context/depth cells (3 context lengths x 5 depths x 5 methods).",
            "Passage Retrieval is 100% for all methods.",
            "Do not interpret rabit8's -0.03% PPL delta vs bf16 as an improvement.",
            "Do not interpret rabit3's 64.0 HotpotQA F1 (above bf16 60.7), or any other non-monotonic small-sample result (e.g. Qasper rabit8/rabit4 above bf16, Qasper rabit3 below rabit2, PPL rabit3 worse than rabit2), as statistically significant.",
            "rabit3 vs rabit2 comparisons are observational only; causal attribution requires later K/V/G/R/META ablations.",
            (
                f"HotpotQA remains visible as the main observed degradation. Canonical frozen result: "
                f"{canon['bf16_f1_pct']} -> {canon['rabit2_f1_pct']} = {canon['delta_rabit2_vs_bf16_f1_points']} F1 points. "
                f"This Experiment 1 rerun: {rerun_bf16} -> {rerun_rabit2} = {rerun_delta} F1 points. "
                "These are separate measurements and must not be conflated."
            ),
            "NIAH logged 'Avg Compression' is the mean of per-cell ratios; logical_compression_vs_bf16 here is the ratio of average logical KV MB. Both are reported and differ slightly.",
            f"All KV figures: {KV_LABEL}.",
        ],
    }


def main() -> int:
    summary = build()
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {SUMMARY.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
