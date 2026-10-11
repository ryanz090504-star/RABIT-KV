"""
Validate results/mlsys2027/quality_frontier/summary.json against the five
raw Experiment 1 logs.

Checks:
  * each recorded raw log SHA-256 matches the log on disk;
  * every method's source row is verbatim the cited log line, for that method;
  * every raw value appears as a numeric token of its source row;
  * every compression ratio and delta vs bf16 recomputes exactly;
  * computed PPL deltas agree with the logged PPL d% column, and logged
    compression columns agree with the derived ratio;
  * NIAH: exactly 75 distinct method/context/depth cells, all PASS;
  * Passage Retrieval 100% for all methods;
  * HotpotQA canonical_reference: the SHA-256 of results/summary.json is
    verified first, then its values are compared; canonical and rerun
    deltas are kept separate;

SHA-256s are over LF-normalized bytes (CRLF -> LF), so they match the
committed git content regardless of core.autocrlf.
  * numbers quoted in the notes match the data.

Read-only: never modifies the logs, summary.json, or canonical results.
Runs locally; no Modal/GPU. Exits non-zero on any failure.

Usage:
    python benchmarks/mlsys2027/validate_exp1_summary.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "mlsys2027" / "quality_frontier"
SUMMARY = OUT_DIR / "summary.json"
CANONICAL_SUMMARY = ROOT / "results" / "summary.json"

METHODS = ["bf16", "rabit8", "rabit4", "rabit3", "rabit2"]
BENCHMARKS = ["continuation_ppl", "niah", "passage_retrieval", "hotpotqa", "qasper"]
NIAH_CELL = re.compile(r"^(\d+)\s+(0\.\d+)\s+(bf16|rabit\d)\s+(PASS|FAIL)\s")
RAW_KEYS = {
    "ppl", "median_sample_ppl", "logged_compression", "tokens",
    "accuracy_pct", "passed", "total", "logical_kv_mb_avg",
    "logged_avg_compression_mean_of_cells", "logical_kv_mb", "samples",
    "f1_pct", "delta_vs_bf16_ppl_pct_logged",
}


def sha256(path: Path) -> str:
    """SHA-256 of LF-normalized bytes (CRLF -> LF), matching the builder:
    equals the committed git content's hash under any core.autocrlf."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def in_row(value: float, tokens: list[str]) -> bool:
    for t in tokens:
        try:
            if float(t) == float(value):
                return True
        except ValueError:
            pass
    return False


def main() -> int:
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    fails: list[str] = []
    counts = {"log_hashes": 0, "source_rows": 0, "raw_values": 0, "derived_values": 0, "note_numbers": 0}

    # Raw log provenance.
    expected_logs = {f"results/mlsys2027/quality_frontier/{n}.log" for n in BENCHMARKS}
    if set(s["raw_logs"]) != expected_logs:
        fails.append(f"raw_logs keys {sorted(s['raw_logs'])} != the five Experiment 1 logs")
    for rel, digest in s["raw_logs"].items():
        counts["log_hashes"] += 1
        if sha256(ROOT / rel) != digest:
            fails.append(f"{rel}: SHA-256 differs from summary.json")

    for bname in BENCHMARKS:
        b = s["benchmarks"][bname]
        ms = b["methods"]
        if list(ms) != METHODS:
            fails.append(f"{bname}: methods {list(ms)} != {METHODS}")
            continue
        bf = ms["bf16"]
        kv_key = "logical_kv_mb_avg" if "logical_kv_mb_avg" in bf else "logical_kv_mb"
        metric_key = next(k for k in ("ppl", "accuracy_pct", "f1_pct") if k in bf)
        for m, e in ms.items():
            src = e["source"]
            lines = (ROOT / src["log"]).read_text(encoding="utf-8").splitlines()
            actual = lines[src["line"] - 1].rstrip()
            counts["source_rows"] += 1
            if actual != src["row"]:
                fails.append(f"{bname}/{m}: source row differs from log line {src['line']}")
            if actual.split()[0] != m:
                fails.append(f"{bname}/{m}: source row belongs to another method")
            tokens = actual.split()
            for k, v in e.items():
                if k in RAW_KEYS:
                    counts["raw_values"] += 1
                    if not in_row(v, tokens):
                        fails.append(f"{bname}/{m}.{k}={v} not found in source row")

            exp_ratio = round(bf[kv_key] / e[kv_key], 3)
            counts["derived_values"] += 1
            if e["logical_compression_vs_bf16"] != exp_ratio:
                fails.append(f"{bname}/{m}: ratio {e['logical_compression_vs_bf16']} != {exp_ratio}")
            if "logged_compression" in e and abs(e["logged_compression"] - exp_ratio) > 0.0015:
                fails.append(f"{bname}/{m}: logged compression {e['logged_compression']} vs derived {exp_ratio}")

            for k, v in e.items():
                if not k.startswith("delta_vs_bf16") or k == "delta_vs_bf16_ppl_pct_logged":
                    continue
                counts["derived_values"] += 1
                if k == "delta_vs_bf16_ppl_pct_computed":
                    exp = round((e["ppl"] / bf["ppl"] - 1) * 100, 2)
                    if abs(exp - e["delta_vs_bf16_ppl_pct_logged"]) > 0.005 + 1e-9:
                        fails.append(f"{bname}/{m}: computed PPL delta disagrees with logged d%")
                else:
                    exp = round(e[metric_key] - bf[metric_key], 1)
                if v != exp:
                    fails.append(f"{bname}/{m}.{k}={v} != {exp}")

    # NIAH 75/75 cells, re-derived from the log.
    niah_lines = (OUT_DIR / "niah.log").read_text(encoding="utf-8").splitlines()
    cells = [NIAH_CELL.match(l).groups() for l in niah_lines if NIAH_CELL.match(l)]
    distinct = {(c, d, m) for c, d, m, _ in cells}
    all_pass = all(r == "PASS" for *_, r in cells)
    cc = s["benchmarks"]["niah"]["cell_check"]
    if not (len(cells) == 75 and len(distinct) == 75 and all_pass):
        fails.append(f"NIAH log: {len(cells)} cells, {len(distinct)} distinct, all PASS={all_pass}")
    if (cc["cells_found"], cc["cells_pass"], cc["cells_fail"]) != (75, 75, 0):
        fails.append(f"NIAH cell_check {cc} != 75/75/0")

    pr = s["benchmarks"]["passage_retrieval"]["methods"]
    if not all(pr[m]["accuracy_pct"] == 100.0 for m in METHODS):
        fails.append("Passage Retrieval is not 100% for all methods")

    # HotpotQA: canonical vs rerun kept separate.
    # The canonical source's SHA is verified BEFORE any canonical value is
    # trusted; on mismatch the canonical values are not compared at all.
    hp = s["benchmarks"]["hotpotqa"]
    cref = hp["canonical_reference"]
    canonical_sha_ok = (
        cref.get("source") == "results/summary.json"
        and cref.get("source_sha256") == sha256(CANONICAL_SUMMARY)
    )
    counts["canonical_sha"] = int(canonical_sha_ok)
    if not canonical_sha_ok:
        fails.append(
            "hotpotqa.canonical_reference: source/source_sha256 does not match "
            "results/summary.json on disk; canonical values NOT trusted"
        )
    else:
        q = json.loads(CANONICAL_SUMMARY.read_text(encoding="utf-8"))["final_quality"]["longbench_hotpotqa_e"]
        if (cref["bf16_f1_pct"], cref["rabit2_f1_pct"]) != (q["bf16_f1_percent"], q["rabit_f1_percent"]):
            fails.append("hotpotqa.canonical_reference does not match results/summary.json")
        if cref["delta_rabit2_vs_bf16_f1_points"] != round(q["rabit_f1_percent"] - q["bf16_f1_percent"], 1):
            fails.append("hotpotqa canonical delta does not recompute")
    if "NOT part of this rerun" not in cref.get("note", ""):
        fails.append("hotpotqa.canonical_reference is not labelled NOT part of this rerun")

    # Numbers quoted in notes.
    hm = hp["methods"]
    note_checks = {
        "rabit8 PPL -0.03%": s["benchmarks"]["continuation_ppl"]["methods"]["rabit8"]["delta_vs_bf16_ppl_pct_logged"] == -0.03,
        "HotpotQA rabit3 64.0": hm["rabit3"]["f1_pct"] == 64.0,
        "HotpotQA rerun bf16 60.7": hm["bf16"]["f1_pct"] == 60.7,
        "HotpotQA rerun rabit2 55.2": hm["rabit2"]["f1_pct"] == 55.2,
        "HotpotQA rerun delta -5.5": hm["rabit2"]["delta_vs_bf16_f1_points"] == -5.5,
        "HotpotQA canonical 60.6": cref["bf16_f1_pct"] == 60.6,
        "HotpotQA canonical 55.2": cref["rabit2_f1_pct"] == 55.2,
        "HotpotQA canonical delta -5.4": cref["delta_rabit2_vs_bf16_f1_points"] == -5.4,
    }
    hp_note = next(n for n in s["notes"] if n.startswith("HotpotQA"))
    note_checks["HotpotQA note states both deltas"] = (
        "60.6 -> 55.2 = -5.4" in hp_note and "60.7 -> 55.2 = -5.5" in hp_note
    )
    for label, ok in note_checks.items():
        counts["note_numbers"] += 1
        if not ok:
            fails.append(f"note check failed: {label}")

    print(f"raw log SHA-256 verified:       {counts['log_hashes']}")
    print(f"canonical source SHA-256 ok:    {bool(counts['canonical_sha'])}")
    print(f"source rows verified verbatim:  {counts['source_rows']}")
    print(f"raw values traced to log rows:  {counts['raw_values']}")
    print(f"derived values recomputed:      {counts['derived_values']}")
    print(f"NIAH cells: {len(cells)} rows, {len(distinct)} distinct, all PASS={all_pass}")
    print(f"note/canonical checks:          {counts['note_numbers']}")
    print(f"FAILURES: {len(fails)}")
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
