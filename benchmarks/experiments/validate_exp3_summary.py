"""
Validate results/mlsys2027/deployment/summary.json (Experiment 3, attempt 2)
independently against the raw result files.

Checks:
  * SHA-256 (LF-normalized) of all 8 top-level result files matches the
    summary (raw-byte hashes are reported too);
  * 4 ABBA legs (A1/B1/B2/A2) with the right dtype and index, 15 measured
    samples each (reps 0..14, 2048 prompt / 32 output tokens), 30 per dtype;
  * capacities re-derived from blocks x 32 (393,024 and 2,074,592), equal to the
    engine log, identical across duplicate legs; ratio 5.2785x;
  * every per-leg and pooled statistic, signed delta and A1/A2, B1/B2 drift
    recomputed from the raw logs;
  * the summary's samples equal the raw log samples, and the pooled/per-leg
    figures agree with the runner's own matched_capacity_latency_summary.json;
  * integrity 94 passed / 0 failed / 0 not_run / 0 not_evaluated; config diff
    passed; protected_paths_post_run_status clean; archive unchanged (manifest
    flag AND failed_attempt_1 files vs failure_summary.json);
  * every number quoted in the notes.

Read-only. Runs locally; no Modal/GPU. Exits non-zero on any failure.

Usage:
    python benchmarks/mlsys2027/validate_exp3_summary.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results" / "mlsys2027" / "deployment"
RESULT_FILES = [
    "modal_session.log", "correctness_gate.log", "bf16_deployment.log", "rabit_kv2_deployment.log",
    "manifest.json", "matched_config_diff.json", "integrity_check.json", "matched_capacity_latency_summary.json",
]
LEGS = [("A1", 1, "bfloat16"), ("B1", 2, "rabit_kv2"), ("B2", 3, "rabit_kv2"), ("A2", 4, "bfloat16")]
LOG_OF = {"bfloat16": "bf16_deployment.log", "rabit_kv2": "rabit_kv2_deployment.log"}
EXPECTED_CAPACITY = {"bfloat16": 393024, "rabit_kv2": 2074592}
EXPECTED_RATIO_4DP = 5.2785
EXPECTED_COUNTS = {"passed": 94, "failed": 0, "not_run": 0, "not_evaluated": 0}
REPS, PROMPT, OUTPUT = 15, 2048, 32
EPS = 1e-9


def lf_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def raw_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def p90(v: list[float]) -> float:
    return statistics.quantiles(v, n=10, method="inclusive")[8]


def raw_legs(path: Path) -> dict:
    """Independent parser for a per-dtype log."""
    legs, cur = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^===== EXP3 LEG (\w+) \(index (\d+), (\w+)\) =====$", line)
        if m:
            cur = m.group(1)
            legs[cur] = {"index": int(m.group(2)), "dtype": m.group(3), "samples": [], "cap": None,
                         "kv": None, "log_tokens": None}
            continue
        if cur is None:
            continue
        if line.startswith("EXP3_SAMPLE "):
            legs[cur]["samples"].append(json.loads(line[len("EXP3_SAMPLE "):]))
        elif line.startswith("EXP3_CAPACITY="):
            legs[cur]["cap"] = json.loads(line.split("=", 1)[1])
        elif line.startswith("EXP3_KV_DTYPE="):
            legs[cur]["kv"] = json.loads(line.split("=", 1)[1])
        else:
            m = re.search(r"GPU KV cache size: ([\d,]+) tokens", line)
            if m:
                legs[cur]["log_tokens"] = int(m.group(1).replace(",", ""))
    return legs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    out = ap.parse_args(argv).out_dir
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    fails: list[str] = []
    n = {"hashes": 0, "samples": 0, "stats": 0, "deltas": 0, "drift": 0, "notes": 0, "cross": 0, "status": 0}

    def need(cond: bool, msg: str, key: str = "status") -> None:
        n[key] += 1
        if not cond:
            fails.append(msg)

    # 1. Raw file hashes.
    raw_match = 0
    for f in RESULT_FILES:
        rec = s["result_files_sha256"].get(f, {})
        need(rec.get("lf_normalized") == lf_sha(out / f), f"{f}: LF-normalized SHA-256 differs", "hashes")
        raw_match += rec.get("raw") == raw_sha(out / f)

    # 2. Legs, samples and capacity from the raw logs.
    raw = {d: raw_legs(out / LOG_OF[d]) for d in LOG_OF}
    legs = {}
    for label, idx, d in LEGS:
        L = raw[d].get(label)
        need(L is not None and L["index"] == idx and L["dtype"] == d, f"{label}: missing or wrong index/dtype")
        if L is None:
            continue
        legs[label] = L
        smp = L["samples"]
        need(len(smp) == REPS and [r["rep"] for r in smp] == list(range(REPS)), f"{label}: not 15 reps 0..14", "samples")
        need(all(r["prompt_tokens"] == PROMPT and r["output_tokens"] == OUTPUT for r in smp),
             f"{label}: prompt/output token counts wrong", "samples")
        cap = L["cap"]
        need(cap["block_size"] == 32 and cap["num_gpu_blocks"] * 32 == cap["capacity_tokens"] == EXPECTED_CAPACITY[d],
             f"{label}: capacity {cap} is not {EXPECTED_CAPACITY[d]} = blocks x 32")
        need(L["log_tokens"] == cap["capacity_tokens"], f"{label}: engine log capacity differs")
        sl = s["per_leg"][label]
        for m in ("tpot_ms", "ttft_ms", "wall_ms"):
            need(sl[f"{m}_samples"] == [r[m] for r in smp], f"{label}: summary {m} samples differ from raw log", "samples")
            v = [r[m] for r in smp]
            for stat, val in (("median", statistics.median(v)), ("p90", p90(v)), ("mean", statistics.mean(v)),
                              ("min", min(v)), ("max", max(v)), ("n", len(v))):
                need(abs(sl[m][stat] - val) <= EPS, f"{label}.{m}.{stat} {sl[m][stat]} != {val}", "stats")
        need(sl["capacity"]["capacity_tokens"] == cap["capacity_tokens"], f"{label}: summary capacity differs")

    pooled_raw = {}
    for d in LOG_OF:
        labels = [label for label, _, dd in LEGS if dd == d]
        caps = [json.dumps(legs[label]["cap"], sort_keys=True) for label in labels if label in legs]
        need(len(caps) == 2 and len(set(caps)) == 1, f"{d}: duplicate capacities not identical")
        rows = [r for label in labels if label in legs for r in legs[label]["samples"]]
        need(len(rows) == 30, f"{d}: pooled samples {len(rows)} != 30", "samples")
        pooled_raw[d] = {m: [r[m] for r in rows] for m in ("tpot_ms", "ttft_ms", "wall_ms")}
        sp = s["pooled_per_dtype"][d]
        for m, v in pooled_raw[d].items():
            for stat, val in (("median", statistics.median(v)), ("p90", p90(v)), ("mean", statistics.mean(v)),
                              ("n", len(v))):
                need(abs(sp[m][stat] - val) <= EPS, f"pooled {d}.{m}.{stat} mismatch", "stats")

    bf, rk = EXPECTED_CAPACITY["bfloat16"], EXPECTED_CAPACITY["rabit_kv2"]
    need(s["capacity"]["bf16_capacity_tokens"] == bf and s["capacity"]["rabit_kv2_capacity_tokens"] == rk,
         "summary capacities are not 393024 / 2074592")
    need(round(s["capacity"]["ratio_rabit_over_bf16"], 4) == EXPECTED_RATIO_4DP
         and abs(s["capacity"]["ratio_rabit_over_bf16"] - rk / bf) <= EPS, "capacity ratio is not 5.2785x")

    # 3. Signed deltas.
    for key, m, fn in (("tpot_median", "tpot_ms", statistics.median), ("tpot_p90", "tpot_ms", p90),
                       ("ttft_median", "ttft_ms", statistics.median), ("wall_median", "wall_ms", statistics.median)):
        x, y = fn(pooled_raw["bfloat16"][m]), fn(pooled_raw["rabit_kv2"][m])
        dl = s["signed_deltas_rabit_minus_bf16_pooled"][key]
        need(abs(dl["signed_delta_ms"] - (y - x)) <= EPS and abs(dl["signed_delta_pct"] - (y / x - 1) * 100) <= EPS,
             f"delta {key} does not recompute", "deltas")
        need(y - x > 0, f"delta {key} is not positive (RABIT-KV not slower)", "deltas")

    # 4. Drift.
    for key, a, b in (("bf16_A1_vs_A2", "A1", "A2"), ("rabit_kv2_B1_vs_B2", "B1", "B2")):
        for m in ("tpot_ms", "ttft_ms", "wall_ms"):
            va, vb = [r[m] for r in legs[a]["samples"]], [r[m] for r in legs[b]["samples"]]
            ma, mb = statistics.median(va), statistics.median(vb)
            dr = s["drift"][key][m]
            need(abs(dr["median_diff_ms"] - (mb - ma)) <= EPS and abs(dr["median_diff_pct"] - (mb / ma - 1) * 100) <= EPS,
                 f"drift {key}.{m} does not recompute", "drift")
            need(dr["ranges_overlap"] == (not (max(va) < min(vb) or max(vb) < min(va))), f"drift {key}.{m} overlap flag", "drift")
    a_drift = s["drift"]["bf16_A1_vs_A2"]["tpot_ms"]["median_diff_pct"]
    need(f"{a_drift:+.2f}" == "+8.64", f"BF16 A1/A2 TPOT drift is {a_drift:+.2f}%, not +8.64%", "drift")
    need(s["drift"]["bf16_A1_vs_A2"]["tpot_ms"]["ranges_overlap"] is False, "A1/A2 TPOT ranges should not overlap", "drift")

    # 5. Status files.
    integ = json.loads((out / "integrity_check.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    cdiff = json.loads((out / "matched_config_diff.json").read_text(encoding="utf-8"))
    need(integ["counts"] == EXPECTED_COUNTS and s["integrity"]["counts"] == EXPECTED_COUNTS,
         f"integrity counts {integ['counts']} != {EXPECTED_COUNTS}")
    need(len(integ["checks"]) == 94 and all(c["state"] == "passed" for c in integ["checks"]),
         "integrity_check.json does not contain exactly 94 passed checks")
    need(cdiff["status"] == "passed" and cdiff["matched"] is True and not cdiff["violations"]
         and s["matched_config"]["status"] == "passed", "matched_config_diff status is not passed")
    need(sorted(cdiff["fields_differing_between_dtypes"]) == sorted(cdiff["dtype_induced_allowlist"]),
         "differing fields are not exactly the dtype-induced allowlist")
    need(manifest["status"] == "passed", "manifest status is not passed")
    need(manifest["protected_paths_post_run_status"] == "clean", "protected_paths_post_run_status is not clean")
    need(manifest["archived_attempts_unchanged"] is True, "manifest archived_attempts_unchanged is not true")
    arch = out / "failed_attempt_1"
    fs = json.loads((arch / "failure_summary.json").read_text(encoding="utf-8"))
    need(all(raw_sha(arch / f) == h for f, h in fs["files_sha256"].items()),
         "failed_attempt_1 files differ from failure_summary.json")
    need(fs["accepted_scientific_evidence"] is False, "failed_attempt_1 claims accepted evidence")
    need(s["correctness_gate"]["pytest_passed"] == 105 and s["correctness_gate"]["pytest_exit"] == 0
         and s["correctness_gate"]["result"] == {"passed": True}, "correctness gate result mismatch")
    need(not s["processes"]["watchdog_timeouts"] and all(
        not p["timed_out"] and p["returncode"] == 0 and not p["group_processes_remaining"]
        for k, p in s["processes"].items() if k != "watchdog_timeouts"), "process/watchdog record not clean")
    need(all(p["clean"] and not p["compute_apps"] for p in s["gpu_clean_state"]["pre_leg"].values())
         and sorted(s["gpu_clean_state"]["pre_leg"]) == sorted(label for label, _, _ in LEGS),
         "GPU clean state not clean for all four legs")

    # 6. Cross-check with the runner's own summary (independent source).
    ms = json.loads((out / "matched_capacity_latency_summary.json").read_text(encoding="utf-8"))
    for d in LOG_OF:
        h = ms["pooled_per_dtype"][d]["headline"]
        sp = s["pooled_per_dtype"][d]
        for key, val in (("tpot_ms_median", sp["tpot_ms"]["median"]), ("tpot_ms_p90", sp["tpot_ms"]["p90"]),
                         ("ttft_ms_median", sp["ttft_ms"]["median"]), ("wall_ms_median", sp["wall_ms"]["median"])):
            need(abs(h[key] - val) <= EPS, f"{d} {key} differs from runner summary", "cross")
        need(ms["pooled_per_dtype"][d]["capacity_tokens"] == sp["capacity_tokens"], f"{d} capacity differs from runner", "cross")
    for label, _, _ in LEGS:
        need(abs(ms["per_leg"][label]["tpot_ms"]["median"] - s["per_leg"][label]["tpot_ms"]["median"]) <= EPS,
             f"{label} TPOT median differs from runner summary", "cross")

    # 7. Every number quoted in the notes.
    notes = " ".join(s["notes"])
    quoted = [f"{bf:,}", f"{rk:,}", "12,282 x 32", "64,831 x 32", f"{rk / bf:.4f}x", "+8.64%",
              f"{statistics.median([r['tpot_ms'] for r in legs['A1']['samples']]):.3f} -> "
              f"{statistics.median([r['tpot_ms'] for r in legs['A2']['samples']]):.3f} ms"]
    for key, v in s["signed_deltas_rabit_minus_bf16_pooled"].items():
        quoted.append(f"{key.replace('_', ' ')} {v['signed_delta_ms']:+.3f} ms ({v['signed_delta_pct']:+.2f}%)")
    for q in quoted:
        need(q in notes, f"note number not found / not matching data: {q!r}", "notes")
    for phrase in ("PHYSICAL real-engine vLLM allocator capacity", "PHYSICAL real-engine single-request latency",
                   "NOT logical fake-quant", "RABIT-KV is slower than BF16 in all reported pooled latency metrics",
                   "pending independent replication", "historical reference only"):
        need(phrase in notes, f"required note phrase missing: {phrase!r}", "notes")

    print(f"result-file SHA-256 (LF-normalized) verified: {n['hashes']} ({raw_match}/{len(RESULT_FILES)} raw-byte identical)")
    print(f"ABBA legs parsed from raw logs: {len(legs)}; sample checks: {n['samples']}")
    print(f"statistics recomputed: {n['stats']}; deltas: {n['deltas']}; drift: {n['drift']}")
    print(f"cross-checks vs runner summary: {n['cross']}; status checks: {n['status']}; note checks: {n['notes']}")
    print(f"capacity: bf16 {bf:,} / rabit_kv2 {rk:,} / ratio {rk / bf:.4f}x; BF16 A1/A2 TPOT drift {a_drift:+.2f}%")
    print(f"FAILURES: {len(fails)}")
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
