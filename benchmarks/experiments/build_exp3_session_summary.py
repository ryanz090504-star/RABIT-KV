"""
Generic session-level summary builder for MLSys 2027 Experiment 3 (matched
BF16 vs RABIT-KV physical capacity and latency, ABBA).

Works for any Experiment 3 result directory produced by
run_experiment3_deployment.py (e.g. results/mlsys2027/deployment/ for run #1,
results/mlsys2027/deployment/replication_1/ for the independent replication).
Nothing run-specific is hard-coded: every number, every overlap/non-overlap
classification and every sentence of the notes is derived from the raw files.

Inputs (read-only) in --results-dir:
  bf16_deployment.log, rabit_kv2_deployment.log  -- samples, capacity, KV dtype, workload
  manifest.json                                    -- provenance, gate, GPU state, processes, protocol
  integrity_check.json                             -- integrity states, GPU model
  matched_config_diff.json                         -- matched-config result
  matched_capacity_latency_summary.json            -- environment only (GPU UUID, driver, packages)

The notes are accompanied by `note_claims`: one machine-readable entry per
number/overlap word appearing in the notes, so a validator can recompute each
claim from the raw logs.

Usage:
    python benchmarks/mlsys2027/build_exp3_session_summary.py \
        --results-dir results/mlsys2027/deployment/replication_1 \
        --run-label independent_replication_1 \
        --output results/mlsys2027/deployment/replication_1/summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

SCHEMA = "exp3_session_summary/v1"
RESULT_FILES = [
    "modal_session.log", "correctness_gate.log", "bf16_deployment.log", "rabit_kv2_deployment.log",
    "manifest.json", "matched_config_diff.json", "integrity_check.json", "matched_capacity_latency_summary.json",
]
INPUT_FILES = ["bf16_deployment.log", "rabit_kv2_deployment.log", "manifest.json", "integrity_check.json",
               "matched_config_diff.json", "matched_capacity_latency_summary.json"]
LOG_OF = {"bfloat16": "bf16_deployment.log", "rabit_kv2": "rabit_kv2_deployment.log"}
DTYPES = ["bfloat16", "rabit_kv2"]
METRICS = ["tpot_ms", "ttft_ms", "wall_ms"]
METRIC_NAME = {"tpot_ms": "TPOT", "ttft_ms": "TTFT", "wall_ms": "wall"}
DELTA_KEYS = [("tpot_median", "tpot_ms", "median"), ("tpot_p90", "tpot_ms", "p90"),
              ("ttft_median", "ttft_ms", "median"), ("wall_median", "wall_ms", "median")]
P90_METHOD = "statistics.quantiles(values, n=10, method='inclusive')[8]"

LEG_HEADER = re.compile(r"^===== EXP3 LEG (\w+) \(index (\d+), (\w+)\) =====$")
TAG = re.compile(r"^(EXP3_[A-Z_]+)=(\{.*\})\s*$")
ROW = re.compile(r"^(EXP3_SAMPLE|EXP3_WARMUP) (\{.*\})\s*$")
KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
KV_MEM = re.compile(r"Available KV cache memory: ([\d.]+) GiB")


def sha256_raw(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha256_lf(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def parse_dtype_log(path: Path) -> dict:
    legs: dict = {}
    cur = None
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        m = LEG_HEADER.match(line)
        if m:
            cur = m.group(1)
            legs[cur] = {"index": int(m.group(2)), "dtype": m.group(3), "tags": {}, "samples": [], "warmups": [],
                         "kv_log_tokens": None, "kv_log_gib": None}
            continue
        if cur is None:
            continue
        leg = legs[cur]
        m = TAG.match(line)
        if m:
            leg["tags"][m.group(1)] = json.loads(m.group(2))
            continue
        m = ROW.match(line)
        if m:
            (leg["samples"] if m.group(1) == "EXP3_SAMPLE" else leg["warmups"]).append(
                dict(json.loads(m.group(2)), line=lineno))
            continue
        m = KV_TOKENS.search(line)
        if m:
            leg["kv_log_tokens"] = int(m.group(1).replace(",", ""))
        m = KV_MEM.search(line)
        if m:
            leg["kv_log_gib"] = float(m.group(1))
    return legs


def stats(values: list[float]) -> dict:
    return {"n": len(values), "median": statistics.median(values),
            "p90": statistics.quantiles(values, n=10, method="inclusive")[8],
            "mean": statistics.mean(values), "stdev": statistics.stdev(values),
            "min": min(values), "max": max(values)}


def overlap(a: list[float], b: list[float]) -> bool:
    return not (max(a) < min(b) or max(b) < min(a))


def build(results_dir: Path, run_label: str) -> dict:
    manifest = json.loads((results_dir / "manifest.json").read_text(encoding="utf-8"))
    integ = json.loads((results_dir / "integrity_check.json").read_text(encoding="utf-8"))
    cdiff = json.loads((results_dir / "matched_config_diff.json").read_text(encoding="utf-8"))
    runner = json.loads((results_dir / "matched_capacity_latency_summary.json").read_text(encoding="utf-8"))
    parsed = {d: parse_dtype_log(results_dir / LOG_OF[d]) for d in DTYPES}

    # ABBA legs as recorded in the logs, ordered by index.
    legs = sorted(((L["index"], label, L["dtype"]) for d in DTYPES for label, L in parsed[d].items()))
    per_leg = {}
    for idx, label, d in legs:
        L = parsed[d][label]
        cap = L["tags"]["EXP3_CAPACITY"]
        samples = L["samples"]
        per_leg[label] = {
            "index": idx, "kv_cache_dtype": d, "source_log": LOG_OF[d],
            "resolved_kv_dtype": L["tags"]["EXP3_KV_DTYPE"],
            "capacity": {"num_gpu_blocks": cap["num_gpu_blocks"], "block_size": cap["block_size"],
                         "capacity_tokens": cap["capacity_tokens"],
                         "engine_log_gpu_kv_cache_size_tokens": L["kv_log_tokens"],
                         "engine_log_available_kv_cache_memory_gib": L["kv_log_gib"]},
            "workload": L["tags"]["EXP3_WORKLOAD"],
            **{f"{m}_samples": [s[m] for s in samples] for m in METRICS},
            "sample_reps": [s["rep"] for s in samples],
            "sample_prompt_tokens": [s["prompt_tokens"] for s in samples],
            "sample_output_tokens": [s["output_tokens"] for s in samples],
            "sample_log_lines": [s["line"] for s in samples],
            "warmup_tpot_ms_excluded": [w["tpot_ms"] for w in L["warmups"]],
            **{m: stats([s[m] for s in samples]) for m in METRICS},
        }

    pooled = {}
    for d in DTYPES:
        labels = [label for _, label, dd in legs if dd == d]
        caps = [per_leg[label]["capacity"] for label in labels]
        c0 = caps[0]
        pooled[d] = {
            "legs": labels,
            "capacity_tokens": c0["capacity_tokens"], "num_gpu_blocks": c0["num_gpu_blocks"],
            "block_size": c0["block_size"],
            "duplicate_capacity_identical": len({json.dumps(c, sort_keys=True) for c in caps}) == 1,
            "physical_bytes_per_token_implied": c0["engine_log_available_kv_cache_memory_gib"] * 2**30
            / c0["capacity_tokens"],
            **{m: stats([v for label in labels for v in per_leg[label][f"{m}_samples"]]) for m in METRICS},
        }
    a, b = pooled["bfloat16"], pooled["rabit_kv2"]
    deltas = {}
    for key, m, stat in DELTA_KEYS:
        x, y = a[m][stat], b[m][stat]
        deltas[key] = {"bf16": x, "rabit_kv2": y, "signed_delta_ms": y - x, "signed_delta_pct": (y / x - 1) * 100}

    def drift_pair(first: str, second: str) -> dict:
        out = {"legs": [first, second]}
        for m in METRICS:
            va, vb = per_leg[first][f"{m}_samples"], per_leg[second][f"{m}_samples"]
            ma, mb = statistics.median(va), statistics.median(vb)
            out[m] = {f"{first}_median": ma, f"{second}_median": mb, "median_diff_ms": mb - ma,
                      "median_diff_pct": (mb / ma - 1) * 100, f"{first}_range": [min(va), max(va)],
                      f"{second}_range": [min(vb), max(vb)], "ranges_overlap": overlap(va, vb)}
        return out

    a_legs, b_legs = a["legs"], b["legs"]
    drift = {f"bf16_{a_legs[0]}_vs_{a_legs[1]}": drift_pair(*a_legs),
             f"rabit_kv2_{b_legs[0]}_vs_{b_legs[1]}": drift_pair(*b_legs)}
    ratio = b["capacity_tokens"] / a["capacity_tokens"]

    # ---- notes + machine-readable claims (every number / overlap word) ----
    claims: list[dict] = []

    def claim(quantity: str, fmt: str, value) -> str:
        text = fmt.format(value) if fmt != "overlap" else ("overlap" if value else "do not overlap")
        claims.append({"quantity": quantity, "format": fmt, "text": text})
        return text

    wl = per_leg[legs[0][1]]["workload"]
    notes = [
        "Capacity is PHYSICAL real-engine vLLM allocator capacity (num_gpu_blocks x block_size).",
        (f"Latency is PHYSICAL real-engine single-request latency (batch size one, "
         f"{claim('workload.context_tokens', '{:d}', wl['context_tokens'])}-token context, "
         f"{claim('workload.output_tokens', '{:d}', wl['output_tokens'])} output tokens, eager mode, Triton backend)."),
        "These are NOT logical fake-quant memory figures.",
        (f"Measured physical capacity: BF16 {claim('capacity.bf16.tokens', '{:,}', a['capacity_tokens'])} tokens "
         f"({claim('capacity.bf16.blocks', '{:,}', a['num_gpu_blocks'])} x "
         f"{claim('capacity.bf16.block_size', '{:d}', a['block_size'])}), RABIT-KV "
         f"{claim('capacity.rabit_kv2.tokens', '{:,}', b['capacity_tokens'])} tokens "
         f"({claim('capacity.rabit_kv2.blocks', '{:,}', b['num_gpu_blocks'])} x "
         f"{claim('capacity.rabit_kv2.block_size', '{:d}', b['block_size'])}), ratio "
         f"{claim('capacity.ratio', '{:.4f}', ratio)}x; duplicate capacities are "
         + ("identical in both legs of each dtype." if a["duplicate_capacity_identical"] and b["duplicate_capacity_identical"]
            else "NOT identical across legs.")),
    ]
    direction = []
    for key, m, stat in DELTA_KEYS:
        dl = deltas[key]
        word = "slower" if dl["signed_delta_ms"] > 0 else ("faster" if dl["signed_delta_ms"] < 0 else "equal")
        direction.append(word)
        notes.append(
            f"Pooled {METRIC_NAME[m]} {stat}: BF16 {claim(f'pooled.bfloat16.{m}.{stat}', '{:.3f}', dl['bf16'])} ms, "
            f"RABIT-KV {claim(f'pooled.rabit_kv2.{m}.{stat}', '{:.3f}', dl['rabit_kv2'])} ms; RABIT-KV is {word} "
            f"({claim(f'delta.{key}.ms', '{:+.3f}', dl['signed_delta_ms'])} ms, "
            f"{claim(f'delta.{key}.pct', '{:+.2f}', dl['signed_delta_pct'])}%).")
    if len(set(direction)) == 1:
        notes.append(f"RABIT-KV is {direction[0]} than BF16 in all reported pooled latency metrics of this session.")
    else:
        notes.append("The latency direction is NOT consistent across the reported pooled metrics of this session.")
    for (dname, dkey), dr in zip((("BF16", k) if k.startswith("bf16_") else ("RABIT-KV", k) for k in drift),
                                 drift.values()):
        first, second = dr["legs"]
        parts = []
        for m in METRICS:
            v = dr[m]
            parts.append(
                f"{METRIC_NAME[m]} median {claim(f'drift.{dkey}.{m}.median_diff_pct', '{:+.2f}', v['median_diff_pct'])}% "
                f"({claim(f'drift.{dkey}.{m}.first_median', '{:.3f}', v[f'{first}_median'])} -> "
                f"{claim(f'drift.{dkey}.{m}.second_median', '{:.3f}', v[f'{second}_median'])} ms; sample ranges "
                f"{claim(f'drift.{dkey}.{m}.ranges_overlap', 'overlap', v['ranges_overlap'])})")
        notes.append(f"Order effect {dname} {first} -> {second}: " + "; ".join(parts) + ".")
    notes += [
        ("These latency values are a valid result of this matched session; any final paper headline is decided "
         "across independent sessions, not from this session alone."),
        ("The frozen canonical RABIT-KV latency (results/performance/latency.json) is historical reference only "
         "and is not used to derive any value here."),
    ]

    prov = manifest["provenance"]
    gate = manifest["correctness_gate"]
    gc = manifest["gpu_clean_state"]
    env = runner.get("environment") or {}
    gpus = env.get("gpus") or []
    return {
        "schema": SCHEMA,
        "run_label": run_label,
        "experiment": "MLSys 2027 Experiment 3 -- matched BF16 vs RABIT-KV physical capacity and latency (ABBA)",
        "status_of_run": manifest["status"],
        "results_dir": results_dir.as_posix(),
        "derivation": ("Derived from bf16_deployment.log, rabit_kv2_deployment.log, manifest.json, "
                       "integrity_check.json, matched_config_diff.json; matched_capacity_latency_summary.json is used "
                       "for the environment block only. p90 = " + P90_METHOD + ". Pooled = both legs of a dtype. "
                       "Signed deltas are rabit_kv2 - bf16 on pooled statistics. Range overlap is computed from "
                       "the raw sample min/max of the two legs."),
        "generated_by": "benchmarks/mlsys2027/build_exp3_session_summary.py",
        "validated_by": "benchmarks/mlsys2027/validate_exp3_session_summary.py",
        "input_files": INPUT_FILES,
        "result_files_sha256": {f: {"raw": sha256_raw(results_dir / f), "lf_normalized": sha256_lf(results_dir / f)}
                                for f in RESULT_FILES if (results_dir / f).exists()},
        "labels": {
            "capacity": "PHYSICAL real-engine vLLM allocator KV capacity (num_gpu_blocks x block_size)",
            "latency": "PHYSICAL real-engine single-request latency",
            "not_logical": "These are NOT logical fake-quant memory figures.",
        },
        "design": manifest["protocol"],
        "abba_legs": [{"index": i, "leg": label, "kv_cache_dtype": d} for i, label, d in legs],
        "capacity": {
            "bf16_capacity_tokens": a["capacity_tokens"], "rabit_kv2_capacity_tokens": b["capacity_tokens"],
            "ratio_rabit_over_bf16": ratio, "signed_delta_tokens": b["capacity_tokens"] - a["capacity_tokens"],
            "per_leg": {label: per_leg[label]["capacity"] for _, label, _ in legs},
            "duplicate_capacity_identical": {d: pooled[d]["duplicate_capacity_identical"] for d in DTYPES},
        },
        "per_leg": per_leg,
        "pooled_per_dtype": pooled,
        "signed_deltas_rabit_minus_bf16_pooled": deltas,
        "drift": drift,
        "matched_config": {k: cdiff.get(k) for k in ("status", "matched", "fields_compared",
                                                      "fields_differing_between_dtypes", "violations",
                                                      "dtype_induced_allowlist")},
        "correctness_gate": {k: gate.get(k) for k in ("pytest_passed", "pytest_warnings", "pytest_exit", "result",
                                                       "regression_passed_line")}
        | {"pytest_command": (gate.get("begin") or {}).get("pytest_command"),
           "rabit_kv2_sha256_lf": (gate.get("begin") or {}).get("rabit_kv2_sha256_lf")},
        "gpu_clean_state": {
            "baseline_memory_used_mib": gc["baseline"]["memory_used_mib"],
            "baseline_compute_apps": gc["baseline"]["compute_apps"],
            "tolerance_mib": gc["tolerance_mib"],
            "pre_leg": {label: {"clean": p["clean"], "readings": len(p["readings"]),
                                "wait_s": p["readings"][-1]["t"] - p["readings"][0]["t"],
                                "memory_used_mib": p["readings"][-1]["memory_used_mib"],
                                "compute_apps": p["readings"][-1]["compute_apps"]}
                        for label, p in gc["pre_leg"].items()},
        },
        "processes": {label: {k: p[k] for k in ("elapsed_s", "returncode", "timed_out", "signals_sent",
                                                 "group_processes_after_leader_exit", "group_processes_remaining")}
                      for label, p in manifest["processes"]["exits"].items()}
        | {"watchdog_timeouts": manifest["processes"]["watchdog_timeouts"]},
        "integrity": {"counts": integ["counts"], "all_ok": integ["all_ok"], "checks_total": len(integ["checks"])},
        "run_status": {
            "manifest_status": manifest["status"],
            "protected_paths_post_run_status": manifest.get("protected_paths_post_run_status"),
            "archived_attempts_unchanged": manifest.get("archived_attempts_unchanged"),
            "archived_attempts_hashed": len(prov.get("archived_attempts_sha256") or {}),
            "config_diff_status": manifest.get("config_diff_status"),
        },
        "provenance": {
            **{k: prov.get(k) for k in ("git_head", "git_branch", "vllm_kvquant_tree", "rabit_kv2_sha256",
                                        "runner_script_sha256", "modal_app_sha256", "worker_sha256",
                                        "correctness_gate_sha256", "watchdog_sha256",
                                        "canonical_benchmark_deployment_sha256")},
            "vllm_kvquant_snapshot_sha256": (manifest.get("vllm_kvquant_snapshot") or {}).get("sha256"),
            "started_utc": manifest["started_utc"], "completed_utc": manifest["completed_utc"],
            "gpu_model": [g.get("name") for g in gpus],
            "gpu_uuid": [g.get("uuid") for g in gpus],
            "gpu_driver": [g.get("driver_version") for g in gpus],
            "packages": env.get("packages"), "python": env.get("python"),
            "vllm_precompiled_wheel_commit": env.get("vllm_precompiled_wheel_commit"),
            "model_snapshot_dir": (runner.get("model") or {}).get("snapshot_dir"),
            "model_files_sha256": {k: v.get("sha256") for k, v in ((runner.get("model") or {}).get("files") or {}).items()},
        },
        "notes": notes,
        "note_claims": claims,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--run-label", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    summary = build(args.results_dir, args.run_label)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output.as_posix()} (run_label={args.run_label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
