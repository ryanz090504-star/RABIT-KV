"""
Build results/mlsys2027/deployment/summary.json for MLSys 2027 Experiment 3
(matched BF16 vs RABIT-KV physical capacity and latency, ABBA).

Values are derived ONLY from these Experiment 3 result files:
  bf16_deployment.log, rabit_kv2_deployment.log   (samples, capacity, KV dtype)
  manifest.json                                    (provenance, gate, GPU state, processes)
  integrity_check.json                             (integrity counts, GPU model)
  matched_config_diff.json                         (matched-config result)
The runner's own matched_capacity_latency_summary.json and the historical
canonical latency numbers are NOT used to derive anything. The SHA-256 of all
8 top-level result files is recorded (raw bytes and LF-normalized).

Read-only with respect to every result file; the only file written is
summary.json in the deployment directory. Runs locally; no Modal/GPU.

Usage:
    python benchmarks/mlsys2027/build_exp3_summary.py [--out-dir DIR]
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
INPUT_FILES = ["bf16_deployment.log", "rabit_kv2_deployment.log", "manifest.json",
               "integrity_check.json", "matched_config_diff.json"]
LEGS = [(1, "A1", "bfloat16"), (2, "B1", "rabit_kv2"), (3, "B2", "rabit_kv2"), (4, "A2", "bfloat16")]
LOG_OF = {"bfloat16": "bf16_deployment.log", "rabit_kv2": "rabit_kv2_deployment.log"}
DTYPES = ["bfloat16", "rabit_kv2"]
SUMMARY_NAME = "summary.json"
P90_METHOD = "statistics.quantiles(values, n=10, method='inclusive')[8]"

LEG_HEADER = re.compile(r"^===== EXP3 LEG (\w+) \(index (\d+), (\w+)\) =====$")
TAG = re.compile(r"^(EXP3_[A-Z_]+)=(\{.*\})\s*$")
ROW = re.compile(r"^(EXP3_SAMPLE|EXP3_WARMUP) (\{.*\})\s*$")
KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
KV_MEM = re.compile(r"Available KV cache memory: ([\d.]+) GiB")


def sha256_raw(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def parse_dtype_log(path: Path) -> dict:
    """Split a per-dtype log into its legs; each leg records tags, samples
    (with log line numbers), warmups and engine-log capacity lines."""
    legs: dict = {}
    cur = None
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        m = LEG_HEADER.match(line)
        if m:
            cur = m.group(1)
            legs[cur] = {"index": int(m.group(2)), "dtype": m.group(3), "header_line": lineno, "tags": {},
                         "samples": [], "warmups": [], "kv_log_tokens": None, "kv_log_gib": None}
            continue
        if cur is None:
            continue
        L = legs[cur]
        m = TAG.match(line)
        if m:
            L["tags"][m.group(1)] = json.loads(m.group(2))
            continue
        m = ROW.match(line)
        if m:
            row = dict(json.loads(m.group(2)), line=lineno)
            (L["samples"] if m.group(1) == "EXP3_SAMPLE" else L["warmups"]).append(row)
            continue
        m = KV_TOKENS.search(line)
        if m:
            L["kv_log_tokens"] = int(m.group(1).replace(",", ""))
        m = KV_MEM.search(line)
        if m:
            L["kv_log_gib"] = float(m.group(1))
    return legs


def stats(values: list[float]) -> dict:
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p90": statistics.quantiles(values, n=10, method="inclusive")[8],
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def drift(first: dict, second: dict, first_label: str, second_label: str) -> dict:
    out = {"legs": [first_label, second_label]}
    for metric in ("tpot_ms", "ttft_ms", "wall_ms"):
        a, b = first[metric], second[metric]
        out[metric] = {
            f"{first_label}_median": a["median"], f"{second_label}_median": b["median"],
            "median_diff_ms": b["median"] - a["median"],
            "median_diff_pct": (b["median"] / a["median"] - 1) * 100,
            f"{first_label}_range": [a["min"], a["max"]], f"{second_label}_range": [b["min"], b["max"]],
            "ranges_overlap": not (a["max"] < b["min"] or b["max"] < a["min"]),
        }
    return out


def build(out_dir: Path) -> dict:
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    integ = json.loads((out_dir / "integrity_check.json").read_text(encoding="utf-8"))
    cdiff = json.loads((out_dir / "matched_config_diff.json").read_text(encoding="utf-8"))
    parsed = {d: parse_dtype_log(out_dir / LOG_OF[d]) for d in DTYPES}

    per_leg = {}
    for k, label, d in LEGS:
        L = parsed[d][label]
        cap = L["tags"]["EXP3_CAPACITY"]
        samples = L["samples"]
        per_leg[label] = {
            "index": k,
            "kv_cache_dtype": d,
            "source_log": f"{out_dir.name}/{LOG_OF[d]}" if out_dir.name else LOG_OF[d],
            "resolved_kv_dtype": L["tags"]["EXP3_KV_DTYPE"],
            "capacity": {
                "num_gpu_blocks": cap["num_gpu_blocks"],
                "block_size": cap["block_size"],
                "capacity_tokens": cap["capacity_tokens"],
                "engine_log_gpu_kv_cache_size_tokens": L["kv_log_tokens"],
                "engine_log_available_kv_cache_memory_gib": L["kv_log_gib"],
            },
            "workload_prompt_token_ids_sha256": L["tags"]["EXP3_WORKLOAD"]["prompt_token_ids_sha256"],
            "tpot_ms_samples": [s["tpot_ms"] for s in samples],
            "ttft_ms_samples": [s["ttft_ms"] for s in samples],
            "wall_ms_samples": [s["wall_ms"] for s in samples],
            "sample_log_lines": [s["line"] for s in samples],
            "warmup_tpot_ms_excluded": [w["tpot_ms"] for w in L["warmups"]],
            "tpot_ms": stats([s["tpot_ms"] for s in samples]),
            "ttft_ms": stats([s["ttft_ms"] for s in samples]),
            "wall_ms": stats([s["wall_ms"] for s in samples]),
        }

    pooled = {}
    for d in DTYPES:
        labels = [label for _, label, dd in LEGS if dd == d]
        caps = {label: per_leg[label]["capacity"] for label in labels}
        cap = caps[labels[0]]
        rows = {m: [v for label in labels for v in per_leg[label][f"{m}_samples"]]
                for m in ("tpot_ms", "ttft_ms", "wall_ms")}
        pooled[d] = {
            "legs": labels,
            "capacity_tokens": cap["capacity_tokens"],
            "num_gpu_blocks": cap["num_gpu_blocks"],
            "block_size": cap["block_size"],
            "duplicate_capacity_identical": len({json.dumps(c, sort_keys=True) for c in caps.values()}) == 1,
            "physical_bytes_per_token_implied": cap["engine_log_available_kv_cache_memory_gib"] * 2**30
            / cap["capacity_tokens"],
            "tpot_ms": stats(rows["tpot_ms"]),
            "ttft_ms": stats(rows["ttft_ms"]),
            "wall_ms": stats(rows["wall_ms"]),
        }

    a, b = pooled["bfloat16"], pooled["rabit_kv2"]

    def delta(metric: str, stat: str) -> dict:
        x, y = a[metric][stat], b[metric][stat]
        return {"bf16": x, "rabit_kv2": y, "signed_delta_ms": y - x, "signed_delta_pct": (y / x - 1) * 100}

    deltas = {
        "tpot_median": delta("tpot_ms", "median"),
        "tpot_p90": delta("tpot_ms", "p90"),
        "ttft_median": delta("ttft_ms", "median"),
        "wall_median": delta("wall_ms", "median"),
    }
    ratio = b["capacity_tokens"] / a["capacity_tokens"]
    drifts = {
        "bf16_A1_vs_A2": drift(per_leg["A1"], per_leg["A2"], "A1", "A2"),
        "rabit_kv2_B1_vs_B2": drift(per_leg["B1"], per_leg["B2"], "B1", "B2"),
    }
    a_drift = drifts["bf16_A1_vs_A2"]["tpot_ms"]["median_diff_pct"]
    all_slower = all(v["signed_delta_ms"] > 0 for v in deltas.values())

    gpu_obs = next((c["observed"] for c in integ["checks"] if c["check"] == "exactly one GPU, H100"), None)
    prov = manifest["provenance"]
    gate = manifest["correctness_gate"]
    gc = manifest["gpu_clean_state"]

    return {
        "experiment": "MLSys 2027 Experiment 3 -- matched BF16 vs RABIT-KV physical capacity and latency (ABBA), attempt 2",
        "status_of_run": manifest["status"],
        "derivation": (
            "Derived ONLY from bf16_deployment.log, rabit_kv2_deployment.log, manifest.json, integrity_check.json "
            "and matched_config_diff.json. The runner's matched_capacity_latency_summary.json and the historical "
            "canonical latency are not used. p90 = " + P90_METHOD + ". Pooled = both legs of a dtype (30 samples). "
            "Signed deltas are rabit_kv2 - bf16 on pooled statistics."
        ),
        "generated_by": "benchmarks/mlsys2027/build_exp3_summary.py",
        "validated_by": "benchmarks/mlsys2027/validate_exp3_summary.py",
        "result_files_sha256": {
            f: {"raw": sha256_raw(out_dir / f), "lf_normalized": sha256_lf(out_dir / f)} for f in RESULT_FILES
        },
        "input_files": INPUT_FILES,
        "labels": {
            "capacity": "PHYSICAL real-engine vLLM allocator KV capacity (num_gpu_blocks x block_size)",
            "latency": "PHYSICAL real-engine single-request latency (batch 1, 2048-token prompt, 32 output tokens, eager mode)",
            "not_logical": "These are NOT logical fake-quant memory figures.",
        },
        "design": manifest["protocol"],
        "capacity": {
            "bf16_capacity_tokens": a["capacity_tokens"],
            "rabit_kv2_capacity_tokens": b["capacity_tokens"],
            "ratio_rabit_over_bf16": ratio,
            "signed_delta_tokens": b["capacity_tokens"] - a["capacity_tokens"],
            "per_leg": {label: per_leg[label]["capacity"] for _, label, _ in LEGS},
            "duplicate_capacity_identical": {d: pooled[d]["duplicate_capacity_identical"] for d in DTYPES},
        },
        "per_leg": per_leg,
        "pooled_per_dtype": pooled,
        "signed_deltas_rabit_minus_bf16_pooled": deltas,
        "drift": drifts,
        "matched_config": {k: cdiff[k] for k in ("status", "matched", "fields_compared",
                                                   "fields_differing_between_dtypes", "violations",
                                                   "dtype_induced_allowlist")},
        "correctness_gate": {k: gate[k] for k in ("pytest_passed", "pytest_warnings", "pytest_exit", "result",
                                                   "regression_passed_line")}
        | {"pytest_command": gate["begin"]["pytest_command"], "rabit_kv2_sha256_lf": gate["begin"]["rabit_kv2_sha256_lf"]},
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
        "integrity": {"counts": integ["counts"], "all_ok": integ["all_ok"]},
        "run_status": {
            "manifest_status": manifest["status"],
            "protected_paths_post_run_status": manifest["protected_paths_post_run_status"],
            "archived_attempts_unchanged": manifest["archived_attempts_unchanged"],
            "config_diff_status": manifest["config_diff_status"],
        },
        "provenance": {
            **{k: prov[k] for k in ("git_head", "git_branch", "vllm_kvquant_tree", "rabit_kv2_sha256",
                                    "runner_script_sha256", "modal_app_sha256", "worker_sha256",
                                    "correctness_gate_sha256", "watchdog_sha256",
                                    "canonical_benchmark_deployment_sha256")},
            "vllm_kvquant_snapshot_sha256": manifest["vllm_kvquant_snapshot"]["sha256"],
            "gpu_model": gpu_obs,
            "started_utc": manifest["started_utc"],
            "completed_utc": manifest["completed_utc"],
            "note": ("GPU UUID, driver, CUDA/torch/vLLM/Triton versions are recorded in modal_session.log "
                     "(EXP3_ENVIRONMENT); that file is hashed above but is not a derivation input."),
        },
        "notes": [
            "Capacity is PHYSICAL real-engine vLLM allocator capacity (num_gpu_blocks x block_size).",
            "Latency is PHYSICAL real-engine single-request latency (batch 1, 2048-token context, 32 output tokens, eager mode, Triton backend).",
            "These are NOT logical fake-quant memory figures.",
            (f"Measured physical capacity: BF16 {a['capacity_tokens']:,} tokens ({a['num_gpu_blocks']:,} x {a['block_size']}), "
             f"RABIT-KV {b['capacity_tokens']:,} tokens ({b['num_gpu_blocks']:,} x {b['block_size']}), "
             f"ratio {ratio:.4f}x; duplicate capacities identical in both ABBA legs of each dtype."),
            (f"BF16 A1/A2 TPOT drift is {a_drift:+.2f}% (median {per_leg['A1']['tpot_ms']['median']:.3f} -> "
             f"{per_leg['A2']['tpot_ms']['median']:.3f} ms, non-overlapping ranges) and must remain visible."),
            ("RABIT-KV is slower than BF16 in all reported pooled latency metrics: "
             + ", ".join(f"{k.replace('_', ' ')} {v['signed_delta_ms']:+.3f} ms ({v['signed_delta_pct']:+.2f}%)"
                         for k, v in deltas.items()) + "."
             if all_slower else "WARNING: not all pooled latency deltas are positive."),
            ("The exact latency percentages are a valid result of this matched run but remain pending independent "
             "replication before becoming the final paper headline."),
            ("The frozen canonical RABIT-KV latency (results/performance/latency.json) is historical reference only "
             "and is not mixed into, or used to derive, any value of this matched run."),
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    summary = build(args.out_dir)
    (args.out_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {(args.out_dir / SUMMARY_NAME).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
