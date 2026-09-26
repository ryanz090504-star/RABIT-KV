"""
Summary builder for MLSys 2027 Experiment 4 (matched BF16 vs native FP8-E4M3
vs RABIT-KV physical capacity and latency, mirrored A1 B1 C1 C2 B2 A2).

Nothing measured is hard-coded: every number, ratio, delta, drift value,
overlap classification, direction word and every number in the notes is
derived from the raw Experiment 4 evidence in --results-dir (read-only):

  bf16_deployment.log, fp8_e4m3_deployment.log, rabit_kv2_deployment.log
      samples, warmups, capacity, KV dtype, effective engine config, workload
  modal_session.log      environment (GPU/driver/packages), model snapshot, post-run GPU state
  correctness_gate.log   gate details (pytest counts, prep/attention/decode-append exactness)
  manifest.json          provenance, protocol, gate result, GPU clean state, processes, run status
  integrity_check.json   integrity states
  matched_config_diff.json  matched-config result and allowlist

matched_capacity_latency_summary.json (the runner's own summary) is NOT used as
a source; validate_exp4_summary.py cross-checks against it independently.

The notes are accompanied by `note_claims`: one machine-readable entry per
number/overlap phrase in the notes, so the validator can recompute each one.

Usage:
    python benchmarks/mlsys2027/build_exp4_summary.py \
        --results-dir results/mlsys2027/fp8_baseline \
        --output results/mlsys2027/fp8_baseline/summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

SCHEMA = "exp4_summary/v1"
RESULT_FILES = [
    "modal_session.log", "correctness_gate.log", "bf16_deployment.log", "fp8_e4m3_deployment.log",
    "rabit_kv2_deployment.log", "manifest.json", "matched_config_diff.json", "integrity_check.json",
    "matched_capacity_latency_summary.json",
]
INPUT_FILES = ["bf16_deployment.log", "fp8_e4m3_deployment.log", "rabit_kv2_deployment.log", "modal_session.log",
               "correctness_gate.log", "manifest.json", "integrity_check.json", "matched_config_diff.json"]
DTYPES = ["bfloat16", "fp8_e4m3", "rabit_kv2"]
LOG_OF = {"bfloat16": "bf16_deployment.log", "fp8_e4m3": "fp8_e4m3_deployment.log",
          "rabit_kv2": "rabit_kv2_deployment.log"}
SHORT = {"bfloat16": "bf16", "fp8_e4m3": "fp8", "rabit_kv2": "rabit_kv2"}
NAME = {"bfloat16": "BF16", "fp8_e4m3": "native FP8", "rabit_kv2": "RABIT-KV"}
PAIRS = [("fp8_e4m3", "bfloat16"), ("rabit_kv2", "bfloat16"), ("rabit_kv2", "fp8_e4m3")]
METRICS = ["tpot_ms", "ttft_ms", "wall_ms"]
METRIC_NAME = {"tpot_ms": "TPOT", "ttft_ms": "TTFT", "wall_ms": "wall"}
DELTA_KEYS = [("tpot_median", "tpot_ms", "median"), ("tpot_p90", "tpot_ms", "p90"),
              ("ttft_median", "ttft_ms", "median"), ("wall_median", "wall_ms", "median")]
P90_METHOD = "statistics.quantiles(values, n=10, method='inclusive')[8]"

LEG_HEADER = re.compile(r"^===== EXP4 LEG (\w+) \(index (\d+), (\w+)\) =====$")
TAG = re.compile(r"^(EXP4_[A-Z_]+)=(\{.*\})\s*$")
ROW = re.compile(r"^(EXP4_SAMPLE|EXP4_WARMUP) (\{.*\})\s*$")
KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
KV_MEM = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
JIT = "Triton kernel JIT compilation during inference"


def sha256_raw(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha256_lf(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def parse_dtype_log(path: Path) -> dict:
    legs: dict = {}
    cur, phase = None, None
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        m = LEG_HEADER.match(line)
        if m:
            cur, phase = m.group(1), None
            legs[cur] = {"index": int(m.group(2)), "dtype": m.group(3), "tags": {}, "samples": [], "warmups": [],
                         "kv_log_tokens": None, "kv_log_gib": None, "jit": {}}
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
            (leg["samples"] if m.group(1) == "EXP4_SAMPLE" else leg["warmups"]).append(
                dict(json.loads(m.group(2)), line=lineno))
            continue
        s = line.strip()
        if s == "EXP4_WARMUP_BEGIN":
            phase = "warmup"
        elif s == "EXP4_MEASUREMENT_BEGIN":
            phase = "measurement"
        elif s in ("EXP4_WARMUP_END", "EXP4_MEASUREMENT_END"):
            phase = "between" if s == "EXP4_WARMUP_END" else "after"
        if JIT in line:
            leg["jit"][phase or "startup"] = leg["jit"].get(phase or "startup", 0) + 1
        m = KV_TOKENS.search(line)
        if m:
            leg["kv_log_tokens"] = int(m.group(1).replace(",", ""))
        m = KV_MEM.search(line)
        if m:
            leg["kv_log_gib"] = float(m.group(1))
    return legs


def parse_session_top(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("["):
            continue
        m = TAG.match(line.strip())
        if m and m.group(1) in ("EXP4_ENVIRONMENT", "EXP4_MODEL", "EXP4_POST_RUN_GPU_STATE", "EXP4_GPU_BASELINE"):
            out[m.group(1)] = json.loads(m.group(2))
    return out


def parse_gate_log(path: Path) -> dict:
    t = path.read_text(encoding="utf-8")
    m_py = re.search(r"^(\d+) passed, (\d+) warnings? in ([\d.]+)s", t, re.M)
    m_prep = re.search(r"prep exactness: PASSED \((\d+)/(\d+)\)", t)
    m_dec = re.search(r"fast decode append exactness: PASSED \((\d+) state/byte steps", t)
    m_exit = re.search(r"^pytest exit=(-?\d+)$", t, re.M)
    return {
        "dispatch_preflight_passed": "Dispatch preflight: PASSED" in t,
        "prep_exactness_passed": m_prep.group(1) if m_prep else None,
        "prep_exactness_total": m_prep.group(2) if m_prep else None,
        "full_attention_exactness_passed": bool(re.search(r"full attention exactness: PASSED", t)),
        "decode_append_steps": int(m_dec.group(1)) if m_dec else None,
        "pytest_passed": int(m_py.group(1)) if m_py else None,
        "pytest_warnings": int(m_py.group(2)) if m_py else None,
        "pytest_exit": int(m_exit.group(1)) if m_exit else None,
        "regression_passed_line": "RABIT-2 FINAL TARGETED REGRESSION PASSED" in t,
    }


def stats(values: list[float]) -> dict:
    return {"n": len(values), "median": statistics.median(values),
            "p90": statistics.quantiles(values, n=10, method="inclusive")[8],
            "mean": statistics.mean(values), "stdev": statistics.stdev(values),
            "min": min(values), "max": max(values)}


def overlap(a: list[float], b: list[float]) -> bool:
    return not (max(a) < min(b) or max(b) < min(a))


def direction(delta: float) -> str:
    return "slower" if delta > 0 else ("faster" if delta < 0 else "equal")


def fp8_vs_bf16_capacity_wording(ratio: float) -> str:
    if 1.9 <= ratio < 2.0:
        return "nearly doubles"
    if 2.0 <= ratio < 2.1:
        return "roughly doubles"
    return "does not double"


def rabit_vs_fp8_capacity_wording(ratio: float) -> str:
    if ratio >= 1.5:
        return "substantially more"
    if ratio > 1.0:
        return "more"
    return "no more"


def fp8_semantics(kv: dict, eff: dict) -> dict:
    """Source/config-derived FP8 facts; NOT runtime tensor observations."""
    native = (kv.get("engine_cache_dtype") == "fp8_e4m3" and kv.get("kv_quant_mode") == "FP8_PER_TENSOR"
              and kv.get("fp8_storage_view_dtype") == "torch.float8_e4m3fn"
              and kv.get("resolved_kv_torch_dtype") == "torch.uint8")
    default_scale = (eff.get("hf_quantization_config") is None and eff.get("calculate_kv_scales") is False
                     and eff.get("quantization") is None)
    query_fp8 = native and eff.get("attention_backend") == "AttentionBackendEnum.TRITON_ATTN"
    return {
        "native_per_tensor_fp8_e4m3_path": native,
        "checkpoint_quantization_config": eff.get("hf_quantization_config"),
        "calculate_kv_scales": eff.get("calculate_kv_scales"),
        "model_quantization": eff.get("quantization"),
        "default_kv_scale_1_0_implied_by_source": default_scale,
        "native_query_fp8_conversion_implied_by_source": query_fp8,
        "evidence_type": ("source/config-derived: the audited vllm-kvquant source sets k/v/q scales to 1.0 when "
                          "the checkpoint has no quantization_config and calculate_kv_scales is False, and on CUDA "
                          "converts the query to FP8 for fp8/fp8_e4m3 KV caches. The scale tensors and the query "
                          "dtype were not observed at runtime (the worker makes no engine RPC)."),
    }


def build(results_dir: Path) -> dict:
    manifest = json.loads((results_dir / "manifest.json").read_text(encoding="utf-8"))
    integ = json.loads((results_dir / "integrity_check.json").read_text(encoding="utf-8"))
    cdiff = json.loads((results_dir / "matched_config_diff.json").read_text(encoding="utf-8"))
    top = parse_session_top(results_dir / "modal_session.log")
    gate_log = parse_gate_log(results_dir / "correctness_gate.log")
    parsed = {d: parse_dtype_log(results_dir / LOG_OF[d]) for d in DTYPES}

    legs = sorted((L["index"], label, L["dtype"]) for d in DTYPES for label, L in parsed[d].items())
    per_leg = {}
    for idx, label, d in legs:
        L = parsed[d][label]
        cap = L["tags"]["EXP4_CAPACITY"]
        smp = L["samples"]
        per_leg[label] = {
            "index": idx, "kv_cache_dtype": d, "source_log": LOG_OF[d],
            "kv_dtype": L["tags"]["EXP4_KV_DTYPE"],
            "capacity": {"num_gpu_blocks": cap["num_gpu_blocks"], "block_size": cap["block_size"],
                         "capacity_tokens": cap["capacity_tokens"],
                         "engine_log_gpu_kv_cache_size_tokens": L["kv_log_tokens"],
                         "engine_log_available_kv_cache_memory_gib": L["kv_log_gib"],
                         "implied_physical_bytes_per_token": L["kv_log_gib"] * 2**30 / cap["capacity_tokens"]},
            "workload": L["tags"]["EXP4_WORKLOAD"],
            **{f"{m}_samples": [r[m] for r in smp] for m in METRICS},
            "sample_reps": [r["rep"] for r in smp],
            "sample_prompt_tokens": [r["prompt_tokens"] for r in smp],
            "sample_output_tokens": [r["output_tokens"] for r in smp],
            "sample_log_lines": [r["line"] for r in smp],
            "warmups_excluded": len(L["warmups"]),
            "warmup_tpot_ms_excluded": [w["tpot_ms"] for w in L["warmups"]],
            "jit_lines_by_phase": L["jit"],
            **{m: stats([r[m] for r in smp]) for m in METRICS},
        }

    pooled = {}
    for d in DTYPES:
        labels = [label for _, label, dd in legs if dd == d]
        caps = [{k: per_leg[l]["capacity"][k] for k in ("num_gpu_blocks", "block_size", "capacity_tokens")}
                for l in labels]
        c0 = per_leg[labels[0]]["capacity"]
        pooled[d] = {
            "legs": labels,
            "capacity_tokens": c0["capacity_tokens"], "num_gpu_blocks": c0["num_gpu_blocks"],
            "block_size": c0["block_size"],
            "duplicate_capacity_identical": len({json.dumps(c, sort_keys=True) for c in caps}) == 1,
            "implied_physical_bytes_per_token": c0["implied_physical_bytes_per_token"],
            **{m: stats([v for l in labels for v in per_leg[l][f"{m}_samples"]]) for m in METRICS},
        }

    ratios = {}
    for x, y in PAIRS:
        cx, cy = pooled[x]["capacity_tokens"], pooled[y]["capacity_tokens"]
        ratios[f"{SHORT[x]}_over_{SHORT[y]}"] = {SHORT[x]: cx, SHORT[y]: cy, "ratio": cx / cy,
                                                "signed_delta_tokens": cx - cy}
    deltas = {}
    for x, y in PAIRS:
        key = f"{SHORT[x]}_minus_{SHORT[y]}"
        deltas[key] = {}
        for dk, m, st in DELTA_KEYS:
            vx, vy = pooled[x][m][st], pooled[y][m][st]
            deltas[key][dk] = {SHORT[x]: vx, SHORT[y]: vy, "signed_delta_ms": vx - vy,
                               "signed_delta_pct": (vx / vy - 1) * 100, "direction": direction(vx - vy)}

    order = {}
    for d in DTYPES:
        first, second = pooled[d]["legs"]
        entry = {"kv_cache_dtype": d, "first_leg": first, "second_leg": second}
        for m in METRICS:
            va, vb = per_leg[first][f"{m}_samples"], per_leg[second][f"{m}_samples"]
            ma, mb = statistics.median(va), statistics.median(vb)
            entry[m] = {"first_median": ma, "second_median": mb, "median_diff_ms": mb - ma,
                        "median_drift_pct": (mb / ma - 1) * 100, "first_range": [min(va), max(va)],
                        "second_range": [min(vb), max(vb)], "ranges_overlap": overlap(va, vb)}
        order[f"{first}_to_{second}"] = entry

    semantics = {}
    for d in DTYPES:
        labels = pooled[d]["legs"]
        kv = per_leg[labels[0]]["kv_dtype"]
        eff = parsed[d][labels[0]]["tags"]["EXP4_EFFECTIVE_ENGINE_CONFIG"]
        semantics[d] = {
            "requested_dtype": kv["requested_kv_cache_dtype"], "engine_cache_dtype": kv["engine_cache_dtype"],
            "resolved_storage_dtype": kv["resolved_kv_torch_dtype"], "kv_quant_mode": kv["kv_quant_mode"],
            "fp8_storage_view_dtype": kv["fp8_storage_view_dtype"],
            "identical_across_legs": len({json.dumps(per_leg[l]["kv_dtype"], sort_keys=True) for l in labels}) == 1,
            "calculate_kv_scales": eff["calculate_kv_scales"],
            "kv_cache_dtype_skip_layers": eff["kv_cache_dtype_skip_layers"],
            "checkpoint_quantization_config": eff["hf_quantization_config"],
        }
    fp8_eff = parsed["fp8_e4m3"][pooled["fp8_e4m3"]["legs"][0]]["tags"]["EXP4_EFFECTIVE_ENGINE_CONFIG"]
    semantics["fp8_e4m3"]["native_fp8_notes"] = fp8_semantics(per_leg[pooled["fp8_e4m3"]["legs"][0]]["kv_dtype"],
                                                             fp8_eff)

    # ---- notes + machine-readable claims (every number / overlap phrase) ----
    claims: list[dict] = []

    def claim(quantity: str, fmt: str, value) -> str:
        text = fmt.format(value) if fmt != "overlap" else ("overlap" if value else "do not overlap")
        claims.append({"quantity": quantity, "format": fmt, "text": text})
        return text

    first_leg = per_leg[legs[0][1]]
    wl = first_leg["workload"]
    eff0 = parsed[legs[0][2]][legs[0][1]]["tags"]["EXP4_EFFECTIVE_ENGINE_CONFIG"]
    reps = first_leg["tpot_ms"]["n"]
    notes = [
        ("Scope: PHYSICAL real-engine vLLM allocator capacity (num_gpu_blocks x block_size) and PHYSICAL "
         "real-engine single-request latency (batch size one), measured in one matched H100 session, eager mode, "
         f"Triton backend, {claim('workload.context_tokens', '{:d}', wl['context_tokens'])}-token prompt, "
         f"{claim('workload.output_tokens', '{:d}', wl['output_tokens'])} generated tokens, "
         f"{claim('workload.warmups', '{:d}', first_leg['warmups_excluded'])} excluded warmups and "
         f"{claim('workload.reps_per_leg', '{:d}', reps)} measured reps per leg "
         f"(n={claim('workload.reps_per_dtype', '{:d}', pooled['bfloat16']['tpot_ms']['n'])} pooled per dtype)."),
        ("This is NOT an FP8 quality comparison and makes no FP8 quality-equivalence claim."),
        ("No samples from the earlier BF16-vs-RABIT-KV experiment (Exp3) were reused: BF16, native FP8 and RABIT-KV "
         "were all measured fresh in this session."),
        ("Findings apply specifically to the native vLLM FP8-E4M3 KV-cache path measured here (per-tensor, default "
         "unit KV scales for this checkpoint, native query FP8 conversion in the Triton backend) and are not a "
         "statement about FP8 methods in general. The default scale and the query conversion are "
         "source/config-derived facts, not direct runtime tensor observations."),
    ]
    cap_parts = [f"{NAME[d]} {claim(f'capacity.{SHORT[d]}.tokens', '{:,}', pooled[d]['capacity_tokens'])} tokens "
                 f"({claim(f'capacity.{SHORT[d]}.blocks', '{:,}', pooled[d]['num_gpu_blocks'])} x "
                 f"{claim(f'capacity.{SHORT[d]}.block_size', '{:d}', pooled[d]['block_size'])})" for d in DTYPES]
    dup_all = all(pooled[d]["duplicate_capacity_identical"] for d in DTYPES)
    notes.append("Measured physical capacity: " + ", ".join(cap_parts) + "; duplicate capacities are "
                 + ("identical in both legs of every dtype." if dup_all else "NOT identical across legs."))
    r_fb, r_rb, r_rf = (ratios["fp8_over_bf16"]["ratio"], ratios["rabit_kv2_over_bf16"]["ratio"],
                        ratios["rabit_kv2_over_fp8"]["ratio"])
    notes.append(
        f"Native FP8 {fp8_vs_bf16_capacity_wording(r_fb)} physical capacity relative to BF16 "
        f"(FP8/BF16 {claim('ratio.fp8_over_bf16', '{:.4f}', r_fb)}x); RABIT-KV provides "
        f"{rabit_vs_fp8_capacity_wording(r_rf)} physical capacity than native FP8 "
        f"(RABIT/FP8 {claim('ratio.rabit_kv2_over_fp8', '{:.4f}', r_rf)}x; RABIT/BF16 "
        f"{claim('ratio.rabit_kv2_over_bf16', '{:.4f}', r_rb)}x).")
    speedups = []
    for x, y in PAIRS:
        key = f"{SHORT[x]}_minus_{SHORT[y]}"
        words = []
        for dk, m, st in DELTA_KEYS:
            dl = deltas[key][dk]
            words.append(dl["direction"])
            if dl["signed_delta_ms"] < 0:
                speedups.append(f"{NAME[x]} vs {NAME[y]} {METRIC_NAME[m]} {st}")
            notes.append(
                f"Pooled {METRIC_NAME[m]} {st}: {NAME[x]} "
                f"{claim(f'pooled.{SHORT[x]}.{m}.{st}', '{:.3f}', dl[SHORT[x]])} ms vs {NAME[y]} "
                f"{claim(f'pooled.{SHORT[y]}.{m}.{st}', '{:.3f}', dl[SHORT[y]])} ms; {NAME[x]} is {dl['direction']} "
                f"than {NAME[y]} ({claim(f'delta.{key}.{dk}.ms', '{:+.3f}', dl['signed_delta_ms'])} ms, "
                f"{claim(f'delta.{key}.{dk}.pct', '{:+.2f}', dl['signed_delta_pct'])}%).")
        if len(set(words)) == 1 and words[0] == "slower":
            notes.append(f"{NAME[x][0].upper() + NAME[x][1:]} is slower than {NAME[y]} in all four pooled latency metrics of this session "
                         f"(a latency overhead, not a speedup).")
        elif len(set(words)) == 1 and words[0] == "faster":
            notes.append(f"{NAME[x][0].upper() + NAME[x][1:]} is faster than {NAME[y]} in all four pooled latency metrics of this session.")
        else:
            notes.append(f"The latency direction of {NAME[x]} vs {NAME[y]} is mixed across the pooled metrics of "
                         f"this session.")
    if speedups:
        notes.append("Faster (negative) pooled deltas were measured only for: " + "; ".join(speedups) + ".")
    else:
        notes.append("No speedup is claimed: no pairwise pooled latency delta in this session is negative.")
    for okey, oe in order.items():
        parts = []
        for m in METRICS:
            v = oe[m]
            parts.append(
                f"{METRIC_NAME[m]} median {claim(f'order.{okey}.{m}.median_drift_pct', '{:+.2f}', v['median_drift_pct'])}% "
                f"({claim(f'order.{okey}.{m}.first_median', '{:.3f}', v['first_median'])} -> "
                f"{claim(f'order.{okey}.{m}.second_median', '{:.3f}', v['second_median'])} ms; sample ranges "
                f"[{claim(f'order.{okey}.{m}.first_min', '{:.3f}', v['first_range'][0])}, "
                f"{claim(f'order.{okey}.{m}.first_max', '{:.3f}', v['first_range'][1])}] vs "
                f"[{claim(f'order.{okey}.{m}.second_min', '{:.3f}', v['second_range'][0])}, "
                f"{claim(f'order.{okey}.{m}.second_max', '{:.3f}', v['second_range'][1])}] "
                f"{claim(f'order.{okey}.{m}.ranges_overlap', 'overlap', v['ranges_overlap'])})")
        notes.append(f"Order effect {NAME[oe['kv_cache_dtype']]} {oe['first_leg']} -> {oe['second_leg']}: "
                     + "; ".join(parts) + ".")
    bf_key = f"{pooled['bfloat16']['legs'][0]}_to_{pooled['bfloat16']['legs'][1]}"
    bf_tp = order[bf_key]["tpot_ms"]
    notes.append(
        f"The BF16 order effect remains visible and is not corrected for: BF16 TPOT median drift "
        f"{claim(f'order.{bf_key}.tpot_ms.median_drift_pct', '{:+.2f}', bf_tp['median_drift_pct'])}% between its "
        f"two legs, whose sample ranges {claim(f'order.{bf_key}.tpot_ms.ranges_overlap', 'overlap', bf_tp['ranges_overlap'])}.")
    g = manifest["correctness_gate"]
    notes.append(
        f"The frozen RABIT-KV correctness gate passed before any leg: pytest "
        f"{claim('gate.pytest_passed', '{:d}', gate_log['pytest_passed'])} passed / "
        f"{claim('gate.pytest_warnings', '{:d}', gate_log['pytest_warnings'])} warnings, prep exactness "
        f"{claim('gate.prep_passed', '{}', gate_log['prep_exactness_passed'])}/"
        f"{claim('gate.prep_total', '{}', gate_log['prep_exactness_total'])}, full-attention exactness passed, "
        f"fast decode-append exactness over {claim('gate.decode_append_steps', '{:d}', gate_log['decode_append_steps'])} "
        f"state/byte steps." if (g.get("result") or {}).get("passed") and gate_log["full_attention_exactness_passed"]
        and gate_log["dispatch_preflight_passed"] else "The correctness gate did NOT pass.")
    c = integ["counts"]
    notes.append(
        f"Integrity: {claim('integrity.passed', '{:d}', c['passed'])} passed, "
        f"{claim('integrity.failed', '{:d}', c['failed'])} failed, {claim('integrity.not_run', '{:d}', c['not_run'])} "
        f"not_run, {claim('integrity.not_evaluated', '{:d}', c['not_evaluated'])} not_evaluated; matched config "
        f"{cdiff['status']}; protected paths {manifest.get('protected_paths_post_run_status')}.")

    prov = manifest["provenance"]
    gc = manifest["gpu_clean_state"]
    env = top.get("EXP4_ENVIRONMENT") or {}
    gpus = env.get("gpus") or []
    model = top.get("EXP4_MODEL") or {}
    req_models = {parsed[d][l]["tags"]["EXP4_REQUESTED_ENGINE_KWARGS"]["model"] for _, l, d in legs}
    prompt_hashes = {l: per_leg[l]["workload"]["prompt_token_ids_sha256"] for _, l, _ in legs}
    return {
        "schema": SCHEMA,
        "experiment": ("MLSys 2027 Experiment 4 -- matched BF16 vs native FP8-E4M3 vs RABIT-KV physical capacity "
                       "and latency (mirrored A1 B1 C1 C2 B2 A2)"),
        "status_of_run": manifest["status"],
        "results_dir": results_dir.as_posix(),
        "derivation": ("Derived from bf16_deployment.log, fp8_e4m3_deployment.log, rabit_kv2_deployment.log, "
                       "modal_session.log (environment, model snapshot, post-run GPU state), correctness_gate.log, "
                       "manifest.json, integrity_check.json and matched_config_diff.json. The runner's "
                       "matched_capacity_latency_summary.json is not a source (validator cross-check only). p90 = "
                       + P90_METHOD + ". Pooled = both legs of one dtype only. Signed deltas are x - y on pooled "
                       "statistics. Range overlap uses the raw sample min/max of the two legs of a dtype."),
        "generated_by": "benchmarks/mlsys2027/build_exp4_summary.py",
        "validated_by": "benchmarks/mlsys2027/validate_exp4_summary.py",
        "input_files": INPUT_FILES,
        "result_files_sha256": {f: {"raw": sha256_raw(results_dir / f), "lf_normalized": sha256_lf(results_dir / f)}
                                for f in RESULT_FILES if (results_dir / f).exists()},
        "scope": {
            "capacity": "PHYSICAL real-engine vLLM allocator KV capacity (num_gpu_blocks x block_size)",
            "latency": "PHYSICAL real-engine single-request latency (batch size one; one request per generate call)",
            "session": "one matched H100 session (one Modal container, one GPU, fresh engine process per leg)",
            "execution": {"enforce_eager": eff0["enforce_eager"], "attention_backend": eff0["attention_backend"],
                          "compilation_mode": eff0["compilation_mode"], "cudagraph_mode": eff0["cudagraph_mode"]},
            "prompt_tokens": wl["context_tokens"], "generated_tokens": wl["output_tokens"],
            "fp8_quality_equivalence_claimed": False,
            "experiment3_samples_used": False,
            "not_logical": "These are NOT logical fake-quant memory figures.",
        },
        "design": manifest["protocol"],
        "legs": [{"index": i, "leg": label, "kv_cache_dtype": d} for i, label, d in legs],
        "dtype_semantics": semantics,
        "capacity": {
            "per_leg": {label: per_leg[label]["capacity"] for _, label, _ in legs},
            "per_dtype": {d: {k: pooled[d][k] for k in ("capacity_tokens", "num_gpu_blocks", "block_size",
                                                        "duplicate_capacity_identical",
                                                        "implied_physical_bytes_per_token")} for d in DTYPES},
            "ratios": ratios,
        },
        "per_leg": per_leg,
        "pooled_per_dtype": pooled,
        "pairwise_deltas_pooled": deltas,
        "order_effects": order,
        "matched_config": {k: cdiff.get(k) for k in ("status", "matched", "fields_compared",
                                                      "fields_differing_between_dtypes", "violations",
                                                      "dtype_induced_allowlist")},
        "correctness_gate": {"result": g.get("result"), "pytest_exit": g.get("pytest_exit"),
                             "pytest_command": (g.get("begin") or {}).get("pytest_command"),
                             "rabit_kv2_sha256_lf": (g.get("begin") or {}).get("rabit_kv2_sha256_lf"),
                             "from_gate_log": gate_log},
        "gpu_clean_state": {
            "baseline_memory_used_mib": gc["baseline"]["memory_used_mib"],
            "baseline_compute_apps": gc["baseline"]["compute_apps"],
            "tolerance_mib": gc["tolerance_mib"],
            "pre_leg": {label: {"clean": p["clean"], "readings": len(p["readings"]),
                                "wait_s": p["readings"][-1]["t"] - p["readings"][0]["t"],
                                "memory_used_mib": p["readings"][-1]["memory_used_mib"],
                                "compute_apps": p["readings"][-1]["compute_apps"]}
                        for label, p in gc["pre_leg"].items()},
            "post_run": top.get("EXP4_POST_RUN_GPU_STATE"),
        },
        "processes": {label: {k: p[k] for k in ("elapsed_s", "returncode", "timed_out", "timeout_s", "signals_sent",
                                                 "group_processes_after_leader_exit", "group_processes_remaining")}
                      for label, p in manifest["processes"]["exits"].items()}
        | {"watchdog_timeouts": manifest["processes"]["watchdog_timeouts"]},
        "integrity": {"counts": integ["counts"], "all_ok": integ["all_ok"], "checks_total": len(integ["checks"]),
                      "jit_during_measurement": {l: per_leg[l]["jit_lines_by_phase"].get("measurement", 0)
                                                 for _, l, _ in legs},
                      "prompt_token_ids_sha256": prompt_hashes,
                      "prompt_hash_identical_across_legs": len(set(prompt_hashes.values())) == 1,
                      "model_snapshot_dir": model.get("snapshot_dir"),
                      "all_legs_loaded_model_snapshot": req_models == {model.get("snapshot_dir")}},
        "run_status": {
            "manifest_status": manifest["status"],
            "protected_paths_post_run_status": manifest.get("protected_paths_post_run_status"),
            "archived_attempts_unchanged": manifest.get("archived_attempts_unchanged"),
            "exp3_evidence_unchanged": manifest.get("exp3_evidence_unchanged"),
            "config_diff_status": manifest.get("config_diff_status"),
        },
        "provenance": {
            **{k: prov.get(k) for k in ("git_head", "git_branch", "vllm_kvquant_tree", "rabit_kv2_sha256",
                                        "runner_script_sha256", "modal_app_sha256", "worker_sha256",
                                        "correctness_gate_sha256", "watchdog_sha256", "exp3_worker_sha256",
                                        "exp3_modal_app_sha256", "exp3_runner_sha256",
                                        "canonical_benchmark_deployment_sha256")},
            "vllm_kvquant_snapshot_sha256": (manifest.get("vllm_kvquant_snapshot") or {}).get("sha256"),
            "modal_app_ids": sorted(set(re.findall(r"https://modal\.com/apps/\S+?/(ap-\w+)",
                                                   (results_dir / "modal_session.log").read_text(encoding="utf-8")))),
            "started_utc": manifest["started_utc"], "completed_utc": manifest["completed_utc"],
            "gpu_model": [x.get("name") for x in gpus],
            "gpu_uuid": [x.get("uuid") for x in gpus],
            "gpu_driver": [x.get("driver_version") for x in gpus],
            "packages": env.get("packages"), "python": env.get("python"),
            "vllm_precompiled_wheel_commit": env.get("vllm_precompiled_wheel_commit"),
            "rabit_kv2_sha256_in_image": env.get("rabit_kv2_sha256_lf"),
            "model_snapshot_dir": model.get("snapshot_dir"),
            "model_files_sha256": {k: v.get("sha256") for k, v in (model.get("files") or {}).items()},
        },
        "notes": notes,
        "note_claims": claims,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    summary = build(args.results_dir)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
