"""
Validator for the MLSys 2027 Experiment 4 summary produced by
build_exp4_summary.py. Independent of the builder (no shared code): every
expected value is recomputed here from the raw files in --results-dir. The
only constants are protocol expectations (leg plan, audited native FP8
semantics, allowlist, reps), never measured results.

Verifies:
  * raw and LF-normalized SHA-256 of every recorded result file;
  * exactly six legs A1 B1 C1 C2 B2 A2 with A=bfloat16, B=fp8_e4m3,
    C=rabit_kv2, matching the manifest plan; 15 reps per leg, 30 per dtype,
    prompt/output token counts, excluded warmups;
  * capacity = num_gpu_blocks x block_size = engine log; duplicate capacities;
    implied bytes/token; all three capacity ratios;
  * every per-leg and pooled statistic, all 12 pairwise deltas and their
    direction words, all order-effect drifts, ranges and overlap flags;
  * dtype semantics per leg against the audited native paths, the FP8
    scale/override sources, and the source-derived FP8 facts;
  * every note claim (recomputed text, present in notes, every number in the
    notes backed by a claim) and the required scientific wording;
  * matched config and allowlist, integrity counts (recount; all passed),
    correctness gate (manifest + gate log), GPU clean state, watchdog,
    protected paths, Exp3-evidence flag, prompt hash, model snapshot,
    provenance and environment against modal_session.log;
  * agreement with the runner's own matched_capacity_latency_summary.json.

Read-only. Exits non-zero on any failure.

Usage:
    python benchmarks/mlsys2027/validate_exp4_summary.py \
        --results-dir results/mlsys2027/fp8_baseline \
        --summary results/mlsys2027/fp8_baseline/summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

EPS = 1e-9
PLAN = [(1, "A1", "bfloat16"), (2, "B1", "fp8_e4m3"), (3, "C1", "rabit_kv2"),
        (4, "C2", "rabit_kv2"), (5, "B2", "fp8_e4m3"), (6, "A2", "bfloat16")]
FILES = {"bfloat16": "bf16_deployment.log", "fp8_e4m3": "fp8_e4m3_deployment.log",
         "rabit_kv2": "rabit_kv2_deployment.log"}
SH = {"bfloat16": "bf16", "fp8_e4m3": "fp8", "rabit_kv2": "rabit_kv2"}
NM = {"bfloat16": "BF16", "fp8_e4m3": "native FP8", "rabit_kv2": "RABIT-KV"}
PAIRS = (("fp8_e4m3", "bfloat16"), ("rabit_kv2", "bfloat16"), ("rabit_kv2", "fp8_e4m3"))
METRICS = ("tpot_ms", "ttft_ms", "wall_ms")
MN = {"tpot_ms": "TPOT", "ttft_ms": "TTFT", "wall_ms": "wall"}
DK = (("tpot_median", "tpot_ms", "median"), ("tpot_p90", "tpot_ms", "p90"),
      ("ttft_median", "ttft_ms", "median"), ("wall_median", "wall_ms", "median"))
REPS_PER_LEG, REPS_PER_DTYPE, WARMUPS = 15, 30, 5
# Audited native paths (protocol expectations, from the vllm-kvquant source audit).
EXPECTED_KV = {
    "bfloat16": {"engine_cache_dtype": "bfloat16", "resolved_kv_torch_dtype": "torch.bfloat16",
                 "kv_quant_mode": "NONE", "fp8_storage_view_dtype": None},
    "fp8_e4m3": {"engine_cache_dtype": "fp8_e4m3", "resolved_kv_torch_dtype": "torch.uint8",
                 "kv_quant_mode": "FP8_PER_TENSOR", "fp8_storage_view_dtype": "torch.float8_e4m3fn"},
    "rabit_kv2": {"engine_cache_dtype": "rabit_kv2", "resolved_kv_torch_dtype": "torch.uint8",
                  "kv_quant_mode": "RABIT_KV2", "fp8_storage_view_dtype": None},
}
ALLOWLIST = ["requested.kv_cache_dtype", "kv_dtype.requested_kv_cache_dtype", "kv_dtype.engine_cache_dtype",
             "kv_dtype.resolved_kv_torch_dtype", "kv_dtype.kv_quant_mode", "kv_dtype.fp8_storage_view_dtype"]
NUMBER = re.compile(r"(?<![A-Za-z_\d.])[-+]?\d(?:[\d,]*\d)?(?:\.\d+)?")


def lf_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def raw_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def p90(v: list[float]) -> float:
    return statistics.quantiles(v, n=10, method="inclusive")[8]


def load_legs(path: Path) -> dict:
    legs, cur, phase = {}, None, None
    for line in path.read_text(encoding="utf-8").splitlines():
        h = re.match(r"^===== EXP4 LEG (\w+) \(index (\d+), (\w+)\) =====$", line)
        if h:
            cur, phase = h.group(1), None
            legs[cur] = {"index": int(h.group(2)), "dtype": h.group(3), "samples": [], "warmups": [], "t": {},
                         "log_tokens": None, "log_gib": None, "jit_measure": 0}
            continue
        if cur is None:
            continue
        L = legs[cur]
        if line.startswith("EXP4_SAMPLE "):
            L["samples"].append(json.loads(line[len("EXP4_SAMPLE "):]))
        elif line.startswith("EXP4_WARMUP "):
            L["warmups"].append(json.loads(line[len("EXP4_WARMUP "):]))
        elif re.match(r"^EXP4_[A-Z_]+=\{", line):
            k, v = line.split("=", 1)
            L["t"][k] = json.loads(v)
        else:
            s = line.strip()
            if s == "EXP4_MEASUREMENT_BEGIN":
                phase = "m"
            elif s == "EXP4_MEASUREMENT_END":
                phase = None
            if phase == "m" and "Triton kernel JIT compilation during inference" in line:
                L["jit_measure"] += 1
            g = re.search(r"GPU KV cache size: ([\d,]+) tokens", line)
            if g:
                L["log_tokens"] = int(g.group(1).replace(",", ""))
            g = re.search(r"Available KV cache memory: ([\d.]+) GiB", line)
            if g:
                L["log_gib"] = float(g.group(1))
    return legs


def fmt(f: str, v) -> str:
    return ("overlap" if v else "do not overlap") if f == "overlap" else f.format(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    a = ap.parse_args(argv)
    rd = a.results_dir
    s = json.loads(a.summary.read_text(encoding="utf-8"))
    fails: list[str] = []
    cnt: dict[str, int] = {}

    def need(ok, msg: str, cat: str) -> None:
        cnt[cat] = cnt.get(cat, 0) + 1
        if not ok:
            fails.append(f"[{cat}] {msg}")

    def eq(x, y) -> bool:
        return isinstance(x, (int, float)) and isinstance(y, (int, float)) and abs(x - y) <= EPS

    def section(fn, cat):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001  (malformed summary/evidence = failure, never a crash)
            fails.append(f"[{cat}] could not evaluate: {type(exc).__name__}: {exc}")

    manifest = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
    integ = json.loads((rd / "integrity_check.json").read_text(encoding="utf-8"))
    cdiff = json.loads((rd / "matched_config_diff.json").read_text(encoding="utf-8"))
    runner = json.loads((rd / "matched_capacity_latency_summary.json").read_text(encoding="utf-8"))
    session = (rd / "modal_session.log").read_text(encoding="utf-8")
    gate_txt = (rd / "correctness_gate.log").read_text(encoding="utf-8")
    raw = {d: load_legs(rd / f) for d, f in FILES.items()}
    legs = {label: raw[d].get(label) for _, label, d in PLAN}
    R: dict = {}
    pooled: dict = {}

    # ---------------------------------------------------------------- hashes
    def hashes():
        rec = s["result_files_sha256"]
        need(sorted(rec) == sorted(p.name for p in rd.iterdir() if p.is_file() and p.name != "summary.json"),
             f"hashed file set {sorted(rec)} != result files on disk", "hashes")
        for f, h in rec.items():
            need((rd / f).is_file() and h["raw"] == raw_sha(rd / f) and h["lf_normalized"] == lf_sha(rd / f),
                 f"{f}: recorded SHA-256 does not match the file", "hashes")

    # -------------------------------------------------------------- structure
    def structure():
        found = sorted((L["index"], lab, L["dtype"]) for d in raw for lab, L in raw[d].items())
        need(found == PLAN, f"legs in logs {found} != A1 B1 C1 C2 B2 A2 plan", "structure")
        need([(x["index"], x["leg"], x["kv_cache_dtype"]) for x in manifest["legs"]] == PLAN,
             "manifest leg plan differs", "structure")
        need([(x["index"], x["leg"], x["kv_cache_dtype"]) for x in s["legs"]] == PLAN, "summary legs differ", "structure")
        pr = manifest["protocol"]
        need(pr["measured_reps_per_leg"] == REPS_PER_LEG and pr["measured_reps_per_dtype"] == REPS_PER_DTYPE
             and pr["warmups_per_leg_excluded"] == WARMUPS, "manifest protocol reps/warmups differ", "structure")
        for _, lab, d in PLAN:
            L = legs[lab]
            need([r["rep"] for r in L["samples"]] == list(range(REPS_PER_LEG)), f"{lab}: reps != 0..14", "structure")
            need(len(L["warmups"]) == WARMUPS and s["per_leg"][lab]["warmups_excluded"] == WARMUPS,
                 f"{lab}: warmups != {WARMUPS}", "structure")
            need(all(r["prompt_tokens"] == pr["context_tokens"] and r["output_tokens"] == pr["output_tokens"]
                     for r in L["samples"] + L["warmups"]), f"{lab}: prompt/output token counts wrong", "structure")
            need(L["t"]["EXP4_LEG"] == {"leg": lab, "kv_cache_dtype": d}, f"{lab}: EXP4_LEG tag wrong", "structure")
            need(s["per_leg"][lab]["kv_cache_dtype"] == d and s["per_leg"][lab]["index"] == PLAN[[p[1] for p in PLAN].index(lab)][0],
                 f"{lab}: summary dtype/index wrong", "structure")
        wl = legs["A1"]["t"]["EXP4_WORKLOAD"]
        R["workload.context_tokens"], R["workload.output_tokens"] = wl["context_tokens"], wl["output_tokens"]
        R["workload.warmups"], R["workload.reps_per_leg"] = len(legs["A1"]["warmups"]), len(legs["A1"]["samples"])
        need(s["scope"]["prompt_tokens"] == wl["context_tokens"] and s["scope"]["generated_tokens"] == wl["output_tokens"],
             "scope token counts wrong", "structure")

    # --------------------------------------------------------------- capacity
    def capacity():
        for _, lab, d in PLAN:
            L = legs[lab]
            c = L["t"]["EXP4_CAPACITY"]
            sc = s["capacity"]["per_leg"][lab]
            need(c["num_gpu_blocks"] * c["block_size"] == c["capacity_tokens"] == L["log_tokens"],
                 f"{lab}: capacity != blocks x block_size / engine log", "capacity")
            need(sc["num_gpu_blocks"] == c["num_gpu_blocks"] and sc["block_size"] == c["block_size"]
                 and sc["capacity_tokens"] == c["capacity_tokens"]
                 and sc["engine_log_available_kv_cache_memory_gib"] == L["log_gib"]
                 and eq(sc["implied_physical_bytes_per_token"], L["log_gib"] * 2**30 / c["capacity_tokens"]),
                 f"{lab}: summary capacity block wrong", "capacity")
        for d in FILES:
            l1, l2 = [lab for _, lab, dd in PLAN if dd == d]
            c1, c2 = legs[l1]["t"]["EXP4_CAPACITY"], legs[l2]["t"]["EXP4_CAPACITY"]
            dup = c1 == c2
            need(dup, f"{d}: duplicate capacity {l1} != {l2}", "capacity")
            pd = s["capacity"]["per_dtype"][d]
            need(pd["duplicate_capacity_identical"] is dup and s["pooled_per_dtype"][d]["duplicate_capacity_identical"] is dup
                 and pd["capacity_tokens"] == c1["capacity_tokens"] and pd["num_gpu_blocks"] == c1["num_gpu_blocks"],
                 f"{d}: summary per-dtype capacity wrong", "capacity")
            R[f"capacity.{SH[d]}.tokens"], R[f"capacity.{SH[d]}.blocks"] = c1["capacity_tokens"], c1["num_gpu_blocks"]
            R[f"capacity.{SH[d]}.block_size"] = c1["block_size"]
        for x, y in PAIRS:
            k = f"{SH[x]}_over_{SH[y]}"
            cx, cy = R[f"capacity.{SH[x]}.tokens"], R[f"capacity.{SH[y]}.tokens"]
            e = s["capacity"]["ratios"][k]
            need(e[SH[x]] == cx and e[SH[y]] == cy and eq(e["ratio"], cx / cy) and e["signed_delta_tokens"] == cx - cy,
                 f"capacity ratio {k} does not recompute", "capacity")
            R[f"ratio.{k}"] = cx / cy

    # ------------------------------------------------------- samples / stats
    def statistics_():
        for _, lab, d in PLAN:
            sl = s["per_leg"][lab]
            for m in METRICS:
                v = [r[m] for r in legs[lab]["samples"]]
                need(sl[f"{m}_samples"] == v, f"{lab}: {m} samples differ from raw log", "samples")
                for st, val in (("median", statistics.median(v)), ("p90", p90(v)), ("mean", statistics.mean(v)),
                                ("stdev", statistics.stdev(v)), ("min", min(v)), ("max", max(v)), ("n", len(v))):
                    need(eq(sl[m][st], val), f"{lab}.{m}.{st} does not recompute", "stats")
        for d in FILES:
            labs = [lab for _, lab, dd in PLAN if dd == d]
            rows = {m: [r[m] for lab in labs for r in legs[lab]["samples"]] for m in METRICS}
            need(all(len(v) == REPS_PER_DTYPE for v in rows.values()), f"{d}: pooled n != 30", "structure")
            pooled[d] = rows
            sp = s["pooled_per_dtype"][d]
            need(sp["legs"] == labs, f"{d}: pooled legs wrong", "stats")
            for m, v in rows.items():
                for st, val in (("median", statistics.median(v)), ("p90", p90(v)), ("mean", statistics.mean(v)),
                                ("stdev", statistics.stdev(v)), ("min", min(v)), ("max", max(v)), ("n", len(v))):
                    need(eq(sp[m][st], val), f"pooled {d}.{m}.{st} does not recompute", "stats")
        R["workload.reps_per_dtype"] = len(pooled["bfloat16"]["tpot_ms"])

    # ----------------------------------------------------------------- deltas
    def deltas():
        for x, y in PAIRS:
            key = f"{SH[x]}_minus_{SH[y]}"
            for dk, m, st in DK:
                f = statistics.median if st == "median" else p90
                vx, vy = f(pooled[x][m]), f(pooled[y][m])
                e = s["pairwise_deltas_pooled"][key][dk]
                word = "slower" if vx - vy > 0 else ("faster" if vx - vy < 0 else "equal")
                need(eq(e[SH[x]], vx) and eq(e[SH[y]], vy) and eq(e["signed_delta_ms"], vx - vy)
                     and eq(e["signed_delta_pct"], (vx / vy - 1) * 100) and e["direction"] == word,
                     f"delta {key}.{dk} does not recompute", "deltas")
                R[f"pooled.{SH[x]}.{m}.{st}"], R[f"pooled.{SH[y]}.{m}.{st}"] = vx, vy
                R[f"delta.{key}.{dk}.ms"], R[f"delta.{key}.{dk}.pct"] = vx - vy, (vx / vy - 1) * 100
                R[f"_dir.{key}.{dk}"] = word
        n = sum(len(v) for v in s["pairwise_deltas_pooled"].values())
        need(n == 12, f"expected 12 pairwise deltas, found {n}", "deltas")

    # ----------------------------------------------------------- order effects
    def order():
        need(len(s["order_effects"]) == 3, "expected 3 order-effect entries", "order")
        for d in FILES:
            l1, l2 = [lab for _, lab, dd in PLAN if dd == d]
            okey = f"{l1}_to_{l2}"
            e = s["order_effects"][okey]
            need(e["kv_cache_dtype"] == d and e["first_leg"] == l1 and e["second_leg"] == l2,
                 f"order {okey} labels wrong", "order")
            for m in METRICS:
                va, vb = [r[m] for r in legs[l1]["samples"]], [r[m] for r in legs[l2]["samples"]]
                ma, mb = statistics.median(va), statistics.median(vb)
                ov = not (max(va) < min(vb) or max(vb) < min(va))
                x = e[m]
                need(eq(x["first_median"], ma) and eq(x["second_median"], mb) and eq(x["median_diff_ms"], mb - ma)
                     and eq(x["median_drift_pct"], (mb / ma - 1) * 100), f"order {okey}.{m} drift wrong", "order")
                need(x["first_range"] == [min(va), max(va)] and x["second_range"] == [min(vb), max(vb)],
                     f"order {okey}.{m} ranges wrong", "order")
                need(x["ranges_overlap"] is ov, f"order {okey}.{m} overlap claim {x['ranges_overlap']} != {ov}", "order")
                base = f"order.{okey}.{m}"
                R[f"{base}.median_drift_pct"], R[f"{base}.first_median"], R[f"{base}.second_median"] = \
                    (mb / ma - 1) * 100, ma, mb
                R[f"{base}.first_min"], R[f"{base}.first_max"] = min(va), max(va)
                R[f"{base}.second_min"], R[f"{base}.second_max"] = min(vb), max(vb)
                R[f"{base}.ranges_overlap"] = ov

    # --------------------------------------------------------- dtype semantics
    def semantics():
        for _, lab, d in PLAN:
            kv, eff = legs[lab]["t"]["EXP4_KV_DTYPE"], legs[lab]["t"]["EXP4_EFFECTIVE_ENGINE_CONFIG"]
            need(kv["requested_kv_cache_dtype"] == d and all(kv[k] == v for k, v in EXPECTED_KV[d].items()),
                 f"{lab}: KV dtype {kv} is not the audited native {d} path", "semantics")
            need(eff["calculate_kv_scales"] is False and eff["kv_cache_dtype_skip_layers"] == []
                 and eff["hf_quantization_config"] is None and eff["quantization"] is None,
                 f"{lab}: KV scale / dtype override source present", "semantics")
            need(eff["attention_backend"] == "AttentionBackendEnum.TRITON_ATTN" and eff["enforce_eager"] is True,
                 f"{lab}: not eager Triton", "semantics")
            need(s["per_leg"][lab]["kv_dtype"] == kv, f"{lab}: summary kv_dtype differs from log", "semantics")
        for d in FILES:
            sd = s["dtype_semantics"][d]
            kv = legs[[lab for _, lab, dd in PLAN if dd == d][0]]["t"]["EXP4_KV_DTYPE"]
            eff = legs[[lab for _, lab, dd in PLAN if dd == d][0]]["t"]["EXP4_EFFECTIVE_ENGINE_CONFIG"]
            need(sd["requested_dtype"] == kv["requested_kv_cache_dtype"] and sd["engine_cache_dtype"] == kv["engine_cache_dtype"]
                 and sd["resolved_storage_dtype"] == kv["resolved_kv_torch_dtype"] and sd["kv_quant_mode"] == kv["kv_quant_mode"]
                 and sd["fp8_storage_view_dtype"] == kv["fp8_storage_view_dtype"]
                 and sd["calculate_kv_scales"] is eff["calculate_kv_scales"]
                 and sd["kv_cache_dtype_skip_layers"] == eff["kv_cache_dtype_skip_layers"]
                 and sd["checkpoint_quantization_config"] == eff["hf_quantization_config"],
                 f"{d}: summary dtype semantics differ from logs", "semantics")
        f8 = s["dtype_semantics"]["fp8_e4m3"]["native_fp8_notes"]
        eff = legs["B1"]["t"]["EXP4_EFFECTIVE_ENGINE_CONFIG"]
        kv = legs["B1"]["t"]["EXP4_KV_DTYPE"]
        native = all(kv[k] == v for k, v in EXPECTED_KV["fp8_e4m3"].items())
        need(f8["native_per_tensor_fp8_e4m3_path"] is native is True, "FP8 native-path flag wrong", "semantics")
        need(f8["default_kv_scale_1_0_implied_by_source"] is (eff["hf_quantization_config"] is None
                                                             and eff["calculate_kv_scales"] is False
                                                             and eff["quantization"] is None) is True,
             "FP8 default-scale flag wrong", "semantics")
        need(f8["native_query_fp8_conversion_implied_by_source"] is (native and eff["attention_backend"]
                                                                    == "AttentionBackendEnum.TRITON_ATTN") is True,
             "FP8 query-conversion flag wrong", "semantics")
        need(f8["checkpoint_quantization_config"] is None and f8["calculate_kv_scales"] is False
             and "not observed at runtime" in f8["evidence_type"] and "source/config-derived" in f8["evidence_type"],
             "FP8 evidence-type statement wrong", "semantics")
        need(s["scope"]["fp8_quality_equivalence_claimed"] is False and s["scope"]["experiment3_samples_used"] is False,
             "scope flags wrong", "semantics")
        need(runner.get("experiment3_samples_used") is False and runner.get("fp8_quality_evaluated") is False,
             "runner scope flags wrong", "semantics")

    # ------------------------------------------------------------ status / gate
    def status():
        recount = {st: sum(1 for c in integ["checks"] if c["state"] == st)
                   for st in ("passed", "failed", "not_run", "not_evaluated")}
        need(recount == integ["counts"] == s["integrity"]["counts"] == manifest["integrity_counts"],
             f"integrity counts {s['integrity']['counts']} != recount {recount}", "status")
        need(recount == {"passed": len(integ["checks"]), "failed": 0, "not_run": 0, "not_evaluated": 0}
             and integ["all_ok"] is True and s["integrity"]["all_ok"] is True, f"not fully passed: {recount}", "status")
        for k, v in recount.items():
            R[f"integrity.{k}"] = v
        need(cdiff["status"] == "passed" and cdiff["matched"] is True and cdiff["violations"] == []
             and cdiff["dtype_induced_allowlist"] == ALLOWLIST
             and sorted(cdiff["fields_differing_between_dtypes"]) == sorted(ALLOWLIST)
             and s["matched_config"] == {k: cdiff[k] for k in ("status", "matched", "fields_compared",
                                                                 "fields_differing_between_dtypes", "violations",
                                                                 "dtype_induced_allowlist")},
             "matched config / allowlist wrong", "status")
        for _, lab, d in PLAN:  # config diff configs equal the logs' tags
            cfg = cdiff["configs"][lab]
            kv, eff = legs[lab]["t"]["EXP4_KV_DTYPE"], legs[lab]["t"]["EXP4_EFFECTIVE_ENGINE_CONFIG"]
            need(all(cfg[f"kv_dtype.{k}"] == v for k, v in kv.items())
                 and all(cfg.get(f"effective.{k}") == v for k, v in eff.items() if not isinstance(v, dict))
                 and cfg["workload.prompt_token_ids_sha256"] == legs[lab]["t"]["EXP4_WORKLOAD"]["prompt_token_ids_sha256"],
                 f"{lab}: matched_config_diff.json disagrees with the leg log", "status")
        g = manifest["correctness_gate"]
        m_py = re.search(r"^(\d+) passed, (\d+) warnings? in", gate_txt, re.M)
        m_prep = re.search(r"prep exactness: PASSED \((\d+)/(\d+)\)", gate_txt)
        m_dec = re.search(r"fast decode append exactness: PASSED \((\d+) state/byte steps", gate_txt)
        need(m_py and m_prep and m_dec and "Dispatch preflight: PASSED" in gate_txt
             and "full attention exactness: PASSED" in gate_txt
             and "RABIT-2 FINAL TARGETED REGRESSION PASSED" in gate_txt and "pytest exit=0" in gate_txt
             and m_prep.group(1) == m_prep.group(2)
             and g["result"] == {"passed": True} and g["pytest_exit"] == 0
             and g["pytest_passed"] == int(m_py.group(1)) and g["pytest_warnings"] == int(m_py.group(2)),
             "correctness gate not passed / log and manifest disagree", "status")
        if m_py and m_prep and m_dec:
            R["gate.pytest_passed"], R["gate.pytest_warnings"] = int(m_py.group(1)), int(m_py.group(2))
            R["gate.prep_passed"], R["gate.prep_total"] = m_prep.group(1), m_prep.group(2)
            R["gate.decode_append_steps"] = int(m_dec.group(1))
            gl = s["correctness_gate"]["from_gate_log"]
            need(gl["pytest_passed"] == R["gate.pytest_passed"] and gl["pytest_warnings"] == R["gate.pytest_warnings"]
                 and gl["prep_exactness_passed"] == R["gate.prep_passed"]
                 and gl["prep_exactness_total"] == R["gate.prep_total"]
                 and gl["decode_append_steps"] == R["gate.decode_append_steps"]
                 and gl["full_attention_exactness_passed"] is True and gl["dispatch_preflight_passed"] is True,
                 "summary gate block differs from gate log", "status")
        need(manifest["status"] == "passed" and s["run_status"]["manifest_status"] == "passed", "manifest not passed", "status")
        need(manifest["protected_paths_post_run_status"] == "clean"
             and s["run_status"]["protected_paths_post_run_status"] == "clean", "protected paths not clean", "status")
        need(manifest["exp3_evidence_unchanged"] is True and s["run_status"]["exp3_evidence_unchanged"] is True
             and manifest["archived_attempts_unchanged"] is True and s["run_status"]["archived_attempts_unchanged"] is True,
             "Exp3 evidence / archive flags not true", "status")
        pre = manifest["gpu_clean_state"]["pre_leg"]
        base = manifest["gpu_clean_state"]["baseline"]
        tol = manifest["gpu_clean_state"]["tolerance_mib"]
        need(sorted(pre) == sorted(lab for _, lab, _ in PLAN) and not base["compute_apps"]
             and all(p["clean"] and not p["readings"][-1]["compute_apps"]
                     and all(u <= b + tol for u, b in zip(p["readings"][-1]["memory_used_mib"], base["memory_used_mib"]))
                     for p in pre.values())
             and all(v["clean"] for v in s["gpu_clean_state"]["pre_leg"].values()), "GPU clean state wrong", "status")
        ex = manifest["processes"]["exits"]
        need(sorted(ex) == sorted(["gate"] + [lab for _, lab, _ in PLAN])
             and all(e["returncode"] == 0 and e["timed_out"] is False and e["group_processes_remaining"] == [] for e in ex.values())
             and manifest["processes"]["watchdog_timeouts"] == [] and s["processes"]["watchdog_timeouts"] == [],
             "process / watchdog status wrong", "status")
        jit = {lab: legs[lab]["jit_measure"] for _, lab, _ in PLAN}
        need(all(v == 0 for v in jit.values()) and s["integrity"]["jit_during_measurement"] == jit,
             f"JIT during measurement: {jit}", "status")
        ph = {lab: legs[lab]["t"]["EXP4_WORKLOAD"]["prompt_token_ids_sha256"] for _, lab, _ in PLAN}
        need(len(set(ph.values())) == 1 and s["integrity"]["prompt_token_ids_sha256"] == ph
             and s["integrity"]["prompt_hash_identical_across_legs"] is True, "prompt hash inconsistent", "status")
        mdl = [json.loads(ln.split("=", 1)[1]) for ln in session.splitlines() if ln.startswith("EXP4_MODEL=")]
        need(len(mdl) == 1 and all(legs[lab]["t"]["EXP4_REQUESTED_ENGINE_KWARGS"]["model"] == mdl[0]["snapshot_dir"]
                                   for _, lab, _ in PLAN)
             and s["integrity"]["all_legs_loaded_model_snapshot"] is True
             and s["provenance"]["model_files_sha256"] == {k: v["sha256"] for k, v in mdl[0]["files"].items()},
             "model snapshot inconsistent", "status")

    # ------------------------------------------------------------ provenance
    def provenance():
        p, mp = s["provenance"], manifest["provenance"]
        need(all(p[k] == mp[k] for k in ("git_head", "vllm_kvquant_tree", "rabit_kv2_sha256", "runner_script_sha256",
                                         "modal_app_sha256", "worker_sha256", "correctness_gate_sha256",
                                         "watchdog_sha256")), "provenance differs from manifest", "provenance")
        need(p["vllm_kvquant_snapshot_sha256"] == manifest["vllm_kvquant_snapshot"]["sha256"], "snapshot hash differs",
             "provenance")
        env = [json.loads(ln.split("=", 1)[1]) for ln in session.splitlines() if ln.startswith("EXP4_ENVIRONMENT=")]
        need(len(env) == 1, "EXP4_ENVIRONMENT missing from session log", "provenance")
        e = env[0]
        need(p["gpu_uuid"] == [x["uuid"] for x in e["gpus"]] and p["gpu_model"] == [x["name"] for x in e["gpus"]]
             and p["gpu_driver"] == [x["driver_version"] for x in e["gpus"]] and p["packages"] == e["packages"]
             and p["python"] == e["python"] and p["rabit_kv2_sha256_in_image"] == e["rabit_kv2_sha256_lf"] == mp["rabit_kv2_sha256"]
             and len(e["gpus"]) == 1 and "H100" in e["gpus"][0]["name"], "environment provenance wrong", "provenance")
        ids = sorted(set(re.findall(r"https://modal\.com/apps/\S+?/(ap-\w+)", session)))
        need(p["modal_app_ids"] == ids and len(ids) == 1, "Modal app id wrong", "provenance")

    # ----------------------------------------------------------------- notes
    def notes():
        text = " ".join(s["notes"])
        claims = s["note_claims"]
        want = {k for k in R if not k.startswith("_")}
        need({c["quantity"] for c in claims} == want,
             f"note claims != recomputed quantities: extra {sorted({c['quantity'] for c in claims} - want)[:5]} "
             f"missing {sorted(want - {c['quantity'] for c in claims})[:5]}", "notes")
        backed = set()
        for c in claims:
            q = c["quantity"]
            need(q in R and c["text"] == fmt(c["format"], R[q]),
                 f"claim {q}: {c['text']!r} != recomputed {fmt(c['format'], R[q]) if q in R else '?'}", "notes")
            need(c["text"] in text, f"claim text {c['text']!r} not in notes", "notes")
            backed.update(NUMBER.findall(c["text"]))
        for num in NUMBER.findall(text):
            need(num in backed, f"number {num!r} in notes not backed by a claim", "notes")
        # Required scientific wording, re-derived from the recomputed values.
        r_fb, r_rf = R["ratio.fp8_over_bf16"], R["ratio.rabit_kv2_over_fp8"]
        w_fb = ("nearly doubles" if 1.9 <= r_fb < 2.0 else "roughly doubles" if 2.0 <= r_fb < 2.1 else "does not double")
        w_rf = ("substantially more" if r_rf >= 1.5 else "more" if r_rf > 1.0 else "no more")
        need(f"Native FP8 {w_fb} physical capacity relative to BF16" in text, "FP8/BF16 capacity wording wrong", "notes")
        need(f"RABIT-KV provides {w_rf} physical capacity than native FP8" in text, "RABIT/FP8 capacity wording wrong",
             "notes")
        neg = []
        for x, y in PAIRS:
            key = f"{SH[x]}_minus_{SH[y]}"
            words = [R[f"_dir.{key}.{dk}"] for dk, _, _ in DK]
            for (dk, m, st), w in zip(DK, words):
                need(f"{NM[x]} is {w} than {NM[y]} ({fmt('{:+.3f}', R[f'delta.{key}.{dk}.ms'])} ms" in text,
                     f"direction wording {key}.{dk} wrong", "notes")
                if w == "faster":
                    neg.append(key)
            if set(words) == {"slower"}:
                need(f"{NM[x][0].upper() + NM[x][1:]} is slower than {NM[y]} in all four pooled latency metrics" in text,
                     f"{key}: all-slower sentence missing", "notes")
            elif set(words) == {"faster"}:
                need(f"{NM[x][0].upper() + NM[x][1:]} is faster than {NM[y]} in all four pooled latency metrics" in text,
                     f"{key}: all-faster sentence missing", "notes")
            else:
                need(f"latency direction of {NM[x]} vs {NM[y]} is mixed" in text, f"{key}: mixed sentence missing", "notes")
        need(("No speedup is claimed" in text) == (not neg) and ("speedup" not in text.replace("not a speedup", "")
                                                                  .replace("No speedup is claimed", "") or bool(neg)),
             "speedup wording inconsistent with measured signs", "notes")
        dup_all = all(s["capacity"]["per_dtype"][d]["duplicate_capacity_identical"] for d in FILES)
        need(("duplicate capacities are identical in both legs of every dtype" in text) == dup_all,
             "duplicate-capacity wording wrong", "notes")
        bf = f"{'A1'}_to_{'A2'}"
        need(f"BF16 TPOT median drift {fmt('{:+.2f}', R[f'order.{bf}.tpot_ms.median_drift_pct'])}%" in text
             and f"sample ranges {fmt('overlap', R[f'order.{bf}.tpot_ms.ranges_overlap'])}" in text,
             "BF16 drift visibility sentence wrong", "notes")
        for phrase in ("PHYSICAL real-engine vLLM allocator capacity", "PHYSICAL real-engine single-request latency",
                       "one matched H100 session", "eager mode", "Triton backend",
                       "NOT an FP8 quality comparison", "no FP8 quality-equivalence claim",
                       "were reused", "measured fresh in this session",
                       "native vLLM FP8-E4M3 KV-cache path measured here", "not a statement about FP8 methods in general",
                       "not direct runtime tensor observations",
                       f"matched config {cdiff['status']}", f"protected paths {manifest['protected_paths_post_run_status']}"):
            need(phrase in text, f"required phrase missing: {phrase!r}", "notes")

    # --------------------------------------------------- runner cross-check
    def cross():
        for _, lab, d in PLAN:
            rl = runner["per_leg"][lab]
            need(rl["measured_samples"] == [r for r in legs[lab]["samples"]]
                 and rl["capacity"] == legs[lab]["t"]["EXP4_CAPACITY"] and rl["kv_dtype"] == legs[lab]["t"]["EXP4_KV_DTYPE"],
                 f"{lab}: runner summary disagrees with the leg log", "cross")
        for d in FILES:
            h, sp = runner["pooled_per_dtype"][d]["headline"], s["pooled_per_dtype"][d]
            need(eq(h["tpot_ms_median"], sp["tpot_ms"]["median"]) and eq(h["tpot_ms_p90"], sp["tpot_ms"]["p90"])
                 and eq(h["ttft_ms_median"], sp["ttft_ms"]["median"]) and eq(h["wall_ms_median"], sp["wall_ms"]["median"])
                 and runner["pooled_per_dtype"][d]["capacity_tokens"] == sp["capacity_tokens"],
                 f"{d}: runner pooled summary disagrees", "cross")
        rmap = {"fp8_over_bf16": "fp8_over_bf16", "rabit_kv2_over_bf16": "rabit_over_bf16",
                "rabit_kv2_over_fp8": "rabit_over_fp8"}
        for k, rk in rmap.items():
            need(eq(runner["capacity_ratios"][rk]["ratio"], s["capacity"]["ratios"][k]["ratio"]),
                 f"runner capacity ratio {rk} disagrees", "cross")
        dmap = {"fp8_minus_bf16": "fp8_vs_bf16", "rabit_kv2_minus_bf16": "rabit_vs_bf16",
                "rabit_kv2_minus_fp8": "rabit_vs_fp8"}
        for k, rk in dmap.items():
            for dk, _, _ in DK:
                e = runner["signed_latency_deltas_pooled"][rk][dk]
                ms = next(v for kk, v in e.items() if kk.startswith("signed_delta_ms"))
                need(eq(ms, s["pairwise_deltas_pooled"][k][dk]["signed_delta_ms"]), f"runner delta {rk}.{dk} disagrees",
                     "cross")
        for okey in s["order_effects"]:
            re_ = runner["order_effects"][okey.replace("_to_", "_vs_")]
            for m in METRICS:
                need(eq(re_[m]["median_drift_pct"], s["order_effects"][okey][m]["median_drift_pct"])
                     and re_[m]["ranges_overlap"] is s["order_effects"][okey][m]["ranges_overlap"],
                     f"runner order effect {okey}.{m} disagrees", "cross")

    for fn, cat in ((hashes, "hashes"), (structure, "structure"), (capacity, "capacity"), (statistics_, "stats"),
                    (deltas, "deltas"), (order, "order"), (semantics, "semantics"), (status, "status"),
                    (provenance, "provenance"), (notes, "notes"), (cross, "cross")):
        section(fn, cat)

    print(f"results_dir: {rd.as_posix()} | summary: {a.summary.as_posix()}")
    for cat in ("hashes", "structure", "capacity", "samples", "stats", "deltas", "order", "semantics", "status",
                "provenance", "notes", "cross"):
        print(f"  {cat:<10} checks: {cnt.get(cat, 0)}")
    if "ratio.fp8_over_bf16" in R and "delta.fp8_minus_bf16.tpot_median.pct" in R:
        print(f"recomputed: capacity bf16 {R['capacity.bf16.tokens']:,} / fp8 {R['capacity.fp8.tokens']:,} / rabit "
              f"{R['capacity.rabit_kv2.tokens']:,}; FP8/BF16 {R['ratio.fp8_over_bf16']:.4f}x, RABIT/FP8 "
              f"{R['ratio.rabit_kv2_over_fp8']:.4f}x; TPOT median FP8-BF16 "
              f"{R['delta.fp8_minus_bf16.tpot_median.pct']:+.2f}%, RABIT-FP8 "
              f"{R['delta.rabit_kv2_minus_fp8.tpot_median.pct']:+.2f}%")
    print(f"FAILURES: {len(fails)}")
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
