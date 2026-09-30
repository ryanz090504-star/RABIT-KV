"""
RABIT-KV MLSys 2027 -- Experiment 12 runner: larger samples + PAIRED per-example bootstrap CIs (logical fake-quant
quality; not a physical serving benchmark).

Plan (docs/MLSYS_EXPERIMENT_PLAN.md, Experiment 12): are the canonical quality deltas (bf16 vs rabit2; in particular
HotpotQA 60.6 -> 55.2) statistically robust given the small canonical counts (8 / 10 / 15 / 20 / 24)? Control /
treatment: bf16 vs rabit2 (the optional Experiment 1 frontier methods are NOT run -- smallest design). Uncertainty is
reduced by MORE INDEPENDENT UNITS (not reseeding: generation is greedy) and quantified by a PAIRED bootstrap over
examples / windows, computed OFFLINE from the logged per-example scores (paired_bootstrap_ci.py).

Execution: the UNCHANGED canonical scripts benchmarks/quality/*.py (no derived copies, no code change) with only the
sample-count / NIAH-depth CLI arguments enlarged. Each larger selection is a deterministic superset extension of the
canonical slice (first N of the same pinned, filtered pool; first N consecutive WikiText-2 windows; a denser NIAH depth
grid containing the canonical depths).

Validity (kept separate from the statistics): on the CANONICAL SUBSET of every benchmark the bf16 and rabit2 rows must
reproduce the canonical reference -- original Exp1 rules for continuation_ppl / NIAH / passage retrieval, and the
FROZEN post-failure per-example QA control gate (amendment 86b03ea, unchanged thresholds) for HotpotQA / Qasper; the
logical KV MB of every unit must equal the exact accounting model. Nothing in Exp12 can modify any validity threshold.

Usage:
    python benchmarks/mlsys2027/run_experiment12_variance.py --write-protocol   (once, before commit)
    python benchmarks/mlsys2027/run_experiment12_variance.py --dry-run
    python benchmarks/mlsys2027/run_experiment12_variance.py            (NOT until explicitly authorized)
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paired_bootstrap_ci as pb  # noqa: E402
import qa_control_gate as qg  # noqa: E402  (frozen amendment; read-only)
import run_experiment1_quality_frontier as e1  # noqa: E402  (accepted; read-only)
import run_experiment7_kbit_ablation as r7  # noqa: E402  (accepted; read-only)
import run_experiment9_group_ablation as r9  # noqa: E402  (accepted; read-only: storage model)
import run_experiment11_metadata_ablation as r11  # noqa: E402  (accepted; read-only)

ROOT = e1.ROOT
RUNNER_SCRIPT = Path(__file__).resolve()
OUT_DIR = ROOT / "results" / "mlsys2027" / "variance"
MANIFEST = OUT_DIR / "manifest.json"
INTEGRITY = OUT_DIR / "integrity_results.json"
RESULTS = OUT_DIR / "variance_results.json"
PROTOCOL = HERE / "exp12_variance_protocol.json"
QUALITY_DIR = ROOT / "benchmarks" / "quality"
REFERENCE_DIR = ROOT / "results" / "quality"  # the canonical reference run (source of the pinned targets)
METHODS = "bf16,rabit2"
EXP11_EVIDENCE_COMMIT = "ef41a160e6f04aa563ed1d214819b93962ee8db0"
EXP11_FILES = [HERE / "exp11_metadata_scripts.py", HERE / "exp11_metadata", HERE / "run_experiment11_metadata_ablation.py",
               HERE / "test_experiment11_metadata.py", HERE / "exp11_metadata_protocol.json", r11.OUT_DIR]
PROTECTED_PATHS = [*r11.PROTECTED_PATHS, *EXP11_FILES, QUALITY_DIR, REFERENCE_DIR]
STORAGE_TOL_MB = 0.001  # per-unit accounting-integrity gate (printed 3-decimal MB vs exact model)
NIAH_DEPTHS = [f"{k * 0.05:.2f}" for k in range(1, 20)]  # 0.05 .. 0.95, contains the canonical 0.1/0.25/0.5/0.75/0.9
NIAH_CONTEXTS = [4096, 8192, 16384]

# Canonical (Exp1) vs Experiment 12 selections. Every Exp12 selection is a superset extension of the canonical one.
SELECTION = {
    "continuation_ppl": {"canonical_units": 8, "exp12_units": 32, "unit": "independent consecutive 1152-token WikiText-2 "
                         "window (1024 context + 128 scored)", "expansion": "4x (the plan's documented example factor)",
                         "rule": "the first 32 x (1024 + 128) tokens of the tokenized WikiText-2 test stream, split into "
                                 "consecutive non-overlapping windows; windows 1-8 are the canonical windows",
                         "args": ["--context-tokens", "1024", "--eval-tokens", "128", "--samples", "32"]},
    "niah": {"canonical_units": 15, "exp12_units": 57, "unit": "(context length, needle depth) case",
             "expansion": "denser depth grid 0.05..0.95 step 0.05 (19 depths) x the same 3 context lengths",
             "rule": "every (context length, depth) pair; needle at round(filler x depth); the 15 canonical cases are "
                     "the depths 0.10 / 0.25 / 0.50 / 0.75 / 0.90",
             "independence_caveat": "cases share the same needle and WikiText-2 filler; denser depths are not fully "
                                    "independent draws -- the CI is descriptive of this case set",
             "args": ["--context-lengths", ",".join(map(str, NIAH_CONTEXTS)), "--needle-depths", ",".join(NIAH_DEPTHS),
                      "--max-new-tokens", "16"]},
    "passage_retrieval": {"canonical_units": 10, "exp12_units": 200, "unit": "LongBench passage_retrieval_en example",
                          "expansion": "the full pinned pool (200 rows); canonical = rows [0, 10)",
                          "rule": "dataset rows [0, 200) of passage_retrieval_en @ 915b0c6 (test split), no filter",
                          "args": ["--sample-start", "0", "--samples", "200", "--max-input-tokens", "16384",
                                   "--max-new-tokens", "32"]},
    "hotpotqa": {"canonical_units": 20, "exp12_units": 100, "unit": "LongBench-E hotpotqa example",
                 "expansion": "the full 8k+ bucket (100 examples; dataset rows 200-299); canonical = filtered [0, 20)",
                 "rule": "filtered positions [0, 100) of hotpotqa_e @ 92b6c5f (test split) with length >= 8000",
                 "args": ["--sample-start", "0", "--samples", "100", "--length-bucket", "8k+", "--max-input-tokens",
                          "16384", "--max-new-tokens", "32"]},
    "qasper": {"canonical_units": 24, "exp12_units": 24, "unit": "LongBench-E qasper example",
               "expansion": "none possible: the 8k+ bucket has exactly 24 examples and the canonical slice is the full "
                            "bucket (the plan: use the full available 8k+ bucket)",
               "rule": "filtered positions [0, 24) of qasper_e @ 52edb9d (test split) with length >= 8000 -- identical "
                       "to canonical",
               "args": ["--sample-start", "0", "--samples", "24", "--length-bucket", "8k+", "--max-input-tokens",
                        "16384", "--max-new-tokens", "32"]},
}
POOL_FACTS = {"hotpotqa": {"rows": 300, "bucket_8k_plus": 100}, "qasper": {"rows": 224, "bucket_8k_plus": 24},
              "passage_retrieval": {"rows": 200}}  # counted offline from the pinned parquet files
EXPECTED_COUNT = {"continuation_ppl": 32 * 128, "niah": 57, "passage_retrieval": 200, "hotpotqa": 100, "qasper": 24}


def canonical_rabit2() -> dict:
    text = (QUALITY_DIR / "hotpotqa.py").read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == "config_for_method")
    ns: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<config_for_method>", "exec"), ns)  # noqa: S102
    return ns["config_for_method"]("rabit2")


def runs() -> list[dict]:
    return [{"name": b, "script": QUALITY_DIR / f"{b}.py", "args": [*SELECTION[b]["args"], "--methods", METHODS]}
            for b in pb.BENCHMARKS]


def build_protocol() -> dict:
    cfg = canonical_rabit2()
    ref = e1.CANONICAL_REFERENCE
    amendment = qg.load_amendment()
    return {
        "experiment": 12, "type": "logical fake-quant quality: larger samples + paired per-example bootstrap CIs "
                                  "(NOT a physical serving benchmark)",
        "question": ("Are the canonical quality deltas (bf16 vs rabit2; in particular HotpotQA 60.6 -> 55.2 F1) "
                     "statistically robust, or within noise given the current small sample counts (8/10/15/20/24)?"),
        "plan_hypothesis_reported_not_presupposed": ("with paired per-example CIs and larger N, the core directional "
                                                     "findings (small PPL degradation; larger HotpotQA regression) "
                                                     "hold, and the HotpotQA delta's CI excludes zero"),
        "purpose_separation": {
            "addresses": "SAMPLING uncertainty of the reported bf16-vs-rabit2 quality deltas: more independent units "
                         "and a paired example / window bootstrap CI on each delta",
            "does_not_address": "run-to-run generation variability of identical configs (fresh-GPU non-bit-exact "
                                "greedy decoding) -- quantified by the control reproducibility audit and handled for "
                                "validity by the frozen QA control gate; the bootstrap is conditional on this run's "
                                "generations and never used to smooth or hide control instability",
            "treatment_vs_control": "the paired bootstrap CI is on the within-run rabit2 - bf16 delta on identical "
                                    "units; run-to-run variability is an additional, unmodelled component"},
        "conditions": {"bf16": {"role": "reference (uncompressed)"},
                       "rabit2": {"role": "final RABIT policy (canonical, unchanged)", "config": cfg,
                                  "shorthand": "K3/V2/G32/R4/META8g64"}},
        "frontier_methods": "not run (the plan marks them optional; smallest design answering the question)",
        "scope": {"benchmarks": list(pb.BENCHMARKS),
                  "multilingual_ppl": "not included: the plan's question names the canonical counts 8/10/15/20/24 "
                                      "(the five Exp1 benchmarks); multilingual PPL (Exp2) is out of scope here"},
        "code": "the unchanged canonical benchmarks/quality/*.py; only CLI sample-count / NIAH-depth arguments "
                "differ; per-example logging already exists in all five scripts",
        "common": r7.COMMON_FACTS,
        "gpu_note": "scripts request gpu='H100'; Modal has previously assigned an H200 for some runs -- the GPU of "
                    "each run is recorded from its log",
        "selection": SELECTION, "pool_facts": POOL_FACTS,
        "benchmark_facts_canonical": {b: r7.BENCHMARK_FACTS[b] for b in pb.BENCHMARKS},
        "runs": [{"name": r["name"], "script": r["script"].relative_to(ROOT).as_posix(), "args": r["args"]}
                 for r in runs()],
        "expected_count_per_method": EXPECTED_COUNT,
        "metrics": {b: pb.METRIC[b][1] for b in pb.BENCHMARKS},
        "statistical_procedure": {
            **pb.BOOTSTRAP,
            "inferential_benchmarks": list(pb.INFERENTIAL),
            "unit": {"continuation_ppl": "selected WikiText-2 evaluation window (32 deterministic consecutive windows; "
                                         "each scores 128 tokens)",
                     "passage_retrieval": "example", "hotpotqa": "example", "qasper": "example"},
            "ci_description": pb.CI_DESCRIPTION,
            "paired": "each replicate resamples unit indices once and applies them to both bf16 and rabit2",
            "statistic": {"qa_retrieval": "100 * (mean rabit2 unit score - mean bf16 unit score)",
                          "continuation_ppl": "100 * (exp(mean ln PPL_rabit2 - mean ln PPL_bf16) - 1)"},
            "inputs": "per-example values as logged by the canonical scripts (F1 / retrieval score 3 decimals, "
                      "per-window PPL 4 decimals); no re-generation",
            "robustness_descriptive": ["per-unit delta distribution (min / q25 / median / q75 / max / mean)",
                                       "counts rabit2 better / worse / equal",
                                       "largest absolute per-unit delta / contribution",
                                       "share of the aggregate delta explained by the top-1 and top-3 units by "
                                       "|contribution| (and share of total |contribution|)",
                                       "share of the aggregate delta explained by the worst-1 and worst-3 units "
                                       "(most negative rabit2 - bf16 contributions)"],
            "niah": {"inferential_statistics": None,
                     "reason": "the 57 cases (19 depths x 3 context lengths) reuse one synthetic needle / filler "
                               "construction and are not 57 independent draws; no bootstrap and no pseudo-CI",
                     "report": ["overall exact retrieval count / 57 and accuracy per method",
                                "19-case result separately at 4k, 8k and 16k",
                                "failed (context, depth) coordinates, if any",
                                "if both methods are 57/57: the expanded robustness grid did not distinguish them"]},
            "qasper_coverage": "Exp12 does NOT increase Qasper sample coverage (N = 24 = the complete pinned 8k+ "
                               "bucket); it only quantifies uncertainty on that complete bucket",
            "qa_validity_gate_independent_of_statistics": "the frozen QA control gate is a validity check only; it "
                                                          "never enters the bootstrap / effect analysis",
            "no_outlier_removal": True,
            "interpretation": "report the delta, its CI and N; a CI containing 0 is not evidence of no effect; "
                              "descriptive, cautious wording; QA deltas are also compared with the observed "
                              "identical-config generation variability (report-only)"},
        "validity": {
            "separate_from_statistics": True,
            "canonical_subset": {b: SELECTION[b]["canonical_units"] for b in pb.BENCHMARKS},
            "continuation_ppl_niah_passage_retrieval": {
                "rule": "on the canonical subset, bf16 and rabit2 reproduce the canonical targets within the original "
                        "Exp1 tolerances",
                "tolerances": {"ppl": {"relative": e1.PPL_RELATIVE_TOLERANCE},
                               "accuracy_pct": {"absolute_points": e1.PERCENTAGE_ABSOLUTE_TOLERANCE}},
                "targets": {b: {"bf16": ref[b]["bf16"], "rabit2": ref[b]["rabit2"]}
                            for b in ("continuation_ppl", "niah", "passage_retrieval")}},
            "hotpotqa_qasper": {
                "rule": "on the canonical subset (hotpotqa first 20, qasper all 24), bf16 and rabit2 must pass the "
                        "FROZEN post-failure per-example QA control gate against results/quality/<benchmark>.log "
                        "(identical dataset indices, token counts and ground truths; score mismatches and L1 within "
                        "the frozen maxima). Units beyond the canonical subset have no control reference",
                "amendment": qg.AMENDMENT.relative_to(ROOT).as_posix(),
                "amendment_sha256_lf": r11.AMENDMENT_SHA256_LF, "amendment_commit": r11.AMENDMENT_COMMIT,
                "thresholds": {b: {m: {k: amendment["thresholds"][b][m][k] for k in (
                    "historical_max_score_mismatch_count", "historical_max_l1_score_distance")}
                    for m in ("bf16", "rabit2")} for b in ("hotpotqa", "qasper")},
                "new_thresholds_derived": False, "exp12_data_can_modify_thresholds": False},
            "storage": {"rule": "every unit's printed logical KV MB within 0.001 MB of the exact model: rabit2 = the "
                                "accepted canonical accounting at prefix T; bf16 = 32*2*T*8*128*2 bytes; on the "
                                "canonical subset also equal to the canonical per-unit values",
                        "prefix": "continuation 1024; NIAH context - 1; LongBench used tokens - 1",
                        "absolute_mb": STORAGE_TOL_MB},
            "structure": ["exit code 0 and no traceback", "bf16 and rabit2 summary rows present with the expected "
                          "counts", "per-unit bf16 / rabit2 rows complete, identical unit keys, in order",
                          "canonical subset units identical (keys / indices) to the canonical run"]},
        "outputs": {"per benchmark log": "results/mlsys2027/variance/<benchmark>.log",
                    "integrity": "results/mlsys2027/variance/integrity_results.json",
                    "statistics": "results/mlsys2027/variance/variance_results.json (schema: bootstrap spec; per "
                                  "benchmark: metric, delta_definition, n_units, bf16_aggregate, rabit2_aggregate, "
                                  "delta, ci_low, ci_high, confidence, resamples, seed, ci_contains_zero, robustness, "
                                  "per_unit)",
                    "manifest": "results/mlsys2027/variance/manifest.json"},
        "compute_plan": {"gpu_runs": "5 Modal apps (one per benchmark), 2 methods each, sequential",
                         "method_unit_evaluations": 2 * sum(SELECTION[b]["exp12_units"] for b in pb.BENCHMARKS),
                         "expected_wall_minutes": "25-45", "expected_h100_hours": "about 0.5-0.8",
                         "offline": "bootstrap and robustness analysis on CPU from the logs; no GPU re-run"},
        "completion_criterion": "all five benchmarks valid; paired bootstrap 95% CI computed for the bf16-vs-rabit2 "
                                "delta of continuation_ppl, passage_retrieval, hotpotqa and qasper, with overlap or "
                                "non-overlap with zero explicitly reported; NIAH reported as the complete 57-case "
                                "deterministic robustness grid (no inferential statistics)",
        "not_reported": ["physical allocator capacity", "throughput", "latency"],
    }


def load_protocol() -> dict:
    committed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if committed != json.loads(json.dumps(build_protocol())):
        raise RuntimeError("exp12_variance_protocol.json differs from the regenerated protocol")
    return committed


def protected_status() -> str:
    return e1.run_git("status", "--short", "--", *[str(p.relative_to(ROOT)) for p in PROTECTED_PATHS])


# --------------------------------------------------------------------------------------------- validity
def _qa_subset_gate(benchmark: str, text: str, amendment: dict) -> dict:
    """The frozen per-example QA gate applied to the canonical subset (first n_ref examples) of a larger run."""
    ref_text = (REFERENCE_DIR / f"{benchmark}.log").read_text(encoding="utf-8", errors="replace")
    ref, run = qg.parse_text(ref_text), qg.parse_text(text)
    n_ref = len(ref["per_example"]["rabit2"])
    key = lambda ln: ln.split(": ", 1)[1] if ln.startswith("Sample ") else ln  # noqa: E731  (drop "i/N" position)
    hdr = lambda t: [ln for ln in t.splitlines() if qg.IDENTITY_HEADER_RE.match(ln) and not ln.startswith("Samples:")]  # noqa: E731
    smp = lambda t: [key(ln) for ln in t.splitlines() if qg.IDENTITY_SAMPLE_RE.match(ln)]  # noqa: E731
    checks = {"dataset_header_identical": hdr(text) == hdr(ref_text),
              "canonical_subset_identity": smp(text)[:2 * n_ref] == smp(ref_text)}
    controls = {}
    for m in ("bf16", "rabit2"):
        thr = amendment["thresholds"][benchmark][m]
        rr, oo = ref["per_example"][m], run["per_example"][m][:n_ref]
        ok = len(oo) == n_ref and [r["dataset_index"] for r in oo] == [r["dataset_index"] for r in rr]
        c = {"rows_complete_and_ordered": ok}
        if ok:
            d = [(i, qg.milli(o["score"]), qg.milli(r["score"])) for i, (o, r) in enumerate(zip(oo, rr))]
            mism = [i for i, a, b in d if a != b]
            l1 = sum(abs(a - b) for _, a, b in d)
            c.update({"score_mismatch_count": len(mism), "score_mismatch_indices": mism, "l1_score_distance": l1 / 1000,
                      "max_allowed_mismatch_count": thr["historical_max_score_mismatch_count"],
                      "max_allowed_l1_score_distance": thr["historical_max_l1_score_distance"],
                      "text_only_answer_differences": [i for i, (o, r) in enumerate(zip(oo, rr))
                                                       if o["answer"] != r["answer"] and i not in mism]})
            c["passed"] = (len(mism) <= thr["historical_max_score_mismatch_count"]
                           and l1 <= thr["historical_max_l1_score_distance_milli"])
        else:
            c["passed"] = False
        controls[m] = c
    return {"n_canonical_subset": n_ref, "checks": checks, "controls": controls,
            "passed": all(checks.values()) and all(c["passed"] for c in controls.values())}


def _subset_aggregate(benchmark: str, rows: dict, n_ref: int, keys_ref: list) -> dict:
    out = {}
    for m in ("bf16", "rabit2"):
        by_key = {json.dumps(x["key"]): x["value"] for x in rows[m]}
        vals = [by_key[json.dumps(k)] for k in keys_ref]
        vals = [math.log(v) for v in vals] if benchmark == "continuation_ppl" else vals
        out[m] = pb.aggregate(benchmark, vals)
    return out


def integrity(name: str, rc: int, text: str, protocol: dict, amendment: dict) -> dict:
    checks = {"exit_code_zero": rc == 0, "no_traceback": "Traceback (most recent call last)" not in text}
    out = {"benchmark": name}
    metric, qi, mi = r7.ROW_COLUMNS[name]
    ci = r11.COUNT_COLUMNS[name][0]
    summary = {m: e1._last_row_tokens(text, m) for m in ("bf16", "rabit2")}
    checks["summary_rows_present"] = all(t is not None and len(t) > max(qi, mi, ci) for t in summary.values())
    if checks["summary_rows_present"]:
        checks["counts_exact"] = all(int(float(t[ci])) == EXPECTED_COUNT[name] for t in summary.values())
    rows = pb.extract(name, text)
    try:
        keys, _, _ = pb.paired_values(name, rows)
        checks["per_unit_rows_paired"] = len(keys) == SELECTION[name]["exp12_units"]
    except ValueError:
        checks["per_unit_rows_paired"] = False
        return {**out, "checks": checks, "passed": False}
    # canonical subset: identity, storage and control reproduction
    ref_rows = pb.extract(name, (REFERENCE_DIR / f"{name}.log").read_text(encoding="utf-8", errors="replace"))
    ref_keys = [x["key"] for x in ref_rows["rabit2"]]
    idx = {json.dumps(k): i for i, k in enumerate(keys)}
    checks["canonical_subset_units_present"] = all(json.dumps(k) in idx for k in ref_keys) and (
        name == "niah" or keys[:len(ref_keys)] == ref_keys)
    if not checks["canonical_subset_units_present"]:
        return {**out, "checks": checks, "passed": False}
    checks["canonical_subset_kv_mb_identical"] = all(
        rows[m][idx[json.dumps(r["key"])]]["kv_mb"] == r["kv_mb"] for m in ("bf16", "rabit2") for r in ref_rows[m])
    cfg = protocol["conditions"]["rabit2"]["config"]
    bad = []
    for m in ("bf16", "rabit2"):
        for x in rows[m]:
            t = 1024 if name == "continuation_ppl" else x["prefix_tokens"]
            exp_mb = (r7.LAYERS * 2 * t * r7.KV_HEADS * r7.HEAD_DIM * 2 if m == "bf16"
                      else r9.traced_logical_bytes(t, cfg)["total"]) / 2**20
            if abs(x["kv_mb"] - exp_mb) > STORAGE_TOL_MB + 1e-9:
                bad.append({"method": m, "key": x["key"], "observed": x["kv_mb"], "expected": exp_mb})
    checks["per_unit_kv_mb_matches_accounting"] = not bad
    out["storage_mismatches"] = bad[:20]
    if name in ("hotpotqa", "qasper"):
        gate = _qa_subset_gate(name, text, amendment)
        checks["qa_control_gate_canonical_subset"] = gate["passed"]
        out["qa_control_gate"] = gate
    else:
        agg = _subset_aggregate(name, rows, len(ref_keys), ref_keys)
        targets = protocol["validity"]["continuation_ppl_niah_passage_retrieval"]["targets"][name]
        mk = "ppl" if name == "continuation_ppl" else "accuracy_pct"
        tol = protocol["validity"]["continuation_ppl_niah_passage_retrieval"]["tolerances"][mk]
        ok = {}
        for m in ("bf16", "rabit2"):
            tgt = targets[m][mk]
            lim = tol["relative"] * abs(tgt) if "relative" in tol else tol["absolute_points"]
            ok[m] = abs(agg[m] - tgt) <= lim + 1e-9
        checks["canonical_subset_controls_reproduce"] = all(ok.values())
        out["canonical_subset_aggregate"] = {m: {"observed": agg[m], "target": targets[m][mk]} for m in agg}
    return {**out, "checks": checks, "passed": all(checks.values())}


def preflight(dry_run: bool) -> dict:
    status = protected_status()
    if status:
        raise RuntimeError("protected paths are not clean:\n" + status)
    if e1.sha256(e1.RABIT_KV2) != e1.EXPECTED_RABIT_SHA256:
        raise RuntimeError("rabit_kv2.py is not the frozen source")
    for commit, paths in ((r11.EXP10_EVIDENCE_COMMIT, r11.EXP10_FILES), (r11.AMENDMENT_COMMIT, r11.AMENDMENT_FILES),
                          (EXP11_EVIDENCE_COMMIT, EXP11_FILES), ("599d059cc3cad96f8cdf3c4f813f5460e5b35654",
                                                                 [QUALITY_DIR, REFERENCE_DIR])):
        if e1.run_git("diff", "--name-only", commit, "--", *[str(p.relative_to(ROOT)) for p in paths]):
            raise RuntimeError(f"accepted evidence differs from its frozen commit {commit[:7]}")
    if qg.audit.sha256_lf(qg.AMENDMENT) != r11.AMENDMENT_SHA256_LF:
        raise RuntimeError("QA control amendment hash does not match the pinned hash")
    amendment = qg.load_amendment()
    for b in pb.BENCHMARKS:
        text = (QUALITY_DIR / f"{b}.py").read_text(encoding="utf-8")
        if e1.REQUIRED_ALLOWED_LINE not in text or e1.REQUIRED_RABIT2_MARKER not in text:
            raise RuntimeError(f"canonical {b}.py no longer matches the accepted Experiment 1 pins")
    e1.verify_canonical_reference()
    protocol = load_protocol()
    uncommitted = e1.run_git("status", "--short", "--", *[str(p.relative_to(ROOT)) for p in (
        RUNNER_SCRIPT, HERE / "paired_bootstrap_ci.py", HERE / "test_experiment12_variance.py", PROTOCOL)])
    if uncommitted and not dry_run:
        raise RuntimeError("Refusing to run: Experiment 12 harness has uncommitted changes:\n" + uncommitted)
    if MANIFEST.exists() and json.loads(MANIFEST.read_text(encoding="utf-8")).get("status") == "passed":
        raise RuntimeError(f"{MANIFEST} already records a passed run; refusing to overwrite")
    return {"git_head": e1.run_git("rev-parse", "HEAD"), "exp11_evidence_commit": EXP11_EVIDENCE_COMMIT,
            "amendment_commit": r11.AMENDMENT_COMMIT, "amendment_sha256_lf": r11.AMENDMENT_SHA256_LF,
            "canonical_script_sha256": {b: e1.sha256(QUALITY_DIR / f"{b}.py") for b in pb.BENCHMARKS},
            "runner_script_sha256": e1.sha256(RUNNER_SCRIPT), "analysis_sha256": e1.sha256(HERE / "paired_bootstrap_ci.py"),
            "protocol_sha256": e1.sha256(PROTOCOL), "protocol": protocol, "amendment": amendment,
            "uncommitted_files": uncommitted or None}


def build_commands() -> list[dict]:
    return [{"name": r["name"], "command": [sys.executable, "-m", "modal", "run", str(r["script"]), *r["args"]],
             "log": (OUT_DIR / f"{r['name']}.log").relative_to(ROOT).as_posix()} for r in runs()]


def main(argv: list[str] | None = None) -> int:
    e1.make_console_encoding_safe()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-protocol", action="store_true", help="write exp12_variance_protocol.json (pre-commit only)")
    a = ap.parse_args(argv)
    if a.write_protocol:
        if PROTOCOL.exists():
            raise SystemExit(f"{PROTOCOL.name} already exists; the frozen protocol is never overwritten")
        PROTOCOL.write_text(json.dumps(build_protocol(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {PROTOCOL.relative_to(ROOT).as_posix()}")
        return 0
    prov = preflight(a.dry_run)
    print("RABIT-KV MLSys 2027 -- Experiment 12 larger-N paired bootstrap (logical fake-quant quality)")
    print("Preflight OK:", json.dumps({k: prov[k] for k in ("git_head", "exp11_evidence_commit", "amendment_commit",
                                                             "protocol_sha256")}))
    for c in build_commands():
        print(f"  {c['name']}: {' '.join(c['command'][3:])}")
    if prov["uncommitted_files"]:
        print("  WARNING (dry-run only): uncommitted:\n    " + prov["uncommitted_files"].replace("\n", "\n    "))
    if a.dry_run:
        print("\n--dry-run: nothing executed, no files written.")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment": "Experiment 12 -- larger samples + paired bootstrap CIs", "methods": METHODS,
                "status": "running", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "provenance": prov,
                "runs": []}
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    results = []
    for c in build_commands():
        log = ROOT / c["log"]
        rc = e1.stream_command(c["command"], log)
        res = integrity(c["name"], rc, log.read_text(encoding="utf-8", errors="replace"), prov["protocol"],
                        prov["amendment"])
        results.append(res)
        manifest["runs"].append({"name": c["name"], "returncode": rc, "passed": res["passed"],
                                 "log_sha256": e1.sha256(log)})
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not res["passed"]:  # no retry; stop at the first failing benchmark
            break
    INTEGRITY.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ok = len(results) == len(pb.BENCHMARKS) and all(r["passed"] for r in results) and not protected_status()
    if ok:  # statistics only after every benchmark is valid (offline, CPU)
        pb.main(["--results-dir", str(OUT_DIR), "--out", str(RESULTS)])
    manifest.update(status="passed" if ok else "failed", completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nEXPERIMENT 12 {manifest['status'].upper()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
