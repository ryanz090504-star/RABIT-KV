"""
RABIT-KV MLSys 2027 -- Experiment 8 runner: single-axis V-bit ablation (P0-D), LOGICAL fake-quant QUALITY only.

Question: holding K3 (seq_affine) / G32 / R4 / META8g64 fixed, how does quality change as V precision changes --
is V2 justified relative to its neighbours? (V2 is not presupposed to be best.)

Pre-registered sweep (docs/MLSYS_EXPERIMENT_PLAN.md, Experiment 8): V1, V2 (control = canonical rabit2), V3, plus the
bf16 reference row. V4 (rabit2_v4) is configured and its expected storage frozen, but it is run ONLY as the
pre-registered substitute for V1, and only if V1 fails the uniform correctness gate (NaN / Inf / degenerate codes)
-- never because of V1's quality -- and only after review.

Separate from Experiment 7's code (imported read-only; nothing in Exp7 is modified). Scripts are the deterministic
Experiment 8 copies in benchmarks/mlsys2027/exp8_vbit/ (exp8_vbit_scripts.py). Not a physical serving benchmark: no
allocator capacity, throughput or latency; KV numbers are LOGICAL packed-prefix accounting.

Usage:
    python benchmarks/mlsys2027/run_experiment8_vbit_ablation.py --write-protocol   (once, before commit)
    python benchmarks/mlsys2027/run_experiment8_vbit_ablation.py --dry-run
    python benchmarks/mlsys2027/run_experiment8_vbit_ablation.py            (NOT until explicitly authorized)
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp8_vbit_scripts as gen  # noqa: E402
import run_experiment1_quality_frontier as e1  # noqa: E402  (accepted; reused read-only)
import run_experiment7_kbit_ablation as r7  # noqa: E402  (accepted; reused read-only)

ROOT = e1.ROOT
RUNNER_SCRIPT = Path(__file__).resolve()
OUT_DIR = ROOT / "results" / "mlsys2027" / "ablations" / "v_bit"
MANIFEST = OUT_DIR / "manifest.json"
REGRESSION_CHECK = OUT_DIR / "regression_check.json"
RESULTS = OUT_DIR / "vbit_results.json"
PROTOCOL = HERE / "exp8_vbit_protocol.json"
METHODS = "bf16,rabit2_v1,rabit2,rabit2_v3"
CONDITIONS = {"rabit2_v1": 1, "rabit2": 2, "rabit2_v3": 3}  # method -> v_bits (executed sweep)
SUBSTITUTE = {"rabit2_v4": 4}  # pre-registered V1 substitute (correctness failure only; after review)
FROZEN = {"k_bits": 3, "k_style": "seq_affine", "v_style": "group_affine", "k_group": 32, "v_group": 32,
          "residual": 4, "metadata_mode": "int8", "metadata_group_size": 64}
EXP6_FROZEN_COMMIT = r7.EXP6_FROZEN_COMMIT
EXP7_EVIDENCE_COMMIT = "6ce79ed50d6b7421643deeade5dbb28442b7cb2a"
EXP7_FILES = [HERE / "exp7_kbit_scripts.py", HERE / "exp7_kbit", HERE / "run_experiment7_kbit_ablation.py",
              HERE / "test_experiment7_kbit.py", HERE / "exp7_kbit_protocol.json", r7.OUT_DIR]
PROTECTED_PATHS = [*r7.PROTECTED_PATHS, *EXP7_FILES]
# (count column index, expected per-method count) of each printed summary row
COUNT_COLUMNS = {"continuation_ppl": (6, 1024), "niah": (2, 15), "passage_retrieval": (4, 10), "hotpotqa": (4, 20),
                 "qasper": (4, 24)}
QUANTIZE_CALLS_PER_METHOD = {"continuation_ppl": 8, "niah": 15, "passage_retrieval": 10, "hotpotqa": 20, "qasper": 24}
FAILURE_MARKER = "EXP8_CORRECTNESS_FAILURE"
OK_RE = re.compile(r"^EXP8_CORRECTNESS_OK name=(.+?) layers=(\d+) k_bits=(\d+) v_bits=(\d+) nonfinite=0 ", re.M)


def derived_configs(benchmark: str, methods=None) -> dict:
    text = gen.derived_path(benchmark).read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == "config_for_method")
    ns: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<config_for_method>", "exec"), ns)  # noqa: S102
    return {m: ns["config_for_method"](m) for m in (methods or [*CONDITIONS, *SUBSTITUTE])}


def expected_logical_mb(name: str) -> dict:
    prefixes = r7.quantized_prefixes(name)
    out = {}
    for method, v in {**CONDITIONS, **SUBSTITUTE}.items():
        per = [r7.logical_kv_bytes(t, 3, v)["total"] / 2**20 for t in prefixes]
        out[method] = round(sum(per) / len(per), 3)
    out["bf16_reference"] = round(sum(r7.LAYERS * 2 * t * r7.KV_HEADS * r7.HEAD_DIM * 2 / 2**20
                                      for t in prefixes) / len(prefixes), 3)
    return out


def runs() -> list[dict]:
    out = []
    for spec in e1.RUNS:
        args = list(spec["args"])
        args[args.index("--methods") + 1] = METHODS
        out.append({"name": spec["name"], "script": gen.derived_path(spec["name"]), "args": args,
                    "purpose": spec["purpose"].replace("bf16/rabit8/rabit4/rabit3/rabit2 frontier",
                                                       "V-bit ablation").replace("full frontier", "V-bit ablation")})
    return out


def build_protocol() -> dict:
    cfgs = {b: derived_configs(b) for b in gen.BENCHMARKS}
    first = cfgs[gen.BENCHMARKS[0]]
    if any(cfgs[b] != first for b in gen.BENCHMARKS):
        raise RuntimeError("condition configs differ between benchmark scripts")
    control = first["rabit2"]
    proof = {m: sorted(k for k in set(c) | set(control) if c.get(k) != control.get(k) and k != "name")
             for m, c in first.items()}
    ref = e1.CANONICAL_REFERENCE
    metric_key = {"continuation_ppl": "ppl", "niah": "accuracy_pct", "passage_retrieval": "accuracy_pct",
                  "hotpotqa": "f1_pct", "qasper": "f1_pct"}
    shorthand = lambda c: (f"K{c['k_bits']}/V{c['v_bits']}/G{c['k_group']}/R{c['residual']}/"  # noqa: E731
                           f"META8g{c['metadata_group_size']}")
    return {
        "experiment": 8, "type": "logical fake-quant / dequant QUALITY ablation (NOT a physical serving benchmark)",
        "axis": "V bits", "not_reported": ["physical allocator capacity", "throughput", "latency"],
        "question": ("Holding K3 / G32 / R4 / META8g64 fixed, how does quality change as V precision changes? In "
                     "particular, is V=2 justified relative to neighbouring V precisions? V2 is not presupposed best."),
        "methods_argument": METHODS,
        "conditions": {"bf16": {"role": "reference (uncompressed); not an ablation condition"},
                       **{m: {"role": "control" if m == "rabit2" else "treatment", "shorthand": shorthand(c), "config": c}
                          for m, c in first.items() if m in CONDITIONS}},
        "pre_registered_substitute": {
            "rabit2_v4": {"shorthand": shorthand(first["rabit2_v4"]), "config": first["rabit2_v4"]},
            "rule": ("V4 replaces V1 ONLY if V1 fails the uniform correctness gate (crash inside the check, NaN / Inf in "
                     "the dequantized KV, out-of-range or all-identical codes) -- identified before any quality metric "
                     "is examined. V1's quality, however poor, is never grounds for substitution. A substitution is "
                     "not automatic: the runner stops and reports the failure mode for review; the substitution is "
                     "then reported explicitly."),
            "source": "docs/MLSYS_EXPERIMENT_PLAN.md, Experiment 8 pre-registration"},
        "only_v_bits_differs_proof": {
            "control": "rabit2", "fields_differing_from_control_excluding_display_name": proof,
            "holds": all(v == ([] if m == "rabit2" else ["v_bits"]) for m, v in proof.items()),
            "configs_identical_across_all_five_scripts": True},
        "correctness_gate": {
            "where": "inside every derived script: per layer right after quantization (codes) and after "
                     "dequantization, before scoring (finite KV)",
            "applies_to": "every quantized method (V1, V2 control, V3; V4 if ever run) -- uniformly; bf16 is never quantized",
            "checks": ["quantized codes within [0, 2^bits - 1] (K and V)",
                       "quantized codes not all identical within a layer's K or V state (degeneracy)",
                       "every dequantized K / V tensor finite (no NaN / Inf)"],
            "observation_only": "reads tensors; never modifies them",
            "evidence": "one EXP8_CORRECTNESS_OK line per quantize call; EXP8_CORRECTNESS_FAILURE raises",
            "expected_ok_lines_per_method": QUANTIZE_CALLS_PER_METHOD},
        "common": r7.COMMON_FACTS,
        "benchmarks": {b: {**r7.BENCHMARK_FACTS[b], "script": gen.derived_path(b).relative_to(ROOT).as_posix(),
                           "canonical_source": gen.canonical_path(b).relative_to(ROOT).as_posix(),
                           "canonical_source_sha256_lf": gen.g7.sha256_text(gen.canonical_path(b).read_text(encoding="utf-8")),
                           "args": next(r["args"] for r in runs() if r["name"] == b)} for b in gen.BENCHMARKS},
        "control_reproduction": {
            "configuration_equality": "EXACT: the V2 control is the canonical rabit2 config (compiled and compared)",
            "numerical_reproduction": "within the frozen tolerances below (copied from the accepted Experiment 1 "
                                      "methodology); applied to bf16 and the V2 control; never changed after the run starts",
            "tolerances": {"continuation_ppl.ppl": {"relative": e1.PPL_RELATIVE_TOLERANCE},
                           "niah.accuracy_pct": {"absolute_points": e1.PERCENTAGE_ABSOLUTE_TOLERANCE},
                           "passage_retrieval.accuracy_pct": {"absolute_points": e1.PERCENTAGE_ABSOLUTE_TOLERANCE},
                           "hotpotqa.f1_pct": {"absolute_points": e1.PERCENTAGE_ABSOLUTE_TOLERANCE},
                           "qasper.f1_pct": {"absolute_points": e1.PERCENTAGE_ABSOLUTE_TOLERANCE},
                           "avg_logical_kv_mb": {"relative": e1.KV_MB_RELATIVE_TOLERANCE}},
            "single_example_flip": {"niah": "1 of 15 cases = 6.67 points > 1.0 -> fails",
                                    "passage_retrieval": "1 of 10 samples = up to 10.0 points > 1.0 -> fails",
                                    "hotpotqa": "one example's F1 moves the mean by up to 5.0 points (1/20)",
                                    "qasper": "one example's F1 moves the mean by up to 4.17 points (1/24)",
                                    "rule": "1.0-point tolerance is a tight drift detector, identical to Exp1 / Exp7"},
            "canonical_targets": {b: {"bf16": {metric_key[b]: ref[b]["bf16"][metric_key[b]],
                                               "avg_logical_kv_mb": ref[b]["bf16"]["avg_kv_mb"]},
                                      "rabit2_V2": {metric_key[b]: ref[b]["rabit2"][metric_key[b]],
                                                    "avg_logical_kv_mb": ref[b]["rabit2"]["avg_kv_mb"]}}
                                  for b in gen.BENCHMARKS},
            "target_source": "results/summary.json and results/quality/*.log (canonical), pinned in the accepted "
                             "Experiment 1 runner"},
        "logical_storage_expectations": {
            "label": "LOGICAL packed prefix-KV storage (payload bits + uint8-group metadata + BF16 residual); NOT "
                     "physical allocator capacity",
            "formula": ("identical to Experiment 7's frozen formula with k_bits = 3 and v_bits varied: per layer "
                        "V payload = ceil(8*Lq*128*v_bits/8) (Lq = prefix - 4, V grouped along head_dim so no sequence "
                        "padding); K payload, all metadata and the BF16 residual do not depend on v_bits"),
            "only_v_payload_depends_on_v_bits": True,
            "expected_avg_logical_kv_mb": {b: expected_logical_mb(b) for b in gen.BENCHMARKS},
            "gates": {"strictly_increasing": "V1 < V2 < V3",
                      "linearity": {"rule": "|(V3 - V2) - (V2 - V1)| <= tolerance",
                                    "tolerance_mb": r7.LINEARITY_ABS_TOL_MB},
                      "matches_expected": {"rule": "observed V1 / V2 / V3 avg logical KV MB within the relative "
                                                   "tolerance of expected_avg_logical_kv_mb",
                                           "relative": e1.KV_MB_RELATIVE_TOLERANCE}}},
        "execution_gates": ["exit code 0 and no traceback", "all four rows (bf16, V1, V2, V3) present",
                            "per-method counts equal the frozen counts",
                            "EXP8_CORRECTNESS_OK lines == expected quantize calls for V1, V2, V3; no failure marker",
                            "bf16 and V2 control within the frozen tolerances of the canonical targets",
                            "logical storage gates above",
                            "protected paths clean; Exp6 identical to 0f5f6ef; Exp7 identical to 6ce79ed",
                            "stop at the first failing benchmark; no retry"],
        "artifacts": [f"results/mlsys2027/ablations/v_bit/{b}.log" for b in gen.BENCHMARKS]
                     + ["results/mlsys2027/ablations/v_bit/manifest.json",
                        "results/mlsys2027/ablations/v_bit/regression_check.json",
                        "results/mlsys2027/ablations/v_bit/vbit_results.json"],
    }


def load_protocol() -> dict:
    committed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if committed != json.loads(json.dumps(build_protocol())):
        raise RuntimeError("exp8_vbit_protocol.json differs from the regenerated protocol")
    if not committed["only_v_bits_differs_proof"]["holds"]:
        raise RuntimeError("protocol does not prove that conditions differ only in v_bits")
    return committed


def protected_status() -> str:
    return e1.run_git("status", "--short", "--", *[str(p.relative_to(ROOT)) for p in PROTECTED_PATHS])


def preflight(dry_run: bool) -> dict:
    status = protected_status()
    if status:
        raise RuntimeError("protected paths are not clean:\n" + status)
    if e1.sha256(e1.RABIT_KV2) != e1.EXPECTED_RABIT_SHA256:
        raise RuntimeError("rabit_kv2.py is not the frozen source")
    for commit, paths in ((EXP6_FROZEN_COMMIT, [r7.EXP6_DIR]), (EXP7_EVIDENCE_COMMIT, EXP7_FILES)):
        if e1.run_git("diff", "--name-only", commit, "--", *[str(p.relative_to(ROOT)) for p in paths]):
            raise RuntimeError(f"accepted evidence differs from its frozen commit {commit[:7]}")
    ok = gen.check()
    if not all(ok.values()):
        raise RuntimeError(f"Exp8 derived scripts differ from their derivation: {ok}")
    for b in gen.BENCHMARKS:
        text = gen.canonical_path(b).read_text(encoding="utf-8")
        if e1.REQUIRED_ALLOWED_LINE not in text or e1.REQUIRED_RABIT2_MARKER not in text:
            raise RuntimeError(f"canonical {b}.py no longer matches the accepted Experiment 1 pins")
    e1.verify_canonical_reference()
    protocol = load_protocol()
    uncommitted = e1.run_git("status", "--short", "--", *[str(p.relative_to(ROOT)) for p in (
        RUNNER_SCRIPT, gen.DERIVED_DIR, HERE / "exp8_vbit_scripts.py", HERE / "test_experiment8_vbit.py", PROTOCOL)])
    if uncommitted and not dry_run:
        raise RuntimeError("Refusing to run: Experiment 8 harness has uncommitted changes:\n" + uncommitted)
    if MANIFEST.exists() and json.loads(MANIFEST.read_text(encoding="utf-8")).get("status") == "passed":
        raise RuntimeError(f"{MANIFEST} already records a passed run; refusing to overwrite")
    return {"git_head": e1.run_git("rev-parse", "HEAD"), "rabit_kv2_sha256": e1.EXPECTED_RABIT_SHA256,
            "exp6_frozen_commit": EXP6_FROZEN_COMMIT, "exp7_evidence_commit": EXP7_EVIDENCE_COMMIT,
            "canonical_script_sha256": {b: e1.sha256(gen.canonical_path(b)) for b in gen.BENCHMARKS},
            "derived_script_sha256": {b: e1.sha256(gen.derived_path(b)) for b in gen.BENCHMARKS},
            "runner_script_sha256": e1.sha256(RUNNER_SCRIPT), "protocol_sha256": e1.sha256(PROTOCOL),
            "protocol": protocol, "uncommitted_files": uncommitted or None}


def build_commands() -> list[dict]:
    return [{"name": r["name"], "purpose": r["purpose"],
             "command": [sys.executable, "-m", "modal", "run", str(r["script"]), *r["args"]],
             "log": (OUT_DIR / f"{r['name']}.log").relative_to(ROOT).as_posix()} for r in runs()]


def parse_rows(name: str, log_text: str) -> dict:
    metric, qi, mi = r7.ROW_COLUMNS[name]
    ci = COUNT_COLUMNS[name][0]
    out = {}
    for method in ("bf16", *CONDITIONS):
        tokens = e1._last_row_tokens(log_text, method)
        out[method] = None if tokens is None or len(tokens) <= max(qi, mi, ci) else {
            metric: float(tokens[qi]), "avg_logical_kv_mb": float(tokens[mi]), "count": int(float(tokens[ci]))}
    return out


def correctness_counts(log_text: str, names: dict) -> dict:
    counts = {m: 0 for m in names}
    by_name = {v: k for k, v in names.items()}
    for name, layers, kb, vb in OK_RE.findall(log_text):
        m = by_name.get(name)
        if m is not None and int(layers) == r7.LAYERS and int(kb) == 3 and int(vb) == CONDITIONS[m]:
            counts[m] += 1
    return counts


def integrity(name: str, rc: int, log_text: str, protocol: dict | None = None) -> dict:
    protocol = protocol or load_protocol()
    rows = parse_rows(name, log_text)
    names = {m: protocol["conditions"][m]["config"]["name"] for m in CONDITIONS}
    failure_lines = [ln for ln in log_text.splitlines() if FAILURE_MARKER in ln]
    checks = {"exit_code_zero": rc == 0, "no_traceback": "Traceback (most recent call last)" not in log_text,
              "no_correctness_failure": not failure_lines, "all_four_rows_present": all(rows.values())}
    ok_counts = correctness_counts(log_text, names)
    checks["correctness_ok_lines_complete"] = all(c == QUANTIZE_CALLS_PER_METHOD[name] for c in ok_counts.values())
    if checks["all_four_rows_present"]:
        checks["counts_exact"] = all(r["count"] == COUNT_COLUMNS[name][1] for r in rows.values())
        v1, v2, v3 = (rows[m]["avg_logical_kv_mb"] for m in ("rabit2_v1", "rabit2", "rabit2_v3"))
        gates = protocol["logical_storage_expectations"]["gates"]
        checks["kv_mb_strictly_increasing_in_v_bits"] = v1 < v2 < v3
        checks["kv_mb_linear_in_v_bits"] = abs((v3 - v2) - (v2 - v1)) <= gates["linearity"]["tolerance_mb"]
        exp = protocol["logical_storage_expectations"]["expected_avg_logical_kv_mb"][name]
        rel = gates["matches_expected"]["relative"]
        checks["kv_mb_matches_expected"] = all(
            abs(rows[m]["avg_logical_kv_mb"] - exp[m]) <= max(abs(exp[m]) * rel, 1e-9) for m in CONDITIONS)
    reg = e1.check_regression(name, log_text)  # bf16 + V2 control vs the canonical reference
    checks["bf16_and_v2_control_reproduce_canonical"] = reg["all_within_tolerance"]
    failed_method = None
    for ln in failure_lines:
        for m, n in names.items():
            if f"name={n} " in ln:
                failed_method = failed_method or m
    classification = ("passed" if all(checks.values()) else
                      "v1_correctness_failure_substitution_requires_review" if failed_method == "rabit2_v1" else
                      "correctness_failure" if failure_lines else "failed")
    return {"benchmark": name, "rows": rows, "correctness_ok_lines": ok_counts, "correctness_failures": failure_lines,
            "checks": checks, "regression": reg, "classification": classification,
            "passed": classification == "passed"}


def main(argv: list[str] | None = None) -> int:
    e1.make_console_encoding_safe()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-protocol", action="store_true", help="write exp8_vbit_protocol.json (pre-commit only)")
    a = ap.parse_args(argv)
    if a.write_protocol:
        if PROTOCOL.exists():
            raise SystemExit(f"{PROTOCOL.name} already exists; the frozen protocol is never overwritten")
        PROTOCOL.write_text(json.dumps(build_protocol(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {PROTOCOL.relative_to(ROOT).as_posix()}")
        return 0
    prov = preflight(a.dry_run)
    print("RABIT-KV MLSys 2027 -- Experiment 8 V-bit ablation (logical fake-quant quality)")
    print("Preflight OK:", json.dumps({k: prov[k] for k in ("git_head", "exp6_frozen_commit", "exp7_evidence_commit",
                                                             "protocol_sha256")}))
    for c in build_commands():
        print(f"  {c['name']}: {' '.join(c['command'][3:])}")
    if prov["uncommitted_files"]:
        print("  WARNING (dry-run only): uncommitted:\n    " + prov["uncommitted_files"].replace("\n", "\n    "))
    if a.dry_run:
        print("\n--dry-run: nothing executed, no files written.")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment": "Experiment 8 -- V-bit ablation (logical fake-quant quality)", "methods": METHODS,
                "conditions_v_bits": CONDITIONS, "frozen": FROZEN, "status": "running",
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "provenance": prov, "runs": []}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    results = []
    for c in build_commands():
        log = ROOT / c["log"]
        rc = e1.stream_command(c["command"], log)
        res = integrity(c["name"], rc, log.read_text(encoding="utf-8", errors="replace"), prov["protocol"])
        results.append(res)
        manifest["runs"].append({"name": c["name"], "returncode": rc, "classification": res["classification"],
                                 "log_sha256": e1.sha256(log)})
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if not res["passed"]:  # no retry, no automatic V4 substitution; stop for review
            break
    REGRESSION_CHECK.write_text(json.dumps([r["regression"] for r in results], indent=2) + "\n", encoding="utf-8")
    RESULTS.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    ok = len(results) == len(e1.RUNS) and all(r["passed"] for r in results) and not protected_status()
    manifest.update(status="passed" if ok else results[-1]["classification"] if results else "failed",
                    completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nEXPERIMENT 8 {manifest['status'].upper()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
