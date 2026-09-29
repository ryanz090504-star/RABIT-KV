"""
RABIT-KV MLSys 2027 -- Experiment 10 runner: single-axis residual-window ablation, LOGICAL fake-quant QUALITY only.

Question: holding K3 (seq_affine) / V2 (group_affine) / G32 / META8g64 fixed, how does changing the recent BF16
residual window R affect quality and logical storage -- is R4 a favourable quality / storage operating point? (R4 is
not presupposed best.)

Pre-registered sweep (docs/MLSYS_EXPERIMENT_PLAN.md, Experiment 10): R0, R2, R4 (control = canonical rabit2), R8, plus
the bf16 reference row. In the canonical code R (config["residual"]) is read only by q_with_residual: for every layer,
the prefix K tensor and -- in a separate, identical call -- the prefix V tensor, each [1, 8, T, 128], keep their newest
min(R, T) token positions in BF16 and quantize the older T - R positions; R = 0 quantizes all T positions; T <= R keeps
the whole tensor BF16. Quantization happens once, on the prefill prefix; tokens appended afterwards (the final prompt
token, generated tokens, teacher-forced continuation tokens) stay BF16 for every method and are not counted.

Separate from Experiments 7 / 8 / 9 (their code is imported read-only; nothing in them is modified). Scripts are the
deterministic Experiment 10 copies in benchmarks/mlsys2027/exp10_residual/ (exp10_residual_scripts.py). Not a physical
serving benchmark: no allocator capacity, throughput or latency; KV numbers are LOGICAL packed-prefix accounting.

Usage:
    python benchmarks/mlsys2027/run_experiment10_residual_ablation.py --write-protocol   (once, before commit)
    python benchmarks/mlsys2027/run_experiment10_residual_ablation.py --dry-run
    python benchmarks/mlsys2027/run_experiment10_residual_ablation.py            (NOT until explicitly authorized)
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp10_residual_scripts as gen  # noqa: E402
import run_experiment1_quality_frontier as e1  # noqa: E402  (accepted; reused read-only)
import run_experiment7_kbit_ablation as r7  # noqa: E402  (accepted; reused read-only)
import run_experiment8_vbit_ablation as r8  # noqa: E402  (accepted; reused read-only)
import run_experiment9_group_ablation as r9  # noqa: E402  (accepted; reused read-only: storage model, paths)

ROOT = e1.ROOT
RUNNER_SCRIPT = Path(__file__).resolve()
OUT_DIR = ROOT / "results" / "mlsys2027" / "ablations" / "residual_window"
MANIFEST = OUT_DIR / "manifest.json"
REGRESSION_CHECK = OUT_DIR / "regression_check.json"
RESULTS = OUT_DIR / "residual_results.json"
PROTOCOL = HERE / "exp10_residual_protocol.json"
METHODS = "bf16,rabit2_r0,rabit2_r2,rabit2,rabit2_r8"
CONDITIONS = {"rabit2_r0": 0, "rabit2_r2": 2, "rabit2": 4, "rabit2_r8": 8}  # method -> residual R
FROZEN = {"k_bits": 3, "v_bits": 2, "k_style": "seq_affine", "v_style": "group_affine", "k_group": 32, "v_group": 32,
          "metadata_mode": "int8", "metadata_group_size": 64}
EXP6_FROZEN_COMMIT = r7.EXP6_FROZEN_COMMIT
EXP7_EVIDENCE_COMMIT = r8.EXP7_EVIDENCE_COMMIT
EXP8_EVIDENCE_COMMIT = r9.EXP8_EVIDENCE_COMMIT
EXP9_EVIDENCE_COMMIT = "599d059cc3cad96f8cdf3c4f813f5460e5b35654"
EXP9_FILES = [HERE / "exp9_group_scripts.py", HERE / "exp9_group", HERE / "run_experiment9_group_ablation.py",
              HERE / "test_experiment9_group.py", HERE / "exp9_group_protocol.json", r9.OUT_DIR]
PROTECTED_PATHS = [*r9.PROTECTED_PATHS, *EXP9_FILES]
COUNT_COLUMNS = r8.COUNT_COLUMNS  # (count column index, expected per-method count); identical workloads
# Accounting-integrity gate (not a statistical reproduction tolerance): logical accounting is deterministic, so each
# condition's printed avg KV MB (3 decimals -- the only precision the canonical scripts log) must lie within 0.001 MB
# of the FULL-PRECISION expected value (exact integer byte counts, averaged, / 2^20, unrounded). For the
# larger-context benchmarks the Exp1 0.1 % total-storage tolerance is comparable to or wider than the storage change
# from a two-token residual step, so Exp10 uses this tighter gate; the 0.1 % tolerance is kept unchanged for
# control reproduction.
STORAGE_MATCH_ABS_TOL_MB = 0.001

traced_logical_bytes = r9.traced_logical_bytes  # shape-only trace of the canonical accounting (Exp9, accepted)


def derived_configs(benchmark: str, methods=None) -> dict:
    text = gen.derived_path(benchmark).read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == "config_for_method")
    ns: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<config_for_method>", "exec"), ns)  # noqa: S102
    return {m: ns["config_for_method"](m) for m in (methods or CONDITIONS)}


def _avg_mb(name: str, config: dict, key: str = "total") -> float:
    prefixes = r7.quantized_prefixes(name)
    return round(sum(traced_logical_bytes(t, config)[key] for t in prefixes) / len(prefixes) / 2**20, 3)


def expected_logical_mb(name: str, configs: dict) -> dict:
    out = {m: _avg_mb(name, configs[m]) for m in CONDITIONS}
    prefixes = r7.quantized_prefixes(name)
    out["bf16_reference"] = round(sum(r7.LAYERS * 2 * t * r7.KV_HEADS * r7.HEAD_DIM * 2 / 2**20
                                      for t in prefixes) / len(prefixes), 3)
    return out


def expected_bytes(name: str, config: dict) -> list[int]:
    """Exact integer logical byte count of every quantized prefix of the benchmark (one per sample / case)."""
    return [traced_logical_bytes(t, config)["total"] for t in r7.quantized_prefixes(name)]


def expected_full_precision_mb(name: str, configs: dict) -> dict:
    """Unrounded average logical KV MB from the exact integer byte counts (what the storage gate compares against)."""
    out = {}
    for m in CONDITIONS:
        per = expected_bytes(name, configs[m])
        out[m] = sum(per) / len(per) / 2**20
    return out


def storage_breakdown(name: str, configs: dict) -> dict:
    return {m: {k: _avg_mb(name, configs[m], k) for k in ("k_payload", "k_meta", "v_payload", "v_meta", "residual")}
            for m in CONDITIONS}


def expected_order(exp: dict) -> list[str]:
    """Conditions ordered by expected logical KV MB, smallest first (derived from the formula, not assumed)."""
    if len({exp[m] for m in CONDITIONS}) != len(CONDITIONS):
        raise RuntimeError("expected logical KV MB not strictly ordered")
    return sorted(CONDITIONS, key=lambda m: exp[m])


def runs() -> list[dict]:
    out = []
    for spec in e1.RUNS:
        args = list(spec["args"])
        args[args.index("--methods") + 1] = METHODS
        out.append({"name": spec["name"], "script": gen.derived_path(spec["name"]), "args": args,
                    "purpose": spec["purpose"].replace("bf16/rabit8/rabit4/rabit3/rabit2 frontier",
                                                       "residual-window ablation").replace(
                                                           "full frontier", "residual-window ablation")})
    return out


def build_protocol() -> dict:
    cfgs = {b: derived_configs(b) for b in gen.BENCHMARKS}
    first = cfgs[gen.BENCHMARKS[0]]
    if any(cfgs[b] != first for b in gen.BENCHMARKS):
        raise RuntimeError("condition configs differ between benchmark scripts")
    control = first["rabit2"]
    proof = {m: sorted(k for k in set(c) | set(control) if c.get(k) != control.get(k) and k != "name")
             for m, c in first.items()}
    anchors = r9.model_anchors()
    if any(a["model_mb"] != a["exp1_observed_mb"] for b in anchors.values() for a in b.values()):
        raise RuntimeError("storage model does not reproduce the observed Exp1 preset MB")
    expected = {b: expected_logical_mb(b, first) for b in gen.BENCHMARKS}
    ref = e1.CANONICAL_REFERENCE
    metric_key = {"continuation_ppl": "ppl", "niah": "accuracy_pct", "passage_retrieval": "accuracy_pct",
                  "hotpotqa": "f1_pct", "qasper": "f1_pct"}
    shorthand = lambda c: (f"K{c['k_bits']}/V{c['v_bits']}/G{c['k_group']}/R{c['residual']}/"  # noqa: E731
                           f"META8g{c['metadata_group_size']}")
    return {
        "experiment": 10, "type": "logical fake-quant / dequant QUALITY ablation (NOT a physical serving benchmark)",
        "axis": "recent BF16 residual window R (config field: residual)",
        "not_reported": ["physical allocator capacity", "throughput", "latency"],
        "question": ("Holding K3 / V2 / G32 / META8g64 fixed, how does changing the recent BF16 residual window affect "
                     "quality and logical storage? Is R4 a favourable quality / storage operating point? R4 is not "
                     "presupposed best."),
        "source": "docs/MLSYS_EXPERIMENT_PLAN.md, Experiment 10 (control rabit2 R4; treatment R0, R2, R8; new "
                  "config_for_method entries varying only residual)",
        "plan_note": ("the plan's hypothesis text mentions 'R=0, R=1' as low values, but its Treatment list is R0, R2, "
                      "R8; the explicit Treatment list is used; R1 is not run"),
        "methods_argument": METHODS,
        "residual_definition": {
            "counts": "token (sequence) positions of the prefill prefix KV cache, per layer; not bytes, not heads",
            "applied": ("q_with_residual is called separately for K (k_bits) and for V (v_bits) of every layer with "
                        "the same config['residual'], so the SAME R applies independently to K and to V"),
            "which_tokens": "the NEWEST min(R, T) positions of the prefix stay BF16 for both K and V; the older T - R "
                            "positions are quantized (K: seq_affine groups of 32 tokens over the older part only; "
                            "V: group_affine along head_dim)",
            "sequence_length": ("T = quantized prefix length (continuation_ppl: 1024 context tokens; NIAH / LongBench: "
                                "prompt length - 1). R is a fixed token count, independent of T; K's sequence groups "
                                "start at position 0 of the older part and pad Lq = T - R up to a multiple of 32"),
            "r_zero": "residual <= 0: the whole prefix (all T positions) is quantized; no BF16 residual",
            "t_le_r": "T <= R: the whole tensor stays BF16 (never reached here: min T = 1024 > 8)",
            "after_quantization": ("quantization is applied once to the prefill prefix; the final prompt token, "
                                   "generated tokens and teacher-forced continuation tokens are appended by the model "
                                   "in BF16 for EVERY method (including R0) and are not counted in logical KV MB"),
            "read_only_in": "q_with_residual (config['residual']); no other canonical code reads residual"},
        "conditions": {"bf16": {"role": "reference (uncompressed); not an ablation condition"},
                       **{m: {"role": "control" if m == "rabit2" else "treatment", "shorthand": shorthand(c), "config": c}
                          for m, c in first.items()}},
        "only_residual_differs_proof": {
            "control": "rabit2", "fields_differing_from_control_excluding_display_name": proof,
            "holds": all(v == ([] if m == "rabit2" else ["residual"]) for m, v in proof.items())
                     and all(c["residual"] == CONDITIONS[m] for m, c in first.items()),
            "frozen_fields": FROZEN,
            "configs_identical_across_all_five_scripts": True},
        "correctness_gate": ("none injected: the plan specifies code changes for Experiment 10 as config_for_method "
                             "entries only; execution gates below apply to every condition"),
        "common": r7.COMMON_FACTS,
        "benchmarks": {b: {**r7.BENCHMARK_FACTS[b], "script": gen.derived_path(b).relative_to(ROOT).as_posix(),
                           "canonical_source": gen.canonical_path(b).relative_to(ROOT).as_posix(),
                           "canonical_source_sha256_lf": gen.g7.sha256_text(gen.canonical_path(b).read_text(encoding="utf-8")),
                           "args": next(r["args"] for r in runs() if r["name"] == b),
                           "expected_count_per_method": COUNT_COLUMNS[b][1],
                           "min_quantized_prefix_tokens": min(r7.quantized_prefixes(b))} for b in gen.BENCHMARKS},
        "control_reproduction": {
            "configuration_equality": "EXACT: the R4 control is the canonical rabit2 config (compiled and compared)",
            "numerical_reproduction": "within the frozen tolerances below (copied from the accepted Experiment 1 "
                                      "methodology); applied to bf16 and the R4 control; never changed after the run starts",
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
                                    "rule": "1.0-point tolerance is a tight drift detector, identical to Exp1 / Exp7-9"},
            "observed_control_variation": ("identical-config rabit2 runs: Qasper F1 35.6 (canonical) / 36.2 (Exp7) / "
                                           "34.9 (Exp8) / 35.6 (Exp9); HotpotQA F1 55.2 except Exp9 attempt 1 (57.7, "
                                           "one answer changed; invalid, excluded) -- generation is not bit-exact "
                                           "across fresh runs"),
            "canonical_targets": {b: {"bf16": {metric_key[b]: ref[b]["bf16"][metric_key[b]],
                                               "avg_logical_kv_mb": ref[b]["bf16"]["avg_kv_mb"]},
                                      "rabit2_R4": {metric_key[b]: ref[b]["rabit2"][metric_key[b]],
                                                    "avg_logical_kv_mb": ref[b]["rabit2"]["avg_kv_mb"]}}
                                  for b in gen.BENCHMARKS},
            "target_source": "results/summary.json and results/quality/*.log (canonical), pinned in the accepted "
                             "Experiment 1 runner"},
        "logical_storage_expectations": {
            "label": "LOGICAL packed prefix-KV storage (payload bits + uint8-group metadata + BF16 residual); NOT "
                     "physical allocator capacity",
            "formula": ("shape-only trace of the canonical accounting (the accepted Experiment 9 model), per layer (32 "
                        "layers, 8 KV heads, head_dim 128), quantized prefix T, residual R, Lq = T - R, "
                        "Lk = ceil(Lq/32)*32: K payload = ceil(8*Lk*128*3/8); K metadata = 2 * meta(8*(Lk/32)*128); "
                        "V payload = ceil(8*Lq*128*2/8); V metadata = 2 * meta(8*Lq*(128/32)); "
                        "BF16 residual = 2 (K, V) * R * 8 * 128 * 2 bytes; "
                        "meta(n) = ceil(n/64)*64 + 4*ceil(n/64). Average over the benchmark's quantized prefixes, "
                        "/ 2^20, rounded to 3 decimals like the scripts' print."),
            "bf16_residual_contribution": {
                "bytes_per_residual_token_all_layers": r7.LAYERS * 2 * r7.KV_HEADS * r7.HEAD_DIM * 2,
                "mib_per_residual_token_all_layers": r7.LAYERS * 2 * r7.KV_HEADS * r7.HEAD_DIM * 2 / 2**20,
                "by_condition_mib": {m: CONDITIONS[m] * r7.LAYERS * 2 * r7.KV_HEADS * r7.HEAD_DIM * 2 / 2**20
                                     for m in CONDITIONS},
                "note": "exactly linear in R: 0.125 MiB per residual token (K + V, 32 layers, BF16)"},
            "r_dependence": ("R moves (a) the BF16 residual term (+0.125 MiB per token, linear) and (b) the quantized "
                             "part through Lq = T - R: V payload and V metadata shrink per token removed from the "
                             "quantized part (~324 B per token per layer), while K payload / metadata change only when "
                             "Lk = ceil(Lq/32)*32 crosses a 32-token boundary (a whole 32-token K group). The net "
                             "total is therefore NOT linear in R (e.g. HotpotQA R8 - R4 = 0.394 MB vs 0.460 MB on "
                             "NIAH); no linearity gate is used"),
            "model_anchors": {"rule": "the model reproduces the MB printed by the canonical Exp1 presets exactly "
                                      "(rabit8 R0, rabit4 R0, rabit3 R2, rabit2 R4 -- including two residual values)",
                              "values": anchors},
            "expected_avg_logical_kv_mb": expected,
            "expected_avg_logical_kv_mb_full_precision": {b: expected_full_precision_mb(b, first)
                                                          for b in gen.BENCHMARKS},
            "expected_total_logical_bytes": {b: {m: sum(expected_bytes(b, first[m])) for m in CONDITIONS}
                                             for b in gen.BENCHMARKS},
            "expected_delta_vs_r4_mb": {b: {m: round(expected[b][m] - expected[b]["rabit2"], 3) for m in CONDITIONS}
                                        for b in gen.BENCHMARKS},
            "expected_compression_vs_bf16": {b: {m: round(expected[b]["bf16_reference"] / expected[b][m], 3)
                                                 for m in CONDITIONS} for b in gen.BENCHMARKS},
            "expected_breakdown_avg_mb": {b: storage_breakdown(b, first) for b in gen.BENCHMARKS},
            "gates": {"matches_expected": {"rule": "ACCOUNTING-INTEGRITY gate (not a statistical reproduction "
                                                   "tolerance): each observed R0 / R2 / R4 / R8 avg logical KV MB, as "
                                                   "printed by the canonical script (3 decimals -- the only precision "
                                                   "it logs), within absolute_mb of "
                                                   "expected_avg_logical_kv_mb_full_precision (exact integer byte "
                                                   "counts averaged / 2^20, unrounded; the 3-decimal "
                                                   "expected_avg_logical_kv_mb is for display only). The accounting "
                                                   "is deterministic (Exp7 / 8 / 9 matched exactly). For the "
                                                   "larger-context benchmarks the old 0.1 % total-storage tolerance "
                                                   "is comparable to or wider than the storage change from a "
                                                   "two-token residual step, so Exp10 uses this tighter deterministic "
                                                   "accounting-integrity gate",
                                           "absolute_mb": STORAGE_MATCH_ABS_TOL_MB},
                      "ordering": {"rule": "observed conditions ordered (smallest MB first) exactly as the formula "
                                           "orders them",
                                   "expected_order_smallest_first": {b: expected_order(expected[b])
                                                                     for b in gen.BENCHMARKS}}}},
        "execution_gates": ["exit code 0 and no traceback", "all five rows (bf16, R0, R2, R4, R8) present",
                            "per-method counts equal the frozen counts",
                            "bf16 and R4 control within the frozen tolerances of the canonical targets",
                            "logical storage gates above",
                            "protected paths clean; Exp6 identical to 0f5f6ef; Exp7 to 6ce79ed; Exp8 to e13a9b6; "
                            "Exp9 to 599d059",
                            "stop at the first failing benchmark; no retry"],
        "artifacts": [f"results/mlsys2027/ablations/residual_window/{b}.log" for b in gen.BENCHMARKS]
                     + ["results/mlsys2027/ablations/residual_window/manifest.json",
                        "results/mlsys2027/ablations/residual_window/regression_check.json",
                        "results/mlsys2027/ablations/residual_window/residual_results.json"],
    }


def load_protocol() -> dict:
    committed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if committed != json.loads(json.dumps(build_protocol())):
        raise RuntimeError("exp10_residual_protocol.json differs from the regenerated protocol")
    if not committed["only_residual_differs_proof"]["holds"]:
        raise RuntimeError("protocol does not prove that conditions differ only in residual")
    return committed


def protected_status() -> str:
    return e1.run_git("status", "--short", "--", *[str(p.relative_to(ROOT)) for p in PROTECTED_PATHS])


def preflight(dry_run: bool) -> dict:
    status = protected_status()
    if status:
        raise RuntimeError("protected paths are not clean:\n" + status)
    if e1.sha256(e1.RABIT_KV2) != e1.EXPECTED_RABIT_SHA256:
        raise RuntimeError("rabit_kv2.py is not the frozen source")
    for commit, paths in ((EXP6_FROZEN_COMMIT, [r7.EXP6_DIR]), (EXP7_EVIDENCE_COMMIT, r8.EXP7_FILES),
                          (EXP8_EVIDENCE_COMMIT, r9.EXP8_FILES), (EXP9_EVIDENCE_COMMIT, EXP9_FILES)):
        if e1.run_git("diff", "--name-only", commit, "--", *[str(p.relative_to(ROOT)) for p in paths]):
            raise RuntimeError(f"accepted evidence differs from its frozen commit {commit[:7]}")
    ok = gen.check()
    if not all(ok.values()):
        raise RuntimeError(f"Exp10 derived scripts differ from their derivation: {ok}")
    for b in gen.BENCHMARKS:
        text = gen.canonical_path(b).read_text(encoding="utf-8")
        if e1.REQUIRED_ALLOWED_LINE not in text or e1.REQUIRED_RABIT2_MARKER not in text:
            raise RuntimeError(f"canonical {b}.py no longer matches the accepted Experiment 1 pins")
    e1.verify_canonical_reference()
    protocol = load_protocol()
    uncommitted = e1.run_git("status", "--short", "--", *[str(p.relative_to(ROOT)) for p in (
        RUNNER_SCRIPT, gen.DERIVED_DIR, HERE / "exp10_residual_scripts.py", HERE / "test_experiment10_residual.py",
        PROTOCOL)])
    if uncommitted and not dry_run:
        raise RuntimeError("Refusing to run: Experiment 10 harness has uncommitted changes:\n" + uncommitted)
    if MANIFEST.exists() and json.loads(MANIFEST.read_text(encoding="utf-8")).get("status") == "passed":
        raise RuntimeError(f"{MANIFEST} already records a passed run; refusing to overwrite")
    return {"git_head": e1.run_git("rev-parse", "HEAD"), "rabit_kv2_sha256": e1.EXPECTED_RABIT_SHA256,
            "exp6_frozen_commit": EXP6_FROZEN_COMMIT, "exp7_evidence_commit": EXP7_EVIDENCE_COMMIT,
            "exp8_evidence_commit": EXP8_EVIDENCE_COMMIT, "exp9_evidence_commit": EXP9_EVIDENCE_COMMIT,
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


def integrity(name: str, rc: int, log_text: str, protocol: dict | None = None) -> dict:
    protocol = protocol or load_protocol()
    rows = parse_rows(name, log_text)
    checks = {"exit_code_zero": rc == 0, "no_traceback": "Traceback (most recent call last)" not in log_text,
              "all_five_rows_present": all(rows.values())}
    if checks["all_five_rows_present"]:
        checks["counts_exact"] = all(r["count"] == COUNT_COLUMNS[name][1] for r in rows.values())
        gates = protocol["logical_storage_expectations"]["gates"]
        exp = protocol["logical_storage_expectations"]["expected_avg_logical_kv_mb_full_precision"][name]
        tol = gates["matches_expected"]["absolute_mb"]
        checks["kv_mb_matches_expected"] = all(
            abs(rows[m]["avg_logical_kv_mb"] - exp[m]) <= tol + 1e-9 for m in CONDITIONS)
        observed = sorted(CONDITIONS, key=lambda m: rows[m]["avg_logical_kv_mb"])
        checks["kv_mb_ordering_matches_formula"] = (
            observed == gates["ordering"]["expected_order_smallest_first"][name]
            and len({rows[m]["avg_logical_kv_mb"] for m in CONDITIONS}) == len(CONDITIONS))
    reg = e1.check_regression(name, log_text)  # bf16 + R4 control vs the canonical reference
    checks["bf16_and_r4_control_reproduce_canonical"] = reg["all_within_tolerance"]
    return {"benchmark": name, "rows": rows, "checks": checks, "regression": reg, "passed": all(checks.values())}


def main(argv: list[str] | None = None) -> int:
    e1.make_console_encoding_safe()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-protocol", action="store_true", help="write exp10_residual_protocol.json (pre-commit only)")
    a = ap.parse_args(argv)
    if a.write_protocol:
        if PROTOCOL.exists():
            raise SystemExit(f"{PROTOCOL.name} already exists; the frozen protocol is never overwritten")
        PROTOCOL.write_text(json.dumps(build_protocol(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {PROTOCOL.relative_to(ROOT).as_posix()}")
        return 0
    prov = preflight(a.dry_run)
    print("RABIT-KV MLSys 2027 -- Experiment 10 residual-window ablation (logical fake-quant quality)")
    print("Preflight OK:", json.dumps({k: prov[k] for k in ("git_head", "exp6_frozen_commit", "exp7_evidence_commit",
                                                             "exp8_evidence_commit", "exp9_evidence_commit",
                                                             "protocol_sha256")}))
    for c in build_commands():
        print(f"  {c['name']}: {' '.join(c['command'][3:])}")
    if prov["uncommitted_files"]:
        print("  WARNING (dry-run only): uncommitted:\n    " + prov["uncommitted_files"].replace("\n", "\n    "))
    if a.dry_run:
        print("\n--dry-run: nothing executed, no files written.")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment": "Experiment 10 -- residual-window ablation (logical fake-quant quality)",
                "methods": METHODS, "conditions_residual": CONDITIONS, "frozen": FROZEN, "status": "running",
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "provenance": prov, "runs": []}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    results = []
    for c in build_commands():
        log = ROOT / c["log"]
        rc = e1.stream_command(c["command"], log)
        res = integrity(c["name"], rc, log.read_text(encoding="utf-8", errors="replace"), prov["protocol"])
        results.append(res)
        manifest["runs"].append({"name": c["name"], "returncode": rc, "passed": res["passed"],
                                 "log_sha256": e1.sha256(log)})
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if not res["passed"]:  # no retry; stop at the first failing benchmark
            break
    REGRESSION_CHECK.write_text(json.dumps([r["regression"] for r in results], indent=2) + "\n", encoding="utf-8")
    RESULTS.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    ok = len(results) == len(e1.RUNS) and all(r["passed"] for r in results) and not protected_status()
    manifest.update(status="passed" if ok else "failed", completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nEXPERIMENT 10 {manifest['status'].upper()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
