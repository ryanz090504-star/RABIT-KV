"""
RABIT-KV MLSys 2027 -- Experiment 9 runner: single-axis group-size ablation, LOGICAL fake-quant QUALITY only.

Question: holding K3 (seq_affine) / V2 (group_affine) / R4 / META8g64 fixed, how do quality and logical storage change
as the quantization group size G changes -- is G32 a favourable quality / storage operating point? (G32 is not
presupposed best.)

Pre-registered sweep (docs/MLSYS_EXPERIMENT_PLAN.md, Experiment 9): G16, G32 (control = canonical rabit2), G64, plus
the bf16 reference row. G is the rabit2 policy's single group-size knob: k_group = v_group = G. In the canonical code
k_group is the key quantizer's SEQUENCE group (seq_affine: G tokens per channel share a min / scale; the quantized
sequence is padded up to a multiple of G) and v_group is the value quantizer's HEAD_DIM group (group_affine: G of the
128 channels per token share a min / scale).

Separate from Experiments 7 / 8 (their code is imported read-only; nothing in Exp7 / Exp8 is modified). Scripts are the
deterministic Experiment 9 copies in benchmarks/mlsys2027/exp9_group/ (exp9_group_scripts.py). Not a physical serving
benchmark: no allocator capacity, throughput or latency; KV numbers are LOGICAL packed-prefix accounting.

Usage:
    python benchmarks/mlsys2027/run_experiment9_group_ablation.py --write-protocol   (once, before commit)
    python benchmarks/mlsys2027/run_experiment9_group_ablation.py --dry-run
    python benchmarks/mlsys2027/run_experiment9_group_ablation.py            (NOT until explicitly authorized)
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
import exp9_group_scripts as gen  # noqa: E402
import run_experiment1_quality_frontier as e1  # noqa: E402  (accepted; reused read-only)
import run_experiment7_kbit_ablation as r7  # noqa: E402  (accepted; reused read-only)
import run_experiment8_vbit_ablation as r8  # noqa: E402  (accepted; reused read-only)

ROOT = e1.ROOT
RUNNER_SCRIPT = Path(__file__).resolve()
OUT_DIR = ROOT / "results" / "mlsys2027" / "ablations" / "group_size"
MANIFEST = OUT_DIR / "manifest.json"
REGRESSION_CHECK = OUT_DIR / "regression_check.json"
RESULTS = OUT_DIR / "group_results.json"
PROTOCOL = HERE / "exp9_group_protocol.json"
METHODS = "bf16,rabit2_g16,rabit2,rabit2_g64"
CONDITIONS = {"rabit2_g16": 16, "rabit2": 32, "rabit2_g64": 64}  # method -> G (= k_group = v_group)
GROUP_FIELDS = gen.GROUP_FIELDS
FROZEN = {"k_bits": 3, "v_bits": 2, "k_style": "seq_affine", "v_style": "group_affine", "residual": 4,
          "metadata_mode": "int8", "metadata_group_size": 64}
EXP6_FROZEN_COMMIT = r7.EXP6_FROZEN_COMMIT
EXP7_EVIDENCE_COMMIT = r8.EXP7_EVIDENCE_COMMIT
EXP8_EVIDENCE_COMMIT = "e13a9b6ce9a7eaf5008bde27a37e728f40c9abe4"
EXP8_FILES = [HERE / "exp8_vbit_scripts.py", HERE / "exp8_vbit", HERE / "run_experiment8_vbit_ablation.py",
              HERE / "test_experiment8_vbit.py", HERE / "exp8_vbit_protocol.json", r8.OUT_DIR]
PROTECTED_PATHS = [*r8.PROTECTED_PATHS, *EXP8_FILES]
COUNT_COLUMNS = r8.COUNT_COLUMNS  # (count column index, expected per-method count); identical workloads
EXP1_PRESETS = ("rabit8", "rabit4", "rabit3", "rabit2")  # anchors of the storage model (observed Exp1 MB)


# ---------------------------------------------------------------------------------------------------------------
# Logical storage model: a shape-only trace of the canonical accounting (q_with_residual -> q_tensor ->
# q_group_sym / q_group_affine / q_seq_affine -> encode_metadata -> stored_state_logical_bytes), for ANY config.
# ---------------------------------------------------------------------------------------------------------------
def _metadata_bytes(n: int, config: dict) -> int:
    """metadata_bytes(encode_metadata(tensor with n elements)): BF16, or uint8 codes padded to the metadata group
    (max(8, metadata_group_size)) plus one BF16 min and one BF16 scale per metadata group."""
    if str(config["metadata_mode"]).lower() == "bf16":
        return 2 * n
    g = max(8, int(config["metadata_group_size"]))
    npad = math.ceil(n / g) * g
    return npad + 4 * (npad // g)


def _quantized_parts(tokens: int, bits: int, side: str, config: dict) -> tuple[int, int]:
    """(payload bytes, metadata bytes) of q_tensor on one layer's [1, KV_HEADS, tokens, HEAD_DIM] K or V."""
    style, g = config[f"{side}_style"], int(config[f"{side}_group"])
    heads, dim = r7.KV_HEADS, r7.HEAD_DIM
    if style in ("group_sym", "group_affine"):  # grouped along head_dim, head_dim padded up to G
        dim_pad = math.ceil(dim / g) * g
        codes, meta_elems = heads * tokens * dim_pad, heads * tokens * (dim_pad // g)
        tensors = 1 if style == "group_sym" else 2  # absmax | (min, scale)
    elif style == "seq_affine":  # grouped along the sequence, sequence padded up to G
        seq_pad = math.ceil(tokens / g) * g
        codes, meta_elems, tensors = heads * seq_pad * dim, heads * (seq_pad // g) * dim, 2
    else:
        raise ValueError(style)
    return (codes * bits + 7) // 8, tensors * _metadata_bytes(meta_elems, config)


def traced_logical_bytes(prefix_tokens: int, config: dict) -> dict:
    """Logical bytes of one sequence's quantized prefix KV (batch 1, all layers), broken down by component."""
    out = {"k_payload": 0, "k_meta": 0, "v_payload": 0, "v_meta": 0, "residual": 0}
    bf16_token = r7.KV_HEADS * r7.HEAD_DIM * 2
    for side in ("k", "v"):
        residual = min(int(config.get("residual", 0)), prefix_tokens)
        if residual > 0 and prefix_tokens <= residual:
            out["residual"] += prefix_tokens * bf16_token
            continue
        payload, meta = _quantized_parts(prefix_tokens - max(residual, 0), config[f"{side}_bits"], side, config)
        out[f"{side}_payload"] += payload
        out[f"{side}_meta"] += meta
        out["residual"] += max(residual, 0) * bf16_token
    out = {k: r7.LAYERS * v for k, v in out.items()}
    out["total"] = sum(out.values())
    return out


def canonical_configs(benchmark: str, methods=EXP1_PRESETS) -> dict:
    """config_for_method of the CANONICAL script (compiled on its own)."""
    text = gen.canonical_path(benchmark).read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == "config_for_method")
    ns: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<config_for_method>", "exec"), ns)  # noqa: S102
    return {m: ns["config_for_method"](m) for m in methods}


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
    out["bf16_reference"] = round(sum(r7.LAYERS * 2 * t * r7.KV_HEADS * r7.HEAD_DIM * 2 / 2**20
                                      for t in r7.quantized_prefixes(name)) / len(r7.quantized_prefixes(name)), 3)
    return out


def storage_breakdown(name: str, configs: dict) -> dict:
    return {m: {k: _avg_mb(name, configs[m], k) for k in ("k_payload", "k_meta", "v_payload", "v_meta", "residual")}
            for m in CONDITIONS}


def exp1_observed_mb(name: str) -> dict:
    text = (r7.EXP1_DIR / f"{name}.log").read_text(encoding="utf-8")
    mi = r7.ROW_COLUMNS[name][2]
    return {m: float(e1._last_row_tokens(text, m)[mi]) for m in EXP1_PRESETS}


def model_anchors() -> dict:
    """The storage model evaluated on the canonical Exp1 presets vs. the MB those presets actually printed."""
    out = {}
    for b in gen.BENCHMARKS:
        cfgs, obs = canonical_configs(b), exp1_observed_mb(b)
        out[b] = {m: {"model_mb": _avg_mb(b, cfgs[m]), "exp1_observed_mb": obs[m]} for m in EXP1_PRESETS}
    return out


def expected_order(exp: dict) -> list[str]:
    """Conditions ordered by expected logical KV MB, largest first (derived from the formula, not assumed)."""
    order = sorted(CONDITIONS, key=lambda m: -exp[m])
    if len({exp[m] for m in CONDITIONS}) != len(CONDITIONS):
        raise RuntimeError("expected logical KV MB not strictly ordered")
    return order


def runs() -> list[dict]:
    out = []
    for spec in e1.RUNS:
        args = list(spec["args"])
        args[args.index("--methods") + 1] = METHODS
        out.append({"name": spec["name"], "script": gen.derived_path(spec["name"]), "args": args,
                    "purpose": spec["purpose"].replace("bf16/rabit8/rabit4/rabit3/rabit2 frontier",
                                                       "group-size ablation").replace("full frontier",
                                                                                      "group-size ablation")})
    return out


def build_protocol() -> dict:
    cfgs = {b: derived_configs(b) for b in gen.BENCHMARKS}
    first = cfgs[gen.BENCHMARKS[0]]
    if any(cfgs[b] != first for b in gen.BENCHMARKS):
        raise RuntimeError("condition configs differ between benchmark scripts")
    control = first["rabit2"]
    proof = {m: sorted(k for k in set(c) | set(control) if c.get(k) != control.get(k) and k != "name")
             for m, c in first.items()}
    anchors = model_anchors()
    if any(a["model_mb"] != a["exp1_observed_mb"] for b in anchors.values() for a in b.values()):
        raise RuntimeError("storage model does not reproduce the observed Exp1 preset MB")
    expected = {b: expected_logical_mb(b, first) for b in gen.BENCHMARKS}
    ref = e1.CANONICAL_REFERENCE
    metric_key = {"continuation_ppl": "ppl", "niah": "accuracy_pct", "passage_retrieval": "accuracy_pct",
                  "hotpotqa": "f1_pct", "qasper": "f1_pct"}
    shorthand = lambda c: (f"K{c['k_bits']}/V{c['v_bits']}/G{c['k_group']}/R{c['residual']}/"  # noqa: E731
                           f"META8g{c['metadata_group_size']}")
    return {
        "experiment": 9, "type": "logical fake-quant / dequant QUALITY ablation (NOT a physical serving benchmark)",
        "axis": "group size G (k_group = v_group)", "not_reported": ["physical allocator capacity", "throughput", "latency"],
        "question": ("Holding K3 / V2 / R4 / META8g64 fixed, how do quality and logical storage change as the "
                     "quantization group size changes? Is G32 a favourable quality / storage operating point? G32 is "
                     "not presupposed best."),
        "source": "docs/MLSYS_EXPERIMENT_PLAN.md, Experiment 9 (control rabit2 G32; treatment G16 and G64 at fixed "
                  "bit-width; new config_for_method entries varying only k_group / v_group)",
        "methods_argument": METHODS,
        "group_size_definition": {
            "G": "the rabit2 policy's single group-size knob: k_group = v_group = G (the canonical rabit2 has "
                 "k_group = v_group = 32, displayed 'G32'); both fields change together, to the same value",
            "k_group": "SEQUENCE group of the seq_affine key quantizer (q_seq_affine): G consecutive quantized tokens "
                       "of each (head, channel) share one min and one scale; the quantized sequence is padded up to "
                       "a multiple of G (padding codes are counted in the logical payload, as in the canonical code)",
            "v_group": "HEAD_DIM group of the group_affine value quantizer (q_group_affine): G of the 128 channels of "
                       "each (head, token) share one min and one scale; 128 is divisible by 16 / 32 / 64, so no padding",
            "read_only_in": "q_tensor (config[f'{side}_group']); no other canonical code reads k_group / v_group"},
        "conditions": {"bf16": {"role": "reference (uncompressed); not an ablation condition"},
                       **{m: {"role": "control" if m == "rabit2" else "treatment", "shorthand": shorthand(c), "config": c}
                          for m, c in first.items()}},
        "only_group_size_differs_proof": {
            "control": "rabit2", "fields_differing_from_control_excluding_display_name": proof,
            "holds": all(v == ([] if m == "rabit2" else sorted(GROUP_FIELDS)) for m, v in proof.items())
                     and all(c["k_group"] == c["v_group"] == CONDITIONS[m] for m, c in first.items()),
            "frozen_fields": FROZEN,
            "configs_identical_across_all_five_scripts": True},
        "correctness_gate": ("none injected: the plan specifies code changes for Experiment 9 as config_for_method "
                             "entries only (the NaN / Inf / degeneracy check is Experiment 8's pre-registration); "
                             "execution gates below apply to every condition"),
        "common": r7.COMMON_FACTS,
        "benchmarks": {b: {**r7.BENCHMARK_FACTS[b], "script": gen.derived_path(b).relative_to(ROOT).as_posix(),
                           "canonical_source": gen.canonical_path(b).relative_to(ROOT).as_posix(),
                           "canonical_source_sha256_lf": gen.g7.sha256_text(gen.canonical_path(b).read_text(encoding="utf-8")),
                           "args": next(r["args"] for r in runs() if r["name"] == b),
                           "expected_count_per_method": COUNT_COLUMNS[b][1]} for b in gen.BENCHMARKS},
        "control_reproduction": {
            "configuration_equality": "EXACT: the G32 control is the canonical rabit2 config (compiled and compared)",
            "numerical_reproduction": "within the frozen tolerances below (copied from the accepted Experiment 1 "
                                      "methodology); applied to bf16 and the G32 control; never changed after the run starts",
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
                                    "rule": "1.0-point tolerance is a tight drift detector, identical to Exp1 / Exp7 / Exp8"},
            "observed_control_variation": ("Qasper rabit2 F1 across identical-config runs: 35.6 (canonical), 36.2 "
                                           "(Exp7), 34.9 (Exp8) -- all within the 1.0-point tolerance; Qasper "
                                           "differences below ~1 point are within run-to-run variation"),
            "canonical_targets": {b: {"bf16": {metric_key[b]: ref[b]["bf16"][metric_key[b]],
                                               "avg_logical_kv_mb": ref[b]["bf16"]["avg_kv_mb"]},
                                      "rabit2_G32": {metric_key[b]: ref[b]["rabit2"][metric_key[b]],
                                                     "avg_logical_kv_mb": ref[b]["rabit2"]["avg_kv_mb"]}}
                                  for b in gen.BENCHMARKS},
            "target_source": "results/summary.json and results/quality/*.log (canonical), pinned in the accepted "
                             "Experiment 1 runner"},
        "logical_storage_expectations": {
            "label": "LOGICAL packed prefix-KV storage (payload bits + uint8-group metadata + BF16 residual); NOT "
                     "physical allocator capacity",
            "formula": ("shape-only trace of the canonical accounting, per layer (32 layers, 8 KV heads, head_dim "
                        "128), quantized prefix P, R = 4 BF16 residual tokens, Lq = P - 4: "
                        "K (seq_affine): Lk = ceil(Lq/G)*G; payload = ceil(8*Lk*128*3/8); metadata = 2 tensors "
                        "(min, scale) of 8*(Lk/G)*128 values each. "
                        "V (group_affine along head_dim): payload = ceil(8*Lq*128*2/8); metadata = 2 tensors of "
                        "8*Lq*(128/G) values each. "
                        "meta(n) = ceil(n/64)*64 uint8 codes + 4 bytes (BF16 min + scale) per 64-value metadata group. "
                        "Residual = 2*4*8*128*2 bytes. Average over the benchmark's quantized prefixes, / 2^20, "
                        "rounded to 3 decimals like the scripts' print."),
            "group_size_dependence": ("G changes (a) K and V metadata volume, both ~ 1/G, and (b) K payload through "
                                      "the sequence padding Lk = ceil(Lq/G)*G. V payload and the BF16 residual do not "
                                      "depend on G. Storage is NOT linear in G or 1/G (ceil terms); no linearity gate "
                                      "is used -- every condition is checked against its own expected value"),
            "model_anchors": {"rule": "the same model, evaluated on the canonical Exp1 presets (rabit8 G128 R0 "
                                      "META8g256, rabit4 G128 R0, rabit3 G32 R2 group_sym, rabit2), reproduces the "
                                      "MB they printed in the accepted Exp1 run exactly",
                              "values": anchors},
            "expected_avg_logical_kv_mb": expected,
            "expected_compression_vs_bf16": {b: {m: round(expected[b]["bf16_reference"] / expected[b][m], 3)
                                                 for m in CONDITIONS} for b in gen.BENCHMARKS},
            "expected_breakdown_avg_mb": {b: storage_breakdown(b, first) for b in gen.BENCHMARKS},
            "gates": {"matches_expected": {"rule": "observed G16 / G32 / G64 avg logical KV MB within the relative "
                                                   "tolerance of expected_avg_logical_kv_mb",
                                           "relative": e1.KV_MB_RELATIVE_TOLERANCE},
                      "ordering": {"rule": "observed conditions ordered (largest MB first) exactly as the formula "
                                           "orders them",
                                   "expected_order_largest_first": {b: expected_order(expected[b])
                                                                    for b in gen.BENCHMARKS}}}},
        "execution_gates": ["exit code 0 and no traceback", "all four rows (bf16, G16, G32, G64) present",
                            "per-method counts equal the frozen counts",
                            "bf16 and G32 control within the frozen tolerances of the canonical targets",
                            "logical storage gates above",
                            "protected paths clean; Exp6 identical to 0f5f6ef; Exp7 identical to 6ce79ed; "
                            "Exp8 identical to e13a9b6",
                            "stop at the first failing benchmark; no retry"],
        "artifacts": [f"results/mlsys2027/ablations/group_size/{b}.log" for b in gen.BENCHMARKS]
                     + ["results/mlsys2027/ablations/group_size/manifest.json",
                        "results/mlsys2027/ablations/group_size/regression_check.json",
                        "results/mlsys2027/ablations/group_size/group_results.json"],
    }


def load_protocol() -> dict:
    committed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if committed != json.loads(json.dumps(build_protocol())):
        raise RuntimeError("exp9_group_protocol.json differs from the regenerated protocol")
    if not committed["only_group_size_differs_proof"]["holds"]:
        raise RuntimeError("protocol does not prove that conditions differ only in k_group / v_group")
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
                          (EXP8_EVIDENCE_COMMIT, EXP8_FILES)):
        if e1.run_git("diff", "--name-only", commit, "--", *[str(p.relative_to(ROOT)) for p in paths]):
            raise RuntimeError(f"accepted evidence differs from its frozen commit {commit[:7]}")
    ok = gen.check()
    if not all(ok.values()):
        raise RuntimeError(f"Exp9 derived scripts differ from their derivation: {ok}")
    for b in gen.BENCHMARKS:
        text = gen.canonical_path(b).read_text(encoding="utf-8")
        if e1.REQUIRED_ALLOWED_LINE not in text or e1.REQUIRED_RABIT2_MARKER not in text:
            raise RuntimeError(f"canonical {b}.py no longer matches the accepted Experiment 1 pins")
    e1.verify_canonical_reference()
    protocol = load_protocol()
    uncommitted = e1.run_git("status", "--short", "--", *[str(p.relative_to(ROOT)) for p in (
        RUNNER_SCRIPT, gen.DERIVED_DIR, HERE / "exp9_group_scripts.py", HERE / "test_experiment9_group.py", PROTOCOL)])
    if uncommitted and not dry_run:
        raise RuntimeError("Refusing to run: Experiment 9 harness has uncommitted changes:\n" + uncommitted)
    if MANIFEST.exists() and json.loads(MANIFEST.read_text(encoding="utf-8")).get("status") == "passed":
        raise RuntimeError(f"{MANIFEST} already records a passed run; refusing to overwrite")
    return {"git_head": e1.run_git("rev-parse", "HEAD"), "rabit_kv2_sha256": e1.EXPECTED_RABIT_SHA256,
            "exp6_frozen_commit": EXP6_FROZEN_COMMIT, "exp7_evidence_commit": EXP7_EVIDENCE_COMMIT,
            "exp8_evidence_commit": EXP8_EVIDENCE_COMMIT,
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
              "all_four_rows_present": all(rows.values())}
    if checks["all_four_rows_present"]:
        checks["counts_exact"] = all(r["count"] == COUNT_COLUMNS[name][1] for r in rows.values())
        gates = protocol["logical_storage_expectations"]["gates"]
        exp = protocol["logical_storage_expectations"]["expected_avg_logical_kv_mb"][name]
        rel = gates["matches_expected"]["relative"]
        checks["kv_mb_matches_expected"] = all(
            abs(rows[m]["avg_logical_kv_mb"] - exp[m]) <= max(abs(exp[m]) * rel, 1e-9) for m in CONDITIONS)
        observed = sorted(CONDITIONS, key=lambda m: -rows[m]["avg_logical_kv_mb"])
        checks["kv_mb_ordering_matches_formula"] = (
            observed == gates["ordering"]["expected_order_largest_first"][name]
            and len({rows[m]["avg_logical_kv_mb"] for m in CONDITIONS}) == len(CONDITIONS))
    reg = e1.check_regression(name, log_text)  # bf16 + G32 control vs the canonical reference
    checks["bf16_and_g32_control_reproduce_canonical"] = reg["all_within_tolerance"]
    return {"benchmark": name, "rows": rows, "checks": checks, "regression": reg, "passed": all(checks.values())}


def main(argv: list[str] | None = None) -> int:
    e1.make_console_encoding_safe()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-protocol", action="store_true", help="write exp9_group_protocol.json (pre-commit only)")
    a = ap.parse_args(argv)
    if a.write_protocol:
        if PROTOCOL.exists():
            raise SystemExit(f"{PROTOCOL.name} already exists; the frozen protocol is never overwritten")
        PROTOCOL.write_text(json.dumps(build_protocol(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {PROTOCOL.relative_to(ROOT).as_posix()}")
        return 0
    prov = preflight(a.dry_run)
    print("RABIT-KV MLSys 2027 -- Experiment 9 group-size ablation (logical fake-quant quality)")
    print("Preflight OK:", json.dumps({k: prov[k] for k in ("git_head", "exp6_frozen_commit", "exp7_evidence_commit",
                                                             "exp8_evidence_commit", "protocol_sha256")}))
    for c in build_commands():
        print(f"  {c['name']}: {' '.join(c['command'][3:])}")
    if prov["uncommitted_files"]:
        print("  WARNING (dry-run only): uncommitted:\n    " + prov["uncommitted_files"].replace("\n", "\n    "))
    if a.dry_run:
        print("\n--dry-run: nothing executed, no files written.")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment": "Experiment 9 -- group-size ablation (logical fake-quant quality)", "methods": METHODS,
                "conditions_group_size": CONDITIONS, "frozen": FROZEN, "status": "running",
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
    print(f"\nEXPERIMENT 9 {manifest['status'].upper()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
