"""
RABIT-KV MLSys 2027 -- Experiment 11 runner: metadata-POLICY ablation (two contrasts), LOGICAL fake-quant QUALITY + logical storage only.

Question (docs/MLSYS_EXPERIMENT_PLAN.md, Experiment 11): holding K3 (seq_affine) / V2 (group_affine) / G32 / R4 fixed,
does grouped UINT8 metadata (META8g64) cost quality relative to BF16 metadata, and how does the metadata group size
affect the quality / storage trade-off? META8g64 is not presupposed best.

Conditions: rabit2 (control, META8g64 = metadata_mode int8, metadata_group_size 64), rabit2_mbf (BF16 metadata),
rabit2_m32 (META8g32), rabit2_m128 (META8g128), plus the bf16 reference row. Each treatment changes exactly ONE config
field (metadata_mode, or metadata_group_size).

What "metadata" is here: the per-group quantization parameters of the K and V payloads -- K (seq_affine, 32-token
sequence groups): one min and one scale per (head, 32-token group, channel); V (group_affine, 32-channel head_dim
groups): one min and one scale per (head, token, channel group). These PRIMARY parameters are computed in fp32; the
payload codes are computed from the fp32 values; then each min / scale tensor is stored by encode_metadata:
  int8 (META8gM): flattened, padded to a multiple of M (repeating the last value), per M-value group an affine UINT8
                  code per value (1 byte) plus SECONDARY parameters: one BF16 group minimum and one BF16 group scale
                  ((max-min)/255) -- 4 bytes per metadata group;
  bf16:           each primary value rounded to BF16 (2 bytes); no secondary parameters.
Dequantization uses the DECODED min / scale, so the metadata representation changes the reconstructed K / V.

Separate from Experiments 7-10 (imported read-only). HotpotQA / Qasper control validity uses the FROZEN post-failure
per-example QA control gate (qa_control_gate_amendment.json, pinned by hash; not modified, no new thresholds). Not a
physical serving benchmark: no allocator capacity, throughput or latency.

Usage:
    python benchmarks/mlsys2027/run_experiment11_metadata_ablation.py --write-protocol   (once, before commit)
    python benchmarks/mlsys2027/run_experiment11_metadata_ablation.py --dry-run
    python benchmarks/mlsys2027/run_experiment11_metadata_ablation.py            (NOT until explicitly authorized)
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import math
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp11_metadata_scripts as gen  # noqa: E402
import qa_control_gate as qg  # noqa: E402  (frozen amendment; read-only)
import run_experiment1_quality_frontier as e1  # noqa: E402  (accepted; read-only)
import run_experiment7_kbit_ablation as r7  # noqa: E402  (accepted; read-only)
import run_experiment8_vbit_ablation as r8  # noqa: E402  (accepted; read-only)
import run_experiment9_group_ablation as r9  # noqa: E402  (accepted; read-only: storage model)
import run_experiment10_residual_ablation as r10  # noqa: E402  (accepted; read-only)
import run_experiment10_attempt2 as a2  # noqa: E402  (accepted; read-only: evidence paths)

ROOT = e1.ROOT
RUNNER_SCRIPT = Path(__file__).resolve()
OUT_DIR = ROOT / "results" / "mlsys2027" / "ablations" / "metadata"
MANIFEST = OUT_DIR / "manifest.json"
REGRESSION_CHECK = OUT_DIR / "regression_check.json"
RESULTS = OUT_DIR / "metadata_results.json"
PROTOCOL = HERE / "exp11_metadata_protocol.json"
METHODS = "bf16,rabit2_mbf,rabit2_m32,rabit2,rabit2_m128"
CONDITIONS = {"rabit2_mbf": {"metadata_mode": "bf16", "metadata_group_size": 64},
              "rabit2_m32": {"metadata_mode": "int8", "metadata_group_size": 32},
              "rabit2": {"metadata_mode": "int8", "metadata_group_size": 64},
              "rabit2_m128": {"metadata_mode": "int8", "metadata_group_size": 128}}
CHANGED_FIELD = {"rabit2_mbf": ["metadata_mode"], "rabit2_m32": ["metadata_group_size"], "rabit2": [],
                 "rabit2_m128": ["metadata_group_size"]}
FROZEN = {"k_bits": 3, "v_bits": 2, "k_style": "seq_affine", "v_style": "group_affine", "k_group": 32, "v_group": 32,
          "residual": 4}
QA_BENCHMARKS = a2.QA_BENCHMARKS
AMENDMENT_SHA256_LF = a2.AMENDMENT_SHA256_LF
AMENDMENT_COMMIT = "86b03eabaecfed0a838cabd7a7d1cce2f0a2fa38"
EXP6_FROZEN_COMMIT = r7.EXP6_FROZEN_COMMIT
EXP7_EVIDENCE_COMMIT = r8.EXP7_EVIDENCE_COMMIT
EXP8_EVIDENCE_COMMIT = r9.EXP8_EVIDENCE_COMMIT
EXP9_EVIDENCE_COMMIT = r10.EXP9_EVIDENCE_COMMIT
EXP10_EVIDENCE_COMMIT = "01fe47eb4af2199e968747ac3abd4e3a5d54e341"
EXP10_FILES = [HERE / "exp10_residual_scripts.py", HERE / "exp10_residual", HERE / "run_experiment10_residual_ablation.py",
               HERE / "test_experiment10_residual.py", HERE / "exp10_residual_protocol.json",
               HERE / "run_experiment10_attempt2.py", HERE / "test_experiment10_attempt2.py", r10.OUT_DIR]
AMENDMENT_FILES = [HERE / "qa_control_gate.py", HERE / "audit_qa_control_reproducibility.py",
                   qg.AMENDMENT.parent]
PROTECTED_PATHS = [*r10.PROTECTED_PATHS, *EXP10_FILES, *AMENDMENT_FILES]
COUNT_COLUMNS = r8.COUNT_COLUMNS
STORAGE_MATCH_ABS_TOL_MB = r10.STORAGE_MATCH_ABS_TOL_MB  # accounting-integrity gate (Exp10 semantics)
LAYERS, KV_HEADS, HEAD_DIM = r7.LAYERS, r7.KV_HEADS, r7.HEAD_DIM
# Additive components (sum = total). Secondary metadata is listed as its two BF16 tensors separately.
COMPONENTS = ("k_payload", "v_payload",
              "k_meta_primary", "k_meta_secondary_min", "k_meta_secondary_scale",
              "v_meta_primary", "v_meta_secondary_min", "v_meta_secondary_scale",
              "residual")
META_COMPONENTS = tuple(k for k in COMPONENTS if "_meta_" in k)
# Derived subtotals (not additive with COMPONENTS).
SUBTOTALS = ("k_meta_total", "v_meta_total", "meta_primary_total", "meta_secondary_total", "metadata_total")


# ---------------------------------------------------------------------------------------------------------------
# Exact integer logical-storage model with metadata split into PRIMARY (uint8 codes incl. metadata-group padding, or
# BF16 values) and SECONDARY bytes: the BF16 per-metadata-group minimum AND the BF16 per-metadata-group scale.
# ---------------------------------------------------------------------------------------------------------------
def metadata_tensor_bytes(n: int, config: dict) -> tuple[int, int, int]:
    """(primary, secondary_min, secondary_scale) bytes of encode_metadata on one min or scale tensor of n values:
    META8gM -> (ceil(n/M)*M uint8 codes, 2*ceil(n/M) BF16 group minima, 2*ceil(n/M) BF16 group scales);
    BF16    -> (2n, 0, 0)."""
    if str(config["metadata_mode"]).lower() == "bf16":
        return 2 * n, 0, 0
    g = max(8, int(config["metadata_group_size"]))
    groups = math.ceil(n / g)
    return groups * g, 2 * groups, 2 * groups


def metadata_value_counts(prefix_tokens: int, config: dict) -> dict:
    """Primary metadata values per layer, per min OR scale tensor (each quantized region has one min + one scale)."""
    lq = prefix_tokens - int(config["residual"])
    lk = math.ceil(lq / config["k_group"]) * config["k_group"]
    return {"lq": lq, "lk": lk, "k_values_per_tensor": KV_HEADS * (lk // config["k_group"]) * HEAD_DIM,
            "v_values_per_tensor": KV_HEADS * lq * (HEAD_DIM // config["v_group"])}


def logical_bytes(prefix_tokens: int, config: dict) -> dict:
    """Exact logical bytes (all layers, batch 1) by component, for a prefix longer than the residual window."""
    c = metadata_value_counts(prefix_tokens, config)
    kp, kmin, kscl = metadata_tensor_bytes(c["k_values_per_tensor"], config)
    vp, vmin, vscl = metadata_tensor_bytes(c["v_values_per_tensor"], config)
    # each side encodes TWO primary tensors (the quantization min and the quantization scale), each with its own
    # primary codes and its own secondary (BF16 group min + BF16 group scale) -> factor 2 on every metadata term
    per_layer = {"k_payload": math.ceil(KV_HEADS * c["lk"] * HEAD_DIM * config["k_bits"] / 8),
                 "v_payload": math.ceil(KV_HEADS * c["lq"] * HEAD_DIM * config["v_bits"] / 8),
                 "k_meta_primary": 2 * kp, "k_meta_secondary_min": 2 * kmin, "k_meta_secondary_scale": 2 * kscl,
                 "v_meta_primary": 2 * vp, "v_meta_secondary_min": 2 * vmin, "v_meta_secondary_scale": 2 * vscl,
                 "residual": 2 * int(config["residual"]) * KV_HEADS * HEAD_DIM * 2}
    out = {k: LAYERS * v for k, v in per_layer.items()}
    out["k_meta_total"] = sum(out[k] for k in META_COMPONENTS if k.startswith("k_"))
    out["v_meta_total"] = sum(out[k] for k in META_COMPONENTS if k.startswith("v_"))
    out["meta_primary_total"] = out["k_meta_primary"] + out["v_meta_primary"]
    out["meta_secondary_total"] = sum(out[k] for k in META_COMPONENTS if "secondary" in k)
    out["metadata_total"] = sum(out[k] for k in META_COMPONENTS)
    out["total"] = sum(out[k] for k in COMPONENTS)
    return out


def derived_configs(benchmark: str, methods=None) -> dict:
    text = gen.derived_path(benchmark).read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == "config_for_method")
    ns: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<config_for_method>", "exec"), ns)  # noqa: S102
    return {m: ns["config_for_method"](m) for m in (methods or CONDITIONS)}


def _per_prefix(name: str, config: dict, key: str) -> list[int]:
    return [logical_bytes(t, config)[key] for t in r7.quantized_prefixes(name)]


def expected_storage(name: str, configs: dict) -> dict:
    prefixes = r7.quantized_prefixes(name)
    n = len(prefixes)
    bf16_ref = [LAYERS * 2 * t * KV_HEADS * HEAD_DIM * 2 for t in prefixes]
    out = {"bf16_reference_avg_mb": round(sum(bf16_ref) / n / 2**20, 3)}
    for m in CONDITIONS:
        comp = {k: sum(_per_prefix(name, configs[m], k)) for k in (*COMPONENTS, *SUBTOTALS, "total")}
        full = comp["total"] / n / 2**20
        out[m] = {"total_logical_bytes_all_samples": comp["total"],
                  "component_bytes_all_samples": {k: comp[k] for k in COMPONENTS},
                  "subtotal_bytes_all_samples": {k: comp[k] for k in SUBTOTALS},
                  "avg_logical_kv_mb": round(full, 3), "avg_logical_kv_mb_full_precision": full,
                  "avg_metadata_mb": round(comp["metadata_total"] / n / 2**20, 3),
                  "compression_vs_bf16": round(sum(bf16_ref) / comp["total"], 3)}
    ctrl = out["rabit2"]
    for m in CONDITIONS:
        out[m]["delta_vs_meta8g64_mb"] = round(out[m]["avg_logical_kv_mb_full_precision"]
                                               - ctrl["avg_logical_kv_mb_full_precision"], 3)
        out[m]["metadata_bytes_delta_vs_meta8g64"] = (out[m]["subtotal_bytes_all_samples"]["metadata_total"]
                                                      - ctrl["subtotal_bytes_all_samples"]["metadata_total"])
    return out


def expected_order(exp: dict) -> list[str]:
    vals = {m: exp[m]["avg_logical_kv_mb_full_precision"] for m in CONDITIONS}
    if len(set(vals.values())) != len(vals):
        raise RuntimeError("expected logical KV MB not strictly ordered")
    return sorted(CONDITIONS, key=lambda m: vals[m])


# ---------------------------------------------------------------------------------------------------------------
# Numerical-semantics check (pure Python emulation of encode_metadata / decode_metadata; offline, no torch)
# ---------------------------------------------------------------------------------------------------------------
def to_bf16(x: float) -> float:
    """Round-to-nearest-even float32 -> bfloat16, as torch .to(torch.bfloat16)."""
    bits = struct.unpack(">I", struct.pack(">f", x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def emulate_decoded_metadata(values: list[float], config: dict) -> list[float]:
    """decode_metadata(encode_metadata(values)) for the config's metadata mode / group size."""
    if str(config["metadata_mode"]).lower() == "bf16":
        return [to_bf16(v) for v in values]
    g = max(8, int(config["metadata_group_size"]))
    flat = list(values) + [values[-1]] * ((-len(values)) % g)
    out = []
    for i in range(0, len(flat), g):
        grp = flat[i:i + g]
        lo, hi = min(grp), max(grp)
        scale = (hi - lo) / 255.0
        scale = 1.0 if abs(scale) < 1e-12 else scale
        codes = [min(255, max(0, round((v - lo) / scale))) for v in grp]
        lo_b, sc_b = to_bf16(lo), to_bf16(scale)
        out += [c * sc_b + lo_b for c in codes]
    return out[:len(values)]


def runs() -> list[dict]:
    out = []
    for spec in e1.RUNS:
        args = list(spec["args"])
        args[args.index("--methods") + 1] = METHODS
        out.append({"name": spec["name"], "script": gen.derived_path(spec["name"]), "args": args,
                    "purpose": spec["purpose"].replace("bf16/rabit8/rabit4/rabit3/rabit2 frontier",
                                                       "metadata ablation").replace("full frontier",
                                                                                    "metadata ablation")})
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
    storage = {b: expected_storage(b, first) for b in gen.BENCHMARKS}
    ref = e1.CANONICAL_REFERENCE
    metric_key = {"continuation_ppl": "ppl", "niah": "accuracy_pct", "passage_retrieval": "accuracy_pct",
                  "hotpotqa": "f1_pct", "qasper": "f1_pct"}
    amendment = qg.load_amendment()
    return {
        "experiment": 11, "type": "logical fake-quant / dequant QUALITY + logical storage metadata-policy ablation "
                                  "(NOT a physical serving benchmark)",
        "axis": ("metadata-policy ablation (NOT one uniform numerical single-axis sweep) with two contrasts around the "
                 "META8g64 control: (1) metadata representation -- BF16 metadata vs grouped UINT8 metadata; "
                 "(2) within grouped UINT8 -- metadata group size 32 / 64 / 128. BF16 metadata does not lie on the "
                 "group-size axis"),
        "contrasts": {"representation": {"conditions": ["rabit2_mbf", "rabit2"],
                                         "compares": "BF16 metadata vs grouped UINT8 metadata (META8g64 control)"},
                      "uint8_group_size": {"conditions": ["rabit2_m32", "rabit2", "rabit2_m128"],
                                           "compares": "grouped UINT8 metadata at group size 32 / 64 (control) / 128"}},
        "not_reported": ["physical allocator capacity", "throughput", "latency"],
        "question": ("Holding K3 / V2 / G32 / R4 fixed: (1) does grouped UINT8 metadata (META8g64) materially hurt "
                     "quality relative to BF16 metadata? (2) within grouped UINT8 metadata, what quality / logical-storage "
                     "trade-off results from metadata group size 32 / 64 / 128? META8g64 is not presupposed best."),
        "source": "docs/MLSYS_EXPERIMENT_PLAN.md, Experiment 11 (control rabit2 uint8 g64; treatment BF16 metadata "
                  "(metadata_mode='bf16') and uint8 metadata at group 32 and 128; config_for_method entries only)",
        "methods_argument": METHODS,
        "meta8g64_definition": {
            "primary_parameters": {
                "k": "seq_affine K: min and scale per (layer, KV head, 32-token sequence group, head_dim channel); "
                     "per quantized region of Lk = ceil((T-4)/32)*32 tokens: 8*(Lk/32)*128 values per tensor",
                "v": "group_affine V: min and scale per (layer, KV head, token, 32-channel head_dim group); per "
                     "quantized region of Lq = T-4 tokens: 8*Lq*4 values per tensor",
                "computed_in": "fp32 (payload codes are computed from the fp32 min / scale)",
                "applies_to": "both K and V; two tensors (min, scale) per side per layer"},
            "stored_components_meta8g64": [
                {"component": "primary metadata codes", "semantic_role": "the K / V min and scale values, each "
                 "affinely quantized", "dtype": "uint8 (8 bits)", "grouping_axis": "the flattened min (or scale) "
                 "tensor of one quantized region (row-major over its shape), padded by repeating the last value",
                 "group_size": 64, "elements": "ceil(n/64)*64 per tensor of n values", "applies_to": "K and V",
                 "kind": "primary", "bytes": "ceil(n/64)*64"},
                {"component": "second-level minimum", "semantic_role": "offset of each 64-value metadata group",
                 "dtype": "bfloat16 (16 bits)", "grouping_axis": "per metadata group", "group_size": 64,
                 "elements": "ceil(n/64) per tensor", "applies_to": "K and V", "kind": "secondary",
                 "bytes": "2*ceil(n/64)"},
                {"component": "second-level scale", "semantic_role": "(max-min)/255 of each 64-value metadata group "
                 "(1.0 if < 1e-12)", "dtype": "bfloat16 (16 bits)", "grouping_axis": "per metadata group",
                 "group_size": 64, "elements": "ceil(n/64) per tensor", "applies_to": "K and V",
                 "kind": "secondary", "bytes": "2*ceil(n/64)"}],
            "decode": "value = code * bf16_scale + bf16_min (fp32 arithmetic); dequantized K / V = payload_code * "
                      "decoded_scale + decoded_min",
            "bytes_per_metadata_value": {"META8g32": 1.125, "META8g64": 1.0625, "META8g128": 1.03125, "BF16": 2.0},
            "physical_implementation_agrees": ("vllm-kvquant/vllm/v1/attention/ops/kvquant_k3.py "
                                               "encode_metadata_uint8_group_ref (group 64, /255, BF16 min / scale) "
                                               "and rabit_kv2.py encode_metadata_blob_ref (blob = uint8 codes incl. "
                                               "padding, then BF16 minima, then BF16 scales); physical per-page "
                                               "padding is not modelled here (logical experiment)"),
            "bf16_mode": "each primary value rounded to BF16 (2 bytes); no secondary parameters; "
                         "metadata_group_size is not read"},
        "conditions": {"bf16": {"role": "reference (uncompressed); not an ablation condition"},
                       **{m: {"role": "control" if m == "rabit2" else "treatment", "config": c,
                              "changed_fields": proof[m]} for m, c in first.items()}},
        "only_metadata_field_differs_proof": {
            "control": "rabit2", "fields_differing_from_control_excluding_display_name": proof,
            "holds": proof == CHANGED_FIELD and all(
                {k: c[k] for k in ("metadata_mode", "metadata_group_size")} == CONDITIONS[m] for m, c in first.items())
                and all({k: c[k] for k in FROZEN} == FROZEN for c in first.values()),
            "single_field_per_treatment": True,
            "note": "rabit2_mbf keeps metadata_group_size = 64 in its config, but in bf16 mode encode_metadata returns "
                    "before reading it (inert); rabit2_m32 / rabit2_m128 keep metadata_mode = int8",
            "frozen_fields": FROZEN, "configs_identical_across_all_five_scripts": True},
        "quality_semantics": {
            "reconstructed_kv_changes": True,
            "why": ("dequantization uses the DECODED min / scale; uint8 metadata at group M quantizes each min / "
                    "scale with a per-M-group affine 8-bit code and BF16 group parameters, so the decoding error "
                    "depends on M; BF16 metadata rounds each value to BF16 instead. The four conditions therefore "
                    "generally reconstruct different K / V values (not an equivalent re-encoding); quality "
                    "evaluation is scientifically meaningful and GPU evaluation is required"),
            "offline_evidence": "test_experiment11_metadata.py emulates encode/decode_metadata and shows different "
                                "decoded parameters across the four conditions (identical when the config is identical)"},
        "common": r7.COMMON_FACTS,
        "benchmarks": {b: {**r7.BENCHMARK_FACTS[b], "script": gen.derived_path(b).relative_to(ROOT).as_posix(),
                           "canonical_source": gen.canonical_path(b).relative_to(ROOT).as_posix(),
                           "canonical_source_sha256_lf": gen.g7.sha256_text(gen.canonical_path(b).read_text(encoding="utf-8")),
                           "args": next(r["args"] for r in runs() if r["name"] == b),
                           "expected_count_per_method": COUNT_COLUMNS[b][1]} for b in gen.BENCHMARKS},
        "control_reproduction": {
            "configuration_equality": "EXACT: the META8g64 control is the canonical rabit2 config (compiled and compared)",
            "continuation_ppl_niah_passage_retrieval": {
                "rule": "original Exp1 tolerances on bf16 and the control",
                "tolerances": {"continuation_ppl.ppl": {"relative": e1.PPL_RELATIVE_TOLERANCE},
                               "niah.accuracy_pct": {"absolute_points": e1.PERCENTAGE_ABSOLUTE_TOLERANCE},
                               "passage_retrieval.accuracy_pct": {"absolute_points": e1.PERCENTAGE_ABSOLUTE_TOLERANCE},
                               "avg_logical_kv_mb": {"relative": e1.KV_MB_RELATIVE_TOLERANCE}}},
            "hotpotqa_qasper": {
                "rule": "the FROZEN post-failure per-example QA control gate (transparent post-failure "
                        "methodological amendment, not pre-registered), applied unchanged to bf16 and the control; "
                        "the aggregate +/-1.0 F1 check is report-only; bf16 / control logical KV MB keep the 0.1 % "
                        "tolerance",
                "amendment": qg.AMENDMENT.relative_to(ROOT).as_posix(), "amendment_sha256_lf": AMENDMENT_SHA256_LF,
                "amendment_commit": AMENDMENT_COMMIT,
                "thresholds": {b: {m: {k: amendment["thresholds"][b][m][k] for k in (
                    "historical_max_score_mismatch_count", "historical_max_l1_score_distance")}
                    for m in ("bf16", "rabit2")} for b in QA_BENCHMARKS},
                "new_thresholds_derived": False},
            "canonical_targets": {b: {"bf16": {metric_key[b]: ref[b]["bf16"][metric_key[b]],
                                               "avg_logical_kv_mb": ref[b]["bf16"]["avg_kv_mb"]},
                                      "rabit2_META8g64": {metric_key[b]: ref[b]["rabit2"][metric_key[b]],
                                                          "avg_logical_kv_mb": ref[b]["rabit2"]["avg_kv_mb"]}}
                                  for b in gen.BENCHMARKS}},
        "logical_storage_expectations": {
            "label": "LOGICAL packed prefix-KV storage (payload bits + metadata + BF16 residual); NOT physical "
                     "allocator capacity",
            "formula": ("per layer (32 layers, 8 KV heads, head_dim 128), quantized prefix T, R = 4, Lq = T - 4, "
                        "Lk = ceil(Lq/32)*32: K payload = ceil(8*Lk*128*3/8); V payload = ceil(8*Lq*128*2/8); "
                        "n_K = 8*(Lk/32)*128 and n_V = 8*Lq*4 values per min (and per scale) tensor; per tensor of "
                        "n values: META8gM primary = ceil(n/M)*M uint8 bytes, secondary = BF16 group minima "
                        "2*ceil(n/M) bytes + BF16 group scales 2*ceil(n/M) bytes (= 4*ceil(n/M)); BF16 primary = "
                        "2n bytes, secondary = 0; K / V metadata = 2 tensors each; residual = 2*4*8*128*2 bytes. "
                        "Exact integers; averages / 2^20."),
            "linearity": "metadata bytes are not exactly linear in 1/M (ceil padding of each tensor to M values)",
            "model_anchor": "the int8 totals equal the accepted Exp9 / Exp10 storage model; the model reproduces the "
                            "20 observed Exp1 preset MB values exactly",
            "expected": storage,
            "gates": {"matches_expected": {"rule": "ACCOUNTING-INTEGRITY gate: each observed avg logical KV MB (3-decimal "
                                                   "print) within absolute_mb of avg_logical_kv_mb_full_precision",
                                           "absolute_mb": STORAGE_MATCH_ABS_TOL_MB},
                      "ordering": {"rule": "observed conditions ordered (smallest MB first) exactly as the formula "
                                           "orders them",
                                   "expected_order_smallest_first": {b: expected_order(storage[b])
                                                                     for b in gen.BENCHMARKS}}}},
        "execution_gates": ["exit code 0 and no traceback", "all five rows (bf16, mbf, m32, m64 control, m128) present",
                            "per-method counts equal the frozen counts",
                            "continuation_ppl / NIAH / passage retrieval: bf16 and control within the original tolerances",
                            "hotpotqa / qasper: frozen per-example QA control gate for bf16 and the control, and "
                            "bf16 / control logical KV MB within 0.1 %",
                            "logical storage gates above",
                            "protected paths clean; Exp6-10 evidence and the QA amendment unchanged",
                            "stop at the first failing benchmark; no retry"],
        "completion_criterion": "bf16-metadata and uint8 g32 / g64 (control) / g128 measured across the 5 benchmarks "
                                "with all gates passing (plan: g64 row reproduces canonical numbers, under the rules above)",
        "artifacts": [f"results/mlsys2027/ablations/metadata/{b}.log" for b in gen.BENCHMARKS]
                     + ["results/mlsys2027/ablations/metadata/manifest.json",
                        "results/mlsys2027/ablations/metadata/regression_check.json",
                        "results/mlsys2027/ablations/metadata/metadata_results.json"],
    }


def load_protocol() -> dict:
    committed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if committed != json.loads(json.dumps(build_protocol())):
        raise RuntimeError("exp11_metadata_protocol.json differs from the regenerated protocol")
    if not committed["only_metadata_field_differs_proof"]["holds"]:
        raise RuntimeError("protocol does not prove that each treatment changes exactly one metadata field")
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
                          (EXP8_EVIDENCE_COMMIT, r9.EXP8_FILES), (EXP9_EVIDENCE_COMMIT, r10.EXP9_FILES),
                          (EXP10_EVIDENCE_COMMIT, EXP10_FILES), (AMENDMENT_COMMIT, AMENDMENT_FILES)):
        if e1.run_git("diff", "--name-only", commit, "--", *[str(p.relative_to(ROOT)) for p in paths]):
            raise RuntimeError(f"accepted evidence differs from its frozen commit {commit[:7]}")
    if qg.audit.sha256_lf(qg.AMENDMENT) != AMENDMENT_SHA256_LF:
        raise RuntimeError("QA control amendment hash does not match the pinned hash")
    amendment = qg.load_amendment()
    ok = gen.check()
    if not all(ok.values()):
        raise RuntimeError(f"Exp11 derived scripts differ from their derivation: {ok}")
    for b in gen.BENCHMARKS:
        text = gen.canonical_path(b).read_text(encoding="utf-8")
        if e1.REQUIRED_ALLOWED_LINE not in text or e1.REQUIRED_RABIT2_MARKER not in text:
            raise RuntimeError(f"canonical {b}.py no longer matches the accepted Experiment 1 pins")
    e1.verify_canonical_reference()
    protocol = load_protocol()
    uncommitted = e1.run_git("status", "--short", "--", *[str(p.relative_to(ROOT)) for p in (
        RUNNER_SCRIPT, gen.DERIVED_DIR, HERE / "exp11_metadata_scripts.py", HERE / "test_experiment11_metadata.py",
        PROTOCOL)])
    if uncommitted and not dry_run:
        raise RuntimeError("Refusing to run: Experiment 11 harness has uncommitted changes:\n" + uncommitted)
    if MANIFEST.exists() and json.loads(MANIFEST.read_text(encoding="utf-8")).get("status") == "passed":
        raise RuntimeError(f"{MANIFEST} already records a passed run; refusing to overwrite")
    return {"git_head": e1.run_git("rev-parse", "HEAD"), "rabit_kv2_sha256": e1.EXPECTED_RABIT_SHA256,
            "exp10_evidence_commit": EXP10_EVIDENCE_COMMIT, "amendment_commit": AMENDMENT_COMMIT,
            "amendment_sha256_lf": AMENDMENT_SHA256_LF,
            "canonical_script_sha256": {b: e1.sha256(gen.canonical_path(b)) for b in gen.BENCHMARKS},
            "derived_script_sha256": {b: e1.sha256(gen.derived_path(b)) for b in gen.BENCHMARKS},
            "runner_script_sha256": e1.sha256(RUNNER_SCRIPT), "protocol_sha256": e1.sha256(PROTOCOL),
            "protocol": protocol, "amendment": amendment, "uncommitted_files": uncommitted or None}


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


def integrity(name: str, rc: int, log_text: str, protocol: dict, amendment: dict) -> dict:
    rows = parse_rows(name, log_text)
    checks = {"exit_code_zero": rc == 0, "no_traceback": "Traceback (most recent call last)" not in log_text,
              "all_five_rows_present": all(rows.values())}
    if checks["all_five_rows_present"]:
        checks["counts_exact"] = all(r["count"] == COUNT_COLUMNS[name][1] for r in rows.values())
        L = protocol["logical_storage_expectations"]
        exp = L["expected"][name]
        tol = L["gates"]["matches_expected"]["absolute_mb"]
        checks["kv_mb_matches_expected"] = all(
            abs(rows[m]["avg_logical_kv_mb"] - exp[m]["avg_logical_kv_mb_full_precision"]) <= tol + 1e-9
            for m in CONDITIONS)
        observed = sorted(CONDITIONS, key=lambda m: rows[m]["avg_logical_kv_mb"])
        checks["kv_mb_ordering_matches_formula"] = (
            observed == L["gates"]["ordering"]["expected_order_smallest_first"][name]
            and len({rows[m]["avg_logical_kv_mb"] for m in CONDITIONS}) == len(CONDITIONS))
    reg = e1.check_regression(name, log_text)  # bf16 + control vs the canonical reference (original tolerances)
    out = {"benchmark": name, "rows": rows, "regression": reg}
    if name in QA_BENCHMARKS:
        checks["bf16_and_control_logical_kv_mb_reproduce_canonical"] = all(
            c["within_tolerance"] for c in reg["checks"] if c["metric"].endswith(".avg_kv_mb"))
        gate = qg.evaluate(name, log_text, amendment)
        checks["qa_control_gate_passed"] = gate["passed"]
        out.update(qa_control_gate=gate, control_rule="frozen post-failure per-example QA control gate",
                   aggregate_f1_control_check_report_only=[c for c in reg["checks"] if c["metric"].endswith(".f1_pct")])
    else:
        checks["bf16_and_control_reproduce_canonical"] = reg["all_within_tolerance"]
        out["control_rule"] = "original control-reproduction rule"
    return {**out, "checks": checks, "passed": all(checks.values())}


def main(argv: list[str] | None = None) -> int:
    e1.make_console_encoding_safe()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-protocol", action="store_true", help="write exp11_metadata_protocol.json (pre-commit only)")
    a = ap.parse_args(argv)
    if a.write_protocol:
        if PROTOCOL.exists():
            raise SystemExit(f"{PROTOCOL.name} already exists; the frozen protocol is never overwritten")
        PROTOCOL.write_text(json.dumps(build_protocol(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {PROTOCOL.relative_to(ROOT).as_posix()}")
        return 0
    prov = preflight(a.dry_run)
    print("RABIT-KV MLSys 2027 -- Experiment 11 metadata ablation (logical fake-quant quality + storage)")
    print("Preflight OK:", json.dumps({k: prov[k] for k in ("git_head", "exp10_evidence_commit", "amendment_commit",
                                                             "amendment_sha256_lf", "protocol_sha256")}))
    for c in build_commands():
        print(f"  {c['name']}: {' '.join(c['command'][3:])}")
    if prov["uncommitted_files"]:
        print("  WARNING (dry-run only): uncommitted:\n    " + prov["uncommitted_files"].replace("\n", "\n    "))
    if a.dry_run:
        print("\n--dry-run: nothing executed, no files written.")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment": "Experiment 11 -- metadata ablation (logical fake-quant quality + storage)",
                "methods": METHODS, "conditions": CONDITIONS, "frozen": FROZEN, "status": "running",
                "qa_control_gate": "frozen post-failure per-example reproducibility envelope",
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "provenance": prov, "runs": []}
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    results = []
    for c in build_commands():
        log = ROOT / c["log"]
        rc = e1.stream_command(c["command"], log)
        res = integrity(c["name"], rc, log.read_text(encoding="utf-8", errors="replace"), prov["protocol"],
                        prov["amendment"])
        results.append(res)
        manifest["runs"].append({"name": c["name"], "returncode": rc, "passed": res["passed"],
                                 "control_rule": res["control_rule"], "log_sha256": e1.sha256(log)})
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not res["passed"]:  # no retry; stop at the first failing benchmark
            break
    REGRESSION_CHECK.write_text(json.dumps([r["regression"] for r in results], indent=2) + "\n", encoding="utf-8")
    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ok = len(results) == len(e1.RUNS) and all(r["passed"] for r in results) and not protected_status()
    manifest.update(status="passed" if ok else "failed", completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nEXPERIMENT 11 {manifest['status'].upper()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
