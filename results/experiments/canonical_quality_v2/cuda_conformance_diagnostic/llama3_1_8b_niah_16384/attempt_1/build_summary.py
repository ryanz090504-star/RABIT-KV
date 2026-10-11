"""Aggregates result.json (all layers of the model) into summary.json. Stdlib only; descriptive; no tolerance, no threshold.
Usage (repository root):
    python results/mlsys2027/canonical_quality_v2/cuda_conformance_diagnostic/llama3_1_8b_niah_16384/attempt_1/build_summary.py
"""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
res = json.loads((HERE / "result.json").read_text(encoding="utf-8"))
rec = json.loads((HERE / "record.json").read_text(encoding="utf-8"))
layers = res["layers"]


def aggregate(section: str) -> dict:
    """Per field over the layers: layers equal, differing / total elements, worst errors, first mismatch."""
    out = {}
    for name in layers[0][section]:
        rows = [(r["layer"], r[section][name]) for r in layers]
        equal = [bool(v.get("equal")) and v.get("bitwise_equal", True) is True for _, v in rows]
        agg = {"layers_equal": sum(equal), "layers": len(rows)}
        if "total" in rows[0][1]:
            agg.update(dtype=rows[0][1]["dtype"][0], total_elements=sum(v["total"] for _, v in rows),
                       differing_elements=sum(v.get("differing", 0) for _, v in rows))
            for key in ("max_abs_error", "max_rel_error", "max_abs_code_difference"):
                vals = [v[key] for _, v in rows if v.get(key) is not None]
                if vals:
                    agg[key] = max(vals)
            first = next(((li, v["first_mismatch"]) for li, v in rows if "first_mismatch" in v), None)
            if first:
                agg["first_mismatch"] = {"layer": first[0], **first[1]}
        out[name] = agg
    return out


def total(section: str, key: str) -> dict:
    out = collections.Counter()
    for r in layers:
        for side, d in r[section].items():
            if isinstance(d.get(key), (int, float)):
                out[side] += d[key]
    return dict(out)


probe = {}
for name in ("k.primary_scale", "v.primary_scale"):
    rows = [r["division_probe"][name] for r in layers]
    probe[name] = {"divisor": rows[0]["divisor"],
                   "layers_with_max_and_min_bit_identical_on_both_devices": sum(
                       x["max_and_min_bit_identical_on_both_devices"] for x in rows),
                   **{k: sum(x[k] for x in rows) for k in (
                       "elements", "device_scale_equals_cpu_true_division",
                       "device_scale_equals_cpu_multiply_by_fp32_reciprocal", "cpu_scale_equals_cpu_true_division")}}
codes = {s: {k: sum(r["codes_vs_primary_parameters"][s][k] for r in layers)
             for k in layers[0]["codes_vs_primary_parameters"][s]} for s in ("k", "v")}

summary = {
    "kind": "canonical-quality-v2 GPU semantic-conformance diagnostic: summary of result.json (descriptive; NOT a "
            "quality result; no scoring / PPL / generation)",
    "diagnostic_valid": rec["diagnostic_valid"], "classification": rec["classification"],
    "source_commit": rec["source_commit"], "result_sha256": rec["result_sha256"], "app_ids": rec["app_ids_new"],
    "app_cleanup_verified": rec["cleanup"]["verified"], "app_final_states": rec["cleanup"]["final_states"],
    "hardware": res["hardware"]["gpus"], "environment": res["environment"],
    "canonical_implementation_sha256_lf": res["files"]["sha256_lf"]["benchmarks/mlsys2027/canonical_rabit_quality.py"],
    "oracle": res["oracle"], "model": {k: res["model"][k] for k in ("model_id", "model_revision", "manifest_sha256",
                                                                   "files_checked", "passed")},
    "window": {**res["dataset"], **res["window"]},
    "raw_kv": {"dtype": layers[0]["raw"]["dtype"], "shape": layers[0]["raw"]["shape"], "device": layers[0]["raw"]["device"],
               "host_copy_bitwise_identical_layers": sum(r["raw"]["host_copy_bitwise_identical"] for r in layers),
               "sha256_per_layer": [{"layer": r["layer"], "k": r["raw"]["k_sha256"], "v": r["raw"]["v_sha256"]}
                                    for r in layers]},
    "primary_cuda_canonical_vs_cuda_oracle": {
        "layers_all_fields_bit_exact": sum(r["summary"]["cuda_canonical_equals_cuda_oracle"] for r in layers),
        "layers_canonical_cache_bit_exact": sum(r["summary"]["canonical_cache_equals_cuda_oracle"] for r in layers),
        "layers_accepted_t1_comparison_clean": sum(r["accepted_t1_comparison_on_cuda"] == [] for r in layers),
        "layers_residual_equals_raw": sum(all(r["residual_equals_raw"].values()) for r in layers),
        "fields": aggregate("cuda_canonical_vs_cuda_oracle"),
        "canonical_cache_fields": aggregate("canonical_cache_vs_cuda_oracle")},
    "cpu_canonical_vs_cpu_oracle": {
        "layers_all_fields_bit_exact": sum(r["summary"]["cpu_canonical_equals_cpu_oracle"] for r in layers)},
    "cpu_oracle_vs_cuda_oracle": {
        "layers_all_fields_bit_exact": sum(r["summary"]["cpu_oracle_equals_cuda_oracle"] for r in layers),
        "fields": aggregate("cpu_oracle_vs_cuda_oracle")},
    "descriptive_cpu_canonical_vs_cuda_canonical": {
        "note": "a = CPU, b = CUDA; descriptive only -- no tolerance is defined and none is used",
        "layers_all_fields_bit_exact": sum(r["summary"]["cpu_canonical_equals_cuda_canonical"] for r in layers),
        "fields": aggregate("cpu_canonical_vs_cuda_canonical"),
        "trace_fields_in_dependency_order": aggregate("trace_cpu_vs_cuda"),
        "earliest_divergence": {s: dict(collections.Counter(str(r["earliest_divergence"][s]) for r in layers))
                                for s in ("k", "v")},
        "codes_vs_primary_parameters": codes, "division_probe": probe},
    "attempt1_gate_reevaluated": res["attempt1_gate_reevaluated"],
    "synthetic_cuda_canonical_vs_cuda_oracle": {
        "lengths": res["synthetic"]["lengths"], "device": res["synthetic"]["device"],
        **{g: {k: (len(v) if k.endswith("failures") else v) for k, v in G.items()}
           for g, G in res["synthetic"]["geometries"].items()}},
    "no_scoring": res["no_scoring"],
}
(HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")

d = summary["descriptive_cpu_canonical_vs_cuda_canonical"]
print("PRIMARY", {k: v for k, v in summary["primary_cuda_canonical_vs_cuda_oracle"].items() if k.startswith("layers")})
print("PRIMARY fields not equal in all layers:", [k for k, v in summary["primary_cuda_canonical_vs_cuda_oracle"]["fields"].items() if v["layers_equal"] != len(layers)])
print("EARLIEST", d["earliest_divergence"])
for name, v in d["trace_fields_in_dependency_order"].items():
    print(f"TRACE {name:32} eq_layers={v['layers_equal']:2} diff={v['differing_elements']:>9}/{v['total_elements']:<10} "
          f"abs={v.get('max_abs_error', v.get('max_abs_code_difference'))} rel={v.get('max_rel_error')}")
for name, v in d["fields"].items():
    if "total_elements" in v:
        print(f"FIELD {name:32} eq_layers={v['layers_equal']:2} diff={v['differing_elements']:>9}/{v['total_elements']:<10} "
              f"abs={v.get('max_abs_error', v.get('max_abs_code_difference'))} rel={v.get('max_rel_error')} {v['dtype']}")
    else:
        print(f"FIELD {name:32} eq_layers={v['layers_equal']}")
print("CODES", json.dumps(codes))
print("PROBE", json.dumps(probe))
print("ORACLE cpu-vs-cuda unequal fields:", sorted(k for k, v in summary["cpu_oracle_vs_cuda_oracle"]["fields"].items() if v["layers_equal"] != len(layers)))
print("GATE", summary["attempt1_gate_reevaluated"])
print("SYN", {g: summary["synthetic_cuda_canonical_vs_cuda_oracle"][g] for g in ("llama3_1_8b", "qwen2_5_7b")})
print("HW", summary["hardware"], summary["environment"]["torch"], summary["environment"]["torch_cuda"], summary["environment"]["python"])
print("WINDOW", {k: v for k, v in summary["window"].items() if k not in ("url",)}, "APPS", summary["app_final_states"])
