"""Builds summary.json for the registered Llama canonical long-context Attempt 1 from record.json / result.json
(stdlib only; descriptive; no threshold). Usage (repository root):
    python results/mlsys2027/canonical_quality_v2/long_context/attempt_1/build_summary.py
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
rec = json.loads((HERE / "record.json").read_text(encoding="utf-8"))
raw = (HERE / "result.json").read_bytes()
res = json.loads(raw.decode("utf-8"))
ev, st = rec["evaluation"], rec["evaluation"]["statistics"]


def paired(x: dict) -> dict:
    r = x["robustness"]
    return {k: x[k] for k in ("n", "bf16", "rabit", "delta_points", "relative_delta_pct", "ci_low", "ci_high", "confidence",
                              "resamples", "seed", "examples_score_changed", "examples_rabit_worse", "examples_rabit_better",
                              "examples_prediction_text_changed")} | {
        "ci_contains_zero": x["ci_low"] <= 0.0 <= x["ci_high"],
        "per_example_delta_points": r["per_unit_delta_summary"],
        "concentration": {"top1_by_abs": r["top1_by_abs"], "top3_by_abs": r["top3_by_abs"], "worst1": r["worst1"],
                          "worst3": r["worst3"]},
        "worst_examples": x["worst_examples"],
        "changed_examples": [{"key": k, "bf16": b, "rabit": q} for k, b, q in zip(
            x["per_example"]["keys"], x["per_example"]["bf16"], x["per_example"]["rabit"]) if b != q]}


def arm_facts(task: str) -> dict:
    rows = res["tasks"][task]
    return {a: {"mean_kept_tokens": sum(len(r["arms"][a]["generated_ids"]) for r in rows) / len(rows),
                "stopped_on_eos": sum(r["arms"][a]["stopped_on_eos"] for r in rows),
                "seconds_total": sum(r["arms"][a]["seconds"] for r in rows)} for a in ("bf16", "rabit")}


niah = st["niah"]
summary = {
    "kind": "canonical-quality-v2 Llama long-context suite: registered Attempt 1 -- summary (logical quality; NOT physical "
            "serving evidence)",
    "attempt": rec["attempt"], "valid": rec["valid"], "status": rec["status"], "source_commit": rec["source_commit"],
    "gates": ev["gates"], "bf16_validity_checks": ev["bf16_control"], "examples_equal_exp12": ev["examples_equal_exp12"],
    "started_utc": rec["started_utc"], "completed_utc": rec["completed_utc"], "elapsed_s": rec["elapsed_s"],
    "container_seconds": res["timing"], "modal_returncode": rec["modal_returncode"], "timed_out": rec["timed_out"],
    "app_ids": rec["app_ids_new"], "app_final_states": rec["cleanup"]["final_states"],
    "cleanup_verified": rec["cleanup"]["verified"], "result_sha256": rec["result_sha256"],
    "result_sha256_matches_file": hashlib.sha256(raw).hexdigest() == rec["result_sha256"],
    "protocol_sha256_lf": rec["protocol_sha256_lf"], "hardware": res["hardware"]["gpus"],
    "runtime_environment": res["runtime_environment"], "environment": res["environment"],
    "model": {k: res["model"][k] for k in ("model_id", "model_revision", "manifest_sha256", "files_checked", "passed")},
    "datasets": res["datasets"], "prompt_sets": res["prompt_sets"], "policy": res["policy"],
    "canonical_rabit_quality_sha256_lf": res["files"]["sha256_lf"]["canonical_rabit_quality.py"],
    "artifact_persistence": res["artifact_persistence"], "incremental_rows_copied": rec["incremental_rows_copied"],
    "niah": {**{k: niah[k] for k in ("bf16", "rabit", "delta_points", "cases_outcome_changed", "cases_answer_text_changed",
                                     "statistics")}, "generation": arm_facts("niah")},
    "passage_retrieval": {**paired(st["passage_retrieval"]), "generation": arm_facts("passage_retrieval")},
    "hotpotqa": {"primary_legacy_scorer": paired(st["hotpotqa"]["primary_legacy_scorer"]),
                 "secondary_official_scorer": paired(st["hotpotqa"]["secondary_official_scorer"]),
                 "generation": arm_facts("hotpotqa")},
    "legacy_comparison": st["legacy_comparison"],
}
(HERE / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

print("VALID", rec["valid"], all(ev["gates"].values()), len(ev["gates"]), rec["elapsed_s"], res["timing"], rec["app_ids_new"],
      [(v["State"], v["Tasks"]) for v in rec["cleanup"]["final_states"].values()], rec["source_commit"][:7])
print("HW", res["hardware"]["gpus"], res["runtime_environment"]["passed"], summary["model"])
print("BF16 CHECKS", json.dumps({k: {x: v[x] for x in v if x not in ("rule", "historical_exp12_rule")} for k, v in ev["bf16_control"].items()}))
for a in ("bf16", "rabit"):
    print("NIAH", a, niah[a]["passed"], niah[a]["cases"], niah[a]["per_context"], niah[a]["failed_coordinates"])
print("NIAH changed", niah["cases_outcome_changed"], "answer text changed", niah["cases_answer_text_changed"])
for name, x in (("PASSAGE", summary["passage_retrieval"]), ("HOTPOT primary", summary["hotpotqa"]["primary_legacy_scorer"]),
                ("HOTPOT secondary", summary["hotpotqa"]["secondary_official_scorer"])):
    print(name, {k: x[k] for k in ("n", "bf16", "rabit", "delta_points", "relative_delta_pct", "ci_low", "ci_high",
                                   "ci_contains_zero", "examples_score_changed", "examples_rabit_worse",
                                   "examples_rabit_better", "examples_prediction_text_changed")})
    print("   delta dist", x["per_example_delta_points"])
    c = x["concentration"]
    print("   top1", c["top1_by_abs"]["units"], c["top1_by_abs"]["share_of_aggregate_delta"], "top3 share", c["top3_by_abs"]["share_of_aggregate_delta"],
          "worst3 share", c["worst3"]["share_of_aggregate_delta"])
    print("   changed", [(e["key"], round(e["bf16"], 3), round(e["rabit"], 3)) for e in x["changed_examples"]])
for e in summary["hotpotqa"]["primary_legacy_scorer"]["worst_examples"]:
    print("   WORST", e["key"], round(e["delta_points"], 1), e["answers"], "| bf16:", repr(e["bf16_prediction"]), "| rabit:", repr(e["rabit_prediction"]))
for k, v in st["legacy_comparison"].items():
    print("LEGACY", k, {x: v[x] for x in ("label", "n", "bf16", "rabit2", "delta_points")}, v["summary_lines_in_log"], v["bf16_reproducibility"])
print("GEN", {t: summary[t]["generation"] for t in ("niah", "passage_retrieval", "hotpotqa")})
