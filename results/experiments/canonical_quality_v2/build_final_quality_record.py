"""Builds the acceptance record of the Llama canonical long-context Attempt 1, the FINAL paper-ready canonical quality
record (continuation PPL + long-context) and the updated evidence index from the accepted summaries (stdlib only;
documentation / freezing; runs nothing). Usage (repository root):
    python results/mlsys2027/canonical_quality_v2/build_final_quality_record.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
lc = json.loads((HERE / "long_context/attempt_1/summary.json").read_text(encoding="utf-8"))
ppl = json.loads((HERE / "canonical_ppl_paper_ready_record.json").read_text(encoding="utf-8"))
lproto = json.loads((ROOT / "benchmarks/mlsys2027/canonical_longctx_protocol.json").read_text(encoding="utf-8"))

WORDING = ("On Llama-3.1-8B, the frozen canonical RABIT operating point preserves exact retrieval on the tested NIAH and "
           "Passage Retrieval suites. On the 100-example HotpotQA subset, canonical RABIT is approximately 1 F1 point below "
           "BF16, with the paired confidence interval spanning zero and the observed differences concentrated in a small "
           "number of examples.")
NOT_TO_WRITE = ["HotpotQA is statistically equivalent", "HotpotQA is proven unaffected", "RABIT causes no degradation",
                "quality is universally preserved"]
CONCLUSION = ("RABIT's physical representation and canonical cache semantics transfer across the tested Llama and Qwen "
              "geometries, but the quality robustness of a single frozen low-bit operating point is strongly "
              "model-dependent.")
CLAIMS = {
    "supported": [CONCLUSION, WORDING,
                  "The same frozen operating point exhibits strongly model-dependent quality behavior (Llama continuation "
                  "PPL +1.56 %; Qwen continuation PPL +1555.9 %)."],
    "not_supported": ["RABIT quality generalizes across models.",
                      "The frozen K3/V2/G32/R4/META8g64 operating point is universally suitable.",
                      "Qwen degradation is caused by GQA, projection bias, outliers, fallback attention, or any other "
                      "specific mechanism.", *NOT_TO_WRITE],
    "note": "a confidence interval spanning zero does NOT establish equivalence; mechanism of the Qwen degradation unknown"}
EVIDENCE = {
    "canonical_implementation": "c360697 (canonical_rabit_quality.py, sha256_lf 195cb489...)",
    "cpu_canonical_vs_cpu_oracle_parity": "8fa9a9c",
    "llama_identity_audit": "3bde966",
    "cuda_conformance_llama_1k": "ec80638", "cuda_conformance_qwen_1k": "fd8d275", "cuda_conformance_llama_16k": "2637457",
    "validity_gate_amendment": "d7ba819",
    "continuation_ppl_attempt_2": "10b0973 (Llama run 01cced9); accepted bae70af",
    "long_context_harness_and_protocol": "a4d8ac5 (protocol sha256_lf 64513be6...)", "long_context_smoke_test": "432033c",
    "long_context_attempt_1": "7896fcd",
    "invalid": "continuation PPL Attempt 1: INVALID, NO QUALITY RESULT (5d8d212, 3cba625); never pooled"}


def r(x, n=2):
    return round(x, n)


def hot(x: dict) -> dict:
    return {"n": x["n"], "bf16_f1": x["bf16"], "rabit_f1": x["rabit"], "delta_points": x["delta_points"],
            "relative_delta_pct": x["relative_delta_pct"], "ci95_paired_bootstrap": [x["ci_low"], x["ci_high"]],
            "ci_contains_zero": x["ci_contains_zero"], "scores_changed": x["examples_score_changed"],
            "worse": x["examples_rabit_worse"], "better": x["examples_rabit_better"],
            "identical_scores": x["n"] - x["examples_score_changed"],
            "median_paired_delta_points": x["per_example_delta_points"]["median"],
            "prediction_text_changed": x["examples_prediction_text_changed"],
            "changed_examples": x["changed_examples"],
            "rounded": {"bf16": r(x["bf16"]), "rabit": r(x["rabit"]), "delta": r(x["delta_points"]),
                        "relative_delta_pct": r(x["relative_delta_pct"]), "ci95": [r(x["ci_low"]), r(x["ci_high"])]}}


niah, pr, hq = lc["niah"], lc["passage_retrieval"], lc["hotpotqa"]
leg = lc["legacy_comparison"]
long_context = {
    "status": "ACCEPTED (2026-10-03): the final canonical long-context quality evidence", "evidence_commit": "7896fcd",
    "registered_attempt": 1, "validity_gates_passed": f"{sum(lc['gates'].values())} / {len(lc['gates'])}",
    "model": lc["model"], "hardware": lc["hardware"][0]["name"], "runtime": lc["runtime_environment"]["runtime"],
    "modal_app": lc["app_ids"][0], "wall_clock_seconds": lc["elapsed_s"],
    "protocol": {"file": "benchmarks/mlsys2027/canonical_longctx_protocol.json (sha256_lf " + lc["protocol_sha256_lf"] + ")",
                 "arms": "BF16 vs canonical RABIT (K3 / V2 / G32 / R4 / META8g64), identical prompt token ids per pair",
                 "generation": lproto["generation"], "tasks": lproto["tasks"], "prompt_sets": lproto["prompt_sets"],
                 "statistics": lproto["statistics"]},
    "niah": {"n": 57, "bf16_passed": niah["bf16"]["passed"], "rabit_passed": niah["rabit"]["passed"], "delta": 0,
             "per_context": {c: {"bf16": f"{niah['bf16']['per_context'][c]['passed']} / {niah['bf16']['per_context'][c]['cases']}",
                                 "rabit": f"{niah['rabit']['per_context'][c]['passed']} / {niah['rabit']['per_context'][c]['cases']}"}
                             for c in ("4096", "8192", "16384")},
             "score_changes": len(niah["cases_outcome_changed"]), "answer_text_changed": niah["cases_answer_text_changed"],
             "inferential_statistics": "none (deterministic grid; cases are not independent draws)"},
    "passage_retrieval": {"n": pr["n"], "bf16": pr["bf16"], "rabit": pr["rabit"], "delta_points": pr["delta_points"],
                          "ci95_paired_bootstrap": [pr["ci_low"], pr["ci_high"]], "score_changes": pr["examples_score_changed"],
                          "prediction_text_changed": pr["examples_prediction_text_changed"]},
    "hotpotqa_primary_legacy_compatible_scorer": hot(hq["primary_legacy_scorer"]),
    "hotpotqa_secondary_official_scorer": hot(hq["secondary_official_scorer"]),
    "hotpotqa_scorer_note": "PRIMARY = the accepted Exp12 scorer verbatim (articles not removed), for comparability with "
                            "Exp12; SECONDARY = standards-aligned LongBench qa_f1_score; both preregistered and computed "
                            "from the identical predictions",
    "concentration": {"niah": "0 / 57 score changes", "passage_retrieval": "0 / 200 score changes",
                      "hotpotqa_primary": "8 / 100 score changes (5 worse, 3 better); 92 / 100 identical scores; median "
                                          "paired delta 0; the difference is concentrated rather than broadly distributed",
                      "notable_examples_descriptive_only": hq["primary_legacy_scorer"]["worst_examples"],
                      "rule": "nothing is excluded; formatting-sensitive examples are NOT removed post hoc"},
    "bf16_reproducibility_vs_exp12": {
        "niah": "57 / 57 reproduced", "passage_retrieval": "200 / 200 scores reproduced",
        "hotpotqa": "99 / 100 scores reproduced; one example (dataset index 289) differs, the known greedy-decoding "
                    "run-to-run variability; the accepted BF16 validity gate passed; not rerun",
        "detail": {k: leg[k]["bf16_reproducibility"] for k in leg}},
    "bf16_validity_checks": lc["bf16_validity_checks"],
    "legacy_exp12_side_by_side_provenance_only": {
        k: {"label": leg[k]["label"], "legacy_bf16": leg[k]["bf16"], "legacy_rabit2": leg[k]["rabit2"],
            "source": leg[k]["summary_lines_in_log"]} for k in leg},
    "interpretation": WORDING}

final = {
    "kind": "FINAL PAPER-READY CANONICAL QUALITY RECORD (canonical-quality-v2; logical quality; NOT physical serving evidence)",
    "status": "FROZEN 2026-10-03",
    "policy": "K3 / V2 / G32 / R4 / META8g64 (one frozen operating point; never retuned)",
    "summary": {
        "llama_continuation_ppl": "7.515 -> 7.633 (+1.56 %)",
        "llama_long_context": {"niah": "no score loss (57 / 57 vs 57 / 57)",
                               "passage_retrieval": "no score loss (100.0 vs 100.0)",
                               "hotpotqa": "about -1 F1 (59.30 -> 58.27); paired 95 % CI [-4.24, +2.20] spans zero"},
        "qwen_continuation_ppl": "6.787 -> 112.39 (+1555.9 %)",
        "conclusion": CONCLUSION},
    "continuation_ppl": {"status": ppl["status"], "protocol": ppl["protocol"], "models": ppl["models"],
                         "interpretation": ppl["interpretation"], "qualification": ppl["qualification"]},
    "long_context": long_context,
    "claims": CLAIMS,
    "limitations": [
        "Logical quality (HF transformers; canonical cache semantics); not a physical serving measurement.",
        "One frozen operating point; no retuning and no ablation under canonical semantics.",
        "Long-context evidence is Llama-only: Qwen NIAH / Passage Retrieval / HotpotQA were NOT run.",
        "Qwen shows severe continuation-PPL degradation at this operating point; the mechanism is undiagnosed and no "
        "causal decomposition of the legacy-vs-canonical difference has been run.",
        "HotpotQA: 100 examples of one pinned 8k+ bucket; the confidence interval spans zero, which does NOT establish "
        "equivalence; the bootstrap is conditional on this run's greedy generations and does not model run-to-run GPU "
        "variability (one BF16 example differs from Exp12).",
        "NIAH and Passage Retrieval are saturated for BF16 (100 %) on these suites; they show no loss but cannot resolve "
        "small effects. NIAH cases share one needle / filler construction and are not independent draws.",
        "Continuation PPL: WikiText-2, N = 32 windows, 4096 scored tokens per arm.",
        "Qasper and multilingual PPL have NOT been run under canonical semantics.",
        "CPU and CUDA canonical states are not bit-identical; validity rests on same-device conformance with the "
        "independent oracle (no cross-device tolerance).",
        "The HotpotQA primary scorer keeps the historical Exp12 article behaviour; the official scorer is reported as "
        "secondary.",
        "The Llama revision was recovered by content matching of the preserved historical cache; the dataset file hashes "
        "are newly recovered provenance (Exp12 recorded commits only)."],
    "quality_experiments_complete": ["canonical evaluator validation", "CPU / CUDA semantic conformance",
                                     "two-model canonical continuation PPL", "Llama canonical long-context validation"],
    "not_to_be_launched_unless_reopened": ["Qwen NIAH", "Qwen Passage Retrieval", "Qwen HotpotQA", "Qasper", "multilingual",
                                           "K / V / group / residual / metadata ablations", "quality retuning"],
    "legacy": "Exp1, Exp2, Exp7-Exp11, Exp12 (quality) and Exp14 (legacy quality) are LEGACY LOGICAL-EVALUATOR RESULTS: "
              "kept unchanged; shown side by side for provenance only; never pooled with canonical measurements",
    "evidence": EVIDENCE,
    "full_detail": ["continuation_ppl/two_model_attempt_2/summary.json", "long_context/attempt_1/summary.json"]}
(HERE / "canonical_quality_final_record.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n",
                                                          encoding="utf-8", newline="\n")

acceptance = {"kind": "canonical-quality-v2 Llama long-context suite: ACCEPTANCE RECORD of registered Attempt 1",
              "status": "accepted", "accepted_utc_date": "2026-10-03", "evidence_commit": "7896fcd",
              "authoritative": "the final canonical long-context quality evidence for the paper",
              "results": {k: long_context[k] for k in ("niah", "passage_retrieval", "hotpotqa_primary_legacy_compatible_scorer",
                                                       "hotpotqa_secondary_official_scorer")},
              "interpretation": WORDING, "do_not_write": NOT_TO_WRITE, "concentration": long_context["concentration"],
              "bf16_reproducibility_vs_exp12": long_context["bf16_reproducibility_vs_exp12"],
              "follow_up_experiments_launched": "none"}
(HERE / "long_context/attempt_1/acceptance_record.json").write_text(
    json.dumps(acceptance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

index = json.loads((HERE / "EVIDENCE_INDEX.json").read_text(encoding="utf-8"))
index["kind"] = "CANONICAL QUALITY EVIDENCE INDEX (canonical-quality-v2); FINAL status as of 2026-10-03"
index["authoritative_canonical_quality_evidence"] = [
    index["authoritative_canonical_quality_evidence"][0],
    {"item": "Llama and Qwen CPU / CUDA semantic conformance (1k both models; 16k Llama)",
     "commits": ["8fa9a9c", "ec80638", "fd8d275", "2637457", "d7ba819"],
     "paths": ["results/mlsys2027/canonical_quality_v2/cuda_conformance_diagnostic/"]},
    index["authoritative_canonical_quality_evidence"][2],
    {"item": "Registered Llama long-context Attempt 1: NIAH 57/57 vs 57/57; Passage Retrieval 100.0 vs 100.0; HotpotQA "
             "59.30 vs 58.27 (-1.03 F1, CI [-4.24, +2.20])", "commits": ["a4d8ac5", "432033c", "7896fcd"],
     "paths": ["results/mlsys2027/canonical_quality_v2/long_context/attempt_1/",
               "results/mlsys2027/canonical_quality_v2/canonical_quality_final_record.json"]}]
index["complete"] = final["quality_experiments_complete"]
index["llama_long_context_validation"] = "COMPLETE"
index["final_paper_ready_record"] = "results/mlsys2027/canonical_quality_v2/canonical_quality_final_record.json"
index["not_run_under_canonical_semantics"] = ["Qwen NIAH / Passage Retrieval / HotpotQA", "Qasper", "multilingual PPL",
                                              "K / V / group / residual / metadata ablations"]
index["not_to_be_launched_unless_reopened"] = final["not_to_be_launched_unless_reopened"]
index["generalization_claims"] = CLAIMS
(HERE / "EVIDENCE_INDEX.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                                          newline="\n")
for k in ("hotpotqa_primary_legacy_compatible_scorer", "hotpotqa_secondary_official_scorer"):
    x = long_context[k]
    print(k, x["bf16_f1"], x["rabit_f1"], x["delta_points"], x["relative_delta_pct"], x["ci95_paired_bootstrap"],
          x["scores_changed"], x["worse"], x["better"], x["identical_scores"], x["median_paired_delta_points"])
print(long_context["niah"]["per_context"], long_context["passage_retrieval"])
