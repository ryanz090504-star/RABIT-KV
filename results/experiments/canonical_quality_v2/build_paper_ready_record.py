"""Builds the acceptance record, the paper-ready result record and the canonical quality evidence index from the
accepted Attempt-2 summary (stdlib only; documentation / freezing; runs nothing). Usage (repository root):
    python results/mlsys2027/canonical_quality_v2/build_paper_ready_record.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
A2 = HERE / "continuation_ppl/two_model_attempt_2"
s = json.loads((A2 / "summary.json").read_text(encoding="utf-8"))
proto = json.loads((HERE.parents[2] / "benchmarks/mlsys2027/canonical_ppl_protocol.json").read_text(encoding="utf-8"))

CASE_1 = {"case": 1,
          "condition": proto["interpretation_cases"]["case_1"]["condition"],
          "interpretation": "The frozen operating point shows model-specific quality sensitivity; mechanism remains undiagnosed.",
          "applied": "descriptively, by review decision; no numerical definition of 'near BF16' is introduced"}
QUALIFICATION = {
    "statement": "The old logical evaluator materially distorted the MAGNITUDE of the Qwen degradation (legacy Qwen RABIT "
                 "PPL 299.62 vs canonical 112.39), but the severe Qwen sensitivity remains under canonical semantics. The "
                 "Qwen degradation is therefore NOT merely an old-evaluator artifact.",
    "no_causal_decomposition": "The change from 299.62 to 112.39 is not attributed to the metadata-layout fix or to the "
                               "aging fix individually; no causal decomposition has been run."}
CLAIMS = {
    "supported": [
        "RABIT's physical representation and canonical cache semantics transfer across the tested Llama and Qwen KV "
        "geometries.",
        "The same frozen operating point exhibits strongly model-dependent quality behavior."],
    "not_supported": [
        "RABIT quality generalizes across models.",
        "The frozen K3/V2/G32/R4/META8g64 operating point is universally suitable.",
        "Qwen degradation is caused by GQA, projection bias, outliers, fallback attention, or any other specific "
        "mechanism."],
    "mechanism": "unknown"}
EVIDENCE = {
    "canonical_implementation": "c360697 (benchmarks/mlsys2027/canonical_rabit_quality.py, sha256_lf 195cb489...)",
    "cpu_canonical_vs_cpu_oracle_parity": "8fa9a9c (276 / 276 cases, bit-exact)",
    "llama_identity_audit": "3bde966",
    "llama_cuda_conformance": "ec80638 (32 / 32 layers, every canonical field, bit-exact vs the frozen oracle on CUDA)",
    "qwen_cuda_conformance": "fd8d275 (28 / 28 layers, every canonical field, bit-exact vs the frozen oracle on CUDA)",
    "post_failure_validity_gate_amendment": "d7ba819",
    "final_harness_and_protocol": "05eb29e (protocol sha256_lf cfa990a7...)",
    "attempt_2_llama_run_pushed_uninspected": "01cced9",
    "attempt_2_evidence": "10b0973",
    "attempt_1": "INVALID HARNESS / VALIDITY-GATE FAILURE, NO QUALITY RESULT PRODUCED (5d8d212, 3cba625, 2a5aec0); never pooled"}


def model_block(key: str) -> dict:
    m = s["models"][key]
    w, c, leg, ctl = (m["paired_window_delta_nll"], m["concentration_descriptive"], m["legacy_accepted_evidence"],
                      m["bf16_reproducibility"])
    return {
        "model_id": m["model"]["model_id"], "revision": m["model"]["model_revision"],
        "manifest_sha256": m["model"]["manifest_sha256"], "files": m["model"]["files_checked"],
        "geometry": m["geometry"], "hardware": m["hardware"][0]["name"],
        "runtime": m["runtime_environment"]["runtime"], "modal_app": m["app_ids"][0],
        "scored_tokens_per_arm": m["scored_tokens_per_arm"], "windows": 32,
        "bf16_ppl": m["bf16_ppl"], "rabit_ppl": m["rabit_ppl"], "relative_delta_pct": m["delta_pct"],
        "relative_delta_pct_ci95_paired_bootstrap": m["delta_pct_ci95_paired_bootstrap"],
        "mean_nll": {"bf16": m["aggregate"]["bf16"]["mean_nll"], "rabit": m["aggregate"]["rabit"]["mean_nll"]},
        "rounded": {"bf16_ppl": round(m["bf16_ppl"], 3), "rabit_ppl": round(m["rabit_ppl"], 3 if m["rabit_ppl"] < 100 else 2),
                    "relative_delta_pct": round(m["delta_pct"], 2 if m["delta_pct"] < 100 else 1),
                    "ci95_pct": [round(x, 2 if x < 100 else 0) for x in m["delta_pct_ci95_paired_bootstrap"]]},
        "per_window_paired_nll_delta": {k: w[k] for k in ("mean", "median", "stdev", "min", "max", "windows_positive",
                                                          "windows_negative")},
        "largest_positive_windows": w["largest_positive"][:3], "smallest_windows": w["largest_negative_or_smallest"][:3],
        "concentration": {k: c[k] for k in ("share_of_summed_window_delta_from_top_1", "share_from_top_3",
                                            "share_from_top_8_of_32", "aggregate_delta_pct_without_top_3_windows")}
        | {"tokens_with_positive_delta_of_4096": c["token_level"]["tokens_with_positive_delta"]},
        "legacy_logical_evaluator_context_only": {
            "source": leg["summary_lines_in_log"], "bf16_ppl": leg["bf16_aggregate_ppl"],
            "legacy_rabit_ppl": leg["rabit2_legacy_aggregate_ppl"], "legacy_relative_delta_pct": leg["rabit2_legacy_delta_pct"]},
        "bf16_reproducibility": {"legacy_bf16_ppl": ctl["legacy_bf16_aggregate_ppl"],
                                 "bf16_batched_control_ppl": ctl["bf16_batched_aggregate_ppl"],
                                 "relative_difference": ctl["relative_difference"],
                                 "stepwise_vs_batched_relative_difference": ctl["stepwise_vs_batched_relative_difference"]}}


models = {k: model_block(k) for k in ("llama3_1_8b", "qwen2_5_7b")}
paper = {
    "kind": "PAPER-READY RESULT RECORD: canonical continuation perplexity, two models (logical quality under canonical "
            "RABIT semantics; NOT physical serving evidence)",
    "status": "ACCEPTED (2026-10-03): the authoritative canonical continuation-PPL evidence",
    "protocol": {"dataset": "WikiText-2 test (raw text), sha256 " + proto["dataset"]["sha256"],
                 "windows": "N = 32 consecutive non-overlapping windows; context 1024 tokens; continuation 128 tokens; no BOS",
                 "scoring": "BF16 prefill; first continuation token scored from the prefill logit; remaining tokens "
                            "teacher-forced one at a time; canonical cache aging at every decode step; identical paired "
                            "token inputs in both arms",
                 "arms": "BF16 vs canonical RABIT (canonical_rabit_quality.py, c360697)",
                 "policy": proto["policy"], "statistics": "PPL = exp(total NLL / total tokens); delta = 100 * (exp(mean "
                 "per-window NLL difference) - 1); paired percentile bootstrap over windows, 10000 resamples, seed 20270929",
                 "hardware": "one NVIDIA H100 80GB per model", "registered_attempt": 2,
                 "protocol_file": "benchmarks/mlsys2027/canonical_ppl_protocol.json (sha256_lf cfa990a7...; commit 05eb29e)"},
    "models": models,
    "interpretation": CASE_1, "qualification": QUALIFICATION, "generalization_claims": CLAIMS,
    "per_window_concentration_summary": {
        "llama3_1_8b": "small and spread: 30 of 32 windows slightly worse, 2 slightly better",
        "qwen2_5_7b": "broadly distributed: all 32 windows worse; the top 3 windows contribute about 12 % of the summed "
                      "window delta"},
    "limitations": [
        "Logical quality (HF transformers, quantize-dequantize of the cache); not a physical serving measurement.",
        "One benchmark (WikiText-2 continuation perplexity), N = 32 windows, 4096 scored tokens per arm, two models.",
        "One frozen operating point (K3 / V2 / G32 / R4 / META8g64); no retuning, no ablation under canonical semantics.",
        "The mechanism of the Qwen degradation is undiagnosed; no causal decomposition of the legacy-vs-canonical "
        "difference has been run.",
        "The bootstrap CI reflects sampling over windows conditional on this run; run-to-run GPU variability is not "
        "modelled.",
        "CPU and CUDA canonical states are not bit-identical (device floating-point semantics); validity rests on "
        "same-device conformance with the independent oracle, with no cross-device tolerance.",
        "NIAH, Passage Retrieval, HotpotQA, Qasper and multilingual PPL have NOT been run under canonical semantics.",
        "The Llama revision was recovered by content matching of the preserved historical cache; the historical master "
        "pointer itself was not recovered."],
    "evidence": EVIDENCE,
    "full_detail": "continuation_ppl/two_model_attempt_2/summary.json (all 32 paired per-window NLL values per model)",
}
(HERE / "canonical_ppl_paper_ready_record.json").write_text(json.dumps(paper, indent=2) + "\n", encoding="utf-8", newline="\n")

acceptance = {
    "kind": "canonical-quality-v2 continuation PPL: ACCEPTANCE RECORD of registered two-model Attempt 2",
    "status": "accepted", "accepted_utc_date": "2026-10-03", "attempt": 2, "evidence_commit": "10b0973",
    "llama_uninspected_run_commit": "01cced9", "source_commit": s["models"]["llama3_1_8b"]["source_commit"],
    "authoritative": "this is the authoritative canonical continuation-PPL evidence",
    "results": {k: {"bf16_ppl": v["bf16_ppl"], "rabit_ppl": v["rabit_ppl"], "relative_delta_pct": v["relative_delta_pct"],
                    "ci95_pct": v["relative_delta_pct_ci95_paired_bootstrap"]} for k, v in models.items()},
    "interpretation": CASE_1, "qualification": QUALIFICATION, "generalization_claims": CLAIMS,
    "attempt_1": "permanently INVALID, NO QUALITY RESULT; never pooled or reused",
    "follow_up_experiments_launched": "none"}
(A2 / "acceptance_record.json").write_text(json.dumps(acceptance, indent=2) + "\n", encoding="utf-8", newline="\n")

index = {
    "kind": "CANONICAL QUALITY EVIDENCE INDEX (canonical-quality-v2); status as of 2026-10-03",
    "authoritative_canonical_quality_evidence": [
        {"item": "validated canonical evaluator / independent parity", "commits": ["c360697", "8fa9a9c", "257f0fb"],
         "paths": ["benchmarks/mlsys2027/canonical_rabit_quality.py", "results/mlsys2027/quality_semantic_audit/"]},
        {"item": "Llama and Qwen CPU / CUDA semantic conformance", "commits": ["8fa9a9c", "ec80638", "fd8d275", "d7ba819"],
         "paths": ["results/mlsys2027/canonical_quality_v2/cuda_conformance_diagnostic/"]},
        {"item": "Registered Attempt 2 continuation PPL: Llama +1.56 %, Qwen +1555.9 % (Case 1)",
         "commits": ["01cced9", "10b0973"],
         "paths": ["results/mlsys2027/canonical_quality_v2/continuation_ppl/two_model_attempt_2/",
                   "results/mlsys2027/canonical_quality_v2/canonical_ppl_paper_ready_record.json"]}],
    "two_model_n32_continuation_ppl_validation": "COMPLETE",
    "invalid_attempts": [{"attempt": 1, "status": "INVALID HARNESS / VALIDITY-GATE FAILURE; NO QUALITY RESULT PRODUCED",
                          "path": "results/mlsys2027/canonical_quality_v2/continuation_ppl/two_model_attempt_1/"}],
    "supporting_records": {"llama_identity_audit": "results/mlsys2027/canonical_quality_v2/llama_identity_audit/ (3bde966)",
                           "offline_proof_diagnostic": "results/mlsys2027/canonical_quality_v2/ppl_offline_proof_diagnostic/ "
                                                       "(bfe43db; the eb3f4b1 proof record is SUPERSEDED)",
                           "offline_proof": "continuation_ppl/offline_proof_record_v2.json (1e5a7ff)"},
    "legacy_logical_evaluator_results": {
        "classification": "LEGACY LOGICAL-EVALUATOR RESULTS",
        "rule": "kept unchanged; usable for historical / provenance discussion; NOT final canonical-RABIT quality evidence",
        "experiments": {"Exp1": "results/mlsys2027/quality_frontier/", "Exp2": "results/mlsys2027/multilingual_frontier/",
                        "Exp7": "results/mlsys2027/ablations/k_bit/", "Exp8": "results/mlsys2027/ablations/v_bit/",
                        "Exp9": "results/mlsys2027/ablations/group_size/", "Exp10": "results/mlsys2027/ablations/residual_window/",
                        "Exp11": "results/mlsys2027/ablations/metadata/", "Exp12 quality": "results/mlsys2027/variance/",
                        "Exp14 legacy quality": "results/mlsys2027/second_model/quality/"}},
    "not_run_under_canonical_semantics": ["NIAH", "Passage Retrieval", "HotpotQA", "Qasper", "multilingual PPL",
                                          "K / V / group / residual / metadata ablations"],
    "generalization_claims": CLAIMS,
}
(HERE / "EVIDENCE_INDEX.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8", newline="\n")
for k, v in models.items():
    print(k, v["bf16_ppl"], v["rabit_ppl"], v["relative_delta_pct"], v["relative_delta_pct_ci95_paired_bootstrap"], v["rounded"])
