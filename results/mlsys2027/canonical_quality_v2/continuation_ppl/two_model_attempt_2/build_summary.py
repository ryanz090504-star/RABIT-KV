"""Builds summary.json for the registered two-model canonical PPL Attempt 2 from the two result.json / record.json files
and the accepted LEGACY logs. Stdlib only; descriptive; no threshold. Usage (repository root):
    python results/mlsys2027/canonical_quality_v2/continuation_ppl/two_model_attempt_2/build_summary.py
"""
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "benchmarks/mlsys2027"))
import run_canonical_ppl as runner  # noqa: E402  (legacy_reference: parses the accepted legacy logs; read-only)

BASE = HERE.parent
ATTEMPT = 2
out = {"kind": "canonical-quality-v2 continuation PPL: registered two-model Attempt 2 -- summary (logical quality; NOT "
               "physical serving evidence)",
       "attempt": ATTEMPT, "two_model_record": json.loads((HERE / "record.json").read_text(encoding="utf-8")), "models": {}}

for model in ("llama3_1_8b", "qwen2_5_7b"):
    d = BASE / model / f"attempt_{ATTEMPT}"
    rec = json.loads((d / "record.json").read_text(encoding="utf-8"))
    raw = (d / "result.json").read_bytes()
    res = json.loads(raw.decode("utf-8"))
    ev, arms = rec["evaluation"], res["arms"]
    n = {a: [r["loss_sum"] / r["tokens"] for r in arms[a]] for a in arms}  # per-window mean token NLL
    tot = {a: (sum(r["loss_sum"] for r in arms[a]), sum(r["tokens"] for r in arms[a])) for a in arms}
    delta = [b - a for a, b in zip(n["bf16"], n["rabit"])]  # paired per-window NLL delta (rabit - bf16)
    order = sorted(range(len(delta)), key=lambda i: delta[i])
    total_delta = sum(delta)
    by_size = sorted(delta, reverse=True)
    pos = [x for x in delta if x > 0]
    # token-level view (4096 paired tokens)
    tb = [x for r in arms["bf16"] for x in r["token_nll"]]
    tr = [x for r in arms["rabit"] for x in r["token_nll"]]
    td = sorted((b - a for a, b in zip(tb, tr)), reverse=True)
    first_tok = [arms["rabit"][i]["token_nll"][0] - arms["bf16"][i]["token_nll"][0] for i in range(len(delta))]
    ref = runner.legacy_reference(model)
    log = (ROOT / runner.LEGACY_LOG[model]).read_text(encoding="utf-8", errors="replace")
    legacy_lines = [f"{runner.LEGACY_LOG[model]}:{k + 1}: {ln.rstrip()}" for k, ln in enumerate(log.splitlines())
                    if re.match(r"(bf16|rabit2)\s+\d", ln)]
    st = ev["statistics"]
    out["models"][model] = {
        "valid": rec["valid"], "gates": ev["gates"], "source_commit": rec["source_commit"],
        "started_utc": rec["started_utc"], "completed_utc": rec["completed_utc"],
        "app_ids": rec["app_ids_new"], "app_final_states": rec["cleanup"]["final_states"],
        "cleanup_verified": rec["cleanup"]["verified"], "modal_returncode": rec["modal_returncode"],
        "result_sha256": rec["result_sha256"], "result_sha256_matches_file": hashlib.sha256(raw).hexdigest() == rec["result_sha256"],
        "hardware": res["hardware"]["gpus"], "environment": res["environment"], "runtime_environment": res["runtime_environment"],
        "model": {k: res["model"][k] for k in ("model_id", "model_revision", "manifest_sha256", "files_checked", "passed", "dir")},
        "dataset": res["dataset"], "geometry": res["geometry"], "policy": res["policy"],
        "canonical_rabit_quality_sha256_lf": res["files"]["sha256_lf"]["canonical_rabit_quality.py"],
        "structure": {"windows": {a: len(arms[a]) for a in arms}, "tokens_per_window": sorted({r["tokens"] for a in arms for r in arms[a]}),
                      "token_nll_values_per_arm": {a: sum(len(r["token_nll"]) for r in arms[a]) for a in arms},
                      "all_finite": all(math.isfinite(x) for a in arms for r in arms[a] for x in r["token_nll"]),
                      "paired_inputs": "both stepwise arms score the same 32 (context, continuation) windows of the pinned "
                                       "token pool (token_pool_sha256 above) with the same loop; first token from the "
                                       "prefill logit",
                      "first_token_nll_identical_in_bf16_and_rabit": all(x == 0.0 for x in first_tok),
                      "rabit_cache_class": sorted({r["cache_class"] for r in arms["rabit"]}),
                      "decode_forwards": sorted({r["decode_forwards"] for a in ("bf16", "rabit") for r in arms[a]})},
        "scored_tokens_per_arm": tot["bf16"][1],
        "aggregate": {a: {"loss_sum": tot[a][0], "tokens": tot[a][1], "mean_nll": tot[a][0] / tot[a][1],
                          "ppl": math.exp(tot[a][0] / tot[a][1])} for a in arms},
        "bf16_ppl": st["bf16_ppl"], "rabit_ppl": st["rabit_ppl"], "delta_pct": st["delta_pct"],
        "delta_pct_ci95_paired_bootstrap": [st["ci_low"], st["ci_high"]],
        "aggregate_nll_delta": tot["rabit"][0] / tot["rabit"][1] - tot["bf16"][0] / tot["bf16"][1],
        "per_window": [{"window": i + 1, "bf16_nll": n["bf16"][i], "rabit_nll": n["rabit"][i], "delta_nll": delta[i],
                        "bf16_ppl": math.exp(n["bf16"][i]), "rabit_ppl": math.exp(n["rabit"][i])} for i in range(len(delta))],
        "paired_window_delta_nll": {
            "mean": statistics.fmean(delta), "median": statistics.median(delta), "stdev": statistics.stdev(delta),
            "min": min(delta), "max": max(delta), "windows_positive": len(pos), "windows_negative": sum(x < 0 for x in delta),
            "quartiles": statistics.quantiles(delta, n=4),
            "largest_positive": [{"window": i + 1, "delta_nll": delta[i], "bf16_ppl": math.exp(n["bf16"][i]),
                                  "rabit_ppl": math.exp(n["rabit"][i])} for i in order[::-1][:5]],
            "largest_negative_or_smallest": [{"window": i + 1, "delta_nll": delta[i], "bf16_ppl": math.exp(n["bf16"][i]),
                                              "rabit_ppl": math.exp(n["rabit"][i])} for i in order[:5]]},
        "concentration_descriptive": {
            "share_of_summed_window_delta_from_top_1": by_size[0] / total_delta,
            "share_from_top_3": sum(by_size[:3]) / total_delta, "share_from_top_8_of_32": sum(by_size[:8]) / total_delta,
            "aggregate_delta_pct_without_top_3_windows": 100 * (math.exp(statistics.fmean(sorted(delta)[:-3])) - 1),
            "token_level": {"tokens": len(td), "tokens_with_positive_delta": sum(x > 0 for x in td),
                            "share_of_total_delta_from_top_1pct_tokens": sum(td[:len(td) // 100]) / sum(td),
                            "share_from_top_10pct_tokens": sum(td[:len(td) // 10]) / sum(td),
                            "max_token_delta": td[0], "min_token_delta": td[-1]}},
        "robustness_from_runner": st.get("robustness"),
        "legacy_accepted_evidence": {
            "log": runner.LEGACY_LOG[model], "bf16_aggregate_ppl": ref["bf16_aggregate_ppl"],
            "rabit2_legacy_aggregate_ppl": ref["rabit2_legacy_aggregate_ppl"],
            "rabit2_legacy_delta_pct": 100 * (ref["rabit2_legacy_aggregate_ppl"] / ref["bf16_aggregate_ppl"] - 1),
            "source": "aggregate of the 32 per-window values printed (4 decimals) in the accepted legacy log",
            "summary_lines_in_log": legacy_lines,
            "note": "legacy logical evaluator (different V metadata order; no decode aging); context only, never pooled"},
        "bf16_reproducibility": ev["control"],
    }

(HERE / "summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
for model, m in out["models"].items():
    print("=====", model, m["valid"], all(m["gates"].values()), len(m["gates"]))
    print("HW", m["hardware"], m["runtime_environment"]["passed"], m["environment"]["torch"], m["environment"]["torch_cuda"], m["environment"]["python"])
    print("MODEL", m["model"]); print("STRUCT", m["structure"])
    print("AGG", {a: (round(v["mean_nll"], 6), round(v["ppl"], 4), v["tokens"]) for a, v in m["aggregate"].items()})
    print("PPL", m["bf16_ppl"], m["rabit_ppl"], m["delta_pct"], m["delta_pct_ci95_paired_bootstrap"], "nll_delta", m["aggregate_nll_delta"])
    p = m["paired_window_delta_nll"]; print("WIN", {k: p[k] for k in ("mean", "median", "stdev", "min", "max", "windows_positive", "windows_negative", "quartiles")})
    print("TOP+", [(x["window"], round(x["delta_nll"], 4), round(x["bf16_ppl"], 3), round(x["rabit_ppl"], 3)) for x in p["largest_positive"]])
    print("LOW", [(x["window"], round(x["delta_nll"], 4), round(x["bf16_ppl"], 3), round(x["rabit_ppl"], 3)) for x in p["largest_negative_or_smallest"]])
    print("CONC", json.dumps(m["concentration_descriptive"]))
    print("LEGACY", {k: v for k, v in m["legacy_accepted_evidence"].items() if k not in ("note", "source")})
    print("CONTROL", m["bf16_reproducibility"]); print("APPS", m["app_ids"], [(v["State"], v["Tasks"]) for v in m["app_final_states"].values()], m["cleanup_verified"])
