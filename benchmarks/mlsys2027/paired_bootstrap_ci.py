"""
RABIT-KV MLSys 2027 -- Experiment 12 offline analysis: per-example extraction and PAIRED bootstrap confidence intervals.

CPU only; reads the raw logs written by the unchanged canonical quality scripts (no GPU, no model, no re-generation).
For each benchmark it extracts the per-example bf16 and rabit2 values of ONE run (the same examples, in the same
order), and computes:

  * the aggregate metric of each method and the rabit2 - bf16 delta, exactly as the official metric defines them;
  * a PAIRED percentile bootstrap CI on the delta: the resampling unit is the example (QA / retrieval / NIAH case) or
    the independent text window (continuation PPL); each bootstrap replicate draws n unit indices with replacement and
    uses the SAME indices for bf16 and rabit2, so the pairing is preserved;
  * descriptive robustness: per-example delta distribution, counts better / worse / equal, and how much of the
    aggregate delta the top-1 / top-3 examples (by |contribution|) account for. Nothing is removed.

Per-unit values (official metrics unchanged):
    hotpotqa / qasper      LongBench qa_f1_score per example (0-1), aggregate = mean x 100 (F1 points)
    passage_retrieval      LongBench retrieval score per example (0-1), aggregate = mean x 100
    niah                   1 if PASS else 0 per case, aggregate = mean x 100 (accuracy %)
    continuation_ppl       per-window PPL; every window scores the same number of tokens, so the aggregate PPL =
                           exp(mean_w ln PPL_w) and delta % = 100 * (exp(mean ln PPL_rabit2 - mean ln PPL_bf16) - 1)

The bootstrap CI quantifies SAMPLING uncertainty over examples / windows, conditional on this run's generations. It
does not capture run-to-run generation variability of identical configs (see the control reproducibility audit).

Usage:
    python benchmarks/mlsys2027/paired_bootstrap_ci.py --results-dir results/mlsys2027/variance
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
import statistics
from pathlib import Path

METHODS = ("bf16", "rabit2")
BENCHMARKS = ("continuation_ppl", "niah", "passage_retrieval", "hotpotqa", "qasper")
BOOTSTRAP = {"resamples": 10000, "base_seed": 20270929, "confidence": 0.95, "method": "paired percentile bootstrap",
             "seed_offset_by_benchmark": {b: i for i, b in enumerate(BENCHMARKS)},
             "index_draw": "int(rng.random() * n) with rng = random.Random(base_seed + offset) (Mersenne Twister)",
             "percentiles": "sorted replicate deltas; lower = s[floor(0.025*B)], upper = s[ceil(0.975*B) - 1]"}
METRIC = {"continuation_ppl": ("ppl", "PPL delta % (rabit2 vs bf16)"),
          "niah": ("accuracy_pct", "accuracy delta (points)"),
          "passage_retrieval": ("accuracy_pct", "retrieval score delta (points)"),
          "hotpotqa": ("f1_pct", "F1 delta (points)"), "qasper": ("f1_pct", "F1 delta (points)")}

PPL_RUNNING_RE = re.compile(r"^Running (\S+)\.\.\.$")
PPL_SAMPLE_RE = re.compile(r"^  sample (\d+)/(\d+): PPL=([0-9.]+), KV=([0-9.]+) MB")
NIAH_ROW_RE = re.compile(r"^(\d+)\s+([0-9.]+)\s+(\S+)\s+(PASS|FAIL)\s+([0-9.]+)\s")
SAMPLE_RE = re.compile(r"^Sample (\d+)/(\d+) \(dataset index (\d+)\): (\d+) original tokens, (\d+) used tokens")
ROW_RE = re.compile(r"^\s+(\S+)\s+score=([0-9.]+) KV=([0-9.]+) MB answer=(.*)$")


# ------------------------------------------------------------------------------------------------ extraction
def extract(benchmark: str, text: str) -> dict:
    """Per-unit rows {method: [ {key, value, kv_mb, ...}, ... ]} for bf16 and rabit2 only (other rows skipped)."""
    lines = text.splitlines()
    out = {m: [] for m in METHODS}
    if benchmark == "continuation_ppl":
        current = None
        for ln in lines:
            r = PPL_RUNNING_RE.match(ln)
            if r:
                current = r.group(1)
                continue
            s = PPL_SAMPLE_RE.match(ln)
            if s and current in out:
                out[current].append({"key": int(s.group(1)), "value": float(s.group(3)), "kv_mb": float(s.group(4)),
                                     "prefix_tokens": None})
    elif benchmark == "niah":
        for ln in lines:
            r = NIAH_ROW_RE.match(ln)
            if r and r.group(3) in out:
                ctx, depth = int(r.group(1)), round(float(r.group(2)), 2)
                out[r.group(3)].append({"key": [ctx, depth], "value": 1.0 if r.group(4) == "PASS" else 0.0,
                                        "kv_mb": float(r.group(5)), "prefix_tokens": ctx - 1})
    else:
        current = None
        for ln in lines:
            s = SAMPLE_RE.match(ln)
            if s:
                current = {"key": int(s.group(3)), "prefix_tokens": int(s.group(5)) - 1}
                continue
            r = ROW_RE.match(ln)
            if r and current is not None and r.group(1) in out:
                out[r.group(1)].append({**current, "value": float(r.group(2)), "kv_mb": float(r.group(3)),
                                        "answer": ast.literal_eval(r.group(4))})
    return out


def paired_values(benchmark: str, rows: dict) -> tuple[list, list, list]:
    """(keys, bf16 unit values, rabit2 unit values) aligned by unit key; raises if the pairing is not exact."""
    b, r = rows["bf16"], rows["rabit2"]
    keys_b, keys_r = [x["key"] for x in b], [x["key"] for x in r]
    if keys_b != keys_r or len(set(map(json.dumps, keys_b))) != len(keys_b) or not keys_b:
        raise ValueError(f"{benchmark}: bf16 / rabit2 units are not an identical, duplicate-free, non-empty set")
    tf = (lambda v: math.log(v)) if benchmark == "continuation_ppl" else (lambda v: v)
    return keys_b, [tf(x["value"]) for x in b], [tf(x["value"]) for x in r]


# ------------------------------------------------------------------------------------------------ statistics
def aggregate(benchmark: str, unit_values: list[float]) -> float:
    m = sum(unit_values) / len(unit_values)
    return math.exp(m) if benchmark == "continuation_ppl" else 100.0 * m


def delta(benchmark: str, bf16: list[float], rabit2: list[float]) -> float:
    if benchmark == "continuation_ppl":  # unit values are ln PPL_w
        return 100.0 * (math.exp(sum(rabit2) / len(rabit2) - sum(bf16) / len(bf16)) - 1.0)
    return 100.0 * (sum(rabit2) - sum(bf16)) / len(bf16)


def bootstrap_indices(n: int, resamples: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[int(rng.random() * n) for _ in range(n)] for _ in range(resamples)]


def paired_bootstrap(benchmark: str, bf16: list[float], rabit2: list[float], resamples: int, seed: int,
                     confidence: float = 0.95) -> dict:
    n = len(bf16)
    reps = []
    for idx in bootstrap_indices(n, resamples, seed):  # the SAME indices for both methods: pairing preserved
        reps.append(delta(benchmark, [bf16[i] for i in idx], [rabit2[i] for i in idx]))
    reps.sort()
    alpha = (1.0 - confidence) / 2.0
    lo, hi = reps[math.floor(alpha * resamples)], reps[math.ceil((1.0 - alpha) * resamples) - 1]
    return {"ci_low": lo, "ci_high": hi, "confidence": confidence, "resamples": resamples, "seed": seed,
            "replicate_mean": sum(reps) / resamples}


def robustness(benchmark: str, keys: list, bf16: list[float], rabit2: list[float]) -> dict:
    """Descriptive per-unit analysis of the delta; nothing is removed."""
    n = len(bf16)
    if benchmark == "continuation_ppl":
        d = [r - b for b, r in zip(bf16, rabit2)]  # per-window delta in ln PPL (mean token NLL)
        contrib = [x / n for x in d]  # contribution to the mean ln-PPL delta
        unit = "ln PPL (mean token NLL) per window; contribution = d/n to the mean ln-PPL delta"
        total = sum(contrib)
    else:
        d = [100.0 * (r - b) for b, r in zip(bf16, rabit2)]  # per-example delta in points
        contrib = [x / n for x in d]  # contribution to the aggregate delta (points)
        unit = "points (x100 of the per-example official score); contribution = d/n to the aggregate delta"
        total = sum(contrib)
    order = sorted(range(n), key=lambda i: (-abs(contrib[i]), i))
    q = statistics.quantiles(d, n=4, method="inclusive") if n >= 2 else [d[0]] * 3

    def top(k):
        sel = order[:k]
        s = sum(contrib[i] for i in sel)
        return {"units": [{"key": keys[i], "delta": d[i], "contribution": contrib[i]} for i in sel],
                "sum_contribution": s,
                "share_of_aggregate_delta": (s / total) if total != 0 else None,
                "share_of_total_abs_contribution": (sum(abs(contrib[i]) for i in sel) / sum(map(abs, contrib)))
                if any(contrib) else None}
    return {"unit": unit, "n": n, "n_rabit2_better": sum(x > 0 for x in d), "n_rabit2_worse": sum(x < 0 for x in d),
            "n_equal": sum(x == 0 for x in d),
            "per_unit_delta_summary": {"min": min(d), "q25": q[0], "median": q[1], "q75": q[2], "max": max(d),
                                       "mean": sum(d) / n},
            "largest_abs_contribution": top(1)["units"][0],
            "top1": top(1), "top3": top(min(3, n))}


def analyze(benchmark: str, text: str) -> dict:
    rows = extract(benchmark, text)
    keys, b, r = paired_values(benchmark, rows)
    boot = paired_bootstrap(benchmark, b, r, BOOTSTRAP["resamples"],
                            BOOTSTRAP["base_seed"] + BOOTSTRAP["seed_offset_by_benchmark"][benchmark],
                            BOOTSTRAP["confidence"])
    dlt = delta(benchmark, b, r)
    return {"benchmark": benchmark, "metric": METRIC[benchmark][0], "delta_definition": METRIC[benchmark][1],
            "n_units": len(keys), "bf16_aggregate": aggregate(benchmark, b), "rabit2_aggregate": aggregate(benchmark, r),
            "delta": dlt, **boot,
            "ci_contains_zero": boot["ci_low"] <= 0.0 <= boot["ci_high"],
            "interpretation_rule": "descriptive: a CI containing 0 is NOT evidence of no effect; it reports the "
                                   "sampling uncertainty of this run's delta",
            "robustness": robustness(benchmark, keys, b, r),
            "per_unit": {"keys": keys, "bf16": b, "rabit2": r,
                         "scale": "ln PPL per window" if benchmark == "continuation_ppl" else "official per-unit score (0-1)"}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    d = Path(a.results_dir)
    out = {"bootstrap": BOOTSTRAP, "benchmarks": {b: analyze(b, (d / f"{b}.log").read_text(encoding="utf-8",
                                                                                          errors="replace"))
                                                  for b in BENCHMARKS}}
    Path(a.out or d / "variance_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                                                          encoding="utf-8")
    for b, x in out["benchmarks"].items():
        print(f"{b:18s} n={x['n_units']:4d} delta={x['delta']:+.3f} 95% CI [{x['ci_low']:+.3f}, {x['ci_high']:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
