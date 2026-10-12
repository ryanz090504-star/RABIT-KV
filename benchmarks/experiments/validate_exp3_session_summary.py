"""
Generic validator for Experiment 3 session summaries produced by
build_exp3_session_summary.py. No run-specific constants: every expected value
is recomputed here, independently, from the raw files in --results-dir.

Verifies:
  * LF-normalized SHA-256 of every recorded result file (raw-byte match reported);
  * ABBA legs from the logs match the manifest plan and have the ABBA dtype
    pattern (x, y, y, x); reps per leg / per dtype and prompt/output token
    counts match the manifest protocol;
  * capacity = num_gpu_blocks x block_size = engine-log value; duplicate
    capacities per dtype; the summary's capacity block and ratio;
  * every per-leg and pooled statistic, signed delta, drift value and
    range-overlap flag, recomputed from the raw samples;
  * every note claim: the claimed quantity recomputes to the claimed text,
    the text appears in the notes, every number in the notes is covered by a
    claim, and direction / overlap / duplicate-capacity wording is correct;
  * integrity counts equal a recount of integrity_check.json states and the
    run is fully passed; matched-config status; correctness gate (also against
    correctness_gate.log); protected-path status; archive status (and, if a
    failed_attempt_* folder is inside --results-dir, its files vs its
    failure_summary.json); provenance and GPU UUID vs the raw files;
  * agreement with the runner's own matched_capacity_latency_summary.json.

Read-only. Exits non-zero on any failure.

Usage:
    python benchmarks/mlsys2027/validate_exp3_session_summary.py \
        --results-dir results/mlsys2027/deployment/replication_1 \
        --summary results/mlsys2027/deployment/replication_1/summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

EPS = 1e-9
LOG_OF = {"bfloat16": "bf16_deployment.log", "rabit_kv2": "rabit_kv2_deployment.log"}
METRICS = ("tpot_ms", "ttft_ms", "wall_ms")
DELTA_KEYS = (("tpot_median", "tpot_ms", "median"), ("tpot_p90", "tpot_ms", "p90"),
              ("ttft_median", "ttft_ms", "median"), ("wall_median", "wall_ms", "median"))
NUMBER = re.compile(r"(?<![A-Za-z_\d.])[-+]?\d[\d,]*(?:\.\d+)?")


def lf_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def raw_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def p90(v: list[float]) -> float:
    return statistics.quantiles(v, n=10, method="inclusive")[8]


def parse(path: Path) -> dict:
    legs, cur = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^===== EXP3 LEG (\w+) \(index (\d+), (\w+)\) =====$", line)
        if m:
            cur = m.group(1)
            legs[cur] = {"index": int(m.group(2)), "dtype": m.group(3), "samples": [], "cap": None, "wl": None,
                         "log_tokens": None, "log_gib": None}
        elif cur is None:
            continue
        elif line.startswith("EXP3_SAMPLE "):
            legs[cur]["samples"].append(json.loads(line[len("EXP3_SAMPLE "):]))
        elif line.startswith("EXP3_CAPACITY="):
            legs[cur]["cap"] = json.loads(line.split("=", 1)[1])
        elif line.startswith("EXP3_WORKLOAD="):
            legs[cur]["wl"] = json.loads(line.split("=", 1)[1])
        else:
            m = re.search(r"GPU KV cache size: ([\d,]+) tokens", line)
            if m:
                legs[cur]["log_tokens"] = int(m.group(1).replace(",", ""))
            m = re.search(r"Available KV cache memory: ([\d.]+) GiB", line)
            if m:
                legs[cur]["log_gib"] = float(m.group(1))
    return legs


def fmt_claim(fmt: str, value) -> str:
    if fmt == "overlap":
        return "overlap" if value else "do not overlap"
    return fmt.format(value)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    args = ap.parse_args(argv)
    rd = args.results_dir
    s = json.loads(args.summary.read_text(encoding="utf-8"))
    fails: list[str] = []
    n: dict[str, int] = {}

    def need(cond, msg: str, cat: str) -> None:
        n[cat] = n.get(cat, 0) + 1
        if not cond:
            fails.append(msg)

    def close(a, b) -> bool:
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= EPS

    manifest = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
    integ = json.loads((rd / "integrity_check.json").read_text(encoding="utf-8"))
    cdiff = json.loads((rd / "matched_config_diff.json").read_text(encoding="utf-8"))
    runner = json.loads((rd / "matched_capacity_latency_summary.json").read_text(encoding="utf-8"))
    proto = manifest["protocol"]

    # -- hashes
    raw_ok = 0
    need(bool(s.get("result_files_sha256")), "no result-file hashes recorded", "hashes")
    for f, h in s.get("result_files_sha256", {}).items():
        need((rd / f).exists() and h["lf_normalized"] == lf_sha(rd / f), f"{f}: LF-normalized SHA-256 differs", "hashes")
        raw_ok += (rd / f).exists() and h["raw"] == raw_sha(rd / f)

    # -- legs and ABBA structure (plan from manifest, facts from logs)
    raw = {d: parse(rd / LOG_OF[d]) for d in LOG_OF}
    plan = [(l["index"], l["leg"], l["kv_cache_dtype"]) for l in manifest["legs"]]
    found = sorted((L["index"], label, L["dtype"]) for d in raw for label, L in raw[d].items())
    need(found == sorted(plan) and len(found) == 4, f"legs in logs {found} != manifest plan {plan}", "structure")
    dts = [d for _, _, d in sorted(plan)]
    need(len(dts) == 4 and dts[0] == dts[3] != dts[1] == dts[2] and dts[0] == "bfloat16",
         f"dtype order {dts} is not ABBA with A=bfloat16", "structure")
    need([(x["index"], x["leg"], x["kv_cache_dtype"]) for x in s["abba_legs"]] == sorted(plan),
         "summary abba_legs differ from plan", "structure")
    legs = {label: raw[d][label] for _, label, d in plan}
    reps_leg, reps_dtype = proto["measured_reps_per_leg"], proto["measured_reps_per_dtype"]
    R: dict = {}  # independently recomputed quantities, keyed like note claims
    for idx, label, d in sorted(plan):
        L = legs[label]
        smp = L["samples"]
        need(len(smp) == reps_leg and [r["rep"] for r in smp] == list(range(reps_leg)),
             f"{label}: {len(smp)} samples, expected reps 0..{reps_leg - 1}", "structure")
        need(all(r["prompt_tokens"] == proto["context_tokens"] and r["output_tokens"] == proto["output_tokens"]
                 for r in smp), f"{label}: prompt/output token counts differ from protocol", "structure")
        cap = L["cap"]
        need(cap["num_gpu_blocks"] * cap["block_size"] == cap["capacity_tokens"] == L["log_tokens"],
             f"{label}: capacity {cap} != blocks x block_size / engine log {L['log_tokens']}", "capacity")
        sl = s["per_leg"][label]
        need(sl["capacity"]["capacity_tokens"] == cap["capacity_tokens"]
             and sl["capacity"]["num_gpu_blocks"] == cap["num_gpu_blocks"], f"{label}: summary capacity differs", "capacity")
        for m in METRICS:
            v = [r[m] for r in smp]
            need(sl[f"{m}_samples"] == v, f"{label}: summary {m} samples differ from raw log", "samples")
            for stat, val in (("median", statistics.median(v)), ("p90", p90(v)), ("mean", statistics.mean(v)),
                              ("stdev", statistics.stdev(v)), ("min", min(v)), ("max", max(v)), ("n", len(v))):
                need(close(sl[m][stat], val), f"{label}.{m}.{stat} {sl[m][stat]} != {val}", "stats")
    wl0 = legs[sorted(plan)[0][1]]["wl"]
    R["workload.context_tokens"], R["workload.output_tokens"] = wl0["context_tokens"], wl0["output_tokens"]
    need(len({json.dumps(legs[l]["wl"], sort_keys=True) for l in legs}) == 1, "workload differs between legs", "structure")

    pooled = {}
    for d in LOG_OF:
        labels = [label for _, label, dd in sorted(plan) if dd == d]
        caps = {json.dumps(legs[l]["cap"], sort_keys=True) for l in labels}
        dup = len(caps) == 1
        need(dup == s["pooled_per_dtype"][d]["duplicate_capacity_identical"] == s["capacity"]["duplicate_capacity_identical"][d],
             f"{d}: duplicate-capacity flag wrong", "capacity")
        need(dup, f"{d}: duplicate capacities differ", "capacity")
        cap = legs[labels[0]]["cap"]
        R[f"capacity.{d if d == 'rabit_kv2' else 'bf16'}.tokens"] = cap["capacity_tokens"]
        R[f"capacity.{d if d == 'rabit_kv2' else 'bf16'}.blocks"] = cap["num_gpu_blocks"]
        R[f"capacity.{d if d == 'rabit_kv2' else 'bf16'}.block_size"] = cap["block_size"]
        rows = {m: [r[m] for l in labels for r in legs[l]["samples"]] for m in METRICS}
        need(all(len(v) == reps_dtype for v in rows.values()), f"{d}: pooled samples != {reps_dtype}", "structure")
        pooled[d] = rows
        sp = s["pooled_per_dtype"][d]
        for m, v in rows.items():
            for stat, val in (("median", statistics.median(v)), ("p90", p90(v)), ("mean", statistics.mean(v)),
                              ("n", len(v))):
                need(close(sp[m][stat], val), f"pooled {d}.{m}.{stat} mismatch", "stats")
                if (m, stat) in {(mm, st) for _, mm, st in DELTA_KEYS}:  # headline stats quoted in notes
                    R[f"pooled.{d}.{m}.{stat}"] = val
    bf, rk = R["capacity.bf16.tokens"], R["capacity.rabit_kv2.tokens"]
    R["capacity.ratio"] = rk / bf
    c = s["capacity"]
    need(c["bf16_capacity_tokens"] == bf and c["rabit_kv2_capacity_tokens"] == rk and close(c["ratio_rabit_over_bf16"], rk / bf)
         and c["signed_delta_tokens"] == rk - bf, "summary capacity block/ratio wrong", "capacity")

    # -- deltas
    for key, m, stat in DELTA_KEYS:
        fn = statistics.median if stat == "median" else p90
        x, y = fn(pooled["bfloat16"][m]), fn(pooled["rabit_kv2"][m])
        dl = s["signed_deltas_rabit_minus_bf16_pooled"][key]
        need(close(dl["bf16"], x) and close(dl["rabit_kv2"], y) and close(dl["signed_delta_ms"], y - x)
             and close(dl["signed_delta_pct"], (y / x - 1) * 100), f"delta {key} does not recompute", "deltas")
        R[f"delta.{key}.ms"], R[f"delta.{key}.pct"] = y - x, (y / x - 1) * 100

    # -- drift and overlap
    for d in LOG_OF:
        first, second = [label for _, label, dd in sorted(plan) if dd == d]
        dkey = f"{'bf16' if d == 'bfloat16' else d}_{first}_vs_{second}"
        need(dkey in s["drift"], f"drift entry {dkey} missing", "drift")
        for m in METRICS:
            va, vb = [r[m] for r in legs[first]["samples"]], [r[m] for r in legs[second]["samples"]]
            ma, mb = statistics.median(va), statistics.median(vb)
            ov = not (max(va) < min(vb) or max(vb) < min(va))
            dr = s["drift"][dkey][m]
            need(close(dr["median_diff_ms"], mb - ma) and close(dr["median_diff_pct"], (mb / ma - 1) * 100),
                 f"drift {dkey}.{m} does not recompute", "drift")
            need(dr["ranges_overlap"] is ov, f"drift {dkey}.{m} overlap claim {dr['ranges_overlap']} != {ov}", "drift")
            need(dr[f"{first}_range"] == [min(va), max(va)] and dr[f"{second}_range"] == [min(vb), max(vb)],
                 f"drift {dkey}.{m} ranges wrong", "drift")
            R[f"drift.{dkey}.{m}.median_diff_pct"] = (mb / ma - 1) * 100
            R[f"drift.{dkey}.{m}.first_median"], R[f"drift.{dkey}.{m}.second_median"] = ma, mb
            R[f"drift.{dkey}.{m}.ranges_overlap"] = ov

    # -- notes and claims
    notes = " ".join(s["notes"])
    claims = s.get("note_claims", [])
    need({cl["quantity"] for cl in claims} == set(R), "note claims do not cover exactly the recomputed quantities", "notes")
    claim_numbers = set()
    for cl in claims:
        q = cl["quantity"]
        need(q in R and cl["text"] == fmt_claim(cl["format"], R[q]),
             f"note claim {q}: text {cl['text']!r} != recomputed {fmt_claim(cl['format'], R.get(q)) if q in R else '?'}", "notes")
        need(cl["text"] in notes, f"note claim text {cl['text']!r} not present in notes", "notes")
        claim_numbers.update(NUMBER.findall(cl["text"]))
    for num in NUMBER.findall(notes):
        need(num in claim_numbers, f"number {num!r} in notes is not backed by a claim", "notes")
    words = []
    for key, m, stat in DELTA_KEYS:
        sign = R[f"delta.{key}.ms"]
        word = "slower" if sign > 0 else ("faster" if sign < 0 else "equal")
        words.append(word)
        need(f"RABIT-KV is {word} ({fmt_claim('{:+.3f}', sign)} ms" in notes, f"direction wording for {key} wrong", "notes")
    if len(set(words)) == 1:
        need(f"RABIT-KV is {words[0]} than BF16 in all reported pooled latency metrics" in notes,
             "all-metrics direction sentence missing/wrong", "notes")
    else:
        need("NOT consistent" in notes, "inconsistent-direction sentence missing", "notes")
    need(("duplicate capacities are identical" in notes) == all(s["capacity"]["duplicate_capacity_identical"].values()),
         "duplicate-capacity wording wrong", "notes")
    for phrase in ("PHYSICAL real-engine vLLM allocator capacity", "PHYSICAL real-engine single-request latency",
                   "NOT logical fake-quant", "historical reference only"):
        need(phrase in notes, f"required phrase missing: {phrase!r}", "notes")

    # -- status: integrity, config, gate, protected paths, archive
    recount = {st: sum(1 for ch in integ["checks"] if ch["state"] == st)
               for st in ("passed", "failed", "not_run", "not_evaluated")}
    need(recount == integ["counts"] == s["integrity"]["counts"], f"integrity counts {s['integrity']['counts']} != recount {recount}", "status")
    need(recount["passed"] == len(integ["checks"]) and integ["all_ok"] is True and s["integrity"]["all_ok"] is True,
         f"session is not fully passed: {recount}", "status")
    need(cdiff["status"] == "passed" and cdiff["matched"] is True and not cdiff["violations"]
         and s["matched_config"]["status"] == cdiff["status"] and s["matched_config"]["violations"] == cdiff["violations"],
         "matched config status not passed / not matching raw", "status")
    need(sorted(cdiff["fields_differing_between_dtypes"]) == sorted(cdiff["dtype_induced_allowlist"])
         and s["matched_config"]["fields_compared"] == cdiff["fields_compared"], "config diff fields mismatch", "status")
    g = manifest["correctness_gate"]
    glog = (rd / "correctness_gate.log").read_text(encoding="utf-8")
    m = re.search(r"^(\d+) passed, (\d+) warnings", glog, re.M)
    need(g["pytest_exit"] == 0 and g["result"] == {"passed": True} and m and int(m.group(1)) == g["pytest_passed"]
         and "RABIT-2 FINAL TARGETED REGRESSION PASSED" in glog
         and s["correctness_gate"]["pytest_passed"] == g["pytest_passed"]
         and s["correctness_gate"]["pytest_exit"] == 0, "correctness gate not passed / not matching log", "status")
    need(manifest["status"] == "passed" and s["run_status"]["manifest_status"] == "passed", "manifest status not passed", "status")
    need(manifest.get("protected_paths_post_run_status") == "clean"
         and s["run_status"]["protected_paths_post_run_status"] == "clean", "protected paths not clean", "status")
    need(manifest.get("archived_attempts_unchanged") is True and s["run_status"]["archived_attempts_unchanged"] is True,
         "archive-integrity flag not true", "status")
    for arch in sorted(rd.glob("failed_attempt_*")):
        fs = json.loads((arch / "failure_summary.json").read_text(encoding="utf-8"))
        need(all(raw_sha(arch / f) == h for f, h in fs["files_sha256"].items()),
             f"{arch.name}: files differ from failure_summary.json", "status")
    need(all(p["clean"] and not p["compute_apps"] for p in s["gpu_clean_state"]["pre_leg"].values())
         and sorted(s["gpu_clean_state"]["pre_leg"]) == sorted(l for _, l, _ in plan), "GPU clean state wrong", "status")
    need(not manifest["processes"]["watchdog_timeouts"] and s["processes"]["watchdog_timeouts"] == [],
         "watchdog timeout recorded", "status")

    # -- provenance and cross-check with the runner summary
    prov, mp = s["provenance"], manifest["provenance"]
    need(all(prov[k] == mp[k] for k in ("git_head", "vllm_kvquant_tree", "rabit_kv2_sha256", "runner_script_sha256",
                                        "modal_app_sha256", "worker_sha256", "correctness_gate_sha256",
                                        "watchdog_sha256")), "provenance differs from manifest", "provenance")
    need(prov["gpu_uuid"] == [gg["uuid"] for gg in runner["environment"]["gpus"]], "GPU UUID differs from raw environment", "provenance")
    for d in LOG_OF:
        h = runner["pooled_per_dtype"][d]["headline"]
        sp = s["pooled_per_dtype"][d]
        need(close(h["tpot_ms_median"], sp["tpot_ms"]["median"]) and close(h["tpot_ms_p90"], sp["tpot_ms"]["p90"])
             and close(h["ttft_ms_median"], sp["ttft_ms"]["median"]) and close(h["wall_ms_median"], sp["wall_ms"]["median"])
             and runner["pooled_per_dtype"][d]["capacity_tokens"] == sp["capacity_tokens"],
             f"{d}: disagrees with runner summary", "cross")

    print(f"run_label: {s.get('run_label')} | results_dir: {rd.as_posix()}")
    print(f"hashes (LF) verified: {n.get('hashes', 0)} ({raw_ok}/{len(s.get('result_files_sha256', {}))} raw-byte identical)")
    for cat in ("structure", "capacity", "samples", "stats", "deltas", "drift", "notes", "status", "provenance", "cross"):
        print(f"  {cat:<10} checks: {n.get(cat, 0)}")
    print(f"recomputed: capacity {bf:,} / {rk:,} ratio {rk / bf:.4f}x; TPOT delta {R['delta.tpot_median.pct']:+.2f}%, "
          f"wall delta {R['delta.wall_median.pct']:+.2f}%; integrity {recount}")
    print(f"FAILURES: {len(fails)}")
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
