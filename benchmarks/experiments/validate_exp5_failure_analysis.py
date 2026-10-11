"""
Validator for results/mlsys2027/context_scaling/failed_attempt_1/failure_analysis.json
(MLSys 2027 Experiment 5, attempt 1 -- a FAILED run; not a scientific summary).

Independent of build_exp5_failure_analysis.py (no shared code): every value is
recomputed here by re-parsing the raw archived evidence (modal_session.log,
correctness_gate.log, manifest.json, integrity_check.json). The only constants
are the pre-registered plan (cell labels, conditioning labels, the 900 s cell
watchdog) and the required status/wording, never measured values.

Read-only. Exits non-zero on any failure.

Usage:
    python benchmarks/mlsys2027/validate_exp5_failure_analysis.py \
        --attempt-dir results/mlsys2027/context_scaling/failed_attempt_1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

EPS = 1e-9
PLAN = ["A512", "B512", "B2048", "A2048", "A4096", "B4096", "B8192", "A8192", "A16384", "B16384", "B32768", "A32768"]
CONDITIONING = {"conditioning_A512", "conditioning_B512"}
CELL_WATCHDOG_S = 900
REQUIRED_WORDING = ("The current RABIT-KV implementation exhibits a severe performance cliff when the workload "
                    "crosses max_num_batched_tokens and activates its non-initial chunked-prefill path.")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def q90(v):
    return statistics.quantiles(v, n=10, method="inclusive")[8]


def reparse(session: str) -> tuple[dict, dict]:
    """Group every line by cell label via the index announced in EXP5_LEG_START."""
    label_of: dict[str, str] = {}
    cells: dict[str, dict] = {}
    top = {"exit": {}, "proc": {}, "pre": {}, "verdict": {}, "timeouts": [], "start": {}}
    for raw in session.splitlines():
        if raw.startswith("[leg"):
            head, _, body = raw.partition("] ")
            idx = head[len("[leg"):].split(":")[0]
            lab = label_of.get(idx)
            if lab is None:
                continue
            c = cells[lab]
            b = body.strip()
            c["lines"].append(body)
            if b.startswith("EXP5_WARMUP ") and b[12:13] == "{":
                c["warm"].append(json.loads(b[12:]))
            elif b.startswith("EXP5_SAMPLE ") and b[12:13] == "{":
                c["samp"].append(json.loads(b[12:]))
            elif b.startswith("EXP5_WORKLOAD="):
                c["wl"] = json.loads(b.split("=", 1)[1])
            elif b.startswith("EXP5_CAPACITY="):
                c["cap"] = json.loads(b.split("=", 1)[1])
            elif b == "EXP5_MEASUREMENT_BEGIN":
                c["in_meas"] = True
            elif b == "EXP5_WORKER_COMPLETE":
                c["complete"] = True
            if "Triton kernel JIT compilation during inference" in b:
                c["jit_meas" if c["in_meas"] else "jit_other"] += 1
            if "CUDA out of memory" in b or "OutOfMemoryError" in b:
                c["oom"] += 1
            m = re.search(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[triton_attn\.py:\d+\] "
                          r"RABIT2_STAGE3C_CHUNKED_PREFILL_ACTIVE q_len=(\d+) context_len=(\d+)", b)
            if m:
                c["stage3c"].append((m.group(1), int(m.group(2)), int(m.group(3))))
            continue
        s = raw.strip()
        if not s.startswith("EXP5_") or "={" not in s:
            continue
        tag, payload = s.split("=", 1)
        try:
            p = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if tag == "EXP5_LEG_START":
            label_of[str(p["index"])] = p["leg"]
            top["start"][p["leg"]] = p
            cells[p["leg"]] = {"lines": [], "warm": [], "samp": [], "wl": None, "cap": None, "in_meas": False,
                               "complete": False, "jit_meas": 0, "jit_other": 0, "oom": 0, "stage3c": []}
        elif tag == "EXP5_LEG_EXIT":
            top["exit"][p["leg"]] = p["returncode"]
        elif tag == "EXP5_PROCESS_EXIT":
            top["proc"][p["label"]] = p
        elif tag == "EXP5_PRE_LEG_GPU_STATE":
            top["pre"][p["leg"]] = p
        elif tag == "EXP5_CELL_VERDICT":
            top["verdict"][p["leg"]] = p["verdict"]
        elif tag == "EXP5_WATCHDOG_TIMEOUT":
            top["timeouts"].append(p)
    return cells, top


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attempt-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    d = a.attempt_dir
    f = json.loads((d / "failure_analysis.json").read_text(encoding="utf-8"))
    fails: list[str] = []
    n = {"checks": 0}

    def need(ok, msg):
        n["checks"] += 1
        if not ok:
            fails.append(msg)

    def eq(x, y):
        return isinstance(x, (int, float)) and isinstance(y, (int, float)) and abs(x - y) <= EPS

    def guarded(fn, name):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 (malformed input is a failure, never a crash)
            fails.append(f"{name}: could not evaluate ({type(exc).__name__}: {exc})")

    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    integ = json.loads((d / "integrity_check.json").read_text(encoding="utf-8"))
    gate = (d / "correctness_gate.log").read_text(encoding="utf-8")
    cells, top = reparse((d / "modal_session.log").read_text(encoding="utf-8"))

    def status():
        need(f["accepted_scientific_summary"] is False, "accepted_scientific_summary must be false")
        need(f["run_status"] == "failed" == manifest["status"], "run_status must be failed (and match manifest)")
        need(f["failure_stage"] == "watchdog_timeout" == manifest["failure"]["stage"],
             "failure_stage must be watchdog_timeout (and match manifest)")
        need(f["failed_cell"] == "B32768", "failed_cell must be B32768")
        need(f["wording"] == REQUIRED_WORDING, "required cliff wording missing/altered")
        need(any("algorithmic limitation" in x for x in f["not_claimed"])
             and any("maximum feasible" in x for x in f["not_claimed"]), "not_claimed disclaimers missing")

    def hashes():
        files = sorted(p.name for p in d.iterdir() if p.is_file() and p.name != "failure_analysis.json")
        need(sorted(f["raw_files_sha256"]) == files and len(files) == 19,
             f"hashed file set != archived raw files ({len(files)})")
        for name in files:
            need(f["raw_files_sha256"].get(name) == sha(d / name), f"{name}: recorded SHA-256 differs from file")

    def counts():
        def ok(lab):
            c = cells.get(lab)
            return (c is not None and c["wl"] is not None and c["complete"] and top["exit"].get(lab) == 0
                    and top["verdict"].get(lab) == "ok" and len(c["warm"]) == c["wl"]["warmups"]
                    and len(c["samp"]) == c["wl"]["reps"])
        done = [l for l in PLAN if ok(l) and cells[l]["samp"]]
        cc = f["cell_counts"]
        need(done == ["A512", "B512", "B2048", "A2048", "A4096", "B4096", "B8192", "A8192", "A16384", "B16384"],
             f"recomputed completed official cells {done}")
        need(cc["official_measured_cells_completed"] == done and cc["official_measured_cells_completed_before_B32768"]
             == len(done) == 10, "completed official cell list/count wrong")
        need(cc["official_measured_cells_planned"] == len(PLAN), "planned official cell count wrong")
        cond_done = sorted(l for l in CONDITIONING if ok(l))
        need(sorted(cc["conditioning_cells_completed_not_official"]) == cond_done
             and not (set(cc["official_measured_cells_completed"]) & CONDITIONING),
             "conditioning cells counted as official / wrong")
        b = cells["B32768"]
        need(cc["B32768"] == f"partial only - {len(b['warm'])}/{b['wl']['warmups']} warmups completed, "
                             f"{len(b['samp'])} measured reps", "B32768 partial description wrong")
        need(cc["A32768"] == ("run" if "A32768" in top["start"] else "not run") == "not run", "A32768 status wrong")
        need(f["a32768"]["run"] is False and "A32768" not in cells, "a32768.run must be false")

    def b32768():
        c, x = cells["B32768"], f["b32768"]
        wl = c["wl"]
        need(x["context_point"] == top["start"]["B32768"]["context_point"] == 32768, "B32768 context_point")
        need(x["actual_prompt_tokens"] == wl["prompt_tokens"] == 32736, "B32768 actual prompt tokens")
        need(x["output_tokens_setting"] == wl["output_tokens"] == 32, "B32768 output tokens")
        need(x["prompt_tokens_seen_by_engine"] == sorted({w["prompt_tokens"] for w in c["warm"]}) == [32736]
             and x["output_tokens_produced"] == sorted({w["output_tokens"] for w in c["warm"]}) == [32],
             "B32768 per-request token counts")
        marks = c["stage3c"]
        need(x["stage3c_chunked_prefill_marker"] is bool(marks) is True, "B32768 Stage3C marker flag")
        need(len({(q, cl) for _, q, cl in marks}) == 1, "B32768 markers disagree on q_len/context_len")
        q_len, ctx = marks[0][1], marks[0][2]
        need(q_len == wl["prompt_tokens"] - 16384 == 16352 and ctx == 16384, "B32768 q_len/context_len from raw")
        need(all(m["q_len"] == q_len and m["context_len"] == ctx for m in x["chunked_prefill_markers"])
             and [m["time"] for m in x["chunked_prefill_markers"]] == [t for t, _, _ in marks],
             "recorded markers differ from raw")
        need(x["chunk_marker_count"] == len(marks) == 32, f"layer marker count {len(marks)}")
        ts = [datetime.strptime("2000-" + t, "%Y-%m-%d %H:%M:%S") for t, _, _ in marks]
        gaps = [(b_ - a_).total_seconds() for a_, b_ in zip(ts, ts[1:])]
        need(eq(x["chunk_marker_interval_s_median"], statistics.median(gaps)), "marker interval median")
        need(x["warmups_completed"] == len(c["warm"]) == 3 and x["measured_reps_completed"] == len(c["samp"]) == 0,
             "B32768 warmup / rep counts")
        need(len(x["warmups"]) == len(c["warm"]), "B32768 warmup list length")
        for rec, w in zip(x["warmups"], c["warm"]):
            need(rec["rep"] == w["rep"] and eq(rec["ttft_ms"], w["ttft_ms"]) and eq(rec["tpot_ms"], w["tpot_ms"])
                 and eq(rec["wall_ms"], w["wall_ms"]), f"B32768 warmup {w['rep']} TTFT/TPOT/wall differ from raw")
        wd = [t for t in top["timeouts"] if t.get("label") == "B32768"]
        pe = top["proc"]["B32768"]
        need(len(wd) == 1 and x["watchdog"]["timed_out"] is True and eq(x["watchdog"]["elapsed_s"], wd[0]["elapsed_s"])
             and x["watchdog"]["timeout_s"] == wd[0]["timeout_s"] == CELL_WATCHDOG_S
             and CELL_WATCHDOG_S <= wd[0]["elapsed_s"] <= CELL_WATCHDOG_S + 5,
             "B32768 watchdog record wrong")
        need(x["watchdog"]["signals_sent"] == wd[0]["signals_sent"] == ["SIGTERM", "SIGKILL"]
             and pe["timed_out"] is True, "B32768 signals must be SIGTERM then SIGKILL")
        need(eq(pe["elapsed_s"], wd[0]["elapsed_s"]) and pe["timeout_s"] == CELL_WATCHDOG_S
             and CELL_WATCHDOG_S <= pe["elapsed_s"] <= CELL_WATCHDOG_S + 5
             and pe["signals_sent"] == wd[0]["signals_sent"] and eq(x["process"]["elapsed_s"], pe["elapsed_s"]),
             "B32768 process-exit record disagrees with the watchdog record / window")
        need(x["oom_detected"] is (c["oom"] > 0) is False, "B32768 OOM flag wrong")
        need(x["full_process_group_cleanup"] is (pe["group_processes_after_leader_exit"] == []
                                                 and pe["group_processes_remaining"] == []) is True,
             "B32768 process-group cleanup wrong")
        need(x["jit_lines_by_phase"].get("measurement", 0) == c["jit_meas"] == 0, "B32768 JIT in measurement")

    def b16384():
        c, x = cells["B16384"], f["b16384"]
        need(x["actual_prompt_tokens"] == c["wl"]["prompt_tokens"] == 16384, "B16384 prompt")
        need(x["stage3c_chunked_prefill_marker"] is bool(c["stage3c"]) is False, "B16384 Stage3C marker flag")
        need(x["completed_successfully"] is True and c["complete"] and top["verdict"]["B16384"] == "ok",
             "B16384 completion")
        need(eq(x["ttft_ms_median"], statistics.median(s["ttft_ms"] for s in c["samp"])), "B16384 TTFT median")

    def discontinuity():
        dd, b16, b32 = f["discontinuity"], cells["B16384"], cells["B32768"]
        t16 = statistics.median(s["ttft_ms"] for s in b16["samp"])
        t32 = statistics.median(w["ttft_ms"] for w in b32["warm"])
        need(dd["b16384_prompt_tokens"] == b16["wl"]["prompt_tokens"] and dd["b32768_prompt_tokens"]
             == b32["wl"]["prompt_tokens"] and eq(dd["prompt_ratio"], b32["wl"]["prompt_tokens"]
                                                   / b16["wl"]["prompt_tokens"]), "discontinuity prompt fields")
        need(eq(dd["b16384_ttft_ms_median_measured"], t16) and eq(dd["b32768_ttft_ms_median_over_completed_warmups"], t32)
             and eq(dd["ttft_ratio_b32768_warmup_over_b16384"], t32 / t16), "discontinuity TTFT fields")
        need(eq(dd["b32768_tpot_ms_median_over_completed_warmups"], statistics.median(w["tpot_ms"] for w in b32["warm"]))
             and eq(dd["b16384_tpot_ms_median_measured"], statistics.median(s["tpot_ms"] for s in b16["samp"])),
             "discontinuity TPOT fields")

    def descriptive():
        de = f["descriptive_evidence_from_failed_run"]
        need("NOT Experiment 5 results" in de["note"], "descriptive-evidence disclaimer missing")
        for lab, rec in de["cells"].items():
            s = cells[lab]["samp"]
            tp = [x["tpot_ms"] for x in s]
            need(rec["tpot_ms_samples"] == tp and eq(rec["tpot_ms_median"], statistics.median(tp))
                 and eq(rec["tpot_ms_p90"], q90(tp))
                 and eq(rec["ttft_ms_median"], statistics.median(x["ttft_ms"] for x in s))
                 and eq(rec["wall_ms_median"], statistics.median(x["wall_ms"] for x in s))
                 and rec["actual_prompt_tokens"] == cells[lab]["wl"]["prompt_tokens"],
                 f"descriptive evidence for {lab} differs from raw")

    def gate_integrity_provenance():
        g = f["correctness_gate"]
        m_py = re.search(r"^(\d+ passed, \d+ warnings?)", gate, re.M)
        m_prep = re.search(r"prep exactness: PASSED \((\d+)/(\d+)\)", gate)
        m_dec = re.search(r"decode append exactness: PASSED \((\d+) state", gate)
        need(g["passed"] is True and manifest["correctness_gate"]["result"] == {"passed": True}
             and g["pytest"] == m_py.group(1) and g["prep_exactness"] == f"{m_prep.group(1)}/{m_prep.group(2)}"
             and m_prep.group(1) == m_prep.group(2) and g["decode_append_steps"] == int(m_dec.group(1))
             and g["dispatch_preflight_passed"] is ("Dispatch preflight: PASSED" in gate) is True
             and g["full_attention_exactness_passed"] is ("full attention exactness: PASSED" in gate) is True
             and "RABIT-2 FINAL TARGETED REGRESSION PASSED" in gate, "correctness gate fields wrong")
        recount = {s: sum(1 for c in integ["checks"] if c["state"] == s)
                   for s in ("passed", "failed", "not_run", "not_evaluated")}
        need(f["integrity_counts"] == integ["counts"] == recount, "integrity counts differ from recount")
        need(f["protected_paths_post_run_status"] == manifest["protected_paths_post_run_status"] == "clean",
             "protected paths not clean")
        need(f["prior_evidence_unchanged"] is manifest["prior_evidence_unchanged"] is True,
             "prior-evidence flag wrong")
        mp = manifest["provenance"]
        need(all(f["provenance"][k] == mp[k] for k in f["provenance"]) and len(f["provenance"]) == 6,
             "provenance differs from manifest")

    for fn, name in ((status, "status"), (hashes, "hashes"), (counts, "counts"), (b32768, "b32768"),
                     (b16384, "b16384"), (discontinuity, "discontinuity"), (descriptive, "descriptive"),
                     (gate_integrity_provenance, "gate/integrity/provenance")):
        guarded(fn, name)

    print(f"attempt: {d.as_posix()} | checks: {n['checks']}")
    if "B32768" in cells:
        w = cells["B32768"]["warm"]
        print(f"recomputed: completed official cells {sum(1 for l in PLAN if cells.get(l) and cells[l]['complete'])}; "
              f"B32768 markers {len(cells['B32768']['stage3c'])}, warmup TTFT ms {[round(x['ttft_ms'], 1) for x in w]}")
    print(f"FAILURES: {len(fails)}")
    for x in fails:
        print(f"  - {x}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
