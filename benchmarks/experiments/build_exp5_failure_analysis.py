"""
Failure analysis for MLSys 2027 Experiment 5, attempt 1 (NOT a scientific
summary). Derived only from the raw evidence archived under
results/mlsys2027/context_scaling/failed_attempt_1/ (read-only):
modal_session.log (per-cell lines, process exits, watchdog, GPU state),
correctness_gate.log, manifest.json, integrity_check.json.

Every number is parsed from those files. The 512-16K measurements are kept only
as descriptive evidence from a failed run; they are not Experiment 5 results.

Usage:
    python benchmarks/mlsys2027/build_exp5_failure_analysis.py \
        --attempt-dir results/mlsys2027/context_scaling/failed_attempt_1
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import statistics
from pathlib import Path

CONDITIONING = ["conditioning_A512", "conditioning_B512"]
OFFICIAL = ["A512", "B512", "B2048", "A2048", "A4096", "B4096", "B8192", "A8192", "A16384", "B16384", "B32768",
            "A32768"]
LEG_PREFIX = re.compile(r"^\[leg(\d+):(\w+)\] (.*)$")
TAG = re.compile(r"^(EXP5_[A-Z_]+)=(\{.*\})\s*$")
CHUNK_MARK = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[triton_attn\.py:\d+\] "
                        r"RABIT2_STAGE3C_CHUNKED_PREFILL_ACTIVE q_len=(\d+) context_len=(\d+)")
JIT = "Triton kernel JIT compilation during inference"
OOM = ("CUDA out of memory", "OutOfMemoryError")
WORDING = ("The current RABIT-KV implementation exhibits a severe performance cliff when the workload crosses "
           "max_num_batched_tokens and activates its non-initial chunked-prefill path.")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def p90(v: list[float]) -> float:
    return statistics.quantiles(v, n=10, method="inclusive")[8]


def parse(session: str) -> tuple[dict, dict]:
    cells: dict = {}
    top: dict = {"leg_start": {}, "leg_exit": {}, "process_exit": {}, "pre_leg": {}, "verdicts": {},
                 "watchdog_timeouts": []}
    index_to_label: dict[int, str] = {}
    for line in session.splitlines():
        m = LEG_PREFIX.match(line)
        if m:
            label = index_to_label.get(int(m.group(1)))
            if label is None:
                continue
            c = cells[label]
            body = m.group(3)
            s = body.strip()
            if s.startswith("EXP5_SAMPLE "):
                c["samples"].append(json.loads(s[len("EXP5_SAMPLE "):]))
            elif s.startswith("EXP5_WARMUP "):
                c["warmups"].append(json.loads(s[len("EXP5_WARMUP "):]))
            elif TAG.match(s):
                tag, payload = s.split("=", 1)
                c["tags"].setdefault(tag, []).append(json.loads(payload))
            elif s in ("EXP5_WARMUP_BEGIN", "EXP5_WARMUP_END", "EXP5_MEASUREMENT_BEGIN", "EXP5_MEASUREMENT_END",
                       "EXP5_WORKER_COMPLETE"):
                c["markers"].append(s)
                c["phase"] = "measurement" if s == "EXP5_MEASUREMENT_BEGIN" else c["phase"]
            cm = CHUNK_MARK.search(body)
            if cm:
                c["chunk_markers"].append({"time": cm.group(1), "q_len": int(cm.group(2)),
                                           "context_len": int(cm.group(3))})
            if JIT in body:
                c["jit"][c["phase"]] = c["jit"].get(c["phase"], 0) + 1
            if any(x in body for x in OOM):
                c["oom_lines"].append(s[:300])
            continue
        tm = TAG.match(line.strip())
        if not tm:
            continue
        tag, payload = tm.group(1), json.loads(tm.group(2))
        if tag == "EXP5_LEG_START":
            index_to_label[payload["index"]] = payload["leg"]
            cells[payload["leg"]] = {"start": payload, "tags": {}, "samples": [], "warmups": [], "markers": [],
                                     "chunk_markers": [], "jit": {}, "oom_lines": [], "phase": "warmup_or_init"}
            top["leg_start"][payload["leg"]] = payload
        elif tag == "EXP5_LEG_EXIT":
            top["leg_exit"][payload["leg"]] = payload
        elif tag == "EXP5_PROCESS_EXIT":
            top["process_exit"][payload["label"]] = payload
        elif tag == "EXP5_PRE_LEG_GPU_STATE":
            top["pre_leg"][payload["leg"]] = payload
        elif tag == "EXP5_CELL_VERDICT":
            top["verdicts"][payload["leg"]] = payload
        elif tag == "EXP5_WATCHDOG_TIMEOUT":
            top["watchdog_timeouts"].append(payload)
    return cells, top


def one(c: dict, tag: str) -> dict | None:
    v = c["tags"].get(tag)
    return v[0] if v else None


def build(attempt: Path) -> dict:
    session = (attempt / "modal_session.log").read_text(encoding="utf-8")
    manifest = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
    integ = json.loads((attempt / "integrity_check.json").read_text(encoding="utf-8"))
    gate_txt = (attempt / "correctness_gate.log").read_text(encoding="utf-8")
    cells, top = parse(session)

    def completed(label: str) -> bool:
        c = cells.get(label)
        wl = one(c, "EXP5_WORKLOAD") if c else None
        return bool(c) and wl is not None and (top["leg_exit"].get(label) or {}).get("returncode") == 0 \
            and (top["verdicts"].get(label) or {}).get("verdict") == "ok" \
            and len(c["warmups"]) == wl["warmups"] and len(c["samples"]) == wl["reps"] \
            and c["markers"][-1:] == ["EXP5_WORKER_COMPLETE"]

    official_done = [l for l in OFFICIAL if completed(l) and cells[l]["samples"]]
    conditioning_done = [l for l in CONDITIONING if completed(l)]

    def latency(label: str) -> dict:
        s = cells[label]["samples"]
        tp, tt, wa = [x["tpot_ms"] for x in s], [x["ttft_ms"] for x in s], [x["wall_ms"] for x in s]
        return {"n": len(s), "tpot_ms_median": statistics.median(tp), "tpot_ms_p90": p90(tp),
                "ttft_ms_median": statistics.median(tt), "wall_ms_median": statistics.median(wa),
                "tpot_ms_samples": tp}

    def facts(label: str) -> dict:
        c = cells[label]
        wl = one(c, "EXP5_WORKLOAD") or {}
        pe = top["process_exit"].get(label) or {}
        return {
            "kv_cache_dtype": c["start"]["kv_cache_dtype"],
            "context_point": c["start"]["context_point"],
            "actual_prompt_tokens": wl.get("prompt_tokens"),
            "output_tokens_setting": wl.get("output_tokens"),
            "prompt_tokens_seen_by_engine": sorted({r["prompt_tokens"] for r in c["warmups"] + c["samples"]}),
            "output_tokens_produced": sorted({r["output_tokens"] for r in c["warmups"] + c["samples"]}),
            "prompt_token_ids_sha256": wl.get("prompt_token_ids_sha256"),
            "engine_initialized": one(c, "EXP5_CAPACITY") is not None,
            "capacity": one(c, "EXP5_CAPACITY"),
            "stage3c_chunked_prefill_marker": bool(c["chunk_markers"]),
            "chunked_prefill_markers": c["chunk_markers"],
            "warmups_completed": len(c["warmups"]),
            "warmups_planned": wl.get("warmups"),
            "measured_reps_completed": len(c["samples"]),
            "measured_reps_planned": wl.get("reps"),
            "oom_detected": bool(c["oom_lines"]),
            "jit_lines_by_phase": c["jit"],
            "process": {k: pe.get(k) for k in ("elapsed_s", "returncode", "timed_out", "timeout_s", "signals_sent",
                                               "group_processes_after_leader_exit", "group_processes_remaining")},
            "gpu_clean_before": (top["pre_leg"].get(label) or {}).get("clean"),
        }

    b16, b32 = facts("B16384"), facts("B32768")
    b16["completed_successfully"] = "B16384" in official_done
    b16["ttft_ms_median"] = latency("B16384")["ttft_ms_median"] if b16["completed_successfully"] else None
    w = cells["B32768"]["warmups"]
    b32["warmups"] = [{"rep": x["rep"], "ttft_ms": x["ttft_ms"], "tpot_ms": x["tpot_ms"], "wall_ms": x["wall_ms"],
                       "prompt_tokens": x["prompt_tokens"], "output_tokens": x["output_tokens"]} for x in w]
    marks = cells["B32768"]["chunk_markers"]
    # vLLM log timestamps carry no year; a fixed leap year only anchors the parse (intervals are year-free).
    times = [dt.datetime.strptime("2000-" + m["time"], "%Y-%m-%d %H:%M:%S") for m in marks]
    gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    b32["chunk_marker_count"] = len(marks)
    b32["chunk_marker_interval_s_median"] = statistics.median(gaps) if gaps else None
    b32["chunk_marker_note"] = ("The marker is logged once per attention-layer implementation (first request only), "
                                "so consecutive markers bracket one layer's non-initial chunk; the median interval "
                                "is a per-layer duration estimate at log-timestamp (1 s) resolution.")
    wd = [x for x in top["watchdog_timeouts"] if x.get("label") == "B32768"]
    b32["watchdog"] = {"timed_out": bool(wd), "timeout_s": wd[0].get("timeout_s") if wd else None,
                       "elapsed_s": wd[0].get("elapsed_s") if wd else None,
                       "signals_sent": wd[0].get("signals_sent") if wd else None}
    b32["full_process_group_cleanup"] = (b32["process"]["group_processes_after_leader_exit"] == []
                                         and b32["process"]["group_processes_remaining"] == [])
    a32_run = "A32768" in top["leg_start"]
    w_ttft = [x["ttft_ms"] for x in w]
    disc = {
        "b16384_prompt_tokens": b16["actual_prompt_tokens"],
        "b32768_prompt_tokens": b32["actual_prompt_tokens"],
        "prompt_ratio": b32["actual_prompt_tokens"] / b16["actual_prompt_tokens"],
        "b16384_ttft_ms_median_measured": b16["ttft_ms_median"],
        "b32768_ttft_ms_median_over_completed_warmups": statistics.median(w_ttft) if w_ttft else None,
        "ttft_ratio_b32768_warmup_over_b16384": (statistics.median(w_ttft) / b16["ttft_ms_median"])
        if w_ttft and b16["ttft_ms_median"] else None,
        "b32768_tpot_ms_median_over_completed_warmups": statistics.median(x["tpot_ms"] for x in w) if w else None,
        "b16384_tpot_ms_median_measured": latency("B16384")["tpot_ms_median"],
        "note": ("Descriptive only. B32768 values are unmeasured warmups (not official samples); B16384 values are "
                 "measured reps from the same failed session."),
    }
    mf = manifest.get("failure", {})
    return {
        "schema": "exp5_failure_analysis/v1",
        "accepted_scientific_summary": False,
        "run_status": manifest["status"],
        "failure_stage": mf.get("stage"),
        "failed_cell": "B32768",
        "reason": "RABIT-KV 32K chunked-prefill path exceeded the pre-registered 900 s watchdog",
        "wording": WORDING,
        "not_claimed": ["an algorithmic limitation of low-bit KV quantization",
                        "a maximum feasible / supported context or memory ceiling",
                        "final Experiment 5 results for the 512-16K cells"],
        "generated_by": "benchmarks/mlsys2027/build_exp5_failure_analysis.py",
        "derived_from": ["modal_session.log", "correctness_gate.log", "manifest.json", "integrity_check.json"],
        "raw_files_sha256": {p.name: sha256(p) for p in sorted(attempt.iterdir())
                             if p.is_file() and p.name != "failure_analysis.json"},
        "cell_counts": {
            "official_measured_cells_planned": len(OFFICIAL),
            "official_measured_cells_completed_before_B32768": len(official_done),
            "official_measured_cells_completed": official_done,
            "B32768": (f"partial only - {len(w)}/{b32['warmups_planned']} warmups completed, "
                       f"{b32['measured_reps_completed']} measured reps"),
            "A32768": "run" if a32_run else "not run",
            "conditioning_cells_completed_not_official": conditioning_done,
            "note": "The two conditioning cells are unmeasured and are never counted as official measured cells.",
        },
        "correctness_gate": {
            "passed": (manifest.get("correctness_gate") or {}).get("result") == {"passed": True},
            "pytest": re.search(r"^(\d+ passed, \d+ warnings?)", gate_txt, re.M).group(1),
            "dispatch_preflight_passed": "Dispatch preflight: PASSED" in gate_txt,
            "prep_exactness": re.search(r"prep exactness: PASSED \((\d+/\d+)\)", gate_txt).group(1),
            "full_attention_exactness_passed": "full attention exactness: PASSED" in gate_txt,
            "decode_append_steps": int(re.search(r"decode append exactness: PASSED \((\d+) state", gate_txt).group(1)),
        },
        "integrity_counts": integ["counts"],
        "protected_paths_post_run_status": manifest.get("protected_paths_post_run_status"),
        "prior_evidence_unchanged": manifest.get("prior_evidence_unchanged"),
        "b16384": b16,
        "b32768": b32,
        "a32768": {"run": a32_run, "reason": "sweep stopped by the B32768 watchdog timeout"},
        "discontinuity": disc,
        "descriptive_evidence_from_failed_run": {
            "note": ("NOT Experiment 5 results. Measured reps of the official cells that completed before the "
                     "stop, kept only as descriptive evidence of a failed run."),
            "cells": {l: {"kv_cache_dtype": cells[l]["start"]["kv_cache_dtype"],
                          "context_point": cells[l]["start"]["context_point"],
                          "actual_prompt_tokens": one(cells[l], "EXP5_WORKLOAD")["prompt_tokens"],
                          **latency(l)} for l in official_done},
        },
        "provenance": {k: (manifest.get("provenance") or {}).get(k) for k in
                       ("git_head", "vllm_kvquant_tree", "rabit_kv2_sha256", "runner_script_sha256",
                        "modal_app_sha256", "worker_sha256")},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attempt-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    out = args.attempt_dir / "failure_analysis.json"
    out.write_text(json.dumps(build(args.attempt_dir), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
