"""
RABIT-KV performance-risk PROFILING DIAGNOSTIC -- local runner (ONE authorized run; DIAGNOSTIC evidence only).

Authorized by the story-lock amendment (docs/MLSYS_PAPER_STORY_LOCK.md, section F): three fixed cases, Llama-3.1-8B
only, no quality scoring, no optimization, no change to the frozen RABIT-KV operating point or to any vLLM source.
The profiled timings are for attribution only and never replace the accepted Experiment 5 / Experiment 6 numbers.

    python benchmarks/mlsys2027/run_perf_profile_diagnostic.py --dry-run
    python benchmarks/mlsys2027/run_perf_profile_diagnostic.py            # launch + monitor + analyze
    python benchmarks/mlsys2027/run_perf_profile_diagnostic.py --monitor  # resume monitoring (never spawns)
    python benchmarks/mlsys2027/run_perf_profile_diagnostic.py --reparse  # re-analyze the downloaded log

Preflight: clean tree, HEAD pushed, the vendored vLLM tree and the canonical quality evaluator unchanged since the
story-lock amendment commit, the three serving sources equal to the accepted hashes, offline tests green, attempt
directory absent. Launch: `modal run --detach` (spawned FunctionCall; the remote log is tee'd to a Modal Volume and
downloaded when DONE.json appears).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import perf_profile_analysis as pa  # noqa: E402

MODAL_APP = HERE / "perf_profile_modal.py"
OUT_DIR = ROOT / "results" / "mlsys2027" / "diagnostics" / "perf_risk_profile" / "attempt_1"
RUN_ID = "perf_risk_profile_attempt_1"
LOG_VOLUME = "rabit-kv-mlsys2027-perf-profile-logs"
STORY_LOCK_COMMIT = "8273a6a"  # story-lock amendment; the vLLM tree must be unchanged since
QUALITY_FREEZE_COMMIT = "164c17f"
FROZEN_SINCE_STORY_LOCK = ("vllm-kvquant",)
FROZEN_SINCE_QUALITY = ("benchmarks/mlsys2027/canonical_rabit_quality.py", "results/mlsys2027/canonical_quality_v2")
IDENTITY = {  # LF-normalized SHA-256 recorded by the accepted serving evidence (Experiments 5 / 6)
    "rabit_kv2.py": "7e628c94eebb9fe689bf416ea61f748c0f909a82d0f229c463edd1a0df92e6ae",
    "triton_attn.py": "0c75dd06d3893c4f6240c4d70bd51d4888855100d220b673c6b77c1a0a4e16cf",
    "rabit_kv2_stage3c_shared_decode.py": "ace859940728721d6198a84f0111e793c8d85a2eee609b4c5df1981586aade1f",
}
IDENTITY_PATHS = {
    "rabit_kv2.py": "vllm-kvquant/vllm/v1/attention/ops/rabit_kv2.py",
    "triton_attn.py": "vllm-kvquant/vllm/v1/attention/backends/triton_attn.py",
    "rabit_kv2_stage3c_shared_decode.py": "vllm-kvquant/vllm/v1/attention/ops/rabit_kv2_stage3c_shared_decode.py",
}
HARNESS = ("perf_profile_plugin.py", "perf_profile_worker.py", "perf_profile_modal.py", "perf_profile_analysis.py",
           "run_perf_profile_diagnostic.py", "test_perf_profile.py", "exp6_workload.py", "exp3_correctness_gate.py",
           "exp3_watchdog.py")
# label, case, dtype, mode, conditioning prefix, measured prefix (c3: reps), watchdog seconds -- FIXED before the run
LEGS = (
    ("c1_rabit_off", "c1", "rabit_kv2", "off", 64, 96, 900),
    ("c1_rabit_events", "c1", "rabit_kv2", "cuda_events", 64, 96, 1200),
    ("c1_rabit_torch", "c1", "rabit_kv2", "torch_profiler", 64, 16, 1200),
    ("c1_bf16_off", "c1", "bfloat16", "off", 64, 96, 900),
    ("c1_bf16_events", "c1", "bfloat16", "cuda_events", 64, 96, 900),
    ("c2_rabit_off", "c2", "rabit_kv2", "off", 64, 96, 1500),
    ("c2_rabit_events", "c2", "rabit_kv2", "cuda_events", 64, 96, 2400),
    ("c2_bf16_off", "c2", "bfloat16", "off", 64, 96, 900),
    ("c2_bf16_events", "c2", "bfloat16", "cuda_events", 64, 96, 900),
    ("c3_rabit_off", "c3", "rabit_kv2", "off", 0, 1, 1500),
    ("c3_rabit_events", "c3", "rabit_kv2", "cuda_events", 0, 1, 2400),
    ("c3_bf16_off", "c3", "bfloat16", "off", 0, 1, 900),
    ("c3_bf16_events", "c3", "bfloat16", "cuda_events", 0, 1, 900),
)
CASES = {"c1": {"prompt_tokens": 2048, "concurrency": 8, "max_num_seqs": 8},
         "c2": {"prompt_tokens": 8192, "concurrency": 32, "max_num_seqs": 32},
         "c3": {"prompt_tokens": 32736, "concurrency": 1, "max_num_seqs": 32}}
ACCEPTED_REFERENCE = {  # accepted paper numbers (context only; never replaced by this diagnostic)
    "c1": {"source": "Experiment 6 L2048 C8", "requests_per_s": {"bfloat16": 7.218, "rabit_kv2": 1.844}},
    "c2": {"source": "Experiment 6 L8192 C32", "requests_per_s": {"bfloat16": 2.346, "rabit_kv2": 0.870}},
    "c3": {"source": "Experiment 5 B32768 (32,736-token prompt)", "ttft_ms": {"bfloat16": 4435.0,
                                                                              "rabit_kv2": 114376.8}},
}
POLL_S, MONITOR_TIMEOUT_S = 60, 6 * 3600


def git(*args: str) -> str:
    p = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{p.stdout}\n{p.stderr}")
    return p.stdout.strip()


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def legs_arg() -> str:
    return ",".join(f"{label}={case}:{dtype}:{mode}:{cond}:{meas}:{timeout}"
                    for label, case, dtype, mode, cond, meas, timeout in LEGS)


def preflight() -> dict:
    checks = {}
    checks["tree_clean"] = git("status", "--porcelain") == ""
    head = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    git("fetch", "origin", branch)
    checks["head_pushed"] = git("rev-parse", f"origin/{branch}") == head
    checks["vllm_tree_unchanged_since_story_lock"] = subprocess.run(
        ["git", "diff", "--quiet", STORY_LOCK_COMMIT, "HEAD", "--", *FROZEN_SINCE_STORY_LOCK], cwd=ROOT).returncode == 0
    checks["quality_evidence_unchanged"] = subprocess.run(
        ["git", "diff", "--quiet", QUALITY_FREEZE_COMMIT, "HEAD", "--", *FROZEN_SINCE_QUALITY], cwd=ROOT).returncode == 0
    actual = {n: sha256_lf(ROOT / p) for n, p in IDENTITY_PATHS.items()}
    checks["serving_sources_equal_accepted_hashes"] = actual == IDENTITY
    t = subprocess.run([sys.executable, str(HERE / "test_perf_profile.py")], cwd=ROOT, text=True, capture_output=True,
                       encoding="utf-8", errors="replace")
    checks["offline_tests_pass"] = t.returncode == 0
    return {"checks": checks, "all_passed": all(checks.values()), "head": head, "branch": branch,
            "serving_source_sha256_lf": actual, "offline_tests_tail": (t.stdout + t.stderr)[-600:],
            "harness_sha256_lf": {n: sha256_lf(HERE / n) for n in HARNESS}}


def build_snapshot() -> dict:
    out = Path(tempfile.mkdtemp(prefix="perf_vllm_snapshot_")) / "vllm_kvquant_snapshot.zip"
    git("-c", "core.autocrlf=false", "archive", "--format=zip", "-o", str(out), "HEAD:vllm-kvquant")
    return {"path": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "bytes": out.stat().st_size,
            "source": "git archive --format=zip HEAD:vllm-kvquant (core.autocrlf=false)"}


def volume_get(remote: str, local: Path) -> bool:
    r = subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", LOG_VOLUME, remote, str(local)],
                       cwd=ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace")
    return r.returncode == 0 and local.is_file()


def monitor() -> dict:
    deadline = time.time() + MONITOR_TIMEOUT_S
    done_path = OUT_DIR / "DONE.json"
    while time.time() < deadline:
        if volume_get(f"/{RUN_ID}/DONE.json", done_path):
            break
        print(f"[{now()}] waiting for /{RUN_ID}/DONE.json ...", flush=True)
        time.sleep(POLL_S)
    else:
        raise RuntimeError("monitor timeout: DONE.json never appeared")
    if not volume_get(f"/{RUN_ID}/remote_session.log", OUT_DIR / "remote_session.log"):
        raise RuntimeError("could not download remote_session.log")
    return json.loads(done_path.read_text(encoding="utf-8"))


def gates(legs: dict, top: list[str], done: dict) -> list[dict]:
    out = []

    def add(name, ok, observed=None):
        out.append({"gate": name, "passed": bool(ok), "observed": observed})

    add("remote function completed", done.get("status") == "complete", done)
    ident = (pa.tagged(top, "PERFRUN_SOURCE_IDENTITY") or [{}])[0]
    add("serving sources in the image equal the accepted hashes", ident.get("match") is True and ident.get("actual") == IDENTITY,
        ident.get("actual"))
    env = (pa.tagged(top, "PERFRUN_ENVIRONMENT") or [{}])[0]
    gpus = env.get("gpus") or []
    add("exactly one NVIDIA H100", len(gpus) == 1 and "H100" in gpus[0].get("name", ""), [g.get("name") for g in gpus])
    gate = (pa.tagged(top, "PERFRUN_GATE_EXIT") or [{}])[0]
    add("frozen RABIT-KV correctness gate passed", gate.get("returncode") == 0 and gate.get("timed_out") is False, gate)
    exits = {e["label"]: e for e in pa.tagged(top, "PERFRUN_LEG_EXIT")}
    for label, case, dtype, mode, cond, meas, _ in LEGS:
        r = legs.get(label)
        e = exits.get(label, {})
        if r is None:
            add(f"{label}: leg present", False)
            continue
        cfg, c = r["effective_engine_config"] or {}, CASES[case]
        rs = r["request_stats"]
        n_req = meas if case != "c3" else meas
        add(f"{label}: process exit 0, worker complete", e.get("returncode") == 0 and not e.get("timed_out") and r["complete"],
            {"returncode": e.get("returncode"), "timed_out": e.get("timed_out"), "failure": r["failure"]})
        add(f"{label}: engine configuration is the accepted shape",
            cfg.get("max_num_seqs") == c["max_num_seqs"] and cfg.get("max_num_batched_tokens") == 16384
            and cfg.get("enable_chunked_prefill") is True and cfg.get("enable_prefix_caching") is False
            and cfg.get("block_size") == 32 and cfg.get("max_model_len") == 32768 and cfg.get("enforce_eager") is True
            and cfg.get("engine_cache_dtype") == dtype, cfg)
        add(f"{label}: workload shape", rs["requests"] == n_req and rs["prompt_tokens"] == [c["prompt_tokens"]]
            and rs["output_tokens"] == [32] and rs["finish_reasons"] == ["length"],
            {k: rs[k] for k in ("requests", "prompt_tokens", "output_tokens", "finish_reasons")})
        if dtype == "rabit_kv2":
            s3 = r["stage_impl"] or {}
            add(f"{label}: Stage3C shared_decode / query block 32, in-tree profilers off",
                s3.get("effective_impl") == "shared_decode" and s3.get("effective_query_block") == 32
                and all(v in (None, "0") for v in (s3.get("intree_profiling_env") or {}).values()), s3)
        loaded = (r["plugin_module_loaded"] or {}).get("perf_profile_plugin_in_sys_modules")
        if mode == "off":
            add(f"{label}: no profiler in the process", loaded is False and not r["installs"] and not r["profile_phases"],
                {"loaded": loaded, "installs": len(r["installs"]), "phases": r["profile_phases"]})
        elif mode == "cuda_events":
            b = r.get("breakdown_measured") or {}
            add(f"{label}: profiler installed and the measured phase recorded in both domains",
                bool(r["installs"]) and all(i.get("ok") for i in r["installs"]) and b.get("steps", 0) > 0
                and b.get("meta_errors") == 0 and (r.get("measured_profile_raw") or {}).get("gpu_domain") is True
                and b["step_windows_total"]["gpu_span_ms"] > 0,
                {"installs": [i.get("ok") for i in r["installs"]], "steps": b.get("steps"),
                 "meta_errors": b.get("meta_errors")})
        else:
            add(f"{label}: torch.profiler plugin installed", bool(r["installs"]) and all(i.get("ok") for i in r["installs"]),
                [i.get("ok") for i in r["installs"]])
    return out


def analyze(write: bool) -> dict:
    text = (OUT_DIR / "remote_session.log").read_text(encoding="utf-8", errors="replace")
    done = json.loads((OUT_DIR / "DONE.json").read_text(encoding="utf-8"))
    raw, top = pa.split_legs(text)
    legs = {label: pa.leg_report(lines) for label, lines in raw.items()}
    g = gates(legs, top, done)
    cases = {}
    for case in CASES:
        entry = {"shape": CASES[case], "accepted_reference": ACCEPTED_REFERENCE[case], "legs": {}}
        for label, c, dtype, mode, *_ in LEGS:
            if c != case or label not in legs:
                continue
            r = legs[label]
            entry["legs"][label] = {
                "dtype": dtype, "mode": mode, "measured_wall_s": (r["measured_summary"] or {}).get("wall_s"),
                "requests": r["request_stats"]["requests"], "requests_per_s": r.get("requests_per_s"),
                "mean_ttft_ms": r["request_stats"]["mean_ttft_ms"], "mean_tpot_ms": r["request_stats"]["mean_tpot_ms"],
                "mean_gpu_utilization_pct": (r["gpu_utilization"] or {}).get("mean_utilization_gpu_pct"),
                "gpu_utilization_samples": (r["gpu_utilization"] or {}).get("samples")}
        short = {"rabit_kv2": "rabit", "bfloat16": "bf16"}
        entry["instrumentation_overhead"] = {
            dtype: pa.overhead(legs[f"{case}_{s}_off"], legs[f"{case}_{s}_events"])
            for dtype, s in short.items() if f"{case}_{s}_off" in legs and f"{case}_{s}_events" in legs}
        for dtype, s in short.items():
            b = legs.get(f"{case}_{s}_events", {}).get("breakdown_measured")
            if b:
                entry[f"breakdown_{s}"] = b
        if case == "c1" and "c1_rabit_torch" in legs:
            entry["torch_profiler_cross_check"] = legs["c1_rabit_torch"]["torch_profiler"]
            entry["torch_profiler_leg_wall_s"] = (legs["c1_rabit_torch"]["measured_summary"] or {}).get("wall_s")
        cases[case] = entry
    summary = {
        "title": "RABIT-KV performance-risk profiling diagnostic (attempt 1)",
        "classification": "DIAGNOSTIC EVIDENCE ONLY -- attribution; NOT paper performance evidence; the accepted "
                          "Experiment 5 / Experiment 6 numbers are not replaced",
        "valid": all(x["passed"] for x in g), "gates": g, "cases": cases,
        "domains": "gpu_span_ms = CUDA event-pair span on the device timeline (device work plus device waiting for "
                   "the host inside the window); host_ms = perf_counter on the engine thread. Never added together.",
        "generated_utc": now(),
    }
    if write:
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        (OUT_DIR / "legs.json").write_text(json.dumps(legs, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--monitor", action="store_true")
    ap.add_argument("--reparse", action="store_true")
    args = ap.parse_args(argv)
    if args.reparse:
        s = analyze(write=True)
        print(json.dumps({"valid": s["valid"], "failed_gates": [x for x in s["gates"] if not x["passed"]]}, indent=1))
        return 0 if s["valid"] else 1
    if not args.monitor:
        pre = preflight()
        print(json.dumps({"preflight": pre["checks"], "head": pre["head"], "legs": legs_arg()}, indent=1))
        if not pre["all_passed"]:
            print(pre["offline_tests_tail"])
            raise SystemExit("PREFLIGHT FAILED; nothing launched")
        if args.dry_run:
            return 0
        if OUT_DIR.exists():
            raise SystemExit(f"{OUT_DIR} exists: this diagnostic is authorized for ONE run")
        OUT_DIR.mkdir(parents=True)
        snap = build_snapshot()
        launch_record = OUT_DIR / "launch_record.json"
        cmd = [sys.executable, "-m", "modal", "run", "--detach", f"{MODAL_APP}::main", "--legs", legs_arg(),
               "--identity", json.dumps(IDENTITY, sort_keys=True), "--run-id", RUN_ID,
               "--launch-record", str(launch_record), "--git-commit", pre["head"]]
        manifest = {"run_id": RUN_ID, "git_commit": pre["head"], "branch": pre["branch"], "preflight": pre,
                    "vllm_kvquant_snapshot": snap, "legs": [dict(zip(("label", "case", "dtype", "mode", "conditioning",
                                                                      "measured", "watchdog_s"), leg)) for leg in LEGS],
                    "cases": CASES, "command": cmd, "launched_utc": now(),
                    "authorization": "docs/MLSYS_PAPER_STORY_LOCK.md section F (profiling only; no optimization)"}
        (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        env = {**os.environ, "PERF_VLLM_SNAPSHOT": snap["path"], "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        with (OUT_DIR / "launch.log").open("w", encoding="utf-8") as log:
            code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True).returncode
        print(f"modal run --detach exit code {code}; launch record present: {launch_record.is_file()}", flush=True)
        if not launch_record.is_file():
            raise SystemExit("launch failed before a FunctionCall was spawned (see launch.log)")
    done = monitor()
    print(json.dumps(done), flush=True)
    s = analyze(write=True)
    print(json.dumps({"valid": s["valid"], "failed_gates": [x for x in s["gates"] if not x["passed"]]}, indent=1))
    return 0 if s["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
