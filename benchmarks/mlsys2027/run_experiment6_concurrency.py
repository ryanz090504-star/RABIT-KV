"""
RABIT-KV MLSys 2027 -- Experiment 6: concurrency / throughput scaling, BF16 vs
RABIT-KV (finalized: K3/V2/G32/R4/META8g64, Stage3C shared_decode, QUERY_BLOCK 32).

EXECUTION-PATH AMENDMENT (reviewed, before any performance run): the plan
preferred `vllm bench throughput`. Audit: the CLI accepts
--kv-cache-dtype rabit_kv2 (block size multiple of 32) but samples at
temperature 1.0 / top_p 1.0 and exposes no per-request timestamps / token IDs,
which the pre-registered metrics and correctness audit need. The sweep therefore
uses vLLM's normal multi-request LLM.generate API -- the path validated by the
accepted concurrency correctness smoke (results/mlsys2027/concurrency_smoke/).
Benchmark-harness change only; rabit_kv2.py, shared_decode and triton_attn.py
are untouched. CONCURRENCY AMENDMENT (reviewed): max_num_seqs = target
concurrency C at EVERY point, identical for both dtypes (the plan's text kept
the canonical 32 for C <= 32). SHADOW-CONDITIONING AMENDMENT (reviewed; the
final warmup amendment, after the JIT-contaminated L2048 attempts 1 and 2; it
supersedes attempt 2's C-request conditioning): every point first runs an
UNMEASURED shadow workload -- ONE queued batch of 256 fixed shadow prompts of the
sweep length (one set per length, the same for both dtypes / all trials / all C),
32 greedy output tokens, under the point's own engine (max_num_seqs = C, closed
loop) -- then the original 2 warmup requests, then the unchanged 256 measured
requests. A point is interpretable only if measured-phase JIT == 0 and its
shadow-conditioning pass is valid.

REMOTE-LOG CAPTURE (harness-only, after the infrastructure-aborted L2048
attempt 3, whose local Modal client disconnected): the sweep is launched with
`modal run --detach` under a unique run id; the Modal app tees its complete
stdout to a Modal Volume and writes DONE.json at the end. The runner downloads
remote_session.log + DONE.json (waiting for DONE.json if the local client
stream ended early) and analyzes the REMOTE log; the locally streamed
modal_session.log is kept as a secondary record. Protocol unchanged.

Frozen workload (exp6_workload.py; hashes frozen in exp6_protocol.json): prompt
lengths 2048 and 8192 (separate sweeps / separate Modal runs); concurrency
{1,4,8,16,32,64}; per point 2 warmup requests (discarded) + 256 measured
requests (one fixed ordered set of 256 distinct prompts per length, reused across
dtype / trial / concurrency); 32 greedy output tokens (ignore_eos); 3 trials,
dtype order T1 BF16->RABIT, T2 RABIT->BF16, T3 BF16->RABIT; ascending
concurrency within each dtype. One fresh engine per dtype x length x concurrency
x trial. Every point is attempted independently.

Per point: wall time, requests/s, output and total tokens/s, per-request
end-to-end latency (median / p90 / p99), TTFT and TPOT (median / p90),
target vs OBSERVED max IN-FLIGHT (overlapping) concurrency and the all-C in-flight
overlap window (from the engine-core scheduled / last-token timestamps; see
CONCURRENCY_TERMINOLOGY), completion counts and
output-length validity, preemptions (vllm:num_preemptions delta; logged
"Preemptions" fallback; else reported unavailable), allocator capacity,
nvidia-smi GPU memory (not request KV memory), OOM / failure / timeout, and JIT
lines by phase. Outcome classes (never collapsed into "failure"):
engine_or_request_failure | oom_or_allocation_failure |
completed_with_preemption | target_concurrency_not_reached |
sustained_target_concurrency. The paper reports the highest successfully tested
concurrency; no true maximum is claimed unless the sweep brackets the boundary.
RABIT-only extension points {128, 256} are pre-registered but NOT enabled.

Usage:
    python benchmarks/mlsys2027/run_experiment6_concurrency.py --prompt-tokens 2048 --dry-run
    python benchmarks/mlsys2027/run_experiment6_concurrency.py --prompt-tokens 2048
    python benchmarks/mlsys2027/run_experiment6_concurrency.py --write-protocol   (once, before commit)
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import statistics
import sys
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp6_workload as wl  # noqa: E402
import run_experiment6_smoke as rs6  # noqa: E402  (committed; accepted configuration + protected paths)
import run_stage3c_cliff_diagnostic as rd  # noqa: E402  (committed; AST helpers)
from run_experiment3_deployment import (  # noqa: E402
    FAILED, NOT_EVALUATED, NOT_RUN, PASSED, _function, _module_assign, make_console_encoding_safe, now, rel,
    run_git, sha256, sha256_raw, stream_command,
)

ROOT = rs6.ROOT
HERE = ROOT / "benchmarks" / "mlsys2027"
RUNNER_SCRIPT = Path(__file__).resolve()
MODAL_APP = HERE / "exp6_modal.py"
WORKER = HERE / "exp6_worker.py"
WORKLOAD = HERE / "exp6_workload.py"
PROTOCOL = HERE / "exp6_protocol.json"
OFFLINE_TESTS = HERE / "test_experiment6_concurrency.py"
REF_MODAL_APP = rs6.MODAL_APP
EXP5_WORKER = rs6.EXP5_WORKER
SMOKE_DIR = rs6.OUT_DIR
BASE_OUT = ROOT / "results" / "mlsys2027" / "concurrency_scaling"
# Failed diagnostic attempts: archived byte-exact, never interpreted, never pooled with any other attempt.
DIAGNOSTIC_ATTEMPTS = {2048: [BASE_OUT / "L2048" / "jit_contaminated_attempt_1",
                              BASE_OUT / "L2048" / "jit_contaminated_attempt_2"]}
ATTEMPT_DIR_NAME = "shadow_conditioned"  # output directory of the accepted-protocol sweep per prompt length
# Infrastructure-aborted attempts: excluded and never pooled, but NOT part of the frozen protocol file.
INFRA_ABORTED_ATTEMPTS = {2048: [BASE_OUT / "L2048" / "infrastructure_aborted_attempt_3"]}


def excluded_attempts(length: int) -> list[Path]:
    return [*DIAGNOSTIC_ATTEMPTS.get(length, []), *INFRA_ABORTED_ATTEMPTS.get(length, [])]


_ALL_EXCLUDED = [d for L in sorted({*DIAGNOSTIC_ATTEMPTS, *INFRA_ABORTED_ATTEMPTS}) for d in excluded_attempts(L)]
EVIDENCE_DIRS = [*rs6.EVIDENCE_DIRS, SMOKE_DIR, *_ALL_EXCLUDED]
PROTECTED_PATHS = [*rs6.PROTECTED_PATHS, SMOKE_DIR, rs6.RUNNER_SCRIPT, rs6.MODAL_APP, rs6.WORKER,
                   *_ALL_EXCLUDED]
SESSION_LOG_VOLUME = "rabit-kv-mlsys2027-exp6-session-logs"
REMOTE_POLL_S = 120
MUST_BE_COMMITTED = [RUNNER_SCRIPT, MODAL_APP, WORKER, WORKLOAD, PROTOCOL, OFFLINE_TESTS, rs6.rsd.GATE,
                     rs6.rsd.WATCHDOG]
EXPECTED_CAPACITY = {"bfloat16": 393024, "rabit_kv2": 2074592}
GATE_TIMEOUT_S, POINT_TIMEOUT_S, MODAL_FUNCTION_TIMEOUT_S, POINTS_PER_SWEEP = 600, 1200, 45000, 36
WATCHDOG_BUDGET_S = GATE_TIMEOUT_S + POINTS_PER_SWEEP * POINT_TIMEOUT_S  # 43800
TAG = re.compile(r"^(EXP6_[A-Z0-9_]+)=(\{.*\})\s*$")
TOP_TAG = re.compile(r"^(S3C_[A-Z0-9_]+)=(\{.*\})\s*$")
LOGGED_PREEMPTIONS = re.compile(r"Preemptions: (\d+)")
MARKERS = ["EXP6_SHADOW_CONDITIONING_BEGIN", "EXP6_SHADOW_CONDITIONING_END", "EXP6_WARMUP_BEGIN", "EXP6_WARMUP_END",
           "EXP6_MEASURED_BEGIN", "EXP6_MEASURED_END", "EXP6_WORKER_COMPLETE"]
JIT_PHASES = ("setup", "shadow_conditioning", "original_warmup", "measured", "after")
JIT = "Triton kernel JIT compilation during inference"
OOM = ("CUDA out of memory", "OutOfMemoryError", "out of memory")
CLASSES = ("engine_or_request_failure", "oom_or_allocation_failure", "completed_with_preemption",
           "target_concurrency_not_reached", "sustained_target_concurrency")
# Classification precedence (unchanged): OOM -> engine/request failure -> completed with preemption ->
# target not reached -> sustained target concurrency.
CLASSES_ORDER = ("oom_or_allocation_failure", "engine_or_request_failure", "completed_with_preemption",
                 "target_concurrency_not_reached", "sustained_target_concurrency")
SCOPE = ("Experiment 6 concurrency / throughput scaling (closed-loop, fixed concurrency, 256 measured requests per "
         "point). Throughput numbers are from this harness only. Highest successfully tested concurrency is reported; "
         "no true maximum is claimed unless the sweep brackets the failure boundary.")
CONCURRENCY_TERMINOLOGY = (
    "Concurrency derived from request [scheduled_ts, last_token_ts] intervals is OVERLAPPING IN-FLIGHT concurrency "
    "(requests that have been scheduled and not yet produced their last token). It is NOT strict GPU-resident or "
    "continuously-executing concurrency: a preempted request stays in flight while not resident. If preemption occurs "
    "(or its status is unavailable), the point's overlap statistics are reported but must not by themselves be used to "
    "claim sustained residency at target C.")
AMENDMENTS = {
    "execution_path": ("vllm bench throughput accepts kv_cache_dtype=rabit_kv2 but samples at temperature 1.0 and "
                       "exposes no per-request timestamps / token IDs required by the pre-registered metrics and audit; "
                       "the sweep uses the vLLM multi-request LLM.generate API validated by the accepted concurrency "
                       "correctness smoke. Benchmark-harness change only."),
    "max_num_seqs": "max_num_seqs = target concurrency C at every point, identical for BF16 and RABIT.",
    "compile_conditioning_superseded": (
        "SUPERSEDED by shadow_conditioning after L2048 attempt 2 (jit_contaminated_attempt_2) still showed "
        "measured-phase JIT at RABIT C16 / C32. Original text: "
        "Point-matched compile-conditioning phase (reviewed amendment, pre-registered before the rerun). "
        "Justification: the original two warmup requests contain only 4096 prompt tokens at L2048 and cannot "
        "exercise concurrency-induced chunked prefill; at C>=16, C*2048 > max_num_batched_tokens=16384, and the "
        "H100 diagnostic run (L2048 jit_contaminated_attempt_1, Modal ap-OwRGAH3ak5JiNMafSwvN6Q) showed the Stage3C "
        "marker at every RABIT C>=16 point, none at C<=8, and measured-phase compilation of "
        "_rabit2_shared_decode_closed_page_partial_kernel, _rabit2_tile32_reduce_partials_kernel and (C32/C64) "
        "_rabit2_tail_partial_kernel. Before the 2 original warmup requests, every point (BF16 and RABIT alike) "
        "runs exactly C fixed conditioning prompts of the sweep prompt length, 32 greedy output tokens, "
        "concurrently in one llm.generate call under max_num_seqs=C, disjoint from the measured and warmup prompts. "
        "Outputs and timings of this phase are discarded from all result tables. This is a pre-measurement "
        "kernel-conditioning fix, not a change to the measured workload."),
    "shadow_conditioning": (
        "Full point-matched shadow-workload conditioning (reviewed; the FINAL warmup amendment, pre-registered before "
        "the rerun). Attempt 1: the 2-request warmup did not exercise concurrency-induced Stage3C. Attempt 2 (Modal "
        "ap-l630ypsTpdW1bItQYkNG1x): a one-shot C-request conditioning batch also failed to reproduce the measured "
        "closed-loop scheduler states (measured-phase JIT of _rabit2_shared_decode_closed_page_partial_kernel and "
        "_rabit2_tile32_reduce_partials_kernel at RABIT C16, _rabit2_tail_partial_kernel at C32, all three trials). "
        "Observed cause: the measured workload continuously admits new requests while existing requests decode, so "
        "prompt chunks share the 16,384-token scheduler budget at different offsets. The final conditioning protocol "
        "therefore mirrors the full measured closed-loop workload shape instead of targeting specific RABIT kernels: "
        "before the 2 original warmup requests, every point runs ONE queued batch of 256 UNMEASURED shadow requests "
        "(same prompt length, 32 greedy output tokens with ignore_eos, max_num_seqs=C, same engine, same closed-loop "
        "admission; different prompt token IDs: one fixed set per prompt length, disjoint from the measured and warmup "
        "prompts, identical for BF16 and RABIT, all trials and all C). Symmetric across dtypes, independent of RABIT "
        "kernel names. It is a benchmark warmup / JIT-conditioning change only and is NOT part of the measured "
        "workload; its throughput / latency never enter performance tables."),
}


def out_dir(length: int) -> Path:
    return BASE_OUT / f"L{length}" / ATTEMPT_DIR_NAME


# ------------------------------------------------------------------ protocol metadata (frozen, committed)
def build_protocol() -> dict:
    return {
        "experiment": "Experiment 6 -- concurrency / throughput scaling", "amendments": AMENDMENTS,
        "prompt_lengths": list(wl.PROMPT_LENGTHS), "concurrency_grid": list(wl.CONCURRENCY_GRID),
        "rabit_extension_grid_pre_registered_not_enabled": list(wl.RABIT_EXTENSION_GRID),
        "rabit_extension_rule": ("RABIT-only points {128, 256} may be run ONLY after review, and only if BF16 fails "
                                 "first (BF16 is not sustained_target_concurrency at some base point where RABIT is, "
                                 "in every trial); never chosen from the throughput shape"),
        "measured_requests": wl.MEASURED_REQUESTS, "warmup_requests": wl.WARMUP_REQUESTS,
        "output_tokens": wl.OUTPUT_TOKENS, "sampling": {"temperature": 0.0, "ignore_eos": True},
        "trials": wl.TRIALS, "trial_dtype_order": {str(k): list(v) for k, v in wl.TRIAL_DTYPE_ORDER.items()},
        "within_trial_order": "ascending concurrency", "max_num_seqs_rule": "max_num_seqs = target concurrency",
        "prompt_construction": ("BOS 128000 + pseudo-random ordinary vocabulary IDs in [1000, 30000) from "
                                "SHA-256(tag:length:index:position); see exp6_workload.py"),
        "prompt_sets": {str(L): {"measured": wl.set_digest(wl.measured_prompts(L)),
                                 "warmup": wl.set_digest(wl.warmup_prompts(L))} for L in wl.PROMPT_LENGTHS},
        "shadow_conditioning": {
            "rule": ("per point, before the 2 original warmup requests: ONE queued llm.generate batch of the 256 "
                     "shadow-conditioning prompts of the sweep length, 32 greedy output tokens (ignore_eos), under the "
                     "point's engine (max_num_seqs=C; the scheduler replenishes slots as they free, as in the measured "
                     "phase); identical for BF16 and RABIT; unmeasured (outputs / timings never enter result tables; "
                     "emitted only for validity checks)"),
            "requests_per_point": wl.SHADOW_CONDITIONING_REQUESTS, "output_tokens": wl.OUTPUT_TOKENS,
            "same_set_for": "both dtypes, all trials and all concurrency values of a prompt length",
            "phase_order": ["engine setup", "shadow conditioning (256)", "2 original warmup", "256 measured"],
            "validity": ["256/256 completed", "every prompt exactly the sweep prompt length and pinned hash",
                         "every output 32 tokens (finish_reason length)",
                         "target overlapping in-flight concurrency C reached (observed max >= C, all-C overlap > 0)",
                         "no OOM / request failure / watchdog", "same engine (max_num_seqs=C, dtype, selector / QB, "
                         "profiling off) as the measured phase"],
            "jit_accounting_phases": list(JIT_PHASES),
            "jit_rule": ("JIT is expected / allowed in setup, shadow_conditioning and original_warmup; a point is "
                         "interpretable only if measured_phase_jit == 0 (and its shadow pass is valid). If ANY measured "
                         "point still has JIT: STOP -- no new warmup scheme, no rerun, affected numbers not accepted; "
                         "return for review."),
            "prompt_sets": {str(L): wl.set_digest(wl.shadow_conditioning_prompts(L)) for L in wl.PROMPT_LENGTHS}},
        "excluded_attempts": {str(L): {"dirs": [rel(d) for d in ds], "use": "diagnostic only; never interpreted or "
                                       "pooled with any other attempt"} for L, ds in DIAGNOSTIC_ATTEMPTS.items()},
        "points": {str(L): wl.plan_points(L) for L in wl.PROMPT_LENGTHS},
        "outcome_classes": list(CLASSES), "outcome_class_order": list(CLASSES_ORDER),
        "concurrency_terminology": CONCURRENCY_TERMINOLOGY, "expected_capacity_tokens": EXPECTED_CAPACITY,
        "watchdogs": {"gate_s": GATE_TIMEOUT_S, "point_s": POINT_TIMEOUT_S, "modal_backstop_s": MODAL_FUNCTION_TIMEOUT_S},
    }


def load_protocol() -> dict:
    committed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if committed != json.loads(json.dumps(build_protocol())):
        raise RuntimeError("exp6_protocol.json differs from the regenerated protocol (workload drift)")
    return committed


# ------------------------------------------------------------------ preflight
def assert_protected_paths_clean(context: str) -> None:
    status = run_git("status", "--short", "--", *[rel(p) for p in PROTECTED_PATHS])
    if status:
        raise RuntimeError(f"CRITICAL: protected paths changed ({context}):\n" + status)


def prior_evidence_digest() -> dict:
    out = {}
    for d in EVIDENCE_DIRS:
        if d.is_dir():
            for f in sorted(p for p in d.rglob("*") if p.is_file()):
                out[f"{rel(d)}/{f.relative_to(d).as_posix()}"] = sha256_raw(f)
    return out


def verify_equivalence(cfg: dict) -> dict:
    w, w5 = (ast.parse(p.read_text(encoding="utf-8")) for p in (WORKER, EXP5_WORKER))
    m, mr = (ast.parse(p.read_text(encoding="utf-8")) for p in (MODAL_APP, REF_MODAL_APP))
    if rd._const(w, "BASE_ENGINE_KWARGS") != rd._const(w5, "BASE_ENGINE_KWARGS"):
        raise RuntimeError("Exp6 BASE_ENGINE_KWARGS differ from the frozen Experiment 5 worker")
    kw = next(ast.unparse(n.value) for n in ast.walk(_function(w, "main"))
              if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "kwargs")
    if kw != ("{'model': args.model_dir, **BASE_ENGINE_KWARGS, 'max_num_seqs': args.concurrency, "
              "'kv_cache_dtype': args.kv_cache_dtype}"):
        raise RuntimeError(f"engine kwargs must differ from the frozen kwargs only by max_num_seqs / kv dtype: {kw}")
    src = ast.unparse(_function(w, "main"))
    if "temperature=0.0" not in src or "wl.OUTPUT_TOKENS" not in src or "collective_rpc" in ast.unparse(w):
        raise RuntimeError("Exp6 worker must be greedy, use the frozen output length and no engine RPC")
    if ast.dump(_module_assign(m, "image")) != ast.dump(_module_assign(mr, "image")):
        raise RuntimeError("Exp6 Modal image differs from the canonical image")
    for fn in ("_emit", "_sha256_file", "_gpu_query", "_compute_apps", "_gpu_state", "_require_clean", "_run_guarded"):
        if ast.dump(_function(m, fn)) != ast.dump(_function(mr, fn)):
            raise RuntimeError(f"Modal helper {fn}() differs from the accepted smoke app")
    for const, mine in (("GATE_TIMEOUT_S", GATE_TIMEOUT_S), ("POINT_TIMEOUT_S", POINT_TIMEOUT_S),
                        ("POINTS_PER_SWEEP", POINTS_PER_SWEEP), ("GPU_CLEAN_TOLERANCE_MIB", 256),
                        ("EXPECTED_RABIT_SHA256_LF", rs6.a2.r5.EXPECTED_RABIT_SHA256_LF)):
        if rd._const(m, const) != mine:
            raise RuntimeError(f"Modal {const} differs")
    backstop = next(ast.literal_eval(k.value) for dec in _function(m, "sweep").decorator_list
                    for k in getattr(dec, "keywords", []) if k.arg == "timeout")
    if not (backstop == MODAL_FUNCTION_TIMEOUT_S > WATCHDOG_BUDGET_S):
        raise RuntimeError(f"Modal backstop {backstop} must exceed the watchdog budget {WATCHDOG_BUDGET_S}")
    rs6.verify_equivalence(cfg)  # accepted shared_decode / triton_attn / rabit_kv2 sources, smoke app unchanged
    return {"engine_kwargs_equal_frozen_exp5_except_max_num_seqs_and_kv_dtype": True, "greedy": True,
            "image_and_helpers_equal_accepted_smoke_app": True, "accepted_implementation_sources": True,
            "watchdog_budget_s": WATCHDOG_BUDGET_S, "modal_backstop_s": backstop}


def preflight(length: int, dry_run: bool) -> dict:
    assert_protected_paths_clean("preflight")
    cfg = rs6.final_config()
    protocol = load_protocol()
    eq = verify_equivalence(cfg)
    if not rs6.run_git("ls-files", rel(SMOKE_DIR / "smoke_analysis.json")):
        raise RuntimeError("the accepted concurrency smoke evidence must be committed")
    smoke = json.loads((SMOKE_DIR / "smoke_analysis.json").read_text(encoding="utf-8"))
    if not (smoke["all_integrity_passed"] and all(v["equal_count"] == 4 for v in smoke["per_dtype"].values())):
        raise RuntimeError("accepted smoke evidence is not a 4/4 pass for both dtypes")
    uncommitted = run_git("status", "--short", "--", *[rel(p) for p in MUST_BE_COMMITTED])
    if uncommitted and not dry_run:
        raise RuntimeError("Refusing to run: Exp6 harness has uncommitted changes:\n" + uncommitted)
    d = out_dir(length)
    if any(d == x or x in d.parents or d in x.parents for x in excluded_attempts(length)):
        raise RuntimeError(f"Refusing to run: {rel(d)} overlaps an archived diagnostic attempt")
    leftovers = sorted(p.name for p in d.iterdir()) if d.is_dir() else []
    if leftovers and not dry_run:
        raise RuntimeError(f"Refusing to run: {rel(d)} is not empty ({leftovers})")
    return {"git_head": run_git("rev-parse", "HEAD"), "vllm_kvquant_tree": run_git("rev-parse", "HEAD:vllm-kvquant"),
            "final_config": cfg, "equivalence": eq, "protocol_sha256": sha256(PROTOCOL),
            "prompt_set_sha256": protocol["prompt_sets"][str(length)]["measured"]["ordered_set_sha256"],
            "runner_script_sha256": sha256(RUNNER_SCRIPT), "modal_app_sha256": sha256(MODAL_APP),
            "worker_sha256": sha256(WORKER), "workload_sha256": sha256(WORKLOAD),
            "protected_paths": [rel(p) for p in PROTECTED_PATHS], "prior_evidence_sha256_raw": prior_evidence_digest(),
            "uncommitted_files": uncommitted or None, "existing_output_files": leftovers or None}


def points_arg(length: int) -> str:
    return ",".join(f"{p['label']}={p['dtype']}:{p['prompt_tokens']}:{p['concurrency']}:{p['trial']}"
                    for p in wl.plan_points(length))


def build_command(length: int, prompt_set_sha: str, cfg: dict, run_id: str) -> list[str]:
    return [sys.executable, "-m", "modal", "run", "--detach", str(MODAL_APP), "--points", points_arg(length),
            "--prompt-set-sha256", prompt_set_sha, "--stage3c-impl", cfg["impl"], "--query-block", str(cfg["query_block"]),
            "--run-id", run_id]


def make_run_id(length: int, git_head: str) -> str:
    return f"exp6-L{length}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{git_head[:7]}"


def modal_volume_get(remote: str, local: Path) -> bool:
    """Download one file from the session-log volume; False if it does not exist (yet)."""
    env = {**__import__("os").environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", SESSION_LOG_VOLUME, remote,
                        str(local)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=env)
    return r.returncode == 0 and local.is_file()


def fetch_remote_log(run_id: str, dest: Path, deadline_s: float, getter=modal_volume_get, sleep=time.sleep,
                     clock=time.monotonic, poll_s: float = REMOTE_POLL_S) -> dict:
    """Wait (up to deadline_s from now) for /<run_id>/DONE.json, then download remote_session.log.
    Without DONE.json by the deadline, the latest committed partial log (if any) is still downloaded."""
    t_end = clock() + deadline_s
    done_path, log_path = dest / "remote_DONE.json", dest / "remote_session.log"
    polls = 0
    while True:
        polls += 1
        if getter(f"/{run_id}/DONE.json", done_path):
            break
        if clock() >= t_end:
            done_path = None
            break
        sleep(poll_s)
    have_log = getter(f"/{run_id}/remote_session.log", log_path)
    done = json.loads(done_path.read_text(encoding="utf-8")) if done_path else None
    return {"run_id": run_id, "volume": SESSION_LOG_VOLUME, "polls": polls, "done": done,
            "done_found": done is not None, "remote_log_found": have_log,
            "remote_log_sha256": sha256_raw(log_path) if have_log else None,
            "remote_log_bytes": log_path.stat().st_size if have_log else None,
            "complete": bool(done and done.get("status") == "complete" and have_log)}


def expected_runtime(length: int) -> dict:
    """Planning estimate only, from Experiment 5 attempt 2 single-request medians: per point ~ engine start
    + 256 x TTFT (prefill does not parallelize) + 256 x 31 x TPOT / C (decode batches)."""
    s = json.loads((rs6.a2.ATTEMPT_DIR / "context_scaling_summary.json").read_text(encoding="utf-8"))
    total, per = 0.0, {}
    for dtype, letter in (("bfloat16", "A"), ("rabit_kv2", "B")):
        lat = s["cells"][f"{letter}{length}"]["latency"]
        ttft, tpot = lat["ttft_ms"]["median"] / 1000.0, lat["tpot_ms"]["median"] / 1000.0
        pts = {c: 40.0 + 2 * (ttft + 31 * tpot) + 256 * ttft + 256 * 31 * tpot / c for c in wl.CONCURRENCY_GRID}
        per[dtype] = {str(c): round(v) for c, v in pts.items()}
        total += wl.TRIALS * sum(pts.values())
    fixed = 600 + 70 + 120
    return {"per_point_s_per_trial": per, "sweep_points_s": round(total), "fixed_s": fixed,
            "total_estimate_s": round(total + fixed),
            "note": "planning estimate; decode batching assumed ideal, so large-C points may be slower"}


# ------------------------------------------------------------------ parsing / metrics (pure)
def demux(text: str, plan: list[dict]):
    prefixes = {f"[pt{k}:{p['label']}] ": p["label"] for k, p in enumerate(plan, start=1)}
    points, gate, top = {p["label"]: [] for p in plan}, [], []
    for line in text.splitlines():
        if line.startswith("[gate] "):
            gate.append(line[len("[gate] "):])
            continue
        for pre, label in prefixes.items():
            if line.startswith(pre):
                points[label].append(line[len(pre):])
                break
        else:
            top.append(line)
    return points, gate, top


def parse_point(lines: list[str]) -> dict:
    out = {"tags": {}, "requests": [], "shadow_requests": [], "markers": [], "jit": dict.fromkeys(JIT_PHASES, 0),
           "oom_lines": 0, "malformed": [], "logged_preemptions_measured": 0, "logged_preemption_lines": 0,
           "gpu_memory": {}, "failure": None}
    phase = "setup"
    for line in lines:
        s = line.strip()
        if s in MARKERS:
            out["markers"].append(s)
            phase = {"EXP6_SHADOW_CONDITIONING_BEGIN": "shadow_conditioning",
                     "EXP6_SHADOW_CONDITIONING_END": "setup_after_shadow",
                     "EXP6_WARMUP_BEGIN": "original_warmup", "EXP6_WARMUP_END": "setup_after_warmup",
                     "EXP6_MEASURED_BEGIN": "measured", "EXP6_MEASURED_END": "after",
                     "EXP6_WORKER_COMPLETE": "after"}[s]
            continue
        if s.startswith("EXP6_") and "={" in s:
            m = TAG.match(s)
            if not m:
                out["malformed"].append(s[:200])
                continue
            tag, p = m.group(1), json.loads(m.group(2))
            if tag == "EXP6_REQUEST":
                out["requests"].append(p)
            elif tag == "EXP6_SHADOW_CONDITIONING_REQUEST":
                out["shadow_requests"].append(p)
            elif tag == "EXP6_GPU_MEMORY":
                out["gpu_memory"][p["phase"]] = p["memory_used_mib"]
            elif tag == "EXP6_REQUEST_FAILURE":
                out["failure"] = p
            else:
                out["tags"][tag] = p
            continue
        if JIT in s:
            key = phase if phase in JIT_PHASES else "setup"
            out["jit"][key] += 1
        if any(o in s for o in OOM):
            out["oom_lines"] += 1
        mp = LOGGED_PREEMPTIONS.search(s)
        if mp and phase == "measured":
            out["logged_preemptions_measured"] += int(mp.group(1))
            out["logged_preemption_lines"] += 1
    return out


def inflight_concurrency(rows: list[dict], target: int) -> dict:
    """Overlapping in-flight concurrency: sweep-line over [scheduled_ts, last_token_ts] (engine-core clock).
    See CONCURRENCY_TERMINOLOGY -- not GPU residency."""
    iv = [(r.get("scheduled_ts"), r.get("last_token_ts")) for r in rows]
    if not iv or any(not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and 0 < a <= b) for a, b in iv):
        return {"evaluable": False}
    events = sorted([(a, 1) for a, _ in iv] + [(b, -1) for _, b in iv], key=lambda e: (e[0], e[1]))
    inflight, peak, t_prev, at_level, all_c, longest, run = 0, 0, events[0][0], {}, 0.0, 0.0, 0.0
    for t, d in events:
        dt = t - t_prev
        if dt > 0:
            at_level[inflight] = at_level.get(inflight, 0.0) + dt
            if inflight >= target:
                all_c += dt
                run += dt
                longest = max(longest, run)
            else:
                run = 0.0
        inflight += d
        peak = max(peak, inflight)
        t_prev = t
    span = events[-1][0] - events[0][0]
    return {"evaluable": True, "kind": "overlapping in-flight concurrency (not GPU residency)",
            "observed_max_inflight_concurrency": peak, "all_c_inflight_overlap_total_s": all_c,
            "all_c_inflight_overlap_longest_s": longest, "measured_span_s": span,
            "time_fraction_by_inflight_count": {str(k): v / span for k, v in sorted(at_level.items())} if span else {}}


def pct(values: list[float], q: int) -> float | None:
    if len(values) < 2:
        return values[0] if values else None
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1]


def point_metrics(p: dict, spec: dict, pinned_hashes: list[str]) -> dict:
    rows = sorted(p["requests"], key=lambda r: r["i"])
    L, C = spec["prompt_tokens"], spec["concurrency"]
    valid = [r for r in rows if r["output_tokens"] == wl.OUTPUT_TOKENS and r["finish_reason"] == "length"
             and r["prompt_tokens"] == L and r["i"] < len(pinned_hashes)
             and r["prompt_token_ids_sha256"] == pinned_hashes[r["i"]]]
    ms = p["tags"].get("EXP6_MEASURED_SUMMARY") or {}
    wall = ms.get("wall_s")
    lat = [r["last_token_ts"] - r["queued_ts"] for r in valid if r.get("queued_ts") and r.get("last_token_ts")]
    ttft = [r["first_token_ts"] - r["queued_ts"] for r in valid if r.get("queued_ts") and r.get("first_token_ts")]
    tpot = [(r["last_token_ts"] - r["first_token_ts"]) / (wl.OUTPUT_TOKENS - 1) for r in valid
            if r.get("first_token_ts") and r.get("last_token_ts")]
    conc = inflight_concurrency(valid, C) if len(valid) == len(rows) and rows else {"evaluable": False}
    if ms.get("preemption_counter_available"):
        preempt, source = ms["preemptions_after"] - ms["preemptions_before"], "vllm:num_preemptions counter delta"
    elif p["logged_preemption_lines"]:
        preempt, source = p["logged_preemptions_measured"], "logged 'Preemptions' lines (lower bound)"
    else:
        preempt, source = None, "unavailable"
    ok_rate = wall and wall > 0 and len(valid) == wl.MEASURED_REQUESTS
    return {"completed_requests": len(valid), "returned_requests": len(rows),
            "failed_requests": wl.MEASURED_REQUESTS - len(valid),
            "output_token_count_valid": bool(rows) and all(r["output_tokens"] == wl.OUTPUT_TOKENS for r in rows),
            "wall_s": wall, "requests_per_s": len(valid) / wall if ok_rate else None,
            "output_tokens_per_s": len(valid) * wl.OUTPUT_TOKENS / wall if ok_rate else None,
            "total_tokens_per_s": len(valid) * (L + wl.OUTPUT_TOKENS) / wall if ok_rate else None,
            "latency_s": {"median": statistics.median(lat) if lat else None, "p90": pct(lat, 90), "p99": pct(lat, 99),
                          "definition": "last_token_ts - queued_ts (engine core; includes queueing under saturation)"},
            "ttft_s": {"median": statistics.median(ttft) if ttft else None, "p90": pct(ttft, 90),
                       "definition": "first_token_ts - queued_ts"},
            "tpot_s": {"median": statistics.median(tpot) if tpot else None, "p90": pct(tpot, 90)},
            "target_concurrency": C, "concurrency": conc, "preemptions": preempt, "preemption_source": source,
            "overlap_stats_are_not_residency_evidence": preempt is None or preempt > 0}


def shadow_validity(p: dict, spec: dict, pinned_hashes: list[str]) -> dict:
    """Validity of the unmeasured 256-request shadow-conditioning pass (never used for performance)."""
    rows = sorted(p["shadow_requests"], key=lambda r: r["i"])
    N = wl.SHADOW_CONDITIONING_REQUESTS
    L, C = spec["prompt_tokens"], spec["concurrency"]
    exact = [r for r in rows if r["prompt_tokens"] == L and r["i"] < len(pinned_hashes)
             and r["prompt_token_ids_sha256"] == pinned_hashes[r["i"]]
             and r["output_tokens"] == wl.OUTPUT_TOKENS and r["finish_reason"] == "length"]
    conc = inflight_concurrency(rows, C) if rows else {"evaluable": False}
    ran = "EXP6_SHADOW_CONDITIONING_END" in p["markers"]
    failed_here = (p["failure"] or {}).get("phase") == "shadow_conditioning"
    completed = ran and not failed_here and len(rows) == len(exact) == N == len(pinned_hashes)
    reached = (bool(conc.get("evaluable")) and conc["observed_max_inflight_concurrency"] >= C
               and conc["all_c_inflight_overlap_total_s"] > 0)
    return {"requests": len(rows), "exact_requests": len(exact), "expected_requests": N, "target_concurrency": C,
            "completed_all_exact": completed,
            "observed_max_inflight_concurrency": conc.get("observed_max_inflight_concurrency"),
            "all_c_inflight_overlap_total_s": conc.get("all_c_inflight_overlap_total_s"),
            "target_inflight_concurrency_reached": reached, "failure_in_conditioning": failed_here,
            "jit_lines": p["jit"]["shadow_conditioning"], "valid": completed and reached and not failed_here}


def classify(p: dict, proc: dict | None, metrics: dict) -> str:
    started = "EXP6_CAPACITY" in p["tags"]
    oom = p["oom_lines"] > 0 or (p["failure"] or {}).get("kind") == "request_oom"
    if oom:
        return "oom_or_allocation_failure"
    if (not started or (proc or {}).get("timed_out") or (proc or {}).get("returncode") != 0 or p["failure"]
            or metrics["completed_requests"] != wl.MEASURED_REQUESTS or "EXP6_WORKER_COMPLETE" not in p["markers"]):
        return "engine_or_request_failure"
    if metrics["preemptions"] is not None and metrics["preemptions"] > 0:
        return "completed_with_preemption"
    c = metrics["concurrency"]
    if not c.get("evaluable") or c["observed_max_inflight_concurrency"] < metrics["target_concurrency"] \
            or c["all_c_inflight_overlap_total_s"] <= 0:
        return "target_concurrency_not_reached"
    return "sustained_target_concurrency"


def highest_successful(results: dict) -> dict:
    """Per dtype: highest C that is sustained_target_concurrency (and JIT-clean) in EVERY trial; also per trial."""
    out = {}
    for dtype in ("bfloat16", "rabit_kv2"):
        per_trial = {}
        for t in range(1, wl.TRIALS + 1):
            ok = [c for c in wl.CONCURRENCY_GRID if (results.get((dtype, c, t)) or {}).get("interpretable_success")]
            per_trial[str(t)] = max(ok) if ok else None
        all_ok = [c for c in wl.CONCURRENCY_GRID
                  if all((results.get((dtype, c, t)) or {}).get("interpretable_success") for t in range(1, wl.TRIALS + 1))]
        top = max(all_ok) if all_ok else None
        contiguous = None
        for c in wl.CONCURRENCY_GRID:  # highest C with EVERY lower grid point also successful in every trial
            if c not in all_ok:
                break
            contiguous = c
        bracketed = contiguous is not None and contiguous != max(wl.CONCURRENCY_GRID)
        out[dtype] = {"highest_successfully_tested_concurrency_all_trials": top,
                      "highest_contiguous_successful_concurrency_all_trials": contiguous,
                      "non_monotonic": top != contiguous, "per_trial": per_trial,
                      "failure_boundary_bracketed_by_grid": bracketed}
    return out


def extension_eligibility(results: dict) -> dict:
    """Pre-registered rule, reported only: BF16 fails first. Extension is NOT enabled in this harness."""
    rabit_all = lambda c: all((results.get(("rabit_kv2", c, t)) or {}).get("interpretable_success")  # noqa: E731
                              for t in range(1, wl.TRIALS + 1))
    bf16_fails = [c for c in wl.CONCURRENCY_GRID if rabit_all(c) and not all(
        (results.get(("bfloat16", c, t)) or {}).get("interpretable_success") for t in range(1, wl.TRIALS + 1))]
    eligible = bool(bf16_fails) and rabit_all(max(wl.CONCURRENCY_GRID))
    return {"bf16_fails_first_at": bf16_fails, "rabit_sustains_64_all_trials": rabit_all(max(wl.CONCURRENCY_GRID)),
            "eligible_for_pre_registered_rabit_only_extension": eligible,
            "extension_grid": list(wl.RABIT_EXTENSION_GRID), "enabled": False,
            "note": "requires review before any 128 / 256 point"}


def parse_top(lines: list[str]) -> dict:
    out = {"pre": {}, "proc": {}, "start": {}, "exit": {}, "timeouts": [], "stopped": None, "complete": None}
    for line in lines:
        m = TOP_TAG.match(line.strip())
        if not m:
            continue
        tag, p = m.group(1), json.loads(m.group(2))
        if tag == "S3C_PRE_LEG_GPU_STATE":
            out["pre"][p["leg"]] = p
        elif tag == "S3C_PROCESS_EXIT":
            out["proc"][p["label"]] = p
        elif tag == "S3C_SERIES_START":
            out["start"][p["series"]] = p
        elif tag == "S3C_SERIES_EXIT":
            out["exit"][p["series"]] = p
        elif tag == "S3C_WATCHDOG_TIMEOUT":
            out["timeouts"].append(p)
        elif tag == "S3C_STOPPED":
            out["stopped"] = p
        elif tag == "S3C_SWEEP_COMPLETE":
            out["complete"] = p
        else:
            out[tag] = p
    return out


def analyze(text: str, length: int, cfg: dict, protocol: dict) -> tuple[dict, dict]:
    plan = wl.plan_points(length)
    pts_lines, gate_lines, top_lines = demux(text, plan)
    top = parse_top(top_lines)
    gate = rs6.a2.r5.parse_gate(gate_lines)
    pinned = protocol["prompt_sets"][str(length)]["measured"]["per_prompt_sha256"]
    shadow_set = protocol["shadow_conditioning"]["prompt_sets"][str(length)]
    checks = []

    def add(name, cat, state, observed=None):
        if isinstance(state, bool):
            state = PASSED if state else FAILED
        checks.append({"check": name, "category": cat, "state": state, "observed": observed})

    env = top.get("S3C_ENVIRONMENT", {})
    add("exactly one H100", "environment",
        (len(env.get("gpus", [])) == 1 and "H100" in env["gpus"][0].get("name", "")) if env else NOT_EVALUATED)
    add("point order as pre-registered (36 points)", "environment",
        env.get("point_labels") == [p["label"] for p in plan] if env else NOT_EVALUATED)
    add("gate passed", "gate", (top.get("S3C_GATE_EXIT", {}).get("returncode") == 0
                                and (gate.get("result") or {}).get("passed") is True) if "S3C_GATE_START" in top else NOT_RUN)
    add("sweep completed all 36 points (no stop)", "completion",
        bool(top["complete"]) and top["complete"].get("points") == POINTS_PER_SWEEP and not top["stopped"],
        top["stopped"] or None)
    base = top.get("S3C_GPU_BASELINE", {})
    results, points_out = {}, []
    for spec in plan:
        label, dtype, C, t = spec["label"], spec["dtype"], spec["concurrency"], spec["trial"]
        p = parse_point(pts_lines[label])
        tg = p["tags"]
        started = label in top["start"]
        pre = top["pre"].get(label)
        add(f"{label}: GPU clean before point", "gpu_clean", rs6.a2.r5.gpu_leg_clean(pre, base) if pre else NOT_RUN)
        m = point_metrics(p, spec, pinned)
        cv = shadow_validity(p, spec, shadow_set["per_prompt_sha256"])
        cls = classify(p, top["proc"].get(label), m) if started else None
        st = lambda ok: ok if started else NOT_RUN  # noqa: E731
        add(f"{label}: point identity", "point", st(tg.get("EXP6_POINT") == {
            "label": label, "kv_cache_dtype": dtype, "trial": t, "prompt_tokens": spec["prompt_tokens"],
            "target_concurrency": C}), tg.get("EXP6_POINT"))
        s3 = tg.get("EXP6_STAGE_IMPL") or {}
        if dtype == "rabit_kv2":
            ok = (s3.get("applicable") is True and s3.get("requested_impl") == s3.get("effective_impl") == cfg["impl"]
                  and s3.get("requested_query_block") == s3.get("effective_query_block") == cfg["query_block"]
                  and s3.get("env") == {"VLLM_RABIT2_STAGE3C_IMPL": cfg["impl"],
                                        "VLLM_RABIT2_SHARED_DECODE_QUERY_BLOCK": str(cfg["query_block"])}
                  and s3.get("shared_decode_module_sha256") == cfg["accepted_shared_decode_module_sha256"])
        else:
            ok = s3.get("applicable") is False and not any((s3.get("env") or {}).values())
        ok = ok and all(v in (None, "0") for v in (s3.get("profiling_env") or {"x": "missing"}).values())
        add(f"{label}: Stage3C selection ({'shared_decode / QB' if dtype == 'rabit_kv2' else 'none'}) and profiling off",
            "stage3c", st(ok), s3 or None)
        engine_up = "EXP6_CAPACITY" in tg
        eff = tg.get("EXP6_EFFECTIVE_ENGINE_CONFIG", {})
        add(f"{label}: max_num_seqs == target concurrency {C} (requested and effective)", "config",
            st((tg.get("EXP6_REQUESTED_ENGINE_KWARGS") or {}).get("max_num_seqs") == C
               and (eff.get("max_num_seqs") == C if engine_up else True)))
        wkl = tg.get("EXP6_WORKLOAD") or {}
        add(f"{label}: frozen prompt set (hash) and 32 greedy outputs", "workload",
            st(not engine_up or (wkl.get("prompt_set", {}).get("ordered_set_sha256")
                                 == protocol["prompt_sets"][str(spec["prompt_tokens"])]["measured"]["ordered_set_sha256"]
                                 and wkl.get("output_tokens") == wl.OUTPUT_TOKENS and wkl.get("temperature") == 0.0
                                 and wkl.get("measured_requests") == wl.MEASURED_REQUESTS
                                 and wkl.get("warmup_requests") == wl.WARMUP_REQUESTS
                                 and wkl.get("shadow_conditioning_requests") == wl.SHADOW_CONDITIONING_REQUESTS
                                 and (wkl.get("shadow_conditioning_prompt_set") or {}).get("ordered_set_sha256")
                                 == shadow_set["ordered_set_sha256"])))
        cap = (tg.get("EXP6_CAPACITY") or {}).get("capacity_tokens")
        add(f"{label}: allocator capacity == expected {EXPECTED_CAPACITY[dtype]} tokens", "capacity",
            st(cap == EXPECTED_CAPACITY[dtype]) if engine_up else NOT_EVALUATED, cap)
        add(f"{label}: shadow conditioning completed 256/256 exact requests (prompt length, pinned hash, 32 "
            f"outputs)", "conditioning", st(cv["completed_all_exact"]) if engine_up else NOT_EVALUATED,
            {k: cv[k] for k in ("requests", "exact_requests", "failure_in_conditioning")})
        add(f"{label}: shadow conditioning reached target overlapping in-flight concurrency {C}", "conditioning",
            st(cv["target_inflight_concurrency_reached"]) if engine_up else NOT_EVALUATED,
            cv["observed_max_inflight_concurrency"])
        add(f"{label}: no Triton JIT during the measured phase", "jit",
            st(p["jit"]["measured"] == 0) if engine_up else NOT_EVALUATED, p["jit"])
        add(f"{label}: no malformed machine lines", "point", st(not p["malformed"]), p["malformed"] or None)
        add(f"{label}: outcome classified", "outcome", st(cls in CLASSES), cls)
        interpretable = cls == "sustained_target_concurrency" and p["jit"]["measured"] == 0 and cv["valid"]
        results[(dtype, C, t)] = {"class": cls, "interpretable_success": interpretable}
        points_out.append({**spec, "outcome_class": cls, "measured_jit_lines": p["jit"]["measured"],
                           "interpretable": p["jit"]["measured"] == 0 and cls is not None and cv["valid"],
                           "shadow_conditioning": cv,
                           "jit_lines_by_phase": p["jit"], "capacity": tg.get("EXP6_CAPACITY"),
                           "gpu_memory_mib": p["gpu_memory"], "oom_lines": p["oom_lines"], "failure": p["failure"],
                           "process": top["proc"].get(label), **m})
    post = top["pre"].get("post_run")
    add("GPU clean after the run", "gpu_clean", rs6.a2.r5.gpu_leg_clean(post, base) if post else NOT_RUN)
    counts = {s: sum(1 for c in checks if c["state"] == s) for s in (PASSED, FAILED, NOT_RUN, NOT_EVALUATED)}
    integ = {"checks": checks, "counts": counts, "all_ok": counts[PASSED] == len(checks),
             "failed_categories": sorted({c["category"] for c in checks if c["state"] == FAILED})}
    summary = {"scope": SCOPE, "amendments": AMENDMENTS, "prompt_tokens": length, "final_rabit_configuration": cfg,
               "all_integrity_passed": integ["all_ok"], "integrity_counts": counts, "points": points_out,
               "outcome_class_counts": {c: sum(1 for p in points_out if p["outcome_class"] == c) for c in CLASSES},
               "highest_successfully_tested_concurrency": highest_successful(results),
               "rabit_only_extension": extension_eligibility(results),
               "excluded_attempts_not_pooled": [rel(d) for d in excluded_attempts(length)],
               "labels": {"concurrency": CONCURRENCY_TERMINOLOGY,
                          "gpu_memory": "device-wide nvidia-smi memory.used; NOT request KV memory",
                          "capacity": "MEASURED allocator capacity (num_gpu_blocks x block_size)"}}
    return integ, summary


def main(argv: list[str] | None = None) -> int:
    make_console_encoding_safe()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt-tokens", type=int, choices=wl.PROMPT_LENGTHS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-protocol", action="store_true", help="write exp6_protocol.json (pre-commit only)")
    ap.add_argument("--rabit-extension", action="store_true", help="NOT enabled until reviewed")
    a = ap.parse_args(argv)
    if a.rabit_extension:
        raise SystemExit("RABIT-only extension points are pre-registered but NOT enabled until review")
    if a.write_protocol:
        if PROTOCOL.exists():
            raise SystemExit(f"{rel(PROTOCOL)} already exists; the frozen protocol is never overwritten")
        PROTOCOL.write_text(json.dumps(build_protocol(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {rel(PROTOCOL)}")
        return 0
    if a.prompt_tokens is None:
        raise SystemExit("--prompt-tokens {2048,8192} is required (prompt-length sweeps run separately)")
    L = a.prompt_tokens
    print(f"RABIT-KV MLSys 2027 -- Experiment 6 concurrency scaling, prompt length {L} (one Modal run)")
    prov = preflight(L, a.dry_run)
    cfg = prov["final_config"]
    print("Preflight OK:", json.dumps({k: prov[k] for k in ("git_head", "vllm_kvquant_tree", "protocol_sha256",
                                                             "prompt_set_sha256")}))
    print("  equivalence:", json.dumps(prov["equivalence"]))
    print("  final RABIT:", json.dumps({k: cfg[k] for k in ("impl", "query_block")}))
    print("  expected runtime:", json.dumps(expected_runtime(L)))
    if prov["uncommitted_files"]:
        print("  WARNING (dry-run only): uncommitted:\n    " + prov["uncommitted_files"].replace("\n", "\n    "))
    print("Order:", [p["label"] for p in wl.plan_points(L)])
    run_id = make_run_id(L, prov["git_head"])
    print("Local command:\n  " + " ".join(build_command(L, prov["prompt_set_sha256"], cfg, run_id))[:600] + " ...")
    if a.dry_run:
        print("\n--dry-run: nothing executed, no files written.")
        return 0
    d = out_dir(L)
    d.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment": "Experiment 6 concurrency scaling", "prompt_tokens": L, "scope": SCOPE,
                "amendments": AMENDMENTS, "attempt": ATTEMPT_DIR_NAME,
                "excluded_attempts_not_pooled": [rel(x) for x in excluded_attempts(L)], "run_id": run_id,
                "launch": "modal run --detach", "started_utc": now(), "status": "running", "provenance": prov}
    mpath = d / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    snap = Path(tempfile.mkdtemp(prefix="exp6_vllm_snapshot_")) / "vllm_kvquant_snapshot.zip"
    run_git("-c", "core.autocrlf=false", "archive", "--format=zip", "-o", str(snap), "HEAD:vllm-kvquant")
    manifest["vllm_kvquant_snapshot"] = {"sha256": sha256_raw(snap), "bytes": snap.stat().st_size}
    cmd = build_command(L, prov["prompt_set_sha256"], cfg, run_id)
    t_launch = time.monotonic()
    code = stream_command(cmd, d / "modal_session.log", {"EXP6_VLLM_SNAPSHOT": str(snap)})
    manifest["modal_returncode"] = code  # local client only; with --detach the remote sweep may outlive it
    mpath.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    remaining = max(0.0, MODAL_FUNCTION_TIMEOUT_S + 1800 - (time.monotonic() - t_launch))
    print(f"Waiting for the remote session log of {run_id} (up to {round(remaining)} s) ...", flush=True)
    remote = fetch_remote_log(run_id, d, remaining)
    manifest["remote_session_log"] = remote
    source = d / ("remote_session.log" if remote["remote_log_found"] else "modal_session.log")
    manifest["analyzed_log"] = source.name
    text = source.read_text(encoding="utf-8", errors="replace")
    integ, summary = analyze(text, L, cfg, load_protocol())
    pts, gate_lines, _ = demux(text, wl.plan_points(L))
    (d / "correctness_gate.log").write_text("\n".join(gate_lines) + "\n", encoding="utf-8")
    for p in wl.plan_points(L):
        name = f"{'bf16' if p['dtype'] == 'bfloat16' else 'rabit_kv2'}_conc{p['concurrency']}_t{p['trial']}.log"
        (d / name).write_text("\n".join(pts[p["label"]]) + "\n", encoding="utf-8")
    (d / "integrity_check.json").write_text(json.dumps(integ, indent=2, default=str) + "\n", encoding="utf-8")
    (d / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    manifest.update(status="completed" if (remote["complete"] and integ["all_ok"]) else "failed",
                    completed_utc=now(), integrity_counts=integ["counts"])
    try:
        assert_protected_paths_clean("post-run")
        manifest["protected_paths_post_run_status"] = "clean"
    except RuntimeError as e:
        manifest.update(protected_paths_post_run_status="check_failed", status="failed", protected_paths_error=str(e))
    manifest["prior_evidence_unchanged"] = prior_evidence_digest() == prov["prior_evidence_sha256_raw"]
    if not manifest["prior_evidence_unchanged"]:
        manifest["status"] = "failed"
    mpath.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nEXP6 L{L} {manifest['status'].upper()}: integrity {integ['counts']}; outcomes "
          f"{summary['outcome_class_counts']}")
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
