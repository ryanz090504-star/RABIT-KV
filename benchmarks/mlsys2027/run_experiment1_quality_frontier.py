"""
RABIT-KV MLSys 2027 -- Experiment 1 runner (P0-A, quality-compression frontier).

Orchestrates the five already-existing quality benchmarks
(continuation_ppl, niah, passage_retrieval, hotpotqa, qasper) across the
bf16/rabit8/rabit4/rabit3/rabit2 method frontier that each script already
supports natively. See docs/MLSYS_EXPERIMENT_PLAN.md, Experiment 1.

This script:
  * does NOT modify any of the five quality scripts;
  * does NOT modify vllm-kvquant/vllm/v1/attention/ops/rabit_kv2.py;
  * does NOT write to results/quality/, results/performance/, or
    results/summary.json;
  * writes ONLY under results/mlsys2027/quality_frontier/.

All KV-memory numbers produced by the underlying scripts are LOGICAL
(HF-side one-shot quantize/dequantize) packed-prefix-KV accounting. They
are NOT physical vLLM allocator capacity. See
docs/MLSYS_EXPERIMENT_PLAN.md Ground Rule 3.

Usage:
    python benchmarks/mlsys2027/run_experiment1_quality_frontier.py --dry-run
        Runs every static/preflight check and prints the exact commands
        this script would execute, without invoking Modal or touching the
        GPU. Writes nothing to disk.

    python benchmarks/mlsys2027/run_experiment1_quality_frontier.py
        Executes the full experiment (Modal H100 runs). Not to be used
        until explicitly authorized.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER_SCRIPT = Path(__file__).resolve()
QUALITY_DIR = ROOT / "benchmarks" / "quality"
OUT_DIR = ROOT / "results" / "mlsys2027" / "quality_frontier"
MANIFEST = OUT_DIR / "manifest.json"
REGRESSION_CHECK = OUT_DIR / "regression_check.json"

RABIT_KV2 = ROOT / "vllm-kvquant" / "vllm" / "v1" / "attention" / "ops" / "rabit_kv2.py"
# Mirrors benchmarks/quality/run_suite.py's EXPECTED_RABIT_SHA256. The five
# quality scripts below do not import rabit_kv2.py -- they are self-contained
# HF/transformers fake-quant scripts running on their own Modal image -- so
# this is a project-wide provenance invariant (confirms the repo is in the
# expected frozen state), not a functional dependency of this experiment.
EXPECTED_RABIT_SHA256 = (
    "108ba8afaac862bc4e070d2dc812493aaeae4e458837c4bda2ac703e72a57524"
)

# Paths this experiment must never modify. Checked via `git status --short`
# before and after every run; any change here aborts/fails loudly.
PROTECTED_PATHS = [
    ROOT / "results" / "quality",
    ROOT / "results" / "performance",
    ROOT / "results" / "summary.json",
    ROOT / "vllm-kvquant",
]

METHODS = "bf16,rabit8,rabit4,rabit3,rabit2"
REQUIRED_ALLOWED_LINE = 'allowed = {"bf16", "rabit8", "rabit4", "rabit3", "rabit2"}'
REQUIRED_RABIT2_MARKER = '"name": "2b META8g64 K3V2 G32 R4"'

# Canonical args mirror benchmarks/quality/run_suite.py's RUNS list exactly
# (sample counts, context lengths, length buckets), with --methods extended
# from "bf16,rabit2" to the full frontier already supported by each script.
# Per-sample/example seeding is hardcoded as seed(0) inside each script and
# is therefore preserved automatically -- this runner never touches it.
RUNS = [
    {
        "name": "continuation_ppl",
        "script": "continuation_ppl.py",
        "args": [
            "--context-tokens", "1024",
            "--eval-tokens", "128",
            "--samples", "8",
            "--methods", METHODS,
        ],
        "purpose": (
            "WikiText-2 teacher-forced continuation PPL, "
            "bf16/rabit8/rabit4/rabit3/rabit2 frontier"
        ),
    },
    {
        "name": "niah",
        "script": "niah.py",
        "args": [
            "--context-lengths", "4096,8192,16384",
            "--needle-depths", "0.1,0.25,0.5,0.75,0.9",
            "--max-new-tokens", "16",
            "--methods", METHODS,
        ],
        "purpose": "Needle-in-a-Haystack: 4K/8K/16K x five depths, full frontier",
    },
    {
        "name": "passage_retrieval",
        "script": "passage_retrieval.py",
        "args": [
            "--sample-start", "0",
            "--samples", "10",
            "--max-input-tokens", "16384",
            "--max-new-tokens", "32",
            "--methods", METHODS,
        ],
        "purpose": "LongBench passage_retrieval_en, full 10-sample slice, full frontier",
    },
    {
        "name": "hotpotqa",
        "script": "hotpotqa.py",
        "args": [
            "--sample-start", "0",
            "--samples", "20",
            "--length-bucket", "8k+",
            "--max-input-tokens", "16384",
            "--max-new-tokens", "32",
            "--methods", METHODS,
        ],
        "purpose": "LongBench-E HotpotQA 8K+, full 20-sample slice, full frontier",
    },
    {
        "name": "qasper",
        "script": "qasper.py",
        "args": [
            "--sample-start", "0",
            "--samples", "24",
            "--length-bucket", "8k+",
            "--max-input-tokens", "16384",
            "--max-new-tokens", "32",
            "--methods", METHODS,
        ],
        "purpose": "LongBench-E Qasper 8K+, full 24-example bucket, full frontier",
    },
]

# ---------------------------------------------------------------------------
# Canonical bf16/rabit2 reference values, for regression checking ONLY.
#
# Sourced read-only from results/summary.json (frozen, Ground Rule 1) and,
# for the one figure summary.json omits (NIAH's bf16 avg KV MB), from the
# raw committed results/quality/niah.log. These are pinned constants, not
# derived by a live parse of those files -- verify_canonical_reference()
# below cross-checks them against the frozen files at preflight time so any
# future drift in the canonical files (which should never happen) is caught
# loudly instead of silently producing a wrong comparison.
# ---------------------------------------------------------------------------
CANONICAL_REFERENCE = {
    "continuation_ppl": {
        "bf16": {"ppl": 8.5020, "avg_kv_mb": 128.000},
        "rabit2": {"ppl": 8.6317, "avg_kv_mb": 24.710},
    },
    "niah": {
        "bf16": {"accuracy_pct": 100.0, "avg_kv_mb": 1194.542},
        "rabit2": {"accuracy_pct": 100.0, "avg_kv_mb": 226.786},
    },
    "passage_retrieval": {
        "bf16": {"accuracy_pct": 100.0, "avg_kv_mb": 1517.375},
        "rabit2": {"accuracy_pct": 100.0, "avg_kv_mb": 288.090},
    },
    "hotpotqa": {
        "bf16": {"f1_pct": 60.6, "avg_kv_mb": 1896.106},
        "rabit2": {"f1_pct": 55.2, "avg_kv_mb": 359.850},
    },
    "qasper": {
        "bf16": {"f1_pct": 35.9, "avg_kv_mb": 1783.531},
        "rabit2": {"f1_pct": 35.6, "avg_kv_mb": 338.428},
    },
}

# ---------------------------------------------------------------------------
# Regression tolerances (bf16/rabit2 only -- rabit8/rabit4/rabit3 have no
# canonical baseline to compare against; they are new evidence, recorded
# but not regression-checked).
#
# Exact bit-for-bit reproduction is NOT required and is not a meaningful
# bar here: these scripts run HF `transformers` forward passes on a freshly
# provisioned Modal H100 instance, with TF32 matmul enabled
# (`torch.backends.cuda.matmul.allow_tf32 = True`) and no call to
# `torch.use_deterministic_algorithms(True)`. cuBLAS/cuDNN reduction order
# and TF32 rounding are not guaranteed bit-identical across GPU instances or
# driver/library minor versions, even with a fixed seed(0). The canonical
# committed results were themselves produced by a single, non-repeated run
# (per the repository audit), so no empirical run-to-run variance estimate
# exists to calibrate against -- these tolerances are deliberately
# conservative and reasoned from first principles, not fit to observed
# noise:
#
# - PPL (continuation_ppl): 0.5% relative tolerance. Roughly an order of
#   magnitude tighter than the canonical rabit2 quality effect itself
#   (+1.53%), so it will not mask a genuine regression while comfortably
#   absorbing expected GPU floating-point noise in a 1024-token
#   cross-entropy aggregate.
# - Accuracy / F1 percentages (niah, passage_retrieval, hotpotqa, qasper):
#   1.0 percentage-point absolute tolerance. This is a TIGHT DRIFT DETECTOR,
#   not headroom meant to absorb a flipped example -- a single discrete
#   correct/incorrect flip is much larger than 1.0 point at these sample
#   counts: passage_retrieval has 10 examples, so one flip moves the
#   aggregate by 100/10 = 10.0 points; NIAH has 15 cases, so one flip moves
#   it by 100/15 = 6.67 points. Both exceed the 1.0-point tolerance by a
#   wide margin. That is intentional: these benchmarks depend on
#   greedy-decoded text, and floating-point nondeterminism could in rare
#   cases flip a token near a decision boundary and change one example's
#   result. If that happens, the aggregate will land far outside tolerance,
#   `check_regression()` will report `all_within_tolerance=False`, and this
#   runner stops immediately for investigation (see the enforcement in
#   `main()`) rather than silently accepting a result that may reflect a
#   real example-level change, not noise. 1.0 point only absorbs the kind
#   of sub-point aggregation/print-rounding noise that does NOT correspond
#   to any example changing correctness.
# - Avg KV MB (logical fake-quant memory accounting): 0.1% relative
#   tolerance. These numbers are deterministic byte-accounting derived from
#   fixed tensor shapes, bit-widths, and group sizes -- not from GPU kernel
#   execution -- so they should reproduce almost exactly every run. The
#   small allowance only covers the scripts' own `.3f` print rounding, not
#   measurement noise; any larger deviation indicates a real discrepancy
#   (e.g. different context tokenization) and must be treated as a hard
#   failure, not smoothed over.
# ---------------------------------------------------------------------------
PPL_RELATIVE_TOLERANCE = 0.005
PERCENTAGE_ABSOLUTE_TOLERANCE = 1.0
KV_MB_RELATIVE_TOLERANCE = 0.001


def run_git(*args: str) -> str:
    p = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{p.stdout}\n{p.stderr}")
    return p.stdout.strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_status() -> str:
    rel = [str(p.relative_to(ROOT)) for p in PROTECTED_PATHS]
    return run_git("status", "--short", "--", *rel)


def assert_protected_paths_clean(context: str) -> None:
    """Raise immediately if any protected canonical path (results/quality/,
    results/performance/, results/summary.json, vllm-kvquant/) shows an
    uncommitted change. Called before the experiment, after EVERY benchmark
    run (not just at the start/end), and after the full loop -- so a
    violation is caught at the earliest possible point, not only once
    everything has already finished."""
    status = protected_status()
    if status:
        raise RuntimeError(
            f"CRITICAL: protected canonical paths changed ({context}). "
            "This must never happen. Investigate immediately:\n" + status
        )


def verify_canonical_reference() -> None:
    """Fail loudly if CANONICAL_REFERENCE has drifted from what is actually
    committed in results/summary.json / results/quality/niah.log. Read-only;
    never writes to either file."""
    summary_path = ROOT / "results" / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    q = json.loads(summary_path.read_text(encoding="utf-8"))["final_quality"]

    pairs = [
        (CANONICAL_REFERENCE["continuation_ppl"]["bf16"]["ppl"],
         q["continuation_ppl_wikitext2"]["bf16_ppl"]),
        (CANONICAL_REFERENCE["continuation_ppl"]["rabit2"]["ppl"],
         q["continuation_ppl_wikitext2"]["rabit_ppl"]),
        (CANONICAL_REFERENCE["continuation_ppl"]["bf16"]["avg_kv_mb"],
         q["continuation_ppl_wikitext2"]["bf16_avg_logical_kv_mb"]),
        (CANONICAL_REFERENCE["continuation_ppl"]["rabit2"]["avg_kv_mb"],
         q["continuation_ppl_wikitext2"]["rabit_avg_logical_kv_mb"]),
        (CANONICAL_REFERENCE["niah"]["bf16"]["accuracy_pct"],
         q["needle_in_a_haystack"]["bf16_accuracy_percent"]),
        (CANONICAL_REFERENCE["niah"]["rabit2"]["accuracy_pct"],
         q["needle_in_a_haystack"]["rabit_accuracy_percent"]),
        (CANONICAL_REFERENCE["niah"]["rabit2"]["avg_kv_mb"],
         q["needle_in_a_haystack"]["rabit_avg_logical_kv_mb"]),
        (CANONICAL_REFERENCE["passage_retrieval"]["bf16"]["accuracy_pct"],
         q["longbench_passage_retrieval"]["bf16_accuracy_percent"]),
        (CANONICAL_REFERENCE["passage_retrieval"]["rabit2"]["accuracy_pct"],
         q["longbench_passage_retrieval"]["rabit_accuracy_percent"]),
        (CANONICAL_REFERENCE["passage_retrieval"]["bf16"]["avg_kv_mb"],
         q["longbench_passage_retrieval"]["bf16_avg_logical_kv_mb"]),
        (CANONICAL_REFERENCE["passage_retrieval"]["rabit2"]["avg_kv_mb"],
         q["longbench_passage_retrieval"]["rabit_avg_logical_kv_mb"]),
        (CANONICAL_REFERENCE["hotpotqa"]["bf16"]["f1_pct"],
         q["longbench_hotpotqa_e"]["bf16_f1_percent"]),
        (CANONICAL_REFERENCE["hotpotqa"]["rabit2"]["f1_pct"],
         q["longbench_hotpotqa_e"]["rabit_f1_percent"]),
        (CANONICAL_REFERENCE["hotpotqa"]["bf16"]["avg_kv_mb"],
         q["longbench_hotpotqa_e"]["bf16_avg_logical_kv_mb"]),
        (CANONICAL_REFERENCE["hotpotqa"]["rabit2"]["avg_kv_mb"],
         q["longbench_hotpotqa_e"]["rabit_avg_logical_kv_mb"]),
        (CANONICAL_REFERENCE["qasper"]["bf16"]["f1_pct"],
         q["longbench_qasper_e"]["bf16_f1_percent"]),
        (CANONICAL_REFERENCE["qasper"]["rabit2"]["f1_pct"],
         q["longbench_qasper_e"]["rabit_f1_percent"]),
        (CANONICAL_REFERENCE["qasper"]["bf16"]["avg_kv_mb"],
         q["longbench_qasper_e"]["bf16_avg_logical_kv_mb"]),
        (CANONICAL_REFERENCE["qasper"]["rabit2"]["avg_kv_mb"],
         q["longbench_qasper_e"]["rabit_avg_logical_kv_mb"]),
    ]
    for pinned, live in pairs:
        if abs(pinned - live) > 1e-6:
            raise RuntimeError(
                "CANONICAL_REFERENCE constant in this script has drifted "
                f"from results/summary.json: pinned={pinned} live={live}. "
                "results/summary.json is supposed to be frozen "
                "(Ground Rule 1) -- investigate before proceeding."
            )

    # NIAH's bf16 avg KV MB has no entry in results/summary.json; sourced
    # from the raw canonical log instead. Verify it is still there.
    niah_log = (ROOT / "results" / "quality" / "niah.log").read_text(
        encoding="utf-8", errors="replace"
    )
    if "bf16        15          15          100.0         1194.542" not in niah_log:
        raise RuntimeError(
            "Pinned NIAH bf16 aggregate row not found verbatim in "
            "results/quality/niah.log -- canonical file may have changed."
        )


def preflight() -> dict:
    branch = run_git("branch", "--show-current")
    head = run_git("rev-parse", "HEAD")

    assert_protected_paths_clean("preflight, before any run")

    if not RABIT_KV2.is_file():
        raise FileNotFoundError(RABIT_KV2)
    rabit_sha = sha256(RABIT_KV2)
    if rabit_sha != EXPECTED_RABIT_SHA256:
        raise RuntimeError(
            "rabit_kv2.py does not match the frozen SHA-256 recorded in "
            "benchmarks/quality/run_suite.py. This experiment does not "
            "read rabit_kv2.py, but a mismatch means the repository is "
            "not in the expected frozen state.\n"
            f"expected={EXPECTED_RABIT_SHA256}\nactual={rabit_sha}"
        )

    script_hashes = {}
    for spec in RUNS:
        path = QUALITY_DIR / spec["script"]
        if not path.is_file():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        if REQUIRED_ALLOWED_LINE not in text:
            raise RuntimeError(
                f"{path} does not expose the expected full-frontier "
                "'allowed' method set. Refusing to run -- this script may "
                "have been modified since the plan was validated."
            )
        if REQUIRED_RABIT2_MARKER not in text:
            raise RuntimeError(
                f"{path} does not expose the frozen RABIT-2 META8g64 "
                "policy marker."
            )
        script_hashes[spec["script"]] = sha256(path)

    verify_canonical_reference()

    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if existing.get("status") == "passed":
            raise RuntimeError(
                f"{MANIFEST} already records a passed run. Refusing to "
                "overwrite prior Experiment 1 evidence. Move or remove "
                "results/mlsys2027/quality_frontier/ first if a genuine "
                "re-run is intended."
            )

    return {
        "git_branch": branch,
        "git_head": head,
        "rabit_kv2_sha256": rabit_sha,
        "quality_script_sha256": script_hashes,
        "runner_script": str(RUNNER_SCRIPT.relative_to(ROOT)).replace("\\", "/"),
        "runner_script_sha256": sha256(RUNNER_SCRIPT),
        "protected_paths_baseline_status": "clean",
    }


def build_commands() -> list[dict]:
    commands = []
    for spec in RUNS:
        script = QUALITY_DIR / spec["script"]
        cmd = [sys.executable, "-m", "modal", "run", str(script), *spec["args"]]
        commands.append(
            {
                "name": spec["name"],
                "purpose": spec["purpose"],
                "command": cmd,
                "log": str((OUT_DIR / f"{spec['name']}.log").relative_to(ROOT)).replace(
                    "\\", "/"
                ),
            }
        )
    return commands


def stream_command(cmd: list[str], log_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    print("\n" + "=" * 118)
    print("RUNNING:", " ".join(cmd))
    print("LOG:", log_path)
    print("=" * 118)

    with log_path.open("w", encoding="utf-8") as log:
        p = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return p.wait()


def _last_row_tokens(log_text: str, method: str) -> list[str] | None:
    """Return the whitespace-split tokens of the LAST line in the log that
    starts with `method` followed by whitespace (the printed summary-table
    row for that method). Returns None if not found."""
    pattern = re.compile(rf"^{re.escape(method)}\s")
    match_line = None
    for line in log_text.splitlines():
        if pattern.match(line):
            match_line = line
    return match_line.split() if match_line is not None else None


def _check(name: str, pinned: float, observed: float, abs_tol: float | None,
           rel_tol: float | None) -> dict:
    diff = abs(observed - pinned)
    if rel_tol is not None:
        tol = max(abs(pinned) * rel_tol, 1e-9)
    else:
        tol = abs_tol
    return {
        "metric": name,
        "canonical": pinned,
        "observed": observed,
        "abs_diff": diff,
        "tolerance": tol,
        "within_tolerance": diff <= tol,
    }


def check_regression(name: str, log_text: str) -> dict:
    """Compare the bf16/rabit2 rows of one benchmark's log against
    CANONICAL_REFERENCE. rabit8/rabit4/rabit3 are not checked (no
    canonical baseline exists for them -- they are new evidence)."""
    ref = CANONICAL_REFERENCE[name]
    checks: list[dict] = []

    if name == "continuation_ppl":
        for method in ("bf16", "rabit2"):
            tokens = _last_row_tokens(log_text, method)
            if tokens is None or len(tokens) < 5:
                checks.append({"metric": f"{method}: row not found", "within_tolerance": False})
                continue
            ppl, avg_kv_mb = float(tokens[1]), float(tokens[4])
            checks.append(_check(f"{method}.ppl", ref[method]["ppl"], ppl, None, PPL_RELATIVE_TOLERANCE))
            checks.append(_check(f"{method}.avg_kv_mb", ref[method]["avg_kv_mb"], avg_kv_mb, None, KV_MB_RELATIVE_TOLERANCE))

    elif name == "niah":
        for method in ("bf16", "rabit2"):
            tokens = _last_row_tokens(log_text, method)
            if tokens is None or len(tokens) < 5:
                checks.append({"metric": f"{method}: row not found", "within_tolerance": False})
                continue
            accuracy, avg_kv_mb = float(tokens[3]), float(tokens[4])
            checks.append(_check(f"{method}.accuracy_pct", ref[method]["accuracy_pct"], accuracy, PERCENTAGE_ABSOLUTE_TOLERANCE, None))
            checks.append(_check(f"{method}.avg_kv_mb", ref[method]["avg_kv_mb"], avg_kv_mb, None, KV_MB_RELATIVE_TOLERANCE))

    elif name == "passage_retrieval":
        for method in ("bf16", "rabit2"):
            tokens = _last_row_tokens(log_text, method)
            if tokens is None or len(tokens) < 4:
                checks.append({"metric": f"{method}: row not found", "within_tolerance": False})
                continue
            accuracy, avg_kv_mb = float(tokens[1]), float(tokens[3])
            checks.append(_check(f"{method}.accuracy_pct", ref[method]["accuracy_pct"], accuracy, PERCENTAGE_ABSOLUTE_TOLERANCE, None))
            checks.append(_check(f"{method}.avg_kv_mb", ref[method]["avg_kv_mb"], avg_kv_mb, None, KV_MB_RELATIVE_TOLERANCE))

    elif name in ("hotpotqa", "qasper"):
        for method in ("bf16", "rabit2"):
            tokens = _last_row_tokens(log_text, method)
            if tokens is None or len(tokens) < 4:
                checks.append({"metric": f"{method}: row not found", "within_tolerance": False})
                continue
            f1, avg_kv_mb = float(tokens[1]), float(tokens[3])
            checks.append(_check(f"{method}.f1_pct", ref[method]["f1_pct"], f1, PERCENTAGE_ABSOLUTE_TOLERANCE, None))
            checks.append(_check(f"{method}.avg_kv_mb", ref[method]["avg_kv_mb"], avg_kv_mb, None, KV_MB_RELATIVE_TOLERANCE))

    else:
        raise ValueError(f"Unknown benchmark name: {name}")

    return {
        "benchmark": name,
        "checks": checks,
        "all_within_tolerance": all(c.get("within_tolerance") for c in checks),
    }


def write_manifest(manifest: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_regression_check(results: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REGRESSION_CHECK.write_text(
        json.dumps(
            {
                "note": (
                    "bf16/rabit2 regression check only. rabit8/rabit4/rabit3 "
                    "have no canonical baseline and are not checked here -- "
                    "see results/mlsys2027/quality_frontier/*.log for their "
                    "raw (new) values. All avg_kv_mb figures are LOGICAL "
                    "fake-quant KV memory, not physical allocator capacity."
                ),
                "tolerances": {
                    "ppl_relative": PPL_RELATIVE_TOLERANCE,
                    "percentage_absolute_points": PERCENTAGE_ABSOLUTE_TOLERANCE,
                    "avg_kv_mb_relative": KV_MB_RELATIVE_TOLERANCE,
                },
                "results": results,
                "all_benchmarks_within_tolerance": all(
                    r["all_within_tolerance"] for r in results
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all static/preflight checks and print the exact commands "
        "this script would execute, without invoking Modal or the GPU. "
        "Writes nothing to disk.",
    )
    args = parser.parse_args(argv)

    print("RABIT-KV MLSys 2027 -- Experiment 1: quality-compression frontier")
    print("Policy under test: bf16 / rabit8 / rabit4 / rabit3 / rabit2 (unchanged)")
    print("Canonical results/quality/, results/performance/, results/summary.json")
    print("are read-only reference points and are never written by this script.")
    print("All KV-memory figures below are LOGICAL fake-quant KV storage.")
    print()

    provenance = preflight()
    print("Preflight OK.")
    print(f"  git branch:        {provenance['git_branch']}")
    print(f"  git HEAD:          {provenance['git_head']}")
    print(f"  rabit_kv2.py SHA:  {provenance['rabit_kv2_sha256']} (matches expected)")
    for script, digest in provenance["quality_script_sha256"].items():
        print(f"  {script:<24} SHA: {digest}")
    print(f"  {provenance['runner_script']:<24} SHA: {provenance['runner_script_sha256']}")
    print()

    commands = build_commands()
    print(f"Would execute {len(commands)} Modal runs, in order:")
    for c in commands:
        print(f"\n[{c['name']}] {c['purpose']}")
        print("  " + " ".join(c["command"]))
        print(f"  -> log: {c['log']}")

    if args.dry_run:
        print("\n--dry-run: no Modal/GPU commands executed, no files written.")
        return 0

    manifest = {
        "experiment": "Experiment 1 -- BF16/8/4/3/2 quality-compression frontier",
        "plan_reference": "docs/MLSYS_EXPERIMENT_PLAN.md#experiment-1",
        "policy": "K3/V2/G32/R4/META8g64 (rabit2), unchanged; frontier adds rabit8/rabit4/rabit3",
        "model": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
        "methods": METHODS.split(","),
        "gpu": "NVIDIA H100 80GB HBM3 (Modal)",
        "kv_memory_label": "LOGICAL fake-quant KV storage (HF-side one-shot quantize/dequantize); NOT physical vLLM allocator capacity",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "completed_utc": None,
        "status": "running",
        "provenance": provenance,
        "regression_tolerances": {
            "ppl_relative": PPL_RELATIVE_TOLERANCE,
            "percentage_absolute_points": PERCENTAGE_ABSOLUTE_TOLERANCE,
            "avg_kv_mb_relative": KV_MB_RELATIVE_TOLERANCE,
        },
        "runs": [],
        "note": (
            "Reproduction/extension run for the frozen RABIT-KV operating "
            "point plus the already-supported 8/4/3-bit frontier. Canonical "
            "committed results (results/quality/, results/performance/, "
            "results/summary.json) remain unchanged."
        ),
    }
    write_manifest(manifest)

    regression_results = []
    for spec, c in zip(RUNS, commands):
        log_path = OUT_DIR / f"{spec['name']}.log"
        row = {
            "name": spec["name"],
            "purpose": spec["purpose"],
            "script": str((QUALITY_DIR / spec["script"]).relative_to(ROOT)).replace("\\", "/"),
            "command": c["command"],
            "log": c["log"],
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "completed_utc": None,
            "returncode": None,
            "status": "running",
        }
        manifest["runs"].append(row)
        write_manifest(manifest)

        code = stream_command(c["command"], log_path)

        # Fix (4): check protected paths immediately after EVERY benchmark,
        # not only before the whole experiment and after all five runs --
        # regardless of whether the benchmark itself succeeded.
        assert_protected_paths_clean(f"immediately after '{spec['name']}'")

        row["returncode"] = code
        row["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()

        if code != 0:
            row["status"] = "failed"
            write_manifest(manifest)
            manifest["status"] = "failed"
            manifest["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            write_manifest(manifest)
            raise SystemExit(
                f"\nEXPERIMENT 1 STOPPED at {spec['name']} (exit={code}). "
                "Completed logs were preserved under "
                "results/mlsys2027/quality_frontier/."
            )

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        reg = check_regression(spec["name"], log_text)
        row["regression_check"] = reg
        regression_results.append(reg)

        # Fix (1): a failed bf16/rabit2 regression check is a hard stop.
        # The log (already on disk from stream_command) and the full
        # regression detail (already attached to `row` above) are
        # preserved before raising. The manifest is never allowed to end
        # in "passed" when a regression check failed.
        if not reg["all_within_tolerance"]:
            row["status"] = "regression_failed"
            write_manifest(manifest)
            write_regression_check(regression_results)
            manifest["status"] = "failed"
            manifest["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            write_manifest(manifest)
            raise SystemExit(
                f"\nEXPERIMENT 1 STOPPED at {spec['name']}: bf16/rabit2 "
                "regression check failed outside tolerance. Log and "
                "regression details preserved under "
                f"results/mlsys2027/quality_frontier/ "
                f"({REGRESSION_CHECK.name}, {log_path.name}, {MANIFEST.name})."
            )

        row["status"] = "passed"
        write_manifest(manifest)

    assert_protected_paths_clean("after all runs completed")

    write_regression_check(regression_results)
    manifest["status"] = "passed"
    manifest["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["protected_paths_post_run_status"] = "clean"
    write_manifest(manifest)

    print("\n" + "=" * 118)
    print("EXPERIMENT 1: ALL RUNS PASSED")
    print(f"Manifest:          {MANIFEST}")
    print(f"Regression check:  {REGRESSION_CHECK}")
    print("Reminder: all KV-memory figures above are LOGICAL fake-quant KV")
    print("storage, not physical vLLM allocator capacity.")
    print("=" * 118)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
