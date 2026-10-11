"""
RABIT-KV MLSys 2027 -- Experiment 2 runner (multilingual quality-compression
frontier).

Runs the existing Chinese/Spanish multilingual continuation-PPL benchmark
(benchmarks/quality/multilingual_ppl.py) across the operating-point sweep
bf16/rabit8/rabit4/rabit3/rabit2, with every canonical setting preserved
(languages zh,es; pinned Wikipedia dataset revision; shuffle seed/buffer;
1024 context + 128 eval tokens; 8 samples per language; Llama-3.1-8B-Instruct).
See docs/MLSYS_EXPERIMENT_PLAN.md.

This script:
  * does NOT modify benchmarks/quality/multilingual_ppl.py (it only verifies
    that the script exposes the expected method set and that its preset /
    quantization code is byte-identical to continuation_ppl.py's);
  * does NOT write to results/quality/, results/performance/,
    results/summary.json, vllm-kvquant/, or the committed Experiment 1
    evidence under results/mlsys2027/quality_frontier/;
  * writes ONLY under results/mlsys2027/multilingual_frontier/.

The canonical multilingual run (results/quality/multilingual_ppl.log) covered
bf16 and rabit2 only, so ONLY bf16/rabit2 are regression-checked here.
rabit8/rabit4/rabit3 have no canonical multilingual baseline and are new
evidence, recorded but not regression-checked.

All KV-memory numbers are LOGICAL fake-quant KV storage (HF-side one-shot
quantize/dequantize accounting), NOT physical vLLM allocator capacity.

Usage:
    python benchmarks/mlsys2027/run_experiment2_multilingual_frontier.py --dry-run
        Runs every static/preflight check and prints the exact command this
        script would execute, without invoking Modal or the GPU. Writes
        nothing to disk.

    python benchmarks/mlsys2027/run_experiment2_multilingual_frontier.py
        Executes the experiment (one Modal H100 run). Refuses to start unless
        this runner and multilingual_ppl.py are committed and unmodified.
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
SCRIPT = QUALITY_DIR / "multilingual_ppl.py"
REFERENCE_SCRIPT = QUALITY_DIR / "continuation_ppl.py"
OUT_DIR = ROOT / "results" / "mlsys2027" / "multilingual_frontier"
LOG = OUT_DIR / "multilingual_ppl.log"
MANIFEST = OUT_DIR / "manifest.json"
REGRESSION_CHECK = OUT_DIR / "regression_check.json"
CANONICAL_SUMMARY = ROOT / "results" / "summary.json"
CANONICAL_LOG = ROOT / "results" / "quality" / "multilingual_ppl.log"

RABIT_KV2 = ROOT / "vllm-kvquant" / "vllm" / "v1" / "attention" / "ops" / "rabit_kv2.py"
# Project-wide provenance invariant; multilingual_ppl.py does not import
# rabit_kv2.py. run_suite.py's frozen EXPECTED_RABIT_SHA256 (108ba8...) is the
# raw-byte hash of a CRLF (Windows core.autocrlf) checkout of this file, so it
# only matches on such a checkout. This runner checks the LF-normalized hash,
# which equals the committed git content (7e628c...) on every platform; the
# CRLF-checkout value is recorded alongside for cross-reference.
EXPECTED_RABIT_SHA256_LF = (
    "7e628c94eebb9fe689bf416ea61f748c0f909a82d0f229c463edd1a0df92e6ae"
)
RUN_SUITE_RABIT_SHA256_CRLF = (
    "108ba8afaac862bc4e070d2dc812493aaeae4e458837c4bda2ac703e72a57524"
)

# Paths this experiment must never modify. Checked via `git status --short`
# before the run, after the run, and on every failure path.
PROTECTED_PATHS = [
    ROOT / "results" / "quality",
    ROOT / "results" / "performance",
    ROOT / "results" / "summary.json",
    ROOT / "vllm-kvquant",
    ROOT / "results" / "mlsys2027" / "quality_frontier",
]

# Files whose committed state defines this experiment. A real run refuses to
# start if either has uncommitted changes, so the recorded SHAs always
# correspond to a commit.
MUST_BE_COMMITTED = [SCRIPT, RUNNER_SCRIPT]

MODEL = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
METHODS = ["bf16", "rabit8", "rabit4", "rabit3", "rabit2"]
LANGUAGES = {"zh": "CHINESE", "es": "SPANISH"}
REQUIRED_ALLOWED_LINE = 'allowed = {"bf16", "rabit8", "rabit4", "rabit3", "rabit2"}'
EXPECTED_PRESETS = {
    "bf16": "BF16 baseline",
    "rabit8": "8b META8g256 G128 R0",
    "rabit4": "4b META8g64 SYM G128 R0",
    "rabit3": "3b META8g64 SYM G32 R2",
    "rabit2": "2b META8g64 K3V2 G32 R4",
}

# Canonical args mirror benchmarks/quality/run_suite.py's multilingual_ppl
# entry exactly, with --methods extended from "bf16,rabit2" to the sweep and
# --model-name passed explicitly (same value as the script default).
ARGS = [
    "--model-name", MODEL,
    "--languages", "zh,es",
    "--context-tokens", "1024",
    "--eval-tokens", "128",
    "--samples", "8",
    "--methods", ",".join(METHODS),
    "--dataset-revision", "cf584d1dc131caa92a5cb910f41a8b7591b12732",
    "--shuffle-seed", "20260804",
    "--shuffle-buffer", "1000",
]
EXPECTED_TOKENS_PER_ROW = 8 * 128

# ---------------------------------------------------------------------------
# Canonical bf16/rabit2 reference values, for regression checking ONLY.
# PPL values are pinned from results/summary.json
# (final_quality.multilingual_continuation_ppl); avg KV MB, which summary.json
# omits, from the per-language table rows of results/quality/multilingual_ppl.log.
# verify_canonical_reference() cross-checks both sources at preflight.
# ---------------------------------------------------------------------------
CANONICAL_REFERENCE = {
    "zh": {
        "bf16": {"ppl": 10.1176, "avg_kv_mb": 128.000},
        "rabit2": {"ppl": 10.3308, "avg_kv_mb": 24.710},
    },
    "es": {
        "bf16": {"ppl": 5.0035, "avg_kv_mb": 128.000},
        "rabit2": {"ppl": 5.1139, "avg_kv_mb": 24.710},
    },
}
CANONICAL_LOG_ROWS = {
    "zh": {
        "bf16": "bf16        10.1176       0.00          10.0736               128.000         1.000           1024",
        "rabit2": "rabit2      10.3308       2.11          10.2241               24.710          5.180           1024",
    },
    "es": {
        "bf16": "bf16        5.0035        0.00          4.4927                128.000         1.000           1024",
        "rabit2": "rabit2      5.1139        2.21          4.5324                24.710          5.180           1024",
    },
}

# ---------------------------------------------------------------------------
# Regression tolerances (bf16/rabit2 only), identical to Experiment 1's for
# the same metric types and for the same reasons (see
# run_experiment1_quality_frontier.py): TF32 matmul, no deterministic-
# algorithms mode, fresh Modal H100 instance.
# - PPL: 0.5% relative. Well below the canonical rabit2 effect (+2.11% zh,
#   +2.21% es), so it cannot mask a real regression. In Experiment 1 the same
#   code path (continuation_ppl.py) reproduced bf16/rabit2 PPL exactly.
# - Avg KV MB: 0.1% relative. Deterministic byte accounting; covers only the
#   script's .3f print rounding.
# ---------------------------------------------------------------------------
PPL_RELATIVE_TOLERANCE = 0.005
KV_MB_RELATIVE_TOLERANCE = 0.001

KV_LABEL = "LOGICAL fake-quant KV storage; not physical vLLM allocator capacity"


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
    """SHA-256 of LF-normalized bytes (CRLF -> LF): equals the committed git
    content's hash regardless of core.autocrlf."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def assert_protected_paths_clean(context: str) -> None:
    """Raise if any protected path shows an uncommitted change."""
    status = run_git("status", "--short", "--", *[rel(p) for p in PROTECTED_PATHS])
    if status:
        raise RuntimeError(
            f"CRITICAL: protected canonical paths changed ({context}). "
            "This must never happen. Investigate immediately:\n" + status
        )


def uncommitted_experiment_files() -> str:
    return run_git("status", "--short", "--", *[rel(p) for p in MUST_BE_COMMITTED])


def quant_region(path: Path) -> str:
    """Preset table + quantize/dequantize + logical byte accounting +
    evaluate_one: from `def encode_metadata(` up to the Configurations print."""
    text = path.read_text(encoding="utf-8")
    start = text.index("    def encode_metadata(")
    end = text.index('    print("Configurations:")')
    return text[start:end]


def verify_script() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    if REQUIRED_ALLOWED_LINE not in text:
        raise RuntimeError(
            f"{rel(SCRIPT)} does not expose the full sweep 'allowed' method "
            "set. Refusing to run."
        )
    for method, name in EXPECTED_PRESETS.items():
        if method != "bf16" and f'"name": "{name}"' not in text:
            raise RuntimeError(f"{rel(SCRIPT)} is missing the {method} preset '{name}'.")
    if quant_region(SCRIPT) != quant_region(REFERENCE_SCRIPT):
        raise RuntimeError(
            f"Preset/quantization code in {rel(SCRIPT)} is no longer "
            f"byte-identical to {rel(REFERENCE_SCRIPT)}. Refusing to run: the "
            "multilingual sweep must use exactly the Experiment 1 presets."
        )


def verify_canonical_reference() -> None:
    """Fail loudly if CANONICAL_REFERENCE has drifted from the frozen
    canonical files. Read-only."""
    q = json.loads(CANONICAL_SUMMARY.read_text(encoding="utf-8"))["final_quality"]
    m = q["multilingual_continuation_ppl"]
    pairs = [
        (CANONICAL_REFERENCE["zh"]["bf16"]["ppl"], m["chinese"]["bf16_ppl"]),
        (CANONICAL_REFERENCE["zh"]["rabit2"]["ppl"], m["chinese"]["rabit_ppl"]),
        (CANONICAL_REFERENCE["es"]["bf16"]["ppl"], m["spanish"]["bf16_ppl"]),
        (CANONICAL_REFERENCE["es"]["rabit2"]["ppl"], m["spanish"]["rabit_ppl"]),
    ]
    for pinned, live in pairs:
        if abs(pinned - live) > 1e-6:
            raise RuntimeError(
                "CANONICAL_REFERENCE has drifted from results/summary.json: "
                f"pinned={pinned} live={live}. Investigate before proceeding."
            )
    if (m["languages"], m["samples_per_language"], m["context_tokens"], m["eval_tokens"]) != (
        ["zh", "es"], 8, 1024, 128
    ):
        raise RuntimeError("Canonical multilingual settings in results/summary.json changed.")

    canonical_tables = parse_language_tables(CANONICAL_LOG.read_text(encoding="utf-8"))
    for lang, rows in CANONICAL_LOG_ROWS.items():
        for method, expected_row in rows.items():
            if canonical_tables.get(lang, {}).get(method, {}).get("row") != expected_row:
                raise RuntimeError(
                    f"Pinned canonical {lang}/{method} row not found verbatim in "
                    f"{rel(CANONICAL_LOG)} -- canonical file may have changed."
                )
            tok = expected_row.split()
            if float(tok[1]) != CANONICAL_REFERENCE[lang][method]["ppl"] or float(
                tok[4]
            ) != CANONICAL_REFERENCE[lang][method]["avg_kv_mb"]:
                raise RuntimeError(f"Pinned {lang}/{method} values disagree with canonical log row.")


def preflight(dry_run: bool) -> dict:
    branch = run_git("branch", "--show-current")
    head = run_git("rev-parse", "HEAD")

    assert_protected_paths_clean("preflight, before any run")

    rabit_sha = sha256(RABIT_KV2)
    if rabit_sha != EXPECTED_RABIT_SHA256_LF:
        raise RuntimeError(
            "rabit_kv2.py does not match the frozen committed content "
            "(LF-normalized SHA-256).\n"
            f"expected={EXPECTED_RABIT_SHA256_LF}\nactual={rabit_sha}"
        )

    verify_script()
    verify_canonical_reference()

    uncommitted = uncommitted_experiment_files()
    if uncommitted and not dry_run:
        raise RuntimeError(
            "Refusing to run: the runner and/or multilingual_ppl.py have "
            "uncommitted changes, so their recorded SHAs would not match a "
            "commit:\n" + uncommitted
        )

    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if existing.get("status") == "passed":
            raise RuntimeError(
                f"{rel(MANIFEST)} already records a passed run. Refusing to "
                "overwrite prior Experiment 2 evidence."
            )

    return {
        "git_branch": branch,
        "git_head": head,
        "rabit_kv2_sha256": rabit_sha,
        "rabit_kv2_sha256_crlf_checkout_equivalent": RUN_SUITE_RABIT_SHA256_CRLF,
        "multilingual_ppl_sha256": sha256(SCRIPT),
        "reference_continuation_ppl_sha256": sha256(REFERENCE_SCRIPT),
        "runner_script": rel(RUNNER_SCRIPT),
        "runner_script_sha256": sha256(RUNNER_SCRIPT),
        "canonical_summary_sha256": sha256(CANONICAL_SUMMARY),
        "canonical_multilingual_log_sha256": sha256(CANONICAL_LOG),
        "sha256_basis": "LF-normalized bytes (CRLF -> LF); equals committed git content",
        "protected_paths_baseline_status": "clean",
        "uncommitted_experiment_files": uncommitted or None,
    }


def build_command() -> list[str]:
    return [sys.executable, "-m", "modal", "run", str(SCRIPT), *ARGS]


def make_console_encoding_safe() -> None:
    """Local console echo must never crash on characters the terminal
    encoding cannot represent. Affects ONLY console display; the .log file is
    written separately as UTF-8 with the original text unchanged."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def console_write(text: str) -> None:
    """Echo `text` to the console, never raising on encoding errors."""
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        sys.stdout.write(
            text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        )
    sys.stdout.flush()


def stream_command(cmd: list[str], log_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    console_write("\n" + "=" * 118 + "\n")
    console_write(f"RUNNING: {' '.join(cmd)}\n")
    console_write(f"LOG: {log_path}\n")
    console_write("=" * 118 + "\n")

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
        try:
            for line in p.stdout:
                # Log first (UTF-8, unchanged), then echo.
                log.write(line)
                log.flush()
                console_write(line)
        except BaseException:
            # Do not leave an orphaned `modal run` client / H100 app behind.
            p.kill()
            p.wait()
            raise
        return p.wait()


SECTION = re.compile(r"^RABIT-KV MULTILINGUAL CONTINUATION PPL \S+ (CHINESE|SPANISH)\s*$")
ROW = re.compile(r"^(bf16|rabit\d)\s")


def parse_language_tables(log_text: str) -> dict:
    """Parse each per-language results table:
    Method PPL PPL_d% MedianSamplePPL AvgKVMB Compression Tokens.
    Rows are taken only from inside a language section (the final summary
    table reuses the method names and is deliberately not parsed here)."""
    code_for = {v: k for k, v in LANGUAGES.items()}
    tables: dict = {}
    current = None
    for lineno, line in enumerate(log_text.splitlines(), start=1):
        section = SECTION.match(line)
        if section:
            current = code_for[section.group(1)]
            tables[current] = {}
            continue
        if line.startswith("MULTILINGUAL RELATIVE-QUALITY SUMMARY"):
            current = None
            continue
        if current and ROW.match(line):
            tok = line.split()
            if len(tok) != 7:
                continue
            tables[current][tok[0]] = {
                "line": lineno,
                "row": line.rstrip(),
                "ppl": float(tok[1]),
                "ppl_delta_pct": float(tok[2]),
                "median_sample_ppl": float(tok[3]),
                "avg_kv_mb": float(tok[4]),
                "compression": float(tok[5]),
                "tokens": int(tok[6]),
            }
    return tables


def parse_configurations(log_text: str) -> dict:
    lines = log_text.splitlines()
    try:
        i = lines.index("Configurations:")
    except ValueError:
        return {}
    out = {}
    for line in lines[i + 1 :]:
        if not line.startswith("  ") or ": " not in line:
            break
        key, value = line.strip().split(": ", 1)
        out[key] = value
    return out


def _check(name: str, pinned: float, observed: float, rel_tol: float) -> dict:
    diff = abs(observed - pinned)
    tol = max(abs(pinned) * rel_tol, 1e-9)
    return {
        "metric": name,
        "canonical": pinned,
        "observed": observed,
        "abs_diff": diff,
        "tolerance": tol,
        "within_tolerance": diff <= tol,
    }


def check_regression(log_text: str) -> dict:
    """Integrity checks on the log (all five methods present per language,
    expected presets, expected token counts) plus bf16/rabit2 regression
    against CANONICAL_REFERENCE. rabit8/rabit4/rabit3 are new evidence."""
    tables = parse_language_tables(log_text)
    checks: list[dict] = []

    configs = parse_configurations(log_text)
    checks.append({
        "metric": "logged presets == expected presets",
        "observed": configs,
        "within_tolerance": configs == EXPECTED_PRESETS,
    })

    for lang in LANGUAGES:
        rows = tables.get(lang, {})
        checks.append({
            "metric": f"{lang}: all five method rows present",
            "observed": sorted(rows),
            "within_tolerance": sorted(rows) == sorted(METHODS),
        })
        for method in METHODS:
            if method in rows:
                checks.append({
                    "metric": f"{lang}.{method}.tokens == {EXPECTED_TOKENS_PER_ROW}",
                    "observed": rows[method]["tokens"],
                    "within_tolerance": rows[method]["tokens"] == EXPECTED_TOKENS_PER_ROW,
                })
        for method in ("bf16", "rabit2"):
            if method not in rows:
                checks.append({"metric": f"{lang}.{method}: row not found", "within_tolerance": False})
                continue
            ref = CANONICAL_REFERENCE[lang][method]
            checks.append(_check(f"{lang}.{method}.ppl", ref["ppl"], rows[method]["ppl"], PPL_RELATIVE_TOLERANCE))
            checks.append(_check(f"{lang}.{method}.avg_kv_mb", ref["avg_kv_mb"], rows[method]["avg_kv_mb"], KV_MB_RELATIVE_TOLERANCE))

    return {
        "benchmark": "multilingual_ppl",
        "checks": checks,
        "all_within_tolerance": all(c.get("within_tolerance") for c in checks),
    }


def write_manifest(manifest: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_regression_check(result: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REGRESSION_CHECK.write_text(
        json.dumps(
            {
                "note": (
                    "bf16/rabit2 regression check only (the canonical multilingual "
                    "run covered bf16 and rabit2 only). rabit8/rabit4/rabit3 have no "
                    "canonical multilingual baseline and are new evidence -- see "
                    "multilingual_ppl.log. All avg_kv_mb figures: " + KV_LABEL + "."
                ),
                "tolerances": {
                    "ppl_relative": PPL_RELATIVE_TOLERANCE,
                    "avg_kv_mb_relative": KV_MB_RELATIVE_TOLERANCE,
                },
                "result": result,
                "all_within_tolerance": result["all_within_tolerance"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def record_runner_failure(manifest: dict, row: dict | None, exc: BaseException) -> None:
    """Exception path: mark the in-flight run (if any) runner_failed, fail the
    manifest, run the protected-path check (recording its own failure
    separately), and persist the manifest. Partial logs are kept. No retry."""
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    error = {"type": type(exc).__name__, "message": str(exc)}
    if row is not None:
        row["status"] = "runner_failed"
        row["completed_utc"] = now
        row["error"] = error
    manifest["runner_error"] = {"run": row["name"] if row is not None else None, **error}
    manifest["status"] = "failed"
    manifest["completed_utc"] = now
    try:
        assert_protected_paths_clean("runner exception path")
    except Exception as check_exc:
        manifest["protected_paths_post_run_status"] = "check_failed"
        manifest["protected_paths_check_error"] = {
            "type": type(check_exc).__name__,
            "message": str(check_exc),
        }
    else:
        manifest["protected_paths_post_run_status"] = "clean"
    write_manifest(manifest)


def run(manifest: dict, command: list[str]) -> int:
    row = {
        "name": "multilingual_ppl",
        "purpose": "Chinese/Spanish Wikipedia continuation PPL, bf16/rabit8/rabit4/rabit3/rabit2 sweep",
        "script": rel(SCRIPT),
        "command": command,
        "log": rel(LOG),
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "completed_utc": None,
        "returncode": None,
        "status": "running",
    }
    manifest["runs"].append(row)
    write_manifest(manifest)

    code = stream_command(command, LOG)
    assert_protected_paths_clean("immediately after 'multilingual_ppl'")

    row["returncode"] = code
    row["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    if code != 0:
        row["status"] = "failed"
        manifest["status"] = "failed"
        manifest["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_manifest(manifest)
        raise SystemExit(
            f"\nEXPERIMENT 2 STOPPED: multilingual_ppl exited {code}. "
            f"Log preserved at {rel(LOG)}."
        )

    reg = check_regression(LOG.read_text(encoding="utf-8", errors="replace"))
    row["regression_check"] = reg
    write_regression_check(reg)
    if not reg["all_within_tolerance"]:
        row["status"] = "regression_failed"
        manifest["status"] = "failed"
        manifest["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_manifest(manifest)
        raise SystemExit(
            "\nEXPERIMENT 2 STOPPED: integrity/bf16-rabit2 regression check "
            f"failed. Details in {rel(REGRESSION_CHECK)}; log at {rel(LOG)}."
        )

    row["status"] = "passed"
    write_manifest(manifest)

    assert_protected_paths_clean("after run completed")
    manifest["status"] = "passed"
    manifest["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["protected_paths_post_run_status"] = "clean"
    write_manifest(manifest)

    print("\n" + "=" * 118)
    print("EXPERIMENT 2: RUN PASSED")
    print(f"Manifest:          {MANIFEST}")
    print(f"Regression check:  {REGRESSION_CHECK}")
    print(f"Reminder: all KV-memory figures are {KV_LABEL}.")
    print("=" * 118)
    return 0


def main(argv: list[str] | None = None) -> int:
    make_console_encoding_safe()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all preflight checks and print the exact command, without "
        "invoking Modal or the GPU. Writes nothing to disk.",
    )
    args = parser.parse_args(argv)

    print("RABIT-KV MLSys 2027 -- Experiment 2: multilingual quality-compression frontier")
    print("Operating-point sweep: bf16 / rabit8 / rabit4 / rabit3 / rabit2 (presets unchanged)")
    print("Regression-checked vs canonical: bf16, rabit2. New evidence: rabit8, rabit4, rabit3.")
    print("Canonical results and Experiment 1 evidence are read-only and never written.")
    print(f"All KV-memory figures: {KV_LABEL}.")
    print()

    provenance = preflight(args.dry_run)
    print("Preflight OK.")
    print(f"  git branch:                 {provenance['git_branch']}")
    print(f"  git HEAD:                   {provenance['git_head']}")
    print(f"  rabit_kv2.py SHA:           {provenance['rabit_kv2_sha256']} (LF; matches committed content)")
    print(f"  multilingual_ppl.py SHA:    {provenance['multilingual_ppl_sha256']}")
    print(f"  continuation_ppl.py SHA:    {provenance['reference_continuation_ppl_sha256']}")
    print(f"  runner SHA:                 {provenance['runner_script_sha256']}")
    print(f"  canonical summary.json SHA: {provenance['canonical_summary_sha256']}")
    print(f"  canonical multilingual log: {provenance['canonical_multilingual_log_sha256']}")
    print("  presets/quantization code:  byte-identical to continuation_ppl.py")
    print("  canonical bf16/rabit2 reference: verified against summary.json and canonical log")
    if provenance["uncommitted_experiment_files"]:
        print("  WARNING (dry-run only): uncommitted experiment files -- a real run")
        print("  would refuse to start until these are committed:")
        for line in provenance["uncommitted_experiment_files"].splitlines():
            print(f"    {line}")
    print()

    command = build_command()
    print("Would execute 1 Modal run:")
    print("  " + " ".join(command))
    print(f"  -> log: {rel(LOG)}")

    if args.dry_run:
        print("\n--dry-run: no Modal/GPU commands executed, no files written.")
        return 0

    manifest = {
        "experiment": "Experiment 2 -- multilingual operating-point quality-compression sweep",
        "plan_reference": "docs/MLSYS_EXPERIMENT_PLAN.md",
        "model": MODEL,
        "languages": list(LANGUAGES),
        "methods": METHODS,
        "presets": EXPECTED_PRESETS,
        "settings": {
            "dataset": "wikimedia/wikipedia 20231101.{zh,es}",
            "dataset_revision": "cf584d1dc131caa92a5cb910f41a8b7591b12732",
            "shuffle_seed": 20260804,
            "shuffle_buffer": 1000,
            "context_tokens": 1024,
            "eval_tokens": 128,
            "samples_per_language": 8,
        },
        "gpu": "NVIDIA H100 80GB HBM3 (Modal)",
        "kv_memory_label": KV_LABEL,
        "regression_scope": "bf16/rabit2 only (canonical multilingual run covered bf16, rabit2); rabit8/rabit4/rabit3 are new evidence",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "completed_utc": None,
        "status": "running",
        "provenance": provenance,
        "regression_tolerances": {
            "ppl_relative": PPL_RELATIVE_TOLERANCE,
            "avg_kv_mb_relative": KV_MB_RELATIVE_TOLERANCE,
        },
        "runs": [],
    }
    write_manifest(manifest)

    try:
        return run(manifest, command)
    except (Exception, KeyboardInterrupt) as exc:
        current_row = manifest["runs"][-1] if manifest["runs"] else None
        if current_row is not None and current_row.get("status") != "running":
            current_row = None
        record_runner_failure(manifest, current_row, exc)
        check_error = manifest.get("protected_paths_check_error")
        protected = (
            f"\nPROTECTED-PATH CHECK ALSO FAILED: {check_error['type']}: {check_error['message']}"
            if check_error
            else ""
        )
        raise SystemExit(
            f"\nEXPERIMENT 2 RUNNER FAILED: {type(exc).__name__}: {exc}{protected}\n"
            f"Manifest marked failed; partial log preserved under {rel(OUT_DIR)}/. Not retried."
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
