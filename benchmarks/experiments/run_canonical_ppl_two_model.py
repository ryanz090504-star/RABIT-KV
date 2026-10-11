"""
Orchestrator: ONE registered two-model canonical continuation-PPL validation attempt (Llama -> Qwen), with no human /
model-dependent decision between the two runs. It only sequences benchmarks/mlsys2027/run_canonical_ppl.py (unchanged
protocol, gates and statistics); it contains no scoring, no quantization, no threshold and no retry.

  0. refuses to start if ANY output of this attempt number already exists (a registered two-model attempt is never
     partially rerun; one model's result is never kept while the other is rerun)
  1. run_canonical_ppl.py --execute --model llama3_1_8b --attempt N     (output captured, NOT printed)
  2. the Llama attempt directory is committed and pushed UNINSPECTED (the per-model preflight requires a clean, pushed
     tree; this is the only way to satisfy it without a human step)
  3. only if the Llama run passed its VALIDITY gates (runner exit code 0 -- the gates never depend on the RABIT
     result): run_canonical_ppl.py --execute --model qwen2_5_7b --attempt N
  4. two_model_attempt_<N>/record.json is written: the attempt is VALID only if BOTH runs are valid
  5. both runner outputs are printed only now.
If either model is invalid, the complete two-model attempt is INVALID (archive; no selective rerun).

Usage:  python benchmarks/mlsys2027/run_canonical_ppl_two_model.py --execute --attempt 1     (ONLY when authorized)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNNER = HERE / "run_canonical_ppl.py"
OUT_BASE = "results/mlsys2027/canonical_quality_v2/continuation_ppl"
ORDER = ["llama3_1_8b", "qwen2_5_7b"]
RULE = ("one registered attempt = both models; valid only if BOTH runs pass every validity gate; if either fails the "
        "whole attempt is invalid and archived; neither result is accepted evidence on its own; no selective rerun")


def command(model: str, attempt: int) -> list:
    return [sys.executable, str(RUNNER), "--execute", "--model", model, "--attempt", str(attempt)]


def run_model(model: str, attempt: int) -> subprocess.CompletedProcess:
    return subprocess.run(command(model, attempt), cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def archive_uninspected(model: str, attempt: int) -> None:
    path = f"{OUT_BASE}/{model}/attempt_{attempt}"
    msg = (f"Archive canonical PPL two-model attempt {attempt}: {model} raw outputs (UNINSPECTED; committed by the "
           "orchestrator before the second model runs)\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
    for cmd in (["git", "add", "--", path], ["git", "commit", "-q", "-m", msg], ["git", "push", "-q"]):
        subprocess.run(cmd, cwd=ROOT, check=True)


def orchestrate(attempt: int, run=run_model, archive=archive_uninspected, root: Path = ROOT, out=print) -> int:
    base = root / OUT_BASE
    record_dir = base / f"two_model_attempt_{attempt}"
    existing = [p for p in [*(base / m / f"attempt_{attempt}" for m in ORDER), record_dir] if p.exists()]
    if existing:
        raise SystemExit(f"attempt {attempt} already has outputs ({[p.name for p in existing]}); a registered "
                         "two-model attempt is never partially rerun -- use a new attempt number for BOTH models")
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    first = run(ORDER[0], attempt)
    if (base / ORDER[0] / f"attempt_{attempt}").is_dir():
        archive(ORDER[0], attempt)
    second = run(ORDER[1], attempt) if first.returncode == 0 else None
    codes = {ORDER[0]: first.returncode, ORDER[1]: None if second is None else second.returncode}
    valid = all(c == 0 for c in codes.values())
    record_dir.mkdir(parents=True, exist_ok=False)
    (record_dir / "record.json").write_text(json.dumps({
        "kind": "canonical-quality-v2 continuation PPL: ONE registered two-model validation attempt",
        "attempt": attempt, "order": ORDER, "started_utc": started,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "runner_exit_codes": codes,
        "models_not_run": [m for m, c in codes.items() if c is None], "rule": RULE, "valid": valid,
        "status": "valid (awaiting review; not accepted evidence until reviewed)" if valid
        else "INVALID two-model attempt (archive BOTH models; never pool; no selective rerun)"}, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    for model, p in zip(ORDER, (first, second)):
        out(f"===== {model} =====")
        if p is None:
            out("NOT RUN (the first model failed a validity gate; the two-model attempt is INVALID)")
        else:
            out(f"runner exit code {p.returncode}\n{p.stdout}\n{p.stderr[-4000:]}")
    out(f"TWO_MODEL_ATTEMPT_{attempt}_VALID={valid}")
    return 0 if valid else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--execute", action="store_true", help="launch the GPU jobs (ONLY when explicitly authorized)")
    a = ap.parse_args(argv)
    if not a.execute:
        raise SystemExit("--execute is required; nothing executed")
    return orchestrate(a.attempt)


if __name__ == "__main__":
    raise SystemExit(main())
