from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
QUALITY_DIR = ROOT / "benchmarks" / "quality"
OUT_DIR = ROOT / "results" / "reproduced" / "quality"
MANIFEST = OUT_DIR / "manifest.json"

EXPECTED_RABIT_SHA256 = "108ba8afaac862bc4e070d2dc812493aaeae4e458837c4bda2ac703e72a57524"

RUNS = [
    {
        "name": "continuation_ppl",
        "script": "continuation_ppl.py",
        "args": [
            "--context-tokens", "1024",
            "--eval-tokens", "128",
            "--samples", "8",
            "--methods", "bf16,rabit2",
        ],
        "purpose": "WikiText-2 teacher-forced continuation PPL, final BF16 vs RABIT-KV",
    },
    {
        "name": "multilingual_ppl",
        "script": "multilingual_ppl.py",
        "args": [
            "--languages", "zh,es",
            "--context-tokens", "1024",
            "--eval-tokens", "128",
            "--samples", "8",
            "--methods", "bf16,rabit2",
            "--dataset-revision", "cf584d1dc131caa92a5cb910f41a8b7591b12732",
            "--shuffle-seed", "20260804",
            "--shuffle-buffer", "1000",
        ],
        "purpose": "Pinned multilingual continuation PPL on Chinese and Spanish Wikipedia",
    },
    {
        "name": "niah",
        "script": "niah.py",
        "args": [
            "--context-lengths", "4096,8192,16384",
            "--needle-depths", "0.1,0.25,0.5,0.75,0.9",
            "--max-new-tokens", "16",
            "--methods", "bf16,rabit2",
        ],
        "purpose": "Needle-in-a-Haystack: 4K/8K/16K x five depths",
    },
    {
        "name": "passage_retrieval",
        "script": "passage_retrieval.py",
        "args": [
            "--sample-start", "0",
            "--samples", "10",
            "--max-input-tokens", "16384",
            "--max-new-tokens", "32",
            "--methods", "bf16,rabit2",
        ],
        "purpose": "LongBench passage_retrieval_en full 10-sample final slice",
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
            "--methods", "bf16,rabit2",
        ],
        "purpose": "LongBench-E HotpotQA 8K+ full 20-sample final slice",
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
            "--methods", "bf16,rabit2",
        ],
        "purpose": "LongBench-E Qasper 8K+ full 24-example bucket",
    },
]


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
        raise RuntimeError(
            f"git {' '.join(args)} failed:\n{p.stdout}\n{p.stderr}"
        )
    return p.stdout.strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight() -> dict:
    branch = run_git("branch", "--show-current")

    status = run_git("status", "--short")
    if status:
        raise RuntimeError(
            "Working tree must be clean before running the quality suite.\n"
            + status
        )

    head = run_git("rev-parse", "HEAD")
    rabit = ROOT / "vllm-kvquant" / "vllm" / "v1" / "attention" / "ops" / "rabit_kv2.py"
    if not rabit.is_file():
        raise FileNotFoundError(rabit)

    rabit_sha = sha256(rabit)
    if rabit_sha != EXPECTED_RABIT_SHA256:
        raise RuntimeError(
            "Frozen performance source SHA mismatch.\n"
            f"expected={EXPECTED_RABIT_SHA256}\n"
            f"actual={rabit_sha}"
        )

    script_hashes = {}
    for spec in RUNS:
        path = QUALITY_DIR / spec["script"]
        if not path.is_file():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        if '"name": "2b META8g64 K3V2 G32 R4"' not in text:
            raise RuntimeError(
                f"{path} does not expose the frozen RABIT-2 META8g64 policy marker."
            )
        script_hashes[spec["script"]] = sha256(path)

    return {
        "git_branch": branch,
        "git_head": head,
        "rabit_kv2_sha256": rabit_sha,
        "quality_script_sha256": script_hashes,
    }


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


def write_manifest(manifest: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    print("RABIT-KV QUALITY SUITE")
    print("Policy: K3 / V2 / G32 / R4 / META8g64")
    print("Comparison: BF16 vs frozen final RABIT-KV only")
    print("Reproduction outputs are written separately from canonical results.")
    print()

    provenance = preflight()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "suite": "RABIT-KV quality suite",
        "policy": "K3/V2/G32/R4/META8g64",
        "model": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
        "methods": ["bf16", "rabit2"],
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "completed_utc": None,
        "status": "running",
        "provenance": provenance,
        "runs": [],
        "note": (
            "Reproduction run for the frozen RABIT-KV operating point. "
            "Canonical committed results remain unchanged."
        ),
    }
    write_manifest(manifest)

    for index, spec in enumerate(RUNS, start=1):
        script = QUALITY_DIR / spec["script"]
        log_path = OUT_DIR / f"{spec['name']}.log"
        cmd = [
            sys.executable,
            "-m",
            "modal",
            "run",
            str(script),
            *spec["args"],
        ]

        row = {
            "index": index,
            "name": spec["name"],
            "purpose": spec["purpose"],
            "script": str(script.relative_to(ROOT)).replace("\\", "/"),
            "command": cmd,
            "log": str(log_path.relative_to(ROOT)).replace("\\", "/"),
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "completed_utc": None,
            "returncode": None,
            "status": "running",
        }
        manifest["runs"].append(row)
        write_manifest(manifest)

        code = stream_command(cmd, log_path)

        row["returncode"] = code
        row["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        row["status"] = "passed" if code == 0 else "failed"
        write_manifest(manifest)

        if code != 0:
            manifest["status"] = "failed"
            manifest["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            write_manifest(manifest)
            raise SystemExit(
                f"\nQUALITY RUN STOPPED at {spec['name']} (exit={code}). "
                "Completed logs were preserved. Do not tune parameters."
            )

    manifest["status"] = "passed"
    manifest["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_manifest(manifest)

    print("\n" + "=" * 118)
    print("QUALITY SUITE: ALL RUNS PASSED")
    print(f"Manifest: {MANIFEST}")
    print(f"Raw logs:  {OUT_DIR}")
    print("Quality suite complete. Results written to results/reproduced/quality/.")
    print("=" * 118)


if __name__ == "__main__":
    main()






