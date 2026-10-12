"""
MLSys 2027 Experiment 14 -- SELF-CONTAINED Modal app for the NON-EVIDENCE shape-gate numerical diagnosis (Attempt 4;
one H100; no model, no engine, no download). CAPTURE-ONLY fix after Attempt 3, whose diagnostic completed but whose
single-line full summary was truncated at 61,515 characters by the Modal log stream.

Result transport: the remote function runs the UNCHANGED diagnostic (exp14_shape_gate_numdiag.py, byte-identical to
e6361eb), extracts the exact EXP14_NUMDIAG_SUMMARY JSON text from the subprocess stdout INSIDE the container, and
RETURNS it through the Modal function-call result (RPC), never through the log stream. The local entrypoint writes that
text to the LOCAL file named by EXP14_NUMDIAG_RESULT_PATH, re-reads and parses it, and prints only a short
EXP14_NUMDIAG_CAPTURE={completed, rows, bytes, sha256} line.

No dependency on sibling Exp14 Python modules: the frozen Exp14 `image = (...)` expression is copied TEXTUALLY from
exp14_deployment_modal.py (the offline test proves equality); the only files appended are the unchanged diagnostic and
the frozen exp14_shape_gate.py it imports. Launched only by benchmarks/mlsys2027/run_exp14_shape_gate_numdiag.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

BASE_COMMIT = "f329ce405b12623fb8b1cf1830f12e5a712523be"
SNAP = Path(os.environ.get("EXP14_VLLM_SNAPSHOT", "/nonexistent/EXP14_VLLM_SNAPSHOT-not-set.zip"))
DIAG_LOCAL = Path(__file__).resolve().parent / "exp14_shape_gate_numdiag.py"
DIAG_REMOTE = "/opt/exp14/exp14_shape_gate_numdiag.py"
SHAPE_GATE_LOCAL = Path(__file__).resolve().parent / "exp14_shape_gate.py"
SHAPE_GATE_REMOTE = "/opt/exp14/exp14_shape_gate.py"
FUNCTION_TIMEOUT_S = 2700  # Modal-side backstop (45 min); the local runner enforces the hard wall-clock limit
DIAG_SUBPROCESS_TIMEOUT_S = 2400
SUMMARY_PREFIX = "EXP14_NUMDIAG_SUMMARY="
RESULT_PATH_ENV = "EXP14_NUMDIAG_RESULT_PATH"

if modal.is_local() and not SNAP.is_file():
    raise RuntimeError(
        "EXP14_VLLM_SNAPSHOT is not set to an existing snapshot zip. Launch via "
        "benchmarks/mlsys2027/run_exp14_shape_gate_numdiag.py."
    )

app = modal.App("rabit-kv-mlsys2027-exp14-shape-gate-numdiag")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "curl", "libnuma1", "libnuma-dev")
    .pip_install(
        "pip>=25", "cmake>=3.26.1", "ninja", "packaging>=24.2",
        "setuptools>=77.0.3,<81.0.0", "setuptools-scm>=8.0",
        "setuptools-rust>=1.9.0", "wheel", "jinja2",
    )
    .run_commands(
        "python -m pip install torch==2.11.0 torchvision==0.26.0 "
        "torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130"
    )
    .add_local_file(
        str(SNAP), "/tmp/rabit2_final_fast_decode_append_snapshot.zip", copy=True
    )
    .run_commands(
        "rm -rf /root/vllm-kvquant && mkdir -p /root/vllm-kvquant && "
        "python -m zipfile -e /tmp/rabit2_final_fast_decode_append_snapshot.zip "
        "/root/vllm-kvquant"
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "VLLM_TARGET_DEVICE": "cuda",
        "VLLM_USE_PRECOMPILED": "1",
        "VLLM_PRECOMPILED_WHEEL_COMMIT": BASE_COMMIT,
        "VLLM_PRECOMPILED_WHEEL_VARIANT": "cu130",
        "VLLM_MAIN_CUDA_VERSION": "13.0",
        "VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX": "1",
        "SETUPTOOLS_SCM_PRETEND_VERSION": "0.10.0+kvquant",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM": "0.10.0+kvquant",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "VLLM_USE_V1": "1",
    })
    .run_commands(
        "cd /root/vllm-kvquant && python use_existing_torch.py --prefix",
        "cd /root/vllm-kvquant && rm -rf build dist *.egg-info /tmp/vllm.log && "
        "(python -m pip install -e . --no-build-isolation > /tmp/vllm.log 2>&1 || "
        "(tail -n 250 /tmp/vllm.log; exit 1))",
    )
    .pip_install("pytest", "modelscope")
)

image = (
    image.add_local_file(str(DIAG_LOCAL), DIAG_REMOTE, copy=True)
    .add_local_file(str(SHAPE_GATE_LOCAL), SHAPE_GATE_REMOTE, copy=True)
)


def split_summary(stdout: str) -> tuple[str | None, str]:
    """(exact JSON text after SUMMARY_PREFIX on its line, or None; stdout WITHOUT that line)."""
    summary, rest = None, []
    for line in stdout.splitlines():
        if summary is None and line.startswith(SUMMARY_PREFIX):
            summary = line[len(SUMMARY_PREFIX):].rstrip("\r")
        else:
            rest.append(line)
    return summary, "\n".join(rest)


def count_rows(summary: dict) -> int:
    return sum(len(rep["checkpoints"]) for g in summary.get("geometries", {}).values()
               for rep in g.get("replays", {}).values())


def write_capture(summary_text: str, path: str) -> dict:
    """Write the returned summary text to a LOCAL file (UTF-8), re-read it, parse it and hash it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(summary_text.encode("utf-8"))
    data = p.read_bytes()
    parsed = json.loads(data.decode("utf-8"))
    return {"completed": parsed.get("completed") is True, "rows": count_rows(parsed), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "roundtrip_identical": data.decode("utf-8") == summary_text}


@app.function(image=image, gpu="H100", timeout=FUNCTION_TIMEOUT_S)
def numdiag() -> dict:
    p = subprocess.run([sys.executable, DIAG_REMOTE], text=True, capture_output=True, timeout=DIAG_SUBPROCESS_TIMEOUT_S)
    summary_text, rest = split_summary(p.stdout)
    print(rest[-20000:], flush=True)  # short lines only; the full summary is RETURNED, not logged
    if p.stderr:
        print(p.stderr[-8000:], flush=True)
    return {"returncode": p.returncode, "summary_text": summary_text,
            "summary_chars": None if summary_text is None else len(summary_text)}


@app.local_entrypoint()
def main():
    res = numdiag.remote()
    meta = {"completed": False, "remote_returncode": res["returncode"], "summary_returned": res["summary_text"] is not None}
    if res["summary_text"] is not None:
        meta.update(write_capture(res["summary_text"], os.environ[RESULT_PATH_ENV]))
        meta["remote_returncode"] = res["returncode"]
        meta["summary_returned"] = True
    print("EXP14_NUMDIAG_CAPTURE=" + json.dumps(meta, sort_keys=True), flush=True)
    if res["returncode"] or not meta.get("completed") or not meta.get("roundtrip_identical"):
        raise SystemExit(1)
