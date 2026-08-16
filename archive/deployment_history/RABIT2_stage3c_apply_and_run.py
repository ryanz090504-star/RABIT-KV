"""Patch the actual vLLM V2 GPU model runner for RABIT-2 Stage 3C and rerun H100 lifecycle smoke.

This fixer assumes the previous Stage 3C installer has already installed:
- Stage 3C registry functions in rabit_kv2.py
- Stage 3C multi-request/chunked-prefill backend in triton_attn.py
- Stage 3C quantization tests

The previous failure was caused by patching the legacy
vllm/v1/worker/gpu_model_runner.py while the engine was actually using
vllm/v1/worker/gpu/model_runner.py (V2 Model Runner).
"""
from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
VLLM_ROOT = PROJECT_ROOT / "vllm-kvquant"

RABIT_PATH = VLLM_ROOT / "vllm/v1/attention/ops/rabit_kv2.py"
TRITON_ATTN_PATH = VLLM_ROOT / "vllm/v1/attention/backends/triton_attn.py"
V2_RUNNER_PATH = VLLM_ROOT / "vllm/v1/worker/gpu/model_runner.py"
TEST_PATH = VLLM_ROOT / "tests/quantization/test_rabit_kv2_stage3c.py"

BACKUP_ROOT = PROJECT_ROOT / "rabit2_stage3c_fixed_v2_backup"
SNAPSHOT_PATH = Path(tempfile.gettempdir()) / "rabit2_stage3c_fixed_v2_vllm_snapshot.zip"
RUNNER_PATH = Path(tempfile.gettempdir()) / "modal_rabit2_stage3c_fixed_v2_test.py"
LOG_PATH = PROJECT_ROOT / "rabit2_stage3c_fixed_v2.log"

V2_MARK = "# === RABIT2_STAGE3C_V2_RUNNER_PATCHED ==="
RUNNER_Z = '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'

IGNORED_DIR_NAMES = {
    ".git", ".github", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "build", "dist", ".venv", "venv", "node_modules",
}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".log"}


def _decode(value: str) -> str:
    return zlib.decompress(base64.b64decode(value.encode("ascii"))).decode("utf-8")


def _backup(path: Path) -> None:
    if not path.exists():
        return
    rel = path.relative_to(PROJECT_ROOT)
    dst = BACKUP_ROOT / rel
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


def _patch_v2_runner(text: str) -> str:
    if V2_MARK in text:
        return text

    init_old = """        self.observability_config = vllm_config.observability_config

        self.device = device
"""
    init_new = """        self.observability_config = vllm_config.observability_config

        if self.cache_config.cache_dtype == "rabit_kv2":
            if self.cache_config.enable_prefix_caching:
                raise NotImplementedError(
                    "RABIT-2 exact R4 sidecar is not compatible with vLLM prefix "
                    "caching yet; use enable_prefix_caching=False."
                )
            if self.speculative_config is not None:
                raise NotImplementedError(
                    "RABIT-2 Stage 3C does not yet support speculative decoding."
                )

        self.device = device
"""
    if init_old not in text:
        raise RuntimeError("Could not locate V2 __init__ guard patch point")
    text = text.replace(init_old, init_new, 1)

    finish_old = """        if preempted_req_ids:
            finished_req_ids = finished_req_ids.union(preempted_req_ids)
        for req_id in finished_req_ids:
            self._remove_request(req_id)
"""
    finish_new = """        if preempted_req_ids:
            finished_req_ids = finished_req_ids.union(preempted_req_ids)

        if self.cache_config.cache_dtype == "rabit_kv2" and finished_req_ids:
            # Sidecars are request-owned GPU-worker state. Release them before
            # vLLM can recycle the request's physical KV block IDs.
            from vllm.v1.attention.ops.rabit_kv2 import rabit2_finish_requests

            rabit2_finish_requests(finished_req_ids)

        for req_id in finished_req_ids:
            self._remove_request(req_id)
"""
    if finish_old not in text:
        raise RuntimeError("Could not locate V2 finished/preempted request patch point")
    text = text.replace(finish_old, finish_new, 1)

    batch_old = """            input_batch = self.prepare_inputs(scheduler_output, batch_desc)
            block_tables, slot_mappings = self.prepare_attn(input_batch)
"""
    batch_new = """            input_batch = self.prepare_inputs(scheduler_output, batch_desc)

            if self.cache_config.cache_dtype == "rabit_kv2":
                # V2 prepare_inputs() establishes the exact request ordering
                # consumed by attention. Publish that ordering plus each request's
                # pre-step context length before attention metadata is built.
                from vllm.v1.attention.ops.rabit_kv2 import rabit2_set_active_batch

                rabit2_set_active_batch(
                    input_batch.req_ids,
                    input_batch.num_scheduled_tokens,
                    input_batch.num_computed_tokens_np,
                )

            block_tables, slot_mappings = self.prepare_attn(input_batch)
"""
    if batch_old not in text:
        raise RuntimeError("Could not locate V2 real-batch prepare_inputs patch point")
    text = text.replace(batch_old, batch_new, 1)

    anchor = "import functools\n"
    if anchor not in text:
        raise RuntimeError("Could not install V2 runner marker")
    text = text.replace(anchor, anchor + V2_MARK + "\n", 1)
    return text


def _install() -> None:
    required = [RABIT_PATH, TRITON_ATTN_PATH, V2_RUNNER_PATH, TEST_PATH]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required Stage 3C sources: {missing}")

    rabit = RABIT_PATH.read_text(encoding="utf-8")
    attn = TRITON_ATTN_PATH.read_text(encoding="utf-8")
    if "RABIT2_STAGE3C_LIFECYCLE_BEGIN" not in rabit:
        raise RuntimeError(
            "Current rabit_kv2.py does not contain Stage 3C lifecycle registry. "
            "Run the original Stage 3C installer first."
        )
    if "Stage-3C multi-request + chunked-prefill" not in attn:
        raise RuntimeError(
            "Current triton_attn.py does not contain the Stage 3C backend. "
            "Run the original Stage 3C installer first."
        )

    _backup(V2_RUNNER_PATH)
    text = V2_RUNNER_PATH.read_text(encoding="utf-8")
    text = _patch_v2_runner(text)
    compile(text, str(V2_RUNNER_PATH), "exec")
    V2_RUNNER_PATH.write_text(text, encoding="utf-8")
    print(f"Installed Stage3C V2 scheduler bridge: {V2_RUNNER_PATH}")


def _preflight() -> None:
    checks = {
        RABIT_PATH: [
            "RABIT2_STAGE3C_LIFECYCLE_BEGIN",
            "rabit2_set_active_batch",
            "rabit2_finish_requests",
        ],
        TRITON_ATTN_PATH: [
            "Stage-3C multi-request + chunked-prefill",
            "rabit2_get_active_batch",
            "RABIT2_STAGE3C_CHUNKED_PREFILL_ACTIVE",
        ],
        V2_RUNNER_PATH: [
            "RABIT2_STAGE3C_V2_RUNNER_PATCHED",
            "rabit2_set_active_batch",
            "rabit2_finish_requests",
            "input_batch.num_computed_tokens_np",
        ],
    }
    for path, needles in checks.items():
        text = path.read_text(encoding="utf-8")
        miss = [x for x in needles if x not in text]
        if miss:
            raise RuntimeError(f"Stage3C-fixed preflight {path} missing {miss}")
        compile(text, str(path), "exec")
    print("Stage 3C fixed-V2 local source preflight: PASSED")


def _include(path: Path) -> bool:
    rel = path.relative_to(VLLM_ROOT)
    return (
        path.is_file()
        and not any(part in IGNORED_DIR_NAMES for part in rel.parts)
        and path.suffix.lower() not in IGNORED_SUFFIXES
        and not path.name.endswith(".egg-info")
    )


def _make_snapshot() -> None:
    tmp = SNAPSHOT_PATH.with_suffix(".zip.tmp")
    for path in (tmp, SNAPSHOT_PATH):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    count = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(VLLM_ROOT.rglob("*")):
            if _include(path):
                zf.write(path, path.relative_to(VLLM_ROOT).as_posix())
                count += 1
    os.replace(tmp, SNAPSHOT_PATH)
    print(
        f"Created frozen vLLM snapshot: {SNAPSHOT_PATH} "
        f"({count} files, {SNAPSHOT_PATH.stat().st_size / 1024**2:.1f} MB)"
    )


def _run_modal() -> int:
    RUNNER_PATH.write_text(_decode(RUNNER_Z), encoding="utf-8")
    time.sleep(2)

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    command = [sys.executable, "-m", "modal", "run", str(RUNNER_PATH)]
    print("Running:", " ".join(command))

    with LOG_PATH.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return process.wait()


def main() -> None:
    print("RABIT-2 Stage 3C FIXED — patch actual V2 GPU model runner")
    print(f"Project root: {PROJECT_ROOT}")
    _install()
    _preflight()
    _make_snapshot()
    code = _run_modal()
    if code != 0:
        raise SystemExit(code)
    print(f"Log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()
