"""vLLM deploy benchmark wrapper.

This module is the bridge between Quantization_v2's policy framework and a
local vLLM fork.  It starts a real OpenAI-compatible ``vllm serve`` process,
runs ``vllm bench serve`` against that server, and parses the saved benchmark
JSON.  Results emitted from here are the only latency rows that should be
labelled ``deploy_latency``.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


@dataclass
class VLLMDeployResult:
    """Metrics from a vLLM serving benchmark run."""

    model: str
    policy_name: str
    policy_signature: str
    nbits: int | None
    kv_cache_dtype: str
    hardware: str | None = None
    vllm_commit: str | None = None

    # Serving metrics
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    itl_ms: float | None = None
    tokens_per_second: float | None = None
    request_throughput: float | None = None
    end_to_end_ms: float | None = None

    # Breakdown
    prefill_latency_ms: float | None = None
    decode_latency_ms: float | None = None

    # Config
    input_len: int = 1024
    output_len: int = 128
    num_prompts: int = 100
    request_rate: float | None = None

    kv_source: str = "vllm_packed_kv_cache"
    quant_kernel: str | None = None
    memory_breakdown: dict[str, Any] = field(default_factory=dict)
    evidence_label: str = "deploy_latency"
    metadata: dict[str, Any] = field(default_factory=dict)


def run_vllm_benchmark(
    model_name: str,
    policy_name: str,
    nbits: int,
    *,
    input_len: int = 1024,
    output_len: int = 128,
    num_prompts: int = 100,
    request_rate: float | None = None,
    max_concurrency: int | None = None,
    vllm_root: str | None = None,
    kv_cache_dtype: str | None = None,
    gpu_memory_utilization: float = 0.85,
    server_start_timeout_s: float = 600.0,
    benchmark_timeout_s: float = 1200.0,
) -> VLLMDeployResult:
    """Run vLLM serving benchmark with quantized KV cache.

    This is the primary entry point for deploy_latency evidence. It requires:
    1. A working vLLM fork with kvquant_k3 support
    2. CUDA GPU
    3. A built/installed vLLM runtime whose Python path points at the fork
    """
    from kvquant.policies import build_policy, policy_signature
    from kvquant.vllm_plugin.config import KVQuantConfig

    dtype = kv_cache_dtype or _kv_dtype(nbits)
    root = Path(vllm_root).expanduser() if vllm_root else None
    commit = _git_commit(root) if root else None
    policy = build_policy(policy_name, nbits=nbits)
    sig = policy_signature(policy)
    layout = KVQuantConfig(policy_name=policy_name, nbits=nbits).memory_layout()

    if root is None or not (root / "vllm").is_dir():
        raise ValueError(
            "vLLM deploy latency requires a local vLLM fork. "
            f"Set --vllm-root to a checkout containing vllm/. Current root: {root}"
        )
    if dtype != "kvquant_k3":
        raise NotImplementedError(
            "This vLLM deploy path is wired only for kvquant_k3. "
            f"Got kv_cache_dtype={dtype!r}; policy={policy_name}; nbits={nbits}."
        )

    hardware = _hardware_name()
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = _vllm_env(root)

    with tempfile.TemporaryDirectory(prefix="kvquant-vllm-bench-") as tmp:
        tmp_dir = Path(tmp)
        server_log = tmp_dir / "vllm_server.log"
        bench_json = tmp_dir / "bench_result.json"
        server_cmd = _serve_cmd(
            model_name=model_name,
            dtype=dtype,
            port=port,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=input_len + output_len,
        )
        bench_cmd = _bench_cmd(
            model_name=model_name,
            base_url=base_url,
            result_file=bench_json,
            input_len=input_len,
            output_len=output_len,
            num_prompts=num_prompts,
            request_rate=request_rate,
            max_concurrency=max_concurrency,
            policy_name=policy_name,
            nbits=nbits,
            dtype=dtype,
        )

        with server_log.open("w", encoding="utf-8") as log_f:
            server = subprocess.Popen(
                server_cmd,
                cwd=str(root),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                ready_error = _wait_for_server(base_url, server, server_start_timeout_s)
                if ready_error is not None:
                    return _error_result(
                        model_name=model_name,
                        policy_name=policy_name,
                        policy_signature=sig,
                        nbits=nbits,
                        dtype=dtype,
                        hardware=hardware,
                        commit=commit,
                        layout=layout,
                        error=ready_error,
                        root=root,
                        server_cmd=server_cmd,
                        bench_cmd=bench_cmd,
                        server_log=server_log,
                    )

                try:
                    bench_proc = subprocess.run(
                        bench_cmd,
                        cwd=str(root),
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=benchmark_timeout_s,
                    )
                except subprocess.TimeoutExpired:
                    return _error_result(
                        model_name=model_name,
                        policy_name=policy_name,
                        policy_signature=sig,
                        nbits=nbits,
                        dtype=dtype,
                        hardware=hardware,
                        commit=commit,
                        layout=layout,
                        error=f"vllm bench serve timed out >{benchmark_timeout_s}s",
                        root=root,
                        server_cmd=server_cmd,
                        bench_cmd=bench_cmd,
                        server_log=server_log,
                    )

                if bench_proc.returncode != 0:
                    return _error_result(
                        model_name=model_name,
                        policy_name=policy_name,
                        policy_signature=sig,
                        nbits=nbits,
                        dtype=dtype,
                        hardware=hardware,
                        commit=commit,
                        layout=layout,
                        error=f"vllm bench serve exited with code {bench_proc.returncode}",
                        root=root,
                        server_cmd=server_cmd,
                        bench_cmd=bench_cmd,
                        server_log=server_log,
                        stdout=bench_proc.stdout,
                        stderr=bench_proc.stderr,
                    )

                metrics = _load_bench_json(bench_json)
            finally:
                _terminate_process(server)

    return VLLMDeployResult(
        model=model_name,
        policy_name=policy_name,
        policy_signature=sig,
        nbits=nbits,
        kv_cache_dtype=dtype,
        hardware=hardware,
        vllm_commit=commit,
        ttft_ms=metrics.get("ttft_ms"),
        tpot_ms=metrics.get("tpot_ms"),
        itl_ms=metrics.get("itl_ms"),
        tokens_per_second=metrics.get("tokens_per_second"),
        request_throughput=metrics.get("request_throughput"),
        end_to_end_ms=metrics.get("end_to_end_ms"),
        input_len=input_len,
        output_len=output_len,
        num_prompts=num_prompts,
        request_rate=request_rate,
        quant_kernel="vllm_triton_kvquant_k3_decode",
        memory_breakdown=layout,
        metadata={
            "vllm_root": str(root),
            "server_command": _quote_cmd(server_cmd),
            "bench_command": _quote_cmd(bench_cmd),
            "base_url": base_url,
            "raw_metrics": metrics,
        },
    )


def _kv_dtype(nbits: int) -> str:
    if nbits >= 16:
        return "auto"
    return f"kvquant_k{nbits}"


def _vllm_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return env


def _serve_cmd(
    *,
    model_name: str,
    dtype: str,
    port: int,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--kv-cache-dtype",
        dtype,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--disable-log-requests",
    ]


def _bench_cmd(
    *,
    model_name: str,
    base_url: str,
    result_file: Path,
    input_len: int,
    output_len: int,
    num_prompts: int,
    request_rate: float | None,
    max_concurrency: int | None,
    policy_name: str,
    nbits: int,
    dtype: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        base_url,
        "--model",
        model_name,
        "--dataset-name",
        "random",
        "--input-len",
        str(input_len),
        "--output-len",
        str(output_len),
        "--num-prompts",
        str(num_prompts),
        "--disable-tqdm",
        "--save-result",
        "--result-dir",
        str(result_file.parent),
        "--result-filename",
        result_file.name,
        "--metadata",
        f"policy_name={policy_name}",
        f"nbits={nbits}",
        f"kv_cache_dtype={dtype}",
    ]
    if request_rate is not None:
        cmd.extend(["--request-rate", str(request_rate)])
    if max_concurrency is not None:
        cmd.extend(["--max-concurrency", str(max_concurrency)])
    return cmd


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(
    base_url: str,
    server: subprocess.Popen[str],
    timeout_s: float,
) -> str | None:
    deadline = time.monotonic() + timeout_s
    models_url = f"{base_url}/v1/models"
    last_error = ""
    while time.monotonic() < deadline:
        if server.poll() is not None:
            return f"vllm serve exited before readiness with code {server.returncode}"
        try:
            with urlopen(models_url, timeout=5) as response:
                if response.status == 200:
                    return None
                last_error = f"HTTP {response.status}"
        except URLError as exc:
            last_error = str(exc)
        except TimeoutError:
            last_error = "readiness request timed out"
        time.sleep(2.0)
    return f"vllm serve did not become ready within {timeout_s}s: {last_error}"


def _load_bench_json(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"vllm bench serve did not create {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return _parse_bench_data(data)


def _parse_bench_data(data: dict[str, Any]) -> dict[str, float]:
    def number(*keys: str) -> float | None:
        for key in keys:
            value = data.get(key)
            if isinstance(value, int | float):
                return float(value)
        return None

    metrics: dict[str, float] = {}
    mapping = {
        "ttft_ms": ("mean_ttft_ms",),
        "tpot_ms": ("mean_tpot_ms",),
        "itl_ms": ("mean_itl_ms",),
        "tokens_per_second": (
            "total_token_throughput",
            "output_throughput",
            "tokens_per_second",
        ),
        "request_throughput": ("request_throughput",),
        "end_to_end_ms": ("mean_e2el_ms",),
    }
    for out_key, source_keys in mapping.items():
        value = number(*source_keys)
        if value is not None:
            metrics[out_key] = value
    metrics["completed"] = float(data.get("completed", 0))
    metrics["failed"] = float(data.get("failed", 0))
    return metrics


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def _tail(path: Path, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[-limit:]


def _quote_cmd(cmd: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(part) for part in cmd)


def _error_result(
    *,
    model_name: str,
    policy_name: str,
    policy_signature: str,
    nbits: int,
    dtype: str,
    hardware: str | None,
    commit: str | None,
    layout: dict[str, Any],
    error: str,
    root: Path,
    server_cmd: list[str],
    bench_cmd: list[str],
    server_log: Path,
    stdout: str = "",
    stderr: str = "",
) -> VLLMDeployResult:
    return VLLMDeployResult(
        model=model_name,
        policy_name=policy_name,
        policy_signature=policy_signature,
        nbits=nbits,
        kv_cache_dtype=dtype,
        hardware=hardware,
        vllm_commit=commit,
        quant_kernel="vllm_triton_kvquant_k3_decode",
        memory_breakdown=layout,
        metadata={
            "error": error,
            "vllm_root": str(root),
            "server_command": _quote_cmd(server_cmd),
            "bench_command": _quote_cmd(bench_cmd),
            "server_log_tail": _tail(server_log),
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        },
    )


def _git_commit(root: Path | None) -> str | None:
    if root is None or not root.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _hardware_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"
