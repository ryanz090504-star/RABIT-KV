"""Latency benchmarks for KV cache quantization.

Two evidence levels are intentionally separated:

* ``kernel_latency_not_deploy_speedup`` measures a single packed-attention
  decode step. It is useful for kernel development, but is not serving speedup.
* ``deploy_latency`` is reserved for a real serving path where vLLM or another
  runtime stores low-bit KV cache and reads it inside attention.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LatencyResult:
    """Deploy latency result from vLLM or custom kernel measurement."""

    backend: str
    """'vllm_bench_serve' or 'triton_packed_attention'"""

    model: str
    policy_name: str
    policy_signature: str
    nbits: int | None
    kv_cache_dtype: str | None = None
    hardware: str | None = None
    vllm_commit: str | None = None
    quant_kernel: str | None = None
    memory_breakdown: dict[str, Any] = field(default_factory=dict)

    # vLLM metrics
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    itl_ms: float | None = None
    tokens_per_second: float | None = None
    request_throughput: float | None = None
    end_to_end_ms: float | None = None

    # Baseline comparison
    baseline_tpot_ms: float | None = None
    tpot_improvement: float | None = None
    throughput_improvement: float | None = None

    # ── Timing breakdown (useful for ablation studies in paper) ──
    quantize_time_ms: float | None = None
    """Time spent quantizing K/V tensors (write path)."""
    dequantize_time_ms: float | None = None
    """Time spent dequantizing K/V tensors (read path)."""
    attention_compute_ms: float | None = None
    """Pure attention computation excluding quantization overhead."""

    kv_source: str | None = None
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
) -> LatencyResult:
    """Run a real vLLM serving benchmark for the forked packed KV path."""
    from kvquant.policies import build_policy, policy_signature
    from kvquant.vllm_runner import run_vllm_benchmark as run_vllm_deploy
    from kvquant.vllm_plugin.config import KVQuantConfig

    cfg = KVQuantConfig(policy_name=policy_name, nbits=nbits)
    expected_dtype = cfg.kv_cache_dtype
    dtype = kv_cache_dtype or expected_dtype
    if dtype != expected_dtype:
        raise ValueError(
            f"kv_cache_dtype {dtype!r} does not match nbits={nbits}; expected {expected_dtype!r}"
        )
    if dtype != "kvquant_k3":
        raise NotImplementedError(
            "The vLLM deploy benchmark is currently wired only for kvquant_k3. "
            f"Got kv_cache_dtype={dtype!r}."
        )

    root = Path(vllm_root).expanduser() if vllm_root else None
    commit = _git_commit(root) if root else None
    policy = build_policy(policy_name, nbits=nbits)
    sig = policy_signature(policy)
    layout = cfg.memory_layout()
    deploy = run_vllm_deploy(
        model_name=model_name,
        policy_name=policy_name,
        nbits=nbits,
        input_len=input_len,
        output_len=output_len,
        num_prompts=num_prompts,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        vllm_root=str(root) if root else None,
        kv_cache_dtype=dtype,
    )

    return LatencyResult(
        backend="vllm_bench_serve",
        model=model_name,
        policy_name=policy_name,
        policy_signature=sig,
        nbits=nbits,
        kv_cache_dtype=dtype,
        hardware=deploy.hardware,
        vllm_commit=deploy.vllm_commit or commit,
        quant_kernel=deploy.quant_kernel,
        memory_breakdown=deploy.memory_breakdown or layout,
        ttft_ms=deploy.ttft_ms,
        tpot_ms=deploy.tpot_ms,
        itl_ms=deploy.itl_ms,
        tokens_per_second=deploy.tokens_per_second,
        request_throughput=deploy.request_throughput,
        end_to_end_ms=deploy.end_to_end_ms,
        kv_source=deploy.kv_source,
        evidence_label=deploy.evidence_label,
        metadata=deploy.metadata,
    )


def run_triton_kernel_benchmark(
    model_name: str,
    policy_name: str,
    nbits: int,
    *,
    batch_size: int = 1,
    heads: int = 4,
    seq_len: int = 255,
    head_dim: int = 64,
    iters: int = 100,
    warmup_iters: int = 10,
    dtype: str = "float16",
    snapshot_dir: str | None = None,
    snapshot_layer_idx: int | None = None,
    kv_cache_dtype: str | None = None,
) -> LatencyResult:
    """Run Triton packed-attention kernel benchmark.

    This is a kernel-level diagnostic. It does not become deploy evidence until
    ``run_packed_attention`` is backed by a true Triton packed-cache kernel.
    """
    import numpy as np

    from kvquant.policies import build_policy, policy_signature
    from kvquant.triton_kernel.packed_attention import (
        TRITON_BACKEND_NAME,
        packed_attention_backend_name,
        run_packed_attention,
    )
    from kvquant.types import QuantizationContext

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(torch, dtype)
    policy = build_policy(policy_name, nbits=nbits)
    kv_cache_dtype = kv_cache_dtype or f"kvquant_k{nbits}"

    # ── Get K/V tensors ──
    if snapshot_dir:
        from kvquant.snapshots import load_snapshots

        snaps = load_snapshots(snapshot_dir)
        target = next(
            (
                s
                for s in snaps
                if snapshot_layer_idx is None or s.layer_idx == snapshot_layer_idx
            ),
            snaps[0],
        )
        keys_np = target.keys.astype(np.float32)
        values_np = target.values.astype(np.float32)
        batch_size, heads, seq_len, head_dim = keys_np.shape
    else:
        rng = np.random.default_rng(7)
        keys_np = rng.normal(0, 1, (batch_size, heads, seq_len, head_dim)).astype(
            np.float32
        )
        values_np = rng.normal(0, 1, (batch_size, heads, seq_len, head_dim)).astype(
            np.float32
        )

    query_np = (
        np.random.default_rng(7)
        .normal(0, 1, (batch_size, heads, head_dim))
        .astype(np.float32)
    )

    # ── Quantize ──
    block = policy.quantize(
        keys_np, values_np, QuantizationContext(layer_idx=0)
    ).to_packed()
    if not hasattr(block.key, "packed") or not hasattr(block.value, "packed"):
        raise ValueError(
            f"policy {policy_name!r} does not produce a packed uniform KV block; "
            "mixed precision/residual policies need a deploy packing format first"
        )

    # ── Baseline: full-precision PyTorch attention ──
    query = torch.as_tensor(query_np, device=device, dtype=torch_dtype)
    b_key = torch.as_tensor(keys_np, device=device, dtype=torch_dtype)
    b_val = torch.as_tensor(values_np, device=device, dtype=torch_dtype)

    def baseline_step():
        import math

        s = torch.matmul(query.unsqueeze(-2), b_key.transpose(-1, -2)) / math.sqrt(
            head_dim
        )
        return torch.matmul(torch.softmax(s, dim=-1), b_val).squeeze(-2)

    key_packed = torch.as_tensor(block.key.packed, device=device, dtype=torch.uint8)
    value_packed = torch.as_tensor(block.value.packed, device=device, dtype=torch.uint8)
    key_min = torch.as_tensor(block.key.minimum, device=device, dtype=torch.float32)
    key_scale = torch.as_tensor(block.key.scale, device=device, dtype=torch.float32)
    value_min = torch.as_tensor(block.value.minimum, device=device, dtype=torch.float32)
    value_scale = torch.as_tensor(block.value.scale, device=device, dtype=torch.float32)

    def quantized_step():
        import math

        return run_packed_attention(
            query=query,
            key_packed=key_packed,
            value_packed=value_packed,
            key_min=key_min,
            key_scale=key_scale,
            value_min=value_min,
            value_scale=value_scale,
            nbits=nbits,
            scale=1.0 / math.sqrt(head_dim),
            key_shape=tuple(keys_np.shape),
            value_shape=tuple(values_np.shape),
        )

    # ── Warmup + benchmark ──
    for _ in range(warmup_iters):
        baseline_step()
        quantized_step()
    _sync_if_cuda()

    baseline_samples = _measure_step_ms(baseline_step, iters)
    quantized_samples = _measure_step_ms(quantized_step, iters)
    baseline_avg = float(np.mean(baseline_samples))
    quantized_avg = float(np.mean(quantized_samples))

    # ── Timing breakdown: quantize / dequantize / attention ──
    def quantize_only_step():
        policy.quantize(
            keys_np, values_np, QuantizationContext(layer_idx=0)
        ).to_packed()

    def dequantize_only_step():
        block.dequantize()

    quantize_samples = _measure_step_ms(quantize_only_step, min(iters, 50))
    dequantize_samples = _measure_step_ms(dequantize_only_step, min(iters, 50))
    quant_time = float(np.mean(quantize_samples))
    dequant_time = float(np.mean(dequantize_samples))
    attn_compute_time = quantized_avg - quant_time - dequant_time  # residual

    sig = policy_signature(policy)
    attention_impl = packed_attention_backend_name()
    kv_source = "real_model_snapshot" if snapshot_dir else "synthetic_random"
    evidence_label = "kernel_latency_not_deploy_speedup"
    improvement = (
        (baseline_avg - quantized_avg) / baseline_avg
        if baseline_avg and baseline_avg > 0
        else None
    )
    baseline_tps = (batch_size * 1000.0 / baseline_avg) if baseline_avg > 0 else None
    quantized_tps = (batch_size * 1000.0 / quantized_avg) if quantized_avg > 0 else None
    throughput_improvement = (
        (quantized_tps - baseline_tps) / baseline_tps
        if baseline_tps and quantized_tps
        else None
    )

    return LatencyResult(
        backend="triton_packed_attention",
        model=model_name,
        policy_name=policy_name,
        policy_signature=sig,
        nbits=nbits,
        kv_cache_dtype=kv_cache_dtype,
        hardware=_hardware_name(),
        quant_kernel=attention_impl,
        tpot_ms=quantized_avg,
        tokens_per_second=quantized_tps,
        baseline_tpot_ms=baseline_avg,
        tpot_improvement=improvement,
        throughput_improvement=throughput_improvement,
        quantize_time_ms=quant_time,
        dequantize_time_ms=dequant_time,
        attention_compute_ms=attn_compute_time,
        kv_source=kv_source,
        evidence_label=evidence_label,
        metadata={
            "attention_impl": attention_impl,
            "baseline_impl": "torch_full_precision_attention",
            "claim_boundary": "kernel_diagnostic_not_serving_speedup",
            "samples": iters,
            "warmup_iters": warmup_iters,
            "batch_size": batch_size,
            "heads": heads,
            "seq_len": int(keys_np.shape[2]),
            "head_dim": int(keys_np.shape[3]),
            "dtype": dtype,
            "snapshot_dir": snapshot_dir,
            "snapshot_layer_idx": snapshot_layer_idx,
            "baseline_tpot_ms_std": float(np.std(baseline_samples)),
            "tpot_ms_std": float(np.std(quantized_samples)),
        },
    )


def _sync_if_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure_step_ms(fn, iters: int) -> list[float]:
    import torch

    samples: list[float] = []
    if torch.cuda.is_available():
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(float(start.elapsed_time(end)))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def _hardware_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


def _git_commit(root: Path | None) -> str | None:
    if root is None or not root.exists():
        return None
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
