"""Quality benchmark: PPL measurement with Transformers.

Runs autoregressive next-token prediction with quantized KV cache
to measure perplexity degradation at each bit width.

Replaces the standalone modal_bench_fixed.py script — now uses
kvquant's unified policy/packing/snapshot stack directly.
"""

from __future__ import annotations

import gc
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QualityResult:
    """Result from one quality benchmark run."""

    model: str
    policy_name: str
    policy_signature: str
    nbits: int | None
    loss: float
    ppl: float
    kv_cache_kb: float
    compression_ratio: float
    seq_len: int
    num_windows: int
    effective_steps: int
    wall_ms: float
    ms_per_token: float
    gpu_ms: float
    gpu_ms_per_token: float
    kv_cache_dtype: str | None = None
    kv_source: str = "transformers_reference_cache"
    hardware: str | None = None
    vllm_commit: str | None = None
    quant_kernel: str | None = "transformers_reference_roundtrip"
    memory_breakdown: dict[str, Any] = field(default_factory=dict)

    # ── 量化误差详细指标 ──
    error_mse: float | None = None
    error_rmse: float | None = None
    error_cosine_similarity: float | None = None
    error_max_abs: float | None = None
    error_snr_db: float | None = None
    error_by_layer: list[dict[str, Any]] = field(default_factory=list)

    # ── Logit-level fidelity ──
    kl_divergence: float | None = None
    """KL divergence between full-precision and quantized output logits."""
    top1_accuracy: float | None = None
    """Fraction of steps where top-1 predicted token matches baseline."""
    top5_accuracy: float | None = None
    """Fraction of steps where top-5 predictions contain the baseline top-1."""

    # ── Attention score distortion ──
    attn_score_mse: float | None = None
    attn_score_cosine: float | None = None
    attn_score_top1_recall: float | None = None
    attn_score_top5_recall: float | None = None

    evidence_label: str = "quality_reference_not_deploy_speedup"
    metadata: dict[str, Any] = field(default_factory=dict)


def run_quality_benchmark(
    model_name: str,
    policy: object,
    *,
    text_file: str,
    max_tokens: int = 256,
    num_windows: int = 4,
    batch_size: int = 1,
    warmup_steps: int = 3,
    dtype: str = "float16",
    seed: int = 0,
    error_breakdown: bool = False,
) -> QualityResult:
    """Run PPL benchmark using a kvquant policy with Transformers.

    This wraps the autoregressive loop from modal_bench_fixed.py but
    uses the policy system for quantization instead of hardcoded min-max.

    Parameters
    ----------
    model_name: HuggingFace model ID or path
    policy: KVQuantPolicy instance
    text_file: path to evaluation text
    max_tokens: tokens per window
    num_windows: number of independent windows
    batch_size: batch size (default 1)
    warmup_steps: initial decode steps excluded from timing
    dtype: model dtype (float16 / bfloat16)
    seed: random seed for reproducibility
    error_breakdown: if True, compute per-layer quantization error stats
    Returns
    -------
    QualityResult with PPL, compression, and timing.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kvquant.policies import policy_signature

    if not torch.cuda.is_available():
        raise RuntimeError("Quality benchmark requires CUDA")

    device = torch.device("cuda")
    torch_dtype = getattr(torch, dtype)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ── Load model and data ──
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=None,
    ).to(device)
    model.eval()

    lines = _load_text_lines(text_file)
    token_ids = _build_token_pool(tokenizer, lines, max_tokens * num_windows * batch_size)
    window_size = max_tokens * batch_size
    windows = [
        token_ids[i * window_size : (i + 1) * window_size].reshape(batch_size, max_tokens).to(device)
        for i in range(num_windows)
    ]

    nbits = getattr(policy, "nbits", None)

    # ── Run trials ──
    losses = []
    kv_sizes = []
    baseline_kv_sizes = []
    all_wall_ms = []
    all_gpu_ms = []
    effective_steps_values = []
    last_raw_cache = None
    # Logit-fidelity from all windows
    all_kl: list[float] = []
    all_top1: list[float] = []
    all_top5: list[float] = []

    for w in windows:
        result = _run_one_quality_window(
            model=model,
            policy=policy,
            input_ids=w,
            target_dtype=torch_dtype,
            nbits=nbits,
            warmup_steps=warmup_steps,
        )
        losses.append(result["loss"])
        kv_sizes.append(result["kv_kb"])
        baseline_kv_sizes.append(result["baseline_kv_kb"])
        all_wall_ms.append(result["wall_ms"])
        all_gpu_ms.append(result["gpu_ms"])
        effective_steps_values.append(result["effective_steps"])
        # Save the raw cache from the last window for error breakdown
        last_raw_cache = result.get("raw_cache")
        # Accumulate logit-fidelity metrics
        kl = result.get("kl_divergence")
        t1 = result.get("top1_accuracy")
        t5 = result.get("top5_accuracy")
        if kl is not None:
            all_kl.append(kl)
        if t1 is not None:
            all_top1.append(t1)
        if t5 is not None:
            all_top5.append(t5)

    # ── Aggregate ──
    median_loss = statistics.median(losses)
    median_kv = statistics.median(kv_sizes)
    baseline_kv = statistics.median(baseline_kv_sizes) if baseline_kv_sizes else median_kv
    effective_steps = int(statistics.median(effective_steps_values)) if effective_steps_values else 0

    # ── Logit fidelity (from last window) ──

    # ── Compute per-layer quantization error if requested ──
    error_mse = None
    error_rmse = None
    error_cosine = None
    error_max_abs = None
    error_snr = None
    error_by_layer = []

    if error_breakdown and nbits not in (16, None) and last_raw_cache:
        from kvquant.metrics import quantization_error_stats as _qes
        from kvquant.metrics import aggregate_error_stats as _aggregate
        kv_tuple = _cache_to_tuple(last_raw_cache)
        layer_stats = []
        all_k_stats = []
        all_v_stats = []
        if kv_tuple:
            for layer_idx, entry in enumerate(kv_tuple):
                if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                    continue
                k_np = entry[0].detach().cpu().float().numpy()
                v_np = entry[1].detach().cpu().float().numpy()
                from kvquant.types import QuantizationContext
                ctx = QuantizationContext(layer_idx=layer_idx)
                block = policy.quantize(k_np, v_np, ctx)
                deq_k, deq_v = block.dequantize()
                k_stats = _qes(k_np, deq_k)
                v_stats = _qes(v_np, deq_v)
                all_k_stats.append(k_stats)
                all_v_stats.append(v_stats)
                layer_stats.append({
                    "layer": layer_idx,
                    "key_mse": k_stats.mse, "key_cosine": k_stats.cosine_similarity,
                    "key_snr_db": k_stats.snr_db,
                    "value_mse": v_stats.mse, "value_cosine": v_stats.cosine_similarity,
                    "value_snr_db": v_stats.snr_db,
                })
        if all_k_stats:
            k_agg = _aggregate(all_k_stats)
            v_agg = _aggregate(all_v_stats)
            error_mse = (k_agg.mse + v_agg.mse) / 2
            error_rmse = (k_agg.rmse + v_agg.rmse) / 2
            error_cosine = (k_agg.cosine_similarity + v_agg.cosine_similarity) / 2
            error_max_abs = max(k_agg.max_abs, v_agg.max_abs)
            error_snr = (k_agg.snr_db + v_agg.snr_db) / 2
            error_by_layer = layer_stats

    kl_val = statistics.mean(all_kl) if all_kl else None
    t1_val = statistics.mean(all_top1) if all_top1 else None
    t5_val = statistics.mean(all_top5) if all_top5 else None

    return QualityResult(
        model=model_name,
        policy_name=policy.name,
        policy_signature=policy_signature(policy),
        nbits=nbits,
        kv_cache_dtype=_kv_cache_dtype(nbits),
        hardware=torch.cuda.get_device_name(0),
        loss=median_loss,
        ppl=math.exp(min(median_loss, 50)),
        kv_cache_kb=median_kv,
        compression_ratio=baseline_kv / median_kv if median_kv > 0 else 1.0,
        seq_len=max_tokens,
        num_windows=num_windows,
        effective_steps=effective_steps,
        wall_ms=statistics.median(all_wall_ms),
        ms_per_token=statistics.median(all_wall_ms) / max(1, effective_steps),
        gpu_ms=statistics.median(all_gpu_ms),
        gpu_ms_per_token=statistics.median(all_gpu_ms) / max(1, effective_steps),
        error_mse=error_mse,
        error_rmse=error_rmse,
        error_cosine_similarity=error_cosine,
        error_max_abs=error_max_abs,
        error_snr_db=error_snr,
        error_by_layer=error_by_layer,
        kl_divergence=kl_val,
        top1_accuracy=t1_val,
        top5_accuracy=t5_val,
    )


def _run_one_quality_window(
    model,
    policy,
    input_ids,
    target_dtype,
    nbits,
    warmup_steps,
) -> dict:
    """Run one autoregressive window with quantized KV cache.

    When *nbits* < 16, also runs a full-precision pass at each step to
    collect matched (fp_logits, quant_logits) pairs for KL divergence and
    top-k fidelity measurement.
    """
    import torch
    import torch.nn.functional as F

    from kvquant.types import QuantizationContext

    steps = input_ids.shape[1] - 1
    eff_warmup = min(max(warmup_steps, 0), max(steps - 1, 0))
    loss_tensors = []
    kv_kb_now = 0.0
    baseline_kv_kb_now = 0.0
    stored_cache = None  # PackedKVBlock per layer
    cache_in = None

    # Logit-fidelity accumulators (populated only when nbits < 16)
    kl_values: list[float] = []
    top1_matches = 0
    top5_matches = 0
    logit_steps = 0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    t_wall_start = None

    with torch.inference_mode():
        for i in range(steps):
            current = input_ids[:, i : i + 1]
            target = input_ids[:, i + 1 : i + 2]

            if i == eff_warmup:
                torch.cuda.synchronize()
                start_event.record()
                t_wall_start = time.perf_counter()

            if nbits == 16 or nbits is None:
                if cache_in is None:
                    outputs = model(input_ids=current, use_cache=True)
                else:
                    outputs = model(input_ids=current, past_key_values=cache_in, use_cache=True)
                cache_in = outputs.past_key_values
                kv_kb_now = _cache_kb(cache_in)
                baseline_kv_kb_now = kv_kb_now
            else:
                # Baseline fp forward
                if stored_cache is None:
                    baseline_outputs = model(input_ids=current, use_cache=True)
                else:
                    cache_arg = _dequant_cache(stored_cache, target_dtype)
                    baseline_outputs = model(input_ids=current, past_key_values=cache_arg, use_cache=True)
                baseline_kv_kb_now = _cache_kb(baseline_outputs.past_key_values)

                # Quantized path: dequant → forward → quant
                if stored_cache is not None:
                    quant_outputs = model(input_ids=current, past_key_values=cache_arg, use_cache=True)
                else:
                    quant_outputs = baseline_outputs
                stored_cache = _quant_cache(quant_outputs.past_key_values, policy)
                kv_kb_now = _stored_kb(stored_cache)

                # Logit fidelity: compare fp vs quant logits (after warmup)
                if i >= eff_warmup:
                    fp_logits = baseline_outputs.logits[:, -1, :].float()
                    q_logits = quant_outputs.logits[:, -1, :].float()
                    try:
                        from kvquant.metrics import kl_divergence as _kl
                        kl_values.append(_kl(fp_logits.cpu().numpy(), q_logits.cpu().numpy()))
                    except Exception:
                        pass
                    fp_top1 = int(torch.argmax(fp_logits, dim=-1).item())
                    q_top5 = torch.topk(q_logits, k=min(5, q_logits.shape[-1]), dim=-1).indices
                    if int(q_top5[0, 0].item()) == fp_top1:
                        top1_matches += 1
                    if fp_top1 in q_top5[0].tolist():
                        top5_matches += 1
                    logit_steps += 1

                outputs = quant_outputs

            logits = outputs.logits[:, -1, :]
            loss = F.cross_entropy(logits.float(), target.reshape(-1))
            loss_tensors.append(loss.detach())

    end_event.record()
    torch.cuda.synchronize()
    t_wall_end = time.perf_counter()
    loss_values = [float(loss.cpu().item()) for loss in loss_tensors]

    return {
        "loss": float(sum(loss_values) / len(loss_values)),
        "kv_kb": float(kv_kb_now),
        "baseline_kv_kb": float(baseline_kv_kb_now),
        "effective_steps": max(0, steps - eff_warmup),
        "wall_ms": (t_wall_end - (t_wall_start or t_wall_end)) * 1000,
        "gpu_ms": start_event.elapsed_time(end_event),
        "raw_cache": outputs.past_key_values,
        # Logit fidelity
        "kl_divergence": statistics.mean(kl_values) if kl_values else None,
        "top1_accuracy": top1_matches / logit_steps if logit_steps > 0 else None,
        "top5_accuracy": top5_matches / logit_steps if logit_steps > 0 else None,
    }


# ── Cache helpers (adapted from old code) ──

def _cache_to_tuple(cache):
    if cache is None:
        return None
    if isinstance(cache, tuple):
        return cache
    if isinstance(cache, list):
        return tuple(cache)
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return tuple((k, v) for k, v in zip(cache.key_cache, cache.value_cache))
    try:
        return tuple(cache)
    except TypeError:
        pass
    raise RuntimeError(f"unsupported cache type: {type(cache)}")


def _cache_kb(cache) -> float:
    kv_tuple = _cache_to_tuple(cache)
    if kv_tuple is None:
        return 0.0
    total = 0
    for entry in kv_tuple:
        if isinstance(entry, (tuple, list)) and len(entry) >= 2:
            for t in (entry[0], entry[1]):
                if hasattr(t, "numel") and hasattr(t, "element_size"):
                    total += t.numel() * t.element_size()
    return total / 1024.0


def _quant_cache(cache, policy):
    """Quantize a HuggingFace cache using the policy."""
    import numpy as np

    from kvquant.types import QuantizationContext

    kv_tuple = _cache_to_tuple(cache)
    blocks = []
    for layer_idx, entry in enumerate(kv_tuple):
        if not isinstance(entry, (tuple, list)) or len(entry) < 2:
            continue
        k_np = entry[0].detach().cpu().float().numpy()
        v_np = entry[1].detach().cpu().float().numpy()
        ctx = QuantizationContext(layer_idx=layer_idx)
        block = policy.quantize(k_np, v_np, ctx)
        blocks.append(block.to_packed())
    return blocks


def _stored_kb(stored_cache) -> float:
    total = 0
    for block in stored_cache:
        total += block.estimated_payload_nbytes()
    return total / 1024.0


def _dequant_cache(stored_cache, target_dtype):
    """Dequantize a stored packed cache back to Transformers DynamicCache."""
    import torch
    from transformers.cache_utils import DynamicCache

    layers = []
    for block in stored_cache:
        dk, dv = block.dequantize()
        layers.append((
            torch.as_tensor(dk, device="cuda", dtype=target_dtype),
            torch.as_tensor(dv, device="cuda", dtype=target_dtype),
        ))

    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(tuple(layers))
    cache = DynamicCache()
    for idx, (k, v) in enumerate(layers):
        cache.update(k, v, idx)
    return cache


# ── Data helpers ──

def _load_text_lines(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _build_token_pool(tokenizer, lines: list[str], needed: int):
    import torch

    ids = []
    for start in range(0, len(lines), 64):
        text = "\n".join(lines[start : start + 64])
        out = tokenizer(text, add_special_tokens=False)
        ids.extend(out["input_ids"])
        if len(ids) >= needed:
            break
    if len(ids) < needed:
        ids = ids * ((needed + len(ids) - 1) // len(ids))
    return torch.tensor(ids[:needed], dtype=torch.long)


def _kv_cache_dtype(nbits: int | None) -> str:
    if nbits is None or nbits >= 16:
        return "fp16"
    return f"kvquant_k{nbits}"
