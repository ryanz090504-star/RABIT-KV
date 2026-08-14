"""Unified Markdown report generation.

Replaces both src/kvquant/report.py and scripts/render_modal_bench_fixed_report.py
with a single, clean report generator.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any


def generate_report(
    inputs: list[str],
    output: str,
    title: str = "KV Cache 量化实验报告",
) -> None:
    """Generate a Markdown report from JSONL result files.

    Parameters
    ----------
    inputs: one or more JSONL file paths
    output: output Markdown file path
    title: report title
    """
    rows = _read_inputs(inputs)
    quality_rows = [r for r in rows if _is_quality_row(r)]
    deploy_rows = [r for r in rows if r.get("evidence_label") == "deploy_latency"]
    kernel_rows = [r for r in rows if r.get("evidence_label") == "kernel_latency_not_deploy_speedup"]
    memory_rows = [r for r in rows if r.get("evidence_label") == "memory_estimate_not_runtime_peak"]
    task_rows = [r for r in rows if r.get("evidence_label") == "quality_task_not_deploy_speedup"]

    lines = [
        f"# {title}",
        "",
        f"**生成日期**: {date.today().isoformat()}",
        f"**输入文件**: {', '.join(inputs)}",
        f"**总行数**: {len(rows)}",
        "",
    ]

    # ── Quality table ──
    if quality_rows:
        lines.extend(_quality_section(quality_rows))
        lines.extend(_error_breakdown_section(quality_rows))
        lines.extend(_logit_fidelity_section(quality_rows))

    # ── Policy comparison matrix ──
    if quality_rows or memory_rows:
        lines.extend(_comparison_matrix(quality_rows, memory_rows, kernel_rows, task_rows))

    # ── Deploy latency ──
    if deploy_rows:
        lines.extend(_deploy_section(deploy_rows))

    # ── Kernel diagnostic latency ──
    if kernel_rows:
        lines.extend(_kernel_section(kernel_rows))

    # ── Task quality ──
    if task_rows:
        lines.extend(_task_section(task_rows))

    # ── Memory ──
    if memory_rows:
        lines.extend(_memory_section(memory_rows))

    # ── Evidence boundary ──
    lines.extend(_evidence_boundary())

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _quality_section(rows: list[dict]) -> list[str]:
    lines = ["## 质量结果 (Perplexity)", ""]
    baseline = next((r for r in rows if r.get("nbits") == 16 or r.get("nbits") is None), None)
    base_ppl = baseline.get("ppl", 1.0) if baseline else 1.0

    lines.append("| Bits | PPL | PPL Delta | KV KB | Compression | ms/token |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: x.get("nbits", 99) or 99, reverse=True):
        nbits = r.get("nbits", "fp16")
        ppl = r.get("ppl", 0)
        delta = (ppl - base_ppl) / base_ppl * 100 if base_ppl > 0 and nbits != 16 else 0
        kv_kb = r.get("kv_cache_kb", r.get("kv_kb", 0))
        comp = r.get("compression_ratio", r.get("compression", 1))
        ms = r.get("ms_per_token", r.get("ms_per_token", 0))
        lines.append(
            f"| {nbits} | {_fmt(ppl, 2)} | {delta:+.1f}% | {_fmt(kv_kb, 1)} | {_fmt(comp, 1)}x | {_fmt(ms, 2)} |"
        )
    return lines + [""]


def _error_breakdown_section(rows: list[dict]) -> list[str]:
    """量化误差详细指标表（仅当数据中有 error_* 字段时显示）"""
    has_errors = any(
        r.get("error_mse") is not None or r.get("error_cosine_similarity") is not None
        for r in rows
    )
    if not has_errors:
        return []

    lines = ["## 量化误差指标", ""]
    lines.append("| Bits | Policy | MSE | RMSE | Cosine | MaxAbs | SNR (dB) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: x.get("nbits", 99) or 99):
        nbits = r.get("nbits", "?")
        policy = r.get("policy_name", r.get("policy", "?"))
        lines.append(
            f"| {nbits} | {policy} | "
            f"{_fmt(r.get('error_mse','N/A'),6)} | {_fmt(r.get('error_rmse','N/A'),4)} | "
            f"{_fmt(r.get('error_cosine_similarity','N/A'),6)} | "
            f"{_fmt(r.get('error_max_abs','N/A'),4)} | "
            f"{_fmt(r.get('error_snr_db','N/A'),1)} |"
        )
    lines.append("")
    lines.append("- Cosine → 1.0 = 方向完全一致; SNR > 30dB = 好; SNR < 10dB = 严重退化")
    lines.append("- MSE/RMSE 越低越好; MaxAbs 越低越好")
    return lines + [""]


def _deploy_section(rows: list[dict]) -> list[str]:
    lines = ["## 部署延迟 (Deploy Latency)", ""]

    real = [r for r in rows if r.get("kv_source") == "real_model_snapshot"]
    synth = [r for r in rows if r.get("kv_source") != "real_model_snapshot"]

    if real:
        lines.append("- ✅ 包含真实模型 KV 张量 (`real_model_snapshot`)")
    if synth:
        lines.append("- ⚠️ 包含合成数据 (`synthetic_random`) — 仅用于 kernel 开发验证")

    lines.extend([
        "",
        "| Bits | KV DType | Hardware | vLLM Commit | KV Source | TPOT ms | Baseline TPOT ms | TPOT ↑ | Throughput ↑ |",
        "|---:|---|---|---|---|---:|---:|---:|---:|",
    ])
    for r in rows:
        nbits = r.get("nbits", "?")
        dtype = r.get("kv_cache_dtype", "?")
        hardware = r.get("hardware", "?")
        commit = r.get("vllm_commit", "?")
        src = r.get("kv_source", "?")
        tpot = r.get("tpot_ms", 0) or 0
        bl = r.get("baseline_tpot_ms", 0) or 0
        t_imp = r.get("tpot_improvement", 0) or 0
        thr_imp = r.get("throughput_improvement", 0) or 0
        lines.append(
            f"| {nbits} | `{dtype}` | {hardware} | `{commit}` | {src} | "
            f"{_fmt(tpot,4)} | {_fmt(bl,4)} | {t_imp*100:+.1f}% | {thr_imp*100:+.1f}% |"
        )
    return lines + [""]


def _kernel_section(rows: list[dict]) -> list[str]:
    lines = [
        "## Kernel 延迟诊断 (Not Serving Speedup)",
        "",
        "- 这里的结果只说明单步 attention 诊断性能，不等同于 vLLM serving 部署加速。",
        "- 如果 `attention_impl` 是 `torch_unpack_dequant_attention`，说明当前仍在先解包/反量化再计算 attention。",
        "",
        "| Bits | KV Source | Attention Impl | Quantized ms | Baseline ms | Delta | Tokens/s |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        meta = r.get("metadata") or {}
        nbits = r.get("nbits", "?")
        src = r.get("kv_source", "?")
        impl = meta.get("attention_impl", "?")
        tpot = r.get("tpot_ms", 0) or 0
        bl = r.get("baseline_tpot_ms", 0) or 0
        delta = r.get("tpot_improvement")
        tps = r.get("tokens_per_second", 0) or 0
        delta_text = "N/A" if delta is None else f"{delta*100:+.1f}%"
        lines.append(
            f"| {nbits} | {src} | `{impl}` | {_fmt(tpot,4)} | {_fmt(bl,4)} | {delta_text} | {_fmt(tps,2)} |"
        )
    return lines + [""]


def _task_section(rows: list[dict]) -> list[str]:
    lines = ["## 任务质量 (Task Quality)", ""]
    lines.append("| Dataset | Task | Bits | Baseline | Quantized | Recovery | Examples |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r.get('dataset','?')} | {r.get('task','?')} | {r.get('nbits','?')} | "
            f"{_fmt(r.get('baseline_score',0),4)} | {_fmt(r.get('quantized_score',0),4)} | "
            f"{r.get('accuracy_recovery',0)*100:.1f}% | {r.get('num_examples','?')} |"
        )
    return lines + [""]


def _memory_section(rows: list[dict]) -> list[str]:
    lines = ["## 存储压缩 (Memory)", ""]
    lines.append("| Bits | KV DType | Baseline GB | Estimated GB | Reduction | Compression | Packed | Scale | Codebook | QJL | Mixed Mask | Page OH |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        br = r.get("memory_breakdown") or {}
        lines.append(
            f"| {r.get('nbits','?')} | `{r.get('kv_cache_dtype','?')}` | {_fmt(r.get('baseline_cache_gb',0),2)} | "
            f"{_fmt(r.get('estimated_cache_gb',0),2)} | "
            f"{r.get('memory_reduction',0)*100:.0f}% | "
            f"{_fmt(r.get('compression_ratio',0),1)}x | "
            f"{_fmt_bytes(br.get('packed_kv_bytes'))} | "
            f"{_fmt_bytes(br.get('scale_bytes'))} | "
            f"{_fmt_bytes(br.get('codebook_bytes'))} | "
            f"{_fmt_bytes(br.get('qjl_residual_bytes'))} | "
            f"{_fmt_bytes(br.get('mixed_precision_mask_bytes'))} | "
            f"{_fmt_bytes(br.get('allocator_page_overhead_bytes'))} |"
        )
    return lines + [""]


def _evidence_boundary() -> list[str]:
    return [
        "## 证据边界",
        "",
        "- `quality_reference_not_deploy_speedup`: 参考后端质量/PPL 证据，不是真实部署加速",
        "- `kernel_latency_not_deploy_speedup`: 单步 kernel/microbench 诊断，不是完整 serving 加速",
        "- `deploy_latency`: 真实部署延迟，必须来自 packed cache + attention backend 的 serving/runtime 路径",
        "- `memory_estimate_not_runtime_peak`: 解析估算，不是运行时峰值",
        "- `quality_task_not_deploy_speedup`: 下游任务准确率证据，不是部署加速",
        "",
        "**论文写作规则**:",
        "- ✅ 可以写: 8-bit near-lossless, 2× 存储压缩",
        "- ✅ 可以写: kernel microbench 的诊断结果（必须注明不是完整 serving）",
        "- ❌ 不可以: 把 reference path Wall ms 写成推理加速",
        "- ❌ 不可以: 把 kernel microbench 写成完整 serving 加速",
        "- ❌ 不可以: 把本仓 vLLM scaffold 的 NotImplemented guard 写成 deploy 结果",
    ]


def _is_quality_row(r: dict) -> bool:
    has_ppl = "ppl" in r or "perplexity" in r
    has_loss = "loss" in r
    is_doc_smoke = r.get("evidence_label", "").startswith("document_algorithm_synthetic")
    return (has_ppl or has_loss) and not is_doc_smoke


def _logit_fidelity_section(rows: list[dict]) -> list[str]:
    """Logit-fidelity table: KL divergence, top-1, top-5 accuracy."""
    has_fidelity = any(
        r.get("kl_divergence") is not None or r.get("top1_accuracy") is not None
        for r in rows
    )
    if not has_fidelity:
        return []

    lines = ["## Logit Fidelity", ""]
    lines.append("| Bits | Policy | KL Divergence | Top-1 Acc | Top-5 Acc |")
    lines.append("|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: x.get("nbits", 99) or 99):
        nbits = r.get("nbits", "?")
        policy = r.get("policy_name", "?")
        kl = r.get("kl_divergence")
        t1 = r.get("top1_accuracy")
        t5 = r.get("top5_accuracy")
        lines.append(
            f"| {nbits} | {policy} | "
            f"{_fmt(kl, 4)} | "
            f"{_fmt_pct(t1)} | "
            f"{_fmt_pct(t5)} |"
        )
    lines.append("")
    lines.append("- KL divergence: lower is better (0 = identical distribution)")
    lines.append("- Top-k accuracy: fraction of steps where quantized top-k contains the baseline top-1 token")
    return lines + [""]


def _comparison_matrix(
    quality_rows: list[dict],
    memory_rows: list[dict],
    kernel_rows: list[dict],
    task_rows: list[dict],
) -> list[str]:
    """Cross-dimension comparison matrix: one row per policy/bit pair."""
    # Build a lookup: (policy_name, nbits) -> combined row
    combined: dict[tuple[str, int], dict[str, Any]] = {}

    for r in quality_rows:
        key = (r.get("policy_name", ""), r.get("nbits"))
        if key not in combined:
            combined[key] = {}
        combined[key].update({
            "ppl": r.get("ppl"),
            "error_mse": r.get("error_mse"),
            "error_snr": r.get("error_snr_db"),
            "kl": r.get("kl_divergence"),
            "top1": r.get("top1_accuracy"),
            "ms_per_token": r.get("ms_per_token"),
        })

    for r in memory_rows:
        key = (r.get("policy_name", ""), r.get("nbits"))
        if key not in combined:
            combined[key] = {}
        combined[key].update({
            "memory_reduction": r.get("memory_reduction"),
            "compression_ratio": r.get("compression_ratio"),
            "effective_bits": r.get("effective_bits_per_element"),
        })

    for r in kernel_rows:
        key = (r.get("policy_name", ""), r.get("nbits"))
        if key not in combined:
            combined[key] = {}
        combined[key].update({
            "tpot_ms": r.get("tpot_ms"),
            "speedup": r.get("tpot_improvement"),
        })

    for r in task_rows:
        key = (r.get("policy_name", ""), r.get("nbits"))
        if key not in combined:
            combined[key] = {}
        combined[key].update({"task_recovery": r.get("accuracy_recovery")})

    if not combined:
        return []

    lines = [
        "## 策略对比矩阵",
        "",
        "| Policy | Bits | PPL | MSE | SNR dB | KL | Mem Red. | Eff. Bits | Speedup | Task Rec. |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (policy, nbits), c in sorted(combined.items(), key=lambda x: (x[0][1] or 0, x[0][0])):
        p_str = str(policy).replace("_", r"\_")
        lines.append(
            f"| {p_str} | {nbits or '?'} | "
            f"{_fmt(c.get('ppl'), 2)} | "
            f"{_fmt(c.get('error_mse'), 4)} | "
            f"{_fmt(c.get('error_snr'), 1)} | "
            f"{_fmt(c.get('kl'), 4)} | "
            f"{_fmt_pct(c.get('memory_reduction'))} | "
            f"{_fmt(c.get('effective_bits'), 2)} | "
            f"{_fmt_pct(c.get('speedup'))} | "
            f"{_fmt_pct(c.get('task_recovery'))} |"
        )
    lines.append("")
    return lines + [""]


def _fmt_pct(val: Any) -> str:
    """Format a 0..1 ratio as a percentage string."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val) * 100:.1f}%"
    except (ValueError, TypeError):
        return str(val)


def _read_inputs(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _fmt(val: Any, digits: int = 2) -> str:
    try:
        v = float(val)
        if abs(v) >= 10000:
            return f"{v:,.0f}"
        return f"{v:.{digits}f}"
    except (ValueError, TypeError):
        return str(val)


def _fmt_bytes(val: Any) -> str:
    try:
        value = float(val or 0)
    except (ValueError, TypeError):
        return str(val)
    if value >= 1024**3:
        return f"{value / 1024**3:.2f} GB"
    if value >= 1024**2:
        return f"{value / 1024**2:.1f} MB"
    if value >= 1024:
        return f"{value / 1024:.1f} KB"
    return f"{value:.0f} B"
