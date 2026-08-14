"""LaTeX table export for paper-ready results.

Converts JSONL result files into LaTeX ``\\begin{table}`` environments
suitable for direct inclusion in academic papers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_to_latex(
    inputs: list[str],
    output: str,
    *,
    caption_prefix: str = "KV cache quantization results",
    label_prefix: str = "tab:kvquant",
) -> None:
    """Export JSONL results to a LaTeX table file.

    Args:
        inputs: Paths to JSONL result files.
        output: Output .tex file path.
        caption_prefix: Prefix for table captions.
        label_prefix: Prefix for LaTeX labels.
    """
    rows = _read_inputs(inputs)
    quality = [r for r in rows if _is_quality(r)]
    latency_rows = [r for r in rows if r.get("evidence_label") in (
        "deploy_latency", "kernel_latency_not_deploy_speedup",
    )]
    memory_rows = [r for r in rows if r.get("evidence_label") == "memory_estimate_not_runtime_peak"]
    task_rows = [r for r in rows if r.get("evidence_label") == "quality_task_not_deploy_speedup"]

    sections: list[str] = []

    if quality:
        sections.append(_quality_table(quality, caption_prefix, label_prefix))
    if latency_rows:
        sections.append(_latency_table(latency_rows, caption_prefix, label_prefix))
    if memory_rows:
        sections.append(_memory_table(memory_rows, caption_prefix, label_prefix))
    if task_rows:
        sections.append(_task_table(task_rows, caption_prefix, label_prefix))

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def _quality_table(rows: list[dict], cap_prefix: str, lab_prefix: str) -> str:
    header = (
        r"\begin{table}[htbp]" "\n"
        r"  \centering" "\n"
        rf"  \caption{{{cap_prefix} — Perplexity and reconstruction error.}}" "\n"
        rf"  \label{{{lab_prefix}:quality}}" "\n"
        r"  \begin{tabular}{lcccccc}" "\n"
        r"    \toprule" "\n"
        r"    Policy & Bits & PPL & PPL $\Delta$\% & MSE & Cosine $\uparrow$ & SNR (dB) \\" "\n"
        r"    \midrule" "\n"
    )
    lines = [header]

    baseline_ppl = next(
        (r.get("ppl", 1.0) for r in rows if r.get("nbits") in (16, None)),
        1.0,
    )

    for r in sorted(rows, key=lambda x: (x.get("nbits", 99) or 99, x.get("policy_name", ""))):
        nbits = r.get("nbits", "fp16")
        policy = r.get("policy_name", "?").replace("_", r"\_")
        ppl = r.get("ppl", 0)
        delta = ((ppl - baseline_ppl) / baseline_ppl * 100) if baseline_ppl > 0 and nbits != 16 else 0
        mse = r.get("error_mse")
        cos = r.get("error_cosine_similarity")
        snr = r.get("error_snr_db")
        lines.append(
            f"    {policy} & {_lbits(nbits)} & {_lfloat(ppl, 2)} & "
            f"{delta:+.1f}\\% & {_lfloat(mse, 4)} & {_lfloat(cos, 4)} & {_lfloat(snr, 1)} \\\\"
        )

    footer = (
        r"    \bottomrule" "\n"
        r"  \end{tabular}" "\n"
        r"\end{table}"
    )
    lines.append(footer)
    return "\n".join(lines)


def _latency_table(rows: list[dict], cap_prefix: str, lab_prefix: str) -> str:
    header = (
        r"\begin{table}[htbp]" "\n"
        r"  \centering" "\n"
        rf"  \caption{{{cap_prefix} — Kernel and deployment latency.}}" "\n"
        rf"  \label{{{lab_prefix}:latency}}" "\n"
        r"  \begin{tabular}{lccccc}" "\n"
        r"    \toprule" "\n"
        r"    Policy & Bits & Quant (ms) & Dequant (ms) & Baseline (ms) & Speedup \\" "\n"
        r"    \midrule" "\n"
    )
    lines = [header]

    for r in sorted(rows, key=lambda x: (x.get("nbits", 99) or 99, x.get("policy_name", ""))):
        policy = r.get("policy_name", "?").replace("_", r"\_")
        nbits = r.get("nbits", "?")
        tpot = r.get("tpot_ms") or 0
        bl = r.get("baseline_tpot_ms") or 0
        imp = r.get("tpot_improvement")
        speedup = f"{imp*100:+.1f}\\%" if imp is not None else "N/A"
        qt = r.get("quantize_time_ms")
        dq = r.get("dequantize_time_ms")
        lines.append(
            f"    {policy} & {_lbits(nbits)} & {_lfloat(qt, 3)} & "
            f"{_lfloat(dq, 3)} & {_lfloat(bl, 3)} & {speedup} \\\\"
        )

    footer = (
        r"    \bottomrule" "\n"
        r"  \end{tabular}" "\n"
        r"\end{table}"
    )
    lines.append(footer)
    return "\n".join(lines)


def _memory_table(rows: list[dict], cap_prefix: str, lab_prefix: str) -> str:
    header = (
        r"\begin{table}[htbp]" "\n"
        r"  \centering" "\n"
        rf"  \caption{{{cap_prefix} — Memory footprint and compression.}}" "\n"
        rf"  \label{{{lab_prefix}:memory}}" "\n"
        r"  \begin{tabular}{lcccccc}" "\n"
        r"    \toprule" "\n"
        r"    Policy & Bits & Base (GB) & Quant (GB) & Reduction & Ratio & Eff. bits \\" "\n"
        r"    \midrule" "\n"
    )
    lines = [header]

    for r in sorted(rows, key=lambda x: (x.get("nbits", 99) or 99, x.get("policy_name", ""))):
        policy = r.get("policy_name", "?").replace("_", r"\_")
        nbits = r.get("nbits", "?")
        base = r.get("baseline_cache_gb", 0) or 0
        est = r.get("estimated_cache_gb", 0) or 0
        red = (r.get("memory_reduction") or 0) * 100
        ratio = r.get("compression_ratio") or 1
        eff = r.get("effective_bits_per_element")
        lines.append(
            f"    {policy} & {_lbits(nbits)} & {_lfloat(base, 2)} & "
            f"{_lfloat(est, 2)} & {red:.0f}\\% & {_lfloat(ratio, 1)}$\\times$ & "
            f"{_lfloat(eff, 2)} \\\\"
        )

    footer = (
        r"    \bottomrule" "\n"
        r"  \end{tabular}" "\n"
        r"\end{table}"
    )
    lines.append(footer)
    return "\n".join(lines)


def _task_table(rows: list[dict], cap_prefix: str, lab_prefix: str) -> str:
    header = (
        r"\begin{table}[htbp]" "\n"
        r"  \centering" "\n"
        rf"  \caption{{{cap_prefix} — Task accuracy (exact match).}}" "\n"
        rf"  \label{{{lab_prefix}:tasks}}" "\n"
        r"  \begin{tabular}{lcccc}" "\n"
        r"    \toprule" "\n"
        r"    Policy & Bits & Baseline & Quantized & Recovery \\" "\n"
        r"    \midrule" "\n"
    )
    lines = [header]

    for r in sorted(rows, key=lambda x: (x.get("nbits", 99) or 99, x.get("policy_name", ""))):
        policy = r.get("policy_name", "?").replace("_", r"\_")
        nbits = r.get("nbits", "?")
        base = r.get("baseline_score", 0) or 0
        quant = r.get("quantized_score", 0) or 0
        rec = (r.get("accuracy_recovery") or 0) * 100
        lines.append(
            f"    {policy} & {_lbits(nbits)} & {_lfloat(base, 3)} & "
            f"{_lfloat(quant, 3)} & {rec:.1f}\\% \\\\"
        )

    footer = (
        r"    \bottomrule" "\n"
        r"  \end{tabular}" "\n"
        r"\end{table}"
    )
    lines.append(footer)
    return "\n".join(lines)


def _lfloat(val: Any, digits: int = 2) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):.{digits}f}"
    except (ValueError, TypeError):
        return str(val)


def _lbits(val: Any) -> str:
    if val is None or val == 16:
        return "fp16"
    return f"{val}"


def _is_quality(r: dict) -> bool:
    return bool(r.get("ppl") or r.get("loss"))


def _read_inputs(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError:
                    pass
    return rows
