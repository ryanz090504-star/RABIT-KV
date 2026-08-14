#!/usr/bin/env python3
"""
RABIT-KV final results aggregator.

This is a fully self-contained local script. It does not require Modal, a GPU,
or any external Markdown/CSV input files.

What it does
------------
1. Stores all frozen experimental measurements directly in Python data.
2. Recomputes:
   - PPL deltas
   - logical compression ratios
   - combined Qasper and HotpotQA scores
   - ablation component effects
   - deployment medians, TPOT, and capacity ratios
3. Validates key consistency conditions.
4. Prints clean final tables.
5. Exports:
   - RABIT_KV_Final_Result_Tables.md
   - RABIT_KV_Final_Results.json
   - RABIT_KV_Main_Results.csv
   - RABIT_KV_LongContext_Results.csv
   - RABIT_KV_2bit_Ablation.csv
   - RABIT_KV_Deployment_Latency.csv

Run
---
    python RABIT_KV_final_results_full.py

Optional
--------
    python RABIT_KV_final_results_full.py --output-dir results
    python RABIT_KV_final_results_full.py --no-export
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


# =============================================================================
# Frozen raw measurements
# =============================================================================

MODEL = "Llama 3.1 8B Instruct"
GPU = "NVIDIA H100 80GB HBM3"

BASELINE_PPL = 11.23
BASELINE_KV_MB = 31.875


@dataclass(frozen=True)
class OperatingPoint:
    method: str
    policy: str
    ppl: float
    kv_mb: float
    effective_bits: float | None = None




# Reported deltas are preserved from the original benchmark output.
# Recomputing from the displayed two-decimal PPL values can differ by about
# 0.1 percentage point because the benchmark used higher-precision internal
# values before rounding PPL for display.
REPORTED_OPERATING_PPL_DELTA_PCT = {
    "BF16": 0.0,
    "RABIT-8": 0.0,
    "RABIT-4": 2.0,
    "RABIT-3": 4.0,
    "RABIT-2 target": 2.6,
}

REPORTED_ABLATION_PPL_DELTA_PCT = {
    "BF16 baseline": 0.0,
    "A Uniform2 G32 R0": 1465.5,
    "B Uniform2 G32 R4": 460.1,
    "C K2/V2 affine R4": 10.6,
    "D K2/V3 affine R4": 7.3,
    "E K3/V2 affine R0": 7.4,
    "F FINAL K3/V2 affine R4": 2.6,
}

OPERATING_POINTS: tuple[OperatingPoint, ...] = (
    OperatingPoint("BF16", "Uncompressed baseline", 11.23, 31.875, 16.00),
    OperatingPoint("RABIT-8", "SYM K32V64 R0", 11.22, 16.685, None),
    OperatingPoint("RABIT-4", "SYM G64 R0", 11.45, 8.467, None),
    OperatingPoint("RABIT-3", "SYM G32 R4", 11.68, 7.363, None),
    OperatingPoint(
        "RABIT-2 target",
        "K3/V2; K sequence-affine; V token-affine; R4",
        11.51,
        7.441,
        3.74,
    ),
)


# Long-context continuation PPL.
CONTINUATION_PPL = {
    "settings": {
        "context_tokens": 1024,
        "continuation_tokens": 128,
        "samples": 1,
    },
    "scores": {
        "BF16": 3.285,
        "RABIT-4": 3.350,
        "RABIT-2 target": 3.397,
    },
}


# Needle-in-a-Haystack smoke tests.
NIAH = {
    "BF16": {4096: True, 8192: True, 16384: True},
    "RABIT-4": {4096: True, 8192: True, 16384: True},
    "RABIT-2 target": {4096: True, 8192: True, 16384: True},
}


# LongBench-E passage_retrieval_en, 10 samples.
PASSAGE_RETRIEVAL = {
    "samples": 10,
    "accuracy": {
        "BF16": 1.00,
        "RABIT-4": 1.00,
        "RABIT-2 target": 1.00,
    },
}


# LongBench-E Qasper, 8K+ bucket.
# Slice 1 is the original first 10 samples.
# Slice 2 is the remaining 14 samples, completing the full 24-example bucket.
QASPER_SLICES = {
    "first_10": {
        "samples": 10,
        "f1": {
            "BF16": 0.235,
            "RABIT-4": 0.240,
            "RABIT-2 target": 0.236,
        },
    },
    "remaining_14": {
        "samples": 14,
        "f1_total": {
            "BF16": 6.17,
            "RABIT-4": 5.24,
            "RABIT-3": 5.14,
            "RABIT-2 target": 5.96,
        },
    },
}


# LongBench-E HotpotQA, 8K+ bucket.
HOTPOT_SLICES = {
    "first_10": {
        "samples": 10,
        "f1": {
            "BF16": 0.642,
            "RABIT-4": 0.636,
            "RABIT-3": 0.643,
            "RABIT-2 target": 0.536,
        },
    },
    "second_10": {
        "samples": 10,
        "f1": {
            "BF16": 0.569,
            "RABIT-4": 0.552,
            "RABIT-3": 0.575,
            "RABIT-2 target": 0.569,
        },
    },
}


@dataclass(frozen=True)
class AblationPoint:
    configuration: str
    ppl: float
    kv_mb: float
    effective_bits: float
    component: str


ABLATION_POINTS: tuple[AblationPoint, ...] = (
    AblationPoint(
        "BF16 baseline",
        11.23,
        31.875,
        16.00,
        "Uncompressed reference",
    ),
    AblationPoint(
        "A Uniform2 G32 R0",
        175.74,
        4.980,
        2.50,
        "Ordinary uniform symmetric 2-bit",
    ),
    AblationPoint(
        "B Uniform2 G32 R4",
        62.87,
        5.402,
        2.71,
        "Add four-token FP16 residual",
    ),
    AblationPoint(
        "C K2/V2 affine R4",
        12.42,
        6.441,
        3.23,
        "Use sequence/token affine axes",
    ),
    AblationPoint(
        "D K2/V3 affine R4",
        12.04,
        7.422,
        3.73,
        "Protect V instead of K",
    ),
    AblationPoint(
        "E K3/V2 affine R0",
        12.06,
        6.988,
        3.51,
        "Mixed K3/V2 without residual",
    ),
    AblationPoint(
        "F FINAL K3/V2 affine R4",
        11.51,
        7.441,
        3.74,
        "Final K3/V2 + affine + R4",
    ),
)


# Controlled production benchmark.
# Both modes used the Triton attention backend.
# TTFT = one-token request latency.
# TPOT = (65-token request latency - one-token request latency) / 64.
DEPLOYMENT_RAW = {
    "environment": {
        "model": MODEL,
        "gpu": GPU,
        "backend": "TRITON_ATTN",
        "execution": "eager",
        "torch_compile": False,
        "cuda_graphs": False,
    },
    "capacity_tokens": {
        "BF16 Triton": 315_200,
        "INT3 kvquant_k3 Triton": 1_551_776,
    },
    "trials": {
        "BF16 Triton": {
            512: {
                "ttft_ms": [21.152684000000477, 19.91454699999906, 19.6886900000095],
                "total65_ms": [880.0967589999971, 907.9826029999936, 885.4309429999887],
            },
            2048: {
                "ttft_ms": [66.79410099999927, 67.7331210000034, 66.64455199999963],
                "total65_ms": [916.9718110000105, 944.968811999999, 928.5126079999912],
            },
            4096: {
                "ttft_ms": [157.65546299999755, 161.12959500000557, 161.54151199998523],
                "total65_ms": [1043.1106430000057, 1027.6736479999897, 1041.5943819999995],
            },
        },
        "INT3 kvquant_k3 Triton": {
            512: {
                "ttft_ms": [53.68309099998214, 52.812603000006675, 52.29759599998829],
                "total65_ms": [2670.930481999989, 2677.5195779999876, 2720.381137000004],
            },
            2048: {
                "ttft_ms": [181.2401560000012, 182.17660999999907, 183.1170259999908],
                "total65_ms": [2792.4339219999865, 2812.0822989999965, 2794.217161000006],
            },
            4096: {
                "ttft_ms": [542.243740999993, 542.0459739999899, 545.855620999987],
                "total65_ms": [3164.899803999987, 3168.5988469999984, 3139.7947520000002],
            },
        },
    },
}


# =============================================================================
# Calculation helpers
# =============================================================================

def ppl_delta_pct(ppl: float, baseline: float = BASELINE_PPL) -> float:
    return (ppl / baseline - 1.0) * 100.0


def compression_ratio(kv_mb: float, baseline_mb: float = BASELINE_KV_MB) -> float:
    return baseline_mb / kv_mb


def weighted_average(parts: Sequence[tuple[int, float]]) -> float:
    total_n = sum(n for n, _ in parts)
    if total_n <= 0:
        raise ValueError("Weighted average requires a positive sample count.")
    return sum(n * value for n, value in parts) / total_n


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate median of an empty sequence.")
    return float(statistics.median(values))


def tpot_trials(ttft_ms: Sequence[float], total65_ms: Sequence[float]) -> list[float]:
    if len(ttft_ms) != len(total65_ms):
        raise ValueError("TTFT and total65 trial counts must match.")
    return [(total - first) / 64.0 for first, total in zip(ttft_ms, total65_ms)]


def format_percent(value: float, digits: int = 1, include_plus: bool = True) -> str:
    if abs(value) < 0.05:
        return "0.0%"
    sign = "+" if include_plus and value > 0 else ""
    return f"{sign}{value:.{digits}f}%"


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def format_f1(value: float) -> str:
    """Format a 0-1 F1 score as percent using conventional half-up rounding."""
    percent = Decimal(str(round(value * 100.0, 10))).quantize(
        Decimal("0.1"),
        rounding=ROUND_HALF_UP,
    )
    return f"{percent:.1f}"


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def plain_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    rendered = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in rendered:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    header_line = " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        for row in rendered
    ]
    return "\n".join([header_line, separator, *body])


# =============================================================================
# Derived result builders
# =============================================================================

def build_operating_point_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in OPERATING_POINTS:
        rows.append(
            {
                "Method": point.method,
                "Policy": point.policy,
                "PPL": point.ppl,
                "PPL delta pct": REPORTED_OPERATING_PPL_DELTA_PCT[point.method],
                "PPL delta recomputed pct": ppl_delta_pct(point.ppl),
                "KV MB": point.kv_mb,
                "Compression": compression_ratio(point.kv_mb),
                "Effective bits": point.effective_bits,
            }
        )
    return rows


def build_qasper_results() -> dict[str, dict[str, Any]]:
    first = QASPER_SLICES["first_10"]
    second = QASPER_SLICES["remaining_14"]

    output: dict[str, dict[str, Any]] = {}

    for method in ("BF16", "RABIT-4", "RABIT-2 target"):
        first_total = first["samples"] * first["f1"][method]
        second_total = second["f1_total"][method]
        total_samples = first["samples"] + second["samples"]
        output[method] = {
            "samples": total_samples,
            "f1": (first_total + second_total) / total_samples,
            "first_10_f1": first["f1"][method],
            "remaining_14_f1": second_total / second["samples"],
        }

    output["RABIT-3"] = {
        "samples": second["samples"],
        "f1": second["f1_total"]["RABIT-3"] / second["samples"],
        "note": "Only the final 14 samples were evaluated.",
    }

    return output


def build_hotpot_results() -> dict[str, dict[str, Any]]:
    first = HOTPOT_SLICES["first_10"]
    second = HOTPOT_SLICES["second_10"]
    methods = ("BF16", "RABIT-4", "RABIT-3", "RABIT-2 target")

    output: dict[str, dict[str, Any]] = {}
    for method in methods:
        combined = weighted_average(
            [
                (first["samples"], first["f1"][method]),
                (second["samples"], second["f1"][method]),
            ]
        )
        output[method] = {
            "samples": first["samples"] + second["samples"],
            "f1": combined,
            "first_10_f1": first["f1"][method],
            "second_10_f1": second["f1"][method],
        }
    return output


def build_long_context_rows() -> list[dict[str, Any]]:
    qasper = build_qasper_results()
    hotpot = build_hotpot_results()
    continuation = CONTINUATION_PPL["scores"]

    methods = ["BF16", "RABIT-4", "RABIT-3", "RABIT-2 target"]
    rows: list[dict[str, Any]] = []

    for method in methods:
        niah_values = NIAH.get(method)
        niah_text = (
            "Pass / Pass / Pass"
            if niah_values and all(niah_values.values())
            else "—"
        )

        passage = PASSAGE_RETRIEVAL["accuracy"].get(method)
        passage_text = f"{passage * 100:.0f}%" if passage is not None else "—"

        qasper_info = qasper.get(method)
        hotpot_info = hotpot.get(method)

        rows.append(
            {
                "Method": method,
                "Continuation PPL": continuation.get(method),
                "Qasper samples": qasper_info["samples"] if qasper_info else None,
                "Qasper F1": qasper_info["f1"] if qasper_info else None,
                "Hotpot samples": hotpot_info["samples"] if hotpot_info else None,
                "Hotpot F1": hotpot_info["f1"] if hotpot_info else None,
                "NIAH 4K/8K/16K": niah_text,
                "Passage retrieval": passage_text,
            }
        )
    return rows


def build_ablation_rows() -> list[dict[str, Any]]:
    return [
        {
            "Configuration": point.configuration,
            "PPL": point.ppl,
            "PPL delta pct": REPORTED_ABLATION_PPL_DELTA_PCT[point.configuration],
            "PPL delta recomputed pct": ppl_delta_pct(point.ppl),
            "KV MB": point.kv_mb,
            "Compression": compression_ratio(point.kv_mb),
            "Effective bits": point.effective_bits,
            "Component": point.component,
        }
        for point in ABLATION_POINTS
    ]


def ablation_component_effects() -> list[dict[str, Any]]:
    by_name = {point.configuration: point for point in ABLATION_POINTS}

    comparisons = [
        (
            "Residual on uniform 2-bit",
            "A Uniform2 G32 R0",
            "B Uniform2 G32 R4",
        ),
        (
            "Affine-axis structure at K2/V2",
            "B Uniform2 G32 R4",
            "C K2/V2 affine R4",
        ),
        (
            "Protect K with 3 bits",
            "C K2/V2 affine R4",
            "F FINAL K3/V2 affine R4",
        ),
        (
            "Residual on final K3/V2",
            "E K3/V2 affine R0",
            "F FINAL K3/V2 affine R4",
        ),
        (
            "K3/V2 vs reversed K2/V3",
            "D K2/V3 affine R4",
            "F FINAL K3/V2 affine R4",
        ),
    ]

    effects: list[dict[str, Any]] = []
    for label, before_name, after_name in comparisons:
        before = by_name[before_name]
        after = by_name[after_name]
        effects.append(
            {
                "Comparison": label,
                "Before": before_name,
                "After": after_name,
                "PPL change": after.ppl - before.ppl,
                "KV MB change": after.kv_mb - before.kv_mb,
            }
        )
    return effects


def build_deployment_rows() -> list[dict[str, Any]]:
    bf16_name = "BF16 Triton"
    int3_name = "INT3 kvquant_k3 Triton"
    rows: list[dict[str, Any]] = []

    for context in (512, 2048, 4096):
        bf16_raw = DEPLOYMENT_RAW["trials"][bf16_name][context]
        int3_raw = DEPLOYMENT_RAW["trials"][int3_name][context]

        bf16_tpot = tpot_trials(bf16_raw["ttft_ms"], bf16_raw["total65_ms"])
        int3_tpot = tpot_trials(int3_raw["ttft_ms"], int3_raw["total65_ms"])

        bf16_ttft_median = median(bf16_raw["ttft_ms"])
        int3_ttft_median = median(int3_raw["ttft_ms"])
        bf16_tpot_median = median(bf16_tpot)
        int3_tpot_median = median(int3_tpot)

        rows.append(
            {
                "Context": context,
                "BF16 TTFT ms": bf16_ttft_median,
                "INT3 TTFT ms": int3_ttft_median,
                "BF16 TPOT ms": bf16_tpot_median,
                "INT3 TPOT ms": int3_tpot_median,
                "INT3/BF16 TTFT": int3_ttft_median / bf16_ttft_median,
                "INT3/BF16 TPOT": int3_tpot_median / bf16_tpot_median,
            }
        )

    return rows


# =============================================================================
# Validation
# =============================================================================

def assert_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isclose(actual, expected, abs_tol=tolerance, rel_tol=0.0):
        raise AssertionError(
            f"{label}: expected approximately {expected}, got {actual}."
        )


def validate_results() -> None:
    operating = {row["Method"]: row for row in build_operating_point_rows()}
    qasper = build_qasper_results()
    hotpot = build_hotpot_results()
    deployment = build_deployment_rows()

    assert_close(
        operating["RABIT-2 target"]["PPL delta pct"],
        2.6,
        0.01,
        "RABIT-2 reported PPL delta",
    )
    assert_close(
        operating["RABIT-2 target"]["PPL delta recomputed pct"],
        2.6,
        0.15,
        "RABIT-2 recomputed PPL delta from rounded PPL",
    )
    assert_close(
        operating["RABIT-2 target"]["Compression"],
        4.28,
        0.02,
        "RABIT-2 compression",
    )
    assert_close(
        qasper["BF16"]["f1"] * 100,
        35.5,
        0.1,
        "Qasper BF16 combined F1",
    )
    assert_close(
        qasper["RABIT-2 target"]["f1"] * 100,
        34.7,
        0.1,
        "Qasper RABIT-2 combined F1",
    )
    assert_close(
        hotpot["RABIT-3"]["f1"] * 100,
        60.9,
        0.1,
        "Hotpot RABIT-3 combined F1",
    )

    capacity = DEPLOYMENT_RAW["capacity_tokens"]
    capacity_ratio = (
        capacity["INT3 kvquant_k3 Triton"] / capacity["BF16 Triton"]
    )
    assert_close(capacity_ratio, 4.923, 0.002, "INT3 capacity ratio")

    expected_tpot_ratios = [3.032, 3.030, 2.980]
    for row, expected in zip(deployment, expected_tpot_ratios):
        assert_close(
            row["INT3/BF16 TPOT"],
            expected,
            0.003,
            f"TPOT ratio at context {row['Context']}",
        )


# =============================================================================
# Rendering
# =============================================================================

def render_markdown() -> str:
    operating = build_operating_point_rows()
    long_context = build_long_context_rows()
    ablation = build_ablation_rows()
    effects = ablation_component_effects()
    deployment = build_deployment_rows()
    qasper = build_qasper_results()
    hotpot = build_hotpot_results()

    operating_md_rows = [
        [
            row["Method"],
            row["Policy"],
            f"{row['PPL']:.2f}",
            format_percent(row["PPL delta pct"]),
            f"{row['KV MB']:.3f}",
            f"{row['Compression']:.2f}×",
            (
                f"{row['Effective bits']:.2f}"
                if row["Effective bits"] is not None
                else "—"
            ),
        ]
        for row in operating
    ]

    long_md_rows = []
    for row in long_context:
        continuation = (
            f"{row['Continuation PPL']:.3f}"
            if row["Continuation PPL"] is not None
            else "—"
        )
        qasper_f1 = (
            format_f1(row["Qasper F1"])
            if row["Qasper F1"] is not None
            else "—"
        )
        qasper_samples = (
            str(row["Qasper samples"])
            if row["Qasper samples"] is not None
            else "—"
        )
        hotpot_f1 = (
            format_f1(row["Hotpot F1"])
            if row["Hotpot F1"] is not None
            else "—"
        )
        hotpot_samples = (
            str(row["Hotpot samples"])
            if row["Hotpot samples"] is not None
            else "—"
        )
        long_md_rows.append(
            [
                row["Method"],
                continuation,
                qasper_f1,
                qasper_samples,
                hotpot_f1,
                hotpot_samples,
                row["NIAH 4K/8K/16K"],
                row["Passage retrieval"],
            ]
        )

    ablation_md_rows = [
        [
            row["Configuration"],
            f"{row['PPL']:.2f}",
            format_percent(row["PPL delta pct"]),
            f"{row['KV MB']:.3f}",
            f"{row['Compression']:.2f}×",
            f"{row['Effective bits']:.2f}",
            row["Component"],
        ]
        for row in ablation
    ]

    effect_md_rows = [
        [
            row["Comparison"],
            f"{row['PPL change']:+.2f}",
            f"{row['KV MB change']:+.3f}",
        ]
        for row in effects
    ]

    deployment_md_rows = [
        [
            row["Context"],
            f"{row['BF16 TTFT ms']:.3f}",
            f"{row['INT3 TTFT ms']:.3f}",
            f"{row['BF16 TPOT ms']:.3f}",
            f"{row['INT3 TPOT ms']:.3f}",
            f"{row['INT3/BF16 TTFT']:.3f}×",
            f"{row['INT3/BF16 TPOT']:.3f}×",
        ]
        for row in deployment
    ]

    capacity = DEPLOYMENT_RAW["capacity_tokens"]
    capacity_ratio = (
        capacity["INT3 kvquant_k3 Triton"] / capacity["BF16 Triton"]
    )

    uniform = next(
        row for row in ablation if row["Configuration"] == "A Uniform2 G32 R0"
    )
    final = next(
        row
        for row in ablation
        if row["Configuration"] == "F FINAL K3/V2 affine R4"
    )
    reduction_vs_uniform = (1.0 - final["PPL"] / uniform["PPL"]) * 100.0

    return f"""# RABIT-KV Final Consolidated Results

## Experimental scope

- Model: {MODEL}
- Production GPU: {GPU}
- The research tables report the frozen RABIT-KV quality policy.
- The production table reports the current INT3 `kvquant_k3` deployment proof-of-concept.
- The current INT3 kernel is not yet an exact deployment of the final K3/V2 affine + R4 policy.

## Table 1. Final operating points

{markdown_table(
    ["Method", "Policy", "PPL", "PPL Δ", "Logical KV MB", "Compression", "Effective bits"],
    operating_md_rows,
)}

**Main quality result:** the RABIT-KV 2-bit target operating point reaches
{final['PPL']:.2f} PPL, {format_percent(final['PPL delta pct'])} relative to BF16,
with {final['Compression']:.2f}× logical KV compression.

## Table 2. Long-context quality

{markdown_table(
    ["Method", "Continuation PPL", "Qasper F1", "Qasper N", "Hotpot F1", "Hotpot N", "NIAH", "Passage retrieval"],
    long_md_rows,
)}

Notes:

- Qasper uses the complete 24-example 8K+ bucket for BF16, RABIT-4, and RABIT-2.
- RABIT-3 Qasper was measured only on the final 14 examples and scored
  {format_f1(qasper['RABIT-3']['f1'])} F1; it is not directly comparable to the 24-example rows.
- HotpotQA uses 20 examples from the 8K+ bucket.
- The continuation-PPL test uses 1,024 context tokens, 128 continuation tokens, and one sample.

## Table 3. Two-bit component ablation

{markdown_table(
    ["Configuration", "PPL", "PPL Δ", "KV MB", "Compression", "Effective bits", "Component"],
    ablation_md_rows,
)}

### Component effects

{markdown_table(
    ["Comparison", "PPL change", "KV MB change"],
    effect_md_rows,
)}

The final policy reduces PPL by {reduction_vs_uniform:.1f}% relative to ordinary
uniform 2-bit quantization, from {uniform['PPL']:.2f} to {final['PPL']:.2f}.

## Table 4. Controlled production deployment

Both modes use the Triton attention backend, eager execution, no CUDA graphs,
and no `torch.compile`.

### Practical KV-cache capacity

| Format | Capacity | Relative capacity |
| --- | ---: | ---: |
| BF16 Triton | {capacity['BF16 Triton']:,} tokens | 1.000× |
| INT3 `kvquant_k3` Triton | {capacity['INT3 kvquant_k3 Triton']:,} tokens | {capacity_ratio:.3f}× |

### TTFT and TPOT

{markdown_table(
    ["Context", "BF16 TTFT ms", "INT3 TTFT ms", "BF16 TPOT ms", "INT3 TPOT ms", "INT3/BF16 TTFT", "INT3/BF16 TPOT"],
    deployment_md_rows,
)}

TTFT is approximated by a one-output-token request. TPOT is calculated as:

`(latency for 65 output tokens - latency for 1 output token) / 64`

**Deployment result:** the current INT3 kernel increases practical KV-cache
capacity by {capacity_ratio:.3f}×, but TPOT is approximately 3× higher than
the controlled BF16 Triton path.

## Final defensible claims

1. The RABIT-KV 2-bit target operating point reaches 11.51 PPL, only 2.6%
   above BF16, with 4.28× logical KV compression.
2. The final policy reduces PPL by {reduction_vs_uniform:.1f}% relative to
   ordinary uniform 2-bit quantization.
3. RABIT-3 maintains BF16-level HotpotQA performance on the current 20-example
   8K+ evaluation: {format_f1(hotpot['RABIT-3']['f1'])} versus
   {format_f1(hotpot['BF16']['f1'])} F1.
4. RABIT-2 remains close to BF16 on the full Qasper 8K+ bucket:
   {format_f1(qasper['RABIT-2 target']['f1'])} versus
   {format_f1(qasper['BF16']['f1'])} F1.
5. The current INT3 vLLM implementation increases practical KV-cache capacity
   by {capacity_ratio:.3f}×.
6. The current INT3 Triton kernel has approximately 3× higher TPOT than the
   controlled BF16 Triton path.

## Required limitations

- “2-bit target” is a target operating point, not literal 2-bit physical
  storage. The final policy uses about 3.74 effective bits after metadata and
  residual storage.
- The current production kernel is INT3 and is not yet an exact implementation
  of K3/V2 affine + R4.
- Small F1 differences should not be described as statistically significant.
- Current LongBench conclusions must retain their sample counts.
"""


def render_console() -> str:
    operating = build_operating_point_rows()
    qasper = build_qasper_results()
    hotpot = build_hotpot_results()
    ablation = build_ablation_rows()
    deployment = build_deployment_rows()

    sections: list[str] = []

    sections.append("RABIT-KV FINAL CONSOLIDATED RESULTS")
    sections.append("=" * 88)

    sections.append("\n1. FINAL OPERATING POINTS")
    sections.append(
        plain_table(
            ["Method", "PPL", "PPL Δ", "KV MB", "Compression", "EffBits"],
            [
                [
                    row["Method"],
                    f"{row['PPL']:.2f}",
                    format_percent(row["PPL delta pct"]),
                    f"{row['KV MB']:.3f}",
                    f"{row['Compression']:.2f}x",
                    (
                        f"{row['Effective bits']:.2f}"
                        if row["Effective bits"] is not None
                        else "-"
                    ),
                ]
                for row in operating
            ],
        )
    )

    sections.append("\n2. LONGBENCH-E 8K+")
    sections.append(
        plain_table(
            ["Method", "Qasper F1", "Qasper N", "Hotpot F1", "Hotpot N"],
            [
                [
                    method,
                    format_f1(qasper[method]["f1"]),
                    qasper[method]["samples"],
                    (
                        format_f1(hotpot[method]["f1"])
                        if method in hotpot
                        else "-"
                    ),
                    hotpot[method]["samples"] if method in hotpot else "-",
                ]
                for method in ("BF16", "RABIT-4", "RABIT-2 target")
            ]
            + [
                [
                    "RABIT-3",
                    format_f1(qasper["RABIT-3"]["f1"]) + "*",
                    qasper["RABIT-3"]["samples"],
                    format_f1(hotpot["RABIT-3"]["f1"]),
                    hotpot["RABIT-3"]["samples"],
                ]
            ],
        )
    )
    sections.append("* RABIT-3 Qasper covers only the final 14 examples.")

    sections.append("\n3. TWO-BIT ABLATION")
    sections.append(
        plain_table(
            ["Configuration", "PPL", "PPL Δ", "KV MB", "Compression", "EffBits"],
            [
                [
                    row["Configuration"],
                    f"{row['PPL']:.2f}",
                    format_percent(row["PPL delta pct"]),
                    f"{row['KV MB']:.3f}",
                    f"{row['Compression']:.2f}x",
                    f"{row['Effective bits']:.2f}",
                ]
                for row in ablation
            ],
        )
    )

    sections.append("\n4. PRODUCTION TTFT / TPOT")
    sections.append(
        plain_table(
            ["Context", "BF16 TTFT", "INT3 TTFT", "BF16 TPOT", "INT3 TPOT", "TPOT ratio"],
            [
                [
                    row["Context"],
                    f"{row['BF16 TTFT ms']:.3f}",
                    f"{row['INT3 TTFT ms']:.3f}",
                    f"{row['BF16 TPOT ms']:.3f}",
                    f"{row['INT3 TPOT ms']:.3f}",
                    f"{row['INT3/BF16 TPOT']:.3f}x",
                ]
                for row in deployment
            ],
        )
    )

    capacity = DEPLOYMENT_RAW["capacity_tokens"]
    ratio = capacity["INT3 kvquant_k3 Triton"] / capacity["BF16 Triton"]
    sections.append(
        f"\nKV capacity: {capacity['BF16 Triton']:,} BF16 tokens -> "
        f"{capacity['INT3 kvquant_k3 Triton']:,} INT3 tokens ({ratio:.3f}x)."
    )

    return "\n".join(sections)


# =============================================================================
# Export
# =============================================================================

def write_csv(path: Path, headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def export_all(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    operating = build_operating_point_rows()
    long_context = build_long_context_rows()
    ablation = build_ablation_rows()
    effects = ablation_component_effects()
    deployment = build_deployment_rows()
    qasper = build_qasper_results()
    hotpot = build_hotpot_results()

    markdown_path = output_dir / "RABIT_KV_Final_Result_Tables.md"
    markdown_path.write_text(render_markdown(), encoding="utf-8")

    json_payload = {
        "metadata": {
            "model": MODEL,
            "gpu": GPU,
            "baseline_ppl": BASELINE_PPL,
            "baseline_kv_mb": BASELINE_KV_MB,
        },
        "operating_points": operating,
        "continuation_ppl": CONTINUATION_PPL,
        "niah": NIAH,
        "passage_retrieval": PASSAGE_RETRIEVAL,
        "qasper": qasper,
        "hotpotqa": hotpot,
        "ablation": ablation,
        "ablation_component_effects": effects,
        "deployment": {
            "environment": DEPLOYMENT_RAW["environment"],
            "capacity_tokens": DEPLOYMENT_RAW["capacity_tokens"],
            "derived_rows": deployment,
            "raw_trials": DEPLOYMENT_RAW["trials"],
        },
    }

    json_path = output_dir / "RABIT_KV_Final_Results.json"
    json_path.write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    main_csv = output_dir / "RABIT_KV_Main_Results.csv"
    write_csv(
        main_csv,
        ["Method", "Policy", "PPL", "PPL delta pct", "KV MB", "Compression", "Effective bits"],
        [
            [
                row["Method"],
                row["Policy"],
                row["PPL"],
                row["PPL delta pct"],
                row["KV MB"],
                row["Compression"],
                row["Effective bits"],
            ]
            for row in operating
        ],
    )

    long_csv = output_dir / "RABIT_KV_LongContext_Results.csv"
    write_csv(
        long_csv,
        [
            "Method",
            "Continuation PPL",
            "Qasper samples",
            "Qasper F1",
            "Hotpot samples",
            "Hotpot F1",
            "NIAH 4K/8K/16K",
            "Passage retrieval",
        ],
        [
            [
                row["Method"],
                row["Continuation PPL"],
                row["Qasper samples"],
                row["Qasper F1"],
                row["Hotpot samples"],
                row["Hotpot F1"],
                row["NIAH 4K/8K/16K"],
                row["Passage retrieval"],
            ]
            for row in long_context
        ],
    )

    ablation_csv = output_dir / "RABIT_KV_2bit_Ablation.csv"
    write_csv(
        ablation_csv,
        [
            "Configuration",
            "PPL",
            "PPL delta pct",
            "KV MB",
            "Compression",
            "Effective bits",
            "Component",
        ],
        [
            [
                row["Configuration"],
                row["PPL"],
                row["PPL delta pct"],
                row["KV MB"],
                row["Compression"],
                row["Effective bits"],
                row["Component"],
            ]
            for row in ablation
        ],
    )

    deployment_csv = output_dir / "RABIT_KV_Deployment_Latency.csv"
    write_csv(
        deployment_csv,
        [
            "Context",
            "BF16 TTFT ms",
            "INT3 TTFT ms",
            "BF16 TPOT ms",
            "INT3 TPOT ms",
            "INT3/BF16 TTFT",
            "INT3/BF16 TPOT",
        ],
        [
            [
                row["Context"],
                row["BF16 TTFT ms"],
                row["INT3 TTFT ms"],
                row["BF16 TPOT ms"],
                row["INT3 TPOT ms"],
                row["INT3/BF16 TTFT"],
                row["INT3/BF16 TPOT"],
            ]
            for row in deployment
        ],
    )

    return [
        markdown_path,
        json_path,
        main_csv,
        long_csv,
        ablation_csv,
        deployment_csv,
    ]


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute, validate, print, and export all frozen RABIT-KV results."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for generated Markdown, JSON, and CSV files.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Print and validate only; do not write output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    validate_results()
    print(render_console())
    print("\nValidation: PASSED")

    if not args.no_export:
        created = export_all(args.output_dir)
        print("\nCreated files:")
        for path in created:
            print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
