"""Paper evidence audit: validate JSONL rows against evidence gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_result_files(inputs: list[str]) -> dict[str, Any]:
    """Validate JSONL result files against paper evidence gates.

    Checks:
    - policy_signature present on all policy rows
    - deploy rows have matching quality rows (linked by signature)
    - deploy rows have real latency metrics and are not fallback/kernel rows
    - memory rows have both baseline and estimated values
    """
    rows = _read_all(inputs)
    checks = []

    # ── policy_signatures_present ──
    sig_required_labels = {
        "quality_reference_not_deploy_speedup",
        "deploy_latency",
        "kernel_latency_not_deploy_speedup",
        "memory_estimate_not_runtime_peak",
        "quality_task_not_deploy_speedup",
    }
    sig_rows = [r for r in rows if r.get("evidence_label") in sig_required_labels]
    missing_sig = [r for r in sig_rows if not r.get("policy_signature")]
    checks.append({
        "name": "policy_signatures_present",
        "status": "pass" if not missing_sig else "fail",
        "message": (
            "所有策略行都有 policy_signature"
            if not missing_sig
            else f"{len(missing_sig)} 行缺少 policy_signature"
        ),
    })

    # ── deploy_quality_trace ──
    deploy_rows = [r for r in rows if r.get("evidence_label") == "deploy_latency"]
    quality_sigs = {r.get("policy_signature") for r in rows if r.get("evidence_label", "").startswith("quality")}
    unmatched = [r for r in deploy_rows if r.get("policy_signature") not in quality_sigs]
    checks.append({
        "name": "deploy_quality_trace",
        "status": "pass" if not unmatched else "fail",
        "message": (
            "无 deploy 行需要质量证据"
            if not deploy_rows
            else "每个 deploy 行都有匹配的质量证据"
            if not unmatched
            else f"缺少质量证据的 deploy 签名: {', '.join(r.get('policy_signature','?') for r in unmatched)}"
        ),
    })

    # ── deploy_metrics_real ──
    bad_deploy = [
        r for r in deploy_rows
        if not _positive_number(r.get("tpot_ms"))
        or _attention_impl(r) == "torch_unpack_dequant_attention"
        or r.get("backend") != "vllm_bench_serve"
        or not r.get("kv_cache_dtype")
        or not r.get("hardware")
        or not r.get("vllm_commit")
        or not r.get("quant_kernel")
        or r.get("kv_source") == "synthetic_random"
    ]
    checks.append({
        "name": "deploy_metrics_real",
        "status": "pass" if not bad_deploy else "fail",
        "message": (
            "无 deploy 行需要真实延迟检查"
            if not deploy_rows
            else "deploy 行包含正 latency，且不是 fallback/synthetic"
            if not bad_deploy
            else f"{len(bad_deploy)} 个 deploy 行缺少真实部署证据"
        ),
    })

    # ── kernel_rows_stay_kernel ──
    bad_kernel_rows = [
        r for r in rows
        if r.get("evidence_label") == "kernel_latency_not_deploy_speedup"
        and r.get("backend") == "vllm_bench_serve"
    ]
    checks.append({
        "name": "kernel_rows_stay_kernel",
        "status": "pass" if not bad_kernel_rows else "fail",
        "message": (
            "kernel 诊断没有混入 vLLM serving backend"
            if not bad_kernel_rows
            else f"{len(bad_kernel_rows)} 个 kernel 行使用了 vLLM serving backend"
        ),
    })

    # ── memory_estimates_present ──
    mem_rows = [r for r in rows if r.get("evidence_label") == "memory_estimate_not_runtime_peak"]
    bad_mem = [
        r for r in mem_rows
        if not r.get("baseline_cache_gb")
        or not r.get("estimated_cache_gb")
        or not isinstance(r.get("memory_breakdown"), dict)
        or not r.get("memory_breakdown", {}).get("total_estimated_bytes")
    ]
    checks.append({
        "name": "memory_estimates_present",
        "status": "pass" if not bad_mem else "fail",
        "message": (
            "内存估算行包含 baseline 和 estimated 缓存大小"
            if not bad_mem
            else f"{len(bad_mem)} 内存行缺少必要字段或 breakdown"
        ),
    })

    # ── schema_fields_present ──
    schema_rows = [r for r in rows if r.get("evidence_label") in sig_required_labels]
    required = {"model", "policy_name", "policy_signature", "nbits", "kv_cache_dtype", "kv_source", "evidence_label"}
    missing_schema = [
        r for r in schema_rows
        if any(field not in r or r.get(field) is None for field in required)
    ]
    checks.append({
        "name": "result_schema_fields_present",
        "status": "pass" if not missing_schema else "fail",
        "message": (
            "结果行包含论文 schema 的核心字段"
            if not missing_schema
            else f"{len(missing_schema)} 行缺少核心 schema 字段"
        ),
    })

    # ── paper_readiness_checklist ──
    paper_checks = _paper_readiness(rows)
    checks.extend(paper_checks)

    all_pass = all(c["status"] == "pass" for c in checks)
    return {
        "status": "pass" if all_pass else "fail",
        "rows": len(rows),
        "checks": checks,
    }


def _paper_readiness(rows: list[dict]) -> list[dict[str, Any]]:
    """Checklist for paper submission readiness.

    These checks are advisory — they only fail when the relevant row types
    exist but are missing expected information. A file containing only kernel
    latency rows is not expected to have quality baselines.
    """
    results: list[dict[str, Any]] = []

    # 1. Baseline (fp16) present in quality rows (if quality rows exist)
    quality_rows = [r for r in rows if r.get("evidence_label") == "quality_reference_not_deploy_speedup"]
    has_baseline = any(
        r.get("nbits") in (16, None) and (r.get("ppl") or r.get("loss"))
        for r in quality_rows
    )
    results.append({
        "name": "paper_baseline_quality_present",
        "status": "pass" if not quality_rows or has_baseline else "fail",
        "message": (
            "基准质量行 (fp16) 存在或无需检查"
            if not quality_rows or has_baseline
            else "质量行缺少 fp16 基准"
        ),
    })

    # 2. At least one 3-bit policy (if quality rows exist)
    has_3bit = any(
        r.get("nbits") == 3 for r in quality_rows
    )
    results.append({
        "name": "paper_3bit_results_present",
        "status": "pass" if not quality_rows or has_3bit else "fail",
        "message": (
            "3-bit 质量结果存在或无需检查"
            if not quality_rows or has_3bit
            else "质量行缺少 3-bit 结果"
        ),
    })

    # 3. Error breakdown present for quantized quality rows (if any quantized rows)
    quant_rows = [r for r in quality_rows if r.get("nbits") not in (16, None)]
    has_error = any(r.get("error_cosine_similarity") is not None for r in quant_rows)
    results.append({
        "name": "paper_error_breakdown_present",
        "status": "pass" if not quant_rows or has_error else "fail",
        "message": "量化误差分析存在或无需检查" if not quant_rows or has_error else "量化行缺少误差分析",
    })

    # 4. Memory estimates (if memory rows exist)
    mem_rows = [r for r in rows if r.get("evidence_label") == "memory_estimate_not_runtime_peak"]
    has_mem_baseline = any(r.get("nbits") in (16, None) for r in mem_rows)
    has_mem_quant = any(r.get("nbits") not in (16, None) for r in mem_rows)
    results.append({
        "name": "paper_memory_coverage",
        "status": "pass" if not mem_rows or (has_mem_baseline and has_mem_quant) else "fail",
        "message": (
            "内存分析覆盖基准和量化方案或无需检查"
            if not mem_rows or (has_mem_baseline and has_mem_quant)
            else f"内存分析不完整 (baseline={has_mem_baseline}, quant={has_mem_quant})"
        ),
    })

    # 5. Policy coverage: check policies across all rows
    policies_tested = set(
        r.get("policy_name") for r in rows if r.get("policy_name")
    )
    results.append({
        "name": "paper_policy_diversity",
        "status": "pass" if len(policies_tested) >= 1 else "fail",
        "message": (
            f"测试了 {len(policies_tested)} 种策略"
            if len(policies_tested) >= 1
            else "没有测试任何策略"
        ),
    })

    # 6. No deploy_latency claims from reference path
    kernel_rows = [r for r in rows if r.get("evidence_label") == "kernel_latency_not_deploy_speedup"]
    deploy_rows = [r for r in rows if r.get("evidence_label") == "deploy_latency"]
    bad_kernel = [
        r for r in kernel_rows
        if r.get("backend") == "vllm_bench_serve"
    ]
    results.append({
        "name": "paper_evidence_labels_correct",
        "status": "pass" if not bad_kernel else "fail",
        "message": (
            "证据标签正确: kernel 诊断不冒充 deploy"
            if not bad_kernel
            else f"{len(bad_kernel)} 个 kernel 行误标为 deploy"
        ),
    })

    return results


def _read_all(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _positive_number(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _attention_impl(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        return None
    impl = metadata.get("attention_impl")
    return str(impl) if impl is not None else None
