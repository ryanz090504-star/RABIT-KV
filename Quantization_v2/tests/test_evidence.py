"""Tests for evidence labels, reporting, and validation guardrails."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kvquant.report import generate_report
from kvquant.validation import validate_result_files


class EvidenceGuardTests(unittest.TestCase):
    def test_kernel_fallback_is_not_deploy_latency(self):
        row = {
            "backend": "triton_packed_attention",
            "model": "dummy",
            "policy_name": "document_naive",
            "policy_signature": "abc123",
            "nbits": 4,
            "kv_cache_dtype": "kvquant_k4",
            "tpot_ms": 1.2,
            "baseline_tpot_ms": 1.0,
            "tpot_improvement": -0.2,
            "kv_source": "synthetic_random",
            "evidence_label": "kernel_latency_not_deploy_speedup",
            "metadata": {"attention_impl": "torch_unpack_dequant_attention"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kernel.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = validate_result_files([str(path)])
        self.assertEqual(result["status"], "pass")

    def test_deploy_fallback_row_fails_validation(self):
        row = {
            "backend": "triton_packed_attention",
            "model": "dummy",
            "policy_name": "document_naive",
            "policy_signature": "abc123",
            "nbits": 4,
            "kv_cache_dtype": "kvquant_k4",
            "tpot_ms": 1.2,
            "baseline_tpot_ms": 1.0,
            "kv_source": "synthetic_random",
            "backend": "vllm_bench_serve",
            "hardware": "A100",
            "vllm_commit": "abc123",
            "quant_kernel": "torch_unpack_dequant_attention",
            "evidence_label": "deploy_latency",
            "metadata": {"attention_impl": "torch_unpack_dequant_attention"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deploy.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = validate_result_files([str(path)])
        self.assertEqual(result["status"], "fail")

    def test_report_labels_kernel_section_as_not_serving_speedup(self):
        row = {
            "backend": "triton_packed_attention",
            "model": "dummy",
            "policy_name": "document_naive",
            "policy_signature": "abc123",
            "nbits": 4,
            "kv_cache_dtype": "kvquant_k4",
            "tpot_ms": 1.2,
            "baseline_tpot_ms": 1.0,
            "tokens_per_second": 833.3,
            "kv_source": "synthetic_random",
            "evidence_label": "kernel_latency_not_deploy_speedup",
            "metadata": {"attention_impl": "torch_unpack_dequant_attention"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "kernel.jsonl"
            out = Path(tmp) / "report.md"
            src.write_text(json.dumps(row) + "\n", encoding="utf-8")
            generate_report([str(src)], str(out))
            text = out.read_text(encoding="utf-8")
        self.assertIn("Kernel 延迟诊断 (Not Serving Speedup)", text)
        self.assertIn("torch_unpack_dequant_attention", text)


if __name__ == "__main__":
    unittest.main()
