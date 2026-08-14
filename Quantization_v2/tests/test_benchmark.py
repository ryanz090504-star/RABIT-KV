"""Tests for benchmark modules (no GPU needed)."""

import unittest
from unittest.mock import patch

from kvquant.cli import main
from kvquant.benchmark.memory import estimate_memory
from kvquant.benchmark.tasks import (
    TaskResult,
    _exact_match,
    _expand_context_lens,
    _normalize_answer,
)
from kvquant.benchmark.latency import run_vllm_benchmark
from kvquant.policies import build_policy
from kvquant.vllm_runner import VLLMDeployResult


class TestMemoryBenchmark(unittest.TestCase):
    def test_estimate_basic(self):
        p = build_policy("document_naive", nbits=4)
        r = estimate_memory(
            "test", p, layers=22, kv_heads=4, head_dim=64, batch_size=8, seq_len=4096
        )
        self.assertGreater(r.baseline_cache_gb, 0)
        self.assertGreater(r.estimated_cache_gb, 0)
        self.assertGreater(r.memory_reduction, 0)
        self.assertIn("packed_kv_bytes", r.memory_breakdown)
        self.assertEqual(r.kv_source, "analytical_estimate")

    def test_eight_bit_saves_half(self):
        p = build_policy("document_naive", nbits=8)
        r = estimate_memory(
            "test", p, layers=22, kv_heads=4, head_dim=64, batch_size=8, seq_len=4096
        )
        self.assertAlmostEqual(r.memory_reduction, 0.5, delta=0.01)

    def test_int3_memory_breakdown_includes_scale_overhead(self):
        p = build_policy("kvquant_int3", nbits=3)
        r = estimate_memory(
            "test", p, layers=2, kv_heads=2, head_dim=8, batch_size=1, seq_len=16
        )
        self.assertEqual(r.kv_cache_dtype, "kvquant_k3")
        self.assertGreater(r.memory_breakdown["scale_bytes"], 0)
        self.assertGreater(r.memory_breakdown["allocator_page_overhead_bytes"], 0)

    def test_turbo_memory_breakdown_includes_qjl(self):
        p = build_policy("turbo_int3", nbits=3)
        r = estimate_memory(
            "test", p, layers=2, kv_heads=2, head_dim=8, batch_size=1, seq_len=16
        )
        self.assertGreater(r.memory_breakdown["codebook_bytes"], 0)
        self.assertGreater(r.memory_breakdown["qjl_residual_bytes"], 0)


class TestTaskResult(unittest.TestCase):
    def test_accuracy_recovery(self):
        r = TaskResult(
            dataset="test",
            task="needle",
            metric="accuracy",
            policy_name="doc_naive",
            nbits=4,
            baseline_score=0.9,
            quantized_score=0.72,
            accuracy_recovery=0.8,
        )
        self.assertEqual(r.evidence_label, "quality_task_not_deploy_speedup")
        self.assertAlmostEqual(r.accuracy_recovery, 0.8)

    def test_exact_match_normalization(self):
        self.assertEqual(_normalize_answer("The Answer: Blue-17!"), "answer blue 17")
        self.assertTrue(_exact_match("The hidden code is Blue-17.", "blue 17"))
        self.assertFalse(_exact_match("The hidden code is red.", "blue 17"))

    def test_context_lens_expansion(self):
        rows = _expand_context_lens([{"id": "a"}, {"id": "b"}], [128, 256])
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["_context_len"] for row in rows}, {128, 256})


class TestLatencyGuards(unittest.TestCase):
    def test_vllm_backend_requires_kvquant_k3(self):
        with self.assertRaisesRegex(NotImplementedError, "kvquant_k3"):
            run_vllm_benchmark("dummy-model", "document_naive", 4)

    def test_vllm_backend_requires_vllm_root(self):
        with self.assertRaisesRegex(ValueError, "vLLM deploy latency requires"):
            run_vllm_benchmark(
                "dummy-model", "kvquant_int3", 3, kv_cache_dtype="kvquant_k3"
            )

    def test_vllm_backend_delegates_to_deploy_runner(self):
        deploy = VLLMDeployResult(
            model="dummy-model",
            policy_name="kvquant_int3",
            policy_signature="sig",
            nbits=3,
            kv_cache_dtype="kvquant_k3",
            hardware="test-gpu",
            vllm_commit="abc123",
            ttft_ms=12.0,
            tpot_ms=3.0,
            request_throughput=4.0,
            tokens_per_second=5.0,
            memory_breakdown={"page_size_bytes": 123},
        )
        with patch("kvquant.vllm_runner.run_vllm_benchmark", return_value=deploy):
            result = run_vllm_benchmark(
                "dummy-model",
                "kvquant_int3",
                3,
                vllm_root="/tmp/vllm",
                kv_cache_dtype="kvquant_k3",
            )
        self.assertEqual(result.backend, "vllm_bench_serve")
        self.assertEqual(result.evidence_label, "deploy_latency")
        self.assertEqual(result.kv_source, "vllm_packed_kv_cache")
        self.assertEqual(result.tpot_ms, 3.0)

    def test_cli_vllm_backend_does_not_fall_back_to_triton(self):
        with patch(
            "kvquant.vllm_runner.run_vllm_benchmark",
            side_effect=RuntimeError("vllm path"),
        ):
            with self.assertRaisesRegex(RuntimeError, "vllm path"):
                main(
                    [
                        "latency",
                        "--backend",
                        "vllm",
                        "--model",
                        "dummy",
                        "--policy",
                        "kvquant_int3",
                        "--nbits",
                        "3",
                        "--vllm-root",
                        "/tmp/vllm",
                        "--kv-cache-dtype",
                        "kvquant_k3",
                    ]
                )

    def test_cli_policy_construction_handles_non_nbits_policies(self):
        main(["memory", "--model", "dummy", "--policy", "no_quant", "--nbits", "4"])
        main(
            [
                "memory",
                "--model",
                "dummy",
                "--policy",
                "attention_mixed",
                "--nbits",
                "3",
            ]
        )


if __name__ == "__main__":
    unittest.main()
