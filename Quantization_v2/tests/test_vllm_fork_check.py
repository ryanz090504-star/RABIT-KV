from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kvquant.vllm_plugin.fork_check import check_vllm_fork


class TestVLLMForkCheck(unittest.TestCase):
    def test_missing_root_fails(self):
        result = check_vllm_fork("/missing/vllm/root/for/kvquant")
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["checks"][0]["name"], "root_exists")

    def test_minimal_ready_tree_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text("[project]\nname='vllm'\n", encoding="utf-8")
            pkg = root / "vllm"
            pkg.mkdir()
            (pkg / "kvquant_k3.py").write_text(
                """
KV_DTYPE = "kvquant_k3"
def cache_layout(block_size, page_size, scale_strategy="per_token_head"):
    return page_size, block_size, scale_strategy
def reshape_and_cache_kvquant_k3(cache):
    return pack(quantize(cache))
def unified_attention_kvquant_k3(attention, cache):
    return dequant(unpack(cache))
def bench_serve_kv_cache_dtype():
    return "kv_cache_dtype=kvquant_k3"
""",
                encoding="utf-8",
            )
            result = check_vllm_fork(root)
        self.assertEqual(result["status"], "pass", result)
        self.assertGreater(result["scanned_files"], 0)

    def test_dtype_only_tree_fails_runtime_surfaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text("[project]\nname='vllm'\n", encoding="utf-8")
            pkg = root / "vllm"
            pkg.mkdir()
            (pkg / "config.py").write_text('KV_DTYPE = "kvquant_k3"\n', encoding="utf-8")
            result = check_vllm_fork(root)
        self.assertEqual(result["status"], "fail")
        failed = {row["name"] for row in result["checks"] if row["status"] == "fail"}
        self.assertIn("cache_write_quantizes_int3", failed)
        self.assertIn("attention_reads_packed_int3", failed)


if __name__ == "__main__":
    unittest.main()
