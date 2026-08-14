"""Tests for vllm_plugin configuration and specs (no GPU needed)."""
import unittest
from kvquant.vllm_plugin.config import KVQuantConfig, KVQUANT_DTYPE_K4, KVQUANT_NBITS_TO_DTYPE
from kvquant.vllm_plugin.attention_spec import KVQuantAttentionSpec


class TestKVQuantConfig(unittest.TestCase):
    def test_kv_cache_dtype_mapping(self):
        self.assertEqual(KVQUANT_NBITS_TO_DTYPE[4], "kvquant_k4")
        self.assertEqual(KVQUANT_NBITS_TO_DTYPE[3], "kvquant_k3")
        self.assertEqual(KVQUANT_NBITS_TO_DTYPE[8], "kvquant_k8")
        self.assertEqual(KVQUANT_NBITS_TO_DTYPE[2], "kvquant_k2")

    def test_page_size_bytes(self):
        cfg = KVQuantConfig(nbits=4, num_kv_heads=4, head_dim=64, block_size=16)
        self.assertEqual(cfg.packed_page_size_bytes, 4096)
        self.assertEqual(cfg.scale_page_size_bytes, 512)
        self.assertEqual(cfg.page_size_bytes, 4608)

    def test_three_bit_layout(self):
        cfg = KVQuantConfig(nbits=3, num_kv_heads=4, head_dim=64, block_size=16)
        self.assertEqual(cfg.kv_cache_dtype, "kvquant_k3")
        self.assertEqual(cfg.packed_page_size_bytes, 3072)
        self.assertEqual(cfg.scale_page_size_bytes, 512)
        self.assertEqual(cfg.page_size_bytes, 3584)
        self.assertEqual(cfg.memory_layout()["scale_strategy"], "per_token_head")

    def test_three_bit_layout_rounds_each_head_to_bytes(self):
        cfg = KVQuantConfig(nbits=3, num_kv_heads=1, head_dim=9, block_size=1)
        self.assertEqual(cfg.packed_page_size_bytes, 8)

    def test_bytes_per_element(self):
        self.assertAlmostEqual(KVQuantConfig(nbits=4).bytes_per_element, 0.5)
        self.assertAlmostEqual(KVQuantConfig(nbits=3).bytes_per_element, 0.375)
        self.assertAlmostEqual(KVQuantConfig(nbits=8).bytes_per_element, 1.0)
        self.assertAlmostEqual(KVQuantConfig(nbits=2).bytes_per_element, 0.25)


class TestKVQuantAttentionSpec(unittest.TestCase):
    def test_dtype_str(self):
        spec = KVQuantAttentionSpec(num_kv_heads=4, head_size=64, nbits=4)
        self.assertEqual(spec.dtype_str, "kvquant_k4")
        self.assertEqual(spec.page_size_bytes, 4608)


if __name__ == "__main__":
    unittest.main()
