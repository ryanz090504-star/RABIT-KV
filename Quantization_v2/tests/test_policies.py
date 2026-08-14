import unittest

import numpy as np

from kvquant.policies import (
    build_policy,
    list_policies,
    policy_nbits,
    policy_quantization_scope,
    policy_signature,
    policy_spec,
)
from kvquant.types import ModalitySpan, QuantizationContext, TokenType


class PolicyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.keys = rng.normal(size=(1, 4, 32, 8)).astype(np.float32)
        self.values = rng.normal(size=(1, 4, 32, 8)).astype(np.float32)

    def test_registry_contains_expected_policies(self):
        names = set(list_policies())
        self.assertIn("document_naive", names)
        self.assertIn("naive", names)
        self.assertIn("per_head", names)
        self.assertIn("attention_mixed", names)
        self.assertIn("kvquant_int3", names)
        self.assertIn("polar_int3", names)
        self.assertIn("turbo_int3", names)

    def test_document_naive_uses_one_global_scale_per_kv_tensor(self):
        policy = build_policy("document_naive", nbits=4)
        block = policy.quantize(self.keys, self.values, QuantizationContext(layer_idx=0))
        self.assertEqual(block.metadata["axis_strategy"], "global")
        self.assertEqual(block.key.minimum.shape, (1, 1, 1, 1))
        self.assertEqual(block.key.scale.shape, (1, 1, 1, 1))
        self.assertEqual(block.value.minimum.shape, (1, 1, 1, 1))
        self.assertEqual(block.value.scale.shape, (1, 1, 1, 1))
        self.assertEqual(block.metadata["policy"], "document_naive")

    def test_policy_signature_tracks_algorithm_configuration(self):
        four_bit = build_policy("document_naive", nbits=4)
        same_four_bit = build_policy("document_naive", nbits=4)
        eight_bit = build_policy("document_naive", nbits=8)

        self.assertEqual(policy_signature(four_bit), policy_signature(same_four_bit))
        self.assertNotEqual(policy_signature(four_bit), policy_signature(eight_bit))
        self.assertEqual(policy_spec(four_bit)["name"], "document_naive")
        self.assertEqual(policy_spec(four_bit)["parameters"]["nbits"], 4)

    def test_policy_table_metadata_reports_bits_and_scope(self):
        self.assertEqual(policy_nbits(build_policy("document_naive", nbits=4)), 4)
        self.assertEqual(policy_quantization_scope(build_policy("document_naive", nbits=4)), "K+V")
        self.assertEqual(
            policy_quantization_scope(
                build_policy("document_naive", nbits=8, quantize_keys=True, quantize_values=False)
            ),
            "K-only",
        )
        self.assertEqual(
            policy_quantization_scope(
                build_policy("document_naive", nbits=8, quantize_keys=False, quantize_values=True)
            ),
            "V-only",
        )
        self.assertEqual(policy_quantization_scope(build_policy("no_quant")), "none")

    def test_per_head_policy_dequantizes(self):
        policy = build_policy("per_head", nbits=4)
        block = policy.quantize(self.keys, self.values, QuantizationContext(layer_idx=2))
        key_deq, value_deq = block.dequantize()
        self.assertEqual(key_deq.shape, self.keys.shape)
        self.assertEqual(value_deq.shape, self.values.shape)
        self.assertGreater(block.compression_ratio(), 1.0)

    def test_no_quant_reports_full_precision_size(self):
        policy = build_policy("no_quant")
        block = policy.quantize(self.keys, self.values, QuantizationContext(layer_idx=0))
        self.assertEqual(block.estimated_payload_nbytes(), block.original_nbytes())
        self.assertEqual(block.compression_ratio(), 1.0)

    def test_attention_policy_uses_attention_scores(self):
        attention = np.zeros((1, 4, 32), dtype=np.float32)
        attention[:, :, -1] = 10
        policy = build_policy("attention_mixed", keep_ratio=0.125)
        block = policy.quantize(self.keys, self.values, QuantizationContext(layer_idx=0, attention_scores=attention))
        self.assertIn("high_token_fraction", block.metadata)
        self.assertGreater(block.metadata["high_token_fraction"], 0)
        self.assertGreater(block.compression_ratio(), 1.0)

    def test_kvquant_int3_uses_per_token_head_scale(self):
        policy = build_policy("kvquant_int3", nbits=3)
        block = policy.quantize(self.keys, self.values, QuantizationContext(layer_idx=0))
        self.assertEqual(block.metadata["axis_strategy"], "per_token_head")
        self.assertEqual(block.key.minimum.shape, (1, 4, 32, 1))
        self.assertEqual(block.key.nbits, 3)
        key_deq, value_deq = block.to_packed().dequantize()
        self.assertEqual(key_deq.shape, self.keys.shape)
        self.assertEqual(value_deq.shape, self.values.shape)

    def test_turbo_int3_adds_residual_payload(self):
        polar = build_policy("polar_int3", nbits=3)
        turbo = build_policy("turbo_int3", nbits=3)
        polar_block = polar.quantize(self.keys, self.values, QuantizationContext(layer_idx=0))
        turbo_block = turbo.quantize(self.keys, self.values, QuantizationContext(layer_idx=0))
        self.assertGreater(turbo_block.estimated_payload_nbytes(), polar_block.estimated_payload_nbytes())
        self.assertIn("not_full_turboquant_reproduction", turbo_block.metadata["claim_boundary"])

    def test_modality_span(self):
        span = ModalitySpan(start=4, end=8, token_type=TokenType.VISUAL, name="image")
        self.assertTrue(span.contains(4))
        self.assertFalse(span.contains(8))


if __name__ == "__main__":
    unittest.main()
