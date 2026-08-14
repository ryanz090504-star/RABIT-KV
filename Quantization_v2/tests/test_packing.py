import unittest

import numpy as np

from kvquant.packing import UniformQuantizedArray, pack_bits, unpack_bits
from kvquant.policies import build_policy
from kvquant.types import QuantizationContext


class PackingTests(unittest.TestCase):
    def test_pack_roundtrip_for_low_bits(self):
        values = np.array([0, 1, 2, 3, 0, 3, 2, 1, 1], dtype=np.uint8)
        packed = pack_bits(values, 2)
        unpacked = unpack_bits(packed, 2, values.size)
        np.testing.assert_array_equal(unpacked, values)

    def test_pack_roundtrip_for_three_bits(self):
        values = np.array([0, 7, 3, 5, 1, 6, 2, 4, 7, 0, 1], dtype=np.uint8)
        packed = pack_bits(values, 3)
        unpacked = unpack_bits(packed, 3, values.size)
        np.testing.assert_array_equal(unpacked, values)

    def test_uniform_quantization_shape_and_error(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=(1, 2, 8, 4)).astype(np.float32)
        q = UniformQuantizedArray.from_float(values, nbits=4, axis_strategy="per_head")
        deq = q.dequantize()
        self.assertEqual(deq.shape, values.shape)
        self.assertLess(q.estimated_payload_nbytes(), values.nbytes)

    def test_quantized_kv_block_pack_roundtrip(self):
        rng = np.random.default_rng(2)
        keys = rng.normal(size=(1, 2, 8, 4)).astype(np.float32)
        values = rng.normal(size=(1, 2, 8, 4)).astype(np.float32)
        block = build_policy("document_naive", nbits=4).quantize(keys, values, QuantizationContext(layer_idx=0))
        packed = block.to_packed()

        key_deq, value_deq = block.dequantize()
        packed_key_deq, packed_value_deq = packed.dequantize()

        np.testing.assert_allclose(packed_key_deq, key_deq)
        np.testing.assert_allclose(packed_value_deq, value_deq)
        self.assertEqual(packed.metadata["storage"], "bit_packed")
        self.assertEqual(packed.estimated_payload_nbytes(), block.estimated_payload_nbytes())


if __name__ == "__main__":
    unittest.main()
