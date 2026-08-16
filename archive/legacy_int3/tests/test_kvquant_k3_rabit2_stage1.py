# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    is_quantized_kv_cache as torch_utils_is_quantized_kv_cache,
    kv_cache_uses_per_token_head_scales as torch_utils_uses_pth_scales,
)
from vllm.v1.attention.ops.kvquant_k3 import (
    RABIT2_GROUP_SIZE,
    RABIT2_METADATA_GROUP_SIZE,
    RABIT2_RESIDUAL_TOKENS,
    attention_rabit2_ref,
    decode_metadata_uint8_group_ref,
    dequantize_int3_per_token_head_ref,
    dequantize_k3_sequence_affine_ref,
    dequantize_rabit2_kv_ref,
    dequantize_v2_group_affine_ref,
    encode_metadata_uint8_group_ref,
    metadata_storage_bytes_ref,
    pack_int2_values,
    pack_int3_values,
    quantize_int3_per_token_head_ref,
    quantize_k3_sequence_affine_ref,
    quantize_rabit2_kv_ref,
    quantize_v2_group_affine_ref,
    rabit2_state_storage_bytes_ref,
    reshape_and_cache_kvquant_k3,
    unified_attention_kvquant_k3,
    unpack_int2_values,
    unpack_int3_values,
    update_rabit2_kv_ref,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVQuantMode,
    get_kv_quant_mode,
    is_quantized_kv_cache,
    kv_cache_uses_per_token_head_scales,
    kvquant_k3_packed_dim,
)


def test_kvquant_k3_dtype_mapping() -> None:
    assert STR_DTYPE_TO_TORCH_DTYPE["kvquant_k3"] is torch.uint8
    assert torch_utils_is_quantized_kv_cache("kvquant_k3")
    assert is_quantized_kv_cache("kvquant_k3")
    assert torch_utils_uses_pth_scales("kvquant_k3")
    assert kv_cache_uses_per_token_head_scales("kvquant_k3")
    assert get_kv_quant_mode("kvquant_k3") == KVQuantMode.INT3_PER_TOKEN_HEAD


@pytest.mark.parametrize("value_count", [1, 2, 3, 7, 8, 9, 16, 17])
def test_pack_unpack_int3_roundtrip(value_count: int) -> None:
    values = (torch.arange(2 * value_count, dtype=torch.uint8) % 8).reshape(
        2, value_count
    )

    packed = pack_int3_values(values)
    unpacked = unpack_int3_values(packed, value_count)

    assert packed.dtype == torch.uint8
    assert packed.shape == (2, math.ceil(value_count * 3 / 8))
    assert torch.equal(unpacked, values)


def test_pack_int3_rejects_out_of_range_codes() -> None:
    with pytest.raises(ValueError, match=r"\[0, 7\]"):
        pack_int3_values(torch.tensor([0, 8], dtype=torch.uint8))


def test_kvquant_k3_page_size_bytes() -> None:
    spec = AttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=64,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.INT3_PER_TOKEN_HEAD,
    )

    packed_dim = kvquant_k3_packed_dim(64)
    real_bytes = 2 * 16 * 4 * packed_dim
    scale_bytes = 2 * 16 * 4 * 4

    assert packed_dim == 24
    assert spec.real_page_size_bytes == real_bytes
    assert spec.page_size_bytes == real_bytes + scale_bytes


def test_kvquant_k3_full_attention_page_size_bytes() -> None:
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=64,
        head_size_v=80,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.INT3_PER_TOKEN_HEAD,
    )

    real_bytes = 16 * 4 * (kvquant_k3_packed_dim(64) + kvquant_k3_packed_dim(80))
    scale_bytes = 2 * 16 * 4 * 4

    assert spec.real_page_size_bytes == real_bytes
    assert spec.page_size_bytes == real_bytes + scale_bytes


def test_triton_backend_kvquant_k3_shape() -> None:
    pytest.importorskip("cbor2", reason="Triton backend import needs vLLM deps")

    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

    shape = TritonAttentionBackend.get_kv_cache_shape(
        num_blocks=3,
        block_size=16,
        num_kv_heads=4,
        head_size=64,
        cache_dtype_str="kvquant_k3",
    )

    assert "kvquant_k3" in TritonAttentionBackend.supported_kv_cache_dtypes
    assert shape == (3, 2, 16, 4, kvquant_k3_packed_dim(64) + 4)


def test_reference_cache_write_uses_packed_area_and_scales() -> None:
    key = torch.tensor(
        [
            [[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5, 2.0, -2.0]],
            [[0.25, -0.25, 0.75, -0.75, 1.25, -1.25, 0.0, 0.5, -0.5]],
        ],
        dtype=torch.float16,
    )
    value = key * 0.5
    block_size = 16
    packed_dim = kvquant_k3_packed_dim(key.shape[-1])
    padded_dim = packed_dim + 4

    key_cache = torch.zeros(1, block_size, 1, padded_dim, dtype=torch.uint8)
    value_cache = torch.zeros_like(key_cache)
    k_scale_cache = torch.ones(1, block_size, 1, dtype=torch.float32)
    v_scale_cache = torch.ones_like(k_scale_cache)
    slot_mapping = torch.tensor([3, 7], dtype=torch.long)

    reshape_and_cache_kvquant_k3(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
    )

    ref_k_packed, ref_k_scales = quantize_int3_per_token_head_ref(key)
    ref_v_packed, ref_v_scales = quantize_int3_per_token_head_ref(value)

    for token_idx, slot in enumerate(slot_mapping.tolist()):
        assert torch.equal(key_cache[0, slot, :, :packed_dim], ref_k_packed[token_idx])
        assert torch.equal(
            value_cache[0, slot, :, :packed_dim], ref_v_packed[token_idx]
        )
        assert torch.equal(
            key_cache[0, slot, :, packed_dim:],
            torch.zeros(1, 4, dtype=torch.uint8),
        )
        assert torch.allclose(k_scale_cache[0, slot], ref_k_scales[token_idx])
        assert torch.allclose(v_scale_cache[0, slot], ref_v_scales[token_idx])

    dequant = dequantize_int3_per_token_head_ref(
        key_cache[0, 3, :, :packed_dim],
        k_scale_cache[0, 3],
        key.shape[-1],
        dtype=key.dtype,
    )
    assert dequant.shape == key[0].shape


def test_attention_read_path_kernel_launches() -> None:
    """Smoke test: the INT3 attention kernel computes output without crashing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    pytest.importorskip("cbor2", reason="Triton backend import needs vLLM deps")

    device = torch.device("cuda")
    num_blocks = 4
    block_size = 16
    num_kv_heads = 2
    num_query_heads = 4
    head_size = 64
    num_tokens = 2
    packed_dim = kvquant_k3_packed_dim(head_size)
    padded_dim = packed_dim + 4

    k_cache = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        padded_dim,
        dtype=torch.uint8,
        device=device,
    )
    v_cache = torch.zeros_like(k_cache)
    k_scale_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, dtype=torch.float32, device=device
    )
    v_scale_cache = torch.zeros_like(k_scale_cache)

    # Write two tokens with known float values.
    key = torch.randn(
        num_tokens, num_kv_heads, head_size, dtype=torch.float16, device=device
    )
    value = torch.randn(
        num_tokens, num_kv_heads, head_size, dtype=torch.float16, device=device
    )
    slot_mapping = torch.tensor([0, 1], dtype=torch.long, device=device)

    reshape_and_cache_kvquant_k3(
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
    )

    q = torch.randn(1, num_query_heads, head_size, dtype=torch.float16, device=device)
    out = torch.zeros(1, num_query_heads, head_size, dtype=torch.float16, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    seqused_k = torch.tensor([2], dtype=torch.int32, device=device)
    block_table = torch.tensor([[0]], dtype=torch.int32, device=device)

    # Sanity: kernel runs without error and produces non-NaN output.
    unified_attention_kvquant_k3(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seqused_k,
        max_seqlen_k=2,
        softmax_scale=1.0 / (head_size**0.5),
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        sinks=None,
        alibi_slopes=None,
        use_alibi_sqrt=False,
        qq_bias=None,
        output_scale=None,
        mm_prefix_range=None,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
    )
    assert not torch.isnan(out).any(), "INT3 attention output contains NaN"
    assert not torch.isinf(out).any(), "INT3 attention output contains Inf"


# ---------------------------------------------------------------------------
# RABIT-2 final-policy correctness reference tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value_count", [1, 2, 3, 4, 7, 8, 9, 16, 17, 32, 33])
def test_pack_unpack_int2_roundtrip(value_count: int) -> None:
    values = (torch.arange(3 * value_count, dtype=torch.uint8) % 4).reshape(
        3, value_count
    )
    packed = pack_int2_values(values)
    unpacked = unpack_int2_values(packed, value_count)
    assert packed.shape == (3, math.ceil(value_count * 2 / 8))
    assert torch.equal(unpacked, values)


def test_pack_int2_rejects_out_of_range_codes() -> None:
    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        pack_int2_values(torch.tensor([0, 4], dtype=torch.uint8))


def test_metadata_uint8_g64_roundtrip_and_accounting() -> None:
    torch.manual_seed(1)
    source = torch.randn(3, 5, 17)
    encoded = encode_metadata_uint8_group_ref(
        source, group_size=RABIT2_METADATA_GROUP_SIZE
    )
    decoded = decode_metadata_uint8_group_ref(encoded)
    assert decoded.shape == source.shape
    assert encoded["codes"].dtype is torch.uint8
    assert encoded["min"].dtype is torch.bfloat16
    assert encoded["scale"].dtype is torch.bfloat16
    assert encoded["group_size"] == 64
    expected_bytes = (
        encoded["codes"].numel()
        + 2 * encoded["min"].numel()
        + 2 * encoded["scale"].numel()
    )
    assert metadata_storage_bytes_ref(encoded) == expected_bytes
    # Error includes UINT8 quantization plus BF16 second-level metadata.
    assert torch.max(torch.abs(decoded - source)).item() < 0.05


def test_k3_sequence_affine_g32_layout_and_dequantization() -> None:
    torch.manual_seed(2)
    key = torch.randn(37, 2, 64, dtype=torch.float32)
    state = quantize_k3_sequence_affine_ref(key)
    decoded = dequantize_k3_sequence_affine_ref(state)

    assert state["bits"] == 3
    assert state["seq_group_size"] == RABIT2_GROUP_SIZE
    assert state["pad_seq"] == 27
    assert state["packed"].shape == (64, 2, math.ceil(64 * 3 / 8))
    assert decoded.shape == key.shape

    # Directly reconstruct the same quality-reference math and metadata path.
    x = key.permute(1, 0, 2).unsqueeze(0)
    x = torch.nn.functional.pad(x, (0, 0, 0, 27))
    grouped = x.reshape(1, 2, 2, 32, 64)
    q_min = grouped.amin(dim=3, keepdim=True)
    q_max = grouped.amax(dim=3, keepdim=True)
    scale = (q_max - q_min) / 7.0
    scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
    codes = torch.round((grouped - q_min) / scale).clamp(0, 7)
    q_min = decode_metadata_uint8_group_ref(
        encode_metadata_uint8_group_ref(q_min)
    )
    scale = decode_metadata_uint8_group_ref(
        encode_metadata_uint8_group_ref(scale)
    )
    expected = (codes * scale + q_min).reshape(1, 2, 64, 64)
    expected = expected[:, :, :37].squeeze(0).permute(1, 0, 2)
    assert torch.equal(decoded, expected)


def test_v2_group_affine_g32_layout_and_dequantization() -> None:
    torch.manual_seed(3)
    value = torch.randn(7, 3, 80, dtype=torch.float32)
    state = quantize_v2_group_affine_ref(value)
    decoded = dequantize_v2_group_affine_ref(state)

    assert state["bits"] == 2
    assert state["group_size"] == RABIT2_GROUP_SIZE
    assert state["pad_dim"] == 16
    assert state["packed"].shape == (7, 3, math.ceil(96 * 2 / 8))
    assert decoded.shape == value.shape

    x = torch.nn.functional.pad(value, (0, 16))
    grouped = x.reshape(7, 3, 3, 32)
    q_min = grouped.amin(dim=-1, keepdim=True)
    q_max = grouped.amax(dim=-1, keepdim=True)
    scale = (q_max - q_min) / 3.0
    scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
    codes = torch.round((grouped - q_min) / scale).clamp(0, 3)
    q_min = decode_metadata_uint8_group_ref(
        encode_metadata_uint8_group_ref(q_min)
    )
    scale = decode_metadata_uint8_group_ref(
        encode_metadata_uint8_group_ref(scale)
    )
    expected = (codes * scale + q_min).reshape(7, 3, 96)[..., :80]
    assert torch.equal(decoded, expected)


def test_rabit2_r4_residual_is_exact_bf16() -> None:
    torch.manual_seed(4)
    key = torch.randn(11, 2, 64, dtype=torch.float32)
    value = torch.randn_like(key)
    state = quantize_rabit2_kv_ref(key, value)
    decoded_k, decoded_v = dequantize_rabit2_kv_ref(state)

    assert state["old_count"] == 7
    assert state["recent_k"].shape[0] == RABIT2_RESIDUAL_TOKENS
    assert torch.equal(decoded_k[-4:], key[-4:].to(torch.bfloat16).float())
    assert torch.equal(decoded_v[-4:], value[-4:].to(torch.bfloat16).float())
    # Old K and V are genuinely quantized and normally differ from BF16 input.
    assert not torch.equal(decoded_k[:7], key[:7].to(torch.bfloat16).float())
    assert not torch.equal(decoded_v[:7], value[:7].to(torch.bfloat16).float())


def test_rabit2_ageing_out_after_fifth_token() -> None:
    torch.manual_seed(5)
    key = torch.randn(4, 1, 32, dtype=torch.float32)
    value = torch.randn_like(key)
    state = quantize_rabit2_kv_ref(key, value)
    assert state["old_count"] == 0

    new_key = torch.randn(1, 1, 32)
    new_value = torch.randn_like(new_key)
    updated = update_rabit2_kv_ref(state, new_key, new_value)
    decoded_k, decoded_v = dequantize_rabit2_kv_ref(updated)

    full_k = torch.cat([key.to(torch.bfloat16).float(), new_key], dim=0)
    full_v = torch.cat([value.to(torch.bfloat16).float(), new_value], dim=0)
    assert updated["old_count"] == 1
    assert updated["recent_k"].shape[0] == 4
    assert torch.equal(decoded_k[-4:], full_k[-4:].to(torch.bfloat16).float())
    assert torch.equal(decoded_v[-4:], full_v[-4:].to(torch.bfloat16).float())


def test_rabit2_storage_breakdown_counts_payload_metadata_and_residual() -> None:
    torch.manual_seed(6)
    key = torch.randn(36, 2, 64)
    value = torch.randn_like(key)
    state = quantize_rabit2_kv_ref(key, value)
    breakdown = rabit2_state_storage_bytes_ref(state)

    assert breakdown["payload"] == (
        state["old_k"]["packed"].numel() + state["old_v"]["packed"].numel()
    )
    assert breakdown["metadata"] > 0
    assert breakdown["residual"] == 2 * 4 * 2 * 64 * 2
    assert breakdown["total"] == (
        breakdown["payload"] + breakdown["metadata"] + breakdown["residual"]
    )


def test_rabit2_attention_oracle_matches_manual_dequantized_attention() -> None:
    torch.manual_seed(7)
    key = torch.randn(9, 2, 32)
    value = torch.randn_like(key)
    q = torch.randn(1, 4, 32, dtype=torch.float16)
    state = quantize_rabit2_kv_ref(key, value)
    actual = attention_rabit2_ref(q, state)

    dequant_k, dequant_v = dequantize_rabit2_kv_ref(state)
    dequant_k = dequant_k.repeat_interleave(2, dim=1)
    dequant_v = dequant_v.repeat_interleave(2, dim=1)
    scores = torch.einsum("qhd,thd->qht", q.float(), dequant_k) / math.sqrt(32)
    expected = torch.einsum(
        "qht,thd->qhd", torch.softmax(scores, dim=-1), dequant_v
    ).to(q.dtype)
    assert torch.equal(actual, expected)
