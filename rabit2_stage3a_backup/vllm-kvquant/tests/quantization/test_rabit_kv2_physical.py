# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.ops.kvquant_k3 import (
    decode_metadata_uint8_group_ref,
    dequantize_k3_sequence_affine_ref,
    dequantize_rabit2_kv_ref,
    encode_metadata_uint8_group_ref,
    quantize_k3_sequence_affine_ref,
    quantize_rabit2_kv_ref,
)
from vllm.v1.attention.ops.rabit_kv2 import (
    Rabit2OnlineStateRef,
    _dequantize_v2_from_primary_ref,
    _quantize_v2_primary_ref,
    decode_metadata_blob_ref,
    decode_rabit2_page_ref,
    encode_metadata_blob_ref,
    encode_rabit2_page_ref,
)
from vllm.v1.kv_cache_interface import (
    rabit2_exact_online_staging_bytes_per_sequence,
    rabit2_page_layout,
)


def test_metadata_blob_roundtrip_matches_reference() -> None:
    torch.manual_seed(1)
    primary = torch.randn(1, 2, 1, 1, 32)
    metadata = encode_metadata_uint8_group_ref(primary)
    blob = encode_metadata_blob_ref(metadata)
    restored = decode_metadata_blob_ref(blob, original_shape=tuple(primary.shape))

    assert blob.dtype == torch.uint8
    assert blob.numel() == 68  # 64 UINT8 codes + two BF16 min/scale values.
    assert torch.equal(restored["codes"], metadata["codes"])
    assert torch.equal(restored["min"], metadata["min"])
    assert torch.equal(restored["scale"], metadata["scale"])
    assert torch.equal(
        decode_metadata_uint8_group_ref(restored),
        decode_metadata_uint8_group_ref(metadata),
    )


def test_complete_page_codec_matches_quality_reference() -> None:
    torch.manual_seed(2)
    key = torch.randn(32, 2, 128, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    layout = rabit2_page_layout(
        block_size=32,
        num_kv_heads=2,
        head_size_k=128,
        head_size_v=128,
    )

    page = encode_rabit2_page_ref(key, value, layout=layout)
    actual_k, actual_v = decode_rabit2_page_ref(page, layout=layout)

    k_state = quantize_k3_sequence_affine_ref(key)
    expected_k = dequantize_k3_sequence_affine_ref(k_state)
    v_packed, v_min_primary, v_scale_primary, _ = _quantize_v2_primary_ref(value)
    v_min = decode_metadata_uint8_group_ref(
        encode_metadata_uint8_group_ref(v_min_primary)
    )
    v_scale = decode_metadata_uint8_group_ref(
        encode_metadata_uint8_group_ref(v_scale_primary)
    )
    expected_v = _dequantize_v2_from_primary_ref(
        v_packed, v_min, v_scale, head_dim=128
    )

    assert page.dtype == torch.uint8
    assert page.numel() == layout.page_bytes
    assert torch.equal(actual_k, expected_k)
    assert torch.equal(actual_v, expected_v)


@pytest.mark.parametrize("token_count", [1, 4, 5, 6, 34, 35, 36, 37, 67, 68, 69])
def test_online_physical_state_matches_quality_oracle(token_count: int) -> None:
    torch.manual_seed(100 + token_count)
    key = torch.randn(token_count, 2, 128, dtype=torch.bfloat16)
    value = torch.randn_like(key)

    online = Rabit2OnlineStateRef(
        num_kv_heads=2,
        head_size_k=128,
        head_size_v=128,
    )
    for token in range(token_count):
        online.append(key[token : token + 1], value[token : token + 1])

    actual_k, actual_v = online.materialize()
    oracle = quantize_rabit2_kv_ref(key, value)
    expected_k, expected_v = dequantize_rabit2_kv_ref(oracle)

    assert online.total_tokens == token_count
    assert torch.equal(actual_k, expected_k)
    assert torch.equal(actual_v, expected_v)


def test_online_state_closes_page_at_36_total_tokens() -> None:
    # Four tokens remain residual, so the first 32-token old page closes at 36.
    torch.manual_seed(5)
    key = torch.randn(36, 2, 128, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    state = Rabit2OnlineStateRef(
        num_kv_heads=2,
        head_size_k=128,
        head_size_v=128,
    )
    state.append(key, value)

    assert len(state.pages) == 1
    assert state.open_k is None
    assert state.recent_k is not None and state.recent_k.shape[0] == 4
    assert state.pages[0].numel() == state.layout.page_bytes


def test_actual_open_state_bytes_match_staging_upper_bound() -> None:
    # At 35 total tokens: 31 open old tokens + four BF16 residual tokens.
    torch.manual_seed(6)
    key = torch.randn(35, 2, 128, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    state = Rabit2OnlineStateRef(
        num_kv_heads=2,
        head_size_k=128,
        head_size_v=128,
    )
    state.append(key, value)
    breakdown = state.storage_breakdown()
    expected = rabit2_exact_online_staging_bytes_per_sequence(
        num_kv_heads=2,
        head_size_k=128,
        head_size_v=128,
    )

    assert len(state.pages) == 0
    assert breakdown["total"] == expected
    assert breakdown == {
        "pages": 0,
        "open_k_bf16": 15872,
        "open_v_payload": 1984,
        "open_v_primary_metadata": 1984,
        "residual": 4096,
        "total": 23936,
    }


def test_llama31_exact_online_staging_is_not_r4_only() -> None:
    assert rabit2_exact_online_staging_bytes_per_sequence(
        num_kv_heads=8,
        head_size_k=128,
        head_size_v=128,
    ) == 95744
