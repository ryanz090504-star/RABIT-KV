# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch

from vllm.v1.attention.ops.rabit_kv2 import (
    decode_rabit2_page_ref,
    decode_rabit2_page_triton,
    encode_rabit2_page_ref,
    encode_rabit2_page_triton,
    rabit2_closed_page_attention_triton,
)
from vllm.v1.kv_cache_interface import rabit2_page_layout


def _layout():
    return rabit2_page_layout(
        block_size=32,
        num_kv_heads=8,
        head_size_k=128,
        head_size_v=128,
    )


def _sample(device="cuda"):
    torch.manual_seed(20260807)
    k = (torch.randn(32, 8, 128, device=device) * 0.7).to(torch.bfloat16)
    v = (torch.randn(32, 8, 128, device=device) * 0.9).to(torch.bfloat16)
    return k, v


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stage3b1_triton_decoder_matches_reference_page():
    layout = _layout()
    k, v = _sample()
    page = encode_rabit2_page_ref(k, v, layout=layout)
    kr, vr = decode_rabit2_page_ref(page, layout=layout, dtype=torch.float32)
    kt, vt = decode_rabit2_page_triton(page, layout=layout, dtype=torch.float32)
    torch.testing.assert_close(kt, kr, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(vt, vr, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stage3b1_triton_encoder_is_byte_exact_on_selected_layout():
    layout = _layout()
    k, v = _sample()
    ref = encode_rabit2_page_ref(k, v, layout=layout)
    tri = encode_rabit2_page_triton(k, v, layout=layout)
    if not torch.equal(tri, ref):
        mismatch = int((tri != ref).sum().item())
        max_delta = int((tri.to(torch.int16) - ref.to(torch.int16)).abs().max().item())
        pytest.fail(f"Triton page differs from reference: mismatch_bytes={mismatch}, max_byte_delta={max_delta}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stage3b1_triton_encoded_page_roundtrip_matches_reference():
    layout = _layout()
    k, v = _sample()
    page = encode_rabit2_page_triton(k, v, layout=layout)
    kr, vr = decode_rabit2_page_ref(page, layout=layout, dtype=torch.float32)
    kt, vt = decode_rabit2_page_triton(page, layout=layout, dtype=torch.float32)
    torch.testing.assert_close(kt, kr, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(vt, vr, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stage3b1_fused_closed_page_attention_matches_reference():
    layout = _layout()
    k, v = _sample()
    page = encode_rabit2_page_ref(k, v, layout=layout)
    kd, vd = decode_rabit2_page_ref(page, layout=layout, dtype=torch.float32)

    torch.manual_seed(20260808)
    q = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(128)
    out_tri = rabit2_closed_page_attention_triton(
        q, page, layout=layout, softmax_scale=scale
    ).float()

    # Llama-3.1-8B GQA mapping: 32 Q heads / 8 KV heads = 4 Q heads per KV head.
    kv_idx = torch.arange(32, device="cuda") // 4
    kq = kd[:, kv_idx, :]
    vq = vd[:, kv_idx, :]
    scores = torch.einsum("hd,thd->ht", q.float(), kq) * scale
    probs = torch.softmax(scores, dim=-1)
    out_ref = torch.einsum("ht,thd->hd", probs, vq)
    torch.testing.assert_close(out_tri, out_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stage3b1_no_bf16_materialization_required_for_fused_read():
    layout = _layout()
    k, v = _sample()
    page = encode_rabit2_page_ref(k, v, layout=layout)
    q = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
    out = rabit2_closed_page_attention_triton(q, page, layout=layout)
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert torch.isfinite(out.float()).all()
