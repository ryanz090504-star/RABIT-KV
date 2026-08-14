
from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("closed_pages", [1, 2, 15, 64])
def test_stage4b3_gqa4_matches_stage4b2_exact(closed_pages):
    import vllm.v1.attention.ops.rabit_kv2 as r

    torch.manual_seed(7000 + closed_pages)
    qh, kvh, d = 32, 8, 128
    rt = r.Rabit2SingleSequenceRuntime(kvh, d, d)
    cache = torch.empty(
        (closed_pages + 8, 1, 1, 1, rt.layout.page_bytes),
        dtype=torch.uint8,
        device="cuda",
    )

    # Fill all physical closed pages with exact page bytes.
    for p in range(closed_pages):
        k = torch.randn((32, kvh, d), dtype=torch.bfloat16, device="cuda")
        v = torch.randn_like(k)
        page = r.encode_rabit2_page_exact_cuda(k, v, layout=rt.layout)
        cache[p].reshape(-1).copy_(page)

    bt = torch.arange(closed_pages + 8, dtype=torch.int32, device="cuda")

    # Exact bounded sidecar tail.
    tk = torch.randn((20, kvh, d), dtype=torch.bfloat16, device="cuda")
    tv = torch.randn_like(tk)
    rt.append(tk, tv, cache, bt)
    rt.closed_pages = closed_pages

    q = torch.randn((1, qh, d), dtype=torch.bfloat16, device="cuda")
    ref = r._rabit2_online_decode_attention_triton_stage4b2_exact(
        q, cache, bt, rt
    )
    got = r.rabit2_online_decode_attention_triton_stage4b3_gqa4(
        q, cache, bt, rt
    )
    torch.cuda.synchronize()
    assert torch.equal(ref, got), (
        closed_pages,
        float((ref.float() - got.float()).abs().max().item()),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stage4b3_gqa4_fallback_for_non_divisible_ratio():
    import vllm.v1.attention.ops.rabit_kv2 as r

    # 8 Q / 8 KV => ratio 1, must use exact Stage4B2 fallback.
    qh, kvh, d = 8, 8, 128
    rt = r.Rabit2SingleSequenceRuntime(kvh, d, d)
    cache = torch.zeros(
        (8, 1, 1, 1, rt.layout.page_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    bt = torch.arange(8, dtype=torch.int32, device="cuda")
    k = torch.randn((4, kvh, d), dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    rt.append(k, v, cache, bt)
    q = torch.randn((1, qh, d), dtype=torch.bfloat16, device="cuda")
    ref = r._rabit2_online_decode_attention_triton_stage4b2_exact(
        q, cache, bt, rt
    )
    got = r.rabit2_online_decode_attention_triton_stage4b3_gqa4(
        q, cache, bt, rt
    )
    torch.cuda.synchronize()
    assert torch.equal(ref, got)
