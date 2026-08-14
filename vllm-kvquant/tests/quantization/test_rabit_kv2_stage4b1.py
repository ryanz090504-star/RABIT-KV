
from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("total_tokens", [5, 6, 13, 20, 27, 35, 70])
def test_stage4b1_matches_stage3c_online_attention(total_tokens: int):
    from vllm.v1.attention.ops.rabit_kv2 import (
        Rabit2SingleSequenceRuntime,
        _rabit2_online_decode_attention_triton_stage3c_exact,
        rabit2_online_decode_attention_triton_stage4b1,
    )

    torch.manual_seed(1000 + total_tokens)
    device = torch.device("cuda")
    qh, kvh, d = 32, 8, 128
    rt = Rabit2SingleSequenceRuntime(kvh, d, d)
    num_blocks = max(8, total_tokens // 32 + 8)
    cache = torch.zeros(
        (num_blocks, 1, 1, 1, rt.layout.page_bytes),
        dtype=torch.uint8,
        device=device,
    )
    bt = torch.arange(num_blocks, dtype=torch.int32, device=device)
    k = torch.randn((total_tokens, kvh, d), dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    rt.append(k, v, cache, bt)
    q = torch.randn((1, qh, d), dtype=torch.bfloat16, device=device)

    ref = _rabit2_online_decode_attention_triton_stage3c_exact(q, cache, bt, rt)
    got = rabit2_online_decode_attention_triton_stage4b1(q, cache, bt, rt)
    torch.cuda.synchronize()

    err = (ref.float() - got.float()).abs()
    assert torch.isfinite(got).all()
    assert float(err.max().item()) <= 0.02
