
from __future__ import annotations
import pytest
import torch

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "total_tokens,seed",
    [(5,1005),(6,1006),(13,1013),(20,1020),(27,1027),(31,3031),(35,1035),(59,4059),(70,4070)],
)
def test_stage4b1_exactmeta_matches_stage3c_online(total_tokens, seed):
    from vllm.v1.attention.ops.rabit_kv2 import (
        Rabit2SingleSequenceRuntime,
        _rabit2_online_decode_attention_triton_stage3c_exact,
        rabit2_online_decode_attention_triton_stage4b1_exactmeta,
    )
    torch.manual_seed(seed)
    qh, kvh, d = 32, 8, 128
    rt = Rabit2SingleSequenceRuntime(kvh, d, d)
    blocks = max(8, total_tokens // 32 + 8)
    cache = torch.zeros((blocks,1,1,1,rt.layout.page_bytes), dtype=torch.uint8, device="cuda")
    bt = torch.arange(blocks, dtype=torch.int32, device="cuda")
    k = torch.randn((total_tokens,kvh,d), dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    rt.append(k, v, cache, bt)
    q = torch.randn((1,qh,d), dtype=torch.bfloat16, device="cuda")
    ref = _rabit2_online_decode_attention_triton_stage3c_exact(q, cache, bt, rt)
    got = rabit2_online_decode_attention_triton_stage4b1_exactmeta(q, cache, bt, rt)
    torch.cuda.synchronize()
    assert torch.equal(ref, got), (total_tokens, float((ref.float()-got.float()).abs().max().item()))
