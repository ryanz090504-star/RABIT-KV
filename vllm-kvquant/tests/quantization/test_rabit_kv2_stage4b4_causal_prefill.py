
from __future__ import annotations

import pytest
import torch


def _alloc(r, tokens, h=8, d=128):
    rt = r.Rabit2SingleSequenceRuntime(h, d, d)
    pages = max(16, (tokens + 31) // 32 + 16)
    cache = torch.zeros(
        (pages, 1, 1, 1, rt.layout.page_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    bt = torch.arange(pages, dtype=torch.int32, device="cuda")
    return rt, cache, bt


def _same_state(a, ca, b, cb):
    assert a.closed_pages == b.closed_pages
    for name in (
        "open_k", "open_v_packed", "open_v_min", "open_v_scale",
        "recent_k", "recent_v",
    ):
        xa, xb = getattr(a, name), getattr(b, name)
        if xa is None or xb is None:
            assert xa is None and xb is None, name
        else:
            assert torch.equal(xa, xb), name
    if a.closed_pages:
        assert torch.equal(ca[:a.closed_pages], cb[:b.closed_pages])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "prefix,chunk",
    [(0,8),(3,8),(4,8),(31,8),(35,16),(60,64),(64,64),(96,64),(127,64)],
)
def test_stage4b4_causal_plan_token_exact(prefix, chunk):
    import vllm.v1.attention.ops.rabit_kv2 as r

    torch.manual_seed(81000 + prefix + chunk)
    h, d, qh = 8, 128, 32
    total = prefix + chunk
    k = torch.randn((total,h,d), dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    q = torch.randn((chunk,qh,d), dtype=torch.bfloat16, device="cuda")

    ref, cref, btref = _alloc(r, total)
    fast, cfast, btfast = _alloc(r, total)

    if prefix:
        ref.append(k[:prefix],v[:prefix],cref,btref)
        fast.append(k[:prefix],v[:prefix],cfast,btfast)

    plan = r.Rabit2CausalChunkPlan(
        fast, k[prefix:], v[prefix:], cfast, btfast
    )

    for j in range(chunk):
        ref.append(
            k[prefix+j:prefix+j+1],
            v[prefix+j:prefix+j+1],
            cref,
            btref,
        )
        plan.apply_step(j)
        _same_state(ref,cref,fast,cfast)

        a = r.rabit2_online_decode_attention_triton(
            q[j:j+1],cref,btref,ref
        )
        b = r.rabit2_online_decode_attention_triton(
            q[j:j+1],cfast,btfast,fast
        )
        torch.cuda.synchronize()
        assert torch.equal(a,b), float((a.float()-b.float()).abs().max())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n", [1,4,5,31,32,36,64,127,256])
def test_stage4b4_bulk_initial_prefill_final_state_exact(n):
    import vllm.v1.attention.ops.rabit_kv2 as r

    torch.manual_seed(82000+n)
    h,d=8,128
    k=torch.randn((n,h,d),dtype=torch.bfloat16,device="cuda")
    v=torch.randn_like(k)
    ref,cref,btref=_alloc(r,n)
    fast,cfast,btfast=_alloc(r,n)

    ref.append(k,v,cref,btref)
    r.rabit2_bulk_append_exact(fast,k,v,cfast,btfast)
    torch.cuda.synchronize()
    _same_state(ref,cref,fast,cfast)
