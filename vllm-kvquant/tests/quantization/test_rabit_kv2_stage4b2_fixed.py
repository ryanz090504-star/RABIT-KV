
from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stage4b2_fixed_exactness_gate():
    import vllm.v1.attention.ops.rabit_kv2 as r

    assert getattr(r, "_RABIT2_STAGE4B2_EXACT_V2_FIXED", False)
    warm = torch.randn((1, 8, 128), dtype=torch.bfloat16, device="cuda")
    r._quantize_v2_primary_ref(warm)
    torch.cuda.synchronize()

    for seed in list(range(1, 257)) + [101, 777, 2026, 4096]:
        torch.manual_seed(seed)
        x = torch.randn((1, 8, 128), dtype=torch.bfloat16, device="cuda")
        ep, emn, esc, ed = r._quantize_v2_primary_ref_stage4b1_exact(x)
        gp, gmn, gsc, gd = r._quantize_v2_primary_ref(x)
        torch.cuda.synchronize()
        assert ed == gd == 128
        assert torch.equal(ep, gp), seed
        assert torch.equal(emn, gmn), seed
        assert torch.equal(esc, gsc), seed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stage4b2_fixed_outputs_survive_subsequent_compiled_calls():
    """Direct regression for the CUDA-graph overwrite that broke Stage4B2."""
    import vllm.v1.attention.ops.rabit_kv2 as r

    torch.manual_seed(111)
    x1 = torch.randn((1, 8, 128), dtype=torch.bfloat16, device="cuda")
    p1, mn1, sc1, _ = r._quantize_v2_primary_ref(x1)
    # Snapshot persistent outputs.
    p1_saved = p1.clone()
    mn1_saved = mn1.clone()
    sc1_saved = sc1.clone()

    # Many later calls must not mutate the first call's returned tensors.
    for seed in range(112, 140):
        torch.manual_seed(seed)
        x = torch.randn((1, 8, 128), dtype=torch.bfloat16, device="cuda")
        r._quantize_v2_primary_ref(x)

    torch.cuda.synchronize()
    assert torch.equal(p1, p1_saved)
    assert torch.equal(mn1, mn1_saved)
    assert torch.equal(sc1, sc1_saved)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stage4b2_fixed_fallback_unchanged():
    import vllm.v1.attention.ops.rabit_kv2 as r

    torch.manual_seed(99)
    x = torch.randn((2, 4, 64), dtype=torch.bfloat16, device="cuda")
    a = r._quantize_v2_primary_ref_stage4b1_exact(x)
    b = r._quantize_v2_primary_ref(x)
    torch.cuda.synchronize()
    for i in range(3):
        assert torch.equal(a[i], b[i])
    assert a[3] == b[3]
