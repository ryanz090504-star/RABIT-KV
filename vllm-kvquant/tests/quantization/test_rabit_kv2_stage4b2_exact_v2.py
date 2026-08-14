
import pytest
import torch

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stage4b2_exact_v2_gate():
    import vllm.v1.attention.ops.rabit_kv2 as r
    warm = torch.randn((1,8,128), dtype=torch.bfloat16, device="cuda")
    r._quantize_v2_primary_ref(warm)
    torch.cuda.synchronize()
    for seed in list(range(1,257)) + [101,777,2026,4096]:
        torch.manual_seed(seed)
        x = torch.randn((1,8,128), dtype=torch.bfloat16, device="cuda")
        a = r._quantize_v2_primary_ref_stage4b1_exact(x)
        b = r._quantize_v2_primary_ref(x)
        torch.cuda.synchronize()
        assert a[3] == b[3] == 128
        assert torch.equal(a[0], b[0]), seed
        assert torch.equal(a[1], b[1]), seed
        assert torch.equal(a[2], b[2]), seed

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stage4b2_fallback_unchanged():
    import vllm.v1.attention.ops.rabit_kv2 as r
    x = torch.randn((2,4,64), dtype=torch.bfloat16, device="cuda")
    a = r._quantize_v2_primary_ref_stage4b1_exact(x)
    b = r._quantize_v2_primary_ref(x)
    for i in range(3):
        assert torch.equal(a[i], b[i])
    assert a[3] == b[3]
