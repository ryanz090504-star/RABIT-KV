import math
from pathlib import Path

import pytest
import torch

from vllm.v1.attention.ops.rabit_kv2 import (
    Rabit2OnlineStateRef,
    Rabit2SingleSequenceRuntime,
    rabit2_online_decode_attention_triton,
)


def _ref_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q0 = q[0].float()
    q_heads = q0.shape[0]
    kv_heads = k.shape[1]
    ratio = q_heads // kv_heads
    kv_idx = torch.arange(q_heads, device=q.device) // ratio
    kk = k[:, kv_idx, :].float()  # [T, QH, D]
    vv = v[:, kv_idx, :].float()
    scores = torch.einsum("hd,thd->ht", q0, kk) * (q0.shape[-1] ** -0.5)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("ht,thd->hd", probs, vv)
    return out.unsqueeze(0).to(q.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stage3b2_online_lifecycle_matches_exact_oracle() -> None:
    torch.manual_seed(20260810)
    device = torch.device("cuda")
    kv_heads = 4
    q_heads = 32
    d = 64

    runtime = Rabit2SingleSequenceRuntime(kv_heads, d, d)
    oracle = Rabit2OnlineStateRef(kv_heads, d, d)
    layout = runtime.layout
    cache = torch.zeros((8, 1, 1, 1, layout.page_bytes), dtype=torch.uint8, device=device)
    block_table = torch.tensor([5, 2, 7, 0, 1, 3, 4, 6], dtype=torch.int32, device=device)

    checkpoints = {1, 4, 5, 31, 32, 35, 36, 37, 63, 64, 67, 68, 69, 70}
    for token_idx in range(70):
        key = torch.randn((1, kv_heads, d), device=device, dtype=torch.bfloat16)
        value = torch.randn((1, kv_heads, d), device=device, dtype=torch.bfloat16)
        runtime.append(key, value, cache, block_table)
        oracle.append(key, value)
        n = token_idx + 1

        assert runtime.total_tokens == n
        assert runtime.closed_pages == len(oracle.pages)
        for page_idx, ref_page in enumerate(oracle.pages):
            physical = int(block_table[page_idx].item())
            got = cache[physical].reshape(-1)
            assert torch.equal(got, ref_page)

        if n in checkpoints:
            query = torch.randn((1, q_heads, d), device=device, dtype=torch.bfloat16)
            got = rabit2_online_decode_attention_triton(
                query, cache, block_table, runtime
            )
            full_k, full_v = oracle.materialize(dtype=torch.float32)
            ref = _ref_attention(query, full_k, full_v)
            max_abs = float((got.float() - ref.float()).abs().max().item())
            assert torch.isfinite(got).all()
            assert max_abs < 2.5e-2, f"n={n} max_abs={max_abs}"

    # Closed pages live in the physical cache only; the bounded sidecar is O(1).
    assert not hasattr(runtime, "pages")
    assert runtime.closed_pages == 2
    assert runtime.sidecar_bytes() > 0


def test_stage3b2_source_guards_and_dispatch_are_installed() -> None:
    import vllm.v1.attention.backends.triton_attn as triton_attn
    import vllm.v1.attention.ops.triton_reshape_and_cache_flash as reshape_mod
    import vllm.v1.attention.ops.triton_unified_attention as unified_mod

    attn_text = Path(triton_attn.__file__).read_text(encoding="utf-8")
    reshape_text = Path(reshape_mod.__file__).read_text(encoding="utf-8")
    unified_text = Path(unified_mod.__file__).read_text(encoding="utf-8")

    assert "def _forward_rabit_kv2" in attn_text
    assert "Rabit2SingleSequenceRuntime" in attn_text
    assert "rabit2_online_decode_attention_triton" in attn_text
    assert "RABIT_KV2 cache update is deferred" in reshape_text
    assert "RABIT_KV2 uses the backend sidecar dispatch" in unified_text
