# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import pytest
import torch

from vllm.v1.attention.ops.rabit_kv2 import (
    Rabit2SingleSequenceRuntime,
    decode_rabit2_page_ref,
    rabit2_finish_requests,
    rabit2_get_active_request_ids,
    rabit2_online_decode_attention_triton,
    rabit2_register_attention_impl,
    rabit2_set_active_batch,
)


def test_stage3c_request_registry_and_cleanup():
    class DummyImpl:
        def __init__(self):
            self._rabit2_runtimes = {"req-a": object(), "req-b": object()}

    dummy = DummyImpl()
    rabit2_register_attention_impl(dummy)
    rabit2_set_active_batch(("req-a", "req-b"), (3, 1), (0, 9))
    assert rabit2_get_active_request_ids() == ("req-a", "req-b")
    removed = rabit2_finish_requests(("req-a",))
    assert removed >= 1
    assert "req-a" not in dummy._rabit2_runtimes
    assert "req-b" in dummy._rabit2_runtimes


def _materialize(runtime, cache, block_row):
    k_parts = []
    v_parts = []
    cache2d = cache.reshape(cache.shape[0], -1)
    for page_idx in range(runtime.closed_pages):
        physical = int(block_row[page_idx].item())
        k, v = decode_rabit2_page_ref(
            cache2d[physical], layout=runtime.layout, dtype=torch.float32
        )
        k_parts.append(k)
        v_parts.append(v)
    tail_k, tail_v = runtime.tail_materialize(torch.float32)
    if tail_k.numel():
        k_parts.append(tail_k)
        v_parts.append(tail_v)
    return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)


def _gqa_ref(q, k, v, softmax_scale):
    # q [QH,D], k/v [T,KVH,D]
    rep = q.shape[0] // k.shape[1]
    k_rep = k.repeat_interleave(rep, dim=1)
    v_rep = v.repeat_interleave(rep, dim=1)
    scores = torch.einsum("hd,thd->ht", q.float(), k_rep.float())
    probs = torch.softmax(scores * softmax_scale, dim=-1)
    return torch.einsum("ht,thd->hd", probs, v_rep.float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_stage3c_two_sequences_share_physical_cache_without_state_aliasing():
    torch.manual_seed(314159)
    device = torch.device("cuda")
    nkv, d, qh = 8, 128, 32

    r1 = Rabit2SingleSequenceRuntime(nkv, d, d)
    r2 = Rabit2SingleSequenceRuntime(nkv, d, d)
    page_bytes = r1.layout.page_bytes
    cache = torch.zeros((8, 1, 1, 1, page_bytes), dtype=torch.uint8, device=device)
    row1 = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
    row2 = torch.tensor([4, 5, 6, 7], dtype=torch.int32, device=device)

    k1 = torch.randn((70, nkv, d), dtype=torch.bfloat16, device=device)
    v1 = torch.randn_like(k1)
    k2 = torch.randn((70, nkv, d), dtype=torch.bfloat16, device=device)
    v2 = torch.randn_like(k2)
    r1.append(k1, v1, cache, row1)
    r2.append(k2, v2, cache, row2)

    assert r1.closed_pages == 2
    assert r2.closed_pages == 2

    scale = d ** -0.5
    for runtime, row in ((r1, row1), (r2, row2)):
        q = torch.randn((1, qh, d), dtype=torch.bfloat16, device=device)
        got = rabit2_online_decode_attention_triton(
            q, cache, row, runtime, softmax_scale=scale
        )[0].float()
        k_full, v_full = _materialize(runtime, cache, row)
        ref = _gqa_ref(q[0], k_full, v_full, scale)
        max_abs = float((got - ref).abs().max().item())
        assert max_abs < 5e-3, max_abs
