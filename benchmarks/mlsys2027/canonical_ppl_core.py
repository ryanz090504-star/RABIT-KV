"""
canonical-quality-v2 continuation PPL -- dataset windows and scorers (torch; no Modal, no file / network access).
Used unchanged inside the GPU container (canonical_ppl_modal.py) and by the offline proofs
(canonical_ppl_offline_proofs.py), which compare it against the frozen legacy protocol source
benchmarks/quality/continuation_ppl.py.

Frozen legacy protocol reproduced here (dataset, windows, order, scorer):
  * WikiText-2 test raw text -> non-empty stripped lines -> blocks of 64 lines joined by "\n" -> tokenizer(text,
    add_special_tokens=False) (no BOS), concatenated until samples x (context + eval) tokens are available; the first
    samples x (context + eval) tokens are split into consecutive non-overlapping windows (context, continuation);
  * BF16 prefill of the context; the FIRST continuation token is scored from the prefill's last logit;
  * score = sum of token cross-entropy over the continuation (logits cast to float32), PPL = exp(loss / tokens).

Arms:
  bf16_batched   the legacy BF16 scorer verbatim (continuation[:-1] in ONE forward, use_cache=False) -- a CONTROL that
                 fingerprints model + windows against the accepted legacy BF16 values; not part of the comparison;
  bf16           BF16 cache, continuation tokens teacher-forced ONE AT A TIME;
  rabit          canonical RABIT cache (canonical_rabit_quality.CanonicalRabitCache, the ONLY RABIT implementation):
                 BF16 prefill -> canonical state; continuation tokens teacher-forced ONE AT A TIME, the cache aging
                 canonically (R4 residual -> open group -> closed 32-token page) at every step.
bf16 and rabit run the SAME stepwise loop on the SAME token ids; they differ only in the cache object.
This module contains no quantization arithmetic.
"""

from __future__ import annotations

import hashlib
import math
import struct

import torch
import torch.nn.functional as F

import canonical_rabit_quality as crq  # the sole RABIT quality implementation

ARMS = ("bf16_batched", "bf16", "rabit")


# ------------------------------------------------------------------------------------------------ windows
def wikitext_lines(text: str) -> list:
    return [line.strip() for line in text.splitlines() if line.strip()]


def build_token_pool(lines: list, tokenizer, samples: int, context_tokens: int, eval_tokens: int,
                     line_block: int = 64) -> list:
    """The first samples x (context + eval) token ids of the block-tokenized stream (no special tokens)."""
    needed = samples * (context_tokens + eval_tokens)
    token_ids: list = []
    for start in range(0, len(lines), line_block):
        text = "\n".join(lines[start:start + line_block])
        token_ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if len(token_ids) >= needed:
            break
    if len(token_ids) < needed:
        raise RuntimeError(f"Need {needed} WikiText tokens, found {len(token_ids)}.")
    return [int(t) for t in token_ids[:needed]]


def pool_sha256(pool: list) -> str:
    """SHA-256 of the token pool as little-endian int64."""
    return hashlib.sha256(struct.pack(f"<{len(pool)}q", *pool)).hexdigest()


def split_windows(pool: list, samples: int, context_tokens: int, eval_tokens: int, device) -> list:
    """[(context_ids [1, C], continuation_ids [1, E]), ...] -- consecutive, non-overlapping, in stream order."""
    span = context_tokens + eval_tokens
    t = torch.tensor(pool, dtype=torch.long)
    out = []
    for i in range(samples):
        seq = t[i * span:(i + 1) * span].unsqueeze(0).to(device)
        out.append((seq[:, :context_tokens], seq[:, context_tokens:]))
    return out


# ------------------------------------------------------------------------------------------------ scorers
def _token_nll(logits: torch.Tensor, continuation_ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), continuation_ids.reshape(-1),
                           reduction="none")


def _row(logits: torch.Tensor, continuation_ids: torch.Tensor) -> dict:
    loss_sum = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), continuation_ids.reshape(-1),
                               reduction="sum")  # the legacy expression
    tokens = int(continuation_ids.numel())
    loss = float(loss_sum.item())
    return {"loss_sum": loss, "tokens": tokens, "ppl": math.exp(min(loss / tokens, 50.0)),
            "token_nll": [float(x) for x in _token_nll(logits, continuation_ids).tolist()]}


def score_bf16_batched(model, context_ids: torch.Tensor, continuation_ids: torch.Tensor) -> dict:
    """The legacy BF16 path verbatim: prefill, then continuation[:-1] in one forward on the prefill cache."""
    with torch.inference_mode():
        prefill = model(input_ids=context_ids, use_cache=True)
        first_logit = prefill.logits[:, -1:, :]
        out = model(input_ids=continuation_ids[:, :-1], past_key_values=prefill.past_key_values, use_cache=False)
        return _row(torch.cat([first_logit, out.logits], dim=1), continuation_ids)


def canonical_cache_from_prefill(prefill_cache):
    return crq.make_canonical_cache_class().from_prefill(prefill_cache)


def score_stepwise(model, context_ids: torch.Tensor, continuation_ids: torch.Tensor, arm: str) -> dict:
    """BF16 prefill; first continuation token from the prefill logit; then one teacher-forced token per forward.
    arm == "bf16": the prefill cache itself; arm == "rabit": the canonical RABIT cache built from the prefill."""
    if arm not in ("bf16", "rabit"):
        raise ValueError(arm)
    context_len, steps = int(context_ids.shape[1]), int(continuation_ids.shape[1]) - 1
    with torch.inference_mode():
        prefill = model(input_ids=context_ids, use_cache=True)
        logits = [prefill.logits[:, -1:, :]]
        cache = prefill.past_key_values if arm == "bf16" else canonical_cache_from_prefill(prefill.past_key_values)
        for t in range(steps):
            out = model(input_ids=continuation_ids[:, t:t + 1], past_key_values=cache, use_cache=True)
            if out.past_key_values is not cache:
                raise RuntimeError("model replaced the cache object")
            if cache.get_seq_length() != context_len + t + 1:
                raise RuntimeError(f"cache length {cache.get_seq_length()} != {context_len + t + 1}")
            logits.append(out.logits)
        if arm == "rabit" and {s.n for s in cache.states} != {context_len + steps}:
            raise RuntimeError("canonical layer states do not all hold context + steps tokens")
        row = _row(torch.cat(logits, dim=1), continuation_ids)
    row["cache_class"] = type(cache).__name__
    row["decode_forwards"] = steps
    return row


def score(model, context_ids: torch.Tensor, continuation_ids: torch.Tensor, arm: str) -> dict:
    if arm == "bf16_batched":
        return score_bf16_batched(model, context_ids, continuation_ids)
    return score_stepwise(model, context_ids, continuation_ids, arm)


# ------------------------------------------------------------------------------------- device / oracle self-check
def prefill_state_parity(model, context_ids: torch.Tensor) -> dict:
    """The canonical cache built ON THE MODEL DEVICE equals, bit for bit, the full (non-incremental) canonical state
    computed on CPU from the same raw prefill K / V (the definition validated by the accepted CPU parity, 8fa9a9c),
    for every layer. Run once before any scoring."""
    with torch.inference_mode():
        prefill = model(input_ids=context_ids, use_cache=True)
        legacy = prefill.past_key_values.to_legacy_cache()
        raw = [(k[0].permute(1, 0, 2).cpu(), v[0].permute(1, 0, 2).cpu()) for k, v in legacy]
        dtype = legacy[0][0].dtype
        cache = canonical_cache_from_prefill(prefill.past_key_values)
        bad = []
        for layer, (rk, rv) in enumerate(raw):
            st = crq.canonical_state(rk, rv)
            ek = st["decoded_k"].permute(1, 0, 2).unsqueeze(0).to(dtype)
            ev = st["decoded_v"].permute(1, 0, 2).unsqueeze(0).to(dtype)
            if not (torch.equal(ek, cache.key_cache[layer].cpu()) and torch.equal(ev, cache.value_cache[layer].cpu())):
                bad.append(layer)
        n = int(context_ids.shape[1])
        return {"layers": len(raw), "tokens": n, "kv_heads": int(raw[0][0].shape[1]),
                "head_dim": int(raw[0][0].shape[2]), "model_dtype": str(dtype), "mismatched_layers": bad,
                "passed": not bad and len(raw) > 0}


def aggregate(rows: list) -> dict:
    loss, tokens = sum(r["loss_sum"] for r in rows), sum(r["tokens"] for r in rows)
    return {"loss": loss / tokens, "ppl": math.exp(min(loss / tokens, 50.0)), "tokens": tokens, "windows": len(rows)}
