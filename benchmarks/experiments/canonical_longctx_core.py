"""
canonical-quality-v2 LONG-CONTEXT suite -- generation core (torch). Imported by the Modal app and by the offline proofs
(canonical_longctx_offline_proofs.py), which compare it with the frozen legacy generate_answer() source.

The frozen legacy generation schedule (benchmarks/quality/{niah,passage_retrieval,hotpotqa}.py: generate_answer):
    prefix = prompt[:-1] is prefilled in BF16; the LAST prompt token is fed as the first decode step; then greedy
    argmax, one token per forward; a token equal to the tokenizer EOS stops generation and is not kept; at most
    max_new_tokens tokens are kept (one further forward follows the last kept token, as in the legacy loop).
Arms -- identical prompt ids, identical loop:
    bf16    the BF16 prefill cache itself (the legacy BF16 path, unchanged)
    rabit   CanonicalRabitCache.from_prefill(BF16 prefill): the canonical RABIT state (K3 / V2 / G32 / R4 / META8g64);
            the last prompt token and every generated token are appended one at a time and age through the R4 residual
            -> open group -> closed 32-token pages. canonical_rabit_quality.py is the only RABIT implementation.
No quantization arithmetic and no scoring lives here.
"""

from __future__ import annotations

import torch

import canonical_rabit_quality as crq

ARMS = ("bf16", "rabit")


def generate(model, prompt_ids: torch.Tensor, arm: str, max_new_tokens: int, eos_id) -> dict:
    """prompt_ids: [1, T]. Returns the kept generated token ids and loop facts (no text, no score)."""
    if arm not in ARMS:
        raise ValueError(arm)
    prefix_ids = prompt_ids[:, :-1]
    final_prompt_token = prompt_ids[:, -1:]
    with torch.inference_mode():
        prefill_output = model(input_ids=prefix_ids, use_cache=True)
        prefix_cache = prefill_output.past_key_values
        if arm == "bf16":
            cache = prefix_cache
        else:
            cache = crq.make_canonical_cache_class().from_prefill(prefix_cache)
        first = cache
        step_output = model(input_ids=final_prompt_token, past_key_values=cache, use_cache=True)
        cache = step_output.past_key_values
        next_token = torch.argmax(step_output.logits[:, -1, :], dim=-1, keepdim=True)
        generated, forwards, stopped_on_eos = [], 1, False
        for _ in range(max_new_tokens):
            token_id = int(next_token.item())
            if eos_id is not None and token_id == eos_id:
                stopped_on_eos = True
                break
            generated.append(token_id)
            step_output = model(input_ids=next_token, past_key_values=cache, use_cache=True)
            cache = step_output.past_key_values
            forwards += 1
            next_token = torch.argmax(step_output.logits[:, -1, :], dim=-1, keepdim=True)
        if cache is not first:
            raise RuntimeError("model replaced the cache object")
        seen = int(cache.get_seq_length())
        if seen != int(prefix_ids.shape[1]) + forwards:
            raise RuntimeError(f"cache length {seen} != prefix {int(prefix_ids.shape[1])} + {forwards} decode forwards")
        if arm == "rabit" and {s.n for s in cache.states} != {seen}:
            raise RuntimeError("canonical layer states do not all hold prefix + decode tokens")
    return {"generated_ids": generated, "cache_class": type(cache).__name__, "decode_forwards": forwards,
            "prefix_tokens": int(prefix_ids.shape[1]), "final_cache_tokens": seen, "stopped_on_eos": stopped_on_eos}
