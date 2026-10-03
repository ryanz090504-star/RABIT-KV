"""
OFFLINE proofs for the canonical-quality-v2 continuation-PPL runner (CPU only; no GPU, no model weights, no Modal).
Needs torch + transformers==4.48.2 and a fixtures directory holding the pinned raw inputs:
    <fixtures>/wikitext2_test.txt                     (sha256 = canonical_ppl_identity.WIKITEXT_SHA256)
    <fixtures>/llama/{tokenizer.json, tokenizer_config.json, special_tokens_map.json}     (pinned manifest hashes)
    <fixtures>/qwen/{tokenizer.json, tokenizer_config.json, vocab.json, merges.txt}       (pinned manifest hashes)

The frozen legacy protocol is taken from its SOURCE: the statements of benchmarks/quality/continuation_ppl.py
(run_quality) are extracted by AST and executed unmodified, then compared with canonical_ppl_core.

  W  dataset / windows: same URL; for BOTH pinned tokenizers the legacy statements and canonical_ppl_core produce
     identical (context, continuation) tensors for N = 32 (and the canonical N = 8 windows are the first 8); no BOS;
     the token-pool SHA-256 equals the pinned constant.
  S  scoring (tiny random BF16 Llama and Qwen2 models with GQA): the legacy evaluate_one(..., "bf16") loss equals
     score_bf16_batched exactly; the stepwise BF16 arm agrees with it to numerical tolerance; the first continuation
     token is scored from the prefill logit in every arm; with the canonical quantizers replaced by identity the
     rabit arm equals the stepwise BF16 arm bit for bit (the arms differ ONLY in the cache content).
  C  canonical path inside the real scoring loop: the cache is CanonicalRabitCache; exactly one token per forward;
     at EVERY decoded() call of EVERY layer the attention-visible K / V equal the full canonical_state() of all raw
     tokens appended so far; pages close during decode (aging exercised); the quantized arm differs from BF16.

Usage:  python canonical_ppl_offline_proofs.py --fixtures <dir> [--out <record.json>] [--print-pool-hashes]
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import canonical_ppl_core as core  # noqa: E402
import canonical_ppl_identity as ident  # noqa: E402
import canonical_rabit_quality as crq  # noqa: E402

LEGACY = ROOT / "benchmarks" / "quality" / "continuation_ppl.py"
BOUND_FILES = ["benchmarks/mlsys2027/canonical_ppl_core.py", "benchmarks/mlsys2027/canonical_ppl_identity.py",
               "benchmarks/mlsys2027/canonical_rabit_quality.py", "benchmarks/mlsys2027/exp14_model_snapshot.py",
               "benchmarks/mlsys2027/canonical_ppl_offline_proofs.py", "benchmarks/quality/continuation_ppl.py"]
TOKENIZER_FILES = {"llama3_1_8b": ("llama", ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]),
                   "qwen2_5_7b": ("qwen", ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"])}
TINY_CONTEXT, TINY_EVAL = 70, 40  # old region 66 -> 105 tokens: the third 32-token page closes during decode


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ------------------------------------------------------------------------------------ legacy source extraction
def legacy_body() -> list:
    tree = ast.parse(LEGACY.read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_quality").body


def _exec(stmts: list, ns: dict) -> dict:
    exec(compile(ast.Module(body=stmts, type_ignores=[]), str(LEGACY), "exec"), ns)  # noqa: S102
    return ns


def legacy_url() -> str:
    stmt = next(s for s in legacy_body() if isinstance(s, ast.Assign) and getattr(s.targets[0], "id", "") == "url")
    return _exec([stmt], {})["url"]


def legacy_windows(text: str, tokenizer, samples: int, context_tokens: int, eval_tokens: int) -> list:
    """Execute the legacy statements from `lines = [...]` through the window loop, unmodified."""
    body = legacy_body()
    first = next(i for i, s in enumerate(body) if isinstance(s, ast.Assign) and getattr(s.targets[0], "id", "") == "lines")
    last = next(i for i, s in enumerate(body) if isinstance(s, ast.For) and getattr(s.target, "id", "") == "sample_index")
    assigned = {getattr(s.targets[0], "id", None) for s in body[first:last + 1] if isinstance(s, ast.Assign)}
    if not {"lines", "needed", "token_ids", "pool", "sample_tensors", "span"} <= assigned:
        raise RuntimeError("legacy window statements not found as expected")
    ns = _exec(body[first:last + 1], {"response": SimpleNamespace(text=text), "tokenizer": tokenizer, "samples": samples,
                                      "context_tokens": context_tokens, "eval_tokens": eval_tokens, "torch": torch,
                                      "device": torch.device("cpu")})
    return ns["sample_tensors"]


class _TorchNoCuda:
    """torch with cuda.empty_cache / synchronize as no-ops (the legacy scorer calls them unconditionally)."""
    cuda = SimpleNamespace(empty_cache=lambda: None, synchronize=lambda: None)

    def __getattr__(self, name):
        return getattr(torch, name)


def legacy_bf16_row(model, context_ids, continuation_ids) -> dict:
    """The legacy evaluate_one(context, continuation, "bf16"), executed from source."""
    wanted = ("cache_to_tuple", "tensor_bytes", "bf16_cache_bytes", "evaluate_one")
    defs = [s for s in legacy_body() if isinstance(s, ast.FunctionDef) and s.name in wanted]
    if [d.name for d in defs] != list(wanted):
        raise RuntimeError("legacy scorer functions not found as expected")
    ns = _exec(defs, {"model": model, "torch": _TorchNoCuda(), "F": F, "gc": gc, "time": time, "math": math,
                      "DynamicCache": DynamicCache, "dtype": torch.bfloat16})
    return ns["evaluate_one"](context_ids, continuation_ids, "bf16")


# ------------------------------------------------------------------------------------------------ W proofs
def window_proofs(fixtures: Path) -> dict:
    raw = (fixtures / "wikitext2_test.txt").read_bytes()
    text = raw.decode("utf-8")
    out = {"legacy_url_identical": legacy_url() == ident.WIKITEXT_URL,
           "wikitext_sha256": hashlib.sha256(raw).hexdigest(), "wikitext_bytes": len(raw), "models": {}}
    out["wikitext_pinned"] = out["wikitext_sha256"] == ident.WIKITEXT_SHA256 and len(raw) == ident.WIKITEXT_BYTES
    s, c, e = ident.SAMPLES, ident.CONTEXT_TOKENS, ident.EVAL_TOKENS
    for key, (sub, files) in TOKENIZER_FILES.items():
        pinned = {p: h for p, _, h in ident.MODELS[key]["files"]}
        tok_ok = all(hashlib.sha256((fixtures / sub / f).read_bytes()).hexdigest() == pinned[f] for f in files)
        tok = AutoTokenizer.from_pretrained(str(fixtures / sub), trust_remote_code=True)
        leg32 = legacy_windows(text, tok, s, c, e)
        leg8 = legacy_windows(text, tok, 8, c, e)
        pool = core.build_token_pool(core.wikitext_lines(text), tok, s, c, e, ident.LINE_BLOCK)
        mine = core.split_windows(pool, s, c, e, torch.device("cpu"))
        same = len(mine) == len(leg32) == s and all(
            torch.equal(a, x) and torch.equal(b, y) and a.dtype == x.dtype and a.shape == x.shape == (1, c)
            and b.shape == y.shape == (1, e) for (a, b), (x, y) in zip(mine, leg32))
        subset = all(torch.equal(a, x) and torch.equal(b, y) for (a, b), (x, y) in zip(mine[:8], leg8))
        bos = tok.bos_token_id
        digest = core.pool_sha256(pool)
        out["models"][key] = {"tokenizer_files_match_manifest": tok_ok, "tokenizer_class": type(tok).__name__,
                              "windows_identical_to_legacy_n32": same, "canonical_n8_windows_are_first_8": subset,
                              "bos_token_id": bos, "no_bos_in_pool": bos is None or bos not in pool,
                              "pool_tokens": len(pool), "token_pool_sha256": digest,
                              "token_pool_sha256_pinned": digest == ident.MODELS[key]["token_pool_sha256"]}
    out["passed"] = out["legacy_url_identical"] and out["wikitext_pinned"] and all(
        m["tokenizer_files_match_manifest"] and m["windows_identical_to_legacy_n32"]
        and m["canonical_n8_windows_are_first_8"] and m["no_bos_in_pool"] and m["token_pool_sha256_pinned"]
        and m["pool_tokens"] == s * (c + e) for m in out["models"].values())
    return out


# ------------------------------------------------------------------------------------------------ S / C proofs
def tiny_models() -> dict:
    common = dict(vocab_size=320, hidden_size=256, intermediate_size=512, num_hidden_layers=3,
                  max_position_embeddings=512)
    torch.manual_seed(1234)
    llama = LlamaForCausalLM(LlamaConfig(num_attention_heads=4, num_key_value_heads=2, **common))  # H_kv 2, D 64
    qwen = Qwen2ForCausalLM(Qwen2Config(num_attention_heads=2, num_key_value_heads=1, **common))  # H_kv 1, D 128
    return {"tiny_llama_gqa_4_2_d64": llama.to(torch.bfloat16).eval(),
            "tiny_qwen2_gqa_2_1_d128": qwen.to(torch.bfloat16).eval()}


def _tiny_ids(seed: int):
    g = torch.Generator().manual_seed(seed)
    seq = torch.randint(0, 320, (1, TINY_CONTEXT + TINY_EVAL), generator=g)
    return seq[:, :TINY_CONTEXT], seq[:, TINY_CONTEXT:]


def scoring_proofs(name: str, model) -> dict:
    ctx, cont = _tiny_ids(7)
    legacy = legacy_bf16_row(model, ctx, cont)
    batched = core.score(model, ctx, cont, "bf16_batched")
    step = core.score(model, ctx, cont, "bf16")
    rabit = core.score(model, ctx, cont, "rabit")
    with torch.inference_mode():  # the prefill's last logit, scored on the first continuation token
        first = float(F.cross_entropy(model(input_ids=ctx).logits[:, -1, :].float(), cont[:, 0], reduction="sum"))
    # identity quantizers: the rabit arm must then equal the stepwise BF16 arm bit for bit
    k3, v2 = crq.k3, crq.v2
    crq.k3 = crq.v2 = lambda raw: {"decoded": raw.detach().float()}
    try:
        ident_arm = core.score(model, ctx, cont, "rabit")
    finally:
        crq.k3, crq.v2 = k3, v2
    rel = abs(step["loss_sum"] - batched["loss_sum"]) / batched["loss_sum"]
    out = {"legacy_bf16_loss_sum": legacy["loss_sum"], "batched_loss_sum": batched["loss_sum"],
           "stepwise_bf16_loss_sum": step["loss_sum"], "rabit_loss_sum": rabit["loss_sum"],
           "legacy_equals_batched_exactly": legacy["loss_sum"] == batched["loss_sum"]
           and legacy["tokens"] == batched["tokens"] == TINY_EVAL and legacy["ppl"] == batched["ppl"],
           "stepwise_vs_batched_rel_diff": rel, "stepwise_matches_batched": rel < 2e-3,
           "first_token_from_prefill_logit_all_arms": batched["token_nll"][0] == step["token_nll"][0]
           == rabit["token_nll"][0] == first,
           "identity_quantizer_rabit_equals_bf16_bitwise": ident_arm["token_nll"] == step["token_nll"],
           "identity_arm_used_canonical_cache": ident_arm["cache_class"] == "CanonicalRabitCache",
           "decode_forwards": [step["decode_forwards"], rabit["decode_forwards"]],
           "one_forward_per_token": step["decode_forwards"] == rabit["decode_forwards"] == TINY_EVAL - 1,
           "quantized_arm_differs_from_bf16": rabit["token_nll"][1:] != step["token_nll"][1:]}
    out["passed"] = all(out[k] for k in ("legacy_equals_batched_exactly", "stepwise_matches_batched",
                                         "first_token_from_prefill_logit_all_arms",
                                         "identity_quantizer_rabit_equals_bf16_bitwise",
                                         "identity_arm_used_canonical_cache", "one_forward_per_token",
                                         "quantized_arm_differs_from_bf16"))
    return out


def canonical_path_proofs(name: str, model) -> dict:
    """Spy on CanonicalLayerState inside core.score_stepwise: every decoded() equals canonical_state(all raw so far)."""
    ctx, cont = _tiny_ids(11)
    raw, stats = {}, {"decoded_calls": 0, "mismatches": 0, "append_sizes": set(), "final_n": set(), "closed_tokens": set()}
    append0, decoded0 = crq.CanonicalLayerState.append, crq.CanonicalLayerState.decoded

    def append(self, k, v):
        pk, pv = raw.get(id(self), (None, None))
        k16, v16 = k.detach().to(torch.bfloat16), v.detach().to(torch.bfloat16)
        raw[id(self)] = (k16 if pk is None else torch.cat([pk, k16]), v16 if pv is None else torch.cat([pv, v16]))
        stats["append_sizes"].add(int(k.shape[0]))
        return append0(self, k, v)

    def decoded(self):
        dk, dv = decoded0(self)
        ref = crq.canonical_state(*raw[id(self)])
        stats["decoded_calls"] += 1
        if not (torch.equal(dk, ref["decoded_k"]) and torch.equal(dv, ref["decoded_v"])):
            stats["mismatches"] += 1
        stats["final_n"].add(self.n)
        stats["closed_tokens"].add(0 if self.closed_k is None else int(self.closed_k.shape[0]))
        return dk, dv

    crq.CanonicalLayerState.append, crq.CanonicalLayerState.decoded = append, decoded
    try:
        row = core.score(model, ctx, cont, "rabit")
        parity = core.prefill_state_parity(model, ctx)
    finally:
        crq.CanonicalLayerState.append, crq.CanonicalLayerState.decoded = append0, decoded0
    layers, steps = model.config.num_hidden_layers, TINY_EVAL - 1
    n_end = TINY_CONTEXT + steps
    out = {"cache_class": row["cache_class"], "decoded_calls": stats["decoded_calls"],
           "expected_decoded_calls": 2 * layers * (1 + steps) - layers * steps,  # score: L*(1+steps); parity: L
           "state_mismatches": stats["mismatches"], "append_sizes": sorted(stats["append_sizes"]),
           "max_n": max(stats["final_n"]), "closed_tokens_seen": sorted(stats["closed_tokens"]),
           "prefill_state_parity": parity}
    out["passed"] = (out["cache_class"] == "CanonicalRabitCache" and out["state_mismatches"] == 0
                     and out["decoded_calls"] == out["expected_decoded_calls"]
                     and out["append_sizes"] == [1, TINY_CONTEXT] and out["max_n"] == n_end
                     and out["closed_tokens_seen"] == [64, 96] and parity["passed"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--out")
    ap.add_argument("--print-pool-hashes", action="store_true")
    a = ap.parse_args(argv)
    ident.check_constants()
    w = window_proofs(Path(a.fixtures))
    if a.print_pool_hashes:
        print(json.dumps({k: m["token_pool_sha256"] for k, m in w["models"].items()}, indent=2))
        return 0
    models = tiny_models()
    s = {n: scoring_proofs(n, m) for n, m in models.items()}
    c = {n: canonical_path_proofs(n, m) for n, m in models.items()}
    rec = {"kind": "canonical-quality-v2 continuation-PPL OFFLINE proofs (CPU; no GPU, no model weights; NOT a "
                   "quality result)",
           "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                           "transformers": str(transformers.__version__)},
           "protocol": {"samples": ident.SAMPLES, "context_tokens": ident.CONTEXT_TOKENS,
                        "eval_tokens": ident.EVAL_TOKENS, "policy": crq.POLICY},
           "bound_file_sha256_lf": {f: sha256_lf(ROOT / f) for f in BOUND_FILES},
           "W_dataset_windows": w, "S_scoring": s, "C_canonical_path": c}
    rec["passed"] = w["passed"] and all(x["passed"] for x in s.values()) and all(x["passed"] for x in c.values())
    text = json.dumps(rec, indent=2) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({"passed": rec["passed"], "W": w["passed"], "S": {n: x["passed"] for n, x in s.items()},
                      "C": {n: x["passed"] for n, x in c.items()}}))
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
