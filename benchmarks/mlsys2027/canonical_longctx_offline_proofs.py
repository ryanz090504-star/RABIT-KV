"""
OFFLINE proofs for the canonical-quality-v2 LONG-CONTEXT suite (CPU only; no GPU, no model weights, no Modal; NOT a
quality result). Needs torch + transformers==4.48.2 + datasets and a fixtures directory holding the pinned raw inputs:
    <fixtures>/wikitext2_test.txt                                  (sha256 = canonical_longctx_tasks.WIKITEXT_SHA256)
    <fixtures>/llama/{tokenizer.json, tokenizer_config.json, special_tokens_map.json}   (pinned manifest hashes)
    <fixtures>/longbench/{passage_retrieval_en.parquet, hotpotqa_e.parquet}             (pinned SHA-256 / sizes)

The frozen legacy protocol is taken from its SOURCE (benchmarks/quality/{niah,passage_retrieval,hotpotqa}.py, the
scripts of the accepted Exp12 run): statements are extracted by AST and executed unmodified, then compared.

  P  prompts / selection: for all 57 + 200 + 100 units the token ids of canonical_longctx_tasks equal the legacy
     build_prompt / build_prompt_ids output; the selected dataset indices equal the legacy selection; every LongBench
     unit equals the accepted Exp12 log (dataset index, original / used token counts, ground truth); the prompt-set
     SHA-256 equal the pinned values.
  S  scorers: on every prediction of the accepted Exp12 logs the frozen scorers equal the legacy scorer functions
     executed from source and reproduce the logged values; the legacy and the official HotpotQA scorers differ exactly
     as documented (articles).
  G  generation (tiny random BF16 Llama with GQA): core.generate(..., "bf16") returns exactly the tokens of the legacy
     generate_answer(..., "bf16") of all three scripts, with and without an EOS stop; with the canonical quantizers
     replaced by identity the rabit arm generates exactly the bf16 tokens (the arms differ ONLY in the cache content).
  C  canonical path inside the real generation loop: the cache is CanonicalRabitCache; exactly one token per forward
     (the last prompt token and every generated token); at EVERY decoded() call of EVERY layer the attention-visible
     K / V equal the full canonical_state() of all raw tokens appended so far (IdentityObserver; no id() keying);
     pages close during generation (aging exercised).

Usage:  python canonical_longctx_offline_proofs.py --fixtures <dir> [--out <record.json>] [--print-pins]
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import canonical_longctx_core as core  # noqa: E402
import canonical_longctx_pins as pins  # noqa: E402
import canonical_longctx_tasks as tasks  # noqa: E402
import canonical_ppl_identity as ident  # noqa: E402
import canonical_rabit_quality as crq  # noqa: E402
import run_canonical_longctx as runner  # noqa: E402  (stdlib helpers: legacy scorers by AST, Exp12 log rows)
from proof_observer import IdentityObserver  # noqa: E402

BOUND_FILES = ["benchmarks/mlsys2027/canonical_longctx_core.py", "benchmarks/mlsys2027/canonical_longctx_tasks.py",
               "benchmarks/mlsys2027/canonical_longctx_pins.py", "benchmarks/mlsys2027/canonical_rabit_quality.py",
               "benchmarks/mlsys2027/canonical_ppl_identity.py", "benchmarks/mlsys2027/exp14_model_snapshot.py",
               "benchmarks/mlsys2027/canonical_longctx_offline_proofs.py", "benchmarks/mlsys2027/proof_observer.py",
               "benchmarks/quality/niah.py", "benchmarks/quality/passage_retrieval.py", "benchmarks/quality/hotpotqa.py"]
ENTRY = {"niah": "run_niah", "passage_retrieval": "run_longbench", "hotpotqa": "run_longbench"}
PARQUET = {"passage_retrieval": "passage_retrieval_en.parquet", "hotpotqa": "hotpotqa_e.parquet"}
OBSERVER = "IdentityObserver v2 (object identity + retained direct references; no id() keying)"


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ------------------------------------------------------------------------------------ legacy source extraction
def legacy_body(task: str) -> list:
    tree = ast.parse((ROOT / runner.LEGACY_SCRIPT[task]).read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == ENTRY[task]).body


def _name(stmt) -> str:
    if isinstance(stmt, ast.FunctionDef):
        return stmt.name
    if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
        return stmt.targets[0].id
    return ""


def legacy_statements(task: str, names: list) -> list:
    """The named top-level assignments / function definitions of the legacy entry function, in source order."""
    stmts = [s for s in legacy_body(task) if _name(s) in names]
    if sorted({_name(s) for s in stmts}) != sorted(names):
        raise RuntimeError(f"{task}: legacy statements not found as expected: {names}")
    return stmts


def _exec(stmts: list, ns: dict, task: str) -> dict:
    exec(compile(ast.Module(body=stmts, type_ignores=[]), runner.LEGACY_SCRIPT[task], "exec"), ns)  # noqa: S102
    return ns


class _TorchNoCuda:
    """torch with cuda.empty_cache / synchronize as no-ops (the legacy generate_answer calls them unconditionally)."""
    cuda = SimpleNamespace(empty_cache=lambda: None, synchronize=lambda: None)

    def __getattr__(self, name):
        return getattr(torch, name)


# ------------------------------------------------------------------------------------------------ P proofs
def niah_prompt_proof(tok, wikitext: str) -> dict:
    ns = _exec(legacy_statements("niah", ["secret_code", "instruction", "needle", "question", "instruction_ids",
                                          "needle_ids", "question_ids", "filler_text", "base_filler_ids", "build_prompt"]),
               {"tokenizer": tok, "torch": torch, "device": torch.device("cpu"), "response": SimpleNamespace(text=wikitext)},
               "niah")
    parts = tasks.niah_parts(tok, wikitext)
    rows, same = [], True
    for c, d in tasks.niah_cases():
        mine = tasks.niah_prompt_ids(parts, c, d)
        leg = ns["build_prompt"](c, d)
        same = same and tuple(leg.shape) == (1, c) and leg.dtype == torch.long and leg[0].tolist() == mine
        rows.append({"key": [c, round(d, 2)], "prompt_tokens": len(mine), "prompt_ids_sha256": tasks.ids_sha256(mine)})
    keys_exp12 = runner.expected_keys("niah") == [x["key"] for x in runner.exp12_rows("niah")["identity"]]
    return {"units": len(rows), "prompts_identical_to_legacy": same, "secret_code_identical": ns["secret_code"] == tasks.NIAH_SECRET,
            "filler_tokens": len(parts["filler"]), "bos_in_any_prompt": tok.bos_token_id in parts["filler"] + parts["instruction"]
            + parts["needle"] + parts["question"], "keys_equal_exp12_grid": keys_exp12,
            "prompt_set_sha256": runner.prompt_set_sha256(rows),
            "passed": same and len(rows) == 57 and keys_exp12 and ns["secret_code"] == tasks.NIAH_SECRET}


def longbench_prompt_proof(task: str, tok, fixtures: Path) -> dict:
    d = tasks.DATASETS[task]
    path = fixtures / "longbench" / PARQUET[task]
    raw = path.read_bytes()
    file_ok = hashlib.sha256(raw).hexdigest() == d["sha256"] and len(raw) == d["bytes"]
    dataset = load_dataset("parquet", data_files={"test": str(path)}, split="test")
    names = ["prompt_template", "build_prompt_ids", "normalize_answers"] + (
        ["in_bucket", "filtered_indices"] if task == "hotpotqa" else [])
    ns = _exec(legacy_statements(task, names),
               {"tokenizer": tok, "torch": torch, "device": torch.device("cpu"), "max_input_tokens": tasks.MAX_INPUT_TOKENS,
                "dataset": dataset, "length_bucket": d["length_bucket"], "json": json, "re": re}, task)
    start, stop = d["sample_start"], d["sample_start"] + d["samples"]
    legacy_indices = ns["filtered_indices"][start:stop] if task == "hotpotqa" else list(range(start, stop))
    mine_indices = tasks.select_indices(task, list(dataset["length"]))
    rows, same, truncated = [], True, 0
    for i in mine_indices:
        ex = dataset[i]
        ids, original = tasks.longbench_prompt_ids(task, ex, tok)
        leg_ids, leg_original, leg_used = ns["build_prompt_ids"](ex)
        same = same and leg_ids[0].tolist() == ids and leg_original == original and leg_used == len(ids) \
            and ns["normalize_answers"](ex["answers"]) == tasks.normalize_answers(ex["answers"])
        truncated += original > len(ids)
        rows.append({"key": i, "prompt_tokens": len(ids), "prompt_ids_sha256": tasks.ids_sha256(ids),
                     "original_tokens": original, "answers": tasks.normalize_answers(ex["answers"])})
    exp12 = runner.exp12_rows(task)["identity"]
    mine_identity = [{"key": r["key"], "original_tokens": r["original_tokens"], "used_tokens": r["prompt_tokens"],
                      "answers": r["answers"]} for r in rows]
    out = {"file_matches_pin": file_ok, "dataset_rows": len(dataset), "units": len(rows),
           "indices_identical_to_legacy_selection": mine_indices == legacy_indices,
           "template_identical": ns["prompt_template"] == tasks.TEMPLATES[task],
           "prompts_identical_to_legacy": same, "units_equal_exp12_log": mine_identity == exp12,
           "units_truncated": truncated, "max_prompt_tokens": max(r["prompt_tokens"] for r in rows),
           "prompt_set_sha256": runner.prompt_set_sha256(rows)}
    out["passed"] = (file_ok and len(dataset) == d["rows"] and len(rows) == d["samples"] and same
                     and out["indices_identical_to_legacy_selection"] and out["template_identical"]
                     and out["units_equal_exp12_log"])
    return out


# ------------------------------------------------------------------------------------------------ S proofs
def scorer_proofs() -> dict:
    out = {}
    e = runner.exp12_rows("niah")
    n = [(m, r) for m in ("bf16", "rabit2") for r in e["rows"][m]]
    out["niah"] = {"predictions": len(n), "outcomes_reproduce_log": all(
        tasks.niah_score(r["answer"])["correct"] == (r["value"] == 1.0) for _, r in n)}
    for task, names, legacy_name in (("passage_retrieval", ["official_retrieval_score", "normalize_answers"], "official_retrieval_score"),
                                     ("hotpotqa", ["normalize_answer", "qa_f1_score", "normalize_answers"], "qa_f1_score")):
        leg, e = runner.legacy_functions(task, names), runner.exp12_rows(task)
        equal = logged = True
        count = 0
        for m in ("bf16", "rabit2"):
            for r, idn in zip(e["rows"][m], e["identity"]):
                mine = tasks.score(task, r["answer"], idn["answers"])["score"]
                theirs = max(leg[legacy_name](r["answer"], a) for a in idn["answers"])
                equal = equal and mine == theirs
                logged = logged and float(f"{mine:.3f}") == r["value"]
                count += 1
        out[task] = {"predictions": count, "frozen_scorer_equals_legacy_source": equal, "scores_reproduce_log": logged}
    off = [tasks.score("hotpotqa", r["answer"], idn["answers"]) for m in ("bf16", "rabit2")
           for r, idn in zip(runner.exp12_rows("hotpotqa")["rows"][m], runner.exp12_rows("hotpotqa")["identity"])]
    out["hotpotqa"].update({
        "legacy_article_example": tasks.qa_f1_score_legacy("the Eiffel Tower", "Eiffel Tower"),
        "official_article_example": tasks.qa_f1_score_official("the Eiffel Tower", "Eiffel Tower"),
        "exp12_predictions_where_the_two_scorers_differ": sum(x["score"] != x["score_official"] for x in off)})
    out["passed"] = (out["niah"]["outcomes_reproduce_log"] and all(
        out[t]["frozen_scorer_equals_legacy_source"] and out[t]["scores_reproduce_log"] for t in ("passage_retrieval", "hotpotqa"))
        and out["hotpotqa"]["legacy_article_example"] == 0.8 and out["hotpotqa"]["official_article_example"] == 1.0)
    return out


# ------------------------------------------------------------------------------------------------ G / C proofs
class _Tok:
    """Minimal tokenizer for the legacy generate_answer: an EOS id and an id-preserving decode."""

    def __init__(self, eos):
        self.eos_token_id = eos

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def tiny_model():
    torch.manual_seed(4321)
    cfg = LlamaConfig(vocab_size=320, hidden_size=256, intermediate_size=512, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)  # H_kv 2, D 64
    return LlamaForCausalLM(cfg).to(torch.bfloat16).eval()


def _prompt(seed: int, length: int) -> torch.Tensor:
    return torch.randint(0, 320, (1, length), generator=torch.Generator().manual_seed(seed))


def legacy_generate(task: str, model, prompt_ids, max_new_tokens: int, eos) -> list:
    """The legacy generate_answer(method='bf16', ...) executed from source; returns the kept token ids."""
    stmts = legacy_statements(task, ["cache_to_tuple", "tensor_bytes", "bf16_cache_bytes", "generate_answer"])
    ns = _exec(stmts, {"model": model, "tokenizer": _Tok(eos), "torch": _TorchNoCuda(), "gc": gc, "time": time, "re": re,
                       "max_new_tokens": max_new_tokens, "secret_code": tasks.NIAH_SECRET, "DynamicCache": DynamicCache}, task)
    r = ns["generate_answer"]("bf16", prompt_ids)
    answer = r["answer"] if task == "niah" else r[0]
    return [int(x) for x in answer.split()] if answer else []


def generation_proofs(model) -> dict:
    out = {}
    for task, length, max_new in (("niah", 71, 16), ("passage_retrieval", 90, 32), ("hotpotqa", 67, 32)):
        prompt = _prompt(100 + length, length)
        free = core.generate(model, prompt, "bf16", max_new, None)
        eos = free["generated_ids"][5]  # an id the model really emits: forces an EOS stop
        stop = core.generate(model, prompt, "bf16", max_new, eos)
        k3, v2 = crq.k3, crq.v2
        crq.k3 = crq.v2 = lambda raw: {"decoded": raw.detach().float()}  # identity quantizers
        try:
            ident_arm = core.generate(model, prompt, "rabit", max_new, eos)
            ident_free = core.generate(model, prompt, "rabit", max_new, None)
        finally:
            crq.k3, crq.v2 = k3, v2
        rabit = core.generate(model, prompt, "rabit", max_new, None)
        o = {"prompt_tokens": length, "max_new_tokens": max_new,
             "bf16_equals_legacy_no_eos": free["generated_ids"] == legacy_generate(task, model, prompt, max_new, None),
             "bf16_equals_legacy_with_eos": stop["generated_ids"] == legacy_generate(task, model, prompt, max_new, eos),
             "kept_tokens_no_eos": len(free["generated_ids"]), "kept_tokens_with_eos": len(stop["generated_ids"]),
             "eos_stop_exercised": stop["stopped_on_eos"] and len(stop["generated_ids"]) < max_new and eos not in stop["generated_ids"],
             "identity_quantizer_rabit_equals_bf16": ident_arm["generated_ids"] == stop["generated_ids"]
             and ident_free["generated_ids"] == free["generated_ids"] and ident_arm["cache_class"] == "CanonicalRabitCache",
             "decode_forwards": [free["decode_forwards"], rabit["decode_forwards"]],
             "one_forward_per_kept_token_plus_last_prompt_token": free["decode_forwards"] == len(free["generated_ids"]) + 1
             and rabit["decode_forwards"] == len(rabit["generated_ids"]) + 1,
             "cache_classes": [free["cache_class"], rabit["cache_class"]],
             "prefix_is_prompt_minus_one": free["prefix_tokens"] == rabit["prefix_tokens"] == length - 1}
        o["passed"] = all(o[k] for k in ("bf16_equals_legacy_no_eos", "bf16_equals_legacy_with_eos", "eos_stop_exercised",
                                         "identity_quantizer_rabit_equals_bf16",
                                         "one_forward_per_kept_token_plus_last_prompt_token", "prefix_is_prompt_minus_one")) \
            and o["cache_classes"] == ["DynamicCache", "CanonicalRabitCache"] and o["kept_tokens_no_eos"] == max_new
        out[task] = o
    return out


def canonical_path_proof(model) -> dict:
    """Spy on CanonicalLayerState inside core.generate: every decoded() equals canonical_state(all raw so far)."""
    length, max_new = 71, 32  # prefix 70: old region 66 -> 99 tokens: the third 32-token page closes during generation
    prompt = _prompt(11, length)
    raw = IdentityObserver()
    stats = {"decoded_calls": 0, "mismatches": 0, "stale": 0, "append_sizes": set(), "closed_tokens": set(), "max_n": 0}
    append0, decoded0 = crq.CanonicalLayerState.append, crq.CanonicalLayerState.decoded

    def append(self, k, v):
        raw.record(self, (k.detach().to(torch.bfloat16), v.detach().to(torch.bfloat16)))
        stats["append_sizes"].add(int(k.shape[0]))
        return append0(self, k, v)

    def decoded(self):
        dk, dv = decoded0(self)
        chunks = raw.items(self)
        raw_k, raw_v = torch.cat([c[0] for c in chunks]), torch.cat([c[1] for c in chunks])
        if int(raw_k.shape[0]) != self.n:
            stats["stale"] += 1
        ref = crq.canonical_state(raw_k, raw_v)
        stats["decoded_calls"] += 1
        if not (torch.equal(dk, ref["decoded_k"]) and torch.equal(dv, ref["decoded_v"])):
            stats["mismatches"] += 1
        stats["max_n"] = max(stats["max_n"], self.n)
        stats["closed_tokens"].add(0 if self.closed_k is None else int(self.closed_k.shape[0]))
        return dk, dv

    crq.CanonicalLayerState.append, crq.CanonicalLayerState.decoded = append, decoded
    try:
        row = core.generate(model, prompt, "rabit", max_new, None)
    finally:
        crq.CanonicalLayerState.append, crq.CanonicalLayerState.decoded = append0, decoded0
    layers = model.config.num_hidden_layers
    out = {"cache_class": row["cache_class"], "decode_forwards": row["decode_forwards"],
           "decoded_calls": stats["decoded_calls"], "expected_decoded_calls": layers * (1 + row["decode_forwards"]),
           "state_mismatches": stats["mismatches"], "stale_state_associations": stats["stale"],
           "observed_layer_states": len(raw), "expected_observed_layer_states": layers,
           "append_sizes": sorted(stats["append_sizes"]), "max_n": stats["max_n"],
           "closed_tokens_seen": sorted(stats["closed_tokens"]), "final_cache_tokens": row["final_cache_tokens"]}
    out["passed"] = (out["cache_class"] == "CanonicalRabitCache" and out["state_mismatches"] == 0
                     and out["stale_state_associations"] == 0 and out["observed_layer_states"] == layers
                     and out["decoded_calls"] == out["expected_decoded_calls"]
                     and out["append_sizes"] == [1, length - 1] and out["decode_forwards"] == max_new + 1
                     and out["max_n"] == length - 1 + max_new + 1 and out["closed_tokens_seen"] == [64, 96])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--out")
    ap.add_argument("--print-pins", action="store_true")
    a = ap.parse_args(argv)
    fx = Path(a.fixtures)
    ident.check_constants()
    raw = (fx / "wikitext2_test.txt").read_bytes()
    wikitext_ok = hashlib.sha256(raw).hexdigest() == tasks.WIKITEXT_SHA256
    pinned = {p: h for p, _, h in ident.MODELS["llama3_1_8b"]["files"]}
    tok_ok = all(hashlib.sha256((fx / "llama" / f).read_bytes()).hexdigest() == pinned[f]
                 for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"))
    tok = AutoTokenizer.from_pretrained(str(fx / "llama"), trust_remote_code=True)
    p = {"niah": niah_prompt_proof(tok, raw.decode("utf-8")),
         "passage_retrieval": longbench_prompt_proof("passage_retrieval", tok, fx),
         "hotpotqa": longbench_prompt_proof("hotpotqa", tok, fx)}
    sets = {t: {"units": p[t]["units"], "prompt_set_sha256": p[t]["prompt_set_sha256"]} for t in p}
    if a.print_pins:
        print(json.dumps(sets, indent=2))
        return 0
    pins_ok = sets == pins.PROMPT_SETS
    eos_ok = tok.eos_token == "<|eot_id|>" and tok.eos_token_id is not None
    s = scorer_proofs()
    model = tiny_model()
    g, c = generation_proofs(model), canonical_path_proof(model)
    rec = {"kind": "canonical-quality-v2 long-context OFFLINE proofs (CPU; no GPU, no model weights; NOT a quality result)",
           "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                           "transformers": str(transformers.__version__), "datasets": str(__import__("datasets").__version__)},
           "observer": OBSERVER, "policy": crq.POLICY,
           "bound_file_sha256_lf": {f: sha256_lf(ROOT / f) for f in BOUND_FILES},
           "inputs": {"wikitext_pinned": wikitext_ok, "tokenizer_files_match_manifest": tok_ok,
                      "tokenizer_eos_token": tok.eos_token, "tokenizer_eos_token_id": tok.eos_token_id,
                      "eos_is_eot_id": eos_ok},
           "P_prompts_and_selection": p, "prompt_sets_equal_pins": pins_ok, "S_scorers": s, "G_generation": g,
           "C_canonical_path": c}
    rec["passed"] = (wikitext_ok and tok_ok and eos_ok and pins_ok and all(x["passed"] for x in p.values()) and s["passed"]
                     and all(x["passed"] for x in g.values()) and c["passed"])
    if a.out:
        Path(a.out).write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"passed": rec["passed"], "inputs": rec["inputs"], "P": {t: x["passed"] for t, x in p.items()},
                      "pins": pins_ok, "S": s["passed"], "G": {t: x["passed"] for t, x in g.items()}, "C": c["passed"]}))
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
