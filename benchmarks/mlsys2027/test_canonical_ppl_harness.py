"""
Static / offline tests of the canonical-quality-v2 continuation-PPL harness (stdlib + modal client only; no torch, no
GPU, no model, no network). Run by run_canonical_ppl.py's preflight. The torch-dependent proofs (legacy window /
scorer equivalence, canonical path) are in canonical_ppl_offline_proofs.py and recorded in
results/mlsys2027/canonical_quality_v2/continuation_ppl/offline_proof_record.json.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import canonical_ppl_identity as ident  # noqa: E402
import exp14_hardware_binding as hb  # noqa: E402
import run_canonical_ppl as runner  # noqa: E402

CORE, MODAL, IDENT, CRQ = (HERE / n for n in ("canonical_ppl_core.py", "canonical_ppl_modal.py",
                                              "canonical_ppl_identity.py", "canonical_rabit_quality.py"))
STDLIB = set(sys.stdlib_module_names)
# identifiers of the LEGACY logical evaluator / oracle that must not appear in any code shipped to the container
LEGACY_NAMES = {"q_seq_affine", "q_group_affine", "q_group_sym", "q_tensor", "q_with_residual", "dequantize_state",
                "quantize_then_dequantize_cache", "encode_metadata", "decode_metadata", "config_for_method",
                "tuple_to_dynamic_cache", "stored_state_logical_bytes", "kvquant_k3", "quantize_rabit2_kv_ref",
                "dequantize_rabit2_kv_ref", "continuation_ppl", "run_quality"}
QUANT_ARITHMETIC = {"round", "clamp", "amin", "amax", "uint8", "floor", "ceil", "quantize"}


def tree(p: Path) -> ast.Module:
    return ast.parse(p.read_text(encoding="utf-8"))


def imports(p: Path) -> set:
    out = set()
    for n in ast.walk(tree(p)):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module.split(".")[0])
    return out


def identifiers(p: Path) -> set:
    out = set()
    for n in ast.walk(tree(p)):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.alias):
            out |= {n.name.split(".")[0], n.asname or ""}
    return out


def attrs_of(p: Path, name: str) -> set:
    return {n.attr for n in ast.walk(tree(p)) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == name}


def modal_app():
    spec = importlib.util.spec_from_file_location("canonical_ppl_modal_under_test", MODAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_ppl_def() -> ast.FunctionDef:
    return next(n for n in tree(MODAL).body if isinstance(n, ast.FunctionDef) and n.name == "run_ppl")


# ------------------------------------------------------------------------------------------------ sole implementation
def test_shipped_files_are_exactly_the_four_canonical_files():
    files = next(ast.literal_eval(n.value) for n in tree(MODAL).body if isinstance(n, ast.Assign)
                 and getattr(n.targets[0], "id", "") == "FILES")
    assert files == runner.SHIPPED == ["canonical_rabit_quality.py", "canonical_ppl_core.py",
                                       "canonical_ppl_identity.py", "exp14_model_snapshot.py"]
    assert all((HERE / f).is_file() for f in files)
    src = MODAL.read_text(encoding="utf-8")
    assert src.count("add_local_file") == 1 and "add_local_dir" not in src and "add_local_python_source" not in src


def test_import_graph_reaches_only_the_canonical_implementation():
    third_party = {"torch", "transformers", "modelscope", "requests", "modal"}
    assert imports(CORE) - STDLIB == {"torch", "canonical_rabit_quality"}
    assert imports(IDENT) - STDLIB == {"exp14_model_snapshot"}
    assert imports(HERE / "exp14_model_snapshot.py") - STDLIB == set()
    assert imports(CRQ) - STDLIB == {"torch", "transformers"}
    assert imports(MODAL) - STDLIB - third_party == {"canonical_ppl_identity", "canonical_ppl_core",
                                                     "canonical_rabit_quality"}
    # module level of the Modal app: no sibling import (parity Attempt 1 failure mode)
    top = {a.name.split(".")[0] for n in tree(MODAL).body if isinstance(n, ast.Import) for a in n.names}
    assert top - STDLIB == {"modal"}


def test_no_legacy_quantizer_identifier_in_shipped_code():
    for p in (CORE, MODAL, IDENT, CRQ, HERE / "exp14_model_snapshot.py"):
        hit = identifiers(p) & LEGACY_NAMES
        assert not hit, (p.name, hit)
    for p in (CORE, MODAL):  # no path to the legacy scripts / oracle / vLLM source either
        src = p.read_text(encoding="utf-8")
        code = "\n".join(ast.unparse(n) for n in tree(p).body if not (isinstance(n, ast.Expr)
                                                                      and isinstance(n.value, ast.Constant)))
        assert "vllm-kvquant" not in src and "benchmarks/quality" not in code, p.name


def test_runner_code_contains_no_quantization_arithmetic():
    for p in (CORE, MODAL):
        assert not (identifiers(p) & QUANT_ARITHMETIC), (p.name, identifiers(p) & QUANT_ARITHMETIC)
    assert attrs_of(CORE, "crq") == {"make_canonical_cache_class", "canonical_state"}
    assert attrs_of(MODAL, "crq") == {"POLICY", "logical_bytes", "__file__"}
    # the rabit arm's cache comes from exactly one place
    assert CORE.read_text(encoding="utf-8").count("make_canonical_cache_class") == 1


def test_canonical_implementation_is_c360697_and_policy_frozen():
    sha = hashlib.sha256(CRQ.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert sha == modal_app().CANONICAL_RABIT_QUALITY_SHA256_LF == "195cb4896c4708cc6ecc450930733962ea3e08e21440fc9406dedbe1bb4c11d5"
    assert runner.git("diff", "--name-only", "c36069781b259e2e9b4d8adf60b7c58ea5df5cac", "--",
                      "benchmarks/mlsys2027/canonical_rabit_quality.py") == ""
    consts = {n.targets[0].id: ast.literal_eval(n.value) for n in tree(CRQ).body if isinstance(n, ast.Assign)
              and isinstance(n.value, ast.Constant)}
    assert (consts["GROUP"], consts["RESIDUAL"], consts["META_GROUP"], consts["K_LEVELS"], consts["V_LEVELS"]) == (32, 4, 64, 7.0, 3.0)


# ------------------------------------------------------------------------------------------------ hardware
def test_strict_h100_selector_and_guard_first():
    fn = run_ppl_def()
    gpu = [kw.value.value for d in fn.decorator_list if isinstance(d, ast.Call) for kw in d.keywords if kw.arg == "gpu"]
    assert gpu == ["H100!:1"]
    assert ast.unparse(fn.body[0]) == "_gpus = _gpu_query()"
    exp14 = next(n for n in hb.guard_statements() if isinstance(n, ast.Assign) and n.targets[0].id == "_hw_ok")
    assert ast.dump(fn.body[1]) == ast.dump(exp14)  # the accepted Exp14 predicate, verbatim
    guard = fn.body[3]
    assert isinstance(guard, ast.If) and ast.unparse(guard.test) == "not _hw_ok" and isinstance(guard.body[-1], ast.Raise)
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for s in fn.body[:4] for n in ast.walk(s))


def test_hardware_predicate():
    h100 = {"name": "NVIDIA H100 80GB HBM3", "memory.total": "81559"}
    assert ident.hardware_ok([h100]) and hb.hardware_ok([h100])
    for bad in ([{"name": "NVIDIA H200", "memory.total": "143771"}], [h100, h100], [],
                [{"name": "NVIDIA A100-SXM4-80GB", "memory.total": "81920"}],
                [{"name": "NVIDIA H100", "memory.total": "40000"}]):
        assert not ident.hardware_ok(bad) and not hb.hardware_ok(bad)


def test_container_order_fails_closed_before_scoring():
    src = ast.unparse(run_ppl_def())
    order = ["if not _hw_ok", "if not files_ok", "snapshot_download(", "ident.verify_dir(", "ident.WIKITEXT_SHA256",
             "m['token_pool_sha256']", "AutoModelForCausalLM.from_pretrained(", "core.prefill_state_parity(",
             "if not parity['passed']", "core.score("]
    pos = [src.index(s) for s in order]
    assert pos == sorted(pos), list(zip(order, pos))
    assert src.count("core.score(") == 1


# ------------------------------------------------------------------------------------------------ identity / protocol
def test_identity_constants():
    ident.check_constants()
    assert (ident.SAMPLES, ident.CONTEXT_TOKENS, ident.EVAL_TOKENS, ident.LINE_BLOCK) == (32, 1024, 128, 64)
    q, l = ident.MODELS["qwen2_5_7b"], ident.MODELS["llama3_1_8b"]
    assert (q["model_id"], q["revision"]) == ("Qwen/Qwen2.5-7B-Instruct", "16c174980d8a1492910551634b4969e69cdc2444")
    assert q["manifest_sha256"] == "9be52dd6573759d4d2c0878dc0889fd72d4a2fbfe8065afa1180a3a1717469d1"
    assert (q["layers"], q["kv_heads"], q["head_dim"], len(q["files"])) == (28, 4, 128, 15)
    assert l["model_id"] == "LLM-Research/Meta-Llama-3.1-8B-Instruct" and len(l["revision"]) == 40
    assert (l["layers"], l["kv_heads"], l["head_dim"], len(l["files"])) == (32, 8, 128, 18)  # Exp12 log: "18 files"
    assert all(len(m["token_pool_sha256"]) == 64 for m in ident.MODELS.values())
    assert set(ident.MODELS) == set(runner.LEGACY_LOG)


def test_protocol_file_is_the_regenerated_protocol():
    p = runner.load_protocol()
    assert p["samples"] == 32 and p["context_tokens"] == 1024 and p["eval_tokens"] == 128
    assert p["hardware"]["selector"] == "H100!:1" and p["quality_threshold"].startswith("NONE")


def test_legacy_references_parse():
    ll, qw = runner.legacy_reference("llama3_1_8b"), runner.legacy_reference("qwen2_5_7b")
    assert len(ll["bf16"]) == len(qw["bf16"]) == 32
    assert abs(ll["bf16_aggregate_ppl"] - 7.5138) < 2e-3 and abs(qw["bf16_aggregate_ppl"] - 6.7837) < 2e-3  # log summary rows
    assert ll["bf16"][:3] == [3.2852, 6.4981, 10.6066]


# ------------------------------------------------------------------------------------------------ transport / evaluation
def _synthetic(model: str = "qwen2_5_7b") -> tuple[dict, dict]:
    m, ref = ident.MODELS[model], runner.legacy_reference(model)

    def rows(ppls, cls):
        out = []
        for i, p in enumerate(ppls, start=1):
            r = {"window": i, "loss_sum": math.log(p) * 128, "tokens": 128, "ppl": p, "token_nll": [math.log(p)] * 128}
            if cls:
                r.update(cache_class=cls, decode_forwards=127)
            out.append(r)
        return out

    res = {"model_key": model,
           "model": {"passed": True, "model_id": m["model_id"], "model_revision": m["revision"],
                     "manifest_sha256": m["manifest_sha256"], "files_checked": len(m["files"])},
           "hardware": {"passed": True, "gpus": [{"name": "NVIDIA H100 80GB HBM3", "memory.total": "81559"}]},
           "environment": {}, "files": {"passed": True, "sha256_lf": {n: runner.sha256_lf(HERE / n) for n in runner.SHIPPED}},
           "dataset": {"wikitext_sha256": ident.WIKITEXT_SHA256, "token_pool_sha256": m["token_pool_sha256"]},
           "geometry": {"layers": m["layers"], "kv_heads": m["kv_heads"], "head_dim": m["head_dim"]},
           "policy": {"k_bits": 3, "v_bits": 2, "group_size": 32, "residual_tokens": 4, "metadata_bits": 8,
                      "metadata_group_size": 64},
           "prefill_state_parity": {"passed": True, "layers": m["layers"], "tokens": 1024}, "logical_kv": {},
           "arms": {"bf16_batched": rows(ref["bf16"], None), "bf16": rows([p * 1.0001 for p in ref["bf16"]], "DynamicCache"),
                    "rabit": rows([p * 1.02 for p in ref["bf16"]], "CanonicalRabitCache")},
           "aggregates": {}, "timing": {},
           "legacy_unreachable": {"passed": True, "repo_files": sorted(f"benchmarks/mlsys2027/{n}" for n in runner.SHIPPED)}}
    return res, ref


def test_evaluate_valid_synthetic_result():
    res, ref = _synthetic()
    ev = runner.evaluate(res, "qwen2_5_7b", ref)
    assert ev["valid"] and len(ev["gates"]) == 10 and all(ev["gates"].values())
    st = ev["statistics"]
    assert abs(st["delta_pct"] - 100 * (1.02 / 1.0001 - 1)) < 1e-9 and st["ci_low"] <= st["delta_pct"] <= st["ci_high"]
    assert (st["resamples"], st["seed"], st["n_windows"], st["scored_tokens_per_arm"]) == (10000, 20270929, 32, 4096)


def test_evaluate_rejects_each_invalid_condition():
    base, ref = _synthetic()

    def bad(gate, mutate):
        r = copy.deepcopy(base)
        mutate(r)
        ev = runner.evaluate(r, "qwen2_5_7b", ref)
        assert not ev["valid"] and ev["gates"][gate] is False, gate

    bad("hardware", lambda r: r["hardware"].update(gpus=[{"name": "NVIDIA H200", "memory.total": "143771"}]))
    bad("model_snapshot", lambda r: r["model"].update(model_revision="master"))
    bad("model_snapshot", lambda r: r["model"].update(passed=False))
    bad("dataset", lambda r: r["dataset"].update(token_pool_sha256="0" * 64))
    bad("shipped_files", lambda r: r["files"]["sha256_lf"].update({"canonical_rabit_quality.py": "0" * 64}))
    bad("geometry_and_policy", lambda r: r["policy"].update(residual_tokens=8))
    bad("gpu_cpu_canonical_state_parity", lambda r: r["prefill_state_parity"].update(passed=False))
    bad("structure", lambda r: r["arms"]["rabit"].pop())
    bad("structure", lambda r: r["arms"]["rabit"][0].update(cache_class="DynamicCache"))
    bad("structure", lambda r: r["arms"]["rabit"][0].update(decode_forwards=1))
    bad("legacy_unreachable", lambda r: r["legacy_unreachable"]["repo_files"].append("benchmarks/quality/hotpotqa.py"))
    bad("bf16_control_reproduces_legacy", lambda r: [x.update(loss_sum=x["loss_sum"] + 128 * math.log(1.01))
                                                     for x in r["arms"]["bf16_batched"]])
    bad("bf16_stepwise_matches_batched", lambda r: [x.update(loss_sum=x["loss_sum"] + 128 * math.log(1.01))
                                                    for x in r["arms"]["bf16"]])
    # a model different from the accepted legacy one fails the control (wrong reference)
    ev = runner.evaluate(copy.deepcopy(base), "qwen2_5_7b", runner.legacy_reference("llama3_1_8b"))
    assert not ev["valid"] and not ev["gates"]["bf16_control_reproduces_legacy"]


def test_strict_json_transport():
    mod = modal_app()
    res, _ = _synthetic()
    assert mod.validate_payload(json.dumps(res))["model_key"] == "qwen2_5_7b"

    class S(str):
        pass

    for badv in (S("2.11.0"), (1, 2), {1: 2}, float("nan"), float("inf"), object()):
        try:
            mod.assert_json_native({"x": badv})
        except TypeError:
            continue
        raise AssertionError(f"accepted {badv!r}")
    for payload in (res, json.dumps({"model_key": "x"})):
        try:
            mod.validate_payload(payload)
        except (TypeError, ValueError):
            continue
        raise AssertionError("invalid payload accepted")


def test_runner_never_executes_by_default():
    for argv in ([], ["--model", "llama3_1_8b"], ["--model", "llama3_1_8b", "--dry-run", "--execute"], ["--execute"]):
        try:
            runner.main(argv)
        except SystemExit as e:
            assert e.code not in (0, None)
            continue
        raise AssertionError(argv)
    src = Path(runner.__file__).read_text(encoding="utf-8")
    assert src.count('"modal", "run"') == 1 and "return run(a.model, a.attempt)" in src


# ------------------------------------------------------------------------------------------------ proof observer
class _Thing:
    """Stand-in for a layer state. Adversarial on purpose: every instance compares equal and hashes alike."""

    def __eq__(self, other):
        return True

    def __hash__(self):
        return 0


def test_id_keyed_bookkeeping_aliases_when_an_address_is_reused():
    """Why raw id()-keying is unsafe (the superseded spy). Deterministic part: two DISTINCT objects that present the
    same address key (what CPython does after a free) share state. Opportunistic part: real CPython address reuse."""
    by_address = {}

    def spy_record(address, item):  # the superseded mechanism: state keyed by an address
        by_address.setdefault(address, []).append(item)
        return by_address[address]

    dead, new = _Thing(), _Thing()
    reused_address = 0xDEAD  # the address `dead` had, handed to `new` after `dead` was freed (simulated)
    spy_record(reused_address, "109 tokens of the dead object")
    assert dead is not new and spy_record(reused_address, "70 tokens of the new object") == [
        "109 tokens of the dead object", "70 tokens of the new object"]  # stale state associated with the new object

    # real interpreter: free an object, allocate another, and look for the same id() (usual in CPython; not required)
    for _ in range(1000):
        a = object.__new__(_Thing)
        address, real = id(a), {}
        real[address] = ["stale"]
        del a
        b = object.__new__(_Thing)
        if id(b) == address:
            assert real.get(id(b)) == ["stale"]  # a brand-new object inherits the dead object's bookkeeping
            break


def test_identity_observer_cannot_associate_stale_state_with_a_new_object():
    from proof_observer import IdentityObserver

    obs = IdentityObserver()
    first = _Thing()
    assert obs.record(first, "a") == 0 and obs.record(first, "b") == 0 and obs.items(first) == ["a", "b"]
    retained_address = id(first)
    del first  # the caller drops it; the observer still holds a direct reference, so it cannot be freed
    fresh = [_Thing() for _ in range(2000)]
    assert all(id(x) != retained_address for x in fresh)  # a live object's address is never handed out again
    for x in fresh[:50]:  # equal-comparing, equal-hashing, never-observed objects get NO state
        try:
            obs.items(x)
        except KeyError:
            continue
        raise AssertionError("state returned for an object that was never observed")
    # new objects start empty and get their own monotonic sequence IDs; earlier state is untouched
    assert [obs.record(x, i) for i, x in enumerate(fresh[:3])] == [1, 2, 3]
    assert [obs.items(x) for x in fresh[:3]] == [[0], [1], [2]] and len(obs) == 4
    assert [obs.sequence_id(x) for x in fresh[:3]] == [1, 2, 3]
    assert obs._entries[0][1] == ["a", "b"] and all(e[0] is not x for e in obs._entries[:1] for x in fresh)


def test_proof_code_never_keys_on_raw_addresses():
    for name in ("canonical_ppl_offline_proofs.py", "proof_observer.py"):
        calls = [n for n in ast.walk(tree(HERE / name)) if isinstance(n, ast.Call) and getattr(n.func, "id", "") in
                 ("id", "hash")]
        assert not calls, (name, [n.lineno for n in calls])
    assert imports(HERE / "proof_observer.py") - STDLIB == set()
    src = (HERE / "canonical_ppl_offline_proofs.py").read_text(encoding="utf-8")
    assert "raw = IdentityObserver()" in src and "raw.items(self)" in src and "raw.record(self, " in src
    # scoring code never sees the observer
    assert all("proof_observer" not in imports(p) for p in (CORE, MODAL, IDENT, CRQ))


if __name__ == "__main__":
    tests =[(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
