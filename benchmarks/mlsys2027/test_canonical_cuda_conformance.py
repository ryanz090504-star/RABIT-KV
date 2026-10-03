"""
Static tests of the canonical-quality-v2 GPU semantic-conformance diagnostic (stdlib + modal client only; no torch, no
GPU, no model, no network). Run by run_canonical_cuda_conformance.py's preflight.
"""

from __future__ import annotations

import ast
import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import canonical_ppl_identity as ident  # noqa: E402
import run_canonical_cuda_conformance as runner  # noqa: E402

MODAL, CONF = HERE / "canonical_cuda_conformance_modal.py", HERE / "canonical_cuda_conformance.py"
SCORING = {"score", "score_stepwise", "score_bf16_batched", "cross_entropy", "generate", "logits", "aggregate", "_row",
           "_token_nll", "log_softmax", "softmax", "argmax", "ppl", "loss", "loss_sum"}
TOLERANCE = {"allclose", "isclose", "atol", "rtol", "tolerance", "threshold"}


def tree(p: Path) -> ast.Module:
    return ast.parse(p.read_text(encoding="utf-8"))


def identifiers(p: Path) -> set:
    out = set()
    for n in ast.walk(tree(p)):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out.add(n.name)
    return out


def fn(name: str) -> ast.FunctionDef:
    return next(n for n in tree(MODAL).body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_no_scoring_generation_or_tolerance_anywhere():
    for p in (MODAL, CONF):
        assert not (identifiers(p) & SCORING), (p.name, identifiers(p) & SCORING)
        assert not (identifiers(p) & TOLERANCE), (p.name, identifiers(p) & TOLERANCE)
    core_attrs = {n.attr for n in ast.walk(tree(MODAL)) if isinstance(n, ast.Attribute)
                  and isinstance(n.value, ast.Name) and n.value.id == "core"}
    assert core_attrs == {"build_token_pool", "wikitext_lines", "pool_sha256", "split_windows",
                          "canonical_cache_from_prefill", "prefill_state_parity"}
    src = ast.unparse(fn("conformance"))
    assert src.count("model(input_ids=context_ids, use_cache=True)") == 1 and src.count("model(") == 1


def test_frozen_implementations_are_only_observed():
    files = next(ast.literal_eval(n.value) for n in tree(MODAL).body if isinstance(n, ast.Assign)
                 and getattr(n.targets[0], "id", "") == "FILES")
    assert files == runner.SHIPPED and all((ROOT / f).is_file() for f in files)
    for k, (commit, paths) in runner.FROZEN.items():
        assert runner.git("diff", "--name-only", commit, "--", *paths) == "", k
    # the oracle path never calls the canonical module; the canonical path never calls the oracle
    conf = tree(CONF)
    body = {n.name: ast.unparse(n) for n in conf.body if isinstance(n, ast.FunctionDef)}
    assert "cq." not in body["oracle_fields"] and "canonical_state" not in body["oracle_fields"]
    assert "o[" not in body["canonical_fields"] and "_ref" not in body["canonical_fields"]
    # the comparison module assigns nothing into the canonical module
    assert not [n for n in ast.walk(conf) if isinstance(n, (ast.Assign, ast.AugAssign)) for t in (
        n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Attribute)
        and isinstance(t.value, ast.Name) and t.value.id in ("cq", "crq")]


def test_strict_h100_selector_and_guard_first():
    f = fn("conformance")
    gpu = [kw.value.value for d in f.decorator_list if isinstance(d, ast.Call) for kw in d.keywords if kw.arg == "gpu"]
    assert gpu == ["H100!:1"] and ast.unparse(f.body[0]) == "_gpus = _gpu_query()"
    guard = f.body[3]
    assert isinstance(guard, ast.If) and ast.unparse(guard.test) == "not _hw_ok" and isinstance(guard.body[-1], ast.Raise)
    src = ast.unparse(f)
    order = ["if not _hw_ok", "if not files_ok", "suite.load_oracle(", "snapshot_download(", "ident.verify_dir(",
             "ident.WIKITEXT_SHA256", "m['token_pool_sha256']", "AutoModelForCausalLM.from_pretrained(",
             "conf.layer_report(", "core.prefill_state_parity(", "conf.synthetic_report("]
    pos = [src.index(s) for s in order]
    assert pos == sorted(pos), list(zip(order, pos))


def _layer(equal: bool = True, heads: int = 8) -> dict:
    return {"layer": 0, "raw": {"dtype": "torch.bfloat16", "shape": [1024, heads, 128], "device": "cuda:0",
                                "host_copy_bitwise_identical": True},
            "trace_reproduces_frozen_code_on_both_devices": True, "accepted_t1_comparison_on_cuda": [],
            "residual_equals_raw": {"cuda_canonical": True, "cuda_oracle": True},
            "summary": {"cuda_canonical_equals_cuda_oracle": equal, "canonical_cache_equals_cuda_oracle": True,
                        "cpu_canonical_equals_cpu_oracle": True, "cpu_canonical_equals_cuda_canonical": False,
                        "cpu_oracle_equals_cuda_oracle": False}}


def _synthetic_result(model: str = "llama3_1_8b") -> dict:
    m = ident.MODELS[model]
    geo = lambda h: {"kv_heads": h, "head_dim": 128, "cases": 60, "aging_cases": 28, "field_failures": [],  # noqa: E731
                     "t1_failures": [], "aging_failures": [], "passed": True}
    return {"model_key": model, "geometry": {k: m[k] for k in ("layers", "kv_heads", "head_dim")},
            "hardware": {"passed": True, "gpus": [{"name": "NVIDIA H100 80GB HBM3", "memory.total": "81559"}]},
            "files": {"passed": True, "sha256_lf": {f: runner.sha256_lf(ROOT / f) for f in runner.SHIPPED}},
            "oracle": {"equals_accepted_parity_oracle": True, "oracle_namespace_references_canonical_module": False},
            "model": {"passed": True, "model_id": m["model_id"], "model_revision": m["revision"],
                      "manifest_sha256": m["manifest_sha256"], "files_checked": len(m["files"])},
            "dataset": {"wikitext_sha256": ident.WIKITEXT_SHA256, "token_pool_sha256": m["token_pool_sha256"]},
            "window": {"context_tokens": 1024, "equals_first_1024_pool_tokens": True, "contains_bos": False},
            "layers": [dict(_layer(heads=m["kv_heads"]), layer=i) for i in range(m["layers"])],
            "attempt1_gate_reevaluated": {"passed": False, "mismatched_layers": list(range(m["layers"]))},
            "synthetic": {"geometries": {"llama3_1_8b": geo(8), "qwen2_5_7b": geo(4)}},
            "no_scoring": {"continuation_tokens_scored": 0, "logits_read": False, "generation": False}}


def test_classification_is_a_pure_function_of_the_same_device_oracle_comparison():
    base = _synthetic_result()
    ev = runner.evaluate(base)
    assert ev["valid"] and ev["classification"] == "A"  # CPU-vs-CUDA differences do NOT affect the classification
    assert ev["descriptive"]["layers_cpu_canonical_equals_cuda_canonical"] == 0
    q = runner.evaluate(_synthetic_result("qwen2_5_7b"), "qwen2_5_7b")
    assert q["valid"] and q["classification"] == "A" and q["primary"]["layers"] == 28
    # a result of one model evaluated as the other is INVALID (model key, identity, geometry)
    assert runner.evaluate(_synthetic_result("qwen2_5_7b"), "llama3_1_8b")["classification"] == "C"
    assert runner.evaluate(_synthetic_result("llama3_1_8b"), "qwen2_5_7b")["classification"] == "C"

    def cls(mutate):
        r = copy.deepcopy(base)
        mutate(r)
        return runner.evaluate(r)["classification"]

    assert cls(lambda r: r["layers"][5]["summary"].update(cuda_canonical_equals_cuda_oracle=False)) == "B"
    assert cls(lambda r: r["layers"][5]["summary"].update(canonical_cache_equals_cuda_oracle=False)) == "B"
    assert cls(lambda r: r["layers"][5].update(accepted_t1_comparison_on_cuda=["V metadata"])) == "B"
    assert cls(lambda r: r["synthetic"]["geometries"]["qwen2_5_7b"].update(passed=False, t1_failures=[{}])) == "B"
    assert cls(lambda r: r["layers"].pop()) == "C"
    assert cls(lambda r: r["hardware"].update(gpus=[{"name": "NVIDIA H200", "memory.total": "143771"}])) == "C"
    assert cls(lambda r: r["model"].update(model_revision="master")) == "C"
    assert cls(lambda r: r["oracle"].update(equals_accepted_parity_oracle=False)) == "C"
    assert cls(lambda r: r["layers"][0].update(trace_reproduces_frozen_code_on_both_devices=False)) == "C"
    assert cls(lambda r: r["no_scoring"].update(continuation_tokens_scored=128)) == "C"
    assert cls(lambda r: r["synthetic"]["geometries"]["llama3_1_8b"].update(cases=59)) == "C"


def test_runner_never_executes_by_default():
    for argv in ([], ["--model", "qwen2_5_7b"], ["--model", "qwen2_5_7b", "--dry-run", "--execute"], ["--execute"]):
        try:
            runner.main(argv)
        except SystemExit as e:
            assert e.code not in (0, None)
            continue
        raise AssertionError(argv)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, test in tests:
        try:
            test()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
