"""
Static / offline tests of the canonical-quality-v2 LONG-CONTEXT suite harness (stdlib + modal client only; no torch, no
GPU, no model, no network). Run by run_canonical_longctx.py's preflight. The torch-dependent proofs (prompt / selection
equality with the legacy source for all 357 units, generation-loop equality, canonical path) are in
canonical_longctx_offline_proofs.py and recorded in results/mlsys2027/canonical_quality_v2/long_context/
offline_proof_record.json.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import canonical_longctx_pins as pins  # noqa: E402
import canonical_longctx_tasks as tasks  # noqa: E402
import canonical_ppl_identity as ident  # noqa: E402
import exp14_hardware_binding as hb  # noqa: E402
import run_canonical_longctx as runner  # noqa: E402

CORE, MODAL, TASKS, PINS, CRQ = (HERE / n for n in ("canonical_longctx_core.py", "canonical_longctx_modal.py",
                                                    "canonical_longctx_tasks.py", "canonical_longctx_pins.py",
                                                    "canonical_rabit_quality.py"))
STDLIB = set(sys.stdlib_module_names)
LEGACY_NAMES = {"q_seq_affine", "q_group_affine", "q_group_sym", "q_tensor", "q_with_residual", "dequantize_state",
                "quantize_then_dequantize_cache", "encode_metadata", "decode_metadata", "config_for_method",
                "tuple_to_dynamic_cache", "stored_state_logical_bytes", "kvquant_k3", "quantize_rabit2_kv_ref",
                "dequantize_rabit2_kv_ref", "run_niah", "run_longbench", "load_oracle"}
QUANT_ARITHMETIC = {"round", "clamp", "amin", "amax", "uint8", "floor", "ceil", "quantize"}
RESULT_NAMES = {"gen", "row", "rows", "results", "prediction", "score", "scores", "answer", "correct", "generated_ids"}


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


def identifiers(node) -> set:
    out = set()
    for n in ast.walk(node if isinstance(node, ast.AST) else tree(node)):
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
    spec = importlib.util.spec_from_file_location("canonical_longctx_modal_under_test", MODAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_suite_def() -> ast.FunctionDef:
    return next(n for n in tree(MODAL).body if isinstance(n, ast.FunctionDef) and n.name == "run_suite")


def task_loop(fn: ast.FunctionDef) -> ast.For:
    """The loop that runs the tasks (the For over TASK_ORDER that calls run_arm)."""
    return next(s for s in fn.body if isinstance(s, ast.For) and ast.unparse(s.iter) == "TASK_ORDER"
                and "run_arm(" in ast.unparse(s))


# ------------------------------------------------------------------------------------------------ sole implementation
def test_shipped_files_and_sole_rabit_implementation():
    mod = modal_app()
    assert mod.FILES == runner.SHIPPED == ["canonical_rabit_quality.py", "canonical_longctx_core.py",
                                           "canonical_longctx_tasks.py", "canonical_longctx_pins.py",
                                           "canonical_ppl_identity.py", "exp14_model_snapshot.py"]
    src = MODAL.read_text(encoding="utf-8")
    assert src.count("add_local_file") == 1 and "add_local_dir" not in src and "add_local_python_source" not in src
    sha = hashlib.sha256(CRQ.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert sha == mod.CANONICAL_RABIT_QUALITY_SHA256_LF == "195cb4896c4708cc6ecc450930733962ea3e08e21440fc9406dedbe1bb4c11d5"
    assert runner.git("diff", "--name-only", "c36069781b259e2e9b4d8adf60b7c58ea5df5cac", "--",
                      "benchmarks/mlsys2027/canonical_rabit_quality.py") == ""
    assert attrs_of(CORE, "crq") == {"make_canonical_cache_class"}  # the rabit cache comes from exactly one place
    assert attrs_of(MODAL, "crq") == {"POLICY", "__file__"}
    assert CORE.read_text(encoding="utf-8").count("make_canonical_cache_class") == 1


def test_legacy_quantizer_and_oracle_unreachable():
    mod = modal_app()
    assert {"niah", "passage_retrieval", "hotpotqa", "qasper", "continuation_ppl", "kvquant_k3", "vllm",
            "canonical_quality_parity_tests", "canonical_cuda_conformance"} <= set(mod.FORBIDDEN_MODULES)
    third_party = {"torch", "transformers", "modelscope", "requests", "modal", "datasets", "huggingface_hub"}
    assert imports(CORE) - STDLIB == {"torch", "canonical_rabit_quality"}
    assert imports(TASKS) - STDLIB == set() and imports(PINS) - STDLIB == set()
    assert imports(CRQ) - STDLIB == {"torch", "transformers"}
    assert imports(MODAL) - STDLIB - third_party == {"canonical_longctx_pins", "canonical_longctx_tasks",
                                                     "canonical_ppl_identity", "canonical_longctx_core",
                                                     "canonical_rabit_quality"}
    top = {a.name.split(".")[0] for n in tree(MODAL).body if isinstance(n, ast.Import) for a in n.names}
    assert top - STDLIB == {"modal"}  # no sibling import at module level
    for p in (CORE, MODAL, TASKS, PINS, CRQ, HERE / "canonical_ppl_identity.py", HERE / "exp14_model_snapshot.py"):
        names = identifiers(p)
        assert not (names & LEGACY_NAMES), (p.name, names & LEGACY_NAMES)
        assert not [n for n in names if n.endswith("_ref")], p.name
    for p in (CORE, MODAL, TASKS):
        code = "\n".join(ast.unparse(n) for n in tree(p).body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)))
        assert "vllm-kvquant" not in code and "benchmarks/quality" not in code, p.name
    assert not (identifiers(CORE) & QUANT_ARITHMETIC)
    # `round` appears only for the needle position (legacy: round(filler x depth)) and the 2-decimal depth key
    assert identifiers(TASKS) & QUANT_ARITHMETIC == {"round"} == identifiers(MODAL) & QUANT_ARITHMETIC
    rounds = [ast.unparse(n) for p in (TASKS, MODAL) for n in ast.walk(tree(p)) if isinstance(n, ast.Call)
              and getattr(n.func, "id", "") == "round"]
    assert rounds == ["round(filler_needed * float(needle_depth))", "round(float(d), 2)"]


# ------------------------------------------------------------------------------------------------ hardware / order
def test_strict_h100_selector_guard_first_and_timeout():
    fn = run_suite_def()
    kw = {k.arg: ast.unparse(k.value) for d in fn.decorator_list if isinstance(d, ast.Call) for k in d.keywords}
    assert kw["gpu"] == "'H100!:1'" and kw["timeout"] == "SUITE_TIMEOUT_S"
    assert modal_app().SUITE_TIMEOUT_S == runner.SUITE_TIMEOUT_S == 12 * 3600
    assert ast.unparse(fn.body[0]) == "_gpus = _gpu_query()"
    exp14 = next(n for n in hb.guard_statements() if isinstance(n, ast.Assign) and n.targets[0].id == "_hw_ok")
    assert ast.dump(fn.body[1]) == ast.dump(exp14)  # the accepted Exp14 predicate, verbatim
    guard = fn.body[3]
    assert isinstance(guard, ast.If) and ast.unparse(guard.test) == "not _hw_ok" and isinstance(guard.body[-1], ast.Raise)
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for s in fn.body[:4] for n in ast.walk(s))
    h100 = {"name": "NVIDIA H100 80GB HBM3", "memory.total": "81559"}
    assert ident.hardware_ok([h100]) and not ident.hardware_ok([h100, h100])
    assert not ident.hardware_ok([{"name": "NVIDIA H200", "memory.total": "143771"}])


def test_container_order_identity_before_model_before_generation():
    src = ast.unparse(run_suite_def())
    order = ["if not _hw_ok", "if not files_ok", "if not runtime_ok", "snapshot_download(", "ident.verify_dir(",
             "hf_hub_download(", "tasks.WIKITEXT_SHA256", "pins.PROMPT_SETS[task]", "AutoModelForCausalLM.from_pretrained(",
             "for position, unit in enumerate(units[task])"]
    pos = [src.index(s) for s in order]
    assert pos == sorted(pos), list(zip(order, pos))
    mod = modal_app()
    assert mod.VALIDATED_RUNTIME == runner.VALIDATED_RUNTIME and mod.TASK_ORDER == tuple(runner.TASK_ORDER) == (
        "niah", "passage_retrieval", "hotpotqa")
    assert "runtime_ok = runtime == VALIDATED_RUNTIME" in src


# ------------------------------------------------------------------------------------------------ no exposure
def test_no_treatment_result_is_exposed_between_tasks():
    """While the tasks run, the container's only output is _progress(task, position, total, arm)."""
    fn = run_suite_def()
    start = fn.body.index(task_loop(fn))
    region = [n for s in fn.body[start:] for n in ast.walk(s)]
    helper_defs = [s for s in fn.body if isinstance(s, ast.FunctionDef)]  # run_arm / persist_row / as_tensor
    region += [n for h in helper_defs for n in ast.walk(h)]
    out_calls = [n for n in region if isinstance(n, ast.Call) and getattr(n.func, "id", "") in ("print", "_emit", "_progress")]
    in_loop = [n for n in ast.walk(task_loop(fn)) if isinstance(n, ast.Call)
               and getattr(n.func, "id", "") in ("print", "_emit", "_progress")]
    assert [ast.unparse(n) for n in in_loop] == ["_progress(task, position + 1, len(units[task]), arm)"]
    assert not [n for h in helper_defs for n in ast.walk(h) if isinstance(n, ast.Call)
                and getattr(n.func, "id", "") in ("print", "_emit", "_progress")]
    after = [ast.unparse(n) for n in out_calls if n not in in_loop]
    assert len(after) == 1 and after[0].startswith("_emit('CANONICAL_LONGCTX_REMOTE'") and "len(results[t])" in after[0]
    assert not (identifiers(ast.parse(after[0])) & (RESULT_NAMES - {"results"}))
    progress = next(n for n in tree(MODAL).body if isinstance(n, ast.FunctionDef) and n.name == "_progress")
    assert [a.arg for a in progress.args.args] == ["task", "position", "total", "arm"]
    # the crash-recovery rows go to a file nothing reads during the run
    persist = next(h for h in helper_defs if h.name == "persist_row")
    assert "open(persist['path'], 'a'" in ast.unparse(persist) and "'r'" not in ast.unparse(persist)
    assert MODAL.read_text(encoding="utf-8").count("rows.jsonl") == 1
    # the runner launches ONE container for the whole suite and prints results only after evaluate()
    rsrc = Path(runner.__file__).read_text(encoding="utf-8")
    assert rsrc.count('"modal", "run"') == 1
    run_fn = next(n for n in tree(Path(runner.__file__)).body if isinstance(n, ast.FunctionDef) and n.name == "run")
    prints = [n for n in ast.walk(run_fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "print"]
    assert len(prints) == 1 and run_fn.body.index(next(s for s in run_fn.body if prints[0] in list(ast.walk(s)))) > \
        run_fn.body.index(next(s for s in run_fn.body if "evaluate(res)" in ast.unparse(s)))


def test_no_result_dependent_control_flow_in_the_container():
    """After the identity checks every unit of every task runs unconditionally; nothing branches on an output."""
    fn = run_suite_def()
    loop = task_loop(fn)
    branches = [n for n in ast.walk(loop) if isinstance(n, (ast.If, ast.While, ast.IfExp, ast.Break, ast.Continue,
                                                            ast.Try, ast.Raise, ast.Return, ast.Assert))]
    assert [ast.unparse(b.test) for b in branches] == ["task == 'niah'"]  # the legacy warm-up choice only
    helper = {h.name: h for h in fn.body if isinstance(h, ast.FunctionDef)}
    arm_tests = [ast.unparse(n.test) for n in ast.walk(helper["run_arm"]) if isinstance(n, (ast.If, ast.While, ast.IfExp))]
    assert arm_tests == ["task == 'niah'"]  # which frozen scorer to apply -- by task, never by value
    assert not [n for n in ast.walk(helper["run_arm"]) if isinstance(n, (ast.Raise, ast.Break, ast.Continue, ast.Try))]
    p_tests = [ast.unparse(n.test) for n in ast.walk(helper["persist_row"]) if isinstance(n, (ast.If, ast.While, ast.IfExp))]
    assert p_tests == ["persist['rows_written'] % 10 == 0"]
    # the statements after the task loop branch only on the forbidden-module list
    start = fn.body.index(loop)
    later = [ast.unparse(n.test) for s in fn.body[start + 1:] for n in ast.walk(s) if isinstance(n, (ast.If, ast.While, ast.IfExp))]
    assert later == ["loaded"]
    # the generation core: the ONLY value-dependent branch is the EOS stop of the legacy loop
    gen = next(n for n in tree(CORE).body if isinstance(n, ast.FunctionDef) and n.name == "generate")
    tests = [ast.unparse(n.test) for n in ast.walk(gen) if isinstance(n, (ast.If, ast.While, ast.IfExp))]
    assert sorted(tests) == sorted([
        "arm not in ARMS", "arm == 'bf16'", "eos_id is not None and token_id == eos_id", "cache is not first",
        "seen != int(prefix_ids.shape[1]) + forwards", "arm == 'rabit' and {s.n for s in cache.states} != {seen}"])
    assert not (identifiers(CORE) & {"score", "cross_entropy", "softmax", "sample", "multinomial", "temperature", "top_k", "top_p"})
    assert ast.unparse(gen).count("torch.argmax(") == 2  # greedy only


# ------------------------------------------------------------------------------------------------ scorers
def test_hotpotqa_legacy_and_official_scorers_and_their_known_difference():
    assert tasks.qa_f1_score_legacy("the Eiffel Tower", "Eiffel Tower") == 0.8  # Exp12: the article is NOT removed
    assert tasks.qa_f1_score_official("the Eiffel Tower", "Eiffel Tower") == 1.0  # official: the article is removed
    assert tasks.normalize_answer_legacy("The Eiffel Tower, a tower.") == "the eiffel tower a tower"
    assert tasks.normalize_answer_official("The Eiffel Tower, a tower.") == "eiffel tower tower"
    assert tasks.qa_f1_score_legacy("", "") == 1.0 and tasks.qa_f1_score_official("", "") == 0.0  # empty prediction
    assert tasks.qa_f1_score_legacy("", "x") == 0.0 == tasks.qa_f1_score_official("", "x")
    s = tasks.score("hotpotqa", "the Eiffel Tower", ["Paris", "Eiffel Tower"])
    assert s == {"score": 0.8, "score_official": 1.0}  # both from the SAME prediction string; legacy is primary
    # the legacy pattern really is the Exp12 one (doubled backslashes inside a raw string)
    pat = [n.args[0].value for n in ast.walk(tree(TASKS)) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "sub"
           and isinstance(n.args[0], ast.Constant) and "an|the" in n.args[0].value]
    leg = [n.args[0].value for n in ast.walk(tree(ROOT / "benchmarks/quality/hotpotqa.py")) if isinstance(n, ast.Call)
           and getattr(n.func, "attr", "") == "sub" and isinstance(n.args[0], ast.Constant) and "an|the" in n.args[0].value]
    assert sorted(pat) == sorted([leg[0], "\\b(a|an|the)\\b"]) and leg[0] == "\\\\b(a|an|the)\\\\b"
    p = runner.load_protocol()
    assert p["scorer_source_sha256"] == runner.function_source_sha256(TASKS, list(p["scorer_source_sha256"]))
    assert "PRIMARY" not in p["tasks"]["hotpotqa"]["metric_secondary"] and "secondary only" in p["tasks"]["hotpotqa"]["metric_secondary"]


def test_frozen_scorers_equal_the_legacy_source_on_every_exp12_prediction():
    e = runner.exp12_rows("niah")
    for m in ("bf16", "rabit2"):
        assert len(e["rows"][m]) == 57 and all(tasks.niah_score(r["answer"])["correct"] == (r["value"] == 1.0) for r in e["rows"][m])
    for task, names, fn in (("passage_retrieval", ["official_retrieval_score", "normalize_answers"], "official_retrieval_score"),
                            ("hotpotqa", ["normalize_answer", "qa_f1_score", "normalize_answers"], "qa_f1_score")):
        leg, e = runner.legacy_functions(task, names), runner.exp12_rows(task)
        assert len(e["identity"]) == runner.COUNTS[task]
        for m in ("bf16", "rabit2"):
            for r, idn in zip(e["rows"][m], e["identity"]):
                mine = tasks.score(task, r["answer"], idn["answers"])["score"]
                assert mine == max(leg[fn](r["answer"], a) for a in idn["answers"]), (task, r["key"])
                assert float(f"{mine:.3f}") == r["value"], (task, r["key"])  # reproduces the accepted Exp12 log
        assert leg["normalize_answers"]('["a", "b"]') == tasks.normalize_answers('["a", "b"]') == ["a", "b"]


# ------------------------------------------------------------------------------------------------ identity / protocol
def test_protocol_pins_and_dataset_provenance():
    p = runner.load_protocol()
    assert p["tasks"]["niah"]["cases"] == 57 and len(tasks.niah_cases()) == 57 and tasks.NIAH_MAX_NEW_TOKENS == 16
    assert tasks.niah_cases()[0] == (4096, 0.05) and tasks.niah_cases()[-1] == (16384, 0.95)
    pr, hq = tasks.DATASETS["passage_retrieval"], tasks.DATASETS["hotpotqa"]
    assert (pr["revision"], pr["bytes"], pr["sha256"], pr["samples"]) == (
        "915b0c6ec0b6dfae1cd44224b7d8995317837f27", 7029836,
        "452f03dbb0e394de2b26d6e016bf1e715a3ebac9dc93844fde73c0e8e74cfd68", 200)
    assert (hq["revision"], hq["bytes"], hq["sha256"], hq["samples"], hq["length_bucket"]) == (
        "92b6c5fbfb0c97b91e92d9ef79802f95ce74b05e", 7196922,
        "44ca413b8c2435a771cd7987b1d9298ab8fce512684533994b5a55c67b539dc3", 100, "8k+")
    assert "NEWLY RECOVERED" in p["dataset_hash_provenance"] and "commits only" in p["dataset_hash_provenance"]
    assert tasks.MAX_INPUT_TOKENS == 16384 and tasks.LONGBENCH_MAX_NEW_TOKENS == 32
    assert tasks.truncate_ids(list(range(20000)), 16384) == list(range(8192)) + list(range(20000 - 8192, 20000))
    assert tasks.truncate_ids([1, 2, 3], 16384) == [1, 2, 3]
    assert tasks.select_indices("hotpotqa", [100] * 200 + [9000] * 100) == list(range(200, 300))
    assert tasks.select_indices("passage_retrieval", [1] * 200) == list(range(200))
    assert {k: v["units"] for k, v in pins.PROMPT_SETS.items()} == runner.COUNTS == {"niah": 57, "passage_retrieval": 200, "hotpotqa": 100}
    assert all(len(v["prompt_set_sha256"]) == 64 for v in pins.PROMPT_SETS.values())
    m = ident.MODELS["llama3_1_8b"]
    assert (p["model"]["revision"], p["model"]["manifest_sha256"]) == (
        "359efdbb8af05b788a4ad4185215c6b8caa9052c", "85d9cffee6980348ad1c334d71f8731f6442553535848542457b68d85d70ce89") == (
        m["revision"], m["manifest_sha256"])
    assert p["policy"].startswith("K3 / V2 / G32 / R4 / META8g64") and p["quality_threshold"].startswith("NONE")
    assert p["rabit_quality_gate"].startswith("NONE") and p["registered_suite"]["order"] == runner.TASK_ORDER
    assert p["statistics"]["seed"] == {"passage_retrieval": 20270931, "hotpotqa": 20270932} and p["statistics"]["resamples"] == 10000
    for t in runner.TASK_ORDER:  # the legacy scripts are the scripts of the accepted Exp12 run
        assert hashlib.sha256((ROOT / runner.LEGACY_SCRIPT[t]).read_bytes()).hexdigest() == runner.LEGACY_SCRIPT_SHA256_RAW[t] \
            or runner.git("diff", "--name-only", "4f767ab03d83e043b2871dd0cd4cf2f8dc862e6b", "--", runner.LEGACY_SCRIPT[t]) == ""
    manifest = json.loads((ROOT / "results/mlsys2027/variance/manifest.json").read_text(encoding="utf-8"))
    assert {t: manifest["provenance"]["canonical_script_sha256"][t] for t in runner.TASK_ORDER} == runner.LEGACY_SCRIPT_SHA256_RAW


def test_cuda_conformance_prerequisites_are_pinned():
    assert set(runner.CONFORMANCE) == {"llama_1k_ppl_window", "llama_16k_niah"}
    for name in runner.CONFORMANCE:
        ev = runner.conformance_evidence(name)
        assert all(v is True for v in ev.values()) and len(ev) == 10, (name, ev)
    saved = copy.deepcopy(runner.CONFORMANCE)
    try:
        runner.CONFORMANCE["llama_16k_niah"]["record_sha256_lf"] = "0" * 64
        assert runner.conformance_evidence("llama_16k_niah")["record_pinned"] is False
        runner.CONFORMANCE["llama_16k_niah"] = dict(saved["llama_1k_ppl_window"], prefill_tokens=16383)
        assert runner.conformance_evidence("llama_16k_niah")["prefill_tokens"] is False
    finally:
        runner.CONFORMANCE.clear()
        runner.CONFORMANCE.update(saved)


# ------------------------------------------------------------------------------------------------ gates / evaluation
def test_bf16_control_gates_are_the_accepted_exp12_rules():
    proto = json.loads(runner.read(runner.EXP12_PROTOCOL))["validity"]
    assert proto["continuation_ppl_niah_passage_retrieval"]["tolerances"]["accuracy_pct"] == {"absolute_points": 1.0}
    assert proto["continuation_ppl_niah_passage_retrieval"]["targets"]["niah"]["bf16"]["accuracy_pct"] == 100.0
    assert proto["continuation_ppl_niah_passage_retrieval"]["targets"]["passage_retrieval"]["bf16"]["accuracy_pct"] == 100.0
    thr = json.loads(runner.read(runner.QA_AMENDMENT))["thresholds"]["hotpotqa"]["bf16"]
    assert (thr["historical_max_score_mismatch_count"], thr["historical_max_l1_score_distance"]) == (1, 0.065)
    assert thr == proto["hotpotqa_qasper"]["thresholds"]["hotpotqa"]["bf16"] | {k: thr[k] for k in thr if k not in (
        "historical_max_score_mismatch_count", "historical_max_l1_score_distance")}
    # applied to the accepted Exp12 BF16 rows the gates reproduce Exp12's own recorded outcome
    for task, units in (("niah", 15), ("passage_retrieval", 10)):
        e = runner.exp12_rows(task)
        g = runner.bf16_subset_gate(task, [x["key"] for x in e["rows"]["bf16"]], [x["value"] for x in e["rows"]["bf16"]])
        assert g["passed"] and g["canonical_subset_units"] == units and g["observed_accuracy_pct"] == 100.0
        bad = [0.0 if i < 2 else v for i, v in enumerate([x["value"] for x in e["rows"]["bf16"]])]
        assert task == "niah" or not runner.bf16_subset_gate(task, [x["key"] for x in e["rows"]["bf16"]], bad)["passed"]
    e = runner.exp12_rows("hotpotqa")
    g = runner.bf16_hotpot_gate(e["identity"], [x["value"] for x in e["rows"]["bf16"]])
    assert g["passed"] and g["n_canonical_subset"] == 20 and (g["score_mismatch_count"], g["score_mismatch_indices"],
                                                             g["l1_score_distance"]) == (1, [3], 0.035)  # = Exp12 record
    scores = [x["value"] for x in e["rows"]["bf16"]]
    two = list(scores)
    two[0], two[1] = two[0] + 0.01, two[1] + 0.01
    assert not runner.bf16_hotpot_gate(e["identity"], two)["passed"]  # more than one mismatch
    far = list(scores)
    far[3] = e["rows"]["bf16"][3]["value"] + 0.2
    assert not runner.bf16_hotpot_gate(e["identity"], far)["passed"]  # L1 beyond the frozen maximum
    wrong = copy.deepcopy(e["identity"])
    wrong[5]["used_tokens"] += 1
    assert not runner.bf16_hotpot_gate(wrong, scores)["passed"]  # identity of the canonical subset
    for fn in ("bf16_subset_gate", "bf16_hotpot_gate"):
        f = next(n for n in tree(Path(runner.__file__)).body if isinstance(n, ast.FunctionDef) and n.name == fn)
        assert "rabit" not in {a.arg for a in f.args.args} and '"rabit2"' not in ast.unparse(f) and "'rabit'" not in ast.unparse(f)


def _synthetic(rabit_quality: str = "same") -> dict:
    """A structurally valid result built from the accepted Exp12 BF16 rows; the rabit arm is set by `rabit_quality`."""
    m = ident.MODELS["llama3_1_8b"]

    def arm(task, prediction, answers, cls, n_ids):
        a = {"prediction": prediction, "generated_ids": list(range(n_ids)), "cache_class": cls, "decode_forwards": n_ids + 1,
             "stopped_on_eos": n_ids < runner.MAX_NEW[task], "seconds": 1.0}
        a.update(tasks.niah_score(prediction) if task == "niah" else tasks.score(task, prediction, answers))
        return a

    t_ = {}
    for task in runner.TASK_ORDER:
        e, rows = runner.exp12_rows(task), []
        for i, (idn, b) in enumerate(zip(e["identity"], e["rows"]["bf16"])):
            answers = [tasks.NIAH_SECRET] if task == "niah" else idn["answers"]
            tokens = idn["key"][0] if task == "niah" else idn["used_tokens"]
            pred_r = {"same": b["answer"], "garbage": "zzz 0 nothing", "empty": ""}[rabit_quality]
            row = {"index": i, "key": idn["key"], "prompt_tokens": tokens,
                   "original_tokens": tokens if task == "niah" else idn["original_tokens"],
                   "prompt_ids_sha256": hashlib.sha256(json.dumps([task, i]).encode()).hexdigest(), "answers": answers,
                   "arms": {"bf16": arm(task, b["answer"], answers, "DynamicCache", 3),
                            "rabit": arm(task, pred_r, answers, "CanonicalRabitCache", 3)}}
            for a in row["arms"].values():
                a.update(prefix_tokens=tokens - 1, final_cache_tokens=tokens + 3)
            rows.append(row)
        t_[task] = rows
    return {"model": {"passed": True, "model_id": m["model_id"], "model_revision": m["revision"],
                      "manifest_sha256": m["manifest_sha256"], "files_checked": len(m["files"])},
            "hardware": {"passed": True, "gpus": [{"name": "NVIDIA H100 80GB HBM3", "memory.total": "81559"}]},
            "environment": {}, "runtime_environment": {"runtime": dict(runner.VALIDATED_RUNTIME),
                                                       "validated": dict(runner.VALIDATED_RUNTIME), "passed": True},
            "files": {"passed": True, "sha256_lf": {n: runner.sha256_lf(HERE / n) for n in runner.SHIPPED}},
            "datasets": {"wikitext_sha256": tasks.WIKITEXT_SHA256,
                         **{k: {"sha256": d["sha256"], "bytes": d["bytes"], "rows": d["rows"]} for k, d in tasks.DATASETS.items()}},
            "prompt_sets": {k: {"units": runner.COUNTS[k], "passed": True, "prompt_set_sha256": runner.prompt_set_sha256(t_[k])}
                            for k in runner.TASK_ORDER},
            "geometry": {"layers": 32, "kv_heads": 8, "head_dim": 128},
            "policy": {"k_bits": 3, "v_bits": 2, "group_size": 32, "residual_tokens": 4, "metadata_bits": 8,
                       "metadata_group_size": 64},
            "task_order": list(runner.TASK_ORDER), "tasks": t_, "artifact_persistence": {}, "timing": {},
            "legacy_unreachable": {"passed": True, "repo_files": sorted(f"benchmarks/mlsys2027/{n}" for n in runner.SHIPPED)}}


def _evaluate(res: dict) -> dict:
    saved = copy.deepcopy(pins.PROMPT_SETS)
    try:  # the synthetic prompt hashes are placeholders: pin them for the duration of the evaluation
        for k in runner.TASK_ORDER:
            pins.PROMPT_SETS[k] = {"units": runner.COUNTS[k], "prompt_set_sha256": runner.prompt_set_sha256(res["tasks"][k])}
        return runner.evaluate(res)
    finally:
        pins.PROMPT_SETS.clear()
        pins.PROMPT_SETS.update(saved)


def test_validity_never_depends_on_rabit_quality():
    base = _evaluate(_synthetic("same"))
    assert base["valid"] and len(base["gates"]) == 14 and all(base["gates"].values())
    assert not [g for g in base["gates"] if "rabit" in g]
    for quality in ("garbage", "empty"):  # a suite with terrible RABIT quality is still VALID, gate for gate
        ev = _evaluate(_synthetic(quality))
        assert ev["valid"] and ev["gates"] == base["gates"] and ev["bf16_control"] == base["bf16_control"], quality
        assert ev["statistics"]["niah"]["rabit"]["passed"] == 0 and ev["statistics"]["passage_retrieval"]["rabit"] == 0.0
    s = base["statistics"]
    assert s["niah"]["bf16"]["passed"] == s["niah"]["rabit"]["passed"] == 57 and s["passage_retrieval"]["delta_points"] == 0.0
    assert set(s["hotpotqa"]) == {"primary_legacy_scorer", "secondary_official_scorer"}
    assert s["hotpotqa"]["primary_legacy_scorer"]["seed"] == 20270932 and s["passage_retrieval"]["seed"] == 20270931
    assert s["legacy_comparison"]["hotpotqa"]["label"] == "LEGACY LOGICAL-EVALUATOR RESULTS"
    # evaluate() reads rabit rows only in statistics() and in harness-integrity checks (never in a control gate)
    ev_fn = next(n for n in tree(Path(runner.__file__)).body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
    ctrl = [s for s in ast.walk(ev_fn) if isinstance(s, ast.Assign) and getattr(s.targets[0], "id", "") == "ctrl"][0]
    assert '"rabit"' not in ast.unparse(ctrl) and "'rabit'" not in ast.unparse(ctrl) and ast.unparse(ctrl).count("'bf16'") == 3


def test_evaluate_rejects_each_invalid_condition():
    def bad(gate, mutate):
        r = _synthetic("same")
        mutate(r)
        ev = _evaluate(r)
        assert not ev["valid"] and ev["gates"].get(gate) is False, (gate, ev["gates"])

    bad("hardware", lambda r: r["hardware"].update(gpus=[{"name": "NVIDIA H200", "memory.total": "143771"}]))
    bad("runtime_environment", lambda r: r["runtime_environment"]["runtime"].update(torch="2.12.0+cu130"))
    bad("model_snapshot", lambda r: r["model"].update(model_revision="master"))
    bad("datasets", lambda r: r["datasets"]["hotpotqa"].update(sha256="0" * 64))
    bad("shipped_files", lambda r: r["files"]["sha256_lf"].update({"canonical_rabit_quality.py": "0" * 64}))
    bad("geometry_and_policy", lambda r: r["policy"].update(residual_tokens=8))
    bad("legacy_unreachable", lambda r: r["legacy_unreachable"]["repo_files"].append("benchmarks/quality/hotpotqa.py"))
    bad("structure", lambda r: r["tasks"]["niah"].pop())
    bad("structure", lambda r: r["tasks"]["hotpotqa"][0]["arms"]["rabit"].update(cache_class="DynamicCache"))
    bad("structure", lambda r: r["tasks"]["hotpotqa"][0]["arms"]["rabit"].update(decode_forwards=99))
    bad("structure", lambda r: r.update(task_order=["passage_retrieval", "niah", "hotpotqa"]))
    bad("scores_equal_frozen_scorers", lambda r: r["tasks"]["hotpotqa"][7]["arms"]["rabit"].update(score=1.0, prediction="no"))
    bad("scores_equal_frozen_scorers", lambda r: r["tasks"]["niah"][0]["arms"]["bf16"].update(correct=False))
    bad("examples_equal_exp12", lambda r: r["tasks"]["passage_retrieval"][3].update(original_tokens=1))
    bad("examples_equal_exp12", lambda r: r["tasks"]["hotpotqa"][50].update(answers=["x"]))
    bad("prompt_sets", lambda r: r["prompt_sets"]["niah"].update(prompt_set_sha256="0" * 64))

    def bf16_wrong(task, n):
        def f(r):
            for row in r["tasks"][task][:n]:
                a = row["arms"]["bf16"]
                a["prediction"] = "zzz 0 nothing"
                a.update(tasks.niah_score(a["prediction"]) if task == "niah" else tasks.score(task, a["prediction"], row["answers"]))
        return f

    bad("bf16_control_passage_retrieval", bf16_wrong("passage_retrieval", 1))  # 90.0 on the 10-row subset
    bad("bf16_control_hotpotqa", bf16_wrong("hotpotqa", 3))
    r = _synthetic("same")  # NIAH: a wrong BF16 answer on a canonical-subset case (depth 0.10 @ 4096) fails the gate ...
    i = next(k for k, row in enumerate(r["tasks"]["niah"]) if row["key"] == [4096, 0.1])
    r["tasks"]["niah"][i]["arms"]["bf16"].update(prediction="none", **tasks.niah_score("none"))
    ev = _evaluate(r)
    assert not ev["valid"] and ev["gates"]["bf16_control_niah"] is False
    r = _synthetic("same")  # ... and one outside the canonical subset (depth 0.05) does not (the accepted Exp12 rule)
    r["tasks"]["niah"][0]["arms"]["bf16"].update(prediction="none", **tasks.niah_score("none"))
    assert _evaluate(r)["gates"]["bf16_control_niah"] is True


def test_runner_never_executes_by_default_and_registered_suite_rule():
    for argv in ([], ["--dry-run", "--execute"]):
        try:
            runner.main(argv)
        except SystemExit as e:
            assert e.code not in (0, None)
            continue
        raise AssertionError(argv)
    src = Path(runner.__file__).read_text(encoding="utf-8")
    assert "return run(a.attempt)" in src and "exist_ok=False" in src
    assert not (identifiers(Path(runner.__file__)) & {"retry", "retries", "sleep"})
    p = runner.load_protocol()
    assert "no selective rerun" in p["registered_suite"]["rule"] and "no automatic retry" in p["registered_suite"]["rule"]
    assert p["registered_suite"]["timeout_hours"] == 12 and "INVALID infrastructure attempt" in p["registered_suite"]["timeout_rule"]
    assert set(p["not_run_here"]) >= {"Qasper", "multilingual", "Qwen long-context", "ablations", "retuning", "serving"}


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
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
