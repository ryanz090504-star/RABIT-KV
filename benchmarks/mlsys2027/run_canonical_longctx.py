"""
Runner: LLAMA-ONLY CANONICAL LONG-CONTEXT VALIDATION (canonical-quality-v2; logical quality, NOT physical serving
evidence). ONE registered suite = ONE container invocation running, in fixed order, NIAH (57) -> Passage Retrieval
(200) -> HotpotQA (100), each example paired BF16 vs canonical RABIT (K3 / V2 / G32 / R4 / META8g64) on identical
prompt token ids, with the frozen legacy Exp12 prompts, selection, truncation, greedy generation schedule and scorers
(canonical_longctx_tasks.py / canonical_longctx_core.py; canonical_rabit_quality.py is the only RABIT implementation).

Validity gates depend ONLY on infrastructure, hardware / runtime, model and dataset identity, harness integrity and the
accepted Exp12 BF16 CONTROL rules. No gate reads a RABIT quality value, and there is no quality threshold:
    NIAH               BF16 accuracy on the 15-case canonical subset within 1.0 point of 100.0   (Exp12 rule)
    Passage Retrieval  BF16 score on the 10-row canonical subset within 1.0 point of 100.0        (Exp12 rule)
    HotpotQA           frozen per-example QA control gate on the first 20 examples vs results/quality/hotpotqa.log:
                       BF16 score mismatches <= 1 and L1 <= 0.065                                 (amendment 86b03ea)
If any gate fails the WHOLE suite is INVALID (archive; no selective rerun; no retry).

Usage:
    python benchmarks/mlsys2027/run_canonical_longctx.py --write-protocol                (once, before commit)
    python benchmarks/mlsys2027/run_canonical_longctx.py --dry-run
    python benchmarks/mlsys2027/run_canonical_longctx.py --execute --attempt 1           (ONLY when authorized)
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import canonical_longctx_pins as pins  # noqa: E402  (stdlib only)
import canonical_longctx_tasks as tasks  # noqa: E402  (stdlib only)
import canonical_ppl_identity as ident  # noqa: E402  (stdlib only)
import paired_bootstrap_ci as pb  # noqa: E402  (accepted Exp12 statistics; stdlib only; read-only)
from run_canonical_quality_parity import poll_cleanup  # noqa: E402  (accepted cleanup polling; stdlib only)

APP_NAME = "rabit-kv-canonical-quality-v2-longctx"
MODEL_KEY = "llama3_1_8b"
MODAL_APP = HERE / "canonical_longctx_modal.py"
STATIC_TESTS = HERE / "test_canonical_longctx_harness.py"
PROTOCOL = HERE / "canonical_longctx_protocol.json"
OUT_BASE = ROOT / "results/mlsys2027/canonical_quality_v2/long_context"
PROOF_RECORD = OUT_BASE / "offline_proof_record.json"
ARTIFACT_VOLUME = "rabit-kv-canonical-longctx-artifacts"
SHIPPED = ["canonical_rabit_quality.py", "canonical_longctx_core.py", "canonical_longctx_tasks.py",
           "canonical_longctx_pins.py", "canonical_ppl_identity.py", "exp14_model_snapshot.py"]
HARNESS = [*[HERE / n for n in SHIPPED], MODAL_APP, Path(__file__).resolve(), STATIC_TESTS,
           HERE / "canonical_longctx_offline_proofs.py", HERE / "proof_observer.py", PROTOCOL, PROOF_RECORD]
TASK_ORDER = ["niah", "passage_retrieval", "hotpotqa"]
ARMS = ["bf16", "rabit"]
COUNTS = {"niah": 57, "passage_retrieval": 200, "hotpotqa": 100}
MAX_NEW = {"niah": tasks.NIAH_MAX_NEW_TOKENS, "passage_retrieval": tasks.LONGBENCH_MAX_NEW_TOKENS,
           "hotpotqa": tasks.LONGBENCH_MAX_NEW_TOKENS}
LEGACY_SCRIPT = {t: f"benchmarks/quality/{t}.py" for t in TASK_ORDER}
LEGACY_SCRIPT_SHA256_RAW = {  # raw-byte hashes recorded by the accepted Exp12 manifest (results/mlsys2027/variance)
    "niah": "845e58c018859554400c6c2a239b160320e6fae8fe3089d050a3de53cbdd3c9e",
    "passage_retrieval": "d9835832b586a7a6c09118436694ad028f7f32473573ca952523cf6fa013dbee",
    "hotpotqa": "67c9b5754bced90a0dc0cac5cbacc748566ad28005eefb5d2dd2ae95696d7288"}
EXP12_LOG = {t: f"results/mlsys2027/variance/{t}.log" for t in TASK_ORDER}  # accepted Exp12 (legacy evaluator)
REFERENCE_LOG = {t: f"results/quality/{t}.log" for t in TASK_ORDER}  # the canonical-subset control reference
EXP12_PROTOCOL = "benchmarks/mlsys2027/exp12_variance_protocol.json"
QA_AMENDMENT = "results/mlsys2027/control_reproducibility_audit/qa_control_gate_amendment.json"
BOOTSTRAP = {"resamples": 10000, "confidence": 0.95, "seed": {"passage_retrieval": 20270931, "hotpotqa": 20270932}}
SUITE_TIMEOUT_S = 12 * 3600
WALL_CLOCK_S = SUITE_TIMEOUT_S + 1800
VALIDATED_RUNTIME = {"python_minor": "3.11", "torch": "2.11.0+cu130", "torch_cuda": "13.0",
                     "cuda_device": "NVIDIA H100 80GB HBM3", "transformers": "4.48.2"}  # == canonical_longctx_modal.py
CONFORMANCE_BASE = "results/mlsys2027/canonical_quality_v2/cuda_conformance_diagnostic"
CONFORMANCE = {  # accepted real-model CUDA canonical <-> CUDA frozen-oracle evidence for Llama (prerequisite)
    "llama_1k_ppl_window": {"dir": f"{CONFORMANCE_BASE}/attempt_1", "prefill_tokens": 1024,
                            "record_sha256_lf": "65f37a9b6053bd3bcf1cb388d6c48cdc69c6ad5d785a0781477bbba3486d1a71",
                            "result_sha256": "593e8a29ce63904278933a6d25855a41571a97b9a75a770d0bbdf6b163999106"},
    "llama_16k_niah": {"dir": f"{CONFORMANCE_BASE}/llama3_1_8b_niah_16384/attempt_1", "prefill_tokens": 16383,
                       "record_sha256_lf": pins.CONFORMANCE_16K["record_sha256_lf"],
                       "result_sha256": pins.CONFORMANCE_16K["result_sha256"]},
}
SCIENTIFIC_FILES = ["benchmarks/mlsys2027/canonical_rabit_quality.py", "benchmarks/mlsys2027/canonical_ppl_identity.py",
                    "benchmarks/mlsys2027/exp14_model_snapshot.py"]
FROZEN = {
    "canonical_impl_c360697": ("c36069781b259e2e9b4d8adf60b7c58ea5df5cac", ["benchmarks/mlsys2027/canonical_rabit_quality.py"]),
    "parity_evidence_8fa9a9c": ("8fa9a9c399245c7db17e64b6b7f206c8f259d6a9", ["results/mlsys2027/quality_semantic_audit/parity_attempt_3"]),
    "legacy_scripts_and_reference": ("599d059cc3cad96f8cdf3c4f813f5460e5b35654", ["benchmarks/quality", "results/quality"]),
    "exp12_evidence_4f767ab": ("4f767ab03d83e043b2871dd0cd4cf2f8dc862e6b", ["results/mlsys2027/variance"]),
    "qa_control_gate_amendment_86b03ea": ("86b03eabaecfed0a838cabd7a7d1cce2f0a2fa38", [QA_AMENDMENT]),
    "model_identity_1e5a7ff": ("1e5a7ff8f5c50cb7c85246dd1eb93b6de8418641", ["benchmarks/mlsys2027/canonical_ppl_identity.py",
                                                                           "benchmarks/mlsys2027/exp14_model_snapshot.py"]),
    "accepted_ppl_attempt_2_bae70af": ("bae70af", ["results/mlsys2027/canonical_quality_v2/continuation_ppl"]),
    "llama_cuda_conformance_ec80638": ("ec80638", [f"{CONFORMANCE_BASE}/attempt_1"]),
    "llama_16k_cuda_conformance_2637457": ("2637457", [f"{CONFORMANCE_BASE}/llama3_1_8b_niah_16384"]),
}
APP_RE = re.compile(r"ap-[A-Za-z0-9]{20,}")
NIAH_CASE_RE = re.compile(r"^\s+(\S+)\s+(PASS|FAIL)\s+KV=[0-9.]+ MB Comp=[0-9.]+x answer=(.*)$")
GROUND_TRUTH_RE = re.compile(r"^Ground truth: (.*)$")


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def env() -> dict:
    e = os.environ.copy()
    e["PYTHONUTF8"], e["PYTHONIOENCODING"] = "1", "utf-8"
    return e


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8", errors="replace")


def apps() -> dict:
    p = subprocess.run([sys.executable, "-m", "modal", "app", "list", "--json"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env(), timeout=180)
    return {r["App ID"]: r for r in json.loads(p.stdout) if r.get("Description") == APP_NAME}


# ------------------------------------------------------------------------------------- accepted legacy evidence
def function_source_sha256(path: Path, names: list) -> dict:
    """SHA-256 of the exact source text of top-level functions (frozen scorer / prompt implementations)."""
    src = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    fns = {n.name: n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    return {n: hashlib.sha256(ast.get_source_segment(src, fns[n]).encode("utf-8")).hexdigest() for n in names}


def legacy_functions(task: str, names: list) -> dict:
    """The named functions of the frozen legacy script, executed from its source by AST (scorers only: stdlib)."""
    src = read(LEGACY_SCRIPT[task])
    fns = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name in names]
    if sorted(n.name for n in fns) != sorted(names):
        raise RuntimeError(f"{task}: legacy functions not found as expected: {names}")
    ns = {"re": re, "json": json}
    exec(compile(ast.Module(body=fns, type_ignores=[]), LEGACY_SCRIPT[task], "exec"), ns)  # noqa: S102
    return ns


def exp12_rows(task: str, log: str | None = None) -> dict:
    """Per-unit rows of an accepted legacy log: {bf16: [...], rabit2: [...]} plus the identity of every unit.
    LongBench units carry dataset index, original / used token counts and ground truths; NIAH units their answers."""
    text = read(log or EXP12_LOG[task])
    rows = pb.extract(task, text)
    lines = text.splitlines()
    if task == "niah":
        answers = {m: [] for m in pb.METHODS}
        for ln in lines:
            r = NIAH_CASE_RE.match(ln)
            if r and r.group(1) in answers:
                answers[r.group(1)].append(ast.literal_eval(r.group(3)))
        for m in pb.METHODS:
            if len(answers[m]) != len(rows[m]):
                raise RuntimeError("NIAH case lines and result table disagree")
            for row, a in zip(rows[m], answers[m]):
                row["answer"] = a
        return {"rows": rows, "identity": [{"key": x["key"]} for x in rows["bf16"]]}
    samples = [pb.SAMPLE_RE.match(ln) for ln in lines]
    truths = [ast.literal_eval(GROUND_TRUTH_RE.match(ln).group(1)) for ln in lines if GROUND_TRUTH_RE.match(ln)]
    ident_rows = [{"key": int(s.group(3)), "original_tokens": int(s.group(4)), "used_tokens": int(s.group(5))}
                  for s in samples if s]
    if len(ident_rows) != len(truths):
        raise RuntimeError(f"{task}: sample lines and ground-truth lines disagree")
    for r, t in zip(ident_rows, truths):
        r["answers"] = t
    return {"rows": rows, "identity": ident_rows}


def legacy_summary(task: str) -> dict:
    """The accepted Exp12 LEGACY LOGICAL-EVALUATOR aggregates (context only; never pooled)."""
    e = exp12_rows(task)
    out = {"label": "LEGACY LOGICAL-EVALUATOR RESULTS", "log": EXP12_LOG[task], "n": len(e["rows"]["bf16"])}
    for m in pb.METHODS:
        out[m] = pb.aggregate(task, [x["value"] for x in e["rows"][m]])
    out["delta_points"] = out["rabit2"] - out["bf16"]
    out["summary_lines_in_log"] = [f"{EXP12_LOG[task]}:{i + 1}: {ln.rstrip()}" for i, ln in
                                   enumerate(read(EXP12_LOG[task]).splitlines()) if re.match(r"(bf16|rabit2)\s+\d", ln)][-2:]
    return out


def milli(score: float) -> int:
    """A score as the legacy evaluator printed it (3 decimals), in exact integer thousandths."""
    return int(round(float(f"{score:.3f}") * 1000))


# ------------------------------------------------------------------------------------- BF16 control gates (Exp12)
def bf16_subset_gate(task: str, keys: list, bf16_values: list) -> dict:
    """The accepted Exp12 control rule for NIAH / Passage Retrieval, BF16 only: on the canonical subset (the units of
    results/quality/<task>.log) the BF16 aggregate reproduces the canonical target within the Exp1 tolerance."""
    proto = json.loads(read(EXP12_PROTOCOL))["validity"]["continuation_ppl_niah_passage_retrieval"]
    target = proto["targets"][task]["bf16"]["accuracy_pct"]
    tol = proto["tolerances"]["accuracy_pct"]["absolute_points"]
    ref_keys = [x["key"] for x in pb.extract(task, read(REFERENCE_LOG[task]))["bf16"]]
    by_key = {json.dumps(k): v for k, v in zip(keys, bf16_values)}
    present = all(json.dumps(k) in by_key for k in ref_keys) and (task == "niah" or keys[:len(ref_keys)] == ref_keys)
    out = {"rule": proto["rule"], "arm": "bf16 only (no RABIT gate)", "canonical_subset_units": len(ref_keys),
           "target_accuracy_pct": target, "tolerance_absolute_points": tol, "canonical_subset_units_present": present}
    if present:
        agg = pb.aggregate(task, [by_key[json.dumps(k)] for k in ref_keys])
        out.update(observed_accuracy_pct=agg, passed=abs(agg - target) <= tol + 1e-9)
    else:
        out["passed"] = False
    return out


def bf16_hotpot_gate(identity: list, bf16_scores: list) -> dict:
    """The frozen per-example QA control gate (amendment 86b03ea) on the canonical subset (first 20 examples), BF16
    only: identical dataset indices / token counts / ground truths, score mismatches and L1 within the frozen maxima."""
    thr = json.loads(read(QA_AMENDMENT))["thresholds"]["hotpotqa"]["bf16"]
    ref = exp12_rows("hotpotqa", REFERENCE_LOG["hotpotqa"])
    rr, n_ref = ref["rows"]["bf16"], len(ref["rows"]["bf16"])
    ident_ok = identity[:n_ref] == ref["identity"]
    out = {"rule": "frozen per-example QA control gate on the canonical subset (results/quality/hotpotqa.log)",
           "arm": "bf16 only (no RABIT gate)", "n_canonical_subset": n_ref, "canonical_subset_identity": ident_ok,
           "max_allowed_mismatch_count": thr["historical_max_score_mismatch_count"],
           "max_allowed_l1_score_distance": thr["historical_max_l1_score_distance"]}
    if not ident_ok or len(bf16_scores) < n_ref:
        out["passed"] = False
        return out
    d = [(i, milli(s), pb_milli(r["value"])) for i, (s, r) in enumerate(zip(bf16_scores[:n_ref], rr))]
    mism = [i for i, a, b in d if a != b]
    l1 = sum(abs(a - b) for _, a, b in d)
    out.update(score_mismatch_count=len(mism), score_mismatch_indices=mism, l1_score_distance=l1 / 1000,
               passed=len(mism) <= thr["historical_max_score_mismatch_count"]
               and l1 <= thr["historical_max_l1_score_distance_milli"])
    return out


def pb_milli(printed: float) -> int:
    return int(round(printed * 1000))


# ------------------------------------------------------------------------------------- prompt-set identity
def prompt_set_sha256(rows: list) -> str:
    """SHA-256 over the ordered (key, token count, token-id SHA-256) of every prompt of a task."""
    return hashlib.sha256("\n".join(f"{json.dumps(r['key'])}\t{r['prompt_tokens']}\t{r['prompt_ids_sha256']}"
                                    for r in rows).encode()).hexdigest()


def expected_keys(task: str) -> list:
    if task == "niah":
        return [[c, round(d, 2)] for c, d in tasks.niah_cases()]
    return [x["key"] for x in exp12_rows(task)["identity"]]


# ------------------------------------------------------------------------------------- conformance evidence
def conformance_evidence(name: str) -> dict:
    """An accepted Llama CUDA canonical <-> CUDA oracle diagnostic: pinned bytes, valid, classification A, exact model
    identity, validated runtime, bit-exact in every layer, and the CURRENT scientific files."""
    c, m = CONFORMANCE[name], ident.MODELS[MODEL_KEY]
    rec_p, res_p = ROOT / c["dir"] / "record.json", ROOT / c["dir"] / "result.json"
    if not (rec_p.is_file() and res_p.is_file()):
        return {"present": False}
    rec, raw = json.loads(rec_p.read_text(encoding="utf-8")), res_p.read_bytes()
    res = json.loads(raw.decode("utf-8"))
    e = res["environment"]
    return {
        "present": True,
        "record_pinned": sha256_lf(rec_p) == c["record_sha256_lf"],
        "result_pinned": hashlib.sha256(raw).hexdigest() == c["result_sha256"] == rec.get("result_sha256"),
        "valid_and_classification_a": rec.get("diagnostic_valid") is True and rec.get("classification") == "A",
        "exact_model_identity": [res["model"][k] for k in ("model_id", "model_revision", "manifest_sha256")]
        == [m["model_id"], m["revision"], m["manifest_sha256"]] and len(res["layers"]) == m["layers"],
        "prefill_tokens": all(r["raw"]["shape"] == [c["prefill_tokens"], m["kv_heads"], m["head_dim"]] for r in res["layers"]),
        "every_layer_cuda_canonical_equals_cuda_oracle": all(
            r["summary"]["cuda_canonical_equals_cuda_oracle"] is True
            and r["summary"]["canonical_cache_equals_cuda_oracle"] is True for r in res["layers"]),
        "validated_runtime": {"python_minor": ".".join(e["python"].split(".")[:2]), "torch": e["torch"],
                              "torch_cuda": e["torch_cuda"], "cuda_device": e["cuda_device"],
                              "transformers": e["packages"]["transformers"]} == VALIDATED_RUNTIME,
        "scientific_files_unchanged_since_conformance": all(
            res["files"]["sha256_lf"].get(f) == sha256_lf(ROOT / f) for f in SCIENTIFIC_FILES),
        "no_scoring_in_diagnostic": res["no_scoring"]["continuation_tokens_scored"] == 0,
    }


# ------------------------------------------------------------------------------------- protocol
def build_protocol() -> dict:
    ident.check_constants()
    m = ident.MODELS[MODEL_KEY]
    tasks_file = HERE / "canonical_longctx_tasks.py"
    return {
        "name": "Llama-only canonical long-context validation (canonical-quality-v2)",
        "type": "logical quality (HF transformers); NOT a physical serving benchmark",
        "model": {k: m[k] for k in ("model_id", "revision", "manifest_sha256", "layers", "kv_heads", "head_dim")},
        "policy": "K3 / V2 / G32 / R4 / META8g64 (frozen; no retuning; no task-specific setting)",
        "rabit_implementation": {"file": "benchmarks/mlsys2027/canonical_rabit_quality.py", "commit": "c360697",
                                 "sole_implementation": True},
        "tasks": {
            "niah": {"cases": COUNTS["niah"], "contexts": list(tasks.NIAH_CONTEXTS), "depths": list(tasks.NIAH_DEPTHS),
                     "max_new_tokens": MAX_NEW["niah"], "order": "context-major, then depth",
                     "prompt": "instruction + WikiText-2 filler with the needle at round(filler x depth) + question; "
                               "no chat template; token ids without special tokens",
                     "filler": {"url": tasks.WIKITEXT_URL, "sha256": tasks.WIKITEXT_SHA256},
                     "warmup": "one unreported BF16 case at 4096 tokens, depth 0.05",
                     "metric": "exact match: regex RABIT-\\d{4} on the upper-cased answer equals RABIT-7291"},
            "passage_retrieval": {**{k: tasks.DATASETS["passage_retrieval"][k] for k in ("revision", "filename", "bytes",
                                                                                         "sha256", "samples")},
                                  "repo": tasks.LONGBENCH_REPO, "selection": "dataset rows [0, 200)",
                                  "max_input_tokens": tasks.MAX_INPUT_TOKENS, "max_new_tokens": MAX_NEW["passage_retrieval"],
                                  "metric": "LongBench retrieval score, mean x 100"},
            "hotpotqa": {**{k: tasks.DATASETS["hotpotqa"][k] for k in ("revision", "filename", "bytes", "sha256", "samples")},
                         "repo": tasks.LONGBENCH_REPO,
                         "selection": "rows with length >= 8000 (bucket 8k+), filtered positions [0, 100)",
                         "max_input_tokens": tasks.MAX_INPUT_TOKENS, "max_new_tokens": MAX_NEW["hotpotqa"],
                         "metric_primary": "qa_f1_score_legacy: the accepted Exp12 scorer verbatim, including its historical "
                                           "article behaviour (articles are NOT removed); max over references; mean x 100",
                         "metric_secondary": "qa_f1_score_official: standards-aligned LongBench qa_f1_score (articles "
                                             "removed); computed from the identical predictions; secondary only"},
        },
        "longbench_common": {"prompt": "LongBench official template through the tokenizer chat template "
                                       "(add_generation_prompt), tokenized without special tokens",
                             "truncation": "if longer than 16384 tokens keep the first 8192 and the last 8192",
                             "warmup": "one 2-token forward before the task"},
        "dataset_hash_provenance": "the parquet SHA-256 / sizes are NEWLY RECOVERED / FROZEN provenance for this suite "
                                   "(2026-10-03); the accepted Exp12 evidence recorded the dataset commits only",
        "prompt_sets": {"definition": "sha256 of the ordered 'key<TAB>token count<TAB>int64-LE token-id sha256' lines",
                        **pins.PROMPT_SETS},
        "task_definition_sha256_lf": {"canonical_longctx_tasks.py": sha256_lf(tasks_file),
                                      "canonical_longctx_core.py": sha256_lf(HERE / "canonical_longctx_core.py"),
                                      "canonical_longctx_pins.py": sha256_lf(HERE / "canonical_longctx_pins.py")},
        "scorer_source_sha256": function_source_sha256(tasks_file, [
            "niah_score", "retrieval_score", "normalize_answer_legacy", "qa_f1_score_legacy",
            "normalize_answer_official", "qa_f1_score_official", "normalize_answers", "score"]),
        "legacy_scripts_sha256_raw": LEGACY_SCRIPT_SHA256_RAW,
        "generation": {"prefill": "prompt[:-1] in BF16", "first_decode_step": "the last prompt token",
                       "decoding": "greedy argmax, one token per forward", "stop": "tokenizer EOS (not kept) or the token limit",
                       "answer": "tokenizer.decode(kept ids, skip_special_tokens=True).strip()"},
        "arms": {"bf16": "the BF16 prefill cache itself", "rabit": "CanonicalRabitCache.from_prefill(BF16 prefill); the "
                 "last prompt token and every generated token age canonically (R4 residual -> open group -> closed page)"},
        "pairing": "per example: bf16 then rabit on the identical prompt token ids",
        "model_settings": {"dtype": "bfloat16", "attention": "sdpa", "seed": "torch.manual_seed(0), cuda.manual_seed_all(0)",
                           "tf32": "enabled (matmul and cudnn)"},
        "registered_suite": {"order": TASK_ORDER, "invocation": "ONE container invocation; no result is printed, "
                             "summarized or used for control flow between tasks",
                             "rule": "valid only if EVERY validity gate of all three tasks passes; otherwise the whole "
                                     "suite is INVALID and archived; no selective rerun; no automatic retry",
                             "timeout_hours": SUITE_TIMEOUT_S // 3600,
                             "timeout_rule": "infrastructure only; a timeout is an INVALID infrastructure attempt"},
        "hardware": {"selector": "H100!:1", "guard": "fail-closed: exactly 1 x NVIDIA H100 80GB before anything else"},
        "validated_runtime": VALIDATED_RUNTIME,
        "conformance_prerequisites": {k: {x: v[x] for x in ("dir", "prefill_tokens", "record_sha256_lf", "result_sha256")}
                                      for k, v in CONFORMANCE.items()},
        "validity_gates": [
            "modal exit code 0, no wall-clock timeout, app cleanup verified",
            "hardware guard passed (1 x H100 80GB); runtime equals the validated conformance runtime",
            "model snapshot at the pinned revision equals the frozen manifest",
            "dataset files equal the pinned SHA-256 / size; WikiText SHA-256 equals the pinned value",
            "every prompt set equals its pinned SHA-256 (computed in the container before the model is loaded)",
            "shipped-file SHA-256 equal the committed harness; canonical_rabit_quality.py equals c360697",
            "structure: fixed task order, 57 / 200 / 100 examples, both arms per example on identical prompt ids, "
            "rabit rows use CanonicalRabitCache, one token per forward",
            "every stored score equals the frozen scorer applied to the stored prediction string",
            "every LongBench example equals the Exp12 log (dataset index, original / used token counts, ground truth); "
            "NIAH cases equal the Exp12 grid",
            "no legacy quality module imported in the container; only the shipped files present",
            "BF16 control, NIAH: 15-case canonical subset accuracy within 1.0 point of 100.0 (Exp12 rule)",
            "BF16 control, Passage Retrieval: 10-row canonical subset score within 1.0 point of 100.0 (Exp12 rule)",
            "BF16 control, HotpotQA: frozen per-example gate on the first 20 examples: mismatches <= 1, L1 <= 0.065"],
        "rabit_quality_gate": "NONE -- no gate reads a RABIT score; a valid suite is never rejected for RABIT quality",
        "quality_threshold": "NONE -- no definition of preserved / healthy / acceptable; exact values are reported",
        "statistics": {"niah": "deterministic grid only (no inferential statistics): exact-retrieval counts overall and "
                               "per context, failed coordinates, cases changed",
                       "passage_retrieval_hotpotqa": "aggregate = mean score x 100; delta = rabit - bf16 (points); paired "
                                                     "percentile bootstrap over examples (Exp12 procedure and seeds)",
                       **BOOTSTRAP,
                       "descriptive": "examples changed, per-example delta distribution, concentration (top-1 / top-3 "
                                      "contributions), worst examples; nothing is removed"},
        "legacy_comparison": "the accepted Exp12 values are quoted as LEGACY LOGICAL-EVALUATOR RESULTS; never pooled",
        "attempts": "results/mlsys2027/canonical_quality_v2/long_context/attempt_<n>/; never overwritten",
        "not_run_here": ["Qasper", "multilingual", "Qwen long-context", "ablations", "retuning", "serving", "profiling",
                         "kernel optimization"],
    }


def load_protocol() -> dict:
    committed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if committed != json.loads(json.dumps(build_protocol())):
        raise RuntimeError("canonical_longctx_protocol.json differs from the regenerated protocol")
    return committed


def preflight(attempt: int, execute: bool) -> dict:
    t = subprocess.run([sys.executable, str(STATIC_TESTS)], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env(), timeout=1800)
    proof = json.loads(PROOF_RECORD.read_text(encoding="utf-8")) if PROOF_RECORD.exists() else {}
    upstream = subprocess.run(["git", "rev-parse", "@{u}"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    checks = {
        "clean_tree": git("status", "--short") == "",
        "head_is_pushed": git("rev-parse", "HEAD") == upstream,
        "harness_committed": all(p.exists() for p in HARNESS)
        and git("status", "--short", "--", *[rel(p) for p in HARNESS if p.exists()]) == "",
        **{f"unchanged_{k}": git("diff", "--name-only", c, "--", *paths) == "" for k, (c, paths) in FROZEN.items()},
        "legacy_scripts_are_the_exp12_scripts": all(
            hashlib.sha256((ROOT / LEGACY_SCRIPT[t]).read_bytes()).hexdigest() == LEGACY_SCRIPT_SHA256_RAW[t]
            for t in TASK_ORDER),
        "static_harness_tests_pass": t.returncode == 0 and bool(re.search(r"^(\d+)/\1 passed$", t.stdout, re.M)),
        "offline_proofs_passed": proof.get("passed") is True,
        "offline_proofs_bound_to_current_files": bool(proof) and all(
            sha256_lf(ROOT / f) == h for f, h in proof.get("bound_file_sha256_lf", {}).items()),
        "protocol_matches": PROTOCOL.exists() and load_protocol() is not None,
        **{f"cuda_conformance_accepted_{k}": all(v is True for v in conformance_evidence(k).values())
           for k in sorted(CONFORMANCE)},
        "attempt_dir_absent": not (OUT_BASE / f"attempt_{attempt}").exists(),
    }
    if execute and not all(checks.values()):
        raise SystemExit(f"pre-run validation failed: {json.dumps(checks, indent=1)}\nstatic tests: {t.stdout[-1500:]}")
    return checks


# ------------------------------------------------------------------------------------------- evaluation (offline)
def _values(task: str, rows: list, arm: str, field: str = None) -> list:
    if task == "niah":
        return [1.0 if r["arms"][arm]["correct"] else 0.0 for r in rows]
    return [float(r["arms"][arm][field or "score"]) for r in rows]


def _rescore(task: str, row: dict, arm: str) -> bool:
    a = row["arms"][arm]
    if task == "niah":
        s = tasks.niah_score(a["prediction"])
        return s["correct"] == a["correct"] and s["extracted_code"] == a["extracted_code"]
    s = tasks.score(task, a["prediction"], row["answers"])
    return all(a.get(k) == v for k, v in s.items())


def _rows_ok(task: str, rows: list) -> bool:
    ok = len(rows) == COUNTS[task] and [r["index"] for r in rows] == list(range(COUNTS[task]))
    for r in rows if ok else []:
        b, q = r["arms"].get("bf16"), r["arms"].get("rabit")
        ok = ok and list(r["arms"]) == ARMS and b["cache_class"] == "DynamicCache" and q["cache_class"] == "CanonicalRabitCache"
        for a in (b, q) if ok else []:
            n = len(a["generated_ids"])
            ok = ok and (a["prefix_tokens"] == r["prompt_tokens"] - 1 and a["decode_forwards"] == n + 1
                         and a["final_cache_tokens"] == r["prompt_tokens"] + n and n <= MAX_NEW[task]
                         and type(a["prediction"]) is str and (a["stopped_on_eos"] or n == MAX_NEW[task]))
    return bool(ok)


def evaluate(res: dict) -> dict:
    """Validity gates (never a RABIT-dependent one) and descriptive statistics. Pure function of the result JSON."""
    m, t_ = ident.MODELS[MODEL_KEY], res.get("tasks", {})
    structure = list(t_) == TASK_ORDER == res.get("task_order") and all(_rows_ok(k, t_[k]) for k in TASK_ORDER)
    gates = {
        "hardware": res["hardware"]["passed"] is True and ident.hardware_ok(res["hardware"]["gpus"]),
        "runtime_environment": res["runtime_environment"]["passed"] is True
        and res["runtime_environment"]["runtime"] == res["runtime_environment"]["validated"] == VALIDATED_RUNTIME,
        "model_snapshot": res["model"]["passed"] is True and res["model"]["model_id"] == m["model_id"]
        and res["model"]["model_revision"] == m["revision"] and res["model"]["manifest_sha256"] == m["manifest_sha256"]
        and res["model"]["files_checked"] == len(m["files"]),
        "datasets": res["datasets"]["wikitext_sha256"] == tasks.WIKITEXT_SHA256 and all(
            res["datasets"][k]["sha256"] == tasks.DATASETS[k]["sha256"] and res["datasets"][k]["bytes"] == tasks.DATASETS[k]["bytes"]
            and res["datasets"][k]["rows"] == tasks.DATASETS[k]["rows"] for k in tasks.DATASETS),
        "shipped_files": res["files"]["passed"] is True
        and res["files"]["sha256_lf"] == {n: sha256_lf(HERE / n) for n in SHIPPED},
        "geometry_and_policy": [res["geometry"][k] for k in ("layers", "kv_heads", "head_dim")]
        == [m["layers"], m["kv_heads"], m["head_dim"]]
        and res["policy"] == {"k_bits": 3, "v_bits": 2, "group_size": 32, "residual_tokens": 4, "metadata_bits": 8,
                              "metadata_group_size": 64},
        "structure": structure,
        "legacy_unreachable": res["legacy_unreachable"]["passed"] is True
        and res["legacy_unreachable"]["repo_files"] == sorted(f"benchmarks/mlsys2027/{n}" for n in SHIPPED),
    }
    out = {"gates": gates}
    if structure:
        gates["prompt_sets"] = all(res["prompt_sets"][k]["passed"] is True
                                   and prompt_set_sha256(t_[k]) == pins.PROMPT_SETS[k]["prompt_set_sha256"]
                                   == res["prompt_sets"][k]["prompt_set_sha256"] for k in TASK_ORDER)
        gates["scores_equal_frozen_scorers"] = all(_rescore(k, r, a) for k in TASK_ORDER for r in t_[k] for a in ARMS)
        idn = {}
        for k in TASK_ORDER:
            if k == "niah":
                idn[k] = [r["key"] for r in t_[k]] == expected_keys(k) and all(r["prompt_tokens"] == r["key"][0] for r in t_[k])
            else:
                mine = [{"key": r["key"], "original_tokens": r["original_tokens"], "used_tokens": r["prompt_tokens"],
                         "answers": r["answers"]} for r in t_[k]]
                idn[k] = mine == exp12_rows(k)["identity"]
        gates["examples_equal_exp12"] = all(idn.values())
        ctrl = {"niah": bf16_subset_gate("niah", [r["key"] for r in t_["niah"]], _values("niah", t_["niah"], "bf16")),
                "passage_retrieval": bf16_subset_gate("passage_retrieval", [r["key"] for r in t_["passage_retrieval"]],
                                                      _values("passage_retrieval", t_["passage_retrieval"], "bf16")),
                "hotpotqa": bf16_hotpot_gate(
                    [{"key": r["key"], "original_tokens": r["original_tokens"], "used_tokens": r["prompt_tokens"],
                      "answers": r["answers"]} for r in t_["hotpotqa"]], _values("hotpotqa", t_["hotpotqa"], "bf16"))}
        for k in TASK_ORDER:
            gates[f"bf16_control_{k}"] = ctrl[k]["passed"] is True
        out["bf16_control"] = ctrl
        out["examples_equal_exp12"] = idn
        out["statistics"] = statistics(t_)
    out["valid"] = all(gates.values()) and len(gates) == 14
    return out


def _paired(task: str, rows: list, field: str) -> dict:
    keys = [r["key"] for r in rows]
    b, r_ = _values(task, rows, "bf16", field), _values(task, rows, "rabit", field)
    d = [100.0 * (y - x) for x, y in zip(b, r_)]
    agg_b, agg_r = pb.aggregate(task, b), pb.aggregate(task, r_)
    worst = sorted(range(len(d)), key=lambda i: (d[i], i))[:5]
    return {"n": len(rows), "bf16": agg_b, "rabit": agg_r, "delta_points": pb.delta(task, b, r_),
            "relative_delta_pct": 100.0 * (agg_r - agg_b) / agg_b if agg_b else None,
            **pb.paired_bootstrap(task, b, r_, BOOTSTRAP["resamples"], BOOTSTRAP["seed"][task], BOOTSTRAP["confidence"]),
            "examples_score_changed": sum(x != y for x, y in zip(b, r_)),
            "examples_rabit_worse": sum(x < 0 for x in d), "examples_rabit_better": sum(x > 0 for x in d),
            "examples_prediction_text_changed": sum(r["arms"]["bf16"]["prediction"] != r["arms"]["rabit"]["prediction"] for r in rows),
            "robustness": pb.robustness(task, keys, b, r_),
            "worst_examples": [{"key": keys[i], "delta_points": d[i], "answers": rows[i]["answers"],
                                "bf16_prediction": rows[i]["arms"]["bf16"]["prediction"],
                                "rabit_prediction": rows[i]["arms"]["rabit"]["prediction"]} for i in worst],
            "per_example": {"keys": keys, "bf16": b, "rabit": r_}}


def statistics(t_: dict) -> dict:
    """Descriptive statistics; RABIT values are read here ONLY, after and independent of the validity gates."""
    niah = {"statistics": "none (deterministic grid; cases are not independent draws)"}
    for arm in ARMS:
        rows = t_["niah"]
        ok = [r["arms"][arm]["correct"] for r in rows]
        niah[arm] = {"passed": sum(ok), "cases": len(ok), "accuracy_pct": 100.0 * sum(ok) / len(ok),
                     "per_context": {str(c): {"passed": sum(o for o, r in zip(ok, rows) if r["key"][0] == c),
                                              "cases": sum(r["key"][0] == c for r in rows)} for c in tasks.NIAH_CONTEXTS},
                     "failed_coordinates": [r["key"] for o, r in zip(ok, rows) if not o],
                     "failed_answers": [{"key": r["key"], "answer": r["arms"][arm]["prediction"]} for o, r in zip(ok, rows) if not o]}
    niah["delta_points"] = niah["rabit"]["accuracy_pct"] - niah["bf16"]["accuracy_pct"]
    niah["cases_outcome_changed"] = [r["key"] for r in t_["niah"] if r["arms"]["bf16"]["correct"] != r["arms"]["rabit"]["correct"]]
    niah["cases_answer_text_changed"] = sum(r["arms"]["bf16"]["prediction"] != r["arms"]["rabit"]["prediction"] for r in t_["niah"])
    out = {"niah": niah, "passage_retrieval": _paired("passage_retrieval", t_["passage_retrieval"], "score"),
           "hotpotqa": {"primary_legacy_scorer": _paired("hotpotqa", t_["hotpotqa"], "score"),
                        "secondary_official_scorer": _paired("hotpotqa", t_["hotpotqa"], "score_official")},
           "legacy_comparison": {}}
    for k in TASK_ORDER:
        leg, e = legacy_summary(k), exp12_rows(k)
        mine = _values(k, t_[k], "bf16")
        ref = [x["value"] for x in e["rows"]["bf16"]]
        diff = [i for i, (a, b) in enumerate(zip(mine, ref)) if (a != b if k == "niah" else milli(a) != pb_milli(b))]
        leg["bf16_reproducibility"] = {
            "note": "this run's BF16 arm vs the accepted Exp12 BF16 arm, per example (3-decimal printed scores)",
            "examples_differing": len(diff), "indices": diff[:20],
            "l1_distance": sum(abs(a - b) for a, b in zip(mine, ref)) if k == "niah" else
            sum(abs(milli(a) - pb_milli(b)) for a, b in zip(mine, ref)) / 1000}
        out["legacy_comparison"][k] = leg
    return out


def run(attempt: int) -> int:
    checks = preflight(attempt, execute=True)
    out = OUT_BASE / f"attempt_{attempt}"
    out.mkdir(parents=True, exist_ok=False)
    result_tmp = Path(tempfile.mkdtemp()) / "result.json"
    e = env()
    e["CANONICAL_LONGCTX_RESULT_PATH"] = str(result_tmp)
    e["CANONICAL_LONGCTX_EXPECTED_FILE_SHA256_LF"] = json.dumps({n: sha256_lf(HERE / n) for n in SHIPPED})
    rec = {"kind": "canonical-quality-v2 long-context suite (logical quality; NOT physical serving evidence)",
           "attempt": attempt, "source_commit": git("rev-parse", "HEAD"),
           "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "preflight": checks,
           "protocol_sha256_lf": sha256_lf(PROTOCOL), "harness_sha256_lf": {rel(p): sha256_lf(p) for p in HARNESS},
           "wall_clock_s": WALL_CLOCK_S}
    pre, t0, parsed = apps(), time.time(), set()
    with (out / "session.log").open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen([sys.executable, "-m", "modal", "run", str(MODAL_APP), "--attempt", str(attempt)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", env=e, cwd=ROOT)

        def pump():
            for line in proc.stdout:
                fh.write(line)
                fh.flush()
                parsed.update(APP_RE.findall(line))

        th = threading.Thread(target=pump, daemon=True)
        th.start()
        try:
            rc, timed_out = proc.wait(timeout=WALL_CLOCK_S), False
        except subprocess.TimeoutExpired:
            proc.kill()
            rc, timed_out = proc.wait(), True
        th.join(timeout=30)
    post = apps()
    new = sorted((set(post) - set(pre)) | (parsed - set(pre)))
    cleanup = poll_cleanup(new, apps, lambda a: subprocess.run([sys.executable, "-m", "modal", "app", "stop", "-y", a],
                                                               env=env(), capture_output=True, timeout=180))
    # best-effort copy of the incremental raw rows written by the container (crash-recovery / provenance only)
    art = subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", ARTIFACT_VOLUME,
                          f"attempt_{attempt}/rows.jsonl", str(out / "incremental_rows.jsonl")], env=env(),
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    res = ev = None
    if result_tmp.is_file():
        (out / "result.json").write_bytes(result_tmp.read_bytes())
        res = json.loads((out / "result.json").read_bytes().decode("utf-8"))
        ev = evaluate(res)
    process_ok = rc == 0 and not timed_out and cleanup["verified"]
    valid = bool(process_ok and ev and ev["valid"])
    rec.update(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_s=round(time.time() - t0, 1),
               modal_returncode=rc, timed_out=timed_out, app_ids_new=new, cleanup=cleanup, process_ok=process_ok,
               incremental_rows_copied=art.returncode == 0 and (out / "incremental_rows.jsonl").is_file(),
               result_sha256=hashlib.sha256((out / "result.json").read_bytes()).hexdigest() if res else None,
               gpu=res["hardware"]["gpus"] if res else None, evaluation=ev, valid=valid,
               status="valid (awaiting review; not accepted evidence until reviewed)" if valid
               else "INVALID registered suite (archive; no selective rerun; never pool)")
    (out / "record.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8", newline="\n")
    st = ev.get("statistics") if ev else None
    print(json.dumps({"attempt": attempt, "valid": valid, "modal_returncode": rc, "timed_out": timed_out,
                      "gates": ev["gates"] if ev else None,
                      "summary": {"niah": {a: st["niah"][a]["passed"] for a in ARMS},
                                  "passage_retrieval": {k: st["passage_retrieval"][k] for k in ("bf16", "rabit", "delta_points")},
                                  "hotpotqa_primary": {k: st["hotpotqa"]["primary_legacy_scorer"][k] for k in ("bf16", "rabit", "delta_points")},
                                  "hotpotqa_secondary": {k: st["hotpotqa"]["secondary_official_scorer"][k] for k in ("bf16", "rabit", "delta_points")}}
                      if st else None}, indent=1))
    return 0 if valid else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true", help="launch the GPU suite (ONLY when explicitly authorized)")
    ap.add_argument("--write-protocol", action="store_true")
    a = ap.parse_args(argv)
    if a.write_protocol:
        if PROTOCOL.exists():
            raise SystemExit(f"{PROTOCOL.name} already exists; the frozen protocol is never overwritten")
        PROTOCOL.write_text(json.dumps(build_protocol(), indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {rel(PROTOCOL)}")
        return 0
    if a.dry_run == a.execute:
        raise SystemExit("exactly one of --dry-run / --execute is required")
    if a.dry_run:
        checks = preflight(a.attempt, execute=False)
        print(json.dumps({"model": {k: ident.MODELS[MODEL_KEY][k] for k in ("model_id", "revision", "manifest_sha256")},
                          "preflight": checks, "all_preflight_checks_pass": all(checks.values()),
                          "command": f"modal run {rel(MODAL_APP)} --attempt {a.attempt}"}, indent=1))
        print("\n--dry-run: nothing executed, no files written.")
        return 0
    return run(a.attempt)


if __name__ == "__main__":
    raise SystemExit(main())
