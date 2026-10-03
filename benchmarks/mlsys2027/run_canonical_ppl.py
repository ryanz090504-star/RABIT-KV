"""
Runner: TWO-MODEL N32 CANONICAL CONTINUATION-PPL VALIDATION (canonical-quality-v2; logical quality, NOT physical
serving evidence). One invocation = one model on one strict H100 80GB Modal container.

Protocol (frozen in canonical_ppl_protocol.json): WikiText-2 continuation PPL, N = 32 windows, context 1024,
continuation 128, no BOS, the frozen legacy windows / order / scorer (benchmarks/quality/continuation_ppl.py, proven
offline); BF16 prefill; first continuation token from the prefill logit; later tokens teacher-forced one at a time;
BF16 vs canonical RABIT (K3 / V2 / G32 / R4 / META8g64, canonical_rabit_quality.py -- the only RABIT implementation),
the canonical cache aging during decode. Nothing is retuned.

Validity gates (never a quality threshold): process / cleanup, strict hardware, model snapshot manifest, pinned
dataset and token pool, shipped-file hashes, GPU == CPU canonical state, result structure, legacy code unreachable,
and BF16 CONTROL REPRODUCTION -- the legacy batched BF16 scorer must reproduce the accepted legacy BF16 aggregate PPL
of the same model (Llama: Exp12; Qwen: Exp14 quality) within the original Exp1 tolerance (0.5 % relative), and the
stepwise BF16 arm must agree with the batched one within the same tolerance.

Usage:
    python benchmarks/mlsys2027/run_canonical_ppl.py --write-protocol                      (once, before commit)
    python benchmarks/mlsys2027/run_canonical_ppl.py --dry-run --model llama3_1_8b
    python benchmarks/mlsys2027/run_canonical_ppl.py --execute --model llama3_1_8b --attempt 1   (ONLY when authorized)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
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
import canonical_ppl_identity as ident  # noqa: E402  (stdlib only)
import paired_bootstrap_ci as pb  # noqa: E402  (accepted Exp12 statistics; stdlib only; read-only)
from run_canonical_quality_parity import poll_cleanup  # noqa: E402  (accepted cleanup polling; stdlib only)

APP_NAME = "rabit-kv-canonical-quality-v2-ppl"
MODAL_APP = HERE / "canonical_ppl_modal.py"
STATIC_TESTS = HERE / "test_canonical_ppl_harness.py"
PROTOCOL = HERE / "canonical_ppl_protocol.json"
OUT_BASE = ROOT / "results/mlsys2027/canonical_quality_v2/continuation_ppl"
PROOF_RECORD = OUT_BASE / "offline_proof_record_v2.json"  # corrected observer; offline_proof_record.json is SUPERSEDED
PROOF_OBSERVER = "IdentityObserver v2 (object identity + retained direct references; no id() keying)"
TWO_MODEL = HERE / "run_canonical_ppl_two_model.py"
SHIPPED = ["canonical_rabit_quality.py", "canonical_ppl_core.py", "canonical_ppl_identity.py", "exp14_model_snapshot.py"]
HARNESS = [*[HERE / n for n in SHIPPED], MODAL_APP, Path(__file__).resolve(), STATIC_TESTS,
           HERE / "canonical_ppl_offline_proofs.py", HERE / "proof_observer.py", TWO_MODEL, PROTOCOL, PROOF_RECORD]
RABIT_KV2 = ROOT / "vllm-kvquant/vllm/v1/attention/ops/rabit_kv2.py"
RABIT_KV2_SHA256_LF = "7e628c94eebb9fe689bf416ea61f748c0f909a82d0f229c463edd1a0df92e6ae"
PARITY_RESULT_SHA256 = "98d3a5d9a16f83256a31d01556c9ecf26233c418f1c08e0bb971f1cb1d7f7d54"
# (frozen commit, paths that must be unchanged since it)
FROZEN = {
    "canonical_impl_c360697": ("c36069781b259e2e9b4d8adf60b7c58ea5df5cac", ["benchmarks/mlsys2027/canonical_rabit_quality.py"]),
    "parity_evidence_8fa9a9c": ("8fa9a9c399245c7db17e64b6b7f206c8f259d6a9", ["results/mlsys2027/quality_semantic_audit/parity_attempt_3"]),
    "harness_amendment_257f0fb": ("257f0fb2d53cc5715664df006bed9fbd8ab6344b", ["results/mlsys2027/quality_semantic_audit"]),
    "legacy_scripts_and_reference": ("599d059cc3cad96f8cdf3c4f813f5460e5b35654", ["benchmarks/quality", "results/quality"]),
    "exp12_evidence_4f767ab": ("4f767ab03d83e043b2871dd0cd4cf2f8dc862e6b", ["results/mlsys2027/variance"]),
    "exp14_quality_evidence_4049642": ("4049642e197e6da0e12cd2e3a4da3fee4d73fff6", ["results/mlsys2027/second_model/quality"]),
    "exp14_model_identity": ("4049642e197e6da0e12cd2e3a4da3fee4d73fff6", ["benchmarks/mlsys2027/exp14_model_snapshot.py"]),
}
LEGACY_LOG = {"llama3_1_8b": "results/mlsys2027/variance/continuation_ppl.log",  # accepted Exp12
              "qwen2_5_7b": "results/mlsys2027/second_model/quality/continuation_ppl.log"}  # accepted Exp14 quality
CONTROL_REL_TOL = 0.005  # the original Exp1 PPL control-reproduction tolerance (unchanged)
BOOTSTRAP = {"resamples": 10000, "confidence": 0.95, "seed": 20270929}  # the Exp12 continuation_ppl bootstrap spec
WALL_CLOCK_S = 3 * 3600
APP_RE = re.compile(r"ap-[A-Za-z0-9]{20,}")


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


def apps() -> dict:
    p = subprocess.run([sys.executable, "-m", "modal", "app", "list", "--json"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env(), timeout=180)
    return {r["App ID"]: r for r in json.loads(p.stdout) if r.get("Description") == APP_NAME}


def legacy_reference(model: str) -> dict:
    """Per-window PPL of the accepted LEGACY evaluator for this model (4-decimal log values)."""
    rows = pb.extract("continuation_ppl", (ROOT / LEGACY_LOG[model]).read_text(encoding="utf-8", errors="replace"))
    if [len(rows[m]) for m in ("bf16", "rabit2")] != [ident.SAMPLES] * 2:
        raise RuntimeError(f"legacy reference log of {model} does not hold {ident.SAMPLES} windows per method")
    out = {m: [x["value"] for x in rows[m]] for m in ("bf16", "rabit2")}
    out["bf16_aggregate_ppl"] = pb.aggregate("continuation_ppl", [math.log(v) for v in out["bf16"]])
    out["rabit2_legacy_aggregate_ppl"] = pb.aggregate("continuation_ppl", [math.log(v) for v in out["rabit2"]])
    return out


def build_protocol() -> dict:
    ident.check_constants()
    return {
        "name": "two-model N32 canonical continuation-PPL validation (canonical-quality-v2)",
        "type": "logical quality (HF transformers); NOT a physical serving benchmark",
        "models": {k: {x: m[x] for x in ("model_id", "revision", "manifest_sha256", "revision_provenance", "layers",
                                         "kv_heads", "head_dim", "token_pool_sha256")} for k, m in ident.MODELS.items()},
        "llama_identity_note": "Exp12 loaded model bytes byte-identical to ModelScope revision "
                               "359efdbb8af05b788a4ad4185215c6b8caa9052c, recovered by content matching of the preserved "
                               "historical cache (results/mlsys2027/canonical_quality_v2/llama_identity_audit: 18 / 18 "
                               "files); the accepted Exp12 evidence itself recorded only '@master' (no revision / "
                               "manifest) and the historical master branch pointer was not directly recovered; equality "
                               "with the Exp12 model is additionally checked by the BF16 control-reproduction gate",
        "dataset": {"name": "WikiText-2 test (raw text)", "url": ident.WIKITEXT_URL, "sha256": ident.WIKITEXT_SHA256,
                    "preprocessing": "non-empty stripped lines joined in blocks of 64; tokenizer(text, "
                                     "add_special_tokens=False) -- no BOS",
                    "windows": "the first 32 x (1024 + 128) tokens, consecutive non-overlapping windows, stream order "
                               "(identical to the accepted legacy protocol; windows 1-8 are the canonical Exp1 windows)"},
        "samples": ident.SAMPLES, "context_tokens": ident.CONTEXT_TOKENS, "eval_tokens": ident.EVAL_TOKENS,
        "scored_tokens_per_arm": ident.SAMPLES * ident.EVAL_TOKENS,
        "policy": "K3 / V2 / G32 / R4 / META8g64 (frozen; no retuning)",
        "rabit_implementation": {"file": "benchmarks/mlsys2027/canonical_rabit_quality.py", "commit": "c360697",
                                 "parity_evidence": "8fa9a9c (276 / 276 cases bit-exact, CPU)", "sole_implementation": True},
        "arms": {"bf16_batched": "CONTROL only: the legacy BF16 scorer verbatim (continuation[:-1] in one forward)",
                 "bf16": "BF16 cache; BF16 prefill; first token from the prefill logit; then one teacher-forced token "
                         "per forward",
                 "rabit": "identical loop and token ids; the cache is CanonicalRabitCache.from_prefill(BF16 prefill) and "
                          "ages canonically at every decode step (R4 residual -> open group -> closed 32-token page)"},
        "order": "arms bf16_batched, bf16, rabit; windows 1..32 inside each arm (the legacy method-major order)",
        "comparison": "bf16 (stepwise) vs rabit (stepwise) on identical paired token inputs",
        "model_settings": {"dtype": "bfloat16", "attention": "sdpa", "seed": "torch.manual_seed(0), cuda.manual_seed_all(0)",
                           "tf32": "enabled (matmul and cudnn)", "warmup": "one untimed 32-token BF16 forward"},
        "image": {"python": "3.11", "torch": "2.11.0", "transformers": "4.48.2",
                  "other": "accelerate, requests, sentencepiece, modelscope (versions recorded at run time)"},
        "hardware": {"selector": "H100!:1", "guard": "fail-closed: exactly 1 x NVIDIA H100 80GB before anything else"},
        "validity_gates": [
            "modal exit code 0, no wall-clock timeout, app cleanup verified",
            "hardware guard passed (1 x H100 80GB)",
            "model snapshot at the pinned revision equals the frozen manifest (every file: size + SHA-256; no extra file)",
            "WikiText SHA-256 and token-pool SHA-256 equal the pinned values",
            "shipped-file SHA-256 equal the committed harness; canonical_rabit_quality.py equals c360697",
            "GPU canonical state == CPU canonical state, bit-exact, every layer (window 1 prefill), before any scoring",
            "structure: 3 arms x 32 windows x 128 tokens; rabit rows use CanonicalRabitCache with 127 one-token forwards",
            "no legacy quality module imported in the container; only the four shipped files present",
            f"BF16 control reproduction: bf16_batched aggregate PPL within {CONTROL_REL_TOL:.1%} (relative) of the "
            "accepted legacy BF16 aggregate of the same model (Llama: Exp12 log; Qwen: Exp14 quality log)",
            f"stepwise BF16 aggregate PPL within {CONTROL_REL_TOL:.1%} (relative) of bf16_batched"],
        "quality_threshold": "NONE -- the result is reported, not gated; no acceptance threshold on the RABIT delta",
        "registered_attempt": "both models form ONE registered validation attempt, run Llama -> Qwen by "
                              "run_canonical_ppl_two_model.py with no human / model-dependent decision in between; "
                              "valid only if BOTH runs pass every validity gate; if either fails the whole attempt is "
                              "invalid and archived; no selective rerun",
        "interpretation_cases": {
            "note": "preregistered before GPU execution; descriptive only; no numerical success threshold",
            "case_1": {"condition": "Llama remains near its BF16 control while Qwen still shows severe degradation.",
                       "interpretation": "the frozen operating point shows model-specific quality sensitivity; "
                                         "mechanism remains undiagnosed."},
            "case_2": {"condition": "Both Llama and Qwen show substantial degradation relative to their BF16 controls.",
                       "interpretation": "the legacy logical evaluator materially understated canonical-RABIT quality "
                                         "loss, and the paper's quality story must be rebuilt."},
            "case_3": {"condition": "Both Llama and Qwen remain close to their BF16 controls.",
                       "interpretation": "the severe legacy Qwen degradation was primarily an artifact of the old "
                                         "logical evaluator semantics."},
            "case_4": {"condition": "Results are mixed or intermediate and do not cleanly fit Cases 1-3.",
                       "interpretation": "report the actual values directly without forcing a categorical narrative."}},
        "statistics": {"aggregate": "PPL = exp(total loss / total tokens) per arm",
                       "delta": "100 * (exp(mean ln PPL_rabit - mean ln PPL_bf16) - 1) over the 32 windows",
                       "ci": "paired percentile bootstrap over windows (Exp12 procedure and seed)", **BOOTSTRAP,
                       "inputs": "full-precision per-window mean token NLL from the result JSON (Exp12 used the "
                                 "4-decimal logged PPL)",
                       "legacy_context": "the accepted legacy-evaluator rabit2 values are quoted for context only; "
                                         "they are a different evaluator and are never pooled or compared as a gate"},
        "attempts": "each attempt writes results/mlsys2027/canonical_quality_v2/continuation_ppl/<model>/attempt_<n>/; "
                    "an existing attempt directory is never overwritten; failed attempts are archived, not pooled",
        "not_run_here": ["NIAH", "Passage Retrieval", "HotpotQA", "Qasper", "ablations", "retuning", "serving",
                         "kernel optimization"],
    }


def load_protocol() -> dict:
    committed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if committed != json.loads(json.dumps(build_protocol())):
        raise RuntimeError("canonical_ppl_protocol.json differs from the regenerated protocol")
    return committed


def preflight(model: str, attempt: int, execute: bool) -> dict:
    """No Modal / GPU / model. Every check must hold for --execute; --dry-run reports them."""
    t = subprocess.run([sys.executable, str(STATIC_TESTS)], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env(), timeout=900)
    proof = json.loads(PROOF_RECORD.read_text(encoding="utf-8")) if PROOF_RECORD.exists() else {}
    parity = json.loads((ROOT / "results/mlsys2027/quality_semantic_audit/parity_attempt_3/record.json").read_text("utf-8"))
    upstream = subprocess.run(["git", "rev-parse", "@{u}"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    checks = {
        "clean_tree": git("status", "--short") == "",
        "head_is_pushed": git("rev-parse", "HEAD") == upstream,
        "harness_committed": git("status", "--short", "--", *[rel(p) for p in HARNESS]) == ""
        and all(p.exists() for p in HARNESS),
        **{f"unchanged_{k}": git("diff", "--name-only", c, "--", *paths) == "" for k, (c, paths) in FROZEN.items()},
        "rabit_kv2_is_frozen_source": sha256_lf(RABIT_KV2) == RABIT_KV2_SHA256_LF,
        "parity_accepted": parity.get("passed") is True and parity.get("result_sha256") == PARITY_RESULT_SHA256,
        "static_harness_tests_pass": t.returncode == 0 and bool(re.search(r"^(\d+)/\1 passed$", t.stdout, re.M)),
        "offline_proofs_passed": proof.get("passed") is True and proof.get("observer") == PROOF_OBSERVER,
        "offline_proofs_bound_to_current_files": bool(proof) and all(
            sha256_lf(ROOT / f) == h for f, h in proof.get("bound_file_sha256_lf", {}).items()),
        "protocol_matches": PROTOCOL.exists() and load_protocol() is not None,
        "attempt_dir_absent": not (OUT_BASE / model / f"attempt_{attempt}").exists(),
    }
    if execute and not all(checks.values()):
        raise SystemExit(f"pre-run validation failed: {json.dumps(checks, indent=1)}\nstatic tests: {t.stdout[-1500:]}")
    return checks


# ------------------------------------------------------------------------------------------- evaluation (offline)
def evaluate(res: dict, model: str, ref: dict) -> dict:
    """Validity gates and statistics from the result JSON (pure; unit-tested)."""
    m = ident.MODELS[model]
    arms = res.get("arms", {})
    s, e = ident.SAMPLES, ident.EVAL_TOKENS

    def rows_ok(a):
        r = arms.get(a, [])
        return (len(r) == s and [x.get("window") for x in r] == list(range(1, s + 1))
                and all(x["tokens"] == e and len(x["token_nll"]) == e and math.isfinite(x["loss_sum"]) for x in r))

    gates = {
        "hardware": res["hardware"]["passed"] is True and ident.hardware_ok(res["hardware"]["gpus"]),
        "model_snapshot": res["model"]["passed"] is True and res["model"]["model_id"] == m["model_id"]
        and res["model"]["model_revision"] == m["revision"] and res["model"]["manifest_sha256"] == m["manifest_sha256"]
        and res["model"]["files_checked"] == len(m["files"]),
        "dataset": res["dataset"]["wikitext_sha256"] == ident.WIKITEXT_SHA256
        and res["dataset"]["token_pool_sha256"] == m["token_pool_sha256"],
        "shipped_files": res["files"]["passed"] is True
        and res["files"]["sha256_lf"] == {n: sha256_lf(HERE / n) for n in SHIPPED},
        "geometry_and_policy": [res["geometry"][k] for k in ("layers", "kv_heads", "head_dim")]
        == [m["layers"], m["kv_heads"], m["head_dim"]]
        and res["policy"] == {"k_bits": 3, "v_bits": 2, "group_size": 32, "residual_tokens": 4, "metadata_bits": 8,
                              "metadata_group_size": 64},
        "gpu_cpu_canonical_state_parity": res["prefill_state_parity"]["passed"] is True
        and res["prefill_state_parity"]["layers"] == m["layers"]
        and res["prefill_state_parity"]["tokens"] == ident.CONTEXT_TOKENS,
        "structure": list(arms) == ["bf16_batched", "bf16", "rabit"] and all(rows_ok(a) for a in arms)
        and all(x.get("cache_class") == "CanonicalRabitCache" and x.get("decode_forwards") == e - 1 for x in arms.get("rabit", []))
        and all(x.get("cache_class") == "DynamicCache" and x.get("decode_forwards") == e - 1 for x in arms.get("bf16", [])),
        "legacy_unreachable": res["legacy_unreachable"]["passed"] is True
        and res["legacy_unreachable"]["repo_files"] == sorted(f"benchmarks/mlsys2027/{n}" for n in SHIPPED),
    }
    out = {"gates": gates}
    if gates["structure"]:
        ln = {a: [x["loss_sum"] / x["tokens"] for x in arms[a]] for a in arms}  # per-window ln PPL
        ppl = {a: pb.aggregate("continuation_ppl", ln[a]) for a in arms}
        ctrl = abs(ppl["bf16_batched"] - ref["bf16_aggregate_ppl"]) / ref["bf16_aggregate_ppl"]
        step = abs(ppl["bf16"] - ppl["bf16_batched"]) / ppl["bf16_batched"]
        gates["bf16_control_reproduces_legacy"] = ctrl <= CONTROL_REL_TOL
        gates["bf16_stepwise_matches_batched"] = step <= CONTROL_REL_TOL
        out["control"] = {
            "legacy_bf16_aggregate_ppl": ref["bf16_aggregate_ppl"], "bf16_batched_aggregate_ppl": ppl["bf16_batched"],
            "relative_difference": ctrl, "tolerance": CONTROL_REL_TOL,
            "max_per_window_relative_difference_report_only": max(
                abs(math.exp(a) - b) / b for a, b in zip(ln["bf16_batched"], ref["bf16"])),
            "stepwise_vs_batched_relative_difference": step}
        out["statistics"] = {
            "n_windows": s, "scored_tokens_per_arm": s * e, "bf16_ppl": ppl["bf16"], "rabit_ppl": ppl["rabit"],
            "delta_pct": pb.delta("continuation_ppl", ln["bf16"], ln["rabit"]),
            **pb.paired_bootstrap("continuation_ppl", ln["bf16"], ln["rabit"], BOOTSTRAP["resamples"], BOOTSTRAP["seed"],
                                  BOOTSTRAP["confidence"]),
            "robustness": pb.robustness("continuation_ppl", list(range(1, s + 1)), ln["bf16"], ln["rabit"]),
            "per_window_ppl": {a: [math.exp(v) for v in ln[a]] for a in arms},
            "legacy_evaluator_context_only": {
                "rabit2_legacy_aggregate_ppl": ref["rabit2_legacy_aggregate_ppl"],
                "rabit2_legacy_delta_pct": 100.0 * (ref["rabit2_legacy_aggregate_ppl"] / ref["bf16_aggregate_ppl"] - 1.0),
                "note": "legacy logical evaluator (different V metadata order; no decode aging); never pooled"}}
    out["valid"] = all(gates.values()) and len(gates) == 10
    return out


def run(model: str, attempt: int) -> int:
    checks = preflight(model, attempt, execute=True)
    ref = legacy_reference(model)
    out = OUT_BASE / model / f"attempt_{attempt}"
    out.mkdir(parents=True, exist_ok=False)
    result_tmp = Path(tempfile.mkdtemp()) / "result.json"
    expected = {n: sha256_lf(HERE / n) for n in SHIPPED}
    e = env()
    e["CANONICAL_PPL_RESULT_PATH"], e["CANONICAL_PPL_EXPECTED_FILE_SHA256_LF"] = str(result_tmp), json.dumps(expected)
    rec = {"kind": "canonical-quality-v2 continuation PPL (logical quality; NOT physical serving evidence)",
           "model_key": model, "attempt": attempt, "source_commit": git("rev-parse", "HEAD"),
           "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "preflight": checks,
           "protocol_sha256_lf": sha256_lf(PROTOCOL), "harness_sha256_lf": {rel(p): sha256_lf(p) for p in HARNESS},
           "wall_clock_s": WALL_CLOCK_S}
    pre, t0, parsed = apps(), time.time(), set()
    with (out / "session.log").open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen([sys.executable, "-m", "modal", "run", str(MODAL_APP), "--model-key", model],
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
    res = ev = None
    if result_tmp.is_file():
        (out / "result.json").write_bytes(result_tmp.read_bytes())
        res = json.loads((out / "result.json").read_bytes().decode("utf-8"))
        ev = evaluate(res, model, ref)
    process_ok = rc == 0 and not timed_out and cleanup["verified"]
    rec.update(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_s=round(time.time() - t0, 1),
               modal_returncode=rc, timed_out=timed_out, app_ids_new=new, cleanup=cleanup, process_ok=process_ok,
               result_sha256=hashlib.sha256((out / "result.json").read_bytes()).hexdigest() if res else None,
               gpu=res["hardware"]["gpus"] if res else None, evaluation=ev,
               valid=bool(process_ok and ev and ev["valid"]),
               status="valid (awaiting review; not accepted evidence until reviewed)" if process_ok and ev and ev["valid"]
               else "INVALID attempt (archive; never pool)")
    (out / "record.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"model": model, "attempt": attempt, "valid": rec["valid"], "modal_returncode": rc,
                      "gates": ev["gates"] if ev else None,
                      "statistics": {k: ev["statistics"][k] for k in ("bf16_ppl", "rabit_ppl", "delta_pct", "ci_low", "ci_high")}
                      if ev and "statistics" in ev else None}, indent=1))
    return 0 if rec["valid"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(ident.MODELS))
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true", help="launch the GPU job (ONLY when explicitly authorized)")
    ap.add_argument("--write-protocol", action="store_true")
    a = ap.parse_args(argv)
    if a.write_protocol:
        if PROTOCOL.exists():
            raise SystemExit(f"{PROTOCOL.name} already exists; the frozen protocol is never overwritten")
        PROTOCOL.write_text(json.dumps(build_protocol(), indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {rel(PROTOCOL)}")
        return 0
    if not a.model or a.dry_run == a.execute:
        raise SystemExit("--model and exactly one of --dry-run / --execute are required")
    if a.dry_run:
        checks, ref = preflight(a.model, a.attempt, execute=False), legacy_reference(a.model)
        print(json.dumps({"model": a.model, "identity": {k: ident.MODELS[a.model][k] for k in (
            "model_id", "revision", "manifest_sha256", "token_pool_sha256")}, "preflight": checks,
            "all_preflight_checks_pass": all(checks.values()),
            "legacy_bf16_control_target_ppl": ref["bf16_aggregate_ppl"],
            "command": f"modal run {rel(MODAL_APP)} --model-key {a.model}"}, indent=1))
        print("\n--dry-run: nothing executed, no files written.")
        return 0
    return run(a.model, a.attempt)


if __name__ == "__main__":
    raise SystemExit(main())
