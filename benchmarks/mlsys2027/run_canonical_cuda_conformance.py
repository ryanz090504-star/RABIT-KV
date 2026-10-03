"""
Runner: canonical-quality-v2 GPU SEMANTIC-CONFORMANCE DIAGNOSTIC (descriptive; NOT a quality experiment; no scoring,
no PPL, no generation, no tolerance). Opened after the INVALID registered PPL Attempt 1 (3cba625), whose
"GPU canonical state == CPU canonical state, bit-exact" gate failed in all 32 Llama layers.

Question: does the canonical implementation ON CUDA equal the frozen independent oracle ON CUDA (bit-exact, every
canonical field), and where exactly do CPU and CUDA differ?

Classification (computed from the result; nothing is tuned):
  A  CUDA canonical == CUDA oracle bit-exactly: every field, all 32 Llama layers, both synthetic geometries
  B  any CUDA canonical vs CUDA oracle difference
  C  diagnostic invalid / incomplete

One invocation = one model. The accepted Llama diagnostic (ec80638) is cuda_conformance_diagnostic/attempt_1/; later
runs are written to cuda_conformance_diagnostic/<model>/attempt_<n>/.

Usage:
    python benchmarks/mlsys2027/run_canonical_cuda_conformance.py --dry-run --model qwen2_5_7b
    python benchmarks/mlsys2027/run_canonical_cuda_conformance.py --execute --model qwen2_5_7b --attempt 1
                                                                                  (ONLY when authorized)
"""

from __future__ import annotations

import argparse
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
import canonical_ppl_identity as ident  # noqa: E402  (stdlib only)
from run_canonical_quality_parity import poll_cleanup  # noqa: E402  (accepted cleanup polling; stdlib only)

APP_NAME = "rabit-kv-canonical-quality-v2-cuda-conformance"
MODAL_APP = HERE / "canonical_cuda_conformance_modal.py"
OUT_BASE = ROOT / "results/mlsys2027/canonical_quality_v2/cuda_conformance_diagnostic"
SHIPPED = ["benchmarks/mlsys2027/canonical_rabit_quality.py", "benchmarks/mlsys2027/canonical_ppl_core.py",
           "benchmarks/mlsys2027/canonical_ppl_identity.py", "benchmarks/mlsys2027/exp14_model_snapshot.py",
           "benchmarks/mlsys2027/canonical_cuda_conformance.py", "benchmarks/mlsys2027/canonical_quality_parity_tests.py",
           "vllm-kvquant/vllm/v1/attention/ops/kvquant_k3.py"]
STATIC_TESTS = HERE / "test_canonical_cuda_conformance.py"
HARNESS = [*[ROOT / f for f in SHIPPED], MODAL_APP, Path(__file__).resolve(), STATIC_TESTS]
FROZEN = {
    "canonical_impl_c360697": ("c36069781b259e2e9b4d8adf60b7c58ea5df5cac", ["benchmarks/mlsys2027/canonical_rabit_quality.py"]),
    "parity_suite_and_oracle_8fa9a9c": ("8fa9a9c399245c7db17e64b6b7f206c8f259d6a9",
                                        ["benchmarks/mlsys2027/canonical_quality_parity_tests.py",
                                         "vllm-kvquant/vllm/v1/attention/ops/kvquant_k3.py"]),
    "ppl_core_and_identity_1e5a7ff": ("1e5a7ff8f5c50cb7c85246dd1eb93b6de8418641",
                                      ["benchmarks/mlsys2027/canonical_ppl_core.py",
                                       "benchmarks/mlsys2027/canonical_ppl_identity.py",
                                       "benchmarks/mlsys2027/canonical_ppl_modal.py",
                                       "benchmarks/mlsys2027/canonical_ppl_protocol.json",
                                       "benchmarks/mlsys2027/exp14_model_snapshot.py"]),
    "llama_conformance_evidence_ec80638": ("ec80638", ["results/mlsys2027/canonical_quality_v2/cuda_conformance_diagnostic/attempt_1"]),
    "attempt_1_archive_3cba625": ("3cba625", ["results/mlsys2027/canonical_quality_v2/continuation_ppl/llama3_1_8b",
                                              "results/mlsys2027/canonical_quality_v2/continuation_ppl/two_model_attempt_1/record.json",
                                              "results/mlsys2027/canonical_quality_v2/continuation_ppl/two_model_attempt_1/attempt_record_addendum.json"]),
}
WALL_CLOCK_S = 3600
APP_RE = re.compile(r"ap-[A-Za-z0-9]{20,}")


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def env() -> dict:
    e = os.environ.copy()
    e["PYTHONUTF8"], e["PYTHONIOENCODING"] = "1", "utf-8"
    return e


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def apps() -> dict:
    p = subprocess.run([sys.executable, "-m", "modal", "app", "list", "--json"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env(), timeout=180)
    return {r["App ID"]: r for r in json.loads(p.stdout) if r.get("Description") == APP_NAME}


def preflight(model: str, attempt: int, execute: bool) -> dict:
    upstream = subprocess.run(["git", "rev-parse", "@{u}"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    t = subprocess.run([sys.executable, str(STATIC_TESTS)], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env(), timeout=900)
    checks = {
        "static_tests_pass": t.returncode == 0 and bool(re.search(r"^(\d+)/\1 passed$", t.stdout, re.M)),
        "clean_tree": git("status", "--short") == "",
        "head_is_pushed": git("rev-parse", "HEAD") == upstream,
        "harness_committed": all(p.exists() for p in HARNESS)
        and git("status", "--short", "--", *[p.relative_to(ROOT).as_posix() for p in HARNESS]) == "",
        **{f"unchanged_{k}": git("diff", "--name-only", c, "--", *paths) == "" for k, (c, paths) in FROZEN.items()},
        "attempt_dir_absent": not (OUT_BASE / model / f"attempt_{attempt}").exists(),
    }
    if execute and not all(checks.values()):
        raise SystemExit(f"pre-run validation failed: {json.dumps(checks, indent=1)}")
    return checks


# ------------------------------------------------------------------------------------------- evaluation (offline)
def evaluate(res: dict, model: str = "llama3_1_8b") -> dict:
    """Validity of the DIAGNOSTIC and the classification. Pure function of the result JSON."""
    m = ident.MODELS[model]
    layers = res.get("layers", [])
    validity = {
        "model_key": res.get("model_key", "llama3_1_8b") == model and res["model"]["model_id"] == m["model_id"]
        and [res["geometry"][k] for k in ("layers", "kv_heads", "head_dim")] == [m["layers"], m["kv_heads"], m["head_dim"]],
        "hardware": res["hardware"]["passed"] is True and ident.hardware_ok(res["hardware"]["gpus"]),
        "shipped_files": res["files"]["passed"] is True
        and res["files"]["sha256_lf"] == {f: sha256_lf(ROOT / f) for f in SHIPPED},
        "oracle_is_the_accepted_parity_oracle": res["oracle"]["equals_accepted_parity_oracle"] is True
        and res["oracle"]["oracle_namespace_references_canonical_module"] is False,
        "model_snapshot": res["model"]["passed"] is True and res["model"]["model_revision"] == m["revision"]
        and res["model"]["manifest_sha256"] == m["manifest_sha256"] and res["model"]["files_checked"] == len(m["files"]),
        "window": res["dataset"]["wikitext_sha256"] == ident.WIKITEXT_SHA256
        and res["dataset"]["token_pool_sha256"] == m["token_pool_sha256"] and res["window"]["context_tokens"] == 1024
        and res["window"]["equals_first_1024_pool_tokens"] is True and res["window"]["contains_bos"] is False,
        "all_layers_reported": [r["layer"] for r in layers] == list(range(m["layers"])),
        "raw_kv": all(r["raw"]["dtype"] == "torch.bfloat16" and r["raw"]["shape"] == [1024, m["kv_heads"], m["head_dim"]]
                      and r["raw"]["host_copy_bitwise_identical"] is True and r["raw"]["device"].startswith("cuda")
                      for r in layers),
        "trace_reproduces_frozen_code": all(r.get("trace_reproduces_frozen_code_on_both_devices") is True for r in layers),
        "synthetic_complete": sorted(res["synthetic"]["geometries"]) == ["llama3_1_8b", "qwen2_5_7b"]
        and all(g["cases"] == 60 and g["aging_cases"] == 28 for g in res["synthetic"]["geometries"].values())
        and [res["synthetic"]["geometries"][g][k] for g in ("llama3_1_8b", "qwen2_5_7b") for k in ("kv_heads", "head_dim")]
        == [8, 128, 4, 128],
        "no_scoring": res["no_scoring"]["continuation_tokens_scored"] == 0 and res["no_scoring"]["logits_read"] is False
        and res["no_scoring"]["generation"] is False,
    }
    out = {"validity": validity, "valid": all(validity.values())}
    if not out["valid"]:
        out["classification"] = "C"
        return out
    eq_bc = [r["summary"]["cuda_canonical_equals_cuda_oracle"] and r["summary"]["canonical_cache_equals_cuda_oracle"]
             and r["accepted_t1_comparison_on_cuda"] == [] and all(r["residual_equals_raw"].values()) for r in layers]
    syn = res["synthetic"]["geometries"]
    out["primary"] = {
        "model": model, "layers": len(layers), "layers_cuda_canonical_equals_cuda_oracle": sum(eq_bc),
        "layers_mismatched_cuda_canonical_vs_cuda_oracle": [r["layer"] for r, e in zip(layers, eq_bc) if not e],
        "synthetic": {g: {"cases": G["cases"], "aging_cases": G["aging_cases"], "passed": G["passed"],
                          "failures": len(G["field_failures"]) + len(G["t1_failures"]) + len(G["aging_failures"])}
                      for g, G in syn.items()}}
    out["descriptive"] = {
        "layers_cpu_canonical_equals_cpu_oracle": sum(r["summary"]["cpu_canonical_equals_cpu_oracle"] for r in layers),
        "layers_cpu_canonical_equals_cuda_canonical": sum(r["summary"]["cpu_canonical_equals_cuda_canonical"] for r in layers),
        "layers_cpu_oracle_equals_cuda_oracle": sum(r["summary"]["cpu_oracle_equals_cuda_oracle"] for r in layers),
        "attempt1_gate_reevaluated_passed": res["attempt1_gate_reevaluated"]["passed"],
        "attempt1_gate_reevaluated_mismatched_layers": len(res["attempt1_gate_reevaluated"]["mismatched_layers"])}
    out["classification"] = "A" if all(eq_bc) and all(G["passed"] for G in syn.values()) else "B"
    return out


def run(model: str, attempt: int) -> int:
    checks = preflight(model, attempt, execute=True)
    out = OUT_BASE / model / f"attempt_{attempt}"
    out.mkdir(parents=True, exist_ok=False)
    result_tmp = Path(tempfile.mkdtemp()) / "result.json"
    e = env()
    e["CANONICAL_CUDA_CONFORMANCE_RESULT_PATH"] = str(result_tmp)
    e["CANONICAL_CUDA_CONFORMANCE_EXPECTED_FILE_SHA256_LF"] = json.dumps({f: sha256_lf(ROOT / f) for f in SHIPPED})
    rec = {"kind": "canonical-quality-v2 GPU semantic-conformance diagnostic (descriptive; NOT a quality result)",
           "model_key": model, "attempt": attempt, "source_commit": git("rev-parse", "HEAD"),
           "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "preflight": checks,
           "harness_sha256_lf": {p.relative_to(ROOT).as_posix(): sha256_lf(p) for p in HARNESS}}
    pre, t0, parsed = apps(), time.time(), set()
    with (out / "session.log").open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen([sys.executable, "-m", "modal", "run", str(MODAL_APP), "--model-key", model], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=e, cwd=ROOT)

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
        ev = evaluate(res, model)
    process_ok = rc == 0 and not timed_out and cleanup["verified"]
    valid = bool(process_ok and ev and ev["valid"])
    rec.update(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_s=round(time.time() - t0, 1),
               modal_returncode=rc, timed_out=timed_out, app_ids_new=new, cleanup=cleanup, process_ok=process_ok,
               result_sha256=hashlib.sha256((out / "result.json").read_bytes()).hexdigest() if res else None,
               gpu=res["hardware"]["gpus"] if res else None, evaluation=ev, diagnostic_valid=valid,
               classification=ev["classification"] if valid else "C")
    (out / "record.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"model": model, "attempt": attempt, "diagnostic_valid": valid, "classification": rec["classification"],
                      "modal_returncode": rc, "validity": ev["validity"] if ev else None,
                      "primary": ev.get("primary") if ev else None, "descriptive": ev.get("descriptive") if ev else None},
                     indent=1))
    return 0 if valid else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(ident.MODELS), required=True)
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true", help="launch the GPU job (ONLY when explicitly authorized)")
    a = ap.parse_args(argv)
    if a.dry_run == a.execute:
        raise SystemExit("exactly one of --dry-run / --execute is required")
    if a.dry_run:
        checks = preflight(a.model, a.attempt, execute=False)
        print(json.dumps({"preflight": checks, "all_preflight_checks_pass": all(checks.values()),
                          "command": f"modal run {MODAL_APP.relative_to(ROOT).as_posix()} --model-key {a.model}"}, indent=1))
        print("\n--dry-run: nothing executed, no files written.")
        return 0
    return run(a.model, a.attempt)


if __name__ == "__main__":
    raise SystemExit(main())
