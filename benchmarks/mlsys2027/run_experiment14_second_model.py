"""
RABIT-KV MLSys 2027 -- Experiment 14 runner: does the FROZEN RABIT policy K3 / V2 / G32 / R4 / META8g64 generalize to a
second model with a different GQA geometry? Model B = Qwen2.5-7B-Instruct (28 layers, 28 query / 4 KV heads, GQA ratio
7, head_dim 128). No policy retuning, no RABIT / vLLM source change.

Three parts, each executed ONCE, in this order (each refuses to overwrite a passed manifest):
  --part probe    NON-EVIDENCE feasibility probe (gates + engine init + sanity generations, no timing); must pass
                  before either measured part may run;
  --part serving  PHYSICAL serving: one H100 container, RABIT gate + Model-B shape gate, then counterbalanced legs
                  A1 B1 B2 A2 (A = bfloat16, B = rabit_kv2), 5 warmups + 30 measured reps per leg; observed allocator
                  capacity (num_gpu_blocks x block_size) and single-request latency (Exp13 methodology);
  --part quality  LOGICAL fake-quant quality (HF transformers scripts, unchanged): the Exp12 larger-N unit selection,
                  bf16 vs rabit2, with the paired per-unit bootstrap (paired_bootstrap_ci.py, unchanged) and the
                  canonical (Exp1-count) subset aggregates.
Logical quality and physical serving are separate evaluations; neither validates the other.

Usage:
    python benchmarks/mlsys2027/run_experiment14_second_model.py --write-protocol   (once, before commit)
    python benchmarks/mlsys2027/run_experiment14_second_model.py --dry-run
    python benchmarks/mlsys2027/run_experiment14_second_model.py --part probe|serving|quality   (only when authorized)
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import urllib.request
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp14_model_snapshot as ms  # noqa: E402  (frozen Model-B identity)
import qa_control_gate as qg  # noqa: E402  (read-only: identity-line regexes)
import run_exp13_turboquant_probe as p1  # noqa: E402  (read-only: image-expression extractor)
import run_experiment1_quality_frontier as e1  # noqa: E402  (accepted; read-only helpers)
import run_experiment7_kbit_ablation as r7  # noqa: E402  (accepted; read-only)
import run_experiment9_group_ablation as r9  # noqa: E402  (accepted; read-only: storage model)
import run_experiment11_metadata_ablation as r11  # noqa: E402  (accepted; read-only)
import run_experiment12_variance as r12  # noqa: E402  (accepted; read-only: Exp12 selection)
import run_experiment13_external_baseline as r13  # noqa: E402  (accepted; read-only: parsing helpers)

pb = r12.pb
ROOT = e1.ROOT
RUNNER_SCRIPT = Path(__file__).resolve()
MODAL_APP = HERE / "exp14_deployment_modal.py"
WORKER = HERE / "exp14_engine_worker.py"
PROBE_WORKER = HERE / "exp14_probe_worker.py"
SHAPE_GATE = HERE / "exp14_shape_gate.py"
MODEL_SNAPSHOT = HERE / "exp14_model_snapshot.py"
MODEL_SCAN_APP = HERE / "exp14_model_snapshot_modal.py"
EXP13_WORKER = HERE / "exp13_engine_worker.py"
EXP4_MODAL = HERE / "exp4_deployment_modal.py"
PROTOCOL = HERE / "exp14_second_model_protocol.json"
TEST_FILE = HERE / "test_experiment14_second_model.py"
QUALITY_DIR = ROOT / "benchmarks" / "quality"
OUT_DIR = ROOT / "results" / "mlsys2027" / "second_model"
PROBE_DIR, SERVING_DIR, QUALITY_OUT = OUT_DIR / "feasibility_probe", OUT_DIR / "serving", OUT_DIR / "quality"
HARNESS_FILES = [RUNNER_SCRIPT, MODAL_APP, WORKER, PROBE_WORKER, SHAPE_GATE, MODEL_SNAPSHOT, MODEL_SCAN_APP, PROTOCOL,
                 TEST_FILE, r13.GATE, r13.WATCHDOG]

EXP13_EVIDENCE_COMMIT = "42c2799f4e7393c6270193a1c852af90eaf7d402"
EXP13_OUT = ROOT / "results" / "mlsys2027" / "external_baseline"
PROTECTED_PATHS = [*r13.PROTECTED_PATHS, EXP13_OUT, *r13.MUST_BE_COMMITTED, QUALITY_DIR, r12.REFERENCE_DIR]

MODEL_B = ms.MODEL_ID
MODEL_REVISION = ms.MODEL_REVISION  # immutable ModelScope commit, required by probe, serving and quality
GEOMETRY = {"architectures": ["Qwen2ForCausalLM"], "num_hidden_layers": 28, "num_attention_heads": 28,
            "num_kv_heads": 4, "head_dim": 128, "sliding_window": None,
            "rabit_gqa4_decode_engaged": False, "rabit_fast_append_engaged": False}
LAYERS, KV_HEADS, HEAD_DIM = 28, 4, 128
FROZEN_RABIT_POLICY = {"k_bits": 3, "v_bits": 2, "group_size": 32, "residual_tokens": 4, "metadata_group_size": 64}
EXPECTED_KV_QUANT_MODE = {"bfloat16": "NONE", "rabit_kv2": "RABIT_KV2"}
RABIT_PAGE_BYTES = 12416  # rabit2_page_layout(block 32, 4 KV heads, head_dim 128)

A, B = "bfloat16", "rabit_kv2"
CONDITIONS = [A, B]
SHORT = {A: "bf16", B: "rabit"}
LEGS = [(1, "A1", A), (2, "B1", B), (3, "B2", B), (4, "A2", A)]
PROBE_LEGS = [(1, "P1", A), (2, "P2", B)]
WARMUPS_PER_LEG, REPS_PER_LEG, SAMPLES_PER_CONDITION = 5, 30, 60
CONTEXT_TOKENS, OUTPUT_TOKENS, REQUESTED_BLOCK_SIZE = 2048, 32, 32
PROMPT_RULE = "filler x 2048 (no BOS token)"
THEORETICAL_BYTES_PER_TOKEN = {
    A: {"bytes": LAYERS * 2 * KV_HEADS * HEAD_DIM * 2, "formula": "28 layers x (K,V) x 4 heads x 128 x 2 B (BF16)"},
    B: {"bytes": LAYERS * RABIT_PAGE_BYTES // 32,
        "formula": "28 layers x 388 B (RABIT physical page 12,416 B per 32-token block per layer: K3 payload 6,144 + "
                   "V2 payload 4,096 + META8g64 K min/scale 2 x 544 + V min/scale 2 x 544)"},
}
DTYPE_INDUCED_ALLOWLIST = ["requested.kv_cache_dtype", "kv_dtype.requested_kv_cache_dtype",
                           "kv_dtype.engine_cache_dtype", "kv_dtype.resolved_kv_torch_dtype", "kv_dtype.kv_quant_mode"]
EXPECTED_BACKEND_EVIDENCE = {"AttentionBackendEnum.TRITON_ATTN"}
IMPLIED_BYTES_REL_TOL = r13.IMPLIED_BYTES_REL_TOL
CAPACITY_LABEL = "OBSERVED PHYSICAL vLLM allocator KV capacity (num_gpu_blocks x block_size) from the live engine"
QUALITY_METHODS = "bf16,rabit2"
STORAGE_TOL_MB = r12.STORAGE_TOL_MB

CLAIM_BOUNDARY = (
    "Generalization of the FROZEN RABIT policy (K3/V2/G32/R4/META8g64, no retuning) to ONE additional model, "
    "Qwen2.5-7B-Instruct (GQA 28:4, head_dim 128). Serving: observed physical allocator capacity and single-request "
    "latency in one H100 session (default V2 runner, eager, 2048-token prompt, 32 output tokens); no throughput claim. "
    "At this geometry the frozen RABIT source serves through its exact FALLBACK paths (GQA ratio 7 is not a multiple "
    "of 4; the final fast append and the compiled V2 quantizer require 8 KV heads x head_dim 128), so RABIT latency on "
    "Model B measures those untuned paths -- it is NOT a measurement of how the Llama-tuned kernels transfer. Quality: "
    "LOGICAL fake-quant evaluation (HF transformers), separate from physical serving; neither validates the other. "
    "Cross-model comparisons with Exp12 / Exp13 (Llama-3.1-8B) are descriptive context from different sessions, never "
    "pooled. No universality claim beyond these two models.")
ATTENTION_CONFORMANCE = {
    "kind": "fixed dtype-aware numerical conformance checks (NOT a rigorous worst-case floating-point theorem, NOT a "
            "universal error bound, NOT a proof of arbitrary-input Triton attention correctness)",
    "r": "UNROUNDED FP32 kernel-semantics attention output computed from the INDEPENDENT Rabit2OnlineStateRef state: "
         "closed pages dequantized in FP32; open-group K / V rounded to BF16 as the runtime does; residual tokens exact "
         "BF16; PyTorch FP32 GQA attention (_gqa_ref)",
    "y": "runtime BF16 decode output converted to FP32 for comparison",
    "u": "2^-8 (BF16 unit roundoff)", "floor": "0.01 * max|r|",
    "C1": "max|y - r| <= u * max|r|; if max|r| == 0, y must be exactly 0",
    "C2": "|y_i - r_i| / |r_i| <= u for every element with |r_i| >= 0.01 * max|r|",
    "applies_to": "identically to Qwen 28Q / 4KV / D128 and the Llama 32Q / 8KV / D128 positive control, every frozen "
                  "replay and checkpoint",
    "justification": ["the runtime attention output is stored in BF16",
                      "the independent reference represents the same intended numerical semantics in FP32",
                      "u = 2^-8 is the BF16 unit-roundoff scale",
                      "the normwise test C1 handles near-zero values",
                      "the pre-existing 1 % floor makes relative error meaningful away from zero"],
    "diagnostics_not_gating": ["old 5e-3 error vs the Attempt-1 FP32 references", "error vs the BF16-rounded reference",
                               "BF16 ULP histogram / max ULP", "C1 / C2 against the runtime-state materialization"]}
CRITERION_AMENDMENT = {
    "kind": "post-failure CORRECTNESS-CRITERION amendment",
    "replaces": "the shape gate's inherited fixed attention criterion: max |runtime - FP32 reference| < 5e-3 over the "
                "runtime-state and reference-state FP32 references (from the single 70-token case of "
                "test_rabit_kv2_stage3c.py)",
    "by": "attention_conformance C1 / C2 (above)",
    "criterion_diff": {"old": "e_rt < 5e-3 and e_ref < 5e-3 (FP32 references, FP32 open-group dequantization)",
                       "new": "C1 and C2 vs r (independent reference state, kernel semantics, unrounded FP32)"},
    "reason": ["feasibility-probe Attempt 1 failed the inherited fixed 5e-3 attention threshold at BOTH geometries "
               "(including the Llama positive control) on short-prefix replays",
               "the valid non-evidence numerical diagnostic (Attempt 4) showed that threshold can be violated by BF16 "
               "output rounding alone while page-byte and state correctness are exact",
               "the replacement criterion was fixed from BF16 numerical semantics (u = 2^-8) and the pre-existing 1 % "
               "floor, not from observed maxima",
               "no capacity, latency, quality or model-engine result had been exposed by the failed Exp14 probe"],
    "old_protocol_sha256": "bea4003ebcb75d54ede04eea2eea4c3ba26f341743d4c6a89e3ae0a069f0b43a",
    "new_protocol_sha256": "recorded in results/mlsys2027/second_model/criterion_amendment_record.json "
                           "(a protocol cannot contain its own hash)",
    "numerical_diagnostic_commit": "4947050bf42d1f1d72ce1096f9c473ea8b2be9f8",
    "offline_evaluation_commit": "f909bc7ad814efd1c76119d90587c2c2c593cece",
    "offline_evaluation": "254 total checkpoints (127 Llama + 127 Qwen): C1 and C2 pass at every checkpoint",
    "failed_probe_attempt_1_archive_commit": "28d83ba1e82582e07d8b4ce0f6fe0cf22ddb2923",
    "excluded": ["probe Attempt 1", "numerical-diagnostic Attempts 1-3"],
    "unchanged": ["exact checks (page bytes, state, token counts, page transitions, dispatch)", "seeds / replays / "
                  "prefixes / geometries", "model / revision", "policy", "RABIT / vLLM source", "serving and quality "
                  "protocols"]}
EXP13_CORRECTNESS_WORDING = (
    "The upstream TurboQuant suite passed under the frozen item-level gate. The exact turboquant_k3v4_nc configuration "
    "was separately validated in the physical feasibility probe for cache initialization, backend routing and sanity "
    "generation.")

TAG = re.compile(r"^(EXP14P?_[A-Z_]+)=(\{.*\}|\[.*\]|null|true|false|-?\d+(?:\.\d+)?|\".*\")\s*$")
ROWLINE = re.compile(r"^(EXP14_SAMPLE|EXP14_WARMUP) (\{.*\})\s*$")
SHAPE_SUMMARY = re.compile(r"^EXP14_SHAPE_GATE_SUMMARY=(\{.*\})\s*$")
JIT_WARNING = "JIT compilation during inference"


# ------------------------------------------------------------------------------------ logical storage model
def quantized_parts_geom(tokens: int, bits: int, side: str, config: dict, heads: int, dim: int) -> tuple[int, int]:
    """r9._quantized_parts with the KV-head count and head_dim as parameters (identical formula)."""
    style, g = config[f"{side}_style"], int(config[f"{side}_group"])
    if style in ("group_sym", "group_affine"):
        dim_pad = math.ceil(dim / g) * g
        codes, meta_elems = heads * tokens * dim_pad, heads * tokens * (dim_pad // g)
        tensors = 1 if style == "group_sym" else 2
    elif style == "seq_affine":
        seq_pad = math.ceil(tokens / g) * g
        codes, meta_elems, tensors = heads * seq_pad * dim, heads * (seq_pad // g) * dim, 2
    else:
        raise ValueError(style)
    return (codes * bits + 7) // 8, tensors * r9._metadata_bytes(meta_elems, config)


def traced_logical_bytes_geom(prefix_tokens: int, config: dict, layers: int = LAYERS, heads: int = KV_HEADS,
                              dim: int = HEAD_DIM) -> dict:
    """r9.traced_logical_bytes with the model geometry as parameters (identical formula; cross-checked by test)."""
    out = {"k_payload": 0, "k_meta": 0, "v_payload": 0, "v_meta": 0, "residual": 0}
    bf16_token = heads * dim * 2
    for side in ("k", "v"):
        residual = min(int(config.get("residual", 0)), prefix_tokens)
        if residual > 0 and prefix_tokens <= residual:
            out["residual"] += prefix_tokens * bf16_token
            continue
        payload, meta = quantized_parts_geom(prefix_tokens - max(residual, 0), config[f"{side}_bits"], side, config,
                                             heads, dim)
        out[f"{side}_payload"] += payload
        out[f"{side}_meta"] += meta
        out["residual"] += max(residual, 0) * bf16_token
    out = {k: layers * v for k, v in out.items()}
    out["total"] = sum(out.values())
    return out


def expected_kv_mb(method: str, tokens: int, config: dict) -> float:
    b = LAYERS * 2 * tokens * KV_HEADS * HEAD_DIM * 2 if method == "bf16" else traced_logical_bytes_geom(tokens, config)["total"]
    return b / 2**20


# ------------------------------------------------------------------------------------------------ protocol
def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def verify_equivalence() -> dict:
    """The Exp14 worker / image are the accepted Exp13 / Exp4 ones except for the documented differences."""
    w14, w13 = _module(WORKER), _module(EXP13_WORKER)
    src = WORKER.read_text(encoding="utf-8")
    return {
        "image_expression_identical_to_exp4": p1.image_expr(MODAL_APP) == p1.image_expr(EXP4_MODAL),
        "base_engine_kwargs_identical_to_exp13": r13._const(w14, "BASE_ENGINE_KWARGS") == r13._const(w13, "BASE_ENGINE_KWARGS"),
        "context_output_tokens_identical": all(r13._const(w14, k) == r13._const(w13, k)
                                               for k in ("CONTEXT_TOKENS", "OUTPUT_TOKENS")),
        "timed_region_identical_to_exp13": ast.dump(r13._fn(r13._fn(w14, "main"), "one")) ==
                                           ast.dump(r13._fn(r13._fn(w13, "main"), "one")),
        "allowed_dtypes": r13._const(w14, "ALLOWED_KV_CACHE_DTYPES") == (A, B),
        "backend_pin_never_removed": 'kwargs.pop("attention_config")' not in src,
    }


def legs_arg(legs=LEGS) -> str:
    return ",".join(f"{label}={d}" for _, label, d in legs)


def build_protocol() -> dict:
    rabit2 = r12.canonical_rabit2()
    return {
        "experiment": 14,
        "type": "generalization of the FROZEN RABIT policy to a second model: physical serving (capacity + single-"
                "request latency) and logical fake-quant quality, reported separately",
        "plan": "docs/MLSYS_EXPERIMENT_PLAN.md Experiment 14 (one additional compatible model); the plan defers the "
                "model choice to a compatibility check",
        "model_b": {"id": MODEL_B, "revision": MODEL_REVISION,
                    "revision_resolution": "git ls-remote of the ModelScope repository during preparation: "
                                           "refs/heads/master = " + MODEL_REVISION + " (no tags exist); the file API "
                                           "returns the identical manifest for Revision=<commit> and Revision=master",
                    "manifest_sha256": ms.MODEL_MANIFEST_SHA256,
                    "manifest_rule": "sha256 of sorted 'path<TAB>size<TAB>sha256' lines of all 15 files",
                    "files": ms.FROZEN_FILES,
                    "key_file_sha256": {f[0]: f[2] for f in ms.FROZEN_FILES if f[0] in (
                        "config.json", "tokenizer_config.json", "tokenizer.json", "generation_config.json",
                        "model.safetensors.index.json")},
                    "enforcement": {
                        "probe_and_serving": "snapshot_download(model, revision=<commit>) in the container, then every "
                                             "frozen file verified (size + sha256, no extra file) before any engine "
                                             "starts; the session records revision, manifest hash and verification",
                        "quality": "the canonical scripts (unmodifiable) download without a revision: before the runs "
                                   "ModelScope master must resolve to <commit> with the identical file manifest; after "
                                   "the runs master must still resolve to <commit> and EVERY Qwen2.5-7B-Instruct "
                                   "snapshot directory in the model volume must match the frozen manifest "
                                   "(exp14_model_snapshot_modal.py, CPU only); otherwise the quality part is invalid",
                        "binding": "probe, serving and quality must all record the same revision and manifest hash"},
                    "geometry": GEOMETRY,
                    "context": "max_position_embeddings 32768, rope_theta 1e6, use_sliding_window false, BF16 weights",
                    "tokenizer": "no BOS token (bos_token null, add_bos_token false); eos <|im_end|>",
                    "selection": "user decision after the compatibility audit: tests a different GQA geometry (the "
                                 "plan's question) with no source change; Llama-3.1-8B is 32:8 (ratio 4)"},
        "policy": {"frozen": "K3/V2/G32/R4/META8g64", "serving_kv_cache_dtype": B,
                   "quality_config": rabit2, "retuning": "none (Exp14 tests the frozen policy, not a new search)"},
        "compatibility_audit": {
            "source_change_required": False,
            "layout": "rabit2_page_layout and Rabit2OnlineStateRef invariants hold: head_dim 128 % 64 == 0; "
                      "4 x 128 = 512 K primaries and 32 x 4 x 4 = 512 V primaries per page are multiples of 64",
            "backend_guards": "no sliding window / softcap / ALiBi / sinks (Qwen2.5 sliding_window resolves to None)",
            "dispatch_at_model_b": {"decode_attention": "Stage4B3 GQA4 entry -> exact Stage4B2 fallback (ratio 7)",
                                    "one_token_append": "final fast append -> _rabit2_final_old_append (needs 8 x 128)",
                                    "v2_quantizer": "compiled (1, 8, 128) path -> exact reference quantizer",
                                    "prefill": "rabit2_bulk_append_exact (shape-general) + dense context attention"},
            "vllm_support": "Qwen2ForCausalLM is registered in the vendored snapshot"},
        "snapshot": {"vllm_kvquant": "the committed tree (unchanged since backport commit " + r13.BACKPORT_COMMIT + ")",
                     "rabit_kv2_sha256_lf": r13.EXPECTED_RABIT_SHA256_LF, "no_source_change": True},
        "gates": {
            "rabit_gate": "exp3_correctness_gate.py, unchanged (Llama shape)",
            "shape_gate": {"script": SHAPE_GATE.relative_to(ROOT).as_posix(),
                           "geometries": {"model_b": "28 / 4 / 128", "positive_control": "32 / 8 / 128 (Llama)"},
                           "replays": "exact serving call sequence: 2048-token bulk prefill + 31 decode steps (2 seeds) "
                                      "and boundary prefills 1, 4, 5, 35, 36, 37, 68 + 8 decode steps",
                           "criteria": ["page bytes / alignment invariants", "closed pages byte-identical to "
                                        "Rabit2OnlineStateRef", "token counts and page transitions identical",
                                        "attention: fixed dtype-aware numerical conformance checks C1 and C2 (see "
                                        "attention_conformance)", "fallback call counts match the predicates"],
                           "exact_checks_dominate": "any exact-invariant failure fails the gate regardless of C1 / C2",
                           "attention_conformance": ATTENTION_CONFORMANCE},
            "order": "probe, serving and quality each run their gates / validity checks before any interpretation"},
        "amendments": [CRITERION_AMENDMENT],
        "feasibility_probe": {
            "status": "NON-EVIDENCE; requires explicit approval; must pass before --part serving / quality",
            "runs": "same image, RABIT gate, shape gate, model provenance, then one fresh engine per dtype "
                    "(bfloat16, rabit_kv2): capacity, geometry, backend, physical-layout consistency, one "
                    "2048-token / 32-token generation, one chat sanity generation; NO timing emitted",
            "output": PROBE_DIR.relative_to(ROOT).as_posix(),
            "binding": "the measured parts run only if a PASSED probe record matches the current harness hashes, "
                       "protocol hash, vllm-kvquant tree and model revision / manifest hash "
                       "(check_probe_prerequisite; tested offline)"},
        "serving": {
            "engine": {"worker_base_engine_kwargs": r13._const(_module(WORKER), "BASE_ENGINE_KWARGS"),
                       "model_runner": "default V2; 'Using V2 Model Runner' required in every leg",
                       "process_topology": "vLLM default multiprocess engine core (Exp4 / Exp13 topology)"},
            "workload": {"prompt": PROMPT_RULE + " (Model B has no BOS; the Exp13 [BOS] + ' the' x 2047 rule is "
                                                "unchanged for tokenizers with BOS)",
                         "context_tokens": CONTEXT_TOKENS, "output_tokens": OUTPUT_TOKENS, "decoding": "greedy",
                         "ignore_eos": True, "requests": "single request per generate call"},
            "session": {"gpu": "one NVIDIA H100 80GB HBM3 (recorded; must be H100 80GB)",
                        "legs": [{"index": k, "label": label, "kv_cache_dtype": d} for k, label, d in LEGS],
                        "order": "A B B A (Exp3 counterbalancing)", "warmups_per_leg": WARMUPS_PER_LEG,
                        "reps_per_leg": REPS_PER_LEG, "samples_per_condition": SAMPLES_PER_CONDITION,
                        "fresh_engine_process_per_leg": True,
                        "failure_rule": "any gate / leg failure, watchdog timeout or unclean GPU aborts; no retry"},
            "capacity": {"primary": CAPACITY_LABEL, "ratio": "rabit / bf16",
                         "secondary_labelled_theoretical": THEORETICAL_BYTES_PER_TOKEN,
                         "implied_bytes_per_token": "reported KV GiB x 2^30 / observed capacity (derived; layout "
                                                    "consistency check within 1 %)"},
            "latency": {"per_leg": list(r13.LEG_STATS), "cross_leg": "median of the two leg-level values",
                        "secondary": "pooled 60-sample median / p90 TPOT (descriptive; not 60 independent replicates)",
                        "drift": "leg 2 - leg 1 per condition (TPOT abs / %, p90 TPOT, TTFT, wall)",
                        "comparison": "RABIT vs BF16 on the cross-leg summaries", "no_inference": True,
                        "jit_warnings": "Triton JIT-during-inference warnings inside measured samples are counted and "
                                        "reported (descriptive, as in the Exp13 acceptance record)"},
            "integrity_gates": ["RABIT gate and shape gate pass", "4 legs exit 0, no watchdog timeout, GPU clean before "
                                "each", "every leg: V2 runner, block_size 32, 5 warmups + 30 samples, prompt 2048 / "
                                "output 32 in every sample, prompt rule '" + PROMPT_RULE + "'",
                                "every leg: served geometry == the frozen Model-B geometry (incl. sliding_window None "
                                "and the two fallback predicates false)", "every leg: backend TRITON_ATTN",
                                "every leg: implied bytes / token within 1 % of the theoretical layout",
                                "RABIT legs: frozen markers present", "identical capacity and config within a "
                                "condition; only dtype-induced fields differ across conditions"],
            "dtype_induced_allowlist": DTYPE_INDUCED_ALLOWLIST},
        "quality": {
            "kind": "LOGICAL fake-quant (HF transformers, unchanged benchmarks/quality/*.py; KV MB is logical)",
            "methods": QUALITY_METHODS, "model_arg": ["--model-name", MODEL_B],
            "selection": {b: {"units": r12.SELECTION[b]["exp12_units"], "canonical_subset": r12.SELECTION[b]["canonical_units"],
                              "args": r12.SELECTION[b]["args"]} for b in pb.BENCHMARKS},
            "why_exp12_selection": "the accepted Exp12 methodology (superset of the Exp1 canonical units) gives both "
                                   "Exp1-count subset aggregates and paired bootstrap CIs in one run",
            "statistics": "paired_bootstrap_ci.py unchanged (10,000 paired resamples; 95 % percentile CI); canonical "
                          "subset aggregates on the canonical unit keys",
            "validity": ["exit 0, no traceback", "log names Model B", "summary counts exact",
                         "per-unit rows paired, count == Exp12 units",
                         "unit identity equals the accepted Exp12 log (unit keys, dataset header, ground truths)",
                         "every per-unit logical KV MB equals the Model-B geometry accounting within 0.001 MB"],
            "not_applicable": "the Exp10 per-example QA control gate and the Exp12 control-reproduction targets are "
                              "Llama reference values; no Model-B reference exists, so the within-run bf16 is the "
                              "control"},
        "fairness": {
            "MATCHED": ["GPU class / container / session per part", "model weights and tokenizer across conditions",
                        "vLLM snapshot", "engine settings (block 32, gpu_memory_utilization 0.82, eager, prefix "
                        "caching off, chunked prefill on, max_model_len 32768)", "prompt / output length", "warmups",
                        "measurement code (timed region AST-identical to Exp13 / Exp4)", "counterbalanced order",
                        "quality units, prompts and scoring across bf16 / rabit2"],
            "METHOD_INHERENT": ["KV representation, metadata and R4 residual"],
            "LIMITATIONS": ["RABIT serves through untuned exact fallback kernels at Model B's geometry (BF16 uses the "
                            "standard TRITON_ATTN path)", "prompt has no BOS token (Model B convention), unlike Exp13",
                            "Model B differs from Llama in size, tokenizer and training; cross-model numbers are "
                            "descriptive context only", "one additional model; no universality claim",
                            "no Model-B quality reference: no control-reproduction gate"]},
        "claim_boundary": CLAIM_BOUNDARY,
        "exp13_correctness_wording_for_paper": EXP13_CORRECTNESS_WORDING,
        "outputs": {"probe": PROBE_DIR.relative_to(ROOT).as_posix(), "serving": SERVING_DIR.relative_to(ROOT).as_posix(),
                    "quality": QUALITY_OUT.relative_to(ROOT).as_posix()},
    }


def load_protocol() -> dict:
    committed = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if committed != json.loads(json.dumps(build_protocol())):
        raise RuntimeError("exp14_second_model_protocol.json differs from the regenerated protocol")
    return committed


# ------------------------------------------------------------------------------------------------- preflight
def protected_status() -> str:
    return e1.run_git("status", "--short", "--", *[str(p.relative_to(ROOT)) for p in PROTECTED_PATHS])


def _manifest_passed(path: Path) -> bool:
    return path.exists() and json.loads(path.read_text(encoding="utf-8")).get("status") == "passed"


PROBE_BINDING_KEYS = ("harness_sha256", "protocol_sha256", "vllm_kvquant_tree", "model_revision",
                      "model_manifest_sha256")


PROBE_RECORD = PROBE_DIR / "probe_record.json"


def load_probe_record() -> dict | None:
    return json.loads(PROBE_RECORD.read_text(encoding="utf-8")) if PROBE_RECORD.exists() else None


def check_probe_prerequisite(record: dict | None, prov: dict) -> None:
    """--part serving / quality may run only after a PASSED non-evidence probe whose harness hashes, protocol hash,
    vllm-kvquant tree and frozen model revision / manifest hash equal the current ones. Raises RuntimeError."""
    if not record:
        raise RuntimeError("no Exp14 feasibility probe record: the probe must pass before serving / quality")
    if record.get("status") != "passed" or record.get("non_evidence") is not True:
        raise RuntimeError(f"the Exp14 feasibility probe has not passed (status={record.get('status')!r})")
    for k in PROBE_BINDING_KEYS:
        if record.get(k) != prov.get(k):
            raise RuntimeError(f"Exp14 probe record does not match the current state: {k}")


def preflight(part: str | None, dry_run: bool) -> dict:
    status = protected_status()
    if status:
        raise RuntimeError("protected paths are not clean:\n" + status)
    for commit, path in ((EXP13_EVIDENCE_COMMIT, EXP13_OUT), (r13.EXP12_EVIDENCE_COMMIT, r12.OUT_DIR)):
        if e1.run_git("diff", "--name-only", commit, "--", path.relative_to(ROOT).as_posix()):
            raise RuntimeError(f"accepted evidence differs from {commit[:7]}: {path}")
    if e1.run_git("diff", "--name-only", "599d059cc3cad96f8cdf3c4f813f5460e5b35654", "--",
                  QUALITY_DIR.relative_to(ROOT).as_posix(), r12.REFERENCE_DIR.relative_to(ROOT).as_posix()):
        raise RuntimeError("canonical quality scripts / reference differ from the accepted state")
    if e1.run_git("diff", "--name-only", r13.BACKPORT_COMMIT, "HEAD", "--", "vllm-kvquant"):
        raise RuntimeError("vllm-kvquant changed after the backport commit")
    rabit = hashlib.sha256((ROOT / "vllm-kvquant/vllm/v1/attention/ops/rabit_kv2.py").read_bytes()
                           .replace(b"\r\n", b"\n")).hexdigest()
    if rabit != r13.EXPECTED_RABIT_SHA256_LF:
        raise RuntimeError("rabit_kv2.py is not the frozen source")
    for b in pb.BENCHMARKS:
        text = (QUALITY_DIR / f"{b}.py").read_text(encoding="utf-8")
        if e1.REQUIRED_ALLOWED_LINE not in text or e1.REQUIRED_RABIT2_MARKER not in text:
            raise RuntimeError(f"canonical {b}.py no longer matches the accepted Experiment 1 pins")
    eq = verify_equivalence()
    if not all(eq.values()):
        raise RuntimeError(f"Exp14 harness is not equivalent to the accepted Exp13 / Exp4 harness: {eq}")
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") is not None:
        raise RuntimeError("VLLM_USE_V2_MODEL_RUNNER must be unset (default V2 runner)")
    protocol = load_protocol()
    head = e1.run_git("rev-parse", "HEAD")
    prov = {"git_head": head, "vllm_kvquant_tree": e1.run_git("rev-parse", "HEAD:vllm-kvquant"),
            "rabit_kv2_sha256_lf": rabit, "equivalence": eq, "protocol_sha256": e1.sha256(PROTOCOL),
            "harness_sha256": {p.name: e1.sha256(p) for p in HARNESS_FILES}, "model_revision": MODEL_REVISION,
            "model_manifest_sha256": ms.MODEL_MANIFEST_SHA256}
    probe_ok = None
    if part in ("serving", "quality") and not dry_run:
        record = load_probe_record()
        check_probe_prerequisite(record, prov)
        e1.run_git("merge-base", "--is-ancestor", record["git_head"], "HEAD")
        probe_ok = record["git_head"]
    uncommitted = e1.run_git("status", "--short", "--", *[str(p.relative_to(ROOT)) for p in HARNESS_FILES])
    if uncommitted and not dry_run:
        raise RuntimeError("Refusing to run: Experiment 14 harness has uncommitted changes:\n" + uncommitted)
    if part:
        m = {"probe": PROBE_DIR / "probe_record.json", "serving": SERVING_DIR / "manifest.json",
             "quality": QUALITY_OUT / "manifest.json"}[part]
        if _manifest_passed(m):
            raise RuntimeError(f"{m} already records a passed run; refusing to overwrite")
    return {**prov, "probe_git_head": probe_ok, "protocol": protocol, "uncommitted_files": uncommitted or None}


# ---------------------------------------------------------------------------------------------- serving parse
def demux(session_text: str, legs=LEGS) -> tuple[dict, list[str], list[str], list[str]]:
    out = {k: [] for k, _, _ in legs}
    gate, shape, top = [], [], []
    prefixes = {f"[leg{k}:{d}] ": k for k, _, d in legs}
    for line in session_text.splitlines():
        if line.startswith("[gate] "):
            gate.append(line[len("[gate] "):])
        elif line.startswith("[shapegate] "):
            shape.append(line[len("[shapegate] "):])
        else:
            for prefix, k in prefixes.items():
                if line.startswith(prefix):
                    out[k].append(line[len(prefix):])
                    break
            else:
                top.append(line)
    return out, gate, shape, top


def parse_worker(lines: list[str]) -> dict:
    out: dict = {"tags": {}, "samples": [], "warmups": [], "kv_log_tokens": None, "kv_log_gib": None,
                 "v2_runner": False, "backend_evidence": [], "jit_warnings_in_measurement": 0}
    measuring = False
    for line in lines:
        s = line.strip()
        if s == "EXP14_MEASUREMENT_BEGIN":
            measuring = True
        elif s == "EXP14_MEASUREMENT_END":
            measuring = False
        if JIT_WARNING in line and measuring:
            out["jit_warnings_in_measurement"] += 1
        m = ROWLINE.match(s)
        if m:
            (out["samples"] if m.group(1) == "EXP14_SAMPLE" else out["warmups"]).append(json.loads(m.group(2)))
            continue
        m = TAG.match(s)
        if m:
            out["tags"].setdefault(m.group(1), json.loads(m.group(2)))
            continue
        if r13.V2_RUNNER in line:
            out["v2_runner"] = True
        for m in r13.BACKEND_LINE.finditer(line):
            ev = m.group(1) or m.group(2) or (f"FA{m.group(3)}" if m.group(3) else None)
            if ev and ev not in out["backend_evidence"]:
                out["backend_evidence"].append(ev)
        m = r13.KV_TOKENS.search(line)
        if m:
            out["kv_log_tokens"] = int(m.group(1).replace(",", ""))
        m = r13.KV_MEM.search(line)
        if m:
            out["kv_log_gib"] = float(m.group(1))
    return out


def parse_gates(gate: list[str], shape: list[str]) -> dict:
    res = next((json.loads(m.group(1)) for ln in gate if (m := r13.GATE_RESULT.match(ln.strip()))), None)
    sg = next((json.loads(m.group(1)) for ln in reversed(shape) if (m := SHAPE_SUMMARY.match(ln.strip()))), None)
    return {"rabit_gate_result": res, "rabit_gate_passed": bool(res and res.get("passed")),
            "shape_gate_summary": sg, "shape_gate_passed": bool(sg and sg.get("passed") is True and
                                                                 set(sg.get("geometries", {})) ==
                                                                 {"model_b_qwen2_5_7b", "control_llama3_1_8b"})}


def parse_top(lines: list[str]) -> dict:
    out: dict = {"leg_exit": {}, "pre_leg": {}, "gate_exit": None, "shape_gate_exit": None, "timeouts": [],
                 "complete": False, "environment": None, "model": None}
    for line in lines:
        s = line.strip()
        if s == "EXP14_MIRRORED_COMPLETE":
            out["complete"] = True
        m = TAG.match(s)
        if not m:
            continue
        tag, p = m.group(1), json.loads(m.group(2))
        if tag == "EXP14_LEG_EXIT":
            out["leg_exit"][p["leg"]] = p["returncode"]
        elif tag == "EXP14_PRE_LEG_GPU_STATE":
            out["pre_leg"][p["leg"]] = p.get("clean")
        elif tag == "EXP14_GATE_EXIT":
            out["gate_exit"] = p["returncode"]
        elif tag == "EXP14_SHAPE_GATE_EXIT":
            out["shape_gate_exit"] = p["returncode"]
        elif tag == "EXP14_WATCHDOG_TIMEOUT":
            out["timeouts"].append(p)
        elif tag == "EXP14_ENVIRONMENT":
            out["environment"] = p
        elif tag == "EXP14_MODEL":
            out["model"] = p
    return out


def model_checks(top: dict) -> dict:
    m = top.get("model") or {}
    v = m.get("verification") or {}
    return {"model_is_model_b": m.get("model") == MODEL_B,
            "model_revision_frozen": m.get("revision") == MODEL_REVISION and v.get("model_revision") == MODEL_REVISION,
            "model_snapshot_verified": v.get("passed") is True and v.get("files_checked") == len(ms.FROZEN_FILES)
                                       and m.get("manifest_sha256") == ms.MODEL_MANIFEST_SHA256}


def capacity(tags: dict, tag: str = "EXP14_CAPACITY") -> int | None:
    c = tags.get(tag)
    if not c or c.get("num_gpu_blocks") is None or c.get("block_size") is None:
        return None
    if c["capacity_tokens"] != c["num_gpu_blocks"] * c["block_size"]:
        return None
    return c["capacity_tokens"]


def implied_bytes_per_token(cap: int | None, gib: float | None) -> float | None:
    return gib * 2**30 / cap if cap and gib else None


def leg_config(p: dict) -> dict:
    flat = {}
    for section, prefix in (("EXP14_REQUESTED_ENGINE_KWARGS", "requested"), ("EXP14_EFFECTIVE_ENGINE_CONFIG", "effective"),
                            ("EXP14_WORKLOAD", "workload"), ("EXP14_KV_DTYPE", "kv_dtype"),
                            ("EXP14_MODEL_GEOMETRY", "geometry")):
        for k, v in (p["tags"].get(section) or {}).items():
            if (prefix, k) in (("requested", "model"), ("effective", "model")):
                continue
            flat[f"{prefix}.{k}"] = v
    return flat


def config_diff(parsed: dict) -> dict:
    configs = {label: leg_config(parsed[k]) for k, label, _ in LEGS}
    dtype_of = {label: d for _, label, d in LEGS}
    violations = []
    for key in sorted(set().union(*configs.values())):
        vals = {label: json.dumps(cfg.get(key), sort_keys=True) for label, cfg in configs.items()}
        if len(set(vals.values())) == 1:
            continue
        if key not in DTYPE_INDUCED_ALLOWLIST:
            violations.append({"field": key, "reason": "non-dtype field differs between legs", "values": vals})
            continue
        for d in CONDITIONS:
            if len({vals[label] for label in vals if dtype_of[label] == d}) != 1:
                violations.append({"field": key, "reason": "differs between the two legs of one condition", "condition": d})
    return {"violations": violations, "passed": not violations}


def serving_integrity(parsed: dict, gates: dict, top: dict, diff: dict) -> dict:
    checks = {"rabit_gate_passed": gates["rabit_gate_passed"] and top["gate_exit"] == 0,
              "shape_gate_passed": gates["shape_gate_passed"] and top["shape_gate_exit"] == 0,
              "no_watchdog_timeout": not top["timeouts"], "session_complete": top["complete"],
              "gpu_is_h100_80gb": any("H100 80GB" in g.get("name", "") for g in ((top.get("environment") or {}).get("gpus") or [])),
              **model_checks(top)}
    for k, label, d in LEGS:
        p = parsed[k]
        eff = p["tags"].get("EXP14_EFFECTIVE_ENGINE_CONFIG") or {}
        wl = p["tags"].get("EXP14_WORKLOAD") or {}
        s = p["samples"]
        cap = capacity(p["tags"])
        checks[f"{label}_exit_0"] = top["leg_exit"].get(label) == 0
        checks[f"{label}_gpu_clean_before"] = top["pre_leg"].get(label) is True
        checks[f"{label}_v2_runner"] = p["v2_runner"]
        checks[f"{label}_block_size_32"] = eff.get("block_size") == REQUESTED_BLOCK_SIZE
        checks[f"{label}_counts"] = len(p["warmups"]) == WARMUPS_PER_LEG and len(s) == REPS_PER_LEG
        checks[f"{label}_tokens"] = bool(s) and all(x["prompt_tokens"] == CONTEXT_TOKENS and
                                                    x["output_tokens"] == OUTPUT_TOKENS for x in s)
        checks[f"{label}_prompt_rule"] = wl.get("prompt_rule") == PROMPT_RULE and wl.get("bos_token_id") is None
        checks[f"{label}_geometry"] = p["tags"].get("EXP14_MODEL_GEOMETRY") == GEOMETRY
        checks[f"{label}_capacity_observed"] = cap is not None
        checks[f"{label}_engine_cache_dtype"] = (p["tags"].get("EXP14_KV_DTYPE") or {}).get("engine_cache_dtype") == d
        checks[f"{label}_backend_evidence"] = set(p["backend_evidence"]) == EXPECTED_BACKEND_EVIDENCE
        ib = implied_bytes_per_token(cap, p["kv_log_gib"])
        checks[f"{label}_physical_layout_consistent"] = ib is not None and abs(
            ib / THEORETICAL_BYTES_PER_TOKEN[d]["bytes"] - 1.0) <= IMPLIED_BYTES_REL_TOL
        checks[f"{label}_no_skip_layers"] = eff.get("kv_cache_dtype_skip_layers") == []
        if d == B:
            mk = p["tags"].get("EXP14_RABIT_MARKERS") or {}
            checks[f"{label}_rabit_markers"] = bool(mk) and all(mk.values())
    for d in CONDITIONS:
        checks[f"{SHORT[d]}_capacity_identical_across_legs"] = len(
            {capacity(parsed[k]["tags"]) for k, _, dd in LEGS if dd == d}) == 1
    checks["config_diff_passed"] = diff["passed"]
    return {"checks": checks, "passed": all(checks.values())}


def build_serving_summary(parsed: dict, top: dict) -> dict:
    by = {d: [(k, label) for k, label, dd in LEGS if dd == d] for d in CONDITIONS}
    cap = {d: capacity(parsed[by[d][0][0]]["tags"]) for d in CONDITIONS}
    per_leg, cross, pooled, drift = {}, {}, {}, {}
    for d in CONDITIONS:
        (k1, l1), (k2, l2) = by[d]
        s1, s2 = r13.leg_stats(parsed[k1]["samples"]), r13.leg_stats(parsed[k2]["samples"])
        per_leg[SHORT[d]] = {l1: s1, l2: s2}
        cross[SHORT[d]] = {f"median_of_leg_{m}": statistics.median([s1[m], s2[m]]) for m in r13.LEG_STATS}
        tp = [s["tpot_ms"] for s in parsed[k1]["samples"] + parsed[k2]["samples"]]
        pooled[SHORT[d]] = {"n": len(tp), "pooled_median_tpot_ms": statistics.median(tp), "pooled_p90_tpot_ms": r13.p90(tp)}
        drift[SHORT[d]] = {"legs": [l1, l2], "tpot_abs_diff_ms": s2["median_tpot_ms"] - s1["median_tpot_ms"],
                           "tpot_pct_diff": 100.0 * (s2["median_tpot_ms"] / s1["median_tpot_ms"] - 1.0),
                           "p90_tpot_abs_diff_ms": s2["p90_tpot_ms"] - s1["p90_tpot_ms"],
                           "ttft_abs_diff_ms": s2["median_ttft_ms"] - s1["median_ttft_ms"],
                           "wall_abs_diff_ms": s2["median_wall_ms"] - s1["median_wall_ms"]}
    cx, cy = cross[SHORT[B]], cross[SHORT[A]]
    ratio = cap[B] / cap[A] if cap[A] and cap[B] else None
    return {"model": MODEL_B, "capacity_label": CAPACITY_LABEL,
            "observed_capacity": {SHORT[d]: {"observed_capacity_tokens": cap[d],
                                             "num_gpu_blocks": parsed[by[d][0][0]]["tags"]["EXP14_CAPACITY"]["num_gpu_blocks"],
                                             "reported_kv_cache_gib": parsed[by[d][0][0]]["kv_log_gib"],
                                             "implied_bytes_per_token_derived": implied_bytes_per_token(
                                                 cap[d], parsed[by[d][0][0]]["kv_log_gib"])} for d in CONDITIONS},
            "capacity_ratio_rabit_over_bf16": ratio,
            "theoretical_bytes_per_token_SEPARATE_FROM_OBSERVED": {SHORT[d]: THEORETICAL_BYTES_PER_TOKEN[d] for d in CONDITIONS},
            "latency_primary_per_leg": per_leg, "latency_primary_cross_leg": cross,
            "latency_secondary_pooled_within_session_descriptive": pooled, "leg_to_leg_drift": drift,
            "rabit_vs_bf16": {"basis": "cross-leg summaries (median of the two leg-level values per condition)",
                              "tpot_abs_diff_ms": cx["median_of_leg_median_tpot_ms"] - cy["median_of_leg_median_tpot_ms"],
                              "tpot_pct_diff": 100.0 * (cx["median_of_leg_median_tpot_ms"] / cy["median_of_leg_median_tpot_ms"] - 1.0),
                              "p90_tpot_abs_diff_ms": cx["median_of_leg_p90_tpot_ms"] - cy["median_of_leg_p90_tpot_ms"],
                              "ttft_abs_diff_ms": cx["median_of_leg_median_ttft_ms"] - cy["median_of_leg_median_ttft_ms"],
                              "wall_abs_diff_ms": cx["median_of_leg_median_wall_ms"] - cy["median_of_leg_median_wall_ms"],
                              "capacity_ratio": ratio, "rabit_serving_path": "exact fallback paths (see claim boundary)"},
            "jit_warnings_in_measured_samples": {label: parsed[k]["jit_warnings_in_measurement"] for k, label, _ in LEGS},
            "statistics_note": "descriptive only: two fresh-engine legs per condition; no significance tests or CIs",
            "model_revision": MODEL_REVISION, "model_manifest_sha256": ms.MODEL_MANIFEST_SHA256,
            "claim_boundary": CLAIM_BOUNDARY, "gpu": (top.get("environment") or {}).get("gpus")}


def analyze_serving(session_text: str) -> dict:
    legs, gate, shape, top_lines = demux(session_text)
    parsed = {k: parse_worker(v) for k, v in legs.items()}
    gates, top = parse_gates(gate, shape), parse_top(top_lines)
    diff = config_diff(parsed)
    integ = serving_integrity(parsed, gates, top, diff)
    summary = build_serving_summary(parsed, top) if integ["passed"] else None
    return {"integrity": {**integ, "gates": gates, "config_diff": diff}, "summary": summary}


def analyze_probe(session_text: str) -> dict:
    legs, gate, shape, top_lines = demux(session_text, PROBE_LEGS)
    gates, top = parse_gates(gate, shape), parse_top(top_lines)
    checks = {"rabit_gate_passed": gates["rabit_gate_passed"] and top["gate_exit"] == 0,
              "shape_gate_passed": gates["shape_gate_passed"] and top["shape_gate_exit"] == 0,
              "no_watchdog_timeout": not top["timeouts"], "session_complete": top["complete"],
              "gpu_is_h100_80gb": any("H100 80GB" in g.get("name", "") for g in ((top.get("environment") or {}).get("gpus") or [])),
              **model_checks(top)}
    facts = {}
    for k, label, d in PROBE_LEGS:
        p = parse_worker(legs[k])
        kv = p["tags"].get("EXP14P_KV") or {}
        wg = p["tags"].get("EXP14P_WORKLOAD_GENERATION") or {}
        cap = capacity(p["tags"], "EXP14P_KV")
        ib = implied_bytes_per_token(cap, p["kv_log_gib"])
        checks[f"{label}_exit_0"] = top["leg_exit"].get(label) == 0
        checks[f"{label}_gpu_clean_before"] = top["pre_leg"].get(label) is True
        checks[f"{label}_v2_runner"] = p["v2_runner"]
        checks[f"{label}_block_size_32"] = kv.get("block_size") == REQUESTED_BLOCK_SIZE
        checks[f"{label}_engine_cache_dtype"] = kv.get("engine_cache_dtype") == d
        checks[f"{label}_kv_quant_mode"] = kv.get("kv_quant_mode") == EXPECTED_KV_QUANT_MODE[d]
        checks[f"{label}_rabit_policy_frozen"] = p["tags"].get("EXP14P_RABIT_POLICY") == FROZEN_RABIT_POLICY
        checks[f"{label}_geometry"] = p["tags"].get("EXP14P_MODEL_GEOMETRY") == GEOMETRY
        checks[f"{label}_backend_evidence"] = set(p["backend_evidence"]) == EXPECTED_BACKEND_EVIDENCE
        checks[f"{label}_physical_layout_consistent"] = ib is not None and abs(
            ib / THEORETICAL_BYTES_PER_TOKEN[d]["bytes"] - 1.0) <= IMPLIED_BYTES_REL_TOL
        checks[f"{label}_workload_generation"] = (wg.get("prompt_tokens") == CONTEXT_TOKENS and
                                                  wg.get("output_tokens") == OUTPUT_TOKENS and wg.get("bos_token_id") is None)
        checks[f"{label}_sanity_generation_completed"] = bool((p["tags"].get("EXP14P_SANITY_GENERATION") or {}).get("output_tokens"))
        facts[label] = {"kv_cache_dtype": d, "capacity_tokens": cap, "kv": kv, "reported_kv_cache_gib": p["kv_log_gib"],
                        "implied_bytes_per_token": ib, "geometry": p["tags"].get("EXP14P_MODEL_GEOMETRY"),
                        "workload_generation": wg, "sanity_generation": p["tags"].get("EXP14P_SANITY_GENERATION")}
    return {"checks": checks, "passed": all(checks.values()), "gates": gates, "facts": facts,
            "model": top.get("model")}


# ---------------------------------------------------------------------------------------------- quality
def quality_runs() -> list[dict]:
    return [{"name": b, "script": QUALITY_DIR / f"{b}.py",
             "args": [*r12.SELECTION[b]["args"], "--methods", QUALITY_METHODS, "--model-name", MODEL_B]}
            for b in pb.BENCHMARKS]


def _identity_lines(text: str) -> dict:
    lines = text.splitlines()
    return {"header": [ln for ln in lines if qg.IDENTITY_HEADER_RE.match(ln)],
            "ground_truth": [ln for ln in lines if ln.startswith("Ground truth:")]}


def quality_integrity(name: str, rc: int, text: str) -> dict:
    checks = {"exit_code_zero": rc == 0, "no_traceback": "Traceback (most recent call last)" not in text,
              "log_names_model_b": f"Model: {MODEL_B}" in text}
    out: dict = {"benchmark": name}
    metric, qi, mi = r7.ROW_COLUMNS[name]
    ci = r11.COUNT_COLUMNS[name][0]
    summary = {m: e1._last_row_tokens(text, m) for m in ("bf16", "rabit2")}
    checks["summary_rows_present"] = all(t is not None and len(t) > max(qi, mi, ci) for t in summary.values())
    if checks["summary_rows_present"]:
        checks["counts_exact"] = all(int(float(t[ci])) == r12.EXPECTED_COUNT[name] for t in summary.values())
    rows = pb.extract(name, text)
    try:
        keys, _, _ = pb.paired_values(name, rows)
        checks["per_unit_rows_paired"] = len(keys) == r12.SELECTION[name]["exp12_units"]
    except ValueError:
        checks["per_unit_rows_paired"] = False
        return {**out, "checks": checks, "passed": False}
    exp12_text = (r12.OUT_DIR / f"{name}.log").read_text(encoding="utf-8", errors="replace")
    e12 = pb.extract(name, exp12_text)
    checks["unit_keys_identical_to_exp12"] = all([x["key"] for x in rows[m]] == [x["key"] for x in e12[m]]
                                                 for m in ("bf16", "rabit2"))
    checks["dataset_identity_identical_to_exp12"] = _identity_lines(text) == _identity_lines(exp12_text)
    cfg = r12.canonical_rabit2()
    bad = []
    for m in ("bf16", "rabit2"):
        for x in rows[m]:
            t = 1024 if name == "continuation_ppl" else x["prefix_tokens"]
            exp_mb = expected_kv_mb(m, t, cfg)
            if abs(x["kv_mb"] - exp_mb) > STORAGE_TOL_MB + 1e-9:
                bad.append({"method": m, "key": x["key"], "observed": x["kv_mb"], "expected": exp_mb})
    checks["per_unit_kv_mb_matches_model_b_accounting"] = not bad
    out["storage_mismatches"] = bad[:20]
    return {**out, "checks": checks, "passed": all(checks.values())}


def canonical_subset_aggregates() -> dict:
    res = {}
    for b in pb.BENCHMARKS:
        rows = pb.extract(b, (QUALITY_OUT / f"{b}.log").read_text(encoding="utf-8", errors="replace"))
        ref_rows = pb.extract(b, (r12.REFERENCE_DIR / f"{b}.log").read_text(encoding="utf-8", errors="replace"))
        ref_keys = [x["key"] for x in ref_rows["rabit2"]]
        agg = r12._subset_aggregate(b, rows, len(ref_keys), ref_keys)
        res[b] = {"n_canonical_units": len(ref_keys), "aggregate": agg,
                  "metric": "ppl" if b == "continuation_ppl" else pb.METRIC[b][0]}
    return res


# ------------------------------------------------------------------------------------------------------ main
def build_snapshot() -> Path:
    out = Path(tempfile.mkdtemp(prefix="exp14_vllm_snapshot_")) / "vllm_kvquant_snapshot.zip"
    e1.run_git("-c", "core.autocrlf=false", "archive", "--format=zip", "-o", str(out), "HEAD:vllm-kvquant")
    return out


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _modal_session(legs: str, log: Path, probe: bool) -> int:
    snap = build_snapshot()
    os.environ["EXP14_VLLM_SNAPSHOT"] = str(snap)
    cmd = [sys.executable, "-m", "modal", "run", str(MODAL_APP), "--legs", legs, "--warmups", str(WARMUPS_PER_LEG),
           "--reps-per-leg", str(REPS_PER_LEG), *(["--probe"] if probe else [])]
    return e1.stream_command(cmd, log)


def _app_ids(log: Path) -> list[str]:
    return sorted(set(re.findall(r"ap-[A-Za-z0-9]{20,}", log.read_text(encoding="utf-8", errors="replace"))))


def run_probe(prov: dict) -> int:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    log = PROBE_DIR / "probe_session.log"
    rec = {"experiment": 14, "kind": "feasibility_probe", "non_evidence": True, "status": "running",
           "git_head": prov["git_head"], "started_utc": _now(), **{k: prov[k] for k in PROBE_BINDING_KEYS}}
    _write(PROBE_DIR / "probe_record.json", rec)
    rc = _modal_session(legs_arg(PROBE_LEGS), log, probe=True)
    res = analyze_probe(log.read_text(encoding="utf-8", errors="replace"))
    ok = rc == 0 and res["passed"] and not protected_status()
    rec.update(status="passed" if ok else "failed", modal_returncode=rc, completed_utc=_now(), modal_app_ids=_app_ids(log),
               session_log_sha256=e1.sha256(log), result=res)
    _write(PROBE_DIR / "probe_record.json", rec)
    print(f"\nEXPERIMENT 14 FEASIBILITY PROBE (NON-EVIDENCE) {rec['status'].upper()}")
    return 0 if ok else 1


def run_serving(prov: dict) -> int:
    SERVING_DIR.mkdir(parents=True, exist_ok=True)
    log = SERVING_DIR / "modal_session.log"
    manifest = {"experiment": 14, "part": "serving", "status": "running", "started_utc": _now(), "provenance": prov}
    _write(SERVING_DIR / "manifest.json", manifest)
    rc = _modal_session(legs_arg(), log, probe=False)
    res = analyze_serving(log.read_text(encoding="utf-8", errors="replace"))
    _write(SERVING_DIR / "integrity_check.json", res["integrity"])
    ok = rc == 0 and res["integrity"]["passed"] and not protected_status()
    if ok:
        _write(SERVING_DIR / "capacity_latency_summary.json", res["summary"])
    manifest.update(status="passed" if ok else "failed", modal_returncode=rc, completed_utc=_now(),
                    modal_app_ids=_app_ids(log), session_log_sha256=e1.sha256(log))
    _write(SERVING_DIR / "manifest.json", manifest)
    print(f"\nEXPERIMENT 14 SERVING {manifest['status'].upper()}")
    return 0 if ok else 1


def remote_model_identity() -> dict:
    """ModelScope master must resolve to the frozen commit with the identical file manifest (read-only)."""
    refs = subprocess.run(["git", "ls-remote", ms.MODEL_GIT_URL, "refs/heads/master"], capture_output=True, text=True,
                          timeout=120).stdout.split()
    master = refs[0] if refs else None
    with urllib.request.urlopen(ms.MODEL_FILES_API.format(revision="master"), timeout=120) as r:
        listing = ms.listing_manifest(json.loads(r.read().decode("utf-8")))
    return {"master_commit": master, "master_is_frozen_revision": master == MODEL_REVISION,
            "master_manifest_sha256": ms.manifest_sha256(listing),
            "manifest_identical": listing == sorted(ms.FROZEN_FILES),
            "passed": master == MODEL_REVISION and listing == sorted(ms.FROZEN_FILES)}


def run_quality(prov: dict) -> int:
    pre = remote_model_identity()
    if not pre["passed"]:
        raise RuntimeError(f"ModelScope master no longer equals the frozen Model-B revision; quality refused: {pre}")
    QUALITY_OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment": 14, "part": "quality", "status": "running", "started_utc": _now(), "provenance": prov,
                "model_identity_pre": pre, "runs": []}
    _write(QUALITY_OUT / "manifest.json", manifest)
    results = []
    for r in quality_runs():
        log = QUALITY_OUT / f"{r['name']}.log"
        rc = e1.stream_command([sys.executable, "-m", "modal", "run", str(r["script"]), *r["args"]], log)
        res = quality_integrity(r["name"], rc, log.read_text(encoding="utf-8", errors="replace"))
        results.append(res)
        manifest["runs"].append({"name": r["name"], "returncode": rc, "passed": res["passed"], "log_sha256": e1.sha256(log)})
        _write(QUALITY_OUT / "manifest.json", manifest)
        if not res["passed"]:  # no retry; stop at the first failing benchmark
            break
    _write(QUALITY_OUT / "integrity_results.json", results)
    post = remote_model_identity()
    scan_log = QUALITY_OUT / "model_snapshot_scan.log"
    scan_rc = e1.stream_command([sys.executable, "-m", "modal", "run", str(MODEL_SCAN_APP)], scan_log)
    scan = next((json.loads(ln.split("=", 1)[1]) for ln in scan_log.read_text(encoding="utf-8", errors="replace")
                 .splitlines() if ln.strip().startswith("EXP14_MODEL_SCAN=")), None)
    manifest.update(model_identity_post=post, model_snapshot_scan={"returncode": scan_rc, "result": scan})
    model_ok = post["passed"] and scan_rc == 0 and bool(scan and scan.get("passed"))
    ok = (len(results) == len(pb.BENCHMARKS) and all(r["passed"] for r in results) and model_ok
          and not protected_status())
    if ok:  # statistics only after every benchmark is valid (offline, CPU)
        pb.main(["--results-dir", str(QUALITY_OUT), "--out", str(QUALITY_OUT / "variance_results.json")])
        _write(QUALITY_OUT / "canonical_subset_aggregates.json", canonical_subset_aggregates())
    manifest.update(status="passed" if ok else "failed", completed_utc=_now())
    _write(QUALITY_OUT / "manifest.json", manifest)
    print(f"\nEXPERIMENT 14 QUALITY {manifest['status'].upper()}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    e1.make_console_encoding_safe()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-protocol", action="store_true", help="write the protocol (pre-commit only)")
    ap.add_argument("--part", choices=("probe", "serving", "quality"))
    a = ap.parse_args(argv)
    if a.write_protocol:
        if PROTOCOL.exists():
            raise SystemExit(f"{PROTOCOL.name} already exists; the frozen protocol is never overwritten")
        PROTOCOL.write_text(json.dumps(build_protocol(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {PROTOCOL.relative_to(ROOT).as_posix()}")
        return 0
    if not a.dry_run and not a.part:
        raise SystemExit("--part probe|serving|quality is required (or --dry-run)")
    prov = preflight(a.part, a.dry_run)
    print("RABIT-KV MLSys 2027 -- Experiment 14 second model (" + MODEL_B + ")")
    print("Preflight OK:", json.dumps({k: prov[k] for k in ("git_head", "vllm_kvquant_tree", "protocol_sha256")}))
    print("  probe legs:  ", legs_arg(PROBE_LEGS))
    print("  serving legs:", legs_arg())
    for r in quality_runs():
        print(f"  quality {r['name']}: {' '.join(r['args'])}")
    if prov["uncommitted_files"]:
        print("  WARNING (dry-run only): uncommitted:\n    " + prov["uncommitted_files"].replace("\n", "\n    "))
    if a.dry_run:
        print("\n--dry-run: nothing executed, no files written.")
        return 0
    return {"probe": run_probe, "serving": run_serving, "quality": run_quality}[a.part](prov)


if __name__ == "__main__":
    raise SystemExit(main())
