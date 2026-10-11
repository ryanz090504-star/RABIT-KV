# RABIT-KV — MLSys Paper Evidence Map

Status: paper-preparation document. All quality experiments are frozen at `164c17f`. Experimental work is closed: QUALITY COMPLETE, SERVING COMPLETE, PERFORMANCE PROFILING COMPLETE, PERFORMANCE OPTIMIZATION NO-GO / CLOSED (`docs/MLSYS_PAPER_STORY_LOCK.md` section I). The final story is Story A (capacity-first system). Every
number below was recovered from committed evidence files (paths given per claim), not from
memory or from earlier prose. This document creates no new result.

Evidence classes used below:

- **PHYSICAL** — real-engine serving evidence (patched vLLM, H100): allocator capacity, latency, throughput.
- **LOGICAL** — logical quality evidence under canonical RABIT semantics (HF transformers, `canonical_rabit_quality.py`).
- **CORRECTNESS** — exactness / parity / conformance evidence.
- **LEGACY** — results of the old logical evaluator; provenance only, never final quality evidence.

Claim status: **SUPPORTED**, **SUPPORTED WITH QUALIFICATION** (SWQ), **NOT SUPPORTED**.

Common hardware unless stated: one NVIDIA H100 80GB HBM3 (81,559 MiB) per run on Modal.
Common serving configuration (Exp3/4/5/13/14): vLLM `0.10.0+kvquant` (base `f329ce4`), eager,
`block_size` 32, `gpu_memory_utilization` 0.82, `max_model_len` 32768, single request unless stated.

---

## 1. RABIT system / representation

Frozen operating point: **K3 / V2 / G32 / R4 / META8g64**. It was never retuned in any experiment.

| Component | Exact meaning | Source |
|---|---|---|
| K3 | Keys: 3-bit affine codes, sequence-axis grouping: one (min, scale) per (head, channel) per 32-token group; incomplete group zero-padded; `scale = (max − min) / 7` | `canonical_rabit_quality.py` (c360697); `kvquant_k3.py` `quantize_k3_sequence_affine_ref` |
| V2 | Values: 2-bit affine codes, per (token, head) over 32-channel groups; `scale = (max − min) / 3` | same; `quantize_v2_group_affine_ref` |
| G32 | Group size 32: 32 tokens per K group and per physical page; 32 channels per V group | same |
| R4 | The newest 4 tokens stay exact BF16 (residual window); older tokens age into the open group, then into closed 32-token pages | same; `RABIT2_RESIDUAL_TOKENS = 4` |
| META8g64 | Each primary min / scale array is itself quantized: 64 values per metadata group, uint8 codes, BF16 group min / scale; dequantization uses the META8-decoded parameters | same; `encode_metadata_uint8_group_ref` |

| ID | Claim | Numbers | Evidence | Class | Status | Wording constraint |
|---|---|---|---|---|---|---|
| S1 | The operating point jointly fixes K/V precision, grouping, residual window and metadata precision | K3 / V2 / G32 / R4 / META8g64 | `benchmarks/mlsys2027/canonical_ppl_protocol.json`; `canonical_rabit_quality.py` @ c360697 | CORRECTNESS | SUPPORTED | Describe as one frozen configuration, not a tuned family |
| S2 | "2-bit target" is an operating-point label, not literal 2 bits/token | Llama physical page: 24,832 B per 32-token block per layer = 776 B per token-layer vs 4,096 B BF16; ≈ 3.03 bits per K/V element (derived: 776 × 8 / 2048) | `external_baseline/exp13/matched_capacity_latency_summary.json` (`theoretical_bytes_per_token`); Qwen page breakdown in `second_model/serving/capacity_latency_summary.json` | PHYSICAL (bytes) + derived | SWQ | Always say "2-bit-target operating point"; never "2 bits per token". The 3.03-bit figure is DERIVED, label it so. The R4 residual is per-sequence state outside the paged pool |
| S3 | Physical page composition (Qwen geometry, per 32-token block per layer) | 12,416 B = K3 payload 6,144 + V2 payload 4,096 + META8g64 K min/scale 2 × 544 + V min/scale 2 × 544 | `second_model/serving/capacity_latency_summary.json` | PHYSICAL | SUPPORTED | The Llama page (24,832 B) scales with 8 KV heads; an explicit Llama breakdown is not in a committed summary — derive and label, or cite the Qwen one |

---

## 2. Physical capacity

"Capacity" below is **physical allocator capacity**: `num_gpu_blocks × block_size` reported by
the live engine under a fixed GPU memory budget (≈ 47.98 GiB KV budget on H100). It is distinct
from **logical compression ratio** (packed bytes accounting used by the legacy quality harness,
5.18–5.27×), which must not be presented as capacity.

| ID | Claim | Numbers (tokens) | Evidence (commit) | Model | Class | Status | Wording constraint |
|---|---|---|---|---|---|---|---|
| C1 | BF16 physical capacity | 393,024 (12,282 blocks; 131,081 B/token implied) | Exp3 `deployment/` (d403df1), replicated (eaa6acc), Exp4 (7a051e6), Exp13 (42c2799) | Llama-3.1-8B | PHYSICAL | SUPPORTED | Identical in every session |
| C2 | FP8 physical capacity | 785,568 (24,549 blocks; 65,540 B/token) | Exp4 `fp8_baseline/` (7a051e6); Exp13 (42c2799) | Llama | PHYSICAL | SUPPORTED | vLLM native per-tensor FP8 e4m3, default 1.0 scales |
| C3 | RABIT physical capacity | 2,074,592 (64,831 blocks; 24,833 B/token) | Exp3, Exp4, Exp5 (e117e7a), Exp13 | Llama | PHYSICAL | SUPPORTED | — |
| C4 | TurboQuant physical capacity | 1,203,200 (37,600 blocks; 42,818 B/token) | Exp13 `external_baseline/exp13/` (42c2799) | Llama | PHYSICAL | SUPPORTED | Config `turboquant_k3v4_nc`: 28 TurboQuant layers + 4 BF16 boundary layers |
| C5 | Capacity ratios | RABIT/BF16 **5.2785×**; RABIT/FP8 **2.6409×**; RABIT/TurboQuant **1.7242×**; TurboQuant/BF16 3.0614×; FP8/BF16 1.9988× | Exp13 `capacity_ratios`; Exp4 `capacity_ratios` | Llama | PHYSICAL | SUPPORTED | Ratios of observed allocator tokens in one configuration. Not a throughput or quality claim |
| C6 | The capacity ratio carries to a second geometry | Qwen2.5-7B: BF16 902,656 → RABIT 4,764,640 tokens = **5.2785×** | Exp14 `second_model/serving/` (7dfaee4) | Qwen2.5-7B | PHYSICAL | SUPPORTED | Capacity only; says nothing about Qwen quality |

Not supported: any statement that 5.28× capacity yields 5.28× more served requests or throughput (see §4).

---

## 3. Serving latency (single request, 2048-token prompt, 32 output tokens)

Absolute latency differs across sessions (BF16 TPOT 13.0–24.2 ms across Exp3 / replication / Exp4 / Exp13).
Only within-session comparisons are valid; never mix absolute numbers from different sessions.
**Primary table = Exp13** (the only session with all four methods, mirrored A B C D D C B A).

| ID | Claim | Numbers | Evidence | Model | Class | Status | Wording constraint |
|---|---|---|---|---|---|---|---|
| L1 | Four-way matched latency (cross-leg medians) | TPOT ms: BF16 24.19, FP8 25.75, TurboQuant 27.79, RABIT 30.10. TTFT ms: 59.99 / 63.43 / 55.03 / 144.65. Wall ms: 809.7 / 861.2 / 916.4 / 1078.8 | Exp13 (42c2799) `latency_primary_cross_leg` | Llama | PHYSICAL | SUPPORTED | Descriptive: two fresh-engine legs per condition; no CI, no significance test |
| L2 | RABIT is slower per token than BF16 | TPOT +24.4% (Exp13); +29.2% (Exp3, d403df1); +32.8% (Exp3 replication, eaa6acc); +32.9% (Exp4, 7a051e6); +35.0% (Exp5 at 2048, e117e7a) | as listed | Llama | PHYSICAL | SUPPORTED | State as a cost. Report the range across sessions, do not average |
| L3 | RABIT TTFT is higher than BF16 at 2048 tokens | 144.65 vs 59.99 ms (Exp13); 145.7 vs 61.9 (Exp3); 116.5 vs 59.6 (Exp4) | as listed | Llama | PHYSICAL | SUPPORTED | — |
| L4 | RABIT vs FP8 | TPOT +16.9% (Exp13), +18.0% (Exp4); capacity 2.64× | Exp13, Exp4 | Llama | PHYSICAL | SUPPORTED | No FP8 quality was measured (`fp8_quality_evaluated: false`) |
| L5 | RABIT vs TurboQuant | TPOT +8.3% (30.10 vs 27.79 ms); TTFT 144.65 vs 55.03 ms; capacity 1.72× | Exp13 `rabit_comparisons` | Llama | PHYSICAL | SWQ | METHOD-NATIVE SYSTEM comparison: TurboQuant runs its own backend + FlashAttention v2 on 4 BF16 boundary layers; others use TRITON_ATTN. Not a quantizer-kernel comparison. No TurboQuant quality measured. Disclose the backported upstream dispatch fix (PR #47609) |
| L6 | Qwen serving works, through fallback paths | Qwen: BF16 TPOT 26.96 ms, RABIT 48.70 ms (**+80.7%**); TTFT 58.3 vs 161.6 ms | Exp14 serving (7dfaee4) | Qwen | PHYSICAL | SWQ | RABIT serves Qwen through exact FALLBACK paths (decode 64/64, append 62/62, V2 quantizer 62/62). This is NOT evidence that the Llama-optimized kernels transfer; say "untuned fallback kernels" |

Not supported: any latency advantage for RABIT at 2048 tokens; the legacy headline "21.61 ms/token" (5 reps, unmatched; `results/performance/`, 3b3b457).

---

## 4. Concurrency (Exp6; closed loop, 256 measured requests per point, 3 trials)

| ID | Claim | Numbers | Evidence | Class | Status | Wording constraint |
|---|---|---|---|---|---|---|
| T1 | At 2048-token prompts both methods sustain every tested concurrency | C ∈ {1, 4, 8, 16, 32, 64}: all 36 points "sustained_target_concurrency" | `concurrency_scaling/L2048/shadow_conditioned/summary.json` (8512566) | PHYSICAL | SUPPORTED | "Highest tested", not a maximum |
| T2 | RABIT throughput is lower than BF16 at every tested concurrency (L2048) | req/s BF16: 1.458, 4.592, 7.218, 10.042, 12.911, 14.361; RABIT: 1.022, 1.612, 1.844, 2.015, 2.083, 1.776 → ratio 0.70, 0.35, 0.26, 0.20, 0.16, 0.12 | same (medians over 3 trials) | PHYSICAL | SUPPORTED | Must be stated explicitly in the paper |
| T3 | At 8192-token prompts BF16 hits its physical ceiling before C = 64; RABIT sustains C = 64 | BF16 C64: "target_concurrency_not_reached" in 3/3 trials, max in-flight 47; RABIT: sustained at C64 in 3/3 trials (max in-flight 64) | `concurrency_scaling/L8192/shadow_conditioned/combined_summary.json` (0f5f6ef) | PHYSICAL | SUPPORTED | 47 matches the allocator bound 393,024 / (8192 + 32 rounded to blocks). Claim "higher offered concurrency", not higher throughput |
| T4 | RABIT throughput is lower at every tested concurrency (L8192) | req/s BF16: 0.937, 1.801, 2.101, 2.282, 2.346, 2.366 (C64 not reached); RABIT: 0.828, 1.259, 1.385, 1.424, 0.870, 0.323 → ratio 0.88, 0.70, 0.66, 0.62, 0.37 (C1–C32) | same (cross-trial medians) | PHYSICAL | SUPPORTED | RABIT throughput *falls* beyond C16 (1.424 → 0.870 → 0.323). At C64 RABIT's 0.323 req/s is below BF16's 2.366 req/s achieved with 47 in flight |

Required sentence: RABIT supports higher *offered* concurrency because of capacity, but the current
implementation has lower throughput at every tested concurrency. Capacity must not be converted into a throughput claim.
Concurrency here is overlapping in-flight concurrency, not GPU-resident concurrency (evidence label).

---

## 5. Context scaling (Exp5 Attempt 2; single request, 15 samples per cell)

Evidence: `context_scaling/attempt_2/context_scaling_summary.json` (e117e7a). RABIT implementation: finalized Stage3C `shared_decode`, query block 32.

| Prompt tokens | BF16 TPOT ms | RABIT TPOT ms | TPOT Δ | BF16 TTFT ms | RABIT TTFT ms | BF16 wall ms | RABIT wall ms |
|---|---|---|---|---|---|---|---|
| 512 | 15.33 | 21.26 | +38.7% | 32.1 | 95.6 | 508.9 | 754.1 |
| 2048 | 15.56 | 21.00 | +35.0% | 59.8 | 119.9 | 542.1 | 770.6 |
| 4096 | 15.60 | 21.03 | +34.8% | 144.6 | 171.3 | 627.8 | 822.9 |
| 8192 | 15.72 | 20.66 | +31.4% | 409.5 | 305.2 | 895.2 | 945.5 |
| 16384 | 15.43 | 21.15 | +37.1% | 1280.8 | 687.8 | 1758.3 | 1344.0 |
| 32736 (32K model-limit point) | 15.56 | 29.38 | +88.9% | 4435.0 | **114,376.8** | 4916.7 | 115,287.4 |

| ID | Claim | Evidence | Class | Status | Wording constraint |
|---|---|---|---|---|---|
| X1 | RABIT per-token decode overhead is roughly constant (+31% to +39%) from 512 to 16,384 tokens | table above | PHYSICAL | SUPPORTED | Single session; descriptive |
| X2 | Allocator capacity is independent of request context | 393,024 vs 2,074,592 at every context | PHYSICAL | SUPPORTED | Derived live paged-KV bytes are DERIVED (5.28×), not measured live memory |
| X3 | RABIT prefill is faster than BF16 at long context | TTFT lower at 8192 and 16384, but higher at 512–4096 and 26× higher at the 32K point | PHYSICAL | **NOT SUPPORTED** | Do NOT claim faster prefill. The sign changes with context and the 32K point is a severe slowdown |
| X4 | Beyond the first 16,384-token prefill chunk RABIT has a chunked-prefill bottleneck | 32K point TTFT 114.4 s vs 4.4 s; diagnostic: ≈ 13–15 ms excess per second-chunk query token | Exp5; `diagnostics/stage3c_cliff/summary.json` (356b2e3); tile32 conclusion (6edae6d) | PHYSICAL (diagnostic) | SUPPORTED (as a limitation) | Must be disclosed. 32,768 is a nominal model-limit point, not a demonstrated maximum feasible context. No formal complexity law is claimed |

---

## 5a. Performance-risk profiling diagnostic (EXPLANATORY ONLY — not performance evidence)

Evidence: `diagnostics/perf_risk_profile/attempt_1/` (`4d07cf6`; addendum `corrected_attribution.json`, `dadc2a0`).
Final record: `docs/MLSYS_PAPER_STORY_LOCK.md` section I. Llama-3.1-8B, one H100, no quality scoring, serving sources
unmodified. The profiler inflated RABIT wall time 3.7×–8.8×: component percentages are diagnostic only, must not be
quoted as precise measurements, and never replace Exp5 / Exp6 / Exp13 numbers.

| ID | Claim | Observation | Evidence | Class | Status | Wording constraint |
|---|---|---|---|---|---|---|
| P1 | The profiling diagnostic is valid | all gates passed; 13/13 legs; correctness gate passed; serving-source hashes equal the accepted ones | `summary.json` | PHYSICAL (diagnostic) | SUPPORTED | Diagnostic evidence only |
| P2 | Unprofiled diagnostic legs reproduce the accepted points (workload identity) | 2048/C8: 1.848 vs 1.844 req/s; 8192/C32: 0.900 vs 0.870 req/s; 32K TTFT 118.3 s vs 114.4 s | `summary.json` vs Exp6 / Exp5 | PHYSICAL (diagnostic) | SWQ | Shortened runs (96 measured requests; 1 request at 32K). Identity check only; NOT additional performance results |
| P3 | The profiler materially perturbed RABIT execution | profiled / unprofiled wall 3.74×, 6.29×, 8.82×; residual after bookkeeping correction 1.35×–1.63× | `corrected_attribution.json` | PHYSICAL (diagnostic) | SUPPORTED | Must be disclosed wherever the profile is discussed (appendix) |
| P4 | Serialized per-request / per-token processing is the dominant qualitative bottleneck family | per-request Python loop per layer; per-token open-tail re-quantization + metadata re-encoding; per-token append / aging; ≈ 10.8 kernel launches per (layer, request, token) in decode, ≈ 6.9 per (layer, token) in a non-initial chunk; one `torch.profiler` cross-check directionally consistent | source (`triton_attn.py`, `rabit_kv2.py`, `rabit_kv2_stage3c_shared_decode.py`); `corrected_attribution.json` | PHYSICAL (diagnostic) | SWQ | Qualitative. Use the frozen sentence: "Profiling identified serialized per-request/per-token processing as the dominant qualitative bottleneck family, but eliminating it requires a substantial implementation redesign rather than a narrow optimization." Never "no dominant bottleneck". Not intrinsic to the representation |
| P5 | The 32K path additionally contains substantial closed-page device work | shared closed-page decode kernel: on the order of 70 s of device span in the profiled 32K leg (approximate) | `corrected_attribution.json` | PHYSICAL (diagnostic) | SWQ | "long-context prefill also contains substantial closed-page device work, so the bottleneck is not solely host-loop overhead." No precise share |
| P6 | Performance optimization is NO-GO for this submission | removing the mechanism needs cross-request / cross-token batching, runtime restructuring and / or new kernels in the hash-frozen serving implementation, with major revalidation | story lock section I | decision | SUPPORTED | Batched runtime / kernel design is future work |

---

## 6. Canonical continuation perplexity (authoritative)

Evidence: registered two-model Attempt 2, `canonical_quality_v2/continuation_ppl/two_model_attempt_2/` (10b0973; Llama run 01cced9; accepted bae70af).
Protocol: WikiText-2 test, N = 32 windows, context 1024, continuation 128, no BOS, 4096 scored tokens per arm,
paired percentile bootstrap (10,000 resamples). Attempt 1 is INVALID with no quality result.

| ID | Claim | Numbers | Model | Class | Status | Wording constraint |
|---|---|---|---|---|---|---|
| Q1 | Llama continuation PPL under canonical RABIT | 7.515 → 7.633, **+1.56%**, paired 95% CI [+1.12%, +2.05%]; 30 of 32 windows worse | Llama-3.1-8B @ `359efdbb` | LOGICAL | SUPPORTED | Logical quality, not serving evidence |
| Q2 | Qwen continuation PPL under canonical RABIT | 6.787 → 112.39, **+1555.9%**, paired 95% CI [+1320%, +1828%]; 32 of 32 windows worse | Qwen2.5-7B @ `16c17498` | LOGICAL | SUPPORTED | Report as a negative result of the same frozen operating point |
| Q3 | Preregistered Case 1 | "The frozen operating point shows model-specific quality sensitivity; mechanism remains undiagnosed." | both | LOGICAL | SUPPORTED | No numerical definition of "near BF16". No mechanism attribution |
| Q3b | The legacy evaluator distorted the magnitude of the Qwen degradation, but the severe sensitivity remains under canonical semantics | legacy Qwen RABIT PPL 299.62 vs canonical 112.39 | both | LOGICAL + LEGACY | SWQ | Do not attribute the change to the metadata-layout fix or the aging fix individually; no causal decomposition was run |

---

## 7. Canonical long-context quality (authoritative; Llama only)

Evidence: registered Attempt 1, `canonical_quality_v2/long_context/attempt_1/` (7896fcd; accepted 164c17f).
Frozen Exp12 prompts / selection / greedy generation / scorers; BF16 vs canonical RABIT on identical prompt ids.

| ID | Claim | Numbers | Class | Status | Wording constraint |
|---|---|---|---|---|---|
| Q4 | NIAH | 57/57 vs 57/57 (19/19 at 4096, 8192, 16384 for both); 0/57 score changes | LOGICAL | SUPPORTED | "No score loss on the tested suite". Saturated; cases are not independent draws; no CI |
| Q5 | Passage Retrieval | 100.0 vs 100.0, Δ 0.0, CI [0.0, 0.0]; 0/200 score changes | LOGICAL | SUPPORTED | Saturated for BF16; cannot resolve small effects |
| Q6 | HotpotQA, primary legacy-compatible scorer | 59.30 vs 58.27, Δ **−1.03 F1** (−1.73%), paired 95% CI [−4.24, +2.20]; 8/100 scores change (5 worse, 3 better), median paired Δ 0 | LOGICAL | SWQ | "approximately 1 F1 point below BF16, with the paired CI spanning zero and the differences concentrated in a small number of examples". NEVER "statistically equivalent" or "unaffected" |
| Q7 | HotpotQA, secondary official scorer | 59.85 vs 58.71, Δ −1.14 F1 (−1.90%), CI [−4.30, +2.03] | LOGICAL | SWQ | Secondary only; both scorers were preregistered. Primary keeps the historical Exp12 article behaviour |
| Q8 | BF16 control reproduces Exp12 | NIAH 57/57, Passage Retrieval 200/200, HotpotQA 99/100 scores | LOGICAL | SUPPORTED | One differing example = known greedy-decoding run-to-run variability |

Frozen wording: "On Llama-3.1-8B, the frozen canonical RABIT operating point preserves exact retrieval on the
tested NIAH and Passage Retrieval suites. On the 100-example HotpotQA subset, canonical RABIT is approximately
1 F1 point below BF16, with the paired confidence interval spanning zero and the observed differences
concentrated in a small number of examples."

Not supported (Qwen long-context was not run under canonical semantics): any Qwen NIAH / retrieval / QA claim.

---

## 8. Correctness / evaluator validation

| ID | Claim | Numbers | Evidence | Class | Status | Wording constraint |
|---|---|---|---|---|---|---|
| V1 | Physical exactness of the serving implementation | Frozen RABIT correctness gate run before the serving legs: 105 pytest items passed (recorded in Exp3 / Exp4 / Exp5); gate recorded as passed in Exp13 and Exp14 (layout, physical page codec vs reference, online state vs oracle, exact metadata, GQA, causal prefill) | gate recorded in Exp3/4/5/13/14 summaries (`correctness_gate`); `vllm-kvquant/tests/quantization/test_rabit_kv2_*.py`, `test_kvquant_k3.py` | CORRECTNESS | SUPPORTED | "Physical pages reproduce the reference state" — cite the tests; do not claim formal verification |
| V2 | Canonical evaluator equals the independent oracle on CPU | 276/276 cases: full state 104/104, sequential aging 56/56, HF cache 12/12 bit-exact; negative control: old harness K equal, V differs at every N ≥ 31 | `quality_semantic_audit/parity_attempt_3/` (8fa9a9c) | CORRECTNESS | SUPPORTED | Oracle = `kvquant_k3.py` `*_ref`, AST-extracted |
| V3 | Canonical equals the oracle on CUDA, Llama real K/V | 32/32 layers, every canonical field bit-exact (1,024-token prefill) | `cuda_conformance_diagnostic/attempt_1/` (ec80638) | CORRECTNESS | SUPPORTED | Same-device comparison |
| V4 | Canonical equals the oracle on CUDA, Qwen real K/V | 28/28 layers, every field bit-exact | `cuda_conformance_diagnostic/qwen2_5_7b/attempt_1/` (fd8d275) | CORRECTNESS | SUPPORTED | — |
| V5 | Canonical equals the oracle on CUDA at 16k | 32/32 layers, 38/38 fields, 0 differing elements (16,383-token prefill: 511 closed pages + 27 open + R4) | `cuda_conformance_diagnostic/llama3_1_8b_niah_16384/attempt_1/` (2637457; erratum beside it) | CORRECTNESS | SUPPORTED | — |
| V6 | Online aging is validated | Token-by-token aging equals the oracle state at every length (CPU 56/56; CUDA synthetic 28/28 per geometry in each conformance run); offline proofs: every `decoded()` call equals the full canonical state, pages close during decode | 8fa9a9c; ec80638 / fd8d275 / 2637457; `continuation_ppl/offline_proof_record_v2.json`; `long_context/offline_proof_record.json` | CORRECTNESS | SUPPORTED | The first offline proof record (eb3f4b1) is SUPERSEDED (id-keyed spy); cite v2 only |
| V7 | CPU and CUDA canonical states are not bit-identical | Earliest divergence: FP32 scale `(max − min) / c`; codes differ by at most 1 (Llama 1k: K 0.074%, V 0.020% of codes) | ec80638, fd8d275, 2637457 summaries; amendment d7ba819 | CORRECTNESS | SUPPORTED (as a limitation) | Validation relies on same-device oracle conformance; no cross-device tolerance is defined |

Legacy logical-evaluator mismatch (semantic audit e88f460, classification A). Keep the main-paper
explanation to two sentences: (1) the old evaluator flattened V META8g64 metadata in (head, token,
V-group) order instead of the canonical (token, head, V-group) order — 0/511 (Qwen) and 0/1022 (Llama)
metadata groups identical; (2) it quantized only the prompt prefix once and left the last prompt token
and all generated tokens in BF16, whereas canonical tokens keep aging (R4 → open group → closed page).
Details belong in the appendix.

---

## 8a. Quality-evaluator execution schedule vs serving (read-only source audit)

Reconstructed from the frozen source, not from prose: `benchmarks/mlsys2027/canonical_ppl_core.py`
(`score_stepwise`), `canonical_longctx_core.py` (`generate`), `canonical_rabit_quality.py`
(`CanonicalRabitCache.from_prefill` / `update`, `CanonicalLayerState`), `canonical_ppl_protocol.json`,
`canonical_longctx_tasks.py`; serving: `triton_attn.py` `_forward_rabit_kv2`, `rabit_kv2.py`
`Rabit2SingleSequenceRuntime`. No experiment was run for this audit.

**Continuation PPL** (per window: 1,024 context + 128 scored tokens; both models):

1. The 1,024 context tokens are processed in ONE dense BF16 forward with a normal HF cache. Prefill attention sees
   BF16 history only.
2. The FIRST scored token is scored from the last logit of that BF16 prefill. It is identical in the BF16 and RABIT arms.
3. `from_prefill` converts the raw BF16 prefill K / V of every layer, in one step, into the canonical state at length
   1,024: 31 closed pages, 28 open-group tokens, 4 residual tokens.
4. Scored tokens 2..128 come from 127 forwards, each feeding ONE teacher-forced token. In each layer the new token's
   K / V are appended, the state ages, and the token attends to the canonical state at its own length (itself included,
   exact, in R4). The last continuation token is never fed.
5. Aging at every append: a token leaves R4 when four newer tokens exist; the open group (zero-padded K3 group, V2 per
   token, META8g64) is recomputed from the raw BF16 tokens at every length; a full 32-token old group closes into a page
   that is decoded once and never recomputed.
6. No generated / teacher-forced token stays BF16 beyond R4. The evaluator never executes a later prefill chunk.

**Long-context generation** (NIAH, Passage Retrieval, HotpotQA; Llama only; prompts at most 16,384 tokens):

1. `prompt[:-1]` is processed in ONE dense BF16 forward. Prefill attention sees BF16 history only.
2. `from_prefill` converts that prefix to the canonical state.
3. The LAST prompt token is fed as the first one-token step on the converted state; its logit produces the first
   generated token. Every generated token is therefore predicted from canonical RABIT state.
4. Greedy argmax, one token per forward, each appended and aged as in PPL step 4–5; EOS stops and is not kept; at most
   16 (NIAH) or 32 (LongBench) tokens are kept.
5. No generated token stays BF16 beyond R4. The evaluator never executes a later prefill chunk.

**Comparison (semantics, not kernel schedule):**

| Row | Serving | PPL evaluator | Long-context evaluator | Classification |
|---|---|---|---|---|
| Initial prompt | first prefill chunk: the whole prompt when it fits the per-step token budget (16,384) | 1,024 context tokens, one forward | all but the last prompt token, one forward | DIFFERENT EXECUTION SCHEDULE (long-context holds back the last prompt token) |
| Initial attention | dense causal attention over the chunk's exact K / V | dense BF16 | dense BF16 | SAME SEMANTIC STATE (exact history) |
| Cache conversion | bulk encode of the chunk into closed pages + open group + residual | one-step `from_prefill` | one-step `from_prefill` | SAME SEMANTIC STATE (each side conforms to the reference: V1 for serving, V2–V6 for the evaluator; no direct serving-vs-evaluator comparison exists) |
| First continuation token | from the exact prefill (last prompt token is inside the dense chunk) | from the BF16 prefill logit; identical in both arms | from a one-token step on RABIT state | PPL: SAME; long-context: DIFFERENT EXECUTION SCHEDULE (the last prompt token attends to quantized history; serving would not) |
| Subsequent decode | append, then attend to the state at the token's own length | same | same | SAME SEMANTIC STATE |
| R4 aging | newest 4 tokens exact | same | same | SAME SEMANTIC STATE |
| Open group | exact open K kept, re-quantized as the group grows; V encoded on leaving R4 | recomputed from raw BF16 tokens at every length | same as PPL | SAME SEMANTIC STATE |
| Closed pages | packed bytes in allocator blocks, decoded in-kernel | decoded once, cached as values | same as PPL | SAME SEMANTIC STATE, different machinery |
| Chunked prefill (later chunks) | later-chunk tokens attend to the packed prefix + exact tail | never executed | never executed | NOT APPLICABLE — not covered by any quality measurement |
| Attention kernels | Triton partial kernels over packed pages | HF attention over decoded K / V cast to the model dtype (BF16) | same as PPL | DIFFERENT EXECUTION SCHEDULE; numerics not shown identical; no serving-vs-evaluator output comparison exists |

| ID | Claim | Status | Wording constraint |
|---|---|---|---|
| E1 | The evaluator reproduces the canonical stored-cache representation and aging semantics | SUPPORTED (V2–V6) | Say "reproduces the stored-cache representation and its aging", never "reproduces the serving execution" |
| E2 | Prompt prefill is dense BF16 in both evaluators; only tokens after the prefill attend to RABIT state | SUPPORTED (source) | Never "every token attends to quantized state". In PPL, 127 of 128 scored tokens per window depend on RABIT state |
| E3 | Quality of prompt tokens attending to an already packed prefix (serving later prefill chunks: prompts beyond 16,384 tokens, or a per-step budget shared among requests) is not measured | SUPPORTED (as a limitation) | Must be stated in Methodology and Limitations |
| E4 | Serving outputs and evaluator outputs were never compared end to end; attention numerics are not shown to be identical (the evaluator attends over decoded values cast to BF16) | SUPPORTED (as a limitation) | Do not claim the quality numbers are the serving engine's outputs; they are logical quality of the representation |

---

## 9. Legacy results — LEGACY LOGICAL-EVALUATOR RESULTS

These are valid measurements of the OLD logical evaluator. They must NEVER be presented as final
canonical quality evidence, and never pooled with canonical measurements. Use for history / provenance only.

| Experiment | Content | Evidence (commit) |
|---|---|---|
| Exp1 | BF16 / 8 / 4 / 3 / 2-bit quality–compression frontier | `quality_frontier/` (0667251) |
| Exp2 | Multilingual PPL frontier | `multilingual_frontier/` (1c516f9) |
| Exp7 | K-bit ablation | `ablations/k_bit/` (6ce79ed) |
| Exp8 | V-bit ablation | `ablations/v_bit/` (e13a9b6) |
| Exp9 | Group-size ablation | `ablations/group_size/` (599d059) |
| Exp10 | Residual-window ablation | `ablations/residual_window/` (01fe47e) |
| Exp11 | Metadata ablation | `ablations/metadata/` (ef41a16) |
| Exp12 (quality) | Llama N-expanded quality + paired CIs: PPL 7.514 → 7.601 (+1.16%); NIAH 57/57 vs 57/57; Passage Retrieval 100.0 → 99.5; HotpotQA 59.53 → 57.59 (−1.93, CI [−4.60, +0.07]); Qasper 35.46 → 34.88 | `variance/` (4f767ab) |
| Exp14 (legacy quality) | Qwen: PPL 6.784 → 299.62 (+4316.8%); NIAH 57/57 → 0/57; Passage Retrieval 99.5 → 38.73; HotpotQA 54.72 → 31.68; Qasper 34.55 → 7.67 | `second_model/quality/` (4049642) |
| Original release numbers | PPL 8.502 → 8.632 (+1.53%, N = 8); HotpotQA 60.6 → 55.2 (N = 20); multilingual; Qasper; TPOT 21.61 ms | `results/quality/`, `results/performance/`, `results/summary.json` (3b3b457) |

Consequence for the paper: **all ablations (Exp7–Exp11) and the bit-width frontier (Exp1–Exp2) exist
only under legacy semantics.** They may motivate the design historically (appendix), but they cannot
be cited as canonical justification of K3 / V2 / G32 / R4 / META8g64.

---

## 10. Conflicting / stale numbers found

| Quantity | Values in the repository | Use in the paper |
|---|---|---|
| Llama WikiText-2 PPL delta | +1.53% (N = 8, legacy, README) · +1.16% (N = 32, legacy Exp12) · **+1.56% (N = 32, canonical)** | canonical only |
| Llama HotpotQA | 60.6 → 55.2 (N = 20, legacy) · 59.53 → 57.59 (N = 100, legacy Exp12) · **59.30 → 58.27 (N = 100, canonical)** | canonical only; the "−5.4 F1" figure must not appear |
| Llama Passage Retrieval RABIT | 99.5 (legacy Exp12) · **100.0 (canonical)** | canonical only |
| Qwen PPL | 299.62 (legacy) · **112.39 (canonical)** | canonical; legacy only as the qualification in Q3b |
| RABIT TPOT (Llama, 2048) | 21.61 ms (legacy 5 reps, unmatched) · 28.15 (Exp3) · 17.30 (Exp3 replication) · 20.38 (Exp4) · 21.00 (Exp5) · **30.10 (Exp13)** | one session per table; Exp13 primary; never quote 21.61 |
| BF16 TPOT (Llama, 2048) | 21.79 (Exp3) · 13.03 (replication) · 15.34 (Exp4) · 15.56 (Exp5) · 24.19 (Exp13) | session-dependent; report relative overhead with its range (+24% to +35% at 2048 tokens) |
| "Compression" | logical 5.18–5.27× (legacy accounting) vs physical capacity **5.2785×** | capacity only; if logical bytes are mentioned, label them logical |
| Exp3/Exp4 decode implementation | manifests do not record the Stage3C implementation; Stage3C `shared_decode` evidence is dated after them | prefer Exp13 / Exp5 / Exp6 (finalized `shared_decode`) for latency figures |
| First PPL offline proof record | eb3f4b1 record passed, but its spy was nondeterministic | cite `offline_proof_record_v2.json` only |

---

## 11. Claims in existing docs that must NOT appear in the paper

Sources: `README.md`, `docs/METHOD.md`, `docs/REPRODUCIBILITY.md`, `results/summary.json` (all pre-date canonical-quality-v2).

- "5.28× physical KV capacity · 21.61 ms/token median TPOT · +1.53% WikiText-2 continuation PPL" (README headline): the TPOT and PPL figures are legacy / unmatched.
- WikiText-2 8.5020 → 8.6317 (+1.53%); Chinese +2.11%; Spanish +2.21%; Qasper 35.9 → 35.6; HotpotQA 60.6 → 55.2 — all legacy evaluator, small N.
- "logical KV compression approximately 5.18–5.27×" presented next to quality results as if it were the capacity result.
- `results/summary.json` "status: final unified quality complete" — superseded by canonical-quality-v2.
- Any phrasing that RABIT quality is validated across models, that Qwen degradation has a known cause, or that RABIT is faster than BF16.

These documents are historical release notes; the paper must be written from this map, not from them.

---

## 12. Novelty and claim boundaries

Do NOT claim as individually novel:

- asymmetric K/V quantization;
- residual (recent-token) windows kept at full precision;
- grouped / per-channel affine quantization;
- low-bit KV-cache compression as such.

Defensible contribution (to be argued, with the evidence above): **a target-bit-aware, physically
packed KV-cache serving system** that jointly accounts for

- asymmetric K/V precision (K3 / V2),
- residual state (R4) and its online aging into open groups and closed pages,
- metadata overhead (META8g64 — second-level quantization of the quantization parameters),
- physical allocator capacity (measured in the real engine, not a byte estimate),
- serving-system cost (measured latency and throughput cost, reported honestly),

Frozen central thesis (Story A): "RABIT-KV is a target-bit-aware, physically packed KV-cache serving system that jointly accounts for asymmetric K/V precision, residual state, metadata overhead, online cache aging, and allocator capacity. On the evaluated H100 setup, it substantially increases physical KV capacity relative to BF16, FP8, and the tested TurboQuant configuration, while exposing explicit latency and throughput tradeoffs."

The contribution is argued together with **a validation methodology**: an independent oracle, same-device bit-exact conformance on
real K/V, and a registered, gate-checked quality protocol that exposed and corrected an evaluator mismatch.

Boundaries:

- Capacity ≠ throughput: RABIT is slower per token and lower in throughput at every tested concurrency.
- Not a speedup paper; these costs are attributed to the current implementation (P4, P5), not claimed to be intrinsic to the representation, and no optimized result is claimed.
- Quality is model-dependent at a single frozen operating point: Llama small loss, Qwen severe loss.
- No claim of generality beyond two models, one GPU type, one engine snapshot.
- TurboQuant comparison is method-native and system-level; FP8 and TurboQuant quality were not measured.
- The Qwen latency result measures fallback kernels.

Supported headline conclusion: "RABIT's physical representation and canonical cache semantics transfer
across the tested Llama and Qwen geometries, but the quality robustness of a single frozen low-bit
operating point is strongly model-dependent."

---

## 13. Proposed paper structure (10-page compatible; no prose drafted)

| § | Section | Purpose | Core claims (IDs) | Likely figure / table | Appendix material |
|---|---|---|---|---|---|
| 1 | Introduction | KV memory limits concurrency and context; byte estimates ≠ servable capacity; state contribution and the honest trade-off | C5, L2, T3, Q1, Q2 (one sentence each) | — (optionally a teaser bar: capacity vs TPOT) | — |
| 2 | Background / Motivation | Paged KV allocators; why metadata and residual state matter at low bit-widths; logical vs physical accounting | S2, C1 vs "logical 5.18–5.27×" distinction | small schematic or none | prior low-bit KV methods detail |
| 3 | RABIT-KV Design | The operating point and the cache lifecycle (R4 → open group → closed page); META8g64; what "2-bit target" means | S1, S2, S3 | **Figure 1** lifecycle / physical layout | exact formulas, page byte breakdown |
| 4 | Physical Implementation | Packed pages in the vLLM allocator; decode / append / chunked-prefill paths; fallback paths for other geometries; correctness gate | V1, L6 (fallback), X4 (bottleneck stated plainly) | implementation diagram (optional, may merge into Fig. 1) | Stage3C diagnostics, kernel details |
| 5 | Experimental Methodology | Evidence classes; matched-session rule; registered attempts and validity gates; canonical evaluator and oracle conformance; BF16-only validity checks | V2–V7, Q8 | methodology table (small) or none | invalid attempts, amendments, identity audit, offline proofs |
| 6.1 | Physical capacity | Measured allocator capacity for four methods | C1–C6 | **Table 1** | per-session capacity logs |
| 6.2 | Serving latency / baselines | Per-token and TTFT cost vs BF16 / FP8 / TurboQuant | L1–L5 | **Figure 2** capacity vs latency | Exp3 / Exp4 / replication tables, order effects |
| 6.3 | Concurrency / context scaling (compact) | Higher offered concurrency, lower throughput; flat TPOT overhead to 16k; 32K prefill bottleneck; one-sentence bottleneck explanation | T1–T4, X1, X2, X4, P4, P5 | **Table 3** compact scaling summary (main) | full concurrency and context-scaling curves (Figure A1), per-point tables, JIT-contaminated attempts, profiling diagnostic and profiler perturbation (P1–P3) |
| 6.4 | Canonical quality | Llama PPL and long-context | Q1, Q4–Q8 | **Table 2** | per-window / per-example statistics, both HotpotQA scorers' details |
| 6.5 | Cross-model behavior | Same representation and capacity on Qwen; severe quality loss; fallback latency | C6, L6, Q2, Q3, Q3b | part of Table 2 (+ one sentence on capacity) | Qwen legacy quality history |
| 7 | Discussion / Limitations | What the system does and does not deliver | all NOT SUPPORTED items; §15 | — | — |
| 8 | Related Work | Position against KV quantization, FP8, TurboQuant, paging systems | novelty boundaries §12 | — | extended comparison |
| 9 | Conclusion | Restate the capacity–cost–quality trade-off and model dependence | supported headline conclusion | — | — |

---

## 14. Figure / table plan (minimum high-value set; none created yet)

| Item | Content | Data source |
|---|---|---|
| Figure 1 | RABIT cache lifecycle and physical page layout: BF16 residual (R4) → open group → closed 32-token page; K3 / V2 payload + META8g64 | design (S1–S3); page bytes from Exp13 / Exp14 summaries |
| Table 1 | Method / storage / capacity: BF16, FP8, TurboQuant, RABIT — bytes per token (theoretical and implied), blocks, capacity tokens, ratio to BF16 | Exp13 `observed_capacity`, `theoretical_bytes_per_token`, `capacity_ratios` |
| Figure 2 | Capacity vs latency trade-off: x = capacity ratio, y = TPOT (and TTFT) for the four methods, one session | Exp13 `latency_primary_cross_leg` |
| Table 3 (main, compact) | BF16 ceiling 47 vs RABIT sustained 64 at 8192 tokens; RABIT / BF16 throughput ratio range at 2048 and 8192 tokens; 32K TTFT 114.4 s vs 4.4 s | Exp6 summaries; Exp5 `per_context` |
| Figure A1 (APPENDIX; formerly Figure 3) | (a) throughput vs concurrency at 2048 and 8192 tokens, with BF16's ceiling at C64 / L8192 marked; (b) TPOT and TTFT vs context length, including the 32K point | Exp6 summaries; Exp5 `per_context` |
| Table 2 | Canonical quality: Llama + Qwen PPL (with CI); Llama NIAH, Passage Retrieval, HotpotQA primary and secondary (with CI) | `canonical_quality_final_record.json` |

Appendix candidates: full concurrency and context-scaling curves; profiling diagnostic details and the profiler perturbation discussion (P1–P6); full legacy ablation history (Exp1, 2, 7–11) clearly labelled legacy; correctness /
parity / conformance details and the CPU-vs-CUDA description; per-window and per-example quality statistics;
invalid attempts and amendments; identity audit; Exp3 / Exp4 session tables.

---

## 15. Limitations (to appear explicitly)

1. Only one frozen operating point (K3 / V2 / G32 / R4 / META8g64) was evaluated under canonical quality.
2. Qwen2.5-7B shows severe quality sensitivity at that point (+1555.9% PPL); mechanism undiagnosed.
3. No optimized kernel for the Qwen geometry: Qwen serving uses fallback paths (+80.7% TPOT).
4. A capacity advantage does not imply a throughput advantage: throughput is lower at every tested concurrency. Frozen wording: "The current implementation prioritizes correctness and physical packing over cross-request and cross-token kernel batching. Profiling indicates serialized per-sequence/per-token work and long-context closed-page processing as major implementation bottlenecks. Consequently, the observed capacity advantage does not translate into higher throughput in the evaluated workloads." These costs are not claimed to be intrinsic to the representation; a batched runtime / kernel design is future work.
5. Long-context quality is Llama-only.
6. HotpotQA: N = 100; the CI spans zero, which does not establish equivalence.
7. NIAH and Passage Retrieval are saturated for BF16; they cannot resolve small effects.
8. WikiText-2 PPL: N = 32 windows (4096 scored tokens per arm).
9. Canonical ablations were not rerun; all ablations are legacy-evaluator results.
10. CPU and CUDA canonical states are not bit-identical; validation relies on same-device oracle conformance.
11. Chunked-prefill bottleneck beyond 16,384 prompt tokens (32K point TTFT 114 s vs 4.4 s); long-context prefill also contains substantial closed-page device work, so the bottleneck is not solely host-loop overhead (P5).
12. FP8 and TurboQuant quality were not measured; the TurboQuant comparison is method-native.
13. One GPU type (H100 80GB), one engine snapshot, single-request latency for baselines; latency statistics are descriptive (no CIs).
14. Greedy-decoding run-to-run variability exists (one BF16 HotpotQA example differs from Exp12).

---

## 16. Biggest remaining paper risks

1. **The system is slower and lower-throughput than BF16 everywhere measured**, so the value proposition rests on capacity (more offered concurrency / longer aggregate context) — reviewers will ask for a workload where that wins end to end; none was measured.
2. **Qwen quality collapse** at the frozen operating point, with no diagnosis and no canonical ablation to show a workable alternative.
3. **No canonical ablations**: the design choices (K3 vs V2, G32, R4, META8g64) are justified only by legacy-evaluator experiments.
4. **32K prefill bottleneck** (114 s TTFT) undermines a long-context story unless framed as a known limitation. Performance optimization is closed (NO-GO, P6): the paper cannot answer "can it be made fast?" with a measurement, only with the diagnosed mechanism (P4, P5) and future work.
5. Baseline quality (FP8, TurboQuant) is absent, so the quality–capacity trade-off cannot be compared across methods.
