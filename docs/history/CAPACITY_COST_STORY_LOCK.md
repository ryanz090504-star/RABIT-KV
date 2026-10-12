# RABIT-KV — Paper Story Lock and Performance-Risk Diagnostic Plan

Status: **FINAL — Story A (capacity-first system) is frozen; performance optimization is NO-GO / CLOSED for this submission (section I).** Sections A–H are the pre-profiling plan and are kept unchanged except where marked. Quality evidence is frozen at `164c17f`; the evidence map is
`docs/MLSYS_PAPER_EVIDENCE_MAP.md` (`8b4121f`). This document runs nothing, changes no scientific
code and reports no new result. Every number is taken from committed evidence (paths given);
every statement about code is taken from the committed source (file and line given).

---

## A. The two candidate paper positions

### Story A — capacity-first system (**the position supported by current evidence**)

Thesis: RABIT-KV turns an aggressive target KV-bit budget into a real, physically packed serving
system by jointly accounting for asymmetric K/V precision, residual state, metadata overhead,
online cache aging, allocator capacity and serving cost.

Primary evidence, in this order:

1. Physical allocator capacity (not logical compression): 393,024 → 2,074,592 tokens.
2. Matched four-way comparison in one session (Exp13): BF16 / FP8 / TurboQuant / RABIT.
3. Ratios: **5.2785×** BF16, **2.6409×** FP8, **1.7242×** TurboQuant.
4. The matched latency cost of that capacity (RABIT is slower per token than all three).
5. Canonical quality: Llama +1.56% PPL, no score loss on NIAH / Passage Retrieval, about −1 F1 on
   HotpotQA (CI spans zero); Qwen +1555.9% PPL (model-dependent robustness).
6. The limits of the current implementation's throughput scaling and long-prefill path, stated plainly.

This is **not** a speedup paper. No headline compares RABIT speed with BF16.

### Story B — capacity + performance system (**CLOSED for this submission; see section I**)

Same system contribution, plus a performance claim. It is viable only if a narrowly scoped
implementation bottleneck can be removed without changing K3 / V2 / G32 / R4 / META8g64 or the
canonical quality semantics, and the improvement is then re-measured under the existing matched
protocols. Until that happens nothing from Story B may be written.

Recommendation as of current evidence: **Story A.** (Final after profiling: Story A; section I.)

---

## B. The performance risk, reconstructed from committed evidence

Rules: absolute timings are comparable only inside one row group (one session). "Ceiling" is the
admission bound DERIVED from measured allocator capacity: `floor(num_gpu_blocks / ceil((L + 32) / 32))`
— it is not a measured quantity except where "observed" is stated. Throughput exists only for Exp6
(closed loop, 256 measured requests per point); Exp13 / Exp5 / Exp14 are single-request runs.
Exp6 TTFT includes closed-loop queueing and is not comparable with single-request TTFT.

### B.1 Exp13 — matched four-way session (single request; Llama; 2048-token prompt, 32 output tokens)

Evidence: `results/mlsys2027/external_baseline/exp13/matched_capacity_latency_summary.json` (42c2799).

| Method | Capacity (tokens) | TPOT ms | TTFT ms | Wall ms | Implementation path | Qualification |
|---|---|---|---|---|---|---|
| BF16 | 393,024 | 24.19 | 59.99 | 809.7 | TRITON_ATTN | cross-leg medians, 2 legs × 30 samples; descriptive, no CI |
| FP8 | 785,568 | 25.75 | 63.43 | 861.2 | TRITON_ATTN, native per-tensor FP8 | no FP8 quality measured |
| TurboQuant | 1,203,200 | 27.79 | 55.03 | 916.4 | TURBOQUANT backend + FlashAttention v2 on 4 BF16 boundary layers | method-native SYSTEM comparison, not a quantizer-kernel comparison; no quality measured |
| RABIT | 2,074,592 | 30.10 | 144.65 | 1078.8 | TRITON_ATTN RABIT path: dense initial prefill + bulk exact append; one-token decode (`rabit2_online_decode_attention_triton`) | no chunked prefill at this length |

RABIT vs BF16 TPOT +24.4%; vs FP8 +16.9%; vs TurboQuant **+8.33%**. No throughput in this experiment.

### B.2 Exp5 — context scaling (single request, C = 1; Llama)

Evidence: `results/mlsys2027/context_scaling/attempt_2/context_scaling_summary.json` (e117e7a).
RABIT implementation: Stage3C `shared_decode`, query block 32. Capacity identical at every context
(393,024 vs 2,074,592).

| Prompt tokens | BF16 TPOT ms | RABIT TPOT ms | BF16 TTFT ms | RABIT TTFT ms | RABIT path for the prompt | Qualification |
|---|---|---|---|---|---|---|
| 512 | 15.33 | 21.26 | 32.1 | 95.6 | dense initial prefill (one chunk) | 15 samples per cell |
| 2048 | 15.56 | 21.00 | 59.8 | 119.9 | dense initial prefill | |
| 4096 | 15.60 | 21.03 | 144.6 | 171.3 | dense initial prefill | |
| 8192 | 15.72 | 20.66 | 409.5 | 305.2 | dense initial prefill | TTFT sign differs from shorter contexts; do not claim faster prefill |
| 16384 | 15.43 | 21.15 | 1280.8 | 687.8 | dense initial prefill (prompt = one 16,384-token chunk) | same caution |
| 32736 (32K point) | 15.56 | 29.38 | 4435.0 | **114,376.8** | first chunk dense; second chunk (16,352 query tokens) through Stage3C `shared_decode` | nominal model-limit point; not a maximum-feasible-context claim |

### B.3 Exp6 — concurrency (closed loop; Llama; 3 trials; medians)

Evidence: `concurrency_scaling/L2048/shadow_conditioned/summary.json` (8512566),
`concurrency_scaling/L8192/shadow_conditioned/combined_summary.json` (0f5f6ef).
`max_num_seqs = C`; `max_num_batched_tokens = 16384`. "Chunked" = the RABIT log marker
`RABIT2_STAGE3C_CHUNKED_PREFILL_ACTIVE` is present in that point's trial-1 log.

Prompt 2048 tokens — derived ceiling: BF16 188, RABIT 997 concurrent requests (both far above the tested C).

| C | BF16 req/s | RABIT req/s | RABIT / BF16 | BF16 TPOT ms | RABIT TPOT ms | BF16 TTFT s | RABIT TTFT s | RABIT chunked path |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.458 | 1.022 | 0.70 | 20 | 26 | 89.5 | 123.2 | no |
| 4 | 4.592 | 1.612 | 0.35 | 21 | 64 | 27.4 | 75.5 | no |
| 8 | 7.218 | 1.844 | 0.26 | 21 | 111 | 17.5 | 67.6 | no |
| 16 | 10.042 | 2.015 | 0.20 | 35 | 219 | 12.4 | 61.4 | yes |
| 32 | 12.911 | 2.083 | 0.16 | 52 | 432 | 9.7 | 57.2 | yes |
| 64 | 14.361 | 1.776 | 0.12 | 108 | 985 | 8.6 | 62.7 | yes |

Prompt 8192 tokens — derived ceiling: BF16 **47**, RABIT 252.

| C | BF16 req/s | RABIT req/s | RABIT / BF16 | BF16 TPOT ms | RABIT TPOT ms | BF16 TTFT s | RABIT TTFT s | RABIT chunked path | Outcome |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.937 | 0.828 | 0.88 | 21 | 28 | 134.2 | 155.6 | no | both sustained |
| 4 | 1.801 | 1.259 | 0.70 | 46 | 82 | 70.8 | 99.5 | yes | both sustained |
| 8 | 2.101 | 1.385 | 0.66 | 96 | 162 | 60.1 | 91.8 | yes | both sustained |
| 16 | 2.282 | 1.424 | 0.62 | 188 | 312 | 56.0 | 92.6 | yes | both sustained |
| 32 | 2.346 | 0.870 | 0.37 | 372 | 1,033 | 54.1 | 145.6 | yes | both sustained |
| 64 | 2.366 | 0.323 | (0.14) | 571 | 6,487 | 54.0 | 327.4 | yes | BF16 target NOT reached (observed max in flight 47, 3/3 trials); RABIT sustained 64 |

Qualification: all 72 points passed integrity; no preemption; concurrency is overlapping in-flight
concurrency, not GPU-resident concurrency. The C64 / L8192 ratio compares RABIT at 64 in flight with
BF16 at 47 in flight.

### B.4 Exp14 — Qwen2.5-7B fallback serving (single request; 2048-token prompt)

Evidence: `results/mlsys2027/second_model/serving/capacity_latency_summary.json` (7dfaee4).

| Method | Capacity (tokens) | TPOT ms | TTFT ms | Path | Qualification |
|---|---|---|---|---|---|
| BF16 | 902,656 | 26.96 | 58.3 | TRITON_ATTN | |
| RABIT | 4,764,640 (5.2785×) | 48.70 (+80.7%) | 161.6 | exact FALLBACK paths (decode 64/64, append 62/62, V2 quantizer 62/62) | measures untuned fallback kernels; NOT a transfer result for the Llama-optimized kernels |

---

## C. The two bottleneck questions, mapped to code paths (no cause inferred from timing)

Source: `vllm-kvquant/vllm/v1/attention/backends/triton_attn.py`, `TritonAttentionImpl._forward_rabit_kv2`
(lines 755–968), and `vllm-kvquant/vllm/v1/attention/ops/rabit_kv2*.py`. The method's own docstring reads:
"Initial prefill chunks use dense causal attention; later chunks are evaluated token-by-token against the
compressed prefix plus exact tail. This is correctness-first; Stage 4 will batch the tail/partial kernels
before latency benchmarking."

Structure established by the source (per attention layer, per scheduler step):

| Stage | What the code does | Where |
|---|---|---|
| Batching behaviour | A Python loop over the requests of the step (`for seq_idx, (req_id, q_len, context_len) in enumerate(active_batch)`); each request owns a `Rabit2SingleSequenceRuntime`; kernels are launched per request, not once for the batch | `triton_attn.py:825` |
| Initial prefill (context 0, q_len > 1) | `rabit2_bulk_append_exact` (quantize + page encode + metadata for the whole chunk), then dense causal `context_attention_fwd` on that one sequence | `triton_attn.py:863–890`; `rabit_kv2.py` (`rabit2_bulk_append_exact`, Stage4D2 page encoders) |
| Non-initial prefill chunk (context > 0, q_len > 1) — "Stage3C" | `rabit2_stage3c_forward_shared_decode`: `Rabit2CausalChunkPlan` (V2 quantization, page encode), closed-page partial kernel over (closed pages × query-head groups × ceil(n / 32) query blocks), plus per-query tail work; else a per-token loop | `triton_attn.py:920–966`; `rabit_kv2_stage3c_shared_decode.py` |
| Decode (q_len = 1) | `runtime.append` (cache aging: residual → open group → closed page; K statistics / codes, K3 pack, META8g64, V2 aging) then `rabit2_online_decode_attention_triton` (closed-page partial kernel + tail partial + reduce) | `triton_attn.py:948–966`; `rabit_kv2.py:1551`, `_rabit2_final_fast_decode_append` (3885) |
| Dequantization | inside the attention kernels (packed pages are decoded in-kernel; the open tail is materialized per call) | `_rabit2_closed_page_partial_kernel`, `runtime.tail_materialize` |
| Python / control overhead | per request per layer: dict lookup, tensor slicing, small-tensor creation, kernel launches | `_forward_rabit_kv2` loop body |
| Triton kernels | closed-page partial, tail partial, reduce, K stats/codes, K3 pack, META8g64 (4 launches), V2 age, shift-recent | `rabit_kv2.py` |

**Question 1 — why does RABIT throughput scale poorly with concurrency despite larger capacity?**
Relevant paths: the per-request loop (batching behaviour); decode (append/aging + attention) executed once
per in-flight request per layer; initial prefill per request; and, whenever `C × L > 16384`, the Stage3C
non-initial-chunk path. Capacity governs admission only; it does not appear in any of these paths.

**Question 2 — why is the 32K RABIT TTFT ≈ 114 s versus ≈ 4.4 s for BF16?**
Relevant path: the prompt is processed as a 16,384-token first chunk (dense) and a 16,352-token second
chunk, and the second chunk runs through Stage3C `shared_decode` (chunk plan + closed-page partials over
511 closed pages + per-query tail work), in each of the 32 layers.

---

## D. What existing committed evidence already explains

| Finding | Evidence | Status |
|---|---|---|
| The 32K slowdown is produced by the non-initial-chunk (Stage3C) path, not by decode or the first chunk | `diagnostics/stage3c_cliff/summary.json` (356b2e3): no excess at second-chunk q_len 1 (decode path by source), excess begins at q_len > 1, ≈ 13–15 ms per second-chunk query token for the then-current implementation, approximately stable from 512 to 8192, no discontinuity at 32 | ESTABLISHED (diagnostic) |
| For the `reference` and `tile32` Stage3C implementations, GPU time is dominated by closed-page work | `diagnostics/stage3c_component_profile/profile_analysis.json` (c75ed71): `closed_page_gpu` = 60–70% of the Stage3C GPU span at q_len 32 / 512 / 2048; host side: per-query launch work is the largest component (30–50% of host wall), profiler overhead 30–40% | ESTABLISHED for those two implementations (shares only; not latency) |
| `tile32` did not remove the bottleneck | `diagnostics/stage3c_tile32_benchmark/summary.json` (6edae6d): "only a modest Stage3C improvement ... does not materially eliminate the long-context chunked-prefill bottleneck" | ESTABLISHED |
| `shared_decode` (query block 32) roughly halved second-chunk TTFT but the 32K point is still ≈ 115 s | `stage3c_shared_decode_final_benchmark` (c2b7627), `..._qb_tiebreak` (161c489): q_len 2048 TTFT ≈ 12.5 s vs 26.9 s (tile32) and 29.2 s (reference); `..._q16352_feasibility` (97fef5f): 114.75 s for one request | ESTABLISHED (one request per point; descriptive) |
| Under concurrency the Stage3C path is entered exactly when `C × L > 16384` | Exp6 logs: marker present at L2048 for C ≥ 16 and at L8192 for C ≥ 4; absent at L2048 C ≤ 8 and L8192 C = 1 | ESTABLISHED (path entered; its time share is not measured) |
| RABIT throughput is already far below BF16 where the Stage3C path is NOT entered | Exp6 L2048: C = 4 ratio 0.35, C = 8 ratio 0.26, with RABIT TPOT 26 → 64 → 111 ms as C goes 1 → 4 → 8 while BF16 TPOT stays 20–21 ms | ESTABLISHED as an observation. Together with the source structure (per-request loop) it shows that RABIT decode is not batched across requests; the share of time per stage is NOT measured |

## What remains unknown

(As of the plan. The profiling diagnostic later addressed items 1–4 qualitatively; see section I.)

1. For the **final `shared_decode` implementation**, how the 32K second-chunk time divides between the
   closed-page kernel, chunk-plan encoding, per-query tail work and host launch overhead. The only component
   profile predates `shared_decode`.
2. Under concurrency, how step time divides between (a) per-request decode attention, (b) append / aging
   work, (c) initial prefill, (d) Stage3C chunks of other requests, (e) host loop overhead, and (f) the rest
   of the model (MLP etc., which is batched). No stage-level timing exists for any multi-request point.
3. Whether the collapse from C = 16 to C = 64 at L8192 (1.424 → 0.870 → 0.323 req/s) is the same mechanism
   as Question 2 (Stage3C chunks scheduled between decode steps) or a separate one.
4. Whether the bottlenecks are implementation artifacts (per-request launches, token-by-token chunk
   evaluation) or intrinsic to the representation (in-kernel page decode cost per attended token).

Existing evidence therefore explains **where** the 32K time goes at path level and **that** decode is
unbatched; it does not isolate the dominant component for the shipped implementation or for concurrency.
A minimal profile is justified.

---

## E. Minimal profiling plan (as proposed; the run that was actually executed is recorded in section I)

Not a quality experiment: fixed prompts, no scoring, greedy output hashes checked only against the
already-recorded hashes (functional sanity). No RABIT parameter changes. No optimization. No variants.

| | Case 1 — concurrency | Case 2 — 32K prefill |
|---|---|---|
| Setup | Llama, 8192-token prompts, **C = 32** (an accepted Exp6 point; both methods sustain it), same engine arguments as Exp6 L8192 | Llama, 32,736-token prompt + 32 output tokens, the accepted Exp5 B32768 cell (prompt hash `dbc5bd88…`), `shared_decode`, query block 32 |
| Load | shortened closed loop: C conditioning requests + 64 measured requests (Exp6 used 256) | 1 conditioning request at 512 tokens + 1 profiled request (Exp5 used 5 warmups + 15 reps) |
| Controls | one BF16 run of the same shape for the model-forward baseline | BF16 single request for the baseline |
| Attribution wanted | per scheduler step and per layer: host and GPU time in `_forward_rabit_kv2` split into initial prefill / Stage3C chunk / decode attention / append-aging / loop overhead, with request counts; attention vs rest-of-model | Stage3C span split into chunk plan, closed-page partial, tail prep (K stats/codes, K3 pack, META8g64), tail partial, reduce, host launch, unattributed |

Instrumentation:

- Extend the existing opt-in profiler (`rabit_kv2_stage3c_profile.py`: CUDA event pairs for GPU leaves,
  `perf_counter` exclusive windows for host; enabled only by `VLLM_RABIT2_STAGE3C_COMPONENT_PROFILE=1`).
  It wraps kernels by monkey-patching, so **`rabit_kv2.py` (SHA pinned by the frozen harnesses) need not be
  edited**. Needed additions: wrappers for the `shared_decode` kernel and for the decode / append entry
  points, and a per-step record around `_forward_rabit_kv2`.
- Files that would change: `vllm-kvquant/vllm/v1/attention/ops/rabit_kv2_stage3c_profile.py` (extended);
  new benchmark files under `benchmarks/mlsys2027/` (a profile worker, Modal app, runner, static tests).
  Preferably no edit to `triton_attn.py` (its accepted hash is recorded in Exp6); if a hook is unavoidable
  it must be a no-op when the environment variable is unset and the frozen correctness gate must be rerun.
- Can instrumentation alter timing materially? **Yes.** In the existing profile the profiler itself is
  30–40% of host wall, and each scope synchronizes the device. Profiled runs therefore yield **shares per
  domain (host and GPU kept separate), never latency**; an unprofiled run of the same shape is the
  reference for absolute time.
- Tool choice: CUDA events + `perf_counter` (already validated in this code base) are appropriate.
  `torch.profiler` is acceptable as a cross-check for kernel-level GPU time on one short run, but adds
  host overhead and large traces. Nsight Systems is not recommended (binary install and privileges in the
  container; not needed for stage-level attribution).

Estimated cost: one H100 80GB.

| Item | GPU minutes |
|---|---|
| Correctness gate + engine starts (4 engines) | ≈ 8 |
| Case 1: RABIT profiled + RABIT unprofiled + BF16 (64 requests each at C = 32) | ≈ 12–18 |
| Case 2: RABIT profiled (≈ 2–4 min per request) + RABIT unprofiled (≈ 2 min) + BF16 | ≈ 8–10 |
| Margin (image build, conditioning, cleanup) | ≈ 10 |
| **Total** | **≈ 40–45 GPU-minutes; budget ≤ 1 GPU-hour** |

Engineering before launch: profiler extension + static tests + a CPU/unit check that wrapping is a no-op
when disabled: roughly one working day. A plumbing smoke test (no measurement) is advisable first.

---

## F. GO / NO-GO rule (fixed before any new profile is seen; amended before profiling)

Amendment (before any profile was run): the earlier proposed numeric boundaries ("one component >= 50%",
"two components >= 60%") are WITHDRAWN. They must not be used as hard decision boundaries, because the
profiler itself materially perturbs host timing and synchronizes GPU work.

**GO for a narrow optimization only if ALL hold:**

1. One or at most two concrete IMPLEMENTATION components account for a clear majority of the relevant GPU
   or host-time domain.
2. The attribution is directionally consistent across CUDA-event instrumentation, source / control-flow
   evidence and, where practical, one `torch.profiler` cross-check.
3. The bottleneck is not intrinsic to K3 / V2 / G32 / R4 / META8g64 semantics.
4. A fix can be implemented without changing the quantization policy, the cache semantics, the physical
   layout definition, the quality protocol or the model policy.
5. The engineering scope is small enough to complete and revalidate before the MLSys deadline.

**NO-GO** if the attribution is diffuse, contradictory, architecture-level, or would require redesign or
retuning.

A GO authorizes only a specific implementation fix and its re-measurement. It does NOT authorize Story B
claims. If NO-GO: performance is frozen and Story A is final.

Profiling cases (amended, fixed before profiling; Llama-3.1-8B only; no quality scoring):

| Case | Workload | Purpose |
|---|---|---|
| 1 | 2048-token prompts, C = 8 (below the chunked-prefill threshold), accepted Exp6 shape, matched BF16 | concurrency / per-request decode behaviour without chunked prefill as a confound |
| 2 | 8192-token prompts, C = 32, accepted Exp6 shape, matched BF16 | combined high-concurrency path with chunked prefill active |
| 3 | 32,736-token prompt, single request, accepted Exp5 setup: one unprofiled RABIT, one profiled RABIT, one BF16 | second-chunk breakdown for the shipped `shared_decode` implementation |

Profiled timing is for attribution only; accepted Exp5 / Exp6 numbers remain authoritative for performance.

---

## G. Main-paper framing regardless of profiling

- Never headline RABIT versus BF16 speed.
- Exp13 is the primary trade-off table: it is the only matched session covering BF16, FP8, TurboQuant and RABIT.
- Present the result as a physical-capacity / latency frontier.
- TurboQuant statement, with its qualification kept verbatim: RABIT provides **1.7242×** the observed
  physical KV capacity of the tested TurboQuant configuration (`turboquant_k3v4_nc`: 28 TurboQuant layers +
  4 BF16 boundary layers) at **+8.33%** median single-request TPOT (30.10 vs 27.79 ms). This is a
  METHOD-NATIVE SYSTEM comparison (TurboQuant uses its own backend + FlashAttention v2 on its BF16 boundary
  layers; the others use TRITON_ATTN), not a quantizer-kernel comparison; single request, 2048-token
  prompt, 32 output tokens; no throughput, quality or generality claim; two legs per condition, no CI.
  RABIT's TTFT in that session is higher (144.65 vs 55.03 ms) and must be shown beside the TPOT.
- The capacity sentence must always be followed by the cost: slower per token than BF16 (+24.4% in Exp13)
  and lower throughput at every tested concurrency.

---

## H. Figure priority review

Recommendation: **the full concurrency / context-scaling figure goes to the appendix; the main paper
keeps a compact statement of the same facts.**

Reasoning: the curves are primarily limitation evidence about the current implementation, not the
contribution. A full-width main figure would make a secondary bottleneck look like the thesis. But the
facts are material to the capacity claim and must not be hidden, so the main paper must contain:

1. one compact table (or a two-line inset in the trade-off figure) with: BF16 ceiling 47 vs RABIT
   sustained 64 at 8192 tokens; RABIT / BF16 throughput ratio range (0.88 → 0.37 at 8192 tokens,
   0.70 → 0.12 at 2048 tokens); the 32K TTFT (114 s vs 4.4 s);
2. the required sentence: RABIT supports higher offered concurrency because of capacity, but the current
   implementation has lower throughput at every tested concurrency;
3. a pointer to the appendix figure and tables.

Revised main set: Figure 1 (lifecycle / layout), Table 1 (capacity, four methods), Figure 2 (capacity vs
latency frontier, Exp13), Table 2 (canonical quality), plus the compact scaling table above. The profile
yielded NO-GO (section I), so this recommendation is final.

---

## Summary

- Final position: **Story A** (frozen; section I).
- The performance risk is fully reconstructed in section B from accepted evidence.
- Existing diagnostics localize the 32K cost to the non-initial-chunk path and show that decode is not
  batched across requests; they do not attribute time by component for the shipped implementation.
- The profile was run once (valid; about 85 GPU-minutes). Outcome: **NO-GO for performance optimization**; performance evidence is frozen at the accepted Exp5 / Exp6 / Exp13 / Exp14 results (section I).


---

## I. FINAL DECISION — profiling result, NO-GO, Story A frozen

This section is the final record. Where it differs from sections A–H above (which are kept as the
pre-profiling plan), this section governs.

### I.1 Outcome of the profiling diagnostic

Evidence: `results/mlsys2027/diagnostics/perf_risk_profile/attempt_1/` — raw run `4d07cf6`
(`summary.json`, `legs.json`, `remote_session.log`, `manifest.json`), overhead-corrected addendum `dadc2a0`
(`corrected_attribution.json`); harness `33c00b8`, `601b897`, `ac9cfbb`. One run, one H100, 13 legs,
Llama-3.1-8B, no quality scoring, no vLLM source file edited (the profiler attached from outside).

1. **The diagnostic was VALID.** All validity gates passed: 13/13 legs exited 0, the frozen correctness
   gate passed, and the three serving sources in the image equal the hashes of the accepted serving evidence.
2. **The profiler materially perturbed RABIT execution.** Profiled / unprofiled wall time was 3.74×
   (Case 1), 6.29× (Case 2) and 8.82× (Case 3). After removing the profiler's recorded bookkeeping a
   residual of 1.35× to 1.63× remains. Component percentages are therefore **DIAGNOSTIC ONLY**: they must
   never be presented as precise measurements and must never replace the accepted Exp5 / Exp6 / Exp13
   performance measurements.
3. **The unprofiled diagnostic legs reproduced the accepted performance points closely enough to validate
   workload identity** (shortened runs; not new performance results):

   | Case | Unprofiled diagnostic leg | Accepted point |
   |---|---|---|
   | 1 (2048 tokens, C = 8) | RABIT 1.848 req/s; BF16 7.473 req/s | Exp6: 1.844; 7.218 |
   | 2 (8192 tokens, C = 32) | RABIT 0.900 req/s; BF16 2.265 req/s | Exp6: 0.870; 2.346 |
   | 3 (32,736-token prompt) | RABIT TTFT 118.3 s; BF16 4.44 s | Exp5: 114.4 s; 4.435 s |

4. **Dominant qualitative mechanism: serialized / unbatched per-sequence and per-token processing.**
   - request-wise RABIT attention path (a Python loop over the requests of a step, per layer);
   - per-token open-tail work (the open K group is re-quantized and its metadata re-encoded for every
     request, layer and token before the tail attention kernel);
   - append / cache-aging work, executed per request, layer and token;
   - repeated launch / control overhead: about 10.8 kernel launches per (layer, request, token) in decode
     and about 6.9 per (layer, token) inside a non-initial prefill chunk (launch counts are exact; time
     shares are not).

   The same mechanism appears in all three cases: through decode at 2048 tokens / C = 8, through both
   chunked prefill and decode at 8192 tokens / C = 32, and through the 16,352-token second chunk
   (523,264 layer-token iterations) at the 32K point. A single `torch.profiler` cross-check (Case 1) points
   the same way: the RABIT kernels account for a small part of device time while the attention path's own
   host time is about half of the traced wall.
5. **The 32K path additionally has substantial closed-page device work** (on the order of 70 s of device
   span for the shared closed-page decode kernel in the profiled leg; approximate). Removing host
   serialization alone would therefore not eliminate the long-context cliff.
6. **Fixing these issues is NOT a narrow optimization.** It requires batching across requests and tokens,
   restructuring the per-sequence runtime, and / or new kernels.
7. **Such changes would modify the currently frozen serving implementation** (`rabit_kv2.py`,
   `triton_attn.py`, the Stage3C modules, whose hashes are recorded by the accepted serving evidence) and
   would require major revalidation of correctness and of every accepted serving experiment.
8. **Decision under the rule fixed in section F: NO-GO FOR PERFORMANCE OPTIMIZATION.**

### I.2 Required wording

Do NOT write "profiling found no dominant bottleneck". Write:

> "Profiling identified serialized per-request/per-token processing as the dominant qualitative bottleneck family, but eliminating it requires a substantial implementation redesign rather than a narrow optimization."

For the 32K path additionally state:

> "long-context prefill also contains substantial closed-page device work, so the bottleneck is not solely host-loop overhead."

Do not claim these costs are intrinsic to RABIT's representation. Frame a batched runtime / kernel design
as future work.

### I.3 Authoritative performance evidence (frozen)

The authoritative performance results are ONLY the accepted unprofiled experiments:

- Exp5 context scaling (`context_scaling/attempt_2/`, e117e7a);
- Exp6 concurrency (`concurrency_scaling/L2048/…` 8512566, `L8192/…` 0f5f6ef);
- Exp13 matched BF16 / FP8 / TurboQuant / RABIT comparison (`external_baseline/exp13/`, 42c2799);
- Exp14 Qwen serving where applicable (`second_model/serving/`, 7dfaee4).

The profiling diagnostic is explanatory evidence only. No accepted timing is replaced by a
profiler-derived value, and the unprofiled diagnostic legs are not additional performance results.

### I.4 Final Story A — central systems thesis (frozen)

> "RABIT-KV is a target-bit-aware, physically packed KV-cache serving system that jointly accounts for asymmetric K/V precision, residual state, metadata overhead, online cache aging, and allocator capacity. On the evaluated H100 setup, it substantially increases physical KV capacity relative to BF16, FP8, and the tested TurboQuant configuration, while exposing explicit latency and throughput tradeoffs."

The paper is NOT a speedup paper. Primary headline evidence:

- **5.2785×** physical KV capacity vs BF16;
- **2.6409×** vs FP8;
- **1.7242×** (1.724×) vs the tested TurboQuant configuration;
- matched TurboQuant TPOT cost of **+8.33%** (with its method-native qualification, section G);
- canonical Llama quality retention;
- model-specific Qwen quality sensitivity.

Story B is closed for this submission: no performance claim beyond the accepted measurements may be made.

### I.5 Performance limitation wording (frozen)

> "The current implementation prioritizes correctness and physical packing over cross-request and cross-token kernel batching. Profiling indicates serialized per-sequence/per-token work and long-context closed-page processing as major implementation bottlenecks. Consequently, the observed capacity advantage does not translate into higher throughput in the evaluated workloads."

### I.6 Main paper vs appendix (frozen)

| Main paper | Appendix |
|---|---|
| capacity comparison | full concurrency curves |
| matched BF16 / FP8 / TurboQuant / RABIT trade-off | full context-scaling curves |
| compact latency / concurrency limitation summary | profiling diagnostic details |
| canonical quality table | profiler perturbation discussion |
| | legacy evaluator history |
| | detailed correctness / parity records |

The 32K result and the throughput limitation are NOT hidden: both are stated clearly in the main paper's
compact evaluation text and in its limitations.

### I.7 Experimental work — closed

| Track | Status |
|---|---|
| QUALITY EXPERIMENTS | COMPLETE |
| SERVING EXPERIMENTS | COMPLETE |
| PERFORMANCE PROFILING | COMPLETE |
| PERFORMANCE OPTIMIZATION | NO-GO / CLOSED |

No benchmark, profile, kernel optimization or change to the scientific or serving implementation is to be
made for this submission. No experiment is reopened unless a concrete paper-review need is identified later.
The next artifact is `docs/MLSYS_PAPER_BLUEPRINT.md`.
