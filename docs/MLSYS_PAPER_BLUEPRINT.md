# RABIT-KV — MLSys Paper Blueprint

Status: paper-preparation document; **no prose is drafted here**. Story A (capacity-first system) is
frozen (`docs/MLSYS_PAPER_STORY_LOCK.md`, section I). Claim IDs refer to
`docs/MLSYS_PAPER_EVIDENCE_MAP.md`; every number in the paper must be taken from the evidence files
named there, never from this blueprint or from earlier prose. Experimental work is closed.

---

## 1. One-sentence thesis

Frozen form (story lock I.4):

> "RABIT-KV is a target-bit-aware, physically packed KV-cache serving system that jointly accounts for
> asymmetric K/V precision, residual state, metadata overhead, online cache aging, and allocator capacity.
> On the evaluated H100 setup, it substantially increases physical KV capacity relative to BF16, FP8, and
> the tested TurboQuant configuration, while exposing explicit latency and throughput tradeoffs."

Working one-sentence form for the abstract's first line (same content, to be worded at drafting time):
RABIT-KV turns an aggressive KV bit budget into measured allocator capacity in a real serving engine —
5.2785× BF16 — and reports what that capacity costs in latency, throughput and model-dependent quality.

The paper is not a speedup paper.

---

## 2. Contribution bullets

| # | Contribution | Claims |
|---|---|---|
| 1 | **A target-bit-aware KV representation with a full cache lifecycle**: one frozen operating point (K3 / V2 / G32 / R4 / META8g64) that jointly fixes asymmetric K/V precision, the exact residual window, online aging into open groups and closed pages, and second-level quantization of the quantization metadata; "2-bit target" is a label for about 3.03 stored bits per element | S1, S2, S3 |
| 2 | **A physically packed implementation in a production serving engine**, whose benefit is measured as allocator capacity rather than estimated bytes: 5.2785× BF16, 2.6409× FP8, 1.7242× the tested TurboQuant configuration on Llama-3.1-8B, and the same 5.2785× on a second model geometry | C1–C6, V1 |
| 3 | **A matched four-method capacity / cost evaluation, with the costs reported**: per-token latency in one session (+8.33% TPOT against the tested TurboQuant configuration, +24.4% against BF16), lower throughput at every tested concurrency, a long-prefill cliff at the 32K point, and a diagnosed bottleneck family | L1–L6, T1–T4, X1–X4, P4, P5 |
| 4 | **A validation methodology for low-bit KV quality claims**: an independent oracle, same-device bit-exact conformance on real K/V, and registered gate-checked quality runs, yielding canonical quality for Llama (small loss) and Qwen (severe loss) and exposing an evaluator mismatch in earlier results | V2–V7, Q1–Q8, Q3b |

Three-bullet fallback if space is short: merge 2 and 3.

---

## 3. Candidate titles

1. RABIT-KV: A Physically Packed, Target-Bit-Aware KV Cache for LLM Serving
2. From Bit Budgets to Servable Capacity: Physically Packed Low-Bit KV Caches in a Real Serving Engine
3. RABIT-KV: Measuring What a 2-Bit-Target KV Cache Buys and Costs in LLM Serving
4. Capacity Is Not Throughput: A Physically Packed Low-Bit KV Cache and Its Serving Trade-offs
5. RABIT-KV: Accounting for Residuals, Metadata and Cache Aging in Low-Bit KV-Cache Serving

Recommendation: 1 for neutrality, 4 if the submission should lead with the trade-off.

---

## 4. Section structure, claims and page budget (10 pages of body)

| § | Section | Subsection | Must establish | Claim IDs | Float | Pages |
|---|---|---|---|---|---|---|
| — | Abstract | — | thesis; the three capacity ratios; +8.33% TPOT vs tested TurboQuant; Llama retention; Qwen sensitivity; throughput not improved | C5, L5, Q1, Q2, T2, T4 | — | 0.2 |
| 1 | Introduction | 1.1 problem | KV memory bounds admitted concurrency and aggregate context; byte estimates are not servable capacity | C1, S2 | — | 1.1 |
| | | 1.2 what RABIT-KV is | the operating point and that it is physically packed in the engine | S1, C3 | — | |
| | | 1.3 result and its cost, one paragraph | capacity ratios, latency cost, lower throughput, Llama vs Qwen quality | C5, L2, L5, T2, T4, X4, Q1, Q2 | — | |
| | | 1.4 contributions | the four bullets of section 2 | — | — | |
| 2 | Background and Motivation | 2.1 paged KV allocation | capacity is a number of allocator blocks | C1 | — | 0.7 |
| | | 2.2 logical vs physical accounting | metadata and residual state move the real bits per element away from the nominal bit-width | S2, S3 | — | |
| 3 | RABIT-KV Design | 3.1 operating point | K3, V2, G32, R4, META8g64 defined exactly | S1 | Figure 1 | 1.5 |
| | | 3.2 cache lifecycle | residual → open group → closed page; online aging | S1, V6 | Figure 1 | |
| | | 3.3 metadata and the real bit cost | 776 vs 4,096 bytes per token-layer; about 3.03 bits per element | S2, S3 | — | |
| 4 | Physical Implementation | 4.1 packed pages in the allocator | pages occupy scheduler-assigned blocks; capacity follows | C3, V1 | — | 1.0 |
| | | 4.2 execution paths | dense initial prefill + bulk append; one-token decode; non-initial prefill chunks; fallback paths for other geometries | L6, X4 | — | |
| | | 4.3 what is not batched | per-request, per-token processing in the current implementation (design fact, stated before the results) | P4 | — | |
| 5 | Methodology | 5.1 evidence classes and matched sessions | physical vs logical evidence; timings compared only within one session | L1 (qualification) | — | 0.9 |
| | | 5.2 canonical evaluator and oracle conformance | CPU and same-device CUDA bit-exact conformance; no cross-device tolerance | V2–V7 | — | |
| | | 5.3 registered runs and validity checks | registered attempts, BF16-only validity checks, an invalid attempt excluded | Q8 | — | |
| 6 | Evaluation | 6.1 physical capacity | four methods, measured blocks and tokens, ratios; second geometry | C1–C6 | Table 1 | 0.6 |
| | | 6.2 matched latency trade-off | Exp13 four-way session; TurboQuant statement with its qualification; TTFT shown beside TPOT | L1–L5 | Figure 2 | 0.9 |
| | | 6.3 concurrency and context scaling (compact) | higher offered concurrency (BF16 ceiling 47 vs RABIT 64 at 8192 tokens); lower throughput at every tested concurrency; flat decode overhead to 16k; 32K TTFT 114.4 s vs 4.4 s; one-sentence bottleneck explanation | T1–T4, X1, X2, X4, P4, P5 | Table 3 | 0.7 |
| | | 6.4 canonical quality | Llama PPL, NIAH, Passage Retrieval, HotpotQA (both scorers) | Q1, Q4–Q8 | Table 2 | 0.8 |
| | | 6.5 cross-model behaviour | same capacity ratio on Qwen; severe quality loss; fallback-path latency | C6, L6, Q2, Q3, Q3b | Table 2 | 0.4 |
| 7 | Discussion and Limitations | 7.1 capacity is not throughput | frozen limitation wording (story lock I.5); not intrinsic; batched runtime as future work | T2, T4, X4, P4–P6 | — | 0.7 |
| | | 7.2 quality scope | one operating point; model dependence; no canonical ablations; saturated suites; N | Q2, Q3, limitations 1–9 | — | |
| | | 7.3 comparison scope | method-native TurboQuant comparison; no FP8 / TurboQuant quality; one GPU, one engine snapshot | L4, L5, limitations 12–13 | — | |
| 8 | Related Work | — | position against KV quantization, FP8 caches, TurboQuant, paged serving; novelty boundaries of evidence map §12 | — | — | 0.5 |
| 9 | Conclusion | — | capacity–cost–quality trade-off and model dependence | supported headline conclusion (evidence map §12) | — | 0.2 |
| | **Total** | | | | | **10.2** → trim §2 or §8 to reach 10.0 |

Floats are counted inside the section budgets. References and appendix are outside the 10 pages.

---

## 5. Main-paper figures and tables

| Item | Content | Data source | What it must prove | What it must not suggest |
|---|---|---|---|---|
| Figure 1 | Cache lifecycle and physical page layout: exact residual (R4) → open group → closed 32-token page; K3 / V2 payload and META8g64 metadata with byte counts | S1–S3; page bytes from Exp13 / Exp14 summaries | The representation is a complete, online cache with accounted metadata, not a quantizer alone | That 2 bits per element are stored |
| Table 1 | BF16, FP8, TurboQuant, RABIT: bytes per token, allocator blocks, capacity in tokens, ratio to BF16; Qwen capacity row | Exp13 `observed_capacity`, `capacity_ratios`; Exp14 serving summary | Capacity is measured in the engine: 5.2785× / 2.6409× / 1.7242×, and 5.2785× on a second geometry | Logical compression; any quality equivalence between methods |
| Figure 2 | Capacity (x) vs TPOT (y) for the four methods in one session, TTFT annotated | Exp13 `latency_primary_cross_leg` | RABIT is the highest-capacity point and the slowest per token; +8.33% TPOT vs tested TurboQuant for 1.7242× its capacity | A speed advantage; a quantizer-kernel comparison with TurboQuant; statistical confidence (two legs, no CI) |
| Table 2 | Canonical quality: Llama and Qwen PPL with CIs; Llama NIAH, Passage Retrieval, HotpotQA primary and secondary scorer with CIs | `canonical_quality_final_record.json` | Llama retains quality at the frozen point (+1.56% PPL, no retrieval score loss, about −1 F1 with a CI spanning zero); Qwen does not (+1555.9% PPL) | Equivalence on HotpotQA; any Qwen long-context result; cross-model robustness |
| Table 3 (compact) | 8192 tokens: BF16 ceiling 47 vs RABIT sustained 64; RABIT / BF16 throughput ratio range at 2048 and 8192 tokens; 32K TTFT 114.4 s vs 4.4 s | Exp6 summaries; Exp5 `per_context` | Capacity raises admissible concurrency but throughput is lower at every tested concurrency, and long prefill has a cliff | That the limitation is hidden or minor; that the costs are intrinsic |

Five floats. If space fails, Table 3 becomes two sentences plus an inset in Figure 2; it is never dropped
without that text.

---

## 6. Appendix allocation

| App. | Content | Claims / evidence |
|---|---|---|
| A | Full concurrency curves (2048 and 8192 tokens, all C, three trials) and per-point tables; contaminated attempts and the warmup amendment | T1–T4; Exp6 |
| B | Full context-scaling curves and table, including the 32K point; earlier Stage3C diagnostics | X1–X4; Exp5; `diagnostics/stage3c_*` |
| C | Profiling diagnostic: cases, method, launch counts, qualitative attribution; profiler perturbation (3.7×–8.8×) and why percentages are diagnostic only | P1–P6 |
| D | Exact formulas and page byte breakdown; kernel and path details; fallback paths | S1–S3, L6 |
| E | Correctness and parity records: correctness gate, CPU and CUDA oracle conformance, CPU-vs-CUDA description, offline proofs | V1–V7 |
| F | Quality detail: per-window and per-example statistics, both HotpotQA scorers, BF16 control reproduction, invalid attempt and amendments, model identity audit | Q1–Q8 |
| G | Legacy evaluator history, labelled "LEGACY LOGICAL-EVALUATOR RESULTS": Exp1, 2, 7–12, legacy Qwen quality; the two-sentence mismatch explanation expanded | evidence map §9, §10, Q3b |
| H | Additional serving sessions (Exp3, Exp4, replication) and session-to-session variation of absolute latency | L2–L4 |
| I | Reproducibility: commits, hashes, environment, registered protocols | evidence index |

---

## 7. Claims that must NOT be made

Performance:

1. RABIT is faster than BF16, FP8 or TurboQuant in any metric, or improves throughput.
2. RABIT prefill is faster at long context (X3 is NOT SUPPORTED), or any complexity law.
3. Any profiler-derived timing or percentage presented as a measurement; any profiled number replacing
   Exp5 / Exp6 / Exp13 / Exp14 values; the unprofiled diagnostic legs presented as results.
4. "Profiling found no dominant bottleneck" — and equally, that the bottleneck is a single small fix.
5. That the latency and throughput costs are intrinsic to the representation, or that a batched
   implementation would achieve any specific speed.
6. 32,768 tokens as a demonstrated maximum feasible context; C = 64 as a maximum concurrency.
7. A throughput comparison at 8192 tokens / C = 64 as like-for-like (BF16 ran with 47 in flight).
8. Statistical significance of latency differences (descriptive medians, no CIs); mixing absolute
   timings across sessions; the legacy 21.61 ms TPOT.

Capacity and baselines:

9. "5.18–5.27× compression" or any logical byte ratio as the capacity result; literal 2 bits per element.
10. The TurboQuant comparison as a quantizer-kernel comparison, as covering TurboQuant in general rather
    than the tested configuration, or as including quality.
11. Any quality statement about FP8 or TurboQuant (not measured).
12. Measured live memory savings (live paged-KV bytes are derived, not measured).

Quality:

13. Quality validated across models; a known cause for the Qwen degradation; any Qwen long-context result
    under canonical semantics.
14. Equivalence to BF16 on HotpotQA (the CI spans zero; that is not equivalence); "lossless".
15. Any legacy-evaluator number as canonical evidence, or pooled with canonical results — including all
    ablations (Exp7–Exp11), the bit-width frontier, "+1.53%", "+1.16%" and "−5.4 F1".
16. Canonical justification of the individual design choices (K3, V2, G32, R4, META8g64): no canonical
    ablation exists.
17. Results from the invalid PPL Attempt 1, or a CPU-vs-CUDA tolerance.
18. That the Qwen latency result reflects the optimized kernels (it measures fallback paths).

Novelty and scope:

19. Individual novelty of asymmetric K/V quantization, residual windows, grouped affine quantization, or
    low-bit KV compression as such.
20. Generality beyond two models, one GPU type and one engine snapshot.

---

## 8. Biggest remaining writing risk

The value proposition has to be argued without an end-to-end win: capacity is higher, but every measured
latency and throughput number favours the baselines, and performance optimization is closed. The
introduction and §6.3 / §7.1 must make "capacity as the contribution, cost as a measured limitation of the
current implementation" read as a deliberate, honest systems result rather than as a concession discovered
on page 8 — without overstating what the diagnosed bottleneck implies about a future implementation.
Secondary risks: the Qwen result with no diagnosis, and design choices supported only by legacy ablations.
