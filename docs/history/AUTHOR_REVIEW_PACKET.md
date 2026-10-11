# RABIT-KV — Author Review Packet

Purpose: everything the human author needs to personally understand, verify and defend the submission.
It does not rewrite the paper. Sources: `paper/main.tex` (main paper), `paper/appendix.tex` (separate
supplementary PDF), `docs/MLSYS_PAPER_EVIDENCE_MAP.md` (claim IDs such as C5, L5, Q1),
`docs/MLSYS_PAPER_STORY_LOCK.md`, `docs/MLSYS_RELATED_WORK_SOURCE_MAP.md`, `docs/MLSYS_PAPER_CLAIM_AUDIT.md`.

Status of the manuscript: structure and experiments frozen. The next prose edits should come from your
review. Things only you can do are marked **AUTHOR ACTION**.

---

## Part A. Section-by-section review

### Abstract

**Purpose.** State the problem (bit-width is not capacity), what RABIT-KV is, the capacity result, its cost,
the quality result on two models, and the throughput limitation, in that order.

**Key claims and evidence.**

| Claim | Evidence |
|---|---|
| 5.2785× / 2.6409× / 1.724× the KV capacity of BF16 / FP8 / tested TurboQuant | C5: allocator tokens 2,074,592 vs 393,024 / 785,568 / 1,203,200 in one session (`external_baseline/exp13/matched_capacity_latency_summary.json`) |
| +8.33% median single-request TPOT vs tested TurboQuant | L5: 30.10 vs 27.79 ms, same file |
| Llama PPL +1.56%; NIAH and Passage Retrieval unchanged; HotpotQA about −1 F1, CI spans zero | Q1, Q4–Q6 (`canonical_quality_v2/canonical_quality_final_record.json`) |
| Qwen: severe model-specific sensitivity | Q2: 6.787 → 112.39 |
| Capacity does not become higher throughput | T2, T4 (Exp6) |

**You must be able to explain.** What "allocator capacity" means (blocks × 32 tokens reported by the running
engine); why TPOT was compared only inside one session; why the Qwen number is in the paper at all.

**Likely questions and answers.**

- *Why should I care about capacity if it is slower?* Capacity determines what can be admitted (BF16 stops
  at 47 in-flight 8192-token requests, RABIT-KV holds 64). The paper measures that this did not turn into
  throughput with the current implementation, and says so. The contribution is the measured trade-off.
- *Is 1.724× vs TurboQuant a fair comparison?* It compares two systems as deployed in the same engine
  snapshot, one TurboQuant configuration, single request. It is not a quantizer comparison and no TurboQuant
  quality was measured.

**Do not strengthen.** Never drop "tested" before TurboQuant. Never say "retains quality" without "on
Llama-3.1-8B". Never say "5.28× more requests".

### 1 Introduction

**Purpose.** Motivate KV memory as the capacity bottleneck, separate logical precision from physical capacity,
introduce RABIT-KV with its headline numbers, and put the throughput limitation on page 1–2.

**Key claims and evidence.**

| Claim | Evidence |
|---|---|
| KV memory bounds context × concurrency under a paged allocator | background fact; vLLM paper |
| Metadata at 2–3 bits is a substantial fraction of payload | arithmetic: 2 × 16 bits per 32 elements = 1 bit per element |
| BF16 ceiling 47 vs RABIT-KV 64 at 8192 tokens | T3 (3/3 trials) |
| Throughput below BF16 in every tested concurrency workload | T2, T4 |
| Large TTFT penalty beyond the first 16,384-token chunk | X4: 114.4 s vs 4.4 s |
| Three contributions (design, physical system, evaluation) | blueprint §2 |

**You must be able to explain.** The operating point K3 / V2 / G32 / R4 / META8g64 in one breath; why "2-bit
target" is a label; what "target-bit-aware" means (start from a bit budget and account for what it costs
physically) and that it is an adjective, not a named technique.

**Likely questions and answers.**

- *What is new here relative to KIVI?* Not the quantizer. The same axes and group size are used. New is the
  specific operating point (3 / 2 bits, 4-token residual, per-token open-group re-quantization, 8-bit
  second-level metadata), its packed realization in a paged engine, and the matched measurement of capacity,
  cost and quality.
- *Why no speed result?* Because there is none to report; the implementation is not batched across requests
  or tokens. The paper says this in the abstract, the introduction, §6.3, §7 and §9.

**Do not strengthen.** Do not write "first", "novel", or "what is missing is". Do not imply the ingredients
are yours.

### 2 Background and Motivation

**Purpose.** Give the three facts the design needs: KV memory per token, paged allocation and what "capacity"
means, and why bit-width alone does not determine capacity.

**Key claims and evidence.**

| Claim | Evidence |
|---|---|
| Llama-3.1-8B: 4,096 B per token per layer, 128 KiB per token in BF16 | 2 × 8 heads × 128 × 2 B; × 32 layers = 131,072 B |
| Pool is sized once from bytes per block | engine behaviour; consistent with implied bytes per token in Exp13 |
| Some prior work reports metadata-inclusive bits (SKVQ) or peak memory (KIVI) | source map §0 |
| Tested TurboQuant configuration leaves four boundary layers in BF16 | C4 |

**You must be able to explain.** Why a full-precision buffer or metadata can sit inside the pool, beside it,
or nowhere, and why that changes the block count.

**Likely question.** *Others already count metadata.* Yes, the text says so and cites SKVQ. Our point is
measurement in the allocator, not being the first to count.

**Do not strengthen.** Keep "often", not "usually" or "always".

### 3 RABIT-KV Design

**Purpose.** Define the operating point exactly, the residual → open group → closed page lifecycle, and the
four-way accounting (label / payload bits / page bytes / allocator capacity).

**Key claims and evidence.**

| Claim | Evidence |
|---|---|
| Axes and group size follow KIVI; other ingredients attributed to KIVI, SKVQ, AsymKV | source map §0 |
| K3: 3-bit, per (head, channel) over 32 tokens, scale = (max − min) / 7; V2: 2-bit, per (token, head) over 32 channels, scale = (max − min) / 3 | S1; `canonical_rabit_quality.py`; reference `kvquant_k3.py` |
| META8g64: 64-value groups, 8-bit codes, BF16 group min / scale; dequantize with decoded parameters | S1 |
| n tokens → ⌊(n − 4) / 32⌋ closed pages, (n − 4) mod 32 open, 4 residual | definition; check: 16,383 → 511 / 27 / 4 (V5) |
| Page = 24.25·H·d bytes; Llama 24,832 B = 776 B per token-layer; about 3.03 bits per element | S2, S3; derivation in appendix C. **Derived**, closed pages only |
| Capacity measures the paged pool; residual and open group (≤ 35 tokens per sequence) are outside it | source: per-sequence runtime is a bounded sidecar |

**You must be able to explain.** Why keys cannot be finalized until the group has 32 tokens while values
can; why the open group is re-quantized every token; the page-byte derivation on a whiteboard:
12Hd + 8Hd + 4(Hd + Hd/16).

**Likely questions and answers.**

- *Why 3 bits for keys, 2 for values, 4 residual tokens?* The point was fixed during development and not
  retuned; the component ablations exist only under the earlier evaluator, so the paper does not claim the
  choices are optimal (§7, appendix G).
- *So it is really a 3-bit cache?* The payload averages 2.5 bits; with metadata a closed page costs about
  3.03 bits per element. "2-bit target" is the name of the operating point. The headline is allocator
  capacity, not bits.
- *Does the 5.28× include the side state?* No. It is the paged pool. Side state is at most 35 tokens per
  active sequence and is outside it; the paper says so in §3.3, §6.1 and the Table 1 caption.

**Do not strengthen.** Never abbreviate to "RABIT uses 3.03 bits". Never say attention "always" reads
quantized state (the serving first chunk is exact).

### 4 Physical Implementation

**Purpose.** Make the capacity result credible: how pages enter the allocator, the three execution paths,
what is not batched, the Llama-specialized vs Qwen fallback paths, and the correctness tests.

**Key claims and evidence.**

| Claim | Evidence |
|---|---|
| A cache type in vLLM's Triton backend; allocator, block tables, scheduler unchanged | source `triton_attn.py`, `kv_cache_interface.py` |
| One block = one closed page per layer; written when the group closes | source (runtime flush path) |
| Initial prefill: dense attention over exact K / V, then bulk encode | source `_forward_rabit_kv2` |
| Later chunks attend to packed prefix + exact tail; closed-page decode shared per 32 queries | source (shared-decode path, query block 32) |
| Per request, per token serialization; rest of model batched | source (Python loop over requests); P4 |
| Qwen served by exact fallback paths | L6 |
| Test suite before every serving experiment; not formal verification | V1 (105 tests) |

**You must be able to explain.** The per-step token budget of 16,384 and why sharing it among requests
triggers the later-chunk path; why per-request serialization hurts throughput even though the GPU kernels
themselves are small.

**Likely questions and answers.**

- *Why not batch?* The implementation was built for correctness and packing first. Batching the tail and
  append paths across requests and tokens needs new kernels and a restructured per-sequence runtime; it is
  future work, and no claim is made about what it would achieve.
- *Is the Qwen latency representative?* No. It measures fallback paths; the paper says it is not evidence of
  how the specialized kernels would transfer.

**Do not strengthen.** Do not say the overhead is "only launch overhead" or give any share of time.

### 5 Experimental Methodology

**Purpose.** State exactly what was measured and how: setup, baselines, capacity, matched latency, the
quality evaluator's two-phase schedule, reference validation, registered protocol, and that earlier
results are excluded.

**Key claims and evidence.**

| Claim | Evidence |
|---|---|
| One H100 80GB; eager; 32-token blocks; memory fraction 0.82; max length 32,768 | serving manifests |
| Matched session, mirrored order A B C D D C B A, 2 legs × 30 samples, 2048-token prompt, 32 tokens | Exp13 |
| Context scaling: 512–32,736 tokens, 15 samples per cell; concurrency: C = 1–64, 256 requests, 3 trials | Exp5, Exp6 |
| Prompt prefilled densely in BF16, then converted in one step; later tokens one per forward | E1, E2 (source audit, evidence map §8a) |
| PPL: first scored token from the BF16 prefill, identical in both arms; 127 of 128 on RABIT state | E2 |
| Long context: last prompt token is the first step on converted state | E2 |
| Later-chunk prefill quality not measured | E3 |
| Evaluator bit-exact vs independent reference, same device; no end-to-end serving comparison | V2–V7, E4 |

**You must be able to explain.** The difference between "reproduces the stored-cache representation and its
aging" and "reproduces the serving execution"; why CPU and GPU states differ in the last bit and why that
led to same-device conformance.

**Likely questions and answers.**

- *Why not score the serving engine's own outputs?* We did not. Quality is a logical measurement of the
  representation; serving and evaluator are each tied to the same reference, but their outputs were not
  compared end to end. This is stated in §5 and §7.
- *Does the evaluator cover chunked prefill?* No. Prompts are at most 16,384 tokens and prefilled in one
  dense pass. Quality when prompt tokens attend to a packed prefix is unmeasured.

**Do not strengthen.** Not "every token attends to quantized state". Not "validated quality" — the validation
is of the state computation, not of model quality.

### 6 Evaluation

**Purpose.** Answer RQ1–RQ4: capacity, its serving cost, scaling behaviour, and quality with cross-model
transfer.

**Key claims and evidence.**

| Claim | Evidence |
|---|---|
| Table 1 capacities and ratios; Qwen 902,656 → 4,764,640 (5.2785×) | C1–C6 |
| TPOT 24.19 / 25.75 / 27.79 / 30.10 ms; TTFT 59.99 / 63.43 / 55.03 / 144.65 ms | L1 |
| +24.4% / +16.9% / +8.33%; +24% to +35% over BF16 across sessions | L2–L5 |
| Table 2: 47 vs 64; 2.282 / 1.424; 2.346 / 0.870; 7.218 / 1.844; 15.43 / 21.15 ms; 4.4 / 114.4 s | T1–T4, X1, X4 |
| Ratios 0.88 → 0.37 (8192), 0.70 → 0.12 (2048); RABIT throughput falls beyond C = 16 | T2, T4 |
| Profiling is qualitative only | P3–P5 |
| Table 3 quality numbers; 30 / 32 and 32 / 32 windows; 92 / 5 / 3 | Q1–Q7 |
| Qwen TPOT 48.70 vs 26.96 ms (+80.7%), fallback | L6 |

**You must be able to explain.** Why Table 1's "pool bytes/token" is accounting and the token column is the
measurement; why 47-vs-64 is admission and not throughput; why lower TTFT at 8K / 16K is not claimed; why
the HotpotQA interval does not show equivalence; why NIAH and Passage Retrieval cannot resolve small effects.

**Likely questions and answers.**

- *RABIT throughput drops as C rises at 8192 tokens — why?* More of the work goes through the serialized
  per-token paths, including later prefill chunks once the step budget is shared. The diagnostic supports
  this qualitatively; we give no percentages because the profiler slowed execution 3.7–8.8×.
- *Is +1.56% PPL significant?* The paired interval is +1.12% to +2.05% and 30 of 32 windows are worse, so
  the loss is small but systematic.
- *What about quality at 32K?* Not measured. The long-context suite stops at 16,384 tokens and does not
  execute the later-chunk path.

**Do not strengthen.** No ratio for the C = 64 / 8192 point. No "prefill is faster at 8–16K". No "no
degradation" on HotpotQA.

### 7 Discussion and Limitations

**Purpose.** State four limits plainly: capacity is not throughput; quality is model-dependent; the scope of
the quality evaluation; the scope of the design space.

**Key claims.** Costs are not claimed to be intrinsic; batching is future work. Operating point not retuned
per model; Qwen mechanism undiagnosed. Serving outputs not scored; later-chunk path unmeasured; long-context
Llama only; saturated suites; N = 100 and N = 32; no FP8 / TurboQuant quality. Ablations exist only under
the earlier evaluator. One GPU type, one engine snapshot, descriptive latency statistics.

**You must be able to explain.** Each limitation without sounding apologetic: these are the boundaries of the
measurement, not excuses.

**Likely question.** *Why should we believe batching would help?* We do not claim it would to any specific
degree. We report where the implementation serializes work and that the 32K path also has substantial
device-side closed-page work.

**Do not strengthen.** Do not promise speedups. Do not speculate on the Qwen cause (GQA, outliers, bias,
fallback kernels are all untested).

### 8 Related Work

**Purpose.** Place RABIT-KV honestly: the representation descends from KIVI; packed low-bit caches in paged
engines already exist; the contribution is one aggressive operating point realized and measured end to end.

**Key claims.** See Part B. Explicit statement: "not the first packed or paged low-bit KV cache, nor the
first to account for storage beyond nominal bits".

**You must be able to explain.** The exact differences from KIVI, Minima-KV and SAW-INT4 (Part B), and that
the TurboQuant numbers are for one preset of the backend in our snapshot.

**AUTHOR ACTION.** Read the four works yourself (Part B) before submission; the passages were read by the
drafting assistant only.

**Do not strengthen.** No "closest", no "unlike prior work, we …" claims that criticize papers for metrics
they did not target.

### 9 Conclusion

**Purpose.** Restate problem, system, capacity, cost, Llama-vs-Qwen quality, and the joint-evaluation lesson.
No new number.

**Do not strengthen.** "Quality is retained closely on Llama-3.1-8B" must keep the model name.

---

## Part B. Prior-work verification packet (four works)

For each work: where to look, what overlaps, what differs, and how to answer. Pointers are section names
from the arXiv HTML versions read on 2026-10-04; page numbers are not given because they differ between
versions. **AUTHOR ACTION: verify each pointer against the PDF.**

### B.1 KIVI

Citation: Liu, Yuan, Jin, Zhong, Xu, Braverman, Chen, Hu. "KIVI: A Tuning-Free Asymmetric 2bit Quantization
for KV Cache." ICML 2024. arXiv:2402.02750.

Inspect:

1. Abstract — the finding that keys are quantized per channel and values per token.
2. Section 2 (preliminary study, Table 1) — simulated group-wise quantization with group size 32; note that
   zero-padding of incomplete groups is described here, for the simulation only.
3. Section 3 (method, Algorithm 1) — the full-precision residual: recent keys and values kept exact and
   quantized in a batch once the residual reaches R tokens.
4. Section 4.1 (settings) — group size 32 and residual length 128 in all experiments; implemented on the
   Hugging Face Transformers codebase.
5. Section 4 efficiency paragraph and Figure 5 — peak memory and throughput as batch size grows on one A100.

Overlap: per-channel keys along the token axis, per-token values, group size 32, a full-precision window of
recent tokens. RABIT-KV's representation is a descendant of this design.

Differences: 3 / 2 bits instead of 2 / 2; a 4-token window instead of 128; an open group re-quantized at
every token instead of batch quantization of the residual; second-level 8-bit quantization of the group
parameters; packed pages inside a paged serving allocator and allocator-measured capacity instead of peak
memory in Transformers.

Why a contribution remains: the paper does not claim the quantizer. It contributes the packed realization
and matched measurement of a more aggressive point, and a negative cross-model result.

Reviewer question: *Is this KIVI with different hyperparameters inside vLLM?*

Careful answer: At the level of quantization axes and group size, yes, and Section 3 says so. The differences
that matter for the paper are the lifecycle (tokens are quantized after four steps and the open group is
re-encoded every token, which is what makes a 32-token page the unit of storage), the quantized metadata,
and the fact that the result is an allocator-visible capacity measured against the engine's own FP8 and
TurboQuant caches, together with its measured cost.

### B.2 TurboQuant and the vLLM TurboQuant backend

Citations: Zandieh, Daliri, Hadian, Mirrokni. "TurboQuant: Online Vector Quantization with Near-optimal
Distortion Rate." arXiv:2504.19874 (2025). vLLM pull request #38479, "[Attention Backend] TurboQuant: 2-bit
KV cache compression with 4x capacity", merged 15 April 2026.

Inspect:

1. Paper abstract — random rotation plus per-coordinate scalar quantization; data-oblivious / online.
2. Paper KV-cache experiments — Llama-3.1-8B-Instruct, needle-in-a-haystack and LongBench; 2.5 and 3.5 bits
   per channel.
3. PR "Summary" — keys: Walsh–Hadamard rotation + Lloyd–Max scalar quantization; values: uniform; fused
   Triton kernels at store time.
4. PR "Compression Presets" table — the four presets, including the 3-bit-key / 4-bit-value preset and its
   slot bytes.
5. PR bullets on boundary-layer protection (layers kept in FP16) and on the attention spec that overrides the
   page size with the packed slot size; PR performance tables.

Overlap: a packed, low-bit, asymmetric-precision KV cache inside vLLM's paged allocator, with full-precision
exemptions, already merged upstream. This is prior art for "packed low-bit KV in a paged engine".

Differences: quantizer family (rotation + scalar quantization per token vs grouped affine with a lifecycle);
exemption by layer vs by recency; RABIT-KV's tested capacity is higher (1.724×) and its TPOT higher (+8.33%)
than the one tested preset.

Why a contribution remains: a different, more aggressive design point measured in the same engine and session,
with quality validated by an independent reference; our results say nothing about other presets.

Reviewer question: *The engine already ships a low-bit backend that is faster per token. Why RABIT-KV?*

Careful answer: The paper does not argue that RABIT-KV should replace it. It reports that this operating point
yields 1.724× the allocator capacity of the tested preset at +8.33% single-request TPOT, with worse TTFT and
no TurboQuant quality measured. Which trade-off is preferable depends on the workload.

**Provenance (audited, read-only).** Every TurboQuant-specific file in our snapshot is byte-identical (git blob
hash) to upstream vLLM at the snapshot's base commit `f329ce4` (2026-07-04). That is a later upstream revision
of the backend introduced by PR #38479, not the PR as merged (upstream changed several files in between). We did
not modify the backend. Details: `docs/MLSYS_RELATED_WORK_SOURCE_MAP.md` section 4. The paper now says the
snapshot "is based on a later upstream revision and contains that revision's TurboQuant backend unmodified".

### B.3 Minima-KV

Citation: Kozyrev, Maiboroda. "Minima-KV: Retention-Preserving KV Cache Compression with Mixed-Format Paged
Attention." arXiv:2608.23834 (August 2026).

Inspect:

1. Abstract and Section 1 (the four contributions).
2. Section 3.1 (paged state and lifecycle; Recent / Anchor in FP8, Stale in a 3-bit rotated scalar format;
   promotion and demotion).
3. Section 3.2 (effective rate; the 18.3 KiB per live token figure and the statement that it is an
   owner-reported aggregate, with capacities labelled analytical).
4. Section 3.3 (mixed-format paged attention: per-format partial maximum, exponential sum and output, merged
   by a global softmax).
5. Section 8 (limitations: one model and GPU; unreconciled aggregate) and Section 6.3 (single-pair
   throughput ratio).

Overlap: a lifecycle from higher-precision recent pages to packed low-bit older pages under paged attention;
partial attention per format merged by an online softmax; memory reported per token against BF16 and FP8.

Differences: three tiers with FP8 for recent and protected pages and a controller that can promote pages,
versus one packed format behind a four-token exact window; rotated scalar quantizer versus grouped affine
with asymmetric bits and quantized metadata; reported memory is an aggregate with analytical capacities,
versus block counts read from the running allocator in a matched four-method session; one model versus two.

Why a contribution remains: a different and more aggressive point (no FP8 tier), allocator-measured capacity,
a matched comparison with the engine's own baselines, and an independently validated quality evaluator with a
negative cross-model result.

Reviewer question: *Minima-KV already has an aging hierarchy with mixed-format paged attention. What is
different?*

Careful answer: The structure is closely related and the paper says so. RABIT-KV keeps only four tokens
exact and everything else in one asymmetric low-bit format, so it sits at a lower-bit point; and its capacity
is the allocator's own block count, not an accounting aggregate. We do not claim the lifecycle idea.

### B.4 SAW-INT4

Citation: Jia, Li, Zhou, Heo, Wang, Dao, Song, Athiwaratkun, Xu, Zhang, Wu. "SAW-INT4: System-AWare 4-Bit
KV-Cache Quantization for Real-World LLM Serving." arXiv:2604.19157 (April 2026).

Inspect:

1. Abstract — serving constraints (paged layouts, regular memory access, fused attention); token-wise INT4
   with block-diagonal Hadamard rotation; "systems co-design problem".
2. Section 1 — the paragraph on residual buffers (KIVI, Kitty) conflicting with uniform paged blocks, and the
   contributions list.
3. Section 2 — the serving-system constraints.
4. Section 4.1–4.2 — the fused kernel and end-to-end latency.
5. Appendix D and the throughput table — serving throughput under concurrent load against BF16.

Overlap: low-bit KV quantization designed for and evaluated inside a paged serving engine, with
dequantization fused into the attention kernel and end-to-end serving measurements.

Differences: 4-bit token-wise quantization for both keys and values with no full-precision buffer, reporting
throughput gains; RABIT-KV uses a lower asymmetric budget with per-channel keys and an exact window held
outside the pool, and reports lower throughput.

Why a contribution remains: a different region of the design space (about 3 bits per element including
metadata instead of 4 bits plus scales), with its capacity and its costs measured.

Reviewer question: *SAW-INT4 shows low-bit KV can improve throughput in a real engine. Why does yours not?*

Careful answer: SAW-INT4 chooses a point whose kernels fit the engine's batched execution. RABIT-KV's
per-channel key groups and open-group re-quantization are implemented per request and per token in the
current code, which is where its throughput is lost. The paper states this as an implementation limitation
and does not claim the cost is intrinsic.

---

## Part C. Claim defense table

| Claim | Evidence | Qualification | Likely objection | Defensible response |
|---|---|---|---|---|
| 5.2785× capacity vs BF16 | 2,074,592 vs 393,024 allocator tokens (64,831 vs 12,282 blocks), one session; same ratio on Qwen | paged pool only; one H100, one engine snapshot | "Just a byte ratio." | It is the engine's own block count; the byte ratio 4096 / 776 predicts it and the measurement confirms it |
| 2.6409× vs FP8 | 785,568 tokens for the engine's native FP8 cache, same session | no FP8 quality measured | "FP8 is nearly lossless; yours is not." | Correct; the comparison is of capacity and latency only, and the paper says so |
| 1.724× vs tested TurboQuant | 1,203,200 tokens, same session | one preset (3-bit keys, 4-bit values, 4 BF16 boundary layers) of the backend in our snapshot | "Cherry-picked configuration." | We state it is one configuration and that results do not characterize the others |
| +8.33% TPOT vs tested TurboQuant | 30.10 vs 27.79 ms median, 2 legs × 30 samples | systems as deployed (different attention backend); single request; no CI; TTFT 144.65 vs 55.03 ms shown beside it | "Not statistically tested." | Descriptive medians; stated as such |
| Lower throughput than BF16 | Exp6: every tested C at 2048 and 8192 tokens (e.g. 0.870 vs 2.346 req/s at C = 32) | three trials, medians | "Then what is the point?" | Capacity raises admissible load (47 vs 64); the paper measures that it does not raise throughput with this implementation and reports it as the central limitation |
| 32K TTFT cliff | 114.4 s vs 4.4 s (Exp5) | 32,736 tokens is the nominal model limit; later-chunk path | "Unusable for long context." | For prompts beyond one 16,384-token chunk with the current implementation, yes; stated in §1, §6.3, §7, §9 |
| Llama +1.56% PPL | 7.515 → 7.633, paired CI [+1.12%, +2.05%], 30 / 32 windows worse | 32 windows; first scored token per window is BF16 in both arms | "Small N." | Paired design, systematic direction; N stated as a limitation |
| Qwen +1555.9% PPL | 6.787 → 112.39, CI [+1320%, +1828%], 32 / 32 windows | same fixed operating point, not retuned; cause undiagnosed | "Your method is broken on Qwen." | At this operating point, yes; reported as a negative result showing that quality does not transfer with the representation |
| NIAH / Passage Retrieval unchanged | 57 / 57 vs 57 / 57; 100.0 vs 100.0; no score changes | BF16 saturated; cannot resolve small effects; NIAH cases not independent | "Too easy." | Agreed and stated; they show no loss on retrieval, not equivalence |
| HotpotQA −1.03 F1 | 59.30 vs 58.27; CI [−4.24, +2.20]; 92 identical, 5 worse, 3 better; secondary scorer −1.14 | N = 100; CI spanning zero is not equivalence | "So no significant loss?" | We do not say that; the interval is wide and the paper says it does not establish equivalence |
| "2-bit target" ≠ 2 bits | payload 2.5 bits; closed page ≈ 3.03 bits per element (derived) | label for the operating point | "Misleading name." | Defined as a label in §1 and §3.3; the headline metric is capacity |
| Capacity ≠ total HBM compression | allocator pool measurement; side state ≤ 35 tokens per sequence outside the pool; live memory not measured | — | "Does 5.28× include everything?" | No; §3.3, §6.1, Table 1 caption and appendix C say exactly what is and is not counted |
| Evaluator ≠ serving execution | source audit (evidence map §8a): dense BF16 prefill, one-step conversion, then token-by-token aging | later-chunk prefill quality unmeasured; no end-to-end comparison | "Your quality numbers are not from the system you benchmark." | Correct; they are logical quality of the representation, validated bit-exactly against an independent reference; stated in §5, §6.4, §7 |
| No new quantization primitive | §3 attributes axes, group size, window, asymmetric bits and reduced-precision parameters to prior work | — | "Then what is the research contribution?" | The packed realization and joint measurement of one aggressive operating point: allocator capacity, serving cost, validated quality, cross-model result |
| Novelty vs KIVI / Minima-KV / SAW-INT4 | Part B | no "first" claim anywhere | "Incremental." | Different design point and a measurement none of them reports: allocator-observed capacity in a matched session with the engine's own baselines, with the costs disclosed |

---

## Part D. Things only the author can do before submission

1. Read the four prior works at the pointers in Part B and confirm the paper's sentences about them.
2. (Done by audit.) The TurboQuant backend in our snapshot is the unmodified upstream backend at the snapshot's
   base commit; see Part B.2.
3. Read the main paper once end to end against Part A and mark any sentence you cannot defend orally.
4. Decide whether the appendix promise "Exact revisions, file hashes, and seeds are recorded with the
   experiment artifacts" matches what you will release.
5. Check anonymity: the system name, the needle text and the engine description do not identify you; no
   repository URL or commit ID appears in either PDF.
6. Confirm the MLSys 2027 dates and rules on the official call (10-page limit excluding references; all
   authors listed in every reference; appendix uploaded separately).
