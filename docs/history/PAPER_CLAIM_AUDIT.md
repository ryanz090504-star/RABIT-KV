# RABIT-KV — Final Claim-Consistency Audit of the Draft

Date: 2026-10-04. Draft audited: `paper/main.tex` in the official MLSys format (277 sentences, about 6,050 words).
Checked against `docs/MLSYS_PAPER_EVIDENCE_MAP.md`, `docs/MLSYS_PAPER_STORY_LOCK.md` and
`docs/MLSYS_RELATED_WORK_SOURCE_MAP.md`. Method: every sentence containing a number, a comparison, a causal
statement or a novelty-related word was read against the claim it relies on; the whole draft was additionally
searched for the risk words listed below. Classification: SUPPORTED (S), SUPPORTED WITH QUALIFICATION (SWQ),
REMOVE / REWRITE (R). No experiment was run.

## 1. Risk-word search (complete draft)

| Term | Occurrences | Finding |
|---|---|---|
| "first" | 12 | 10 are ordinary uses (first token, first chunk, correctness-first). 2 are the explicit disclaimer in Section 8 ("not the first packed or paged low-bit KV cache, nor the first to account for storage beyond nominal bits"). No first-claim. S |
| "novel" | 0 | — |
| "optimal" / "best" | 2 / 2 | only in disclaimers ("we do not claim that each component is individually optimal"; "not … the best trade-off"). S |
| "generalizes" | 0 | — |
| "speedup", "faster", "outperform", "lossless", "state-of-the-art" | 0 | — |
| "higher throughput" | 2 | both negated ("does not convert … into higher throughput"; "did not yield higher throughput"). S (T2, T4) |
| "2-bit" | 10 | "2-bit target" is defined twice as a label, not a storage cost (S2); the others are "2-bit values" / KIVI's 2-bit quantization. No "2-bit cache". S |
| "compression ratio" | 0 | "compression" appears 3 times, never as our result. S |
| "same semantics" / "same implementation" | 0 | — |
| "equivalent" | 2 | both negated ("does not establish equivalence"). S (Q6) |
| "quality … retained" | 1 (+3 "retain") | Conclusion: "Quality is retained closely on Llama-3.1-8B and degrades severely on Qwen2.5-7B"; Section 7: "retains most of its quality"; Abstract: "retain their tested scores". All carry the model / task qualification. SWQ (Q1, Q4–Q6, Q2) |
| "intrinsic" | 2 | both are the disclaimer. S (P6) |
| "transfer" | 5 | capacity transfers, quality does not; fallback latency "is not evidence about how the specialized kernels would transfer". S (C6, L6, Q2) |
| "TurboQuant" | 24 sentences | see section 3 |

## 2. Quantitative claims

| Claim in the draft | Where | Evidence | Class |
|---|---|---|---|
| 5.2785× / 2.6409× / 1.724× capacity; 393,024 / 785,568 / 1,203,200 / 2,074,592 tokens; block counts | Abstract, §1, §6.1, Table 1, §9 | C1–C5 (Exp13 summary: 5.27854, 2.64088, 1.72423) | S |
| Table 1 pool bytes per token 131,072 / 65,536 / 42,816 / 24,832 | Table 1 | Exp13 `theoretical_bytes_per_token` | SWQ — labelled storage accounting, not the measured quantity; excludes side state |
| Qwen 902,656 → 4,764,640, 5.2785× | §6.1, §6.5 | C6 | S |
| "not a reduction of total device memory" / paged pool only | §3.3, §6.1 | S2 note; source audit of per-sequence state | S |
| 776 B per token-layer; 24.25·Hd; about 3.03 bits per element | §3.3 | S2, S3 (Llama breakdown derived and labelled) | SWQ — derived, closed pages only, "not our headline metric" |
| TPOT 24.19 / 25.75 / 27.79 / 30.10 ms; +24.4% / +16.9% / +8.33%; TTFT 59.99 / 63.43 / 55.03 / 144.65 ms | §6.2, Fig. 2 | L1–L5 | S; the TurboQuant statement is SWQ and carries its qualification in the same paragraph |
| "+24% to +35%" over BF16 in other sessions | §6.2 | L2 | SWQ — relative overheads only; absolute timings are not mixed |
| 47 vs 64 in flight; req/s 2.282 / 1.424, 2.346 / 0.870, 7.218 / 1.844; ratios 0.88, 0.37, 0.70, 0.12; 1.424 → 0.870 → 0.323 | §1, §6.3, Table 2 | T1–T4 | S; "admission capacity, not throughput" |
| TPOT overhead +31% to +39% (512–16,384 tokens); 15.43 / 21.15 ms; TTFT 114.4 s vs 4.4 s; 16,352-token later chunk | §6.3, Table 2 | X1, X4 | S |
| lower TTFT at 8K / 16K not interpreted as a prefill advantage | §6.3 | X3 (NOT SUPPORTED as an advantage) | S — stated as a non-claim |
| profiling: bottleneck families, profiler perturbation, no percentages | §6.3, §7 | P3–P6 | SWQ — qualitative only |
| PPL 7.515 → 7.633, +1.56% [+1.12%, +2.05%], 30 of 32 windows | Abstract, §1, §6.4, Table 3 | Q1 | S |
| NIAH 57/57; Passage Retrieval 100.0 / 100.0; saturation caveat | §6.4, §7 | Q4, Q5 | S |
| HotpotQA 59.30 / 58.27, −1.03 [−4.24, +2.20]; 92 identical, 5 worse, 3 better; secondary 59.85 / 58.71, −1.14 [−4.30, +2.03] | §6.4, Table 3 | Q6, Q7 | SWQ — "does not establish equivalence" |
| Qwen PPL 6.787 → 112.39, +1555.9% [+1320%, +1828%], 32 of 32 windows; undiagnosed | §1, §6.5, Table 3 | Q2, Q3 | S |
| Qwen TPOT 48.70 vs 26.96 ms (+80.7%), fallback paths | §6.5 | L6 | SWQ — "characterizes the fallback paths" |
| evaluator schedule; first scored token from BF16 prefill; 127 of 128; later-chunk path unmeasured; no end-to-end comparison | §5, §6.4, §7 | E1–E4 | S |
| same-device bit-exact conformance; not formal verification | §5 | V2–V7 | S |
| correctness tests before every serving experiment | §4 | V1 | S |

## 3. TurboQuant statements

Every sentence that attributes a number to TurboQuant says "tested TurboQuant configuration" or sits in a paragraph
that defines it (28 quantized layers, four BF16 boundary layers, its own attention backend, single request, no
quality measured). One sentence in §6.2 ("… and +8.33% relative to TurboQuant") uses the bare name; it is inside
the matched-session paragraph and is followed immediately by the qualifying paragraph. SWQ, kept.

## 4. Violations found and fixed in this pass

| # | Sentence | Problem | Fix |
|---|---|---|---|
| 1 | §8 "KIVI established that keys should be quantized per channel …" | normative wording | "KIVI quantizes keys per channel and values per token, and combines …" |
| 2 | §8 "… outside the block pool for the same reason" | unsupported causal claim about our design motive | "SAW-INT4 notes that … complicates storage in uniform paged blocks; RABIT-KV stores its residual and open-group state outside the paged pool." |
| 3 | §8 "Minima-KV is closest in structure" | implied a systematic ranking of the literature | "has a closely related tiered serving design" |
| 4 | §8 "one of its presets is our baseline" | implied our snapshot's backend is the upstream PR | "Upstream vLLM integrates a TurboQuant backend … Our engine snapshot contains a TurboQuant backend with the same preset interface; all TurboQuant measurements in this paper refer to one preset of the backend in that snapshot" |
| 5 | §5 baseline description cited the upstream PR as if it were our backend | same provenance issue | "one configuration of the TurboQuant backend in our snapshot (…; cf. the upstream integration)" |
| 6 | §2 and §4 cited the upstream PR directly for our tested configuration / our engine's cache types | same provenance issue | citation removed in both places; §4 now says "the FP8 and TurboQuant cache types of our engine snapshot" |
| 7 | §8 "RABIT-KV contributes no new quantizer to this line" | bare, and silent on what is contributed | "RABIT-KV does not claim a new quantization primitive" + closing paragraph states the contribution as physical realization and joint measurement |
| 8 | §5 "[NIAH]" placeholder | no canonical citation exists for our task | described as "a synthetic needle-in-a-haystack task of our own construction, in which a secret code is inserted at 19 depths into WikiText-2 filler text"; no citation invented |
| 9 | Abstract contained two citations | style | removed (the same citations remain in §1) |

No quantitative claim required a change. No correct claim was weakened.

## 5. Contribution check

1. Design bullet: "A target-bit-aware KV-cache representation with an online cache lifecycle … integrates them into
   one lifecycle whose target bit budget maps to a concrete physical layout." Section 3 now states explicitly that
   the axes and group size are KIVI's scheme and where the other ingredients appear. S.
2. Physical-system bullet: benefit "measured as allocator-observed KV capacity rather than by logical compression
   accounting", with the three ratios. No first-claim. S.
3. Evaluation bullet: matched four-method evaluation plus quality on two models with a validated evaluator,
   "model-dependent quality trade-offs" (the negative Qwen result). S.

None of the five forbidden first-claims appears.

## 6. Items that remain open (not violations)

- Whether the TurboQuant backend in our engine snapshot is identical to upstream PR #38479 is not verified; the
  draft is worded so that it does not depend on it.
- The quoted prior-work passages were read by the drafting assistant; an author should confirm them against the PDFs.
- Appendix material referenced by the main text (full curves, profiling details, earlier ablations, correctness
  records) is not drafted yet.
