# RABIT-KV — MLSys Submission Freeze Record

Freeze time: **2026-10-08T14:30:25Z**. Branch `research/mlsys-2027`.

This record freezes the manuscript state. Nothing has been submitted.

It supersedes the freeze of 2026-10-05T15:03:00Z (paper sources `5ef5e23`, `paper/main.tex` SHA-256
`66988882…634a1c8e`, `paper/frozen/main.pdf` SHA-256 `4915bf8b…5a71ef06`, record commit `6ea33e9`) because of a
narrow reviewer-feedback patch to `paper/main.tex`, listed in section 1c. The patch changes wording and
presentation only: no experiment was run, no implementation changed, and no measured value, percentage, table
entry or figure changed. `paper/appendix.tex`, `paper/references.bib` and `paper/frozen/appendix.pdf` are unchanged.

That freeze had superseded the freeze of 2026-10-05T14:59:04Z (paper sources `a1af636`, `paper/main.tex`
SHA-256 `88d693b5…36a8f16d`, `paper/frozen/main.pdf` SHA-256 `349188a7…952c25cd`, record commit `ba74d63`) only
because of one typo correction in Section 5 of `paper/main.tex`: "prefetched" was corrected to "prefilled"
(section 1b). No other prose, number, citation, experiment, or code changed.

That freeze had in turn superseded the freeze of 2026-10-04T22:36:55Z (paper sources `26cb9a7`, `paper/main.tex` SHA-256
`65667025…fb343a2e3`, `paper/frozen/main.pdf` SHA-256 `609458a8…9e6f2d98`, record commit `33cd36b`). The prior
freeze was superseded by four human-requested factual/wording corrections to `paper/main.tex`, listed in
section 1a. No experimental number, table, figure, or implementation changed; `paper/appendix.tex`,
`paper/references.bib` and `paper/frozen/appendix.pdf` are unchanged.

## 1. Commits

| Item | Commit |
|---|---|
| Paper sources (last commit touching `paper/main.tex`, `paper/appendix.tex`, `paper/references.bib`) | `21b931930dc2c8d433b9b5a42e12418bda4f23a3` |
| Documentation (HEAD when the sources were hashed) | `21b931930dc2c8d433b9b5a42e12418bda4f23a3` |
| Frozen experimental evidence | quality `164c17f`; profiling diagnostic `4d07cf6` + `dadc2a0`; story lock `b4219c1` |

The commit that updates this record and the rebuilt main PDF follows `21b9319`; it changes no source file.

### 1a. Corrections that superseded the 2026-10-04 freeze

All four are in `paper/main.tex`, commit `a1af636`, requested by the human author.

1. Introduction: the KV cache is described as "a major GPU-memory consumer in LLM serving" that "can become
   dominant at long context lengths or high concurrency" (was "the main consumer of GPU memory in an LLM
   serving system").
2. Section 3: the per-channel key / per-token value axes are attributed to KIVI, KVQuant and AsymKV; the group
   size of 32 is attributed to KIVI and AsymKV only (the earlier sentence implied it for KVQuant as well).
3. Section 5, quality evaluator: the general statement now describes a dense BF16 prefill of a prompt prefix,
   since the long-context evaluator prefills all but the last prompt token. The Tasks paragraph is unchanged.
   (The replacement text as committed in `a1af636` read "prefetched tokens"; see section 1b.)
4. Section 6 opening: "We do not set out to show that RABIT-KV is fast, and it is not." is replaced by "Our
   evaluation focuses on the capacity--cost trade-off rather than a speedup claim." The reported performance
   limitations are unchanged.

A comparison of the text extracted from the prior and rebuilt main PDFs finds the same numeric tokens in both,
apart from the punctuation after one citation year in the Section 3 sentence. The Qwen2.5-7B TPOT values
(26.96 ms, 48.70 ms, +80.7%) were rechecked against
`results/mlsys2027/second_model/serving/capacity_latency_summary.json` (`median_of_leg_median_tpot_ms`
26.9561 and 48.7007) and left as they were.

### 1b. Typo correction that superseded the immediately prior freeze

Commit `5ef5e23`, requested by the human author: in Section 5 of `paper/main.tex`, "so those prefetched tokens
attend to exact history" became "so those prefilled tokens attend to exact history". Wording only. The text
extracted from the two main PDFs differs in that one word and nothing else.

### 1c. Reviewer-feedback patch that superseded the 2026-10-05T15:03:00Z freeze

Commit `21b9319`, requested by the human author. All edits are in `paper/main.tex`.

1. Rounding statement (Section 5, Latency and throughput): added "Relative differences are computed from
   unrounded measurements; displayed absolute values are rounded." The percentages +8.33% (TPOT vs. TurboQuant),
   +80.7% (Qwen2.5-7B TPOT) and +1.56% (Llama-3.1-8B perplexity) are unchanged.
2. HotpotQA sentence (Section 6.4): now "RABIT-KV scores 58.27 versus 59.30 for BF16, a difference of -1.03 F1;
   the paired 95% confidence interval [-4.24, +2.20] spans zero." The following sentences, including "The
   interval does not establish equivalence at this sample size.", are unchanged.
3. R4 off-by-one (Section 3.2, Open group): "When a fifth newer token arrives, the token leaves the window" was
   wrong by one and is replaced by "When appending a token would make the residual window exceed four tokens,
   the oldest residual token leaves the window and joins the open group". The rest of the paper and the
   appendix were audited for equivalent wording and are consistent with a four-token window (R4 definition,
   Figure 1 and its caption, the n-token decomposition with (n-4), "at most 35 tokens", the evaluator
   description, and the appendix's 16,383 = 511 x 32 + 27 + 4 case); nothing else needed changing.
4. Appendix references: four occurrences of "the appendix" now read "the supplementary appendix" (Section 5
   scope paragraph, Table 2 caption, Section 6.3 context length, Section 7 design space). Every result the main
   paper relies on is stated in the main paper; the supplement holds full tables, details and history only.
5. Ratio presentation: prose capacity ratios are rounded to 5.28x (BF16), 2.64x (FP8) and 1.72x (tested
   TurboQuant) in the abstract, introduction, contribution bullet, Sections 6.1, 6.2 and 6.5, and the
   conclusion. Table 1 keeps the exact token counts and the exact ratios (1.9988x, 3.0614x, 5.2785x); the
   appendix and all evidence and reproducibility records keep the exact ratios.
6. Eager-mode limitation (Section 5, Setup): added "All methods run in eager mode. Because CUDA graph capture
   can interact differently with attention backends, the reported latency comparisons characterize this eager
   configuration and need not transfer unchanged to graph-enabled deployments."
7. Caveat deduplication, three removals: (a) Section 6.3 Concurrency, the closing sentence "The 47-versus-64
   result therefore demonstrates admission capacity, not throughput." (the same paragraph already says
   "Admission is not throughput, however."); (b) Section 6.3 Where the time goes, "We do not claim that these
   costs are intrinsic to the representation." (kept verbatim in Section 7); (c) Section 6.4, the paragraph
   "What is not measured" (the unmeasured later-chunk prefill path is stated in Section 5, Tasks, and in full
   in Section 7). The Introduction's capacity-versus-throughput warning, the TurboQuant qualification in
   Section 6.2 and all of Section 7 are unchanged.

The LaTeX style is unchanged. A comparison of the numeric tokens in the text extracted from the prior and
rebuilt main PDFs finds only the differences these edits imply: the three prose ratios in their rounded form,
"95" in the HotpotQA sentence, and the numbers that occurred in the removed sentences.

## 2. Source hashes

SHA-256 of the committed (LF) content, with the git blob id.

| File | SHA-256 | Git blob |
|---|---|---|
| `paper/main.tex` | `03bd248ebe4e777f9415c1b5c7adcb67ddfceecb024d555d018c684089694e20` | `2db82d9ffb071b01dcaa440e4f037e63f9fb527d` |
| `paper/references.bib` | `87df77c967cdb8b70aab536a4975420f7327e45ab838e384513908bf70cb7063` | `0fb0c88b28912481977e774e35f4d67efdd0f99b` |
| `paper/appendix.tex` | `35ef775e3f209e7fa96e593442bc4001393cb43ae06338379e58c8da876d27e8` | `789d89367cccc282e29f62aa0c6c2feb38593f7a` |
| `paper/mlsys2025.sty` | `05a9842992b7ef71851fd2380a1058f83b0faafc106602cabc4c169d372ad8e2` | `7a942961bb77039dfd3d18794034b3427802be2c` |
| `paper/mlsys2025.bst` | `c9c9f1b83e32512b93f6208e28ba2989fc691b6f70763ad0657a77d44bc067a7` | `d0576754365bf88458d159f5d21c3b32c8195600` |

Figure sources: Figures 1 and 2 are TikZ code embedded in `paper/main.tex` (two `tikzpicture` environments);
there are no separate figure files and no `\includegraphics`. Their source hash is therefore the hash of
`paper/main.tex` above. The appendix has no figures.

## 3. Built PDFs

Built from the committed sources with Tectonic 0.15.0 (XeTeX engine, `xdvipdfmx`), main paper and appendix
compiled separately. Copies are stored in `paper/frozen/`.

| PDF | SHA-256 | Pages |
|---|---|---|
| `paper/frozen/main.pdf` | `87484abe5746689e948955358a00084f4ea4063397cd55bf7c59701c1a22c4d5` | 10 |
| `paper/frozen/appendix.pdf` | `f7c795ef1854e0143ae555f508f94920b1b714d92afe0d0cd535868006d80afb` | 4 |

A PDF hash identifies this build only: a rebuild with another TeX distribution, or at another time, need not
produce byte-identical files. The source hashes in section 2 are the stable identifiers. The final
submission PDF should be rebuilt and re-inspected by the author with the toolchain they will submit from.

Compile status of this build: no errors, no undefined references or citations, no overfull boxes, in either PDF.
The main PDF was rebuilt for this freeze and all 10 pages were inspected as rendered images: no table or figure
overflows its column or the page, Tables 1-3 and Figures 1-2 are placed and numbered as before, and all 15
references resolve. The appendix source is unchanged, so `paper/frozen/appendix.pdf` is the prior build; the
appendix was recompiled as a check (4 pages, extracted text identical to the frozen file) and its 4 pages were
inspected. Section, table and figure numbering in the main paper did not change.

## 4. Page counts

| Quantity | Value |
|---|---|
| Main-paper body before references | about 8.35 pages (the References heading is about two thirds down the left column of page 9) |
| References | about 0.75 page (rest of page 9 and the top of page 10) |
| Main PDF total | 10 pages |
| Appendix (separate PDF) | 4 pages |
| MLSys limit | 10 pages excluding references — satisfied |

The appendix is not appended to the main PDF.

## 5. Format and anonymity

- Style: official MLSys kit `mlsys2025style.zip` from
  `https://media.mlsys.org/Conferences/MLSYS2025/mlsys2025style.zip` (SHA-256 of the downloaded archive
  `04e77090038f78c985154c71f7a57b7fbb2553ddd3e4e62f431ed420ba3768ef`), which the MLSys 2027 Call for Research
  Papers prescribes. `\usepackage{mlsys2025}` without the `accepted` option (blind review).
- Anonymity status: author block prints "Anonymous Authors" / "Anonymous Institution"; PDF metadata author is
  "Anonymous Authors"; an automated text search of both PDFs and of the three source files found no user
  name, e-mail address, repository URL or commit ID. Citations to upstream vLLM and to the model card contain
  public URLs that do not identify the authors. This is an automated check, not a substitute for the author's
  own anonymity review (for example of the system name and of self-citations, of which there are none).
- References: 15 entries; every entry lists all authors, as the call requires.

## 6. State of the work

- Experimental work closed: quality COMPLETE, serving COMPLETE, performance profiling COMPLETE, performance
  optimization NO-GO / CLOSED.
- Claim audit: `docs/MLSYS_PAPER_CLAIM_AUDIT.md`. Related-work verification and TurboQuant provenance audit:
  `docs/MLSYS_RELATED_WORK_SOURCE_MAP.md` (sections 0 and 4). Author packet: `docs/AUTHOR_REVIEW_PACKET.md`.
- Editing rule from this point: substantive prose changes originate from the human author's review. Permitted
  without it: factual correction, citation correction, formatting repair, and wording changes the author
  requests. Any change to a source file invalidates the hashes above and requires a new freeze record.

## 7. Final human-author checklist

To be completed manually by the author before submission.

- [ ] Read every main-paper paragraph
- [ ] Understand every quantitative claim
- [ ] Verify KIVI overlap personally
- [ ] Verify TurboQuant overlap personally
- [ ] Verify Minima-KV overlap personally
- [ ] Verify SAW-INT4 overlap personally
- [ ] Confirm novelty statement
- [ ] Confirm capacity qualification
- [ ] Confirm evaluator/serving distinction
- [ ] Confirm Qwen limitation
- [ ] Confirm throughput limitation
- [ ] Confirm no accidental first/novel/optimal claim
- [ ] Confirm references
- [ ] Confirm anonymity
- [ ] Confirm separate appendix
- [ ] Confirm 10-page main-paper rule
- [ ] Confirm all authorship / submission-policy requirements

Supporting material for each item: `docs/AUTHOR_REVIEW_PACKET.md` (Part A for the paragraphs and claims, Part B
for the four prior works, Part C for the claim defense table).
