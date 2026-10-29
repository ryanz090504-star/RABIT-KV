# N32 attempt-002: completed and independently audited

Final status: complete, valid, frozen analysis finished. No new attempt was launched during recovery. App `ap-NYwbdORspbtIXZlHsNLdqJ`, call `fc-01M5FAN0WNZ4C4EBHPF735D1V2`, volume `vo-iMNdVWWdzMq8Pl7PctpL62`. One successful call, one container, zero failures/timeouts. Modal stopped at 2026-10-21 10:38:41 America/New_York. Explicit stop confirmed already stopped; final usage shows zero runners, active inputs, and backlog. Persistent evidence retained.

## Scope and exposure

All32 fresh windows, 128 target tokens/window/cell (4096/cell), independent HF control, all four serving cells in frozen order: BF16 single, packed single, packed later, BF16 later. No exclusions or pooled partial results. Windows5-32 are the preregistered originally-uninspected physical set; by attempt002, cancelled attempt001 had exposed BF16 single windows1-32 and packed single windows1-23 plus88 target tokens of24. That partial exposure is disclosed, not relabeled as fresh uninspected data. Historical logical exposure also exists. Windows1-4 remain separate; all32 is secondary.

Model/tokenizer: `LLM-Research/Meta-Llama-3.1-8B-Instruct@359efdbb8af05b788a4ad4185215c6b8caa9052c`. Input SHA256 `b1f90a924f0c8e12f4b4633e964a7d5f660bae9bd6c3cccc17ede6f6236d8670`. The missing historical Llama revision remains unknown. Frozen metadata’s old ephemeral-output wording is preserved; actual unique persistent result-volume creation/indefinite retention is correctly recorded in attempt002 registration, following the additive correction note.

## Independently recomputed quality

| Set | BF16 single | Packed single | Packed later | BF16 later | HF control |
|---|---:|---:|---:|---:|---:|
| Primary windows5-32 | 7.493334 | 7.599709 | 7.639868 | 7.493334 | 7.480139 |
| Inspected windows1-4 | 7.718317 | 7.842640 | 7.939774 | 7.718317 | 7.707687 |
| Secondary all32 | 7.521095 | 7.629659 | 7.676728 | 7.521095 | 7.508211 |

Higher PPL is worse. Paired window bootstrap uses100,000 resamples, frozen seed20261020; primary co-endpoints use97.5% intervals (Bonferroni), secondary95%. Fixed contiguous4-window block sensitivity uses7 blocks forN28 and8 forN32; no meaningful block interval forN4.

| Set / endpoint | Mean delta NLL | Relative PPL change | Paired window interval | Four-window block interval |
|---|---:|---:|---:|---:|
| Primary5-32, single | 0.01409605 | +1.420% | [0.865%, 1.997%] | [0.909%, 1.875%] |
| Primary5-32, later | 0.01936648 | +1.956% | [1.071%, 2.933%] | [1.372%, 2.589%] |
| Inspected1-4, single | 0.01597926 | +1.611% | [0.471%, 2.903%] | not defined |
| Inspected1-4, later | 0.02828845 | +2.869% | [1.550%, 3.909%] | not defined |
| Secondary all32, single | 0.01433146 | +1.443% | [0.991%, 1.912%] | [1.068%, 1.797%] |
| Secondary all32, later | 0.02048173 | +2.069% | [1.358%, 2.834%] | [1.565%, 2.587%] |

Primary packed schedule contrast / interaction: +0.528% ratio change,95% paired interval[-0.078%,1.191%], block sensitivity[-0.064%,1.262%]. Secondary all32 interaction +0.617%,95% interval[0.047%,1.229%]. BF16 single/later scores are exactly identical at all4096 token positions. The primary interaction interval includes zero; do not claim a conclusive schedule interaction from the secondary result.

These are end-to-end physical-serving differences for this pinned model, benchmark, and configuration, not quantization-only effects, broad model-quality proof, or throughput/capacity evidence. Both primary packed-vs-BF16 intervals remain positive under registered block sensitivity. Contiguous benchmark windows may remain dependent beyond four-window blocks.

## Validity and independent audit

The663-file raw archive is1,945,435,135 bytes; independent remote inventory matches local paths/sizes exactly. Every raw-file hash, staged runtime hash, frozen input/protocol hash, model-cache hash, effective configuration, and source identity passed. Returned summary equals archived summary; every child exited0 in fixed order. No failure tensors/artifacts or forbidden fallback were found. The independent audit passed before statistical analysis started.

Checks cover all32 windows/all32 layers: exact IDs and scoring positions (target1024+j from logit1023+j); original-logit scores before forced-ID override; finite4096 scores/cell and raw engine agreement<2e-5; actual1024 versus512+512 prefill schedules; RoPE/positions/causal masks; packed uint8 pages of24832 bytes, physical write/read chronology, sidecar cleanup, aging and independent packed-state/reference checks. Packed attention checkpoint maximum errors were0.0156309605(single) and0.0154438019(later), both below the unchanged strict0.025 gate.

Local tensor replay: six original-logit tensors/cell,21 saved packed-attention tensors(single),39(later), all passed. Detailed tensors are restricted to window1 as preregistered; all windows retain online reference events, token scores, and complete physical paths. Codec reference ancestry is shared with serving; this limits independence. BF16/HF aggregate discrepancy is +0.171594%, below0.5%; maximum absolute BF16-HF per-token NLL difference0.22220206. BF16-HF per-window PPL discrepancies range from -0.499612% to +0.701512%; the registered0.5% gate applies to aggregate PPL, not each window. Per-window discrepancies and every paired per-token difference are exported without exclusion. Passing numerical gates does not imply every token is identical or eliminate all possible bugs.

Eight offline integration tests passed before launch. The detached infrastructure check passed client-exit survival, independent status/result retrieval, persisted mount access, and cleanup. Earlier infrastructure failures (volume hydration and module mount) are preserved and diagnosed; they are not scientific observations. No N32 rerun/fix was required for attempt002.

## Cost and cleanup

Measured execution 3904.008s; startup 6.154s. Estimated compute $4.78-$5.06; archive download $0.072; illustrative five-day retention $0.027. Attempt002 total about $4.88-$5.16; allow another $0.10 conservatively for infrastructure checks. No actual invoice total is exposed. This is an estimate, not a guaranteed billing cap or remaining-credit claim.

Rates assumed: H100$3.95/h,4CPU cores$0.1892/h,32-64GiB RAM$0.256-$0.512/h; storage$0.09/GiB-month and transfer$0.04/GiB. Cached setup is included in execution; no new image build or model download. Existing image/model-cache baseline and earlier scientific attempts are excluded from incremental attempt002 cost; no credits/free allowance deducted. Retention is indefinite until authorized removal, approximately$0.00544/day for this result volume. Setup/state details are in registration and cost-estimate.json.

## Saved evidence and commands executed

`archive.json`, `remote-inventory.json`, `archive-completeness.json`, `artifacts/validity-audit.json`, `independent-audit.log`, `frozen-analysis.json`, `paired-token-differences.csv`, `paired-window-differences.csv`, `supplemental-checks.json`, `cloud.log`, `final-app-info.txt`, `cleanup.log`, `usage-20261021T180759Z/`, `cost-estimate.json`. All are under this attempt directory.

From repository root, using `working-papers/physical-serving-pilot/.venv/Scripts/python.exe`:

```text
-B working-papers/physical-serving-pilot/n32-20261020/collect_detached.py --attempt working-papers/physical-serving-pilot/n32-20261020/runs/attempt-002
-B working-papers/physical-serving-pilot/n32-20261020/audit_n32.py --attempt working-papers/physical-serving-pilot/n32-20261020/runs/attempt-002
-B working-papers/physical-serving-pilot/review-20261020/analyze_n32.py --scores-root working-papers/physical-serving-pilot/n32-20261020/runs/attempt-002/artifacts --output working-papers/physical-serving-pilot/n32-20261020/runs/attempt-002/frozen-analysis.json
-B working-papers/physical-serving-pilot/n32-20261020/export_pairs.py --attempt working-papers/physical-serving-pilot/n32-20261020/runs/attempt-002
```

These commands already completed. They use exclusive/fresh outputs and should not be blindly rerun into these existing files. No further compute is needed for this completed evaluation. Attempt001, failed infrastructure attempts, manuscripts, preregistration, historical evidence, and research/release refs remain preserved; no commit/push/publication/submission.
