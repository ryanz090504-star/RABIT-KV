# Reproducibility and evidence provenance

## Canonical paper and evidence hierarchy

The preserved manuscript is [paper/main.tex](../paper/main.tex), with [supplement](../paper/appendix.tex), [bibliography](../paper/references.bib), [frozen main PDF](../paper/frozen/main.pdf), and [frozen supplement PDF](../paper/frozen/appendix.pdf). The last main-source commit is `21b931930dc2c8d433b9b5a42e12418bda4f23a3`; the starting commit contains the subsequent freeze record and PDF build. The historical [freeze record](history/MANUSCRIPT_FREEZE.md) supplies source/blob/PDF hashes. It is provenance, not a public submission claim.

[results/summary.json](../results/summary.json) is a derived public index. Accepted records and immutable raw artifacts below remain authoritative. An attempt summary's historical “awaiting review” text is superseded by its separate acceptance record, not silently edited.

| Evidence class / reported quantities | Authoritative source and fields | Supporting acceptance / raw evidence |
|---|---|---|
| Physical Llama capacity and matched latency | [matched summary](../results/experiments/matched_serving_baselines/exp13/matched_capacity_latency_summary.json): `observed_capacity`, `capacity_ratios`, `latency_primary_cross_leg`, `rabit_comparisons` | [acceptance](../results/experiments/matched_serving_baselines/exp13/acceptance_record.json), [integrity](../results/experiments/matched_serving_baselines/exp13/integrity_check.json), [raw session](../results/experiments/matched_serving_baselines/exp13/modal_session.log), manifest in the same directory |
| Canonical two-model continuation PPL / paired CIs | [attempt summary](../results/experiments/canonical_quality_v2/continuation_ppl/two_model_attempt_2/summary.json): `models.<model>.aggregate`, `bf16_ppl`, `rabit_ppl`, `delta_pct`, `delta_pct_ci95_paired_bootstrap` | [acceptance](../results/experiments/canonical_quality_v2/continuation_ppl/two_model_attempt_2/acceptance_record.json); model-specific `attempt_2/result.json`, `record.json`, `session.log` under the continuation_ppl directory |
| Canonical Llama retrieval / QA | [attempt summary](../results/experiments/canonical_quality_v2/long_context/attempt_1/summary.json): `niah`, `passage_retrieval`, `hotpotqa.primary_legacy_scorer`, `hotpotqa.secondary_official_scorer` | [acceptance](../results/experiments/canonical_quality_v2/long_context/attempt_1/acceptance_record.json), [raw result](../results/experiments/canonical_quality_v2/long_context/attempt_1/result.json), incremental rows and session log in the same directory |
| Qwen physical serving / fallback paths | [serving summary](../results/experiments/cross_model/serving/capacity_latency_summary.json): `observed_capacity`, `capacity_ratio_rabit_over_bf16`, `latency_primary_cross_leg`, `rabit_vs_bf16` | Integrity, manifest, and session artifacts in that directory; legacy quality there is a separate evidence class |
| Maximum tested context point / TTFT | [context summary](../results/experiments/context_scaling/attempt_2/context_scaling_summary.json): `per_context.32768.actual_prompt_tokens`, `output_tokens`, `latency_signed_deltas_rabit_minus_bf16.ttft_median` | Accepted unprofiled cells, samples, source hashes, and reparse provenance in that directory |
| Concurrency / admission / throughput | [8k combined report](../results/experiments/concurrency_scaling/L8192/shadow_conditioned/combined_report.json): `capacity`, `metrics`, `rabit_over_bf16`, `interpretation`; [2k summary](../results/experiments/concurrency_scaling/L2048/shadow_conditioned/summary.json) | Per-trial accepted records and raw logs; excluded attempts remain archived |
| Stored-state conformance | [canonical final record](../results/experiments/canonical_quality_v2/canonical_quality_final_record.json): `evidence` and `limitations` | `results/experiments/cache_semantics_audit/parity_attempt_3/` and `results/experiments/canonical_quality_v2/cuda_conformance_diagnostic/`; same-device independent oracle |
| Profiling explanation, never replacement latency | [profile summary](../results/experiments/diagnostics/perf_risk_profile/attempt_1/summary.json), [corrected attribution](../results/experiments/diagnostics/perf_risk_profile/attempt_1/corrected_attribution.json) | Raw legs and session in that directory; timing materially perturbed |

## Numerical reporting rules

The [README](../README.md) contains the public tables. Full-precision measurements in the public JSON are copied from the sources above, with source-field mappings. Relative changes use unrounded measurements; displayed absolutes are rounded.

- Capacity ratios are observed RABIT tokens divided by the comparator's observed tokens. Llama ratios are 5.278537697443413 (BF16), 2.6408815023015193 (FP8), and 1.7242287234042553 (tested TurboQuant).
- Matched latency is the median of the two leg-level medians, not the secondary pooled median. TPOT percentage cost is `100 * (RABIT / comparator - 1)`; Llama costs are +24.42% versus BF16 and +8.33% versus tested TurboQuant. No between-session pooling or latency confidence interval is claimed.
- Continuation PPL is `exp(total NLL / tokens)`; each arm scores 4,096 tokens from 32 windows of 128 scored tokens, with 1,024 context tokens per window. Relative delta is `100 * (RABIT_PPL / BF16_PPL - 1)`. Paired percentile bootstraps use 10,000 resamples over windows; CIs are conditional on the recorded evaluation.
- NIAH has 19 depths at each of 4,096, 8,192, and 16,384 tokens, 57 cases total, with 57 passes in both arms. It is a deterministic shared-construction grid, not independent draws. Passage Retrieval has 200 saturated examples. HotpotQA has 100 examples; primary and official secondary scorers use identical predictions. Confidence intervals spanning zero do not establish equivalence or remove greedy-generation variability.
- The context point labeled 32768 actually has 32,736 prompt and 32 output tokens. Its source TTFT values are 114376.83296203613 and 4435.004711151123 ms, displayed as 114.38 and 4.435 seconds. At offered concurrency 64 with 8,192-token prompts, the three accepted trials realize RABIT maximum in-flight 64 and BF16 47. These are admission/overlap observations, not throughput gains or a measured total-memory saving.
- Method byte counts are closed-page accounting, distinct from allocator observations: 24,832 bytes/page/layer, divided by 32 gives 776 bytes/token-layer; `776 * 8 / 2048` gives 3.03125 bits/element. Open-group and residual state are excluded. The policy constants and lifecycle are read from the canonical evaluator and physical implementation, not inferred from historical quality scores.

## Recorded environment and matched controls

Canonical quality records pin model revisions, datasets, prompts, shipped source hashes, and runtime. Llama revision is `359efdbb8af05b788a4ad4185215c6b8caa9052c`; Qwen revision is `16c174980d8a1492910551634b4969e69cdc2444`. Runtime is Python 3.11, PyTorch 2.11.0+cu130, CUDA 13.0, Transformers 4.48.2, on one H100 80GB HBM3 per run. Use each experiment's own manifest rather than assuming every historical session shared that runtime.

Matched serving uses custom vLLM `0.10.0+kvquant`, upstream base `f329ce405b12623fb8b1cf1830f12e5a712523be`, default V2 runner, BF16 model weights, block size 32, GPU memory utilization 0.82, max model length 32768, max batched tokens 16384, max sequences 32, eager execution, chunked prefill enabled, and prefix caching disabled. CUDA graphs and compilation are disabled. The four-condition order is mirrored A B C D D C B A, two fresh-engine legs per condition, 30 measured samples per leg, single request, greedy output with ignore_eos.

FP8 is native per-tensor e4m3 with default 1.0 scales. Tested TurboQuant is `turboquant_k3v4_nc`, with 28 quantized and four BF16 boundary layers. All four methods run on the same snapshot with the upstream V2 cache-dtype integration fix from `fa4321de3d894c50c5ca0766dffa352d3fb07423` backported. The fix changes cache-dtype dispatch, not the TurboQuant quantizer, layout, boundary-layer policy, or kernels. TurboQuant uses its own backend plus FlashAttention v2 for boundary layers; other methods use TRITON_ATTN. The latency bias direction of that backend difference is unknown. This eager system comparison need not transfer unchanged to graph-enabled deployments.

## Historical-result provenance

The old public summary is retained byte-for-byte in Git at `research-backup-2026-10-11:results/summary.json` (commit `45fd03d8c10f2c14f96dcaedfcad25c4f56fe746`). Inspect it read-only with:

```powershell
git show research-backup-2026-10-11:results/summary.json
```

The public JSON replaces its obsolete schema and records that Git object and the raw evidence paths. Consumers of `final_quality` or `frozen_deployment` must migrate to the explicit canonical-quality, matched-serving, and historical-provenance sections.

| Historical evidence | Classification and permitted use |
|---|---|
| `results/legacy/quality/`, `benchmarks/legacy/quality/` | Original small-sample legacy evaluator; includes Qasper and multilingual results. Provenance only, no canonical headline |
| `results/legacy/performance/` | Historical physical deployment/capacity; unmatched latency cannot replace the primary matched comparison |
| `results/legacy/quality_frontier/`, `results/legacy/multilingual_frontier/`, `results/legacy/ablations/` | Legacy evaluator frontier and K/V/group/residual/metadata ablations; no canonical ablation claim |
| `results/legacy/quality_variance/` | Expanded legacy quality, including Qasper; kept distinct from canonical expanded quality |
| `results/experiments/cross_model/quality/` | Legacy Qwen quality; not canonical Qwen long-context evidence |
| `results/experiments/deployment_validation/`, `results/experiments/fp8_baseline/` | Earlier physical sessions; preserve acceptance and excluded attempts; no pooling with primary matched latency |
| Semantic/control audits and invalid attempts | Provenance, correctness, or exclusion records; never convert invalid attempts into accepted quality |

The legacy evaluator flattened V metadata differently and did not age generated tokens. Both differences were corrected for canonical evaluation; no experiment decomposes their separate causal effects. Legacy multilingual and Qasper results remain unvalidated under canonical semantics.

## Runnable public reproduction

Use the new [public adapter](../benchmarks/public/reproduce.py), not the preserved top-level research runners. Its default mode only prints a plan: no output directory is created and no Modal command or GPU job runs. Each invocation selects one experiment and pins its source to preserved commit `45fd03d8c10f2c14f96dcaedfcad25c4f56fe746`. The protocol bytes and regenerated protocol must agree before and after output mapping.

Requirements: Python 3.11 or newer for the local adapter, Git with the preserved history, and an existing external output parent. Execution additionally requires the frozen runners' static tests/dependencies, a configured Modal client/account, model access, and the recorded H100/container environment. The remote scientific runtimes remain pinned by the unchanged workers; the adapter does not install dependencies or relax runtime checks.

The frozen gates require a **clean source checkout at the pinned commit with an existing upstream ref at the same commit**. This public checkout now has mapped browsing paths and uncommitted public edits. Supply a separate existing preserved research checkout using `--source-root` for both planning and execution; the adapter deliberately loads the original `benchmarks/mlsys2027/` and `results/mlsys2027/` paths there. Relocated scientific scripts are browseable archives, not alternate execution entry points. The adapter does not create a checkout, alter refs, commit, or push to satisfy a gate. Missing history, upstream, proofs, conformance artifacts, static checks, dependencies, or protocol identity causes refusal before compute.

From the repository root, replace the absolute paths below with your clean preserved source checkout and an existing external output parent. Choose a new, absent directory for **each** experiment:

```powershell
# Read-only plans; these do not execute any experiment.
python -B benchmarks/public/reproduce.py canonical-ppl --source-root C:/scratch/rabit-research --output-dir C:/scratch/rabit-runs/ppl-run-001 --attempt 1001
python -B benchmarks/public/reproduce.py long-context --source-root C:/scratch/rabit-research --output-dir C:/scratch/rabit-runs/longctx-run-001 --attempt 1001
python -B benchmarks/public/reproduce.py matched-serving --source-root C:/scratch/rabit-research --output-dir C:/scratch/rabit-runs/serving-run-001
```

To launch a reproduction when the plan and prerequisites are ready, repeat the selected command with **`--execute`**. There is no default execution or automatic retry. Existing output directories, links/junctions, repository-contained outputs, and preserved attempt numbers are rejected. Long-context execution also requires a parseable remote crash-recovery-volume listing showing the attempt path absent; if freshness cannot be established it refuses to append. The [Modal volume CLI](https://modal.com/docs/cli/latest/volume) listing is read-only; the adapter recognizes its `filename`/`Filename` table keys and refuses unknown or conflicting schemas. Use a fresh attempt number and avoid concurrent registrations of the same number on the shared Modal volume.

The adapter calls the original per-model PPL and long-context run/evaluate functions, retaining cleanup and validity gates. PPL registration precedes the first model, local raw-file archival precedes the second, and the second runs only if the first passed. Both must pass for the whole attempt to be valid. A failed model or archive invalidates the entire attempt; no one-model result is accepted and no selective rerun is supported. A new attempt uses a new directory and number.

Matched serving reuses the unchanged source snapshot builder, preflight, Modal worker, mirrored schedule, measurement counts, parser, statistics and integrity checks. Historical artifact names in the frozen protocol are mapped to physical files under the external output directory; they are not rewritten in the protocol. The new plan records the historical/physical aliases. The original passed-manifest overwrite check therefore applies to the new destination, while source-history and correctness checks still apply to the preserved research inputs.

Outputs include `plan.json`, an external `tmp/`, experiment artifacts under `artifacts/`, and `completion.json`. PPL also writes `registration.json`, `two_model_record.json`, and per-model local archival hashes. Exceptions retain failure/invalid records. Successful new results are **reproduced/unreviewed**, not accepted canonical evidence. Nothing updates the public summary or overwrites frozen records.

Python file-write checks confine adapter and inherited Python-child writes to the registered output tree. Child commands allow read-only Git, the frozen vLLM snapshot archive into that tree, frozen static tests, and explicit Modal operations; Git mutation/publication and shell commands are denied. Unknown operations fail closed. This is a guard for trusted frozen code and dependencies, not an operating-system security boundary against hostile code. Additional dependency/client behavior can cause refusal and has not been validated on an actual public GPU reproduction.

## Offline adapter validation

Supply an external source checkout at the pinned commit, or a complete byte-bound offline preserved-source fixture. The fixture is test data and cannot replace a Git checkout for production execution. It retains the original paths and all 6,730 tracked files; its `_public_offline_fixture.json` binds the source commit and every raw file hash. The tests reject the migrated public tree as their scientific source and verify fixture coverage/hashes before loading workers.

```powershell
$env:RABIT_PRESERVED_SOURCE = "C:/scratch/rabit-research"
python -B benchmarks/public/test_reproduce.py
```

The offline suite checks output confinement, overwrite rejection, Git-publication denial, explicit execution, exact frozen protocol identity for all three modes, remote-freshness refusal, source readiness, and both-model sequencing/failure behavior. The suite exercises the original frozen preflight/run/evaluation/cleanup functions, with source-cleanliness, static-child responses and Modal transports mocked for offline execution. Actual scientific byte/protocol, validity and statistic checks remain unchanged. No GPU/Modal jobs or model downloads run. If the system temp directory is restricted, set `RABIT_TEST_TEMP` to an existing writable external directory before testing.

The concrete schema-consumer incompatibilities and applied migration record are in [mutable public release history](PUBLIC_RELEASE_HISTORY.md). Scientific limitations remain: logical quality does not validate end-to-end serving or later packed-history prefill chunks; Qwen quality loss is undiagnosed; canonical Qwen long-context, Qasper, multilingual and ablations are absent. No GPU reproduction is claimed for this adapter yet.

## Path and historical-reference resolution

The [complete path manifest](PATH_MIGRATION.json) records every pre-migration tracked path, its current alias, Git blob ID, raw/LF SHA-256 and original status records. Frozen records and historical documents keep their embedded paths and hashes unchanged. For a historical path, find its `old_path` entry and browse `new_path`; historical `commit:path` lookups still use the original path. The [history index](history/README.md), [evidence index](../results/README.md) and [source index](../benchmarks/README.md) navigate the mapped public tree.
