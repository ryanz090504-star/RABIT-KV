# Public release history and path migration

Mutable release-maintenance document, separate from the immutable research freeze and accepted evidence. The reviewed path migration below is now applied. Statements about no moves in earlier sections describe those earlier preparation phases.

## Preserved starting state

The first documentation preparation started from clean commit `45fd03d8c10f2c14f96dcaedfcad25c4f56fe746`. The preservation tag `research-backup-2026-10-11` and branch `release/public-v1` were created at that commit without overwriting existing refs. That phase updated only README.md, docs/METHOD.md, docs/REPRODUCIBILITY.md, and results/summary.json. No manuscript, scientific implementation, frozen benchmark, raw result, or frozen evidence was changed or moved.

There is no root or documentation/result-specific AGENTS.md. The vendored [vLLM instructions](../vllm-kvquant/AGENTS.md) govern contributions there. The preserved research records define canonical/legacy evidence boundaries and freeze the scientific implementation and measurements. The manuscript and historical records retain their original format and names; this public guide assigns no venue or submission status.

## Reviewed experiment-based path mapping — applied

Retain `README.md`, `docs/METHOD.md`, `docs/REPRODUCIBILITY.md`, `results/summary.json`, `paper/`, and `vllm-kvquant/`. Keep frozen paper assets and style filenames unchanged. The following exhaustive prefix mapping records the old and current experiment/result directories; descendants and attempt filenames retain their relative suffixes, statuses, and bytes.

| Historical prefix | Current public prefix |
|---|---|
| `benchmarks/mlsys2027/` | `benchmarks/experiments/` |
| `benchmarks/quality/` | `benchmarks/legacy/quality/` |
| `benchmarks/performance/` | `benchmarks/legacy/performance/` |
| `results/quality/` | `results/legacy/quality/` |
| `results/performance/` | `results/legacy/performance/` |
| `results/mlsys2027/canonical_quality_v2/` | `results/experiments/canonical_quality_v2/` |
| `results/mlsys2027/external_baseline/` | `results/experiments/matched_serving_baselines/` |
| `results/mlsys2027/context_scaling/` | `results/experiments/context_scaling/` |
| `results/mlsys2027/concurrency_scaling/` | `results/experiments/concurrency_scaling/` |
| `results/mlsys2027/concurrency_smoke/` | `results/experiments/concurrency_smoke/` |
| `results/mlsys2027/deployment/` | `results/experiments/deployment_validation/` |
| `results/mlsys2027/fp8_baseline/` | `results/experiments/fp8_baseline/` |
| `results/mlsys2027/second_model/` | `results/experiments/cross_model/` |
| `results/mlsys2027/quality_semantic_audit/` | `results/experiments/cache_semantics_audit/` |
| `results/mlsys2027/control_reproducibility_audit/` | `results/experiments/control_reproducibility_audit/` |
| `results/mlsys2027/diagnostics/` | `results/experiments/diagnostics/` |
| `results/mlsys2027/quality_frontier/` | `results/legacy/quality_frontier/` |
| `results/mlsys2027/multilingual_frontier/` | `results/legacy/multilingual_frontier/` |
| `results/mlsys2027/ablations/` | `results/legacy/ablations/` |
| `results/mlsys2027/variance/` | `results/legacy/quality_variance/` |

Under `cross_model/`, the existing `quality/` subtree remains explicitly legacy, while `serving/` remains physical. Mixed audit and diagnostic directories retain their per-record classifications. Renaming a parent does not change evidence status. All benchmark files beneath the renamed root, including `exp7_kbit/` through the existing ablation directories, retain their names and relative imports initially; no scientific refactor is part of the migration.

| Historical document | Current public document |
|---|---|
| `docs/MLSYS_SUBMISSION_FREEZE.md` | `docs/history/MANUSCRIPT_FREEZE.md` |
| `docs/MLSYS_RELATED_WORK_SOURCE_MAP.md` | `docs/history/RELATED_WORK_SOURCE_MAP.md` |
| `docs/MLSYS_PAPER_STORY_LOCK.md` | `docs/history/CAPACITY_COST_STORY_LOCK.md` |
| `docs/MLSYS_PAPER_EVIDENCE_MAP.md` | `docs/history/PAPER_EVIDENCE_MAP.md` |
| `docs/MLSYS_PAPER_CLAIM_AUDIT.md` | `docs/history/PAPER_CLAIM_AUDIT.md` |
| `docs/MLSYS_PAPER_BLUEPRINT.md` | `docs/history/PAPER_BLUEPRINT.md` |
| `docs/MLSYS_EXPERIMENT_PLAN.md` | `docs/history/EXPERIMENT_PLAN.md` |
| `docs/AUTHOR_REVIEW_PACKET.md` | `docs/history/AUTHOR_REVIEW_PACKET.md` |

These documents preserve historical content; changing their names would not erase their historical venue references. They remain an archive, while the public guides stay venue-neutral.

### Complete reference-migration procedure

1. Inventory every tracked path and every textual occurrence of the old prefixes/document names with `git ls-files` and `rg`, including README/docs, paper sources/bibliography, Python imports, `ROOT / ...` constants, Modal source mounts, subprocess/module commands, protocol JSON, result JSON/JSONL/logs, tests, vendored integration helpers, and `.gitignore`. Include slash/backslash variants, relative links, bare document names, absolute historical paths, and Git `commit:path` references. Map each tracked descendant by the longest matching prefix above; reject collisions or pre-existing destinations. Ignored caches/build outputs are not evidence and should not be copied.
2. Before any separately authorized move, record an old-path/new-path manifest with Git blob id, raw SHA-256, LF-normalized hash where used by a gate, evidence class, attempt/acceptance status, and source commit. Preserve the backup tag and all original refs. Require complete tracked-file coverage; unchanged paths receive identity mappings.
3. Move only after that manifest and scope are reviewed. Copy/move immutable evidence byte-for-byte. Do not rewrite path strings inside frozen protocols, result records, logs, manifests, acceptance records, PDFs, or manuscript sources. Their historical `commit:path` identities remain valid. Retain a read-only compatibility tree or external path resolver for old runtime paths; directory redirects alone are insufficient for Git history checks.
4. In a separately authorized mutable adapter/documentation change, update active Markdown links, summary source paths, script entry points, Python package/import references, source upload mappings, output roots, and ignore patterns. Frozen runners import sibling helpers and contain output paths, protected-path checks, provenance hashes, and historical tree/commit checks: adapt them through an external adapter/resolver rather than silently rewriting frozen scientific code. Keep old Git-object references for provenance and supply current-path aliases in the migration index.
5. Generate new derived indexes/provenance records that reference both original and migrated paths. Hashes of frozen bytes must match before/after; never recalculate acceptance or change an experiment's status merely because it moved. Any changed wrapper or protocol is a new version with separate hashes and validation, not a retroactive edit of an accepted record. Paper references remain historical until a separately authorized new manuscript revision; do not invalidate the current freeze.
6. Validate all local links/anchors and machine-readable source references, imports/mounts/output destinations in the adapter, historical Git-object lookups, path-resolver completeness, and manifest coverage. Compare derived values to unchanged evidence, confirm no accepted/invalid attempts were dropped, and review the full diff for scientific changes. Runtime checks or GPU reruns require separate scope; path validation alone is not reproduction.

## Documentation verification performed

The audit recalculated canonical PPL from raw recorded window loss sums and token counts, checked token-level NLL sums against the recorded FP32 reductions, recalculated both HotpotQA scorer means from raw per-example scores, counted NIAH passes and Passage Retrieval examples, and matched reported confidence intervals to the accepted records. It parsed every matched Llama serving sample from the raw session, recomputed leg medians and cross-leg medians, and checked capacity blocks, ratios, and TPOT percentage costs. Qwen serving samples and the maximum-context cell medians were independently checked, and accepted concurrency comparisons support lower throughput at every tested point.

The primary audit passed 308 assertions, including all 54 local Markdown links in the public guides. Additional checks passed for Qwen/context/concurrency evidence, manuscript source and frozen-PDF hashes against the freeze record, historical-summary Git-blob identity, and complete proposed mapping of the existing experiment directories. Source hashes in the public JSON identify unchanged local evidence bytes. The Git diff contains exactly the four authorized files; HEAD remains the starting commit. These are evidence/document checks, not new scientific experiments or GPU reproduction.

## Public usability follow-up — 2026-10-11

Target-bit-aware wording and the meaning of the 2-bit target are restored in README. Release notes, validation history and proposed path migration now live here; REPRODUCIBILITY is focused on source evidence, environment and runnable adapter instructions. The new mutable files are `benchmarks/public/reproduce.py` and `benchmarks/public/test_reproduce.py`. Existing scientific scripts, protocols, paper sources and evidence remain unchanged. No paths have been moved and no compute jobs have run.

The adapter pins a separate preserved source checkout, checks exact regenerated protocol identity, maps output artifacts externally, requires explicit execution and rejects reused destinations/attempts. It retains the frozen quality run/evaluate functions and serving preflight/parser/statistics/integrity functions. It does not invoke the original two-model orchestrator's Git archive/push helper. First-model raw archival is local and uninspected; both-model registration and failure semantics are retained. Protocol output names are historical aliases, and the plan records the actual external paths.

Current execution limitations are concrete: this public checkout is dirty and has no matching upstream, so the original clean/pushed-source gates refuse it; use a separate existing clean research checkout at the pinned commit. Dependencies and Modal-client behavior have not been validated on a GPU run. Remote long-context artifact freshness must be established from the shared volume listing; ambiguous output/API errors fail closed. Registrations on the shared remote volume must not race for the same attempt number. The Python guard is for trusted frozen code/dependencies, not a replacement for OS isolation.

## Summary schema consumer audit

The repository-wide search covered benchmark code, the vendored implementation, documents and machine-readable evidence for `results/summary.json`, `CANONICAL_SUMMARY`, `final_quality`, `frozen_deployment`, and calls to `verify_canonical_reference`. No mutable executable consumer existed before the public adapter; the existing executable consumers belong to frozen legacy research. None were rewritten or supplied canonical values under legacy keys.

| Frozen direct consumer (original preserved-source path) | Concrete schema-v2 incompatibility | Resolution |
|---|---|---|
| `benchmarks/mlsys2027/build_exp1_summary.py:184` | Reads `final_quality.longbench_hotpotqa_e`; raises KeyError against public v2. The reference is historical small-sample legacy quality despite the old name “canonical”. | Preserve the builder; use its historical checkout/summary Git object when inspecting legacy derivation. Do not rebuild accepted output in place. |
| `benchmarks/mlsys2027/validate_exp1_summary.py:149` and `:159` | Requires the old summary source SHA-256 and then reads `final_quality.longbench_hotpotqa_e`; v2 fails the hash check and the removed-key lookup. | Preserve the validator and accepted provenance; validate historical evidence against the original summary bytes in a separate historical checkout, not v2. |
| `benchmarks/mlsys2027/run_experiment1_quality_frontier.py:282` | `verify_canonical_reference` reads `final_quality` for legacy continuation PPL, NIAH, passage retrieval and Qasper targets/memory values. Its pinned constants cannot be mapped to canonical-v2 results without changing scientific meaning. | Preserve and document as a legacy-only runner, incompatible with public v2. The public adapter never calls this runner's main/preflight/reference check. Its read-only Git/stream helpers used by serving do not read the removed keys. |
| `benchmarks/mlsys2027/run_experiment2_multilingual_frontier.py:234` | Reads `final_quality.multilingual_continuation_ppl`, pinned Chinese/Spanish values and settings; no canonical multilingual counterpart exists. | Preserve and document as legacy-only, incompatible with public v2; no compatibility values are fabricated. |

Transitive callers of the Experiment 1 reference check are `run_experiment7_kbit_ablation.py:328`, `run_experiment8_vbit_ablation.py:221`, `run_experiment9_group_ablation.py:333`, `run_experiment10_residual_ablation.py:316`, `run_experiment11_metadata_ablation.py:413`, and `run_experiment12_variance.py:366` under `benchmarks/mlsys2027/`. These legacy ablation/variance preflights are likewise incompatible with v2.

No executable repository consumer of `frozen_deployment` was found. Its public replacement is `matched_llama_serving`, with explicit per-method capacities, median-of-leg-medians latency and within-session percentage costs; it is not a shape-compatible substitution for the unmatched old deployment result. Removed top-level `model` also has no safe universal replacement: select the model-specific canonical/serving section. Qasper and multilingual keys have no canonical replacement.

Frozen manifests/protocols and `docs/MLSYS_PAPER_EVIDENCE_MAP.md` still reference the old path/schema or call legacy evidence “canonical” in their historical vocabulary. Those bytes and recorded hashes remain unchanged. The v2 summary records `historical_provenance.previous_summary_git_object`, its blob id and the old raw paths. Mutable public readers should branch on `schema_version == 2`, use `canonical_quality`, `matched_llama_serving`, `qwen_serving` and `scaling_limitations`, and keep `historical_provenance` separate. The new public adapter reads frozen protocols/results and does not consume the public summary or restore removed keys.

The earlier four-file validation section above describes the initial documentation phase only. The follow-up export includes the full cumulative diff, new mutable files, offline adapter test output and frozen-file preservation checks; no commit, push or merge is part of either phase.

## Offline follow-up validation

The public adapter suite now includes replay of both models' accepted raw PPL results, the canonical long-context raw result, and the matched serving session through the frozen evaluation/parser/statistic functions. These are offline evidence replays, not new scientific experiments. It also checks inherited Python-child write/publication guards and records invalid attempts when a required source gate fails. No GPU or Modal job is launched by the suite. Detailed test output and preservation hashes are exported outside the repository for review.

The remote freshness parser was checked against Modal's [official volume command implementation](https://github.com/modal-labs/modal-client/blob/main/py/modal/cli/volume.py) and [table serializer](https://github.com/modal-labs/modal-client/blob/main/py/modal/cli/utils.py). It recognizes the current `filename` key and historical `Filename` spelling, rejects missing/non-string/conflicting names, and detects an existing attempt before the unchanged long-context worker may append. Offline tests use representative JSON listings; no remote listing or job was executed during this follow-up.

## Applied path migration — 2026-10-11

Applied all 28 reviewed prefix/document mappings: 911 tracked files moved, with all 6,730 original tracked paths covered by the pre-move [manifest](PATH_MIGRATION.json), including identity mappings. The manifest records Git blob IDs, raw and LF-normalized SHA-256, evidence class and explicit original status sources; missing local acceptance is stated rather than inferred. Canonical PPL model artifacts also cite the separate two-model acceptance record. No failed or legacy result was promoted.

Frozen code, protocols, evidence, historical documents and paper bytes were moved without text edits. Their embedded historical references remain unchanged. The mapping index resolves old paths to current browseable locations, while original Git-object lookups remain valid. Historical documents are navigated from the [history index](history/README.md), scientific workers from the [source index](../benchmarks/README.md), and measurements from the [evidence index](../results/README.md). Internal links in frozen historical files are historical references, not active public navigation links.

Mutable public guides, summary paths and root ignore patterns use the new locations. The public summary retains original Git-object references and original-path provenance alongside the current source aliases. `reproduce.py` is byte-for-byte unchanged: both planning and execution require the separate pinned preserved Git source checkout, with the original path geometry and unchanged gates. The relocated frozen workers must not be launched directly from this public checkout.

Offline tests require `RABIT_PRESERVED_SOURCE` pointing outside the public repository. They use an external pinned checkout or complete original-path fixture with per-file raw hashes and source commit. This migration run created and verified a complete external offline fixture before moving evidence; it is not a fabricated Git checkout or a production execution source. Required scientific checks, statistics, two-model registration, artifact/archival validation, cancellation and frozen cleanup remain in force. Modal and source-readiness transports are mocked only in offline tests.

The self-contained external migration packet includes the complete manifest, move list, mutable-file diffs and validation/test logs. No GPU/remote operations, commit, push, merge or ref changes are part of this migration. Ignored caches and ignored reproduction outputs are outside the tracked migration scope and are not copied or deleted.
