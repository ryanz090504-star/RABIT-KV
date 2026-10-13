# RABIT-KV: Initial public research release (draft)

This is a preparation document, not a published release. The author entry in
`CITATION.cff` is a draft based on Ryan Zhang's account of his contributions.
No named acknowledgement or institutional endorsement is asserted.

## Proposed release description

RABIT-KV provides a target-bit-aware KV-cache representation and physically
packed serving implementation at the frozen K3/V2/G32/R4/META8g64 operating
point. This initial public research release organizes the implementation,
accepted evidence, historical records, and reproduction instructions.

The 2-bit target refers to an operating point centered on 2-bit values, not
two-bit total storage: keys use 3 bits, metadata adds storage, and the newest
four tokens remain BF16.

### Included

- Physical serving implementation in the preserved vLLM fork.
- Canonical logical quality evaluator and accepted result records.
- Matched serving evidence, scaling results, and explicit limitations.
- A guarded public reproduction adapter with planning and explicit execution
  into fresh external output directories.
- Public method and reproduction guides, a schema-v2 results summary, and a
  complete historical-to-current path migration manifest.
- Preserved manuscript sources and frozen PDFs as research artifacts; their
  inclusion does not claim publication, peer review, or conference acceptance.

### Results and boundaries

- On the evaluated H100 configuration, Llama-3.1-8B-Instruct's observed paged
  KV capacity is 5.28x BF16, 2.64x FP8, and 1.72x the tested TurboQuant
  configuration. This is allocator-pool capacity, not total-device-memory
  compression or a throughput multiplier.
- In the matched serving workload, RABIT-KV TPOT is 24.42% higher than BF16
  and 8.33% higher than tested TurboQuant. Throughput is lower in the evaluated
  concurrency workloads.
- Canonical Llama continuation perplexity increases by 1.56%, with a paired
  95% confidence interval of [+1.12%, +2.05%]. Qwen continuation perplexity
  increases by 1555.9% at the same operating point; the cause is undiagnosed.
- Quality results come from a validated logical evaluator. They do not
  establish end-to-end physical serving quality. Legacy quality results remain
  explicitly separate from canonical evidence.
- Evaluation covers one GPU type and one frozen configuration. Long-context
  latency, serialized execution, and model-dependent quality remain material
  limitations.

See the [README](../README.md), [method](METHOD.md),
[reproduction guide](REPRODUCIBILITY.md), and
[public results summary](../results/summary.json) for settings, source records,
confidence intervals, and interpretation.

### Reproduction and validation

The adapter requires a separate clean checkout of the pinned preserved
scientific source. Relocated frozen workers must not be launched directly from
the public checkout. Follow the reproduction guide's environment prerequisites
and validity gates.

Earlier public preparation included offline adapter checks and evidence
replays. These are not new GPU experiments. No new GPU or Modal reproduction
was run during preparation of this metadata draft, and no GitHub CI success
is claimed.

### Provenance

- Reviewed public baseline:
  `52689ca532a833ee615c59b5c61fd16c2d504ce2` (merged PR #1).
- Preserved scientific source:
  `45fd03d8c10f2c14f96dcaedfcad25c4f56fe746`.
- Preservation tag: `research-backup-2026-10-11`.
- Research branch: `research/mlsys-2027`.

These metadata additions do not alter scientific code, raw results, protocols,
paper assets, or the preserved research references.

## Licensing decision

Ryan Zhang has approved Apache-2.0 for his original RABIT-KV software code.
The official license text is in [the root LICENSE](../LICENSE), and
[licensing scope and third-party exceptions](LICENSING.md) define the grant.
Existing licenses and notices remain in place. Paper assets, research data,
and third-party materials are excluded from any blanket relicensing claim.
The `CITATION.cff` license entry describes the covered software subject to that
scope. Frozen scientific files remain byte-identical.

## Finish before publishing

- Confirm the final author metadata with Ryan Zhang. Affiliation is omitted
  from the citation draft until he confirms the desired representation.
- Choose the release tag/version and the exact final release commit. A
  preliminary tag such as `v0.1.0` is a proposal, not an existing release.
- If automatic Zenodo archival is desired, enable the repository in Zenodo
  before publishing the GitHub Release. No DOI or archive record is asserted.
- Add the actual version and release date to `CITATION.cff` when decided;
  add DOI and paper citation only after valid records exist. ORCID is optional.

Keep this document marked as draft until these decisions are complete. Extract
the proposed description and update its provenance when creating the actual
release. Preparing this file does not create a tag, publish a GitHub Release,
submit to arXiv, or submit to a conference.
