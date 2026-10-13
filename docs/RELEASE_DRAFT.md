# RABIT-KV v0.1.0: Initial public research release (draft)

This is a preparation document, not a published release. Ryan Zhang has
confirmed the final software citation author as Ryan Zhang alone, with no
affiliation. `CITATION.cff` records this confirmed authorship.
No named acknowledgement or institutional endorsement is asserted.

The chosen software version is `0.1.0`, with proposed tag `v0.1.0`.
Preparation starts from GitHub main commit
`8c6727fa2b844247829b36dbff66842d0432c891` (merged PR #2).
The tag and GitHub releases were checked during preparation: `v0.1.0` did not
exist, and no GitHub releases existed. Recheck before publishing; do not
overwrite any existing tag or release. No tag or release is created here.
The exact final release commit remains to be selected after review.

## Proposed GitHub Release title

```text
RABIT-KV v0.1.0: Initial public research release
```

## Proposed GitHub Release body

The exact proposed body starts below and ends before "Finish before publishing".
Its links point to the verified preparation baseline.

<!-- BEGIN PROPOSED RELEASE BODY -->

RABIT-KV v0.1.0 provides a target-bit-aware KV-cache representation and physically
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

See the [README](https://github.com/ryanz090504-star/RABIT-KV/blob/8c6727fa2b844247829b36dbff66842d0432c891/README.md),
[method](https://github.com/ryanz090504-star/RABIT-KV/blob/8c6727fa2b844247829b36dbff66842d0432c891/docs/METHOD.md),
[reproduction guide](https://github.com/ryanz090504-star/RABIT-KV/blob/8c6727fa2b844247829b36dbff66842d0432c891/docs/REPRODUCIBILITY.md), and
[public results summary](https://github.com/ryanz090504-star/RABIT-KV/blob/8c6727fa2b844247829b36dbff66842d0432c891/results/summary.json) for settings, source records,
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

### Licensing and authorship

Ryan Zhang has approved Apache-2.0 for his original RABIT-KV software code.
The official license text is in [the root LICENSE](https://github.com/ryanz090504-star/RABIT-KV/blob/8c6727fa2b844247829b36dbff66842d0432c891/LICENSE), and
[licensing scope and third-party exceptions](https://github.com/ryanz090504-star/RABIT-KV/blob/8c6727fa2b844247829b36dbff66842d0432c891/docs/LICENSING.md) define the grant.
Existing licenses and notices remain in place. Paper assets, research data,
third-party materials, and non-code documentation are excluded from this
software grant; existing separate licenses or permissions continue to apply.
The `CITATION.cff` license entry describes the covered software subject to that
scope. Frozen scientific files remain byte-identical.

The confirmed software citation author is Ryan Zhang alone, with no affiliation.
The citation records version `0.1.0`; release date, DOI, and ORCID remain unset.
No DOI, archival record, peer review, or conference acceptance is asserted.

<!-- END PROPOSED RELEASE BODY -->

## Finish before publishing

- Select and verify the exact final release commit after review. Version
  `0.1.0` and proposed tag `v0.1.0` are chosen; recheck tags and releases before
  creating either, without overwriting any existing record.
- If automatic Zenodo archival is desired, enable the repository in Zenodo
  before publishing the GitHub Release. No DOI or archive record is asserted.
- Add the actual release date to `CITATION.cff` when publication is authorized;
  add DOI and paper citation only after valid records exist. ORCID is optional.

Keep this document marked as draft until these decisions are complete. Extract
the proposed description and update its provenance when creating the actual
release. Preparing this file does not create a tag, publish a GitHub Release,
submit to arXiv, or submit to a conference.
