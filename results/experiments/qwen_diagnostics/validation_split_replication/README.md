# Replication on the validation split (H001R)

The two post-hoc findings were repeated on 32 windows of the **WikiText-2 validation split**. The arms, analysis and
decision criteria were fixed before the run (`PROTOCOL.md`, `freeze-record.json`).

| Pre-specified quantity (Qwen2.5-7B) | Value | 97.5% interval |
|---|---:|---:|
| Padding excluded vs frozen policy, PPL | -91.4% | [-92.9, -89.7] |
| Fraction of frozen excess NLL removed by padding exclusion | 0.941 | [0.924, 0.955] |
| Exact keys vs padding excluded, PPL | -13.2% | [-17.0, -9.6] |
| Exact closed pages vs padding excluded, PPL | -12.8% | [-16.7, -9.1] |
| Fraction of the exact-key gap removed by exact closed pages | 0.963 | [0.901, 1.001] |
| Fraction of the exact-key gap removed by the exact open group | 0.031 | [-0.021, 0.093] |

- **Outcome:** both pre-specified decisions met their criteria: H1 (padding exclusion) and H2 (closed-page localization).
- **Agreement:** `audit.json` is the in-run analysis; `reanalysis.json` recomputes it from the per-token scores, and the two agree to within 1e-12.
- **Scope:**
  - This is a replication on another split of the same corpus, in the logical evaluator. It is not a held-out confirmation, and it does not make any change part of RABIT-KV.
  - Padding exclusion and exact keys are not implemented in the serving engine.

**Files.**

- `RUN-RECORD-H001R.md`, `PROTOCOL.md`, `H001R-CHANGES.md`, `freeze-record.json`, `audit.json`, `reanalysis.json`.
- **`artifacts/`:** per-token scores, aggregates and validity gates.
- **`code/`:** the scoring worker, the pre-registered analysis and the recomputation.
