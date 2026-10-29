# Qwen2.5-7B post-hoc diagnostic: which keys matter (8 windows)

A post-hoc diagnostic in the logical evaluator, with FlashAttention fixed and padding exclusion as the baseline arm.
Each arm makes one region of the keys exact.

- **Windows:** eight previously observed windows (1, 5, …, 29); they are not a holdout.
- **Design:** frozen before the run in `code/registration.json`.

| Arm (PPL, 8 windows) | Qwen2.5-7B | vs padding excluded | Llama-3.1-8B |
|---|---:|---:|---:|
| BF16 | 8.003 | | 7.399 |
| Padding excluded | 8.956 | | 7.479 |
| Exact keys | 8.100 | -9.56% [-14.12, -4.81] | 7.494 |
| Exact closed pages | 8.087 | -9.70% [-14.62, -4.61] | 7.507 |
| Exact open group | 8.971 | +0.17% [-0.58, +1.02] | 7.498 |

- **What it shows:** on Qwen2.5-7B, once padding is excluded, making only the **closed-page** keys exact removes the whole mean-NLL gap to exact keys (fraction 1.016), while making only the open group exact removes none of it (-0.017). Effects are not additive.
- **Limits:** these are diagnostic controls, not deployable policies.
- **Replication:** see `../validation_split_replication/`, which repeats the design on another split.
- **Execution:** the run executed on RunPod. Two earlier attempts at the same frozen design on another provider never started (the workspace was paused) and produced no scores.

**Files.**

- `REPORT.md`, `audit.json` (in-run analysis), `independent-check.json` (separately written check).
- **`artifacts/`:** per-token scores, aggregates and validity gates.
- **`code/`:** the registration, the scoring worker and the independent check.
