# Qwen2.5-7B post-hoc diagnostic: open-group padding (all 32 windows)

A post-hoc diagnostic, designed after the Qwen2.5-7B quality failure was known. It runs in the **logical evaluator**,
with the attention backend fixed to FlashAttention, so its absolute values are not interchangeable with the canonical
automatic-backend records.

Arms:

| Arm | What it changes |
|---|---|
| Frozen policy | Nothing: the K3/V2/G32/R4/META8g64 operating point |
| Padding excluded | The open key group's minimum and maximum are taken over its real tokens only, without the zero padding to 32 tokens |
| Exact keys | Keys exact; values as in the frozen policy |

| Model (32 windows, PPL) | BF16 | Frozen policy | Padding excluded | Exact keys |
|---|---:|---:|---:|---:|
| Qwen2.5-7B | 6.791 | 113.93 | 7.841 | 6.864 |
| Llama-3.1-8B | 7.514 | 7.639 | 7.619 | 7.612 |

- **Qwen2.5-7B:** excluding the padding removes 94.9% of the excess mean token NLL of the frozen policy over BF16. It leaves perplexity +15.5% above BF16.
- **Statistics:** paired-window intervals (20,000 resamples, seed 20261022) are in `REPORT.md` and `paired-window-effects.csv`.
- **Scope:** this localizes the failure to the zero-padded open key group, together with the metadata (see the metadata ablation). It does not identify a mechanism.
- **Not in the serving engine:** padding exclusion is a diagnostic arm, not implemented in serving. No capacity, latency or serving-quality result follows from it, and it does not change the RABIT-KV operating point.

**Files.**

- `REPORT.md`, `independent-audit.json`, `paired-window-effects.csv`.
- **`artifacts/`:** per-token scores, aggregates and validity gates.
- **`code/`:** the frozen registration and scoring worker.

**Links inside `REPORT.md`.** The report is kept byte-for-byte, so its links use the original run layout. In this directory:

- `gpu-attempt-008/independent-audit.json` → `independent-audit.json`
- `gpu-attempt-008/paired-window-effects.csv` → `paired-window-effects.csv`
- `gpu-attempt-008/artifacts/` → `artifacts/`
- `stage008/registration.json` → `code/registration.json`

The offline-check and billing records it links to are not published.
