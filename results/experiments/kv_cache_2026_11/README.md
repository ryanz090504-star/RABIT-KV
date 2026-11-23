# RABIT-KV research update (November 2026): evidence

This folder holds the evidence for [RABIT-KV research update: KV-cache quantization on Qwen and Llama](../../../docs/research/KV_CACHE_UPDATE_QWEN_LLAMA.md).

**Configurations:**

| | Configuration | Status |
|---|---|---|
| **A** | released operating point (zero-padded open key group) | the released system |
| **B** | padding fix | research serving patch, not in this repository |
| **C** | padding fix + BF16 K minima in every layer | research serving patch, not in this repository |

**Studies:**

| Study | Status |
|---|---|
| `h01` | exploratory |
| `c3_a2` | exploratory |
| `c4` | confirmatory |
| `h1` | exploratory mechanism diagnostic |

## Contents

| Path | What it is |
|---|---|
| `tables/serving_window_nll.csv` | One row per study, model, window and cell (896 rows). See the column table below |
| `tables/h1_layer_metrics.csv` | H1, one row per model, window and layer (1,920 rows): attention-score RMS error with META8 and with BF16 minima, their difference `dE`, the relative reduction, and the share and quantiles of K minima moved by META8 by more than half a K3 step |
| `scripts/recompute.py` | Recomputes every number of the report from the two tables and compares it with `frozen/`. Exit status 0 = all checks pass (257 checks). The serving part needs only the standard library; the H1 part needs numpy |
| `frozen/` | The analysis outputs as produced at the time, byte for byte except line endings (the two H1 files were written with CRLF on Windows and are published LF-normalized): `h01-summary.json`, `c3-a2-summary.json`, `c4-summary-qwen.json`, `c4-summary-llama.json`, `h1-analysis.json`, and `h1-layer-shares.json` (post hoc: layer shares with explicit denominators, interval count) |
| `frozen_code/` | The frozen analysis code (`h01_analyze.py`, `c3_analyze.py`, `c4_analyze.py`, `analyze_h1.py`), plus the H1 metric (`h1_metric.py`) and capture (`h1_capture.py`), byte for byte. They read the unpublished raw score files and capture outputs; `scripts/recompute.py` re-implements their statistics on the published tables |
| `windows/` | C4's window specifications (`spec-r01-qwen.json`, `spec-r01l-llama.json`: per-window token and target hashes) and its window builder (`c4_windows.py`). The windows can be rebuilt from the public WikiText-2 validation file and each model's tokenizer |
| `provenance.json` | Per cell, the number of raw score files and a digest of their SHA-256 values. Also the SHA-256 of every published file at build time |
| `MANIFEST.sha256` | SHA-256 of every file in this folder, over the repository's LF bytes (a Windows checkout with `core.autocrlf` shows CRLF and different hashes) |

**Columns of `serving_window_nll.csv`:**

| Column | Meaning |
|---|---|
| `study`, `model`, `window_id` | which study, model and window |
| `token_sha256` | SHA-256 of the window's 1,152 token ids, as int64 little-endian |
| `order_index` | the window's position in the frozen analysis order; bootstrap resampling follows this order |
| `cell` | the cell as named in the run |
| `prefill` | `one_chunk` or `two_chunk` |
| `configuration` | `bf16`, `original_padding_zeros` (A), `padding_fix` (B) or `padding_fix+kmin_bf16` (C) |
| `mean_nll` | the window's mean FP32 NLL over its 128 scored tokens |

C4 ran configuration B in two code trees: `base_on_*` and `off_*`. They are bit-identical, and `off_*` is the registered comparator.

## Reproduce

```
cd results/experiments/kv_cache_2026_11
python scripts/recompute.py --out recomputed.json
```

## Not included

The following are not published:
- the raw per-token score files and Pod archives (`provenance.json` holds their digests);
- Pod logs, spend records and model tensors;
- the serving-engine patches for B and C.

The statistics can therefore be re-derived here, but the GPU runs cannot be repeated from this repository alone.
