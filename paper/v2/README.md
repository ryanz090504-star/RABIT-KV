# Paper, October 2026

**RABIT-KV: A Physically Packed, Target-Bit-Aware KV Cache for LLM Serving**, Ryan Zhang. Preprint.

- `main.pdf`, `appendix.pdf`: the paper and its supplementary appendix.
- `main.tex`, `appendix.tex`, `references.bib`, `mlsys2025.sty`, `mlsys2025.bst`: their source. It builds with Tectonic 0.15.0 (`tectonic -X compile main.tex`, then `appendix.tex`).

This version extends the earlier paper in `../` (kept unchanged) with:
- serving-engine perplexity;
- the Qwen2.5-7B post-hoc diagnostics and their validation-split replication;
- the metadata-precision ablation;
- the full storage and capacity accounting;
- dispersion of the latency measurements.

The evidence behind each number is in `../../results/`; see `../../docs/REPRODUCIBILITY.md`.
