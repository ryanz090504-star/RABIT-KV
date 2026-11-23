# RABIT-KV

*Ryan Zhang*

[Paper](paper/v2/main.pdf) · [Software DOI](https://doi.org/10.5281/zenodo.23352595) · [Code: rabit_kv2.py](vllm-kvquant/vllm/v1/attention/ops/rabit_kv2.py) · [Method and cache lifecycle](docs/METHOD.md) · [Latest research report](docs/research/KV_CACHE_UPDATE_QWEN_LLAMA.md)

**On Llama-3.1-8B-Instruct, RABIT-KV held 5.28× as many cached tokens as BF16 in the same paged-cache budget.** In the primary serving quality test, perplexity rose by 1.42% with one-chunk prefill and 1.96% with two-chunk prefill.

A language model keeps keys and values for earlier tokens so it can attend to them when predicting the next one. I built RABIT-KV to answer a practical question: **how much context can a serving engine keep in the same memory, and what does it give up to do so?**

## The design: 3-bit keys, 2-bit values

RABIT-KV packs keys into 3 bits and values into 2 bits, groups keys in blocks of 32 tokens, and keeps the newest four tokens in BF16. I measured the token slots the running engine allocated on one H100, including the metadata stored in each cache page. The capacity figures describe that paged pool; per-sequence state outside it is counted separately in the [capacity breakdown](docs/RESULTS.md#capacity-accounting).

| What I measured on Llama | What I found |
|---|---|
| Cache capacity | 2,074,592 token slots versus 393,024 for BF16: **5.28× BF16**, **2.64× native FP8**, and **1.72× the tested TurboQuant K3/V4 configuration**. |
| Retrieval quality | In a separate evaluation, RABIT-KV and BF16 both scored **57/57** on needle-in-a-haystack and **100%** on 200 passage-retrieval examples. |
| Serving speed | In the matched single-request benchmark, time to first token rose from **59.99 to 144.65 ms** (2.41×); time per output token rose from **24.19 to 30.10 ms** (+24.4%). Throughput was also lower across the tested concurrency settings. |

## What changed when I tested Qwen?

On Qwen2.5-7B, the same design produced a perplexity of 120.86 against 7.78 for BF16. Lower is better. I traced most of the gap to an unfinished group: zeros in unused slots were affecting the range assigned to real keys.

| Qwen serving configuration · one-chunk prefill, same 32 development windows used for debugging | Perplexity ↓ |
|---|---:|
| BF16 reference | 7.78 |
| Original RABIT-KV | 120.86 |
| Calculate the unfinished group’s range from real tokens only | 8.63 |
| Also store the key minima in BF16, keeping the keys at 3 bits | 7.88 |

On **32 fresh Qwen windows**, storing minima in BF16 lowered perplexity by **4.28% with one chunk** and **4.20% with two chunks**, relative to the padding fix alone. The test did not establish that both Qwen settings were within 3% of BF16. On Llama, the two-chunk test could not rule out a perplexity increase above 0.5%, so the overall confirmation did not pass.

That leaves a concrete design choice. Excluding unused slots fixes how a group’s range is calculated. Spending extra bits on its minimum helped Qwen, but on Llama it reduced capacity from **5.28× to 5.08× BF16** without a demonstrated quality benefit. Both changes were tested as research patches; the released engine retains the original configuration.

[Full results and limitations](docs/RESULTS.md) · [November research report](docs/research/KV_CACHE_UPDATE_QWEN_LLAMA.md) · [Tables and statistical reproduction](results/experiments/kv_cache_2026_11/) · [Supplement](paper/v2/appendix.pdf)

Original RABIT-KV code by Ryan Zhang is licensed under [Apache-2.0](LICENSE). See [licensing scope](docs/LICENSING.md) and [citation metadata](CITATION.cff).
