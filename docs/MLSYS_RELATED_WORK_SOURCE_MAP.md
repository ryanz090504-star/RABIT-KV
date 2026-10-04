# RABIT-KV — Related Work Source Map

Status: paper-preparation document for claim safety. **No Section 8 prose is drafted here.** Compiled on
2026-10-04 from the sources linked per entry. Purpose: record what each work verifiably says, so that
RABIT-KV claims nothing as novel that it is not.

Verification levels used below:

- **ABSTRACT** — title, authors, venue and abstract read from the arXiv / official page.
- **FULL-TEXT (automated)** — specific details extracted from the arXiv HTML full text by an automated
  reader answering fixed questions. These details must be re-checked by a human against the PDF before
  any of them is stated in the paper.
- **UNKNOWN** — not verified from the source. Nothing is inferred.

No bibliography entry has been created. Venues are as stated on the source page.

---

## 0. Primary-source verification of every detail used in Section 8 (final pass, 2026-10-04)

Method: the full text of each paper (arXiv HTML) and the GitHub API record of the pull request were downloaded and
the passages were read directly; quotations below are from those texts. This supersedes the "FULL-TEXT (automated)"
level in sections 1–2 for the details listed here. It is a direct reading by the drafting assistant, NOT an
independent human check: an author should still confirm each quoted passage against the PDF before submission.
Details not listed here remain at the level stated in sections 1–2, or UNKNOWN.

| Work | Detail used in the paper | Primary-source passage | Status |
|---|---|---|---|
| KIVI | keys per channel, values per token | abstract: "the key cache should be quantized per-channel, i.e., group elements along the channel dimension and quantize them together. In contrast, the value cache should be quantized per-token" | PRIMARY SOURCE |
| KIVI | groups of 32 | Sec. 4.1: "the group size in Algorithm 1 for quantization is set as 32 across all experiments, the residual length for key and value cache is set to 128" | PRIMARY SOURCE |
| KIVI | full-precision residual, quantized in batches | Sec. 3: the residual key cache "is kept in full precision … once [it] reaches R tokens … we quantize [it] and concatenate it with the previously quantized" part; values: "keep the most recent value cache in full precision" in a queue of length R | PRIMARY SOURCE |
| KIVI | zero-padding of incomplete groups | appears ONLY in the preliminary simulated-quantization study ("For per-channel quantization, if the number of tokens is not divided evenly into groups, we add zero-padding"); the KIVI algorithm itself keeps the incomplete part in the full-precision residual | PRIMARY SOURCE — the paper therefore does NOT say that KIVI zero-pads incomplete groups, and Section 8 does not claim it |
| KIVI | implementation and memory metric | "We use the Hugging Face Transformers codebase and implement the KIVI algorithm upon it"; "report the peak memory usage and throughput" (single A100 80GB, batch size increased until out of memory) | PRIMARY SOURCE (only "peak memory" is used in the paper, in Section 2) |
| KVQuant | methods | "Per-Channel Key Quantization", "Pre-RoPE Key Quantization", "Non-Uniform KV Cache Quantization, where we derive per-layer sensitivity-weighted non-uniform datatypes", "Per-Vector Dense-and-Sparse Quantization, where we isolate outliers separately for each vector" | PRIMARY SOURCE |
| KVQuant | first token full precision; kernels | "keeping only the first token in fp16, we can attain perplexity benefits"; "custom CUDA kernels" | PRIMARY SOURCE |
| SKVQ | channel reordering, clipped dynamic quantization, sliding window | abstract (verbatim, section 1 above) | PRIMARY SOURCE |
| SKVQ | FP8 quantization parameters counted in average bits | "use FP8 to store quantization parameter so that the average bits is equal to …"; "storage overhead for quantization parameters, which is noted as average bits"; Table 4: group size 128 / 64 / 32 → 2.125 / 2.25 / 2.5 average bits | PRIMARY SOURCE |
| AsymKV | key sensitivity; unequal K / V bits by layer | abstract: "the transformer's output loss is more sensitive to the quantization of key matrices"; Sec. 4: "adheres to the quantization scheme outlined in KIVI … per-channel quantization for the key matrix and per-token quantization for the value matrix, with a group size of 32. AsymKV utilizes a combination of higher 2-bit quantization and lower 1-bit quantization" | PRIMARY SOURCE |
| TurboQuant (paper) | method; KV bit-widths | "randomly rotating input vectors … to simply apply optimal scalar quantizers per each coordinate"; "3.5 bits per channel and marginal quality degradation with 2.5 bits per channel"; evaluated on Llama-3.1-8B-Instruct needle-in-a-haystack | PRIMARY SOURCE |
| vLLM TurboQuant backend | what it is; presets; boundary layers; page sizing | PR #38479 (GitHub API; merged 2026-04-15T02:57:14Z): "online KV cache compression to vLLM's v1 attention backend using PolarQuant (WHT rotation + Lloyd-Max scalar quantization) for keys and uniform quantization for values"; presets `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc` ("3-bit MSE + NC" keys, "4-bit uniform + NC" values), `turboquant_3bit_nc`; "Boundary layer protection — first/last N layers keep FP16 KV cache via kv_cache_dtype_skip_layers"; "TQFullAttentionSpec — proper spec subclass that overrides real_page_size_bytes with TQ slot bytes" | PRIMARY SOURCE (official repository). Our engine snapshot lists the same four preset names (`vllm-kvquant/vllm/config/cache.py`) and contains `turboquant_attn.py`; that our snapshot's backend is byte-identical to the merged PR is NOT verified |
| SAW-INT4 | systems co-design; paged, fused kernel; throughput | abstract: "they often violate practical serving constraints such as paged memory layouts, regular memory access, and fused attention execution"; "token-wise INT4 quantization with block-diagonal Hadamard rotation"; "a fused rotation–quantization kernel that integrates directly into paged KV-cache layouts"; "effective KV-cache compression is fundamentally a systems co-design problem"; contributions: "jointly measures accuracy and real serving throughput" | PRIMARY SOURCE |
| SAW-INT4 | residual buffers and paged blocks | Sec. 1: KIVI and Kitty "maintain a fixed-length residual buffer of unquantized key–value pairs … PagedAttention manages cache memory in fixed-size, uniform-type blocks; accommodating two distinct precisions within the same paged pool requires either fragmented memory layouts or separate page tables" | PRIMARY SOURCE |
| SAW-INT4 | serving engine | the BF16 baseline and throughput figures are labelled SGLang; an explicit sentence "implemented in SGLang" was not located | PARTIAL — Section 8 says "paged engines" and does not name SGLang |
| Minima-KV | lifecycle and formats | "Recent and Anchor pages use FP8, while Stale pages use TQ3"; "Old non-anchor pages are encoded with a three-bit rotated scalar quantizer inspired by TurboQuant. Packed codes, scales, and required metadata are stored next to the page" | PRIMARY SOURCE |
| Minima-KV | attention | "Each format-specific kernel computes a partial maximum, exponential sum, and normalized partial output" merged through "a stable global softmax merge without constructing a cache-sized dense shadow" | PRIMARY SOURCE |
| Minima-KV | memory accounting | "Deployment accounting reports 18.3 KiB per live token … 3.497× the BF16 footprint … 1.749× the FP8 footprint"; "We therefore use 18.3 KiB/token only as an owner-reported workload aggregate and label constant-rate capacity values as analytical scenarios rather than per-context measurements" | PRIMARY SOURCE |
| Minima-KV | serving engine, K vs V precision | engine not named ("does not bind … the engine commit"); K / V asymmetry: "preserves this separation in its codec and kernel interfaces, even where a deployment selects the same nominal tier for both tensors" | UNKNOWN engine; not used in the paper |
| vLLM / PagedAttention | paged KV allocation | abstract (verbatim, section 1 above) | PRIMARY SOURCE |

### Exact overlaps with RABIT-KV

**KIVI.** Same: per-channel keys along the token axis; per-token values; group size 32; a full-precision window of
recent tokens; affine round-to-nearest codes. Different: bit allocation (KIVI 2 / 2; RABIT-KV 3 / 2); window length
(KIVI 128 by default, 32 also evaluated; RABIT-KV 4); handling of the incomplete key group (KIVI keeps it exact
inside the residual and quantizes whole batches; RABIT-KV re-quantizes a zero-padded open group at every token);
second-level quantization of the group parameters (RABIT-KV only); KIVI is implemented in Hugging Face Transformers
and reports peak memory and batch-size throughput, RABIT-KV is implemented in vLLM and reports allocator capacity.

**Minima-KV.** Same in kind: a lifecycle in which recent state is stored at higher precision and older pages in a
packed low-bit format under paged attention; per-format partial attention states merged by an online softmax; memory
reported per token against BF16 and FP8. Different: Minima-KV has three tiers with FP8 for Recent and protected
Anchor pages and a 3-bit rotated scalar quantizer for Stale pages, with a controller that can promote pages;
RABIT-KV has one packed format (3-bit keys, 2-bit values, affine) for everything older than a four-token exact
window, no promotion, and quantized metadata. Minima-KV's memory figure is an owner-reported aggregate and its
capacities are labelled analytical; RABIT-KV reads block counts from the running allocator in a matched session.
Minima-KV evaluates one model (Qwen3.6-27B); RABIT-KV two. Minima-KV reports near-parity throughput in a single
pair; RABIT-KV reports lower throughput.

**SAW-INT4.** Same in kind: low-bit KV quantization designed for and evaluated inside a paged serving engine, with
dequantization fused into the attention kernel and end-to-end serving measurements. Different: SAW-INT4 uses 4-bit
token-wise quantization for both keys and values with no full-precision buffer and reports throughput gains;
RABIT-KV uses a lower, asymmetric bit budget with per-channel keys and an exact window, and reports lower throughput.

### Final defensible novelty statement

RABIT-KV is not introduced as the first physical, packed or paged low-bit KV system, nor as the first to account
for storage beyond nominal bits, nor as the first mixed-precision cache lifecycle. Its contribution is the
end-to-end realization and measurement of one specific aggressive asymmetric target-bit operating point, jointly
exposing physical allocator capacity, asymmetric K / V payload, quantized metadata, residual and open-group state,
online aging, serving cost and validated logical quality; its empirical contribution includes the matched
physical-capacity comparison with BF16, FP8 and the tested TurboQuant configuration, and the model-dependent
quality result. No "first" claim is made anywhere in the draft.

---

## 1. Core works (required by the directive)

### KIVI

| Field | Recorded |
|---|---|
| Title | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache |
| Authors / venue / year | Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu; ICML 2024 (ABSTRACT) |
| Main KV representation | Affine quantization; "the key cache should be quantized per-channel, i.e., group elements along the channel dimension and quantize them together. In contrast, the value cache should be quantized per-token" (ABSTRACT) |
| K / V precision | 2-bit (also 4-bit evaluated) for both (ABSTRACT; FULL-TEXT automated) |
| Grouping / axis | keys per-channel, values per-token; group size G = 32 (FULL-TEXT automated) |
| Residual / recent tokens | yes: a residual of recent tokens kept in full precision, residual length R = 128; zero-padding when tokens do not divide into groups during prefill; residual quantized once it reaches R tokens (FULL-TEXT automated) |
| Metadata treatment | zero-point and scale per group; whether their bytes are counted in reported memory: UNKNOWN |
| Physical packed serving | "hardware-friendly implementation" in the Hugging Face Transformers codebase with CUDA kernels (FULL-TEXT automated); integration into a paged serving engine: not mentioned |
| Allocator-visible capacity measured | no allocator metric; reports "2.6× less peak memory (including model weight)" (ABSTRACT) |
| Serving framework | Hugging Face Transformers (FULL-TEXT automated) |
| Latency / throughput claims | "up to 4× larger batch size, bringing 2.35× ∼ 3.47× throughput on real LLM inference workload" (ABSTRACT) |
| Overlap with RABIT-KV | **Very high at the representation level**: key groups along the token axis per channel, value groups within a token, group size 32, a full-precision window of recent tokens, zero-padded incomplete groups |
| RABIT-KV must NOT claim as novel | per-channel (sequence-axis) key quantization; per-token value quantization; group size 32; a full-precision residual of recent tokens; asymmetric treatment of K and V by axis; 2-bit KV quantization |
| Source | https://arxiv.org/abs/2402.02750 ; https://arxiv.org/html/2402.02750 |

### KVQuant

| Field | Recorded |
|---|---|
| Title | KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization |
| Authors / venue / year | Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, Amir Gholami; NeurIPS 2024 (ABSTRACT) |
| Main KV representation | per-channel key quantization, pre-RoPE key quantization, non-uniform (sensitivity-weighted) datatypes, per-vector dense-and-sparse quantization, attention-sink-aware quantization (FULL-TEXT automated) |
| K / V precision | 4-, 3- and 2-bit evaluated; "under 0.1 perplexity degradation with 3-bit quantization" (ABSTRACT / FULL-TEXT automated) |
| Grouping / axis | keys per-channel; values per-token (FULL-TEXT automated) |
| Residual / recent tokens | first token kept in fp16 (attention sink); about 1% outliers stored separately in full precision as a sparse component (FULL-TEXT automated). A recent-token window: not mentioned |
| Metadata treatment | offline calibration for per-channel key scaling factors; online per-token value statistics (FULL-TEXT automated). Byte accounting of metadata: UNKNOWN |
| Physical packed serving | custom CUDA kernels for quantization / dequantization (FULL-TEXT automated); paged serving engine integration: UNKNOWN |
| Allocator-visible capacity measured | no; context-length feasibility claims ("context lengths up to 1 million on a single A100-80GB GPU", ABSTRACT) |
| Serving framework | UNKNOWN |
| Latency / throughput claims | "up to ~1.7x speedups, compared to baseline fp16 matrix-vector multiplications" (ABSTRACT) |
| Overlap with RABIT-KV | per-channel keys / per-token values; 3-bit operating points; dequantization inside custom kernels; some state kept in full precision |
| RABIT-KV must NOT claim as novel | sub-4-bit KV quantization with per-channel keys and per-token values; kernels that operate on the quantized cache; keeping selected state in full precision |
| Source | https://arxiv.org/abs/2401.18079 ; https://arxiv.org/html/2401.18079 |

### SKVQ

| Field | Recorded |
|---|---|
| Title | SKVQ: Sliding-window Key and Value Cache Quantization for Large Language Models |
| Authors / venue / year | Haojie Duanmu, Zhihang Yuan, Xiuhong Li, Jiangfei Duan, Xingcheng Zhang, Dahua Lin; arXiv 2405.06219 (May 2024, revised Nov 2024); peer-reviewed venue: UNKNOWN |
| Main KV representation | channel reordering to make quantization groups homogeneous; "clipped dynamic quantization at the group level" (ABSTRACT) |
| K / V precision | "2-bit keys and 1.5-bit values" (ABSTRACT) |
| Grouping / axis | channel groups after reordering; group sizes 128 (main), 64, 32 (FULL-TEXT automated) |
| Residual / recent tokens | "the most recent window tokens in the KV cache are preserved with high precision" (ABSTRACT); window 128; a few attention-sink tokens also kept at high precision (FULL-TEXT automated) |
| Metadata treatment | scale and zero-point stored in FP8 (E4M3); reported "average bits" INCLUDE the quantization parameters (e.g. 2.5 average bits at group size 32 with FP8 parameters vs 3 with FP16) (FULL-TEXT automated) |
| Physical packed serving | UNKNOWN (no framework named; no actual system implementation reported, per FULL-TEXT automated) |
| Allocator-visible capacity measured | no; "context lengths of up to 1M on an 80GB memory GPU for a 7b model" (ABSTRACT) |
| Serving framework | UNKNOWN |
| Latency / throughput claims | "up to 7 times faster decoding" (ABSTRACT); described as theoretical / estimated by roofline analysis (FULL-TEXT automated) |
| Overlap with RABIT-KV | sliding window of recent high-precision tokens; different K and V bit-widths; **low-precision storage of quantization parameters and metadata-inclusive bit accounting** |
| RABIT-KV must NOT claim as novel | a recent-token full-precision window; asymmetric K / V bit-widths; counting quantization metadata in the effective bits per element; storing quantization parameters at reduced precision |
| Source | https://arxiv.org/abs/2405.06219 ; https://arxiv.org/html/2405.06219 |

### TurboQuant (paper)

| Field | Recorded |
|---|---|
| Title | TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate |
| Authors / venue / year | Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni; arXiv 2504.19874 (April 2025); peer-reviewed venue: UNKNOWN |
| Main KV representation | data-oblivious vector quantization: random rotation, optimal scalar quantizer per coordinate; a two-stage variant adds a 1-bit Quantized JL transform on the residual for unbiased inner products (ABSTRACT) |
| K / V precision | KV-cache experiments at 2.5 and 3.5 bits per channel; "quality neutrality at 3.5 bits per channel" (ABSTRACT / FULL-TEXT automated) |
| Grouping / axis | per-vector; K vs V treated differently: UNKNOWN |
| Residual / recent tokens | quantization applied "even during the streaming generation process" (FULL-TEXT automated); full-precision layers: not mentioned |
| Metadata treatment | UNKNOWN |
| Physical packed serving | UNKNOWN (not reported in the paper) |
| Allocator-visible capacity measured | no |
| Serving framework | none reported in the paper |
| Latency / throughput claims | none for KV-cache serving (FULL-TEXT automated) |
| Overlap with RABIT-KV | low-bit online KV quantization; baseline method |
| RABIT-KV must NOT claim | anything about TurboQuant as a method in general: RABIT-KV was compared with ONE configuration of the vLLM integration below, and no TurboQuant quality was measured |
| Source | https://arxiv.org/abs/2504.19874 ; https://arxiv.org/html/2504.19874 |

### TurboQuant in vLLM (the system RABIT-KV was actually compared with)

| Field | Recorded |
|---|---|
| Title | vLLM pull request #38479, "[Attention Backend] TurboQuant: 2-bit KV cache compression with 4x capacity"; merged April 15, 2026 |
| Main KV representation | per the PR description: PolarQuant-style Walsh-Hadamard rotation + Lloyd-Max scalar quantization for keys, uniform quantization for values |
| K / V precision | presets include `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc` ("3-bit MSE keys + 4-bit values + NC, ~4.3x compression"), `turboquant_3bit_nc` |
| Residual / full-precision state | boundary-layer protection: "first/last N layers keep FP16 KV cache via `kv_cache_dtype_skip_layers`" |
| Physical packed serving | **yes — a dedicated attention backend inside vLLM** |
| Allocator-visible capacity measured | the PR title claims "4x capacity"; how it was measured: UNKNOWN |
| Latency / throughput claims | PR reports, for one preset on Qwen3-4B, "79-100% of baseline throughput across all scenarios" |
| Overlap with RABIT-KV | **a low-bit, asymmetric-precision (3-bit keys, 4-bit values) KV cache packed inside vLLM, with full-precision exemptions, predating this paper** |
| RABIT-KV must NOT claim as novel | being the first low-bit packed KV cache in vLLM; being the first to turn low-bit KV quantization into serving-engine capacity; asymmetric K / V precision in a serving engine |
| Open item | confirm that the `turboquant_k3v4_nc` backend in our engine snapshot is this integration, and cite the integration (not only the paper) wherever "tested TurboQuant configuration" appears |
| Source | https://github.com/vllm-project/vllm/pull/38479 |

### vLLM / PagedAttention

| Field | Recorded |
|---|---|
| Title | Efficient Memory Management for Large Language Model Serving with PagedAttention |
| Authors / venue / year | Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica; SOSP 2023 (ABSTRACT) |
| Content | paging-inspired attention; "near-zero waste in KV cache memory"; "improves the throughput of popular LLMs by 2-4× with the same level of latency" (ABSTRACT) |
| Overlap with RABIT-KV | the block allocator, block tables and scheduler that RABIT-KV reuses unchanged |
| RABIT-KV must NOT claim as novel | paged KV allocation; block-based admission; any allocator mechanism |
| Source | https://arxiv.org/abs/2309.06180 |

vLLM quantized KV cache (documentation): FP8 (`fp8_e4m3`, `fp8_e5m2`) KV cache; "can significantly reduce its
memory footprint"; "enables you to store more tokens in memory, leading to improved throughput and support for
longer context windows"; scales default to 1.0 or calibrated. This is our FP8 baseline.
Source: https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html

---

## 2. Other directly relevant work

| Work | Verified content | Level | Relevance / what RABIT-KV must not claim |
|---|---|---|---|
| **Minima-KV: Retention-Preserving KV Cache Compression with Mixed-Format Paged Attention** — Sergii Kozyrev, Davyd Maiboroda; arXiv 2608.23834, Aug 2026 | "Recent and protected Anchor pages remain in FP8, while older non-anchor pages move to packed TQ3; every live-request page remains addressable. Format-specific kernels compute partial attention states and combine them through a globally normalized online-softmax merge"; "deployment accounting reports 18.3 KiB of attention KV per live token, corresponding to 3.50x compression relative to BF16 and 1.75x relative to FP8"; quality on RULER NIAH and LongBench v2 | ABSTRACT only; everything else UNKNOWN | **Closest system-level prior work found.** Recent pages at higher precision aging into packed low-bit pages, paged attention, per-format partial attention merged by online softmax, per-live-token deployment accounting against BF16 and FP8. RABIT-KV must not claim novelty for: an aging hierarchy from high-precision recent state to packed pages; partial-attention merging across formats; per-token deployment-level accounting. Must be read in full before Section 8 is written |
| **SAW-INT4: System-AWare 4-Bit KV-Cache Quantization for Real-World LLM Serving** — Jinda Jia, Jisen Li, Zhongzhu Zhou, Jung Hwan Heo, Jue Wang, Tri Dao, et al.; arXiv 2604.19157, Apr 2026 | 4-bit K and V, per-token per-head scaling after block-diagonal Hadamard rotation; integrated with SGLang within paged KV-cache layouts; reports serving throughput gains over BF16 under concurrency (e.g. 32 concurrent requests on 2×H100) | FULL-TEXT (automated); allocator-capacity measurement UNKNOWN | A low-bit KV cache evaluated inside a real paged serving engine with system-level throughput results. RABIT-KV must not claim to be the first system-aware / serving-engine evaluation of low-bit KV quantization. Note: it reports throughput gains, RABIT-KV does not |
| **AsymKV: Enabling 1-Bit Quantization of KV Cache with Layer-Wise Asymmetric Quantization Configurations** — Qian Tao, Wenyuan Yu, Jingren Zhou; arXiv 2410.13212, Oct 2024 | "the transformer's output loss is more sensitive to the quantization of key matrices"; "an asymmetric quantization strategy … distinct configurations for key and value matrices" | ABSTRACT | Asymmetric K / V precision motivated by key sensitivity. RABIT-KV must not claim asymmetric K / V bit allocation as novel |
| **GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM** — Hao Kang, Qingru Zhang, Souvik Kundu, Geonhwa Jeong, Zaoxing Liu, Tushar Krishna, Tuo Zhao; arXiv 2403.05527 | quantization + low-rank approximation of the quantization error + sparse outlier matrix; "near-lossless 4-bit KV cache compression with up to 2.38x throughput improvement, while reducing peak-memory size up to 2.29x" | ABSTRACT | Reports measured peak memory and throughput, not only bit-widths |
| **ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification** — Yefei He, Luoming Zhang, Weijia Wu, Jing Liu, Hong Zhou, Bohan Zhuang; arXiv 2405.14256 | channel-separable token-wise quantization; saliency-based mixed precision; compression ratio, latency and GPU memory reported | ABSTRACT (partly summarized) | Mixed-precision by token saliency; reports GPU memory |
| **No Token Left Behind (MiKV)** — June Yong Yang et al.; arXiv 2402.18096 | important KV pairs at higher precision, others at low precision instead of eviction | ABSTRACT (partly summarized) | Mixed-precision retention |
| **VecInfer** — Dingyu Yao et al.; arXiv 2510.06175 | vector quantization with outlier suppression; "an optimized CUDA kernel that fuses computation with dequantization" | ABSTRACT | Fused dequantization inside the attention kernel. RABIT-KV must not claim in-kernel dequantization as novel |
| **CacheGen** — Yuhan Liu et al.; SIGCOMM'24 | KV-cache encoding into compact bitstreams for network transfer; "reduces the KV cache size by 3.5-4.3x" | ABSTRACT | Different goal (transfer / storage, not in-GPU serving capacity); optional citation |

Found by search but NOT verified (titles only; every field UNKNOWN): "High-accuracy Low-Bit KV-Cache Quantization
via Local Distribution Restoration" (arXiv 2607.16248); "MorphServe: Efficient and Workload-Aware LLM Serving via
Runtime Quantized Layer Swapping and KV Cache Resizing" (arXiv 2506.02006); "SPECTRA: Pushing the KV Cache Beyond
the 2-Bit Cliff via Spectral Transform Coding" (arXiv 2608.07915); "XQuant: Achieving Ultra-Low Bit KV Cache
Quantization with Cross-Layer Compression" (arXiv 2510.11236); PM-KVQ (arXiv 2505.18610); PolarQuant (named as the
key quantizer in the vLLM TurboQuant PR). These must be read before Section 8 is written; the search was a single
query and is not a systematic survey.

---

## 3. Novelty check

**1. Which individual RABIT-KV ingredients definitely have prior art?**

- Key quantization grouped along the token axis per channel, and value quantization grouped within a token
  (KIVI, KVQuant). RABIT-KV's "sequence-axis grouping per (head, channel)" is KIVI's "per-channel" key quantization.
- Group size 32 (KIVI).
- A full-precision window of recent tokens (KIVI residual, R = 128; SKVQ sliding window).
- Zero-padding of incomplete groups (KIVI, per automated full-text read).
- Asymmetric K / V bit-widths (AsymKV; SKVQ 2-bit / 1.5-bit; the vLLM TurboQuant presets with 3-bit keys and 4-bit values).
- Reduced-precision storage of quantization parameters and metadata-inclusive bit accounting (SKVQ: FP8 parameters, "average bits").
- Dequantization fused into attention kernels (KVQuant, VecInfer).
- A packed low-bit KV cache inside a paged serving engine (vLLM TurboQuant backend; SAW-INT4 in SGLang; Minima-KV).
- Higher-precision recent state aging into packed low-bit pages, with partial attention merged across formats (Minima-KV, abstract).
- Reporting memory at system level rather than as a bit-width (KIVI and GEAR peak memory; Minima-KV per-live-token deployment accounting).

**2. Which combinations have close prior art?**

- **KIVI** is the closest representation: per-channel keys + per-token values + group size 32 + full-precision
  residual + 2-bit. RABIT-KV differs in the bit allocation (3 / 2), the residual length (4 vs 128), re-quantizing an
  open group every token instead of quantizing in residual-sized batches, and the second-level metadata quantization.
  These are differences of degree and engineering, not of kind.
- **SKVQ** combines a recent-token window with metadata-aware bit accounting and reduced-precision parameters.
- **Minima-KV** is the closest system: an aging hierarchy of page formats under paged attention with per-format
  partial attention, and deployment-level per-token accounting against BF16 and FP8.
- **vLLM TurboQuant** is a merged, packed, asymmetric-precision low-bit KV backend in the same engine.

**3. What remains the safest defensible RABIT-KV contribution?**

Not a "first". The defensible content is:

- one specific, fully specified operating point (including 8-bit second-level quantization of the group
  parameters in 64-value groups, and a per-token open-group / closed-page lifecycle) realized as packed pages in vLLM;
- **measured** allocator capacity in one matched session against BF16, FP8 and the engine's own TurboQuant
  configuration, with the latency, throughput and long-prefill costs reported rather than omitted;
- a validation methodology (independent reference, same-device bit-exact conformance, registered runs) and what it
  exposed: an evaluator mismatch in earlier results;
- a negative cross-model result at a fixed operating point.

The paper should be positioned as a careful system-and-measurement study of one design point, not as the origin of
any mechanism in the list under question 1.

**4. Are any current Introduction / Design sentences too strong after reviewing the literature?**

No sentence claims a first or a novel mechanism, so there is no clear factual conflict and nothing was rewritten.
The following should be weakened or supported by a citation when Section 8 is written:

- Abstract: "Low-bit KV-cache quantization is **usually** described by its logical precision". SKVQ reports
  metadata-inclusive average bits, KIVI and GEAR report peak memory, Minima-KV reports per-live-token deployment
  accounting. "often" is safer than "usually".
- Section 2: "Quantized KV caches … are **usually** described by the bit-width of their codes" — same issue.
- Introduction, paragraph 2: "Our focus is the systems problem of accounting for these effects jointly and measuring
  the resulting allocator-visible capacity and runtime cost" — acceptable as a statement of focus, but Minima-KV and
  SAW-INT4 address closely related systems problems; the paragraph must not be read as "unaddressed".
- Design: the K3 and V2 paragraphs should state that these axes follow KIVI's per-channel / per-token finding, with a
  citation; at present the design reads as if the axes were our choice.
- Contribution bullet 2 ("measured as allocator-observed KV capacity rather than by logical compression
  accounting") is a statement about our method of measurement and is safe, provided Section 8 says that
  system-level memory reporting exists elsewhere.
- Everywhere: "[TurboQuant]" must cite the vLLM integration as well as the paper.

**5. Is "target-bit-aware" genuinely descriptive rather than a claim of first use?**

As used, it is descriptive: it says the design starts from a target bit budget and accounts for what that budget
costs physically. It is not presented as a named technique or a first. Whether the phrase is used elsewhere in the
literature is UNKNOWN (not searched). It should stay an adjective and should not be capitalized, defined as a new
term, or listed as a contribution in itself.

**6. Which works are essential citations in the main Related Work section?**

1. KIVI — closest representation.
2. KVQuant — per-channel / per-token sub-4-bit quantization with custom kernels.
3. SKVQ — recent-token window and metadata-inclusive accounting.
4. AsymKV — asymmetric K / V precision.
5. TurboQuant — the paper and the vLLM integration that is our baseline.
6. vLLM / PagedAttention — the allocator and engine.
7. Minima-KV — closest system-level work (must be read in full first).
8. SAW-INT4 — low-bit KV quantization evaluated inside a paged serving engine with throughput results.

GEAR, ZipCache, MiKV, VecInfer and CacheGen are secondary citations if space permits.

---

## 4. TurboQuant snapshot provenance audit (read-only, 2026-10-04)

Question: what is the relationship between the TurboQuant backend in our frozen engine snapshot (the one used in the
matched four-method experiment) and upstream vLLM pull request #38479?

Method: git blob hashes of the committed snapshot files (`git ls-tree`, identical at the Exp13 evidence commit
`42c2799` and at HEAD) were compared with (a) the per-file blob hashes of PR #38479 from the GitHub API
(PR head `ac46a983…`, merge commit `f4b42df0…`, merged 2026-04-15) and (b) the blob hashes of upstream vLLM at commit
`f329ce405b12623fb8b1cf1830f12e5a712523be` (2026-07-04), which is the base commit of our snapshot. No serving run.

| File (under `vllm/` unless noted) | Local blob | Upstream @ f329ce4 | PR #38479 head | Result |
|---|---|---|---|---|
| `v1/attention/backends/turboquant_attn.py` | `af4ab007…` | `af4ab007…` | `279fcb04…` | identical to upstream base; differs from PR |
| `v1/attention/ops/triton_turboquant_decode.py` | `3adaf261…` | `3adaf261…` | `8b276e31…` | identical to upstream base; differs from PR |
| `v1/attention/ops/triton_turboquant_store.py` | `3ad2d414…` | `3ad2d414…` | `3da3347d…` | identical to upstream base; differs from PR |
| `model_executor/layers/quantization/turboquant/config.py` | `84e3940d…` | `84e3940d…` | `289bed12…` | identical to upstream base; differs from PR |
| `model_executor/layers/quantization/turboquant/__init__.py` | `f9f4384e…` | `f9f4384e…` | `10ee032c…` | identical to upstream base; differs from PR |
| `model_executor/layers/quantization/turboquant/centroids.py` | `49026574…` | `49026574…` | `49026574…` | identical to both |
| `model_executor/layers/quantization/turboquant/quantizer.py` | absent | absent | `aea63c52…` | removed upstream after the PR |
| `tests/quantization/test_turboquant.py` | `ccdc6907…` | `ccdc6907…` | `78c137e6…` | identical to upstream base; differs from PR |
| shared: `config/attention.py`, `engine/arg_utils.py`, `model_executor/layers/attention/attention.py`, `platforms/cuda.py`, `v1/attention/backends/registry.py`, `v1/core/single_type_kv_cache_manager.py`, `v1/worker/utils.py` | — | — | — | each identical to upstream base |
| shared: `config/cache.py` | `dc767ed0…` | `70a58004…` | — | differs: +11 lines, all adding our `kvquant_k3` / `rabit_kv2` cache types |
| shared: `utils/torch_utils.py` | — | — | — | differs: +5 / −1 lines, all for our cache types |
| shared: `v1/kv_cache_interface.py` | `71930a69…` | `323b1e76…` | — | differs: +298 / −1 lines, all for our cache types (the one changed line inserts a branch for our 3-bit mode before the existing 4-bit branch) |

No changed line in the three differing shared files mentions TurboQuant.

Tested preset (`turboquant_k3v4_nc`, from the snapshot's `config.py`): 3-bit keys, 4-bit values, norm correction
enabled. Boundary layers: the backend's default keeps the first two and the last two layers of a dense model in full
precision, which gives the 28 quantized + 4 BF16 layers recorded by the experiment.

**Classification: A, with a precise referent.** Every TurboQuant-specific file in our snapshot is byte-identical
(git blob hash) to **upstream vLLM at the snapshot's base commit `f329ce4`**. It is NOT byte-identical to PR #38479
as merged: upstream changed the backend between the PR (April 2026) and that commit (July 2026), and our snapshot
carries the later upstream revision. We made no modification to the TurboQuant backend; our changes to shared files
only add our own cache types.

Paper wording permitted by this audit: "Our engine snapshot is based on a later upstream revision and contains that
revision's TurboQuant backend unmodified." Not permitted: "the backend of PR #38479" without "a later revision of".
