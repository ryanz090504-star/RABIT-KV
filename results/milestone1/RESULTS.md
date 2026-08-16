# RABIT-KV Milestone 1 Results

## Frozen method

**K3 / V2 / G32 / R4 / META8g64**

- K: 3-bit sequence-axis affine quantization
- V: 2-bit last-dimension/token-axis affine quantization
- Group size: 32
- Newest four K/V tokens: BF16 residual
- Primary metadata: UINT8 grouped metadata, group size 64

## Quality

| Metric | BF16 | RABIT-KV |
|---|---:|---:|
| Perplexity | 11.225819 | **11.477385** |
| Relative PPL change | - | **+2.241%** |
| Logical KV compression | 1.00x | **4.892491x** |
| RABIT logical KV | - | 6.499756 MB |

The initial GitHub snapshot did not contain the raw META8g64 quality logfile.
These values are preserved as the latest consolidated result. After performance
freeze, the unified quality suite will be run once and its raw outputs will
become the final quality evidence.

Existing long-context logs are preserved under `results/quality/pre_meta8/`
and intentionally labeled as predating the final META8g64 rerun.

## Physical KV capacity

| Format | Capacity | Relative |
|---|---:|---:|
| BF16 | 393,024 tokens | 1.000x |
| RABIT `rabit_kv2` | **2,074,592 tokens** | **5.2785x** |

This is a real allocator-capacity result on H100, not a logical byte estimate.

## Stable deployment

**Stage4D2.2 writer-only locked**

- 105 regression tests passed
- real `kv_cache_dtype=rabit_kv2`
- physical compressed cache allocation succeeded

## D3.4 experimental latency prototype

| Microbenchmark | Before | D3.4 | Speedup |
|---|---:|---:|---:|
| Exact tail prep | 0.8215 ms | 0.0619 ms | **13.27x** |
| Full attention | 1.0226 ms | 0.1693 ms | **6.04x** |

D3.4 is exact (`max_abs=0.0`) but is **not production-integrated yet**.
These are prototype microbenchmark numbers, not an end-to-end speedup claim.