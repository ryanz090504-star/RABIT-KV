# Milestone 1

## Stable scope

- K3/V2/G32/R4/META8g64 policy
- latest META8 consolidated quality result
- physical packed KV layout
- `rabit_kv2` vLLM integration
- Stage4D2.2 writer-only locked checkpoint
- 105-test regression
- controlled BF16 vs RABIT allocator capacity

## Experimental

D3.4 fused exact tail preparation:
- 13.27x tail-prep microbenchmark
- 6.04x full-attention microbenchmark
- exact output
- not yet production-integrated

D3.1-D3.3 remain research history, not final-method components.

## Hard performance finish line

After D3.4 integration:

- TPOT <= 20 ms/token: close single-request latency immediately.
- 20 < TPOT <= 30: one targeted optimization, then freeze.
- TPOT > 30: profile once, one targeted fix, then freeze.

**Maximum two performance optimization cycles after D3.4. No third redesign.**
Then freeze performance, run the unified quality suite once, and package v1.0.
