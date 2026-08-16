# Method

Milestone 1 freezes RABIT-KV at **K3/V2/G32/R4/META8g64**.

- Keys: 3-bit sequence-axis affine, group size 32.
- Values: 2-bit last-dimension/token-axis affine, group size 32.
- Recent residual: newest four K/V tokens remain BF16.
- Metadata: UINT8 grouped primary metadata, group size 64, with higher-precision
  second-level parameters in the physical page codec.
- Deployment: packed payloads and metadata are stored in physical vLLM KV pages.

"2-bit target" must not be described as literal two-bit total storage. The
method includes 3-bit K, 2-bit V, metadata, and a BF16 residual.

RABIT-KV therefore reports four separate quantities:

1. quality;
2. logical KV compression;
3. physical allocator capacity;
4. deployment latency.