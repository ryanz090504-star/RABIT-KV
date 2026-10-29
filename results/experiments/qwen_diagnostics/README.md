# Qwen2.5-7B post-hoc diagnostics

All three are logical-evaluator diagnostics of the Qwen2.5-7B quality failure, run after that failure was known:

- [Open-group padding, all 32 windows](padding_exclusion_full32/)
- [Which keys matter, 8 windows](key_region_8window/)
- [Replication on the validation split](validation_split_replication/)

The diagnostic arms (padding exclusion, exact key regions) are **not implemented in the serving engine**. Their quality
must not be combined with the 5.28× capacity of the measured operating point: no system with both properties was
built or measured.
