# Scientific source navigation

Frozen worker/protocol files retain their bytes and original path assumptions. Browse them here; run reproductions through the [public adapter](public/reproduce.py) using a separate pinned source checkout with the original paths. The [reproduction guide](../docs/REPRODUCIBILITY.md) describes prerequisites and external offline fixtures.

- [Canonical logical evaluator](experiments/canonical_rabit_quality.py)
- [Canonical PPL runner](experiments/run_canonical_ppl.py)
- [Canonical long-context runner](experiments/run_canonical_longctx.py)
- [Matched serving runner](experiments/run_experiment13_external_baseline.py)
- [Legacy quality workers](legacy/quality/)
- [Legacy performance workers](legacy/performance/)
- [Complete old/current path manifest](../docs/PATH_MIGRATION.json)
