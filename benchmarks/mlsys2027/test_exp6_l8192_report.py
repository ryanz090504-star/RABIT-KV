"""Offline tests for the read-only L8192 three-trial report (no GPU). Run directly or with pytest."""

from __future__ import annotations

import copy
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp6_l8192_report as rep  # noqa: E402
import run_experiment6_concurrency as r6  # noqa: E402

D = r6.out_dir(8192)


def _inputs():
    combined = json.loads((D / "combined_summary.json").read_text(encoding="utf-8"))
    accepted = {t: json.loads((D / f"trial_{t}" / "ACCEPTED_RUN.json").read_text(encoding="utf-8")) for t in rep.TRIALS}
    return combined, accepted


def test_spread_and_median_of_trial_ratios():
    s = rep.spread({"1": 1.0, "2": 2.0, "3": 4.0})
    assert s["median"] == 2.0 and s["min"] == 1.0 and s["max"] == 4.0 and s["relative_spread"] == 1.5
    combined, accepted = _inputs()
    out = rep.build_report(combined, accepted)
    for c in ("1", "16", "64"):
        b = combined["cross_trial"][f"bfloat16|{c}"]["requests_per_s"]["per_trial"]
        r = combined["cross_trial"][f"rabit_kv2|{c}"]["requests_per_s"]["per_trial"]
        want = statistics.median(r[t] / b[t] for t in rep.TRIALS)
        assert abs(out["rabit_over_bf16"][c]["requests_per_s"]["median"] - want) < 1e-15


def test_uses_only_trial_level_values_and_requires_three_trials():
    combined, accepted = _inputs()
    out = rep.build_report(combined, accepted)
    assert "never concatenated" in out["sample_pooling"] and "NOT a confidence interval" in out["spread_note"]
    m = out["metrics"]["rabit_kv2|64"]["requests_per_s"]
    assert m["median"] == combined["cross_trial"]["rabit_kv2|64"]["requests_per_s"]["cross_trial_median"]
    bad = copy.deepcopy(combined)
    bad["per_trial_validity"].pop("3")
    for args in ((bad, accepted), ({**combined, "all_trials_integrity_passed": False}, accepted),
                 (combined, {t: accepted[t] for t in ("1", "2")})):
        try:
            rep.build_report(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete / invalid input accepted")


def test_capacity_and_c64_reporting():
    combined, accepted = _inputs()
    out = rep.build_report(combined, accepted)
    bf, rb = out["capacity"]["bfloat16"], out["capacity"]["rabit_kv2"]
    assert bf["allocator_capacity_kv_tokens"] == 393024 and bf["allocator_derived_full_length_sequence_ceiling"] == 47
    assert rb["allocator_capacity_kv_tokens"] == 2074592 and rb["allocator_derived_full_length_sequence_ceiling"] == 252
    assert bf["realized_max_inflight_per_trial"]["64"] == {"1": 47, "2": 47, "3": 47}
    assert rb["realized_max_inflight_per_trial"]["64"] == {"1": 64, "2": 64, "3": 64}
    assert set(bf["outcome_class_per_trial"]["64"].values()) == {"target_concurrency_not_reached"}
    assert "NOT a matched-realized-concurrency" in out["c64_comparison_note"]
    assert "no cause" in out["stability"]["attribution"]


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
