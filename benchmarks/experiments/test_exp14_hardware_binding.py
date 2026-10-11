"""Offline tests for the Exp14 scheduler-only hardware binding amendment (no GPU, no Modal call)."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp14_hardware_binding as hb  # noqa: E402
import run_experiment14_second_model as r14  # noqa: E402

ROOT = HERE.parents[1]


def _show(commit, path):
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def test_deployment_differs_by_exactly_selector_and_guard():
    old = _show(hb.ACCEPTED_PROBE_COMMIT, hb.DEPLOYMENT_PATH)
    new = (ROOT / hb.DEPLOYMENT_PATH).read_text(encoding="utf-8")
    d = hb.deployment_diff(old, new)
    assert d == {"old_selector": "H100", "new_selector": "H100!:1", "guard_block_found": True,
                 "everything_else_identical": True, "passed": True}
    # a third change anywhere is rejected
    assert not hb.deployment_diff(old, new.replace("LEG_TIMEOUT_S = 900", "LEG_TIMEOUT_S = 901"))["passed"]
    # the selector alone (without the guard) is not the recorded amendment
    assert not hb.deployment_diff(old, old.replace('gpu="H100",', 'gpu="H100!:1",'))["passed"]
    # a different selector is rejected
    assert not hb.deployment_diff(old, new.replace('gpu="H100!:1"', 'gpu="H100:1"'))["passed"]


def test_runner_changes_only_the_prerequisite_binding():
    old = _show(hb.ACCEPTED_PROBE_COMMIT, hb.RUNNER_PATH)
    new = (ROOT / hb.RUNNER_PATH).read_text(encoding="utf-8")
    d = hb.runner_diff(old, new)
    assert d["passed"] and set(d["changed_or_added_names"]) <= hb.RUNNER_ALLOWED_NAMES
    assert not hb.runner_diff(old, new.replace("WARMUPS_PER_LEG, REPS_PER_LEG, SAMPLES_PER_CONDITION = 5, 30, 60",
                                               "WARMUPS_PER_LEG, REPS_PER_LEG, SAMPLES_PER_CONDITION = 5, 31, 62"))["passed"]


def test_guard_predicate_is_h100_80gb_only():
    h100 = {"name": "NVIDIA H100 80GB HBM3", "memory.total": "81559"}
    assert hb.hardware_ok([h100])
    assert not hb.hardware_ok([{"name": "NVIDIA H200", "memory.total": "143771"}])
    assert not hb.hardware_ok([{"name": "NVIDIA H100 NVL", "memory.total": "95830"}])  # 94 GB variant
    assert not hb.hardware_ok([{"name": "NVIDIA B200", "memory.total": "183359"}])
    assert not hb.hardware_ok([h100, h100])  # exactly one GPU
    assert not hb.hardware_ok([])
    src = (ROOT / hb.DEPLOYMENT_PATH).read_text(encoding="utf-8")
    # the guard runs before the RABIT gate, the shape gate, the model download and every leg
    i = src.index('_emit("EXP14_HARDWARE_CHECK"')
    assert i < src.index("EXP14_GATE_START") < src.index("EXP14_SHAPE_GATE_START") < src.index("snapshot_download(MODEL")


def test_amendment_record_matches_files_and_probe():
    a = json.loads(r14.HW_BINDING_AMENDMENT.read_text(encoding="utf-8"))
    rec = json.loads((ROOT / "results/mlsys2027/second_model/feasibility_probe/probe_record.json").read_text(encoding="utf-8"))
    assert a["kind"] == "scheduler_only_hardware_binding_amendment" and a["scientific_protocol_changed"] is False
    assert a["protocol_sha256"] == rec["protocol_sha256"] == r14.e1.sha256(r14.PROTOCOL)  # protocol unchanged
    for name, v in a["bound_file_changes"].items():
        assert v["old_sha256"] == rec["harness_sha256"][name]
        assert v["new_sha256"] == r14.e1.sha256(HERE / name)
    cur = {p.name: r14.e1.sha256(p) for p in r14.HARNESS_FILES}
    diff = {n for n in cur if cur[n] != rec["harness_sha256"].get(n)}
    assert diff == set(a["bound_file_changes"])  # every OTHER bound file still matches the probe binding


def test_prerequisite_accepts_amendment_and_rejects_anything_else():
    rec = json.loads((ROOT / "results/mlsys2027/second_model/feasibility_probe/probe_record.json").read_text(encoding="utf-8"))
    prov = {k: rec[k] for k in r14.PROBE_BINDING_KEYS}
    prov["harness_sha256"] = {p.name: r14.e1.sha256(p) for p in r14.HARNESS_FILES}
    r14.check_probe_prerequisite(rec, prov)  # accepted: the only harness difference is the recorded amendment
    for mutate in (lambda p: p["harness_sha256"].__setitem__("exp14_shape_gate.py", "0" * 64),
                   lambda p: p["harness_sha256"].__setitem__("exp14_deployment_modal.py", "1" * 64),
                   lambda p: p.__setitem__("protocol_sha256", "0" * 64),
                   lambda p: p.__setitem__("model_revision", "0" * 40)):
        bad = copy.deepcopy(prov)
        mutate(bad)
        try:
            r14.check_probe_prerequisite(rec, bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError("prerequisite accepted a non-amendment difference")


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
