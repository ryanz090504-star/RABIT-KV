"""
MLSys 2027 Experiment 13 -- TurboQuant correctness GATE with frozen item-level accounting (runs INSIDE the Modal
container, cwd /root/vllm-kvquant, BEFORE any measured leg). Post-failure HARNESS amendment after Attempt 1.

Frozen from the authoritative non-evidence collection probe (results/mlsys2027/external_baseline/tq_collect_probe/,
commit 0b4d01b; unchanged Exp4-verbatim image, --confcutdir=/root/vllm-kvquant/tests/quantization):
  * 123 collected items of tests/quantization/test_turboquant.py (45 test definitions) and the full SHA-256 of the
    sorted node-ID list;
  * S_SCIPY = 2 optional SciPy-reference items (the ONLY items allowed to skip, and only if scipy is absent);
  * 15 GPU-only items (must execute and pass; GPGPU_AVAILABLE must be true).
Steps (stop at the first failure; exit 0 only if every check passes):
  1. environment: scipy importability, GPGPU_AVAILABLE;
  2. pytest --collect-only: count == 123 AND sha256(sorted node IDs) == frozen hash, else STOP before executing tests;
  3. pytest run of the UNCHANGED tests with a JUnit XML report: per-item outcomes and skip reasons; summary counts.
Stdlib only; never modifies a test, the source or the environment. Emits EXP13_TQ_GATE_SUMMARY=<json>.
Correctness claim boundary: the upstream suite has NO end-to-end store/decode item for turboquant_k3v4_nc.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

TARGET = "tests/quantization/test_turboquant.py"
CONFCUTDIR = "--confcutdir=/root/vllm-kvquant/tests/quantization"
BASE = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", CONFCUTDIR]
EXPECTED_COUNT = 123
EXPECTED_NODE_IDS_SHA256 = "de2c00c3ded35dbdc3686e2480b81b6ab32f2cc2410aa969890de8663b41c86b"
EXPECTED_NODE_IDS = [
    "tests/quantization/test_turboquant.py::TestCentroids::test_centroids_cached",
    "tests/quantization/test_turboquant.py::TestCentroids::test_centroids_different_dims_not_identical",
    "tests/quantization/test_turboquant.py::TestCentroids::test_centroids_shape[2-4]",
    "tests/quantization/test_turboquant.py::TestCentroids::test_centroids_shape[3-8]",
    "tests/quantization/test_turboquant.py::TestCentroids::test_centroids_shape[4-16]",
    "tests/quantization/test_turboquant.py::TestCentroids::test_centroids_sorted[2]",
    "tests/quantization/test_turboquant.py::TestCentroids::test_centroids_sorted[3]",
    "tests/quantization/test_turboquant.py::TestCentroids::test_centroids_sorted[4]",
    "tests/quantization/test_turboquant.py::TestCentroids::test_centroids_symmetric_around_zero[2]",
    "tests/quantization/test_turboquant.py::TestCentroids::test_centroids_symmetric_around_zero[3]",
    "tests/quantization/test_turboquant.py::TestCentroids::test_centroids_symmetric_around_zero[4]",
    "tests/quantization/test_turboquant.py::TestCentroids::test_centroids_within_4sigma[2]",
    "tests/quantization/test_turboquant.py::TestCentroids::test_centroids_within_4sigma[3]",
    "tests/quantization/test_turboquant.py::TestCentroids::test_centroids_within_4sigma[4]",
    "tests/quantization/test_turboquant.py::TestHadamardRotation::test_hadamard_orthonormal[128]",
    "tests/quantization/test_turboquant.py::TestHadamardRotation::test_hadamard_orthonormal[256]",
    "tests/quantization/test_turboquant.py::TestHadamardRotation::test_hadamard_orthonormal[64]",
    "tests/quantization/test_turboquant.py::TestHadamardRotation::test_hadamard_symmetric[128]",
    "tests/quantization/test_turboquant.py::TestHadamardRotation::test_hadamard_symmetric[256]",
    "tests/quantization/test_turboquant.py::TestHadamardRotation::test_hadamard_symmetric[64]",
    "tests/quantization/test_turboquant.py::TestHybridAttentionIndices::test_attn_type_list_minimax",
    "tests/quantization/test_turboquant.py::TestHybridAttentionIndices::test_layer_types_full_attention",
    "tests/quantization/test_turboquant.py::TestHybridAttentionIndices::test_layers_block_type_jamba",
    "tests/quantization/test_turboquant.py::TestHybridAttentionIndices::test_no_hybrid_hints_returns_empty",
    "tests/quantization/test_turboquant.py::TestLloydMax::test_boundaries_are_midpoints[2]",
    "tests/quantization/test_turboquant.py::TestLloydMax::test_boundaries_are_midpoints[3]",
    "tests/quantization/test_turboquant.py::TestLloydMax::test_boundaries_are_midpoints[4]",
    "tests/quantization/test_turboquant.py::TestLloydMax::test_boundaries_between_centroids[2]",
    "tests/quantization/test_turboquant.py::TestLloydMax::test_boundaries_between_centroids[3]",
    "tests/quantization/test_turboquant.py::TestLloydMax::test_boundaries_between_centroids[4]",
    "tests/quantization/test_turboquant.py::TestLloydMax::test_boundaries_sorted[2]",
    "tests/quantization/test_turboquant.py::TestLloydMax::test_boundaries_sorted[3]",
    "tests/quantization/test_turboquant.py::TestLloydMax::test_boundaries_sorted[4]",
    "tests/quantization/test_turboquant.py::TestLloydMax::test_centroids_match_scipy_reference[3]",
    "tests/quantization/test_turboquant.py::TestLloydMax::test_centroids_match_scipy_reference[4]",
    "tests/quantization/test_turboquant.py::TestLloydMax::test_centroids_sorted[2]",
    "tests/quantization/test_turboquant.py::TestLloydMax::test_centroids_sorted[3]",
    "tests/quantization/test_turboquant.py::TestLloydMax::test_centroids_sorted[4]",
    "tests/quantization/test_turboquant.py::TestLloydMax::test_solve_deterministic",
    "tests/quantization/test_turboquant.py::TestLloydMax::test_solve_dtype_float32",
    "tests/quantization/test_turboquant.py::TestLloydMax::test_solve_shapes[2-4]",
    "tests/quantization/test_turboquant.py::TestLloydMax::test_solve_shapes[3-8]",
    "tests/quantization/test_turboquant.py::TestLloydMax::test_solve_shapes[4-16]",
    "tests/quantization/test_turboquant.py::TestRotationMatrix::test_rotation_matrix_det_is_pm1",
    "tests/quantization/test_turboquant.py::TestRotationMatrix::test_rotation_matrix_deterministic",
    "tests/quantization/test_turboquant.py::TestRotationMatrix::test_rotation_matrix_different_seeds",
    "tests/quantization/test_turboquant.py::TestRotationMatrix::test_rotation_matrix_shape_and_orthogonal[128]",
    "tests/quantization/test_turboquant.py::TestRotationMatrix::test_rotation_matrix_shape_and_orthogonal[256]",
    "tests/quantization/test_turboquant.py::TestRotationMatrix::test_rotation_matrix_shape_and_orthogonal[64]",
    "tests/quantization/test_turboquant.py::TestRotationMatrix::test_rotation_matrix_shape_and_orthogonal[96]",
    "tests/quantization/test_turboquant.py::TestStoreDecodeRoundTrip::test_single_token_roundtrip[turboquant_4bit_nc]",
    "tests/quantization/test_turboquant.py::TestStoreDecodeRoundTrip::test_single_token_roundtrip[turboquant_k8v4]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_all_presets_all_head_dims[128-turboquant_3bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_all_presets_all_head_dims[128-turboquant_4bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_all_presets_all_head_dims[128-turboquant_k3v4_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_all_presets_all_head_dims[128-turboquant_k8v4]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_all_presets_all_head_dims[256-turboquant_3bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_all_presets_all_head_dims[256-turboquant_4bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_all_presets_all_head_dims[256-turboquant_k3v4_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_all_presets_all_head_dims[256-turboquant_k8v4]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_all_presets_all_head_dims[64-turboquant_3bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_all_presets_all_head_dims[64-turboquant_4bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_all_presets_all_head_dims[64-turboquant_k3v4_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_all_presets_all_head_dims[64-turboquant_k8v4]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_all_presets_all_head_dims[96-turboquant_3bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_all_presets_all_head_dims[96-turboquant_4bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_all_presets_all_head_dims[96-turboquant_k3v4_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_all_presets_all_head_dims[96-turboquant_k8v4]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_bits_and_centroids[turboquant_3bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_bits_and_centroids[turboquant_4bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_bits_and_centroids[turboquant_k3v4_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_bits_and_centroids[turboquant_k8v4]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_boundary_skip_layers_basic",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_boundary_skip_layers_cap_at_half",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_boundary_skip_layers_small_model",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_boundary_skip_layers_zero",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_centroid_bits_always_positive[turboquant_3bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_centroid_bits_always_positive[turboquant_4bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_centroid_bits_always_positive[turboquant_k3v4_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_centroid_bits_always_positive[turboquant_k8v4]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_invalid_preset_raises",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_key_mode[turboquant_3bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_key_mode[turboquant_4bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_key_mode[turboquant_k3v4_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_key_mode[turboquant_k8v4]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_key_value_packed_sizes_positive[turboquant_3bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_key_value_packed_sizes_positive[turboquant_4bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_key_value_packed_sizes_positive[turboquant_k3v4_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_key_value_packed_sizes_positive[turboquant_k8v4]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_mse_key_or_fp8_exclusive[turboquant_3bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_mse_key_or_fp8_exclusive[turboquant_4bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_mse_key_or_fp8_exclusive[turboquant_k3v4_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_mse_key_or_fp8_exclusive[turboquant_k8v4]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_n_centroids_is_2_to_mse_bits[turboquant_3bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_n_centroids_is_2_to_mse_bits[turboquant_4bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_n_centroids_is_2_to_mse_bits[turboquant_k3v4_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_n_centroids_is_2_to_mse_bits[turboquant_k8v4]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_norm_correction[turboquant_3bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_norm_correction[turboquant_4bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_norm_correction[turboquant_k3v4_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_norm_correction[turboquant_k8v4]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_packed_sizes[turboquant_3bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_packed_sizes[turboquant_4bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_packed_sizes[turboquant_k3v4_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_packed_sizes[turboquant_k8v4]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_padded_slot_is_even[turboquant_3bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_padded_slot_is_even[turboquant_4bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_padded_slot_is_even[turboquant_k3v4_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_padded_slot_is_even[turboquant_k8v4]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_preset_parses[turboquant_3bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_preset_parses[turboquant_4bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_preset_parses[turboquant_k3v4_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_preset_parses[turboquant_k8v4]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_slot_equals_key_plus_value[turboquant_3bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_slot_equals_key_plus_value[turboquant_4bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_slot_equals_key_plus_value[turboquant_k3v4_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_slot_equals_key_plus_value[turboquant_k8v4]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_value_mode[turboquant_3bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_value_mode[turboquant_4bit_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_value_mode[turboquant_k3v4_nc]",
    "tests/quantization/test_turboquant.py::TestTurboQuantConfig::test_value_mode[turboquant_k8v4]",
    "tests/quantization/test_turboquant.py::TestTurboQuantWorkspaceReservation::test_metadata_builder_reserves_decode_and_continuation_prefill_workspace",
    "tests/quantization/test_turboquant.py::TestTurboQuantWorkspaceReservation::test_metadata_builder_skips_continuation_prefill_when_disabled"
]
SCIPY_NODE_IDS = [
    "tests/quantization/test_turboquant.py::TestLloydMax::test_centroids_match_scipy_reference[3]",
    "tests/quantization/test_turboquant.py::TestLloydMax::test_centroids_match_scipy_reference[4]"
]
GPU_ONLY_NODE_IDS = [
    "tests/quantization/test_turboquant.py::TestHadamardRotation::test_hadamard_orthonormal[128]",
    "tests/quantization/test_turboquant.py::TestHadamardRotation::test_hadamard_orthonormal[256]",
    "tests/quantization/test_turboquant.py::TestHadamardRotation::test_hadamard_orthonormal[64]",
    "tests/quantization/test_turboquant.py::TestHadamardRotation::test_hadamard_symmetric[128]",
    "tests/quantization/test_turboquant.py::TestHadamardRotation::test_hadamard_symmetric[256]",
    "tests/quantization/test_turboquant.py::TestHadamardRotation::test_hadamard_symmetric[64]",
    "tests/quantization/test_turboquant.py::TestRotationMatrix::test_rotation_matrix_det_is_pm1",
    "tests/quantization/test_turboquant.py::TestRotationMatrix::test_rotation_matrix_deterministic",
    "tests/quantization/test_turboquant.py::TestRotationMatrix::test_rotation_matrix_different_seeds",
    "tests/quantization/test_turboquant.py::TestRotationMatrix::test_rotation_matrix_shape_and_orthogonal[128]",
    "tests/quantization/test_turboquant.py::TestRotationMatrix::test_rotation_matrix_shape_and_orthogonal[256]",
    "tests/quantization/test_turboquant.py::TestRotationMatrix::test_rotation_matrix_shape_and_orthogonal[64]",
    "tests/quantization/test_turboquant.py::TestRotationMatrix::test_rotation_matrix_shape_and_orthogonal[96]",
    "tests/quantization/test_turboquant.py::TestStoreDecodeRoundTrip::test_single_token_roundtrip[turboquant_4bit_nc]",
    "tests/quantization/test_turboquant.py::TestStoreDecodeRoundTrip::test_single_token_roundtrip[turboquant_k8v4]"
]
JUNIT = "/tmp/exp13_tq_gate_junit.xml"
COUNT_KEYS = ("passed", "skipped", "failed", "errors", "xfailed", "xpassed")


def node_ids_sha256(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def parse_collect(stdout: str) -> list:
    return [ln.strip() for ln in stdout.splitlines() if "::" in ln and not ln.startswith(" ")]


def parse_junit(xml_text: str) -> dict:
    """node id -> {"outcome": passed | skipped | xfailed | failed | error, "message": str}."""
    out = {}
    for tc in ET.fromstring(xml_text).iter("testcase"):
        cls = tc.get("classname", "").split(".")[-1]
        nid = f"{TARGET}::{cls}::{tc.get('name')}"
        outcome, msg = "passed", ""
        for child in tc:
            if child.tag == "skipped":
                outcome = "xfailed" if child.get("type") == "pytest.xfail" else "skipped"
                msg = child.get("message", "")
            elif child.tag == "failure":
                outcome, msg = "failed", child.get("message", "")
            elif child.tag == "error":
                outcome, msg = "error", child.get("message", "")
        out[nid] = {"outcome": outcome, "message": msg}
    return out


def parse_summary(text: str) -> dict:
    """Counts from pytest's final summary line, e.g. '121 passed, 2 skipped in 12.3s'."""
    counts = {k: 0 for k in COUNT_KEYS}
    for line in reversed(text.splitlines()):
        if re.search(r" in [\d.]+s", line) and re.search(r"\d+ (passed|failed|skipped|errors?|xfailed|xpassed)", line):
            for n, k in re.findall(r"(\d+) (passed|skipped|failed|errors?|xfailed|xpassed)", line):
                counts["errors" if k.startswith("error") else k] = int(n)
            break
    return counts


def evaluate(collected, collect_rc, scipy_importable, gpgpu, run_rc, items, counts) -> dict:
    checks = {"collect_exit_0": collect_rc == 0, "collected_count_123": len(collected) == EXPECTED_COUNT,
              "collected_hash_matches": node_ids_sha256(collected) == EXPECTED_NODE_IDS_SHA256,
              "gpgpu_available": gpgpu is True}
    if not all(checks.values()) or run_rc is None:
        return {"stage": "collection", "checks": checks, "valid": False if not all(checks.values()) else None}
    skipped = sorted(n for n, v in items.items() if v["outcome"] == "skipped")
    expected_skips = [] if scipy_importable else sorted(SCIPY_NODE_IDS)
    n_pass = EXPECTED_COUNT - len(expected_skips)
    checks.update({
        "run_exit_0": run_rc == 0,
        "every_collected_item_reported_once": sorted(items) == sorted(collected),
        "failed_0": counts["failed"] == 0 and not any(v["outcome"] == "failed" for v in items.values()),
        "errors_0": counts["errors"] == 0 and not any(v["outcome"] == "error" for v in items.values()),
        "xfailed_0": counts["xfailed"] == 0 and not any(v["outcome"] == "xfailed" for v in items.values()),
        "xpassed_0": counts["xpassed"] == 0,
        "skipped_exactly_allowed_set": skipped == expected_skips,
        "skip_reasons_missing_scipy": all("scipy" in items[n]["message"].lower() for n in skipped),
        "passed_count": counts["passed"] == n_pass and sum(v["outcome"] == "passed" for v in items.values()) == n_pass,
        "skipped_count": counts["skipped"] == len(expected_skips),
        "gpu_only_all_executed_and_passed": all(items.get(n, {}).get("outcome") == "passed" for n in GPU_ONLY_NODE_IDS),
    })
    return {"stage": "execution", "checks": checks, "valid": all(checks.values()),
            "expected_case": ("scipy absent: 121 passed + the 2 frozen SciPy skips" if not scipy_importable
                              else "scipy present: 123 passed, 0 skipped"),
            "skipped_node_ids": skipped, "counts": counts}


def main() -> int:
    scipy_importable = importlib.util.find_spec("scipy") is not None
    g = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.is_available() or torch.xpu.is_available())"],
                       capture_output=True, text=True)
    gpgpu = g.stdout.strip() == "True"
    collect_cmd = [*BASE, "--collect-only", TARGET]
    c = subprocess.run(collect_cmd, capture_output=True, text=True)
    print(c.stdout[-3000:], flush=True)
    collected = parse_collect(c.stdout)
    env = {"scipy_importable": scipy_importable, "gpgpu_available": gpgpu, "collected_count": len(collected),
           "collected_sha256": node_ids_sha256(collected), "collect_cmd": collect_cmd, "collect_returncode": c.returncode}
    pre = evaluate(collected, c.returncode, scipy_importable, gpgpu, None, {}, {})
    if pre["valid"] is False:  # STOP before executing the test suite
        print("EXP13_TQ_GATE_SUMMARY=" + json.dumps({**env, **pre}, sort_keys=True), flush=True)
        return 3
    run_cmd = [*BASE, "-rA", f"--junitxml={JUNIT}", TARGET]
    r = subprocess.run(run_cmd, capture_output=True, text=True)
    print(r.stdout[-6000:], flush=True)
    if r.stderr:
        print(r.stderr[-2000:], flush=True)
    try:
        with open(JUNIT, encoding="utf-8") as f:
            items = parse_junit(f.read())
    except (OSError, ET.ParseError):
        items = {}
    res = evaluate(collected, c.returncode, scipy_importable, gpgpu, r.returncode, items, parse_summary(r.stdout))
    print("EXP13_TQ_GATE_SUMMARY=" + json.dumps({**env, "run_cmd": run_cmd, "run_returncode": r.returncode, **res},
                                                sort_keys=True), flush=True)
    return 0 if res["valid"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
