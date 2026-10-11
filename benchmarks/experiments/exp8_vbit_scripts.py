"""
MLSys 2027 Experiment 8 -- derive the V-bit-ablation quality scripts from the FROZEN canonical quality scripts.

Separate from Experiment 7's generator (exp7_kbit_scripts.py is not modified, so the accepted Exp7 evidence stays
reproducible). The canonical benchmarks/quality/*.py are never edited. Each Experiment 8 script is a deterministic
copy with exactly these edits, each verified to apply exactly once:

  1. a provenance header line (comment);
  2. the Modal App name (rabit-kv-exp8-vbit-<benchmark>);
  3. the `allowed` method set, extended with rabit2_v1, rabit2_v3 and the pre-registered substitute rabit2_v4 (plus its
     error message);
  4. `config_for_method` branches returning the UNCHANGED `rabit2` config with ONLY `v_bits` replaced (1, 3 or 4) and a
     new display name;
  5. the plan's uniform correctness check (docs/MLSYS_EXPERIMENT_PLAN.md, Experiment 8): two observation-only helpers
     defined before quantize_then_dequantize_cache, called (a) per layer right after quantization, checking the
     quantized codes of K and V (in range, and not all-identical: degeneracy) and (b) after dequantization, checking
     every rebuilt K / V tensor is finite (no NaN / Inf), before scoring. It runs for EVERY quantized method (V1, V2
     control, V3 alike -- bf16 is never quantized), never alters a tensor, prints one EXP8_CORRECTNESS_OK line per
     quantize call and raises EXP8_CORRECTNESS_FAILURE on a violation.

Every quantizer, the metadata codec, the logical accounting, the evaluation loop, datasets, selection, seeds and
prompts stay the canonical code.

Usage:
    python benchmarks/mlsys2027/exp8_vbit_scripts.py --write
    python benchmarks/mlsys2027/exp8_vbit_scripts.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp7_kbit_scripts as g7  # noqa: E402  (read-only reuse of the canonical paths / helpers)

ROOT = g7.ROOT
DERIVED_DIR = ROOT / "benchmarks" / "mlsys2027" / "exp8_vbit"
BENCHMARKS = g7.BENCHMARKS
V_ABLATION = {"rabit2_v1": 1, "rabit2_v3": 3, "rabit2_v4": 4}  # rabit2_v4: pre-registered substitute only
CONTROL = "rabit2"

OLD_ALLOWED = g7.OLD_ALLOWED
NEW_ALLOWED = 'allowed = {"bf16", "rabit8", "rabit4", "rabit3", "rabit2", "rabit2_v1", "rabit2_v3", "rabit2_v4"}'
OLD_USE = g7.OLD_USE
NEW_USE = '"Use bf16,rabit8,rabit4,rabit3,rabit2,rabit2_v1,rabit2_v3,rabit2_v4."'
RAISE_LINE = g7.RAISE_LINE
V_BRANCH = '''        if method in ("rabit2_v1", "rabit2_v3", "rabit2_v4"):
            # Experiment 8 single-axis V-bit ablation: the frozen rabit2 policy with ONLY v_bits changed.
            config = dict(config_for_method("rabit2"))
            config["v_bits"] = {"rabit2_v1": 1, "rabit2_v3": 3, "rabit2_v4": 4}[method]
            config["name"] = f"K3V{config['v_bits']} META8g64 G32 R4 (V-bit ablation of rabit2)"
            return config
'''
QDQ_DEF = "    def quantize_then_dequantize_cache(cache, config):\n"
CHECK_HELPERS = '''    # --- Experiment 8 uniform correctness check (observation only; never alters a tensor) ---
    exp8_layer_stats = []

    def exp8_state_codes(state):
        if state["type"] == "bf16":
            return []
        if state["type"] == "split":
            return exp8_state_codes(state["old"]) + exp8_state_codes(state["recent"])
        return [(state["codes"], int(state["bits"]))]

    def exp8_check_codes(config, quantized_key, quantized_value):
        for side, state in (("k", quantized_key), ("v", quantized_value)):
            for codes, bits in exp8_state_codes(state):
                low, high = int(codes.amin().item()), int(codes.amax().item())
                if low < 0 or high > 2 ** bits - 1:
                    raise RuntimeError(f"EXP8_CORRECTNESS_FAILURE kind=code_out_of_range name={config['name']} "
                                       f"side={side} bits={bits} min={low} max={high}")
                if low == high:
                    raise RuntimeError(f"EXP8_CORRECTNESS_FAILURE kind=degenerate_all_codes_identical "
                                       f"name={config['name']} side={side} bits={bits} code={low}")
                exp8_layer_stats.append((side, high - low))

    def exp8_check_finite(config, rebuilt_layers):
        nonfinite = 0
        for key_tensor, value_tensor in rebuilt_layers:
            for tensor in (key_tensor, value_tensor):
                if not bool(torch.isfinite(tensor).all().item()):
                    nonfinite += 1
        if nonfinite:
            raise RuntimeError(f"EXP8_CORRECTNESS_FAILURE kind=nonfinite_dequantized_kv name={config['name']} "
                               f"tensors={nonfinite}")
        spans = {side: min((s for d, s in exp8_layer_stats if d == side), default=-1) for side in ("k", "v")}
        print(f"EXP8_CORRECTNESS_OK name={config['name']} layers={len(rebuilt_layers)} "
              f"k_bits={config['k_bits']} v_bits={config['v_bits']} nonfinite=0 "
              f"min_code_span_k={spans['k']} min_code_span_v={spans['v']}", flush=True)
        exp8_layer_stats.clear()

'''
LOOP_ANCHOR = "            logical_bytes += stored_state_logical_bytes(quantized_value)\n"
LOOP_CALL = "            exp8_check_codes(config, quantized_key, quantized_value)\n"
RETURN_ANCHOR = "        return (\n            tuple_to_dynamic_cache(tuple(rebuilt_layers)),\n"
RETURN_CALL = "        exp8_check_finite(config, rebuilt_layers)\n"


def derive(benchmark: str, canonical_text: str) -> str:
    """The Experiment 8 copy of one canonical quality script (pure function of its text)."""
    apps = g7.APP_RE.findall(canonical_text)
    if len(apps) != 1:
        raise ValueError(f"{benchmark}: expected exactly one Modal App definition, found {len(apps)}")
    text = g7.APP_RE.sub(f'app = modal.App("rabit-kv-exp8-vbit-{benchmark.replace("_", "-")}")', canonical_text)
    text = g7._replace_once(text, OLD_ALLOWED, NEW_ALLOWED, f"{benchmark}: allowed set")
    text = g7._replace_once(text, OLD_USE, NEW_USE, f"{benchmark}: allowed-set error message")
    text = g7._replace_once(text, RAISE_LINE, V_BRANCH + RAISE_LINE, f"{benchmark}: config_for_method fallthrough")
    text = g7._replace_once(text, QDQ_DEF, CHECK_HELPERS + QDQ_DEF, f"{benchmark}: quantize_then_dequantize_cache")
    text = g7._replace_once(text, LOOP_ANCHOR, LOOP_ANCHOR + LOOP_CALL, f"{benchmark}: per-layer quantize loop")
    text = g7._replace_once(text, RETURN_ANCHOR, RETURN_CALL + RETURN_ANCHOR, f"{benchmark}: rebuilt-cache return")
    header = (f"# GENERATED for MLSys 2027 Experiment 8 (V-bit ablation) from benchmarks/quality/{benchmark}.py "
              f"(LF-normalized sha256 {g7.sha256_text(canonical_text)}) by benchmarks/mlsys2027/exp8_vbit_scripts.py. "
              f"Do not edit.\n")
    return header + text


def canonical_path(benchmark: str) -> Path:
    return g7.canonical_path(benchmark)


def derived_path(benchmark: str) -> Path:
    return DERIVED_DIR / f"{benchmark}.py"


def expected_derived(benchmark: str) -> str:
    return derive(benchmark, canonical_path(benchmark).read_text(encoding="utf-8"))


def check() -> dict:
    return {b: derived_path(b).is_file() and derived_path(b).read_text(encoding="utf-8") == expected_derived(b)
            for b in BENCHMARKS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.write:
        DERIVED_DIR.mkdir(parents=True, exist_ok=True)
        for b in BENCHMARKS:
            derived_path(b).write_text(expected_derived(b), encoding="utf-8", newline="\n")
            print(f"wrote {derived_path(b).relative_to(ROOT).as_posix()}")
        return 0
    res = check()
    print(res)
    return 0 if all(res.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
