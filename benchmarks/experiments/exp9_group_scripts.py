"""
MLSys 2027 Experiment 9 -- derive the group-size-ablation quality scripts from the FROZEN canonical quality scripts.

Separate from Experiments 7 / 8 (exp7_kbit_scripts.py and exp8_vbit_scripts.py are not modified, so the accepted
Exp7 / Exp8 evidence stays reproducible; Exp7's pure helpers are imported read-only). The canonical
benchmarks/quality/*.py are never edited. Each Experiment 9 script is a deterministic copy with exactly these edits
(docs/MLSYS_EXPERIMENT_PLAN.md, Experiment 9: "new config_for_method entries varying only k_group/v_group"), each
verified to apply exactly once:

  1. a provenance header line (comment);
  2. the Modal App name (rabit-kv-exp9-group-<benchmark>);
  3. the `allowed` method set, extended with rabit2_g16 and rabit2_g64 (plus its error message);
  4. `config_for_method` branches returning the UNCHANGED `rabit2` config with ONLY `k_group` and `v_group` replaced
     (both by the same G: 16 or 64) and a new display name.

"G" is the single group-size knob of the rabit2 policy ("G32" = k_group = v_group = 32), exactly as the plan's
G16 / G32 / G64 sweep specifies. In the canonical code k_group is the SEQUENCE group of the seq_affine key quantizer
(G tokens per channel share one min / scale) and v_group is the HEAD_DIM group of the group_affine value quantizer
(G channels per token share one min / scale); both are read only in q_tensor.

Every quantizer, the metadata codec, the logical accounting, the evaluation loop, datasets, selection, seeds and
prompts stay the canonical code. No correctness-check injection (the plan specifies one for Experiment 8 only).

Usage:
    python benchmarks/mlsys2027/exp9_group_scripts.py --write
    python benchmarks/mlsys2027/exp9_group_scripts.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp7_kbit_scripts as g7  # noqa: E402  (read-only reuse of the canonical paths / helpers)

ROOT = g7.ROOT
DERIVED_DIR = ROOT / "benchmarks" / "mlsys2027" / "exp9_group"
BENCHMARKS = g7.BENCHMARKS
G_ABLATION = {"rabit2_g16": 16, "rabit2_g64": 64}  # control: rabit2 (k_group = v_group = 32)
GROUP_FIELDS = ("k_group", "v_group")
CONTROL = "rabit2"

OLD_ALLOWED = g7.OLD_ALLOWED
NEW_ALLOWED = 'allowed = {"bf16", "rabit8", "rabit4", "rabit3", "rabit2", "rabit2_g16", "rabit2_g64"}'
OLD_USE = g7.OLD_USE
NEW_USE = '"Use bf16,rabit8,rabit4,rabit3,rabit2,rabit2_g16,rabit2_g64."'
RAISE_LINE = g7.RAISE_LINE
G_BRANCH = '''        if method in ("rabit2_g16", "rabit2_g64"):
            # Experiment 9 single-axis group-size ablation: the frozen rabit2 policy with ONLY k_group / v_group
            # changed (both to the same G).
            config = dict(config_for_method("rabit2"))
            config["k_group"] = config["v_group"] = {"rabit2_g16": 16, "rabit2_g64": 64}[method]
            config["name"] = f"K3V2 META8g64 G{config['k_group']} R4 (group-size ablation of rabit2)"
            return config
'''


def derive(benchmark: str, canonical_text: str) -> str:
    """The Experiment 9 copy of one canonical quality script (pure function of its text)."""
    apps = g7.APP_RE.findall(canonical_text)
    if len(apps) != 1:
        raise ValueError(f"{benchmark}: expected exactly one Modal App definition, found {len(apps)}")
    text = g7.APP_RE.sub(f'app = modal.App("rabit-kv-exp9-group-{benchmark.replace("_", "-")}")', canonical_text)
    text = g7._replace_once(text, OLD_ALLOWED, NEW_ALLOWED, f"{benchmark}: allowed set")
    text = g7._replace_once(text, OLD_USE, NEW_USE, f"{benchmark}: allowed-set error message")
    text = g7._replace_once(text, RAISE_LINE, G_BRANCH + RAISE_LINE, f"{benchmark}: config_for_method fallthrough")
    header = (f"# GENERATED for MLSys 2027 Experiment 9 (group-size ablation) from benchmarks/quality/{benchmark}.py "
              f"(LF-normalized sha256 {g7.sha256_text(canonical_text)}) by benchmarks/mlsys2027/exp9_group_scripts.py. "
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
