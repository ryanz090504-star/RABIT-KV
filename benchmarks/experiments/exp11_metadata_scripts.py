"""
MLSys 2027 Experiment 11 -- derive the metadata-ablation quality scripts from the FROZEN canonical quality scripts.

Separate from Experiments 7-10 (their generators are not modified; Exp7's pure helpers are imported read-only). The
canonical benchmarks/quality/*.py are never edited. Each Experiment 11 script is a deterministic copy with exactly
these edits (docs/MLSYS_EXPERIMENT_PLAN.md, Experiment 11: "new config_for_method entries selecting
metadata_mode='bf16' or varying metadata_group_size (32/128) while leaving K/V/G/R untouched"), each verified to apply
exactly once:

  1. a provenance header line (comment);
  2. the Modal App name (rabit-kv-exp11-metadata-<benchmark>);
  3. the `allowed` method set, extended with rabit2_mbf, rabit2_m32 and rabit2_m128 (plus its error message);
  4. `config_for_method` branches returning the UNCHANGED `rabit2` config with exactly one metadata field replaced:
       rabit2_mbf   metadata_mode "int8" -> "bf16"   (metadata_group_size stays 64 and is inert in bf16 mode:
                                                      encode_metadata returns before reading it)
       rabit2_m32   metadata_group_size 64 -> 32
       rabit2_m128  metadata_group_size 64 -> 128
     plus a new display name.

Method names are at most 11 characters so every script's summary row keeps a whitespace delimiter after the name.

Every quantizer, the metadata codec, the logical accounting, the evaluation loop, datasets, selection, seeds and
prompts stay the canonical code.

Usage:
    python benchmarks/mlsys2027/exp11_metadata_scripts.py --write
    python benchmarks/mlsys2027/exp11_metadata_scripts.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp7_kbit_scripts as g7  # noqa: E402  (read-only reuse of the canonical paths / helpers)

ROOT = g7.ROOT
DERIVED_DIR = ROOT / "benchmarks" / "mlsys2027" / "exp11_metadata"
BENCHMARKS = g7.BENCHMARKS
# method -> the single metadata field it changes, and its value (control: rabit2 = int8 / 64)
M_ABLATION = {"rabit2_mbf": ("metadata_mode", "bf16"), "rabit2_m32": ("metadata_group_size", 32),
              "rabit2_m128": ("metadata_group_size", 128)}
CONTROL = "rabit2"

OLD_ALLOWED = g7.OLD_ALLOWED
NEW_ALLOWED = 'allowed = {"bf16", "rabit8", "rabit4", "rabit3", "rabit2", "rabit2_mbf", "rabit2_m32", "rabit2_m128"}'
OLD_USE = g7.OLD_USE
NEW_USE = '"Use bf16,rabit8,rabit4,rabit3,rabit2,rabit2_mbf,rabit2_m32,rabit2_m128."'
RAISE_LINE = g7.RAISE_LINE
M_BRANCH = '''        if method in ("rabit2_mbf", "rabit2_m32", "rabit2_m128"):
            # Experiment 11 metadata ablation: the frozen rabit2 policy with exactly one metadata field changed.
            config = dict(config_for_method("rabit2"))
            if method == "rabit2_mbf":
                config["metadata_mode"] = "bf16"
                config["name"] = "K3V2 METAbf16 G32 R4 (metadata ablation of rabit2)"
            else:
                config["metadata_group_size"] = {"rabit2_m32": 32, "rabit2_m128": 128}[method]
                config["name"] = f"K3V2 META8g{config['metadata_group_size']} G32 R4 (metadata ablation of rabit2)"
            return config
'''


def derive(benchmark: str, canonical_text: str) -> str:
    """The Experiment 11 copy of one canonical quality script (pure function of its text)."""
    apps = g7.APP_RE.findall(canonical_text)
    if len(apps) != 1:
        raise ValueError(f"{benchmark}: expected exactly one Modal App definition, found {len(apps)}")
    text = g7.APP_RE.sub(f'app = modal.App("rabit-kv-exp11-metadata-{benchmark.replace("_", "-")}")', canonical_text)
    text = g7._replace_once(text, OLD_ALLOWED, NEW_ALLOWED, f"{benchmark}: allowed set")
    text = g7._replace_once(text, OLD_USE, NEW_USE, f"{benchmark}: allowed-set error message")
    text = g7._replace_once(text, RAISE_LINE, M_BRANCH + RAISE_LINE, f"{benchmark}: config_for_method fallthrough")
    header = (f"# GENERATED for MLSys 2027 Experiment 11 (metadata ablation) from benchmarks/quality/{benchmark}.py "
              f"(LF-normalized sha256 {g7.sha256_text(canonical_text)}) by benchmarks/mlsys2027/exp11_metadata_scripts.py. "
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
