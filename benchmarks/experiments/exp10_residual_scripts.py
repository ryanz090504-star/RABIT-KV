"""
MLSys 2027 Experiment 10 -- derive the residual-window-ablation quality scripts from the FROZEN canonical quality
scripts.

Separate from Experiments 7 / 8 / 9 (their generators are not modified, so their accepted evidence stays reproducible;
Exp7's pure helpers are imported read-only). The canonical benchmarks/quality/*.py are never edited. Each
Experiment 10 script is a deterministic copy with exactly these edits (docs/MLSYS_EXPERIMENT_PLAN.md, Experiment 10:
"new config_for_method entries varying only residual"), each verified to apply exactly once:

  1. a provenance header line (comment);
  2. the Modal App name (rabit-kv-exp10-residual-<benchmark>);
  3. the `allowed` method set, extended with rabit2_r0, rabit2_r2 and rabit2_r8 (plus its error message);
  4. `config_for_method` branches returning the UNCHANGED `rabit2` config with ONLY `residual` replaced (0, 2 or 8)
     and a new display name.

In the canonical code `residual` (R) is read only by q_with_residual: for each layer's prefix K and (separately) V
tensor of T tokens, the newest min(R, T) tokens stay BF16 and the older T - R tokens are quantized; R = 0 quantizes
all T tokens; T <= R keeps the whole tensor BF16.

Every quantizer, the metadata codec, the logical accounting, the evaluation loop, datasets, selection, seeds and
prompts stay the canonical code.

Usage:
    python benchmarks/mlsys2027/exp10_residual_scripts.py --write
    python benchmarks/mlsys2027/exp10_residual_scripts.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp7_kbit_scripts as g7  # noqa: E402  (read-only reuse of the canonical paths / helpers)

ROOT = g7.ROOT
DERIVED_DIR = ROOT / "benchmarks" / "mlsys2027" / "exp10_residual"
BENCHMARKS = g7.BENCHMARKS
R_ABLATION = {"rabit2_r0": 0, "rabit2_r2": 2, "rabit2_r8": 8}  # control: rabit2 (residual = 4)
CONTROL = "rabit2"

OLD_ALLOWED = g7.OLD_ALLOWED
NEW_ALLOWED = 'allowed = {"bf16", "rabit8", "rabit4", "rabit3", "rabit2", "rabit2_r0", "rabit2_r2", "rabit2_r8"}'
OLD_USE = g7.OLD_USE
NEW_USE = '"Use bf16,rabit8,rabit4,rabit3,rabit2,rabit2_r0,rabit2_r2,rabit2_r8."'
RAISE_LINE = g7.RAISE_LINE
R_BRANCH = '''        if method in ("rabit2_r0", "rabit2_r2", "rabit2_r8"):
            # Experiment 10 single-axis residual-window ablation: the frozen rabit2 policy with ONLY residual changed.
            config = dict(config_for_method("rabit2"))
            config["residual"] = {"rabit2_r0": 0, "rabit2_r2": 2, "rabit2_r8": 8}[method]
            config["name"] = f"K3V2 META8g64 G32 R{config['residual']} (residual-window ablation of rabit2)"
            return config
'''


def derive(benchmark: str, canonical_text: str) -> str:
    """The Experiment 10 copy of one canonical quality script (pure function of its text)."""
    apps = g7.APP_RE.findall(canonical_text)
    if len(apps) != 1:
        raise ValueError(f"{benchmark}: expected exactly one Modal App definition, found {len(apps)}")
    text = g7.APP_RE.sub(f'app = modal.App("rabit-kv-exp10-residual-{benchmark.replace("_", "-")}")', canonical_text)
    text = g7._replace_once(text, OLD_ALLOWED, NEW_ALLOWED, f"{benchmark}: allowed set")
    text = g7._replace_once(text, OLD_USE, NEW_USE, f"{benchmark}: allowed-set error message")
    text = g7._replace_once(text, RAISE_LINE, R_BRANCH + RAISE_LINE, f"{benchmark}: config_for_method fallthrough")
    header = (f"# GENERATED for MLSys 2027 Experiment 10 (residual-window ablation) from benchmarks/quality/{benchmark}.py "
              f"(LF-normalized sha256 {g7.sha256_text(canonical_text)}) by benchmarks/mlsys2027/exp10_residual_scripts.py. "
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
