"""
MLSys 2027 Experiment 7 -- derive the K-bit-ablation quality scripts from the FROZEN canonical quality scripts.

The canonical benchmarks/quality/{continuation_ppl,niah,passage_retrieval,hotpotqa,qasper}.py are never edited
(Experiment 1 / 2 runners pin their exact text). Instead each Experiment 7 script is a deterministic copy with
exactly four textual edits, all verified to apply exactly once:

  1. a provenance header line (comment);
  2. the Modal App name (rabit-kv-exp7-kbit-<benchmark>) so Exp7 runs are identifiable;
  3. the `allowed` method set, extended with rabit2_k2 and rabit2_k4 (plus its error message);
  4. two new `config_for_method` branches that return the UNCHANGED `rabit2` config with ONLY `k_bits` replaced
     (2 or 4) and a new display name.

Every quantizer, the metadata codec, the logical-byte accounting, the evaluation loop, datasets, sample selection,
seeds and prompts stay byte-for-byte the canonical code. `rabit2` (K3/V2/G32/R4/META8g64) is the untouched control.

Usage:
    python benchmarks/mlsys2027/exp7_kbit_scripts.py --write   (regenerate the committed derived copies)
    python benchmarks/mlsys2027/exp7_kbit_scripts.py --check   (verify committed copies == derivation)
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUALITY_DIR = ROOT / "benchmarks" / "quality"
DERIVED_DIR = ROOT / "benchmarks" / "mlsys2027" / "exp7_kbit"
BENCHMARKS = ("continuation_ppl", "niah", "passage_retrieval", "hotpotqa", "qasper")
K_ABLATION = {"rabit2_k2": 2, "rabit2_k4": 4}  # control: rabit2 (k_bits = 3)
CONTROL = "rabit2"

OLD_ALLOWED = 'allowed = {"bf16", "rabit8", "rabit4", "rabit3", "rabit2"}'
NEW_ALLOWED = 'allowed = {"bf16", "rabit8", "rabit4", "rabit3", "rabit2", "rabit2_k2", "rabit2_k4"}'
OLD_USE = '"Use bf16,rabit8,rabit4,rabit3,rabit2."'
NEW_USE = '"Use bf16,rabit8,rabit4,rabit3,rabit2,rabit2_k2,rabit2_k4."'
RAISE_LINE = '        raise ValueError(f"No RABIT-KV configuration for {method}.")'
K_BRANCH = '''        if method in ("rabit2_k2", "rabit2_k4"):
            # Experiment 7 single-axis K-bit ablation: the frozen rabit2 policy with ONLY k_bits changed.
            config = dict(config_for_method("rabit2"))
            config["k_bits"] = {"rabit2_k2": 2, "rabit2_k4": 4}[method]
            config["name"] = f"K{config['k_bits']}V2 META8g64 G32 R4 (K-bit ablation of rabit2)"
            return config
'''
APP_RE = re.compile(r'^app = modal\.App\("([^"]+)"\)$', re.MULTILINE)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"{what}: expected exactly one occurrence, found {text.count(old)}")
    return text.replace(old, new)


def derive(benchmark: str, canonical_text: str) -> str:
    """The Experiment 7 copy of one canonical quality script (pure function of its text)."""
    apps = APP_RE.findall(canonical_text)
    if len(apps) != 1:
        raise ValueError(f"{benchmark}: expected exactly one Modal App definition, found {len(apps)}")
    text = APP_RE.sub(f'app = modal.App("rabit-kv-exp7-kbit-{benchmark.replace("_", "-")}")', canonical_text)
    text = _replace_once(text, OLD_ALLOWED, NEW_ALLOWED, f"{benchmark}: allowed set")
    text = _replace_once(text, OLD_USE, NEW_USE, f"{benchmark}: allowed-set error message")
    text = _replace_once(text, RAISE_LINE, K_BRANCH + RAISE_LINE, f"{benchmark}: config_for_method fallthrough")
    header = (f"# GENERATED for MLSys 2027 Experiment 7 (K-bit ablation) from benchmarks/quality/{benchmark}.py "
              f"(LF-normalized sha256 {sha256_text(canonical_text)}) by benchmarks/mlsys2027/exp7_kbit_scripts.py. Do not edit.\n")
    return header + text


def canonical_path(benchmark: str) -> Path:
    return QUALITY_DIR / f"{benchmark}.py"


def derived_path(benchmark: str) -> Path:
    return DERIVED_DIR / f"{benchmark}.py"


def expected_derived(benchmark: str) -> str:
    return derive(benchmark, canonical_path(benchmark).read_text(encoding="utf-8"))


def check() -> dict:
    out = {}
    for b in BENCHMARKS:
        path = derived_path(b)
        ok = path.is_file() and path.read_text(encoding="utf-8") == expected_derived(b)
        out[b] = ok
    return out


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
