"""Hotfix wrapper for Stage4B3 GQA4 integration.

The previous fixed installer still carried the typo inside its compressed Modal
runner payload. This wrapper loads that installer, patches the decoded runner
from SNAPSHOT -> SNAP, recompresses it, and then executes the original main().
"""
from __future__ import annotations

import base64
import importlib.util
import sys
import zlib
from pathlib import Path

ROOT = Path.cwd().resolve()

# Prefer the user's local fixed installer, fall back to the original name only
# if needed. We do NOT execute it during import because it is guarded by
# if __name__ == "__main__".
candidates = [
    ROOT / "RABIT2_stage4b3_gqa4_integrate_and_run_fixed.py",
    ROOT / "RABIT2_stage4b3_gqa4_integrate_and_run.py",
]
target = next((p for p in candidates if p.is_file()), None)
if target is None:
    raise FileNotFoundError(
        "Could not find RABIT2_stage4b3_gqa4_integrate_and_run_fixed.py "
        "or the original integration script in the current directory."
    )

spec = importlib.util.spec_from_file_location("_rabit2_stage4b3_installer", target)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load installer: {target}")

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

if not hasattr(mod, "R"):
    raise RuntimeError("Installer payload variable R was not found")

runner = zlib.decompress(base64.b64decode(mod.R)).decode("utf-8")

before = runner.count("SNAPSHOT")
if before == 0:
    print("Compressed Modal runner already has no SNAPSHOT typo.")
else:
    runner = runner.replace("SNAPSHOT", "SNAP")
    print(f"Patched compressed Modal runner: SNAPSHOT -> SNAP ({before} occurrence(s))")

# Re-encode the corrected Modal runner and execute the original installer.
mod.R = base64.b64encode(zlib.compress(runner.encode("utf-8"), 9)).decode("ascii")

# Defensive proof before doing any source modification.
check = zlib.decompress(base64.b64decode(mod.R)).decode("utf-8")
if "SNAPSHOT" in check:
    raise RuntimeError("Hotfix failed: SNAPSHOT still present in Modal runner")
if "str(SNAP)" not in check:
    raise RuntimeError("Hotfix failed: corrected str(SNAP) not found in Modal runner")

print(f"Loaded installer: {target.name}")
print("Compressed runner hotfix preflight: PASSED")

mod.main()
