"""
MLSys 2027 Experiment 14 -- FROZEN Model-B snapshot identity (stdlib only; imported locally by the runner / tests and
inside the Modal containers).

Model B = ModelScope Qwen/Qwen2.5-7B-Instruct at the IMMUTABLE commit MODEL_REVISION. Resolved during preparation
(2026-09-30): `git ls-remote https://www.modelscope.cn/Qwen/Qwen2.5-7B-Instruct.git` -> refs/heads/master =
16c174980d8a1492910551634b4969e69cdc2444 (the repository has no tags); the ModelScope file API returns an identical
15-file manifest for Revision=<commit> and Revision=master; config.json / tokenizer_config.json /
generation_config.json / model.safetensors.index.json downloaded at the commit hash to the listed SHA-256 values.

Enforcement (the probe, serving and quality parts all require this SAME identity):
  * probe / serving: snapshot_download(MODEL_ID, revision=MODEL_REVISION) inside the container, then verify_dir() on
    the returned directory (every frozen file present with the frozen size and SHA-256, no extra non-hidden file);
    any mismatch aborts before the first engine starts;
  * quality (the canonical scripts call snapshot_download without a revision and may not be modified): before the
    runs, master must resolve to MODEL_REVISION with the identical API manifest; after the runs, master must still
    resolve to MODEL_REVISION and EVERY Qwen2.5-7B-Instruct snapshot directory in the model volume must pass
    verify_dir() -- otherwise the quality part is invalid.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "16c174980d8a1492910551634b4969e69cdc2444"
MODEL_GIT_URL = "https://www.modelscope.cn/Qwen/Qwen2.5-7B-Instruct.git"
MODEL_FILES_API = ("https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-7B-Instruct/repo/files"
                   "?Revision={revision}&Recursive=true")
# (path, size in bytes, sha256) of every file at MODEL_REVISION (ModelScope API listing; sorted by path).
FROZEN_FILES = [
    [".gitattributes", 1519, "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"],
    ["LICENSE", 11343, "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"],
    ["README.md", 6240, "f366f33bbf6bcadbb7d87f0a21a7b65584a56b8d58b0743c77c88bee625b93a6"],
    ["config.json", 663, "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c"],
    ["configuration.json", 2, "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"],
    ["generation_config.json", 243, "3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f"],
    ["merges.txt", 1671839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"],
    ["model-00001-of-00004.safetensors", 3945441440, "a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7"],
    ["model-00002-of-00004.safetensors", 3864726352, "f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185"],
    ["model-00003-of-00004.safetensors", 3864726424, "8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5"],
    ["model-00004-of-00004.safetensors", 3556377672, "1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd"],
    ["model.safetensors.index.json", 27752, "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028"],
    ["tokenizer.json", 7031645, "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"],
    ["tokenizer_config.json", 7305, "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"],
    ["vocab.json", 2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"],
]
SNAPSHOT_DIR_RE = re.compile(r"qwen2(\.|___)5-7b-instruct", re.IGNORECASE)  # new ("Qwen--Qwen2.5-...") and legacy layouts


def manifest_sha256(files=FROZEN_FILES) -> str:
    """SHA-256 of the sorted 'path<TAB>size<TAB>sha256' lines."""
    return hashlib.sha256("\n".join(f"{p}\t{s}\t{h}" for p, s, h in sorted(files)).encode()).hexdigest()


MODEL_MANIFEST_SHA256 = manifest_sha256()


def listing_manifest(api_json: dict) -> list:
    """[[path, size, sha256], ...] from a ModelScope repo/files API response (blobs only, sorted)."""
    return sorted([f["Path"], int(f["Size"]), f["Sha256"]] for f in api_json["Data"]["Files"] if f.get("Type") == "blob")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_dir(path) -> dict:
    """Every frozen file present with the frozen size and SHA-256; no extra non-hidden file."""
    root = Path(path)
    mismatches, observed = [], {}
    for rel, size, sha in FROZEN_FILES:
        p = root / rel
        if not p.is_file():
            mismatches.append({"file": rel, "reason": "missing"})
            continue
        st = p.stat().st_size
        got = _sha256_file(p) if st == size else None
        observed[rel] = {"bytes": st, "sha256": got}
        if st != size or got != sha:
            mismatches.append({"file": rel, "reason": "size/sha256 differ", "bytes": st, "sha256": got})
    frozen = {rel for rel, _, _ in FROZEN_FILES}
    extra = sorted(str(p.relative_to(root)).replace(os.sep, "/") for p in root.rglob("*")
                   if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts[:-1])
                   and not p.name.startswith(".") and str(p.relative_to(root)).replace(os.sep, "/") not in frozen)
    return {"dir": str(root), "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "manifest_sha256": MODEL_MANIFEST_SHA256, "files_checked": len(FROZEN_FILES), "observed": observed,
            "mismatches": mismatches, "extra_files": extra, "passed": not mismatches and not extra}


def find_snapshot_dirs(cache_root) -> list[str]:
    """Every directory under the model volume that holds a Qwen2.5-7B-Instruct snapshot (contains config.json)."""
    out = []
    for cfg in Path(cache_root).rglob("config.json"):
        d = cfg.parent
        if SNAPSHOT_DIR_RE.search(str(d)) and not any(part.startswith(".") for part in d.parts[len(Path(cache_root).parts):]):
            out.append(str(d))
    return sorted(out)


def scan_volume(cache_root) -> dict:
    dirs = find_snapshot_dirs(cache_root)
    results = [verify_dir(d) for d in dirs]
    return {"cache_root": str(cache_root), "snapshot_dirs": dirs,
            "results": [{k: r[k] for k in ("dir", "mismatches", "extra_files", "passed")} for r in results],
            "passed": bool(dirs) and all(r["passed"] for r in results)}


if __name__ == "__main__":
    print(json.dumps({"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                      "manifest_sha256": MODEL_MANIFEST_SHA256}, indent=2))
