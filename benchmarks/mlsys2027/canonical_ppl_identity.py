"""
canonical-quality-v2 continuation PPL -- FROZEN identities and protocol constants (stdlib only; imported locally by the
runner / tests and inside the Modal container).

Models (both loaded with snapshot_download(model_id, revision=<immutable commit>) and verified file-by-file -- size and
SHA-256 -- against the manifest below BEFORE the model is loaded):

  qwen2_5_7b   Qwen/Qwen2.5-7B-Instruct @ 16c174980d8a1492910551634b4969e69cdc2444 -- the frozen Exp14 Model-B identity
               (exp14_model_snapshot.py, unchanged; manifest 9be52dd6...).

  llama3_1_8b  LLM-Research/Meta-Llama-3.1-8B-Instruct @ 359efdbb8af05b788a4ad4185215c6b8caa9052c.
               PROVENANCE (stated exactly; nothing guessed): the accepted Exp12 evidence did NOT record a revision or a
               file manifest. It records only `snapshot_download("LLM-Research/Meta-Llama-3.1-8B-Instruct")` and the log
               line "Downloading 18 files from LLM-Research/Meta-Llama-3.1-8B-Instruct@master" (2026-09-30). The commit
               above is DERIVED: resolved 2026-10-03 by `git ls-remote https://www.modelscope.cn/LLM-Research/
               Meta-Llama-3.1-8B-Instruct.git` -> refs/heads/master = 359efdbb...; its ModelScope commit date is
               2025-02-26 (unix 1740572854), i.e. before every Exp1-Exp12 run, and the file API lists the same 18 blobs
               for Revision=master and Revision=<commit>. The run additionally fingerprints the model + windows by
               reproducing the Exp12 BF16 per-window PPL with the legacy batched scorer (runner validity gate).

Dataset / windows (the frozen legacy protocol, benchmarks/quality/continuation_ppl.py): WikiText-2 test raw text,
non-empty stripped lines joined in blocks of 64, tokenized with add_special_tokens=False (no BOS); the first
32 x (1024 + 128) tokens split into consecutive non-overlapping windows. The raw text and each model's token pool are
pinned by SHA-256 (computed offline with the pinned tokenizer files; canonical_ppl_offline_proofs.py).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import exp14_model_snapshot as qwen  # frozen Exp14 Model-B identity (stdlib only; unchanged)

SAMPLES, CONTEXT_TOKENS, EVAL_TOKENS = 32, 1024, 128
LINE_BLOCK = 64
WIKITEXT_URL = ("https://raw.githubusercontent.com/pytorch/examples/main/"
                "word_language_model/data/wikitext-2/test.txt")
WIKITEXT_SHA256 = "d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0"
WIKITEXT_BYTES = 1256449

QWEN_MANIFEST_SHA256 = "9be52dd6573759d4d2c0878dc0889fd72d4a2fbfe8065afa1180a3a1717469d1"
LLAMA_ID = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
LLAMA_REVISION = "359efdbb8af05b788a4ad4185215c6b8caa9052c"
LLAMA_GIT_URL = "https://www.modelscope.cn/LLM-Research/Meta-Llama-3.1-8B-Instruct.git"
# (path, size in bytes, sha256) of every file at LLAMA_REVISION (ModelScope API listing; sorted by path).
LLAMA_FILES = [
    [".gitattributes", 1519, "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"],
    ["LICENSE", 7627, "64e1b2889b7892e6bbe7a7ed5bfe6ff793c61f9d584345f8f41cf9f5cb30a369"],
    ["README.md", 44044, "ed0e2e86f7a40c38b793b0a04dfd993b2c1dfb34906804f96b999f0bbd8d70e6"],
    ["USE_POLICY.md", 4691, "a568f2ebc73cec3fd74ba2afd992d4e945a8c7a9d851f9b66163aac834b7b859"],
    ["config.json", 855, "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e"],
    ["configuration.json", 48, "a78221ae93dd8977b26fcd1d106fa5ad7e7f1fd3a7d93bfd9d6fba5ed00ec4d2"],
    ["generation_config.json", 184, "189fb0c0d7fd8a527db217c0a60a0e013f0394cd8800f9697a666a9e75e5f7fd"],
    ["model-00001-of-00004.safetensors", 4976698672, "2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668"],
    ["model-00002-of-00004.safetensors", 4999802720, "09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15"],
    ["model-00003-of-00004.safetensors", 4915916176, "fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa"],
    ["model-00004-of-00004.safetensors", 1168138808, "92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b"],
    ["model.safetensors.index.json", 23950, "146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b"],
    ["original/consolidated.00.pth", 16060617592, "ab33d910f405204e5d388bc3521503584800461dc96808e287821dd451c1edac"],
    ["original/params.json", 199, "b15b6b31b2043c0400b028ecc25c8946e21d76ac260e9ac6a357ed8727c8865f"],
    ["original/tokenizer.model", 2183982, "82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55"],
    ["special_tokens_map.json", 296, "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec"],
    ["tokenizer.json", 9085657, "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4"],
    ["tokenizer_config.json", 55351, "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424"],
]
LLAMA_MANIFEST_SHA256 = "85d9cffee6980348ad1c334d71f8731f6442553535848542457b68d85d70ce89"

MODELS = {
    "llama3_1_8b": {"model_id": LLAMA_ID, "revision": LLAMA_REVISION, "files": LLAMA_FILES,
                    "manifest_sha256": LLAMA_MANIFEST_SHA256, "layers": 32, "kv_heads": 8, "head_dim": 128,
                    "revision_provenance": "derived (Exp12 recorded only @master); see module docstring",
                    # sha256 of the int64-LE token pool (first 32 x 1152 tokens), computed offline
                    "token_pool_sha256": "88dadfe6521c338ca71b1edfe5e85f08b8e2466e0884f8003b5b34a7ee3c849f"},
    "qwen2_5_7b": {"model_id": qwen.MODEL_ID, "revision": qwen.MODEL_REVISION, "files": qwen.FROZEN_FILES,
                   "manifest_sha256": QWEN_MANIFEST_SHA256, "layers": 28, "kv_heads": 4, "head_dim": 128,
                   "revision_provenance": "frozen Exp14 Model-B identity (exp14_model_snapshot.py)",
                   "token_pool_sha256": "eb424fb71baccb8b64ae53688613a30c462a54a893f82359c9e6ea5385c0f63c"},
}


def manifest_sha256(files) -> str:
    """SHA-256 of the sorted 'path<TAB>size<TAB>sha256' lines (same definition as exp14_model_snapshot)."""
    return hashlib.sha256("\n".join(f"{p}\t{s}\t{h}" for p, s, h in sorted(files)).encode()).hexdigest()


def check_constants() -> None:
    """The frozen manifests hash to the pinned values (raises otherwise)."""
    for key, m in MODELS.items():
        if manifest_sha256(m["files"]) != m["manifest_sha256"]:
            raise RuntimeError(f"{key}: file manifest does not hash to the pinned manifest_sha256")
    if qwen.MODEL_MANIFEST_SHA256 != QWEN_MANIFEST_SHA256:
        raise RuntimeError("exp14_model_snapshot manifest differs from the pinned Qwen manifest")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_dir(path, key: str) -> dict:
    """Every frozen file of MODELS[key] present with the frozen size and SHA-256; no extra non-hidden file."""
    m, root = MODELS[key], Path(path)
    mismatches = []
    for rel, size, sha in m["files"]:
        p = root / rel
        if not p.is_file():
            mismatches.append({"file": rel, "reason": "missing"})
            continue
        st = p.stat().st_size
        got = _sha256_file(p) if st == size else None
        if st != size or got != sha:
            mismatches.append({"file": rel, "reason": "size/sha256 differ", "bytes": st, "sha256": got})
    frozen = {rel for rel, _, _ in m["files"]}
    extra = sorted(str(p.relative_to(root)).replace(os.sep, "/") for p in root.rglob("*")
                   if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)
                   and str(p.relative_to(root)).replace(os.sep, "/") not in frozen)
    return {"dir": str(root), "model_id": m["model_id"], "model_revision": m["revision"],
            "manifest_sha256": m["manifest_sha256"], "files_checked": len(m["files"]), "mismatches": mismatches,
            "extra_files": extra, "passed": not mismatches and not extra}


def hardware_ok(gpus: list) -> bool:
    """Exactly one NVIDIA H100 80GB (same predicate as the Exp14 hardware-binding guard, 5d7fae2)."""
    return (len(gpus) == 1 and "H100" in gpus[0].get("name", "")
            and 79 * 1024 <= int(float(gpus[0].get("memory.total", 0))) <= 82 * 1024)
