"""Static readiness checks for an editable vLLM fork.

The checks are intentionally conservative. They do not prove that the fork is
correct or fast; they catch missing integration surfaces before a deploy
benchmark is allowed to produce paper evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


SOURCE_SUFFIXES = {".py", ".pyi", ".cu", ".cuh", ".cc", ".cpp", ".h", ".hpp", ".jinja", ".triton"}
SKIP_DIRS = {
    ".git",
    "__pycache__",
    "build",
    "dist",
    ".venv",
    "venv",
    ".eggs",
    ".pytest_cache",
    "tests",
}


def check_vllm_fork(root: str | Path, *, max_files: int = 4000) -> dict[str, Any]:
    """Return readiness checks for a vLLM fork expected to implement kvquant_k3."""
    root_path = Path(root).expanduser()
    checks: list[dict[str, Any]] = []

    exists = root_path.is_dir()
    checks.append(_check("root_exists", exists, f"vLLM root exists: {root_path}"))
    if not exists:
        return _summary(checks, scanned_files=0)

    files = list(_iter_source_files(root_path, max_files=max_files))
    text_index = [(path, _read_text(path)) for path in files]

    source_tree = (root_path / "pyproject.toml").is_file() and (root_path / "vllm").is_dir()
    checks.append(_check("vllm_source_tree", source_tree, "Root looks like a vLLM source checkout"))

    checks.append(_pattern_check(
        "kvquant_dtype_registered",
        text_index,
        required=("kvquant_k3",),
        message="Fork registers a kvquant_k3 cache dtype string",
    ))
    checks.append(_pattern_check(
        "cache_layout_accounts_for_scales",
        text_index,
        required=("kvquant_k3",),
        any_groups=(("page_size", "block_size", "num_blocks"), ("scale", "scales", "scale_strategy")),
        message="KV cache layout accounts for packed bytes and auxiliary scale storage",
    ))
    checks.append(_pattern_check(
        "cache_write_quantizes_int3",
        text_index,
        required=("kvquant_k3", "def reshape_and_cache_kvquant_k3"),
        any_groups=(("quant", "quantize"), ("pack", "packed"), ("int3", "3")),
        message="Cache write path quantizes and packs K/V into INT3 storage",
    ))
    checks.append(_pattern_check(
        "attention_reads_packed_int3",
        text_index,
        required=("kvquant_k3", "def unified_attention_kvquant_k3"),
        any_groups=(("unpack", "dequant"), ("int3", "3")),
        message="Attention backend exposes a packed INT3 read/dequantization entrypoint",
    ))
    checks.append(_forbidden_pattern_check(
        "attention_read_kernel_implemented",
        text_index,
        required=("def unified_attention_kvquant_k3",),
        forbidden=("notimplementederror",),
        message="Packed INT3 attention read entrypoint is implemented, not a stub",
    ))
    checks.append(_pattern_check(
        "bench_serve_can_select_dtype",
        text_index,
        required=("kvquant_k3",),
        any_groups=(("bench", "serve"), ("kv_cache_dtype", "kv-cache-dtype")),
        message="Serving benchmark path can select kvquant_k3",
    ))

    return _summary(checks, scanned_files=len(files))


def _iter_source_files(root: Path, *, max_files: int):
    count = 0
    for path in root.rglob("*"):
        if count >= max_files:
            break
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix in SOURCE_SUFFIXES:
            count += 1
            yield path


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return ""


def _pattern_check(
    name: str,
    text_index: list[tuple[Path, str]],
    *,
    required: tuple[str, ...],
    message: str,
    any_groups: tuple[tuple[str, ...], ...] = (),
) -> dict[str, Any]:
    required_lower = tuple(item.lower() for item in required)
    for path, text in text_index:
        if not all(item in text for item in required_lower):
            continue
        if any_groups and not all(any(option.lower() in text for option in group) for group in any_groups):
            continue
        return _check(name, True, message, file=str(path))
    return _check(name, False, message)


def _forbidden_pattern_check(
    name: str,
    text_index: list[tuple[Path, str]],
    *,
    required: tuple[str, ...],
    forbidden: tuple[str, ...],
    message: str,
) -> dict[str, Any]:
    required_lower = tuple(item.lower() for item in required)
    forbidden_lower = tuple(item.lower() for item in forbidden)
    for path, text in text_index:
        if not all(item in text for item in required_lower):
            continue
        if any(item in text for item in forbidden_lower):
            return _check(name, False, message, file=str(path))
        return _check(name, True, message, file=str(path))
    return _check(name, False, message)


def _check(name: str, passed: bool, message: str, **extra: Any) -> dict[str, Any]:
    row = {
        "name": name,
        "status": "pass" if passed else "fail",
        "message": message,
    }
    row.update(extra)
    return row


def _summary(checks: list[dict[str, Any]], *, scanned_files: int) -> dict[str, Any]:
    passed = all(check["status"] == "pass" for check in checks)
    return {
        "status": "pass" if passed else "fail",
        "scanned_files": scanned_files,
        "checks": checks,
    }
