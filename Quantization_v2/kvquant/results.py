from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def write_jsonl(path: str | Path, rows: list[dict[str, Any]] | dict[str, Any]) -> None:
    """将结果追加写入JSONL文件（每行一个JSON对象）。

    自动创建父目录。支持传入单行或行列表。以追加模式打开，
    多次运行写入同一文件不会覆盖之前的结果。
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, dict):
        rows = [rows]
    with output.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取JSONL文件，返回字典列表。"""
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _json_safe(value: Any) -> Any:
    """将dataclass、numpy标量等转换为JSON安全的Python类型。"""
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value
