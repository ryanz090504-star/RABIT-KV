"""KV cache snapshot capture, save, and load.

Captures real K/V tensors from HuggingFace model inference for use in
deploy-latency benchmarks. The snapshots are saved as .npz files with
metadata, so the latency path can use the exact same KV distribution
that the quality path validates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class KVSnapshot:
    """A single captured K/V pair for one transformer layer at one decode step.

    Captured from the raw (unquantized) HuggingFace cache before any
    quantization is applied, so the values represent the real model's
    KV distribution per layer.
    """

    keys: np.ndarray
    """Key tensor, shape [batch, kv_heads, seq, head_dim], dtype float16/float32."""

    values: np.ndarray
    """Value tensor, same shape as keys."""

    layer_idx: int
    """Zero-based transformer layer index."""

    step: int
    """Decode step (token position) at which this snapshot was captured."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary metadata: model name, dtype, seq_len, etc."""

    @property
    def shape(self) -> tuple[int, ...]:
        return self.keys.shape

    @property
    def batch_size(self) -> int:
        return int(self.keys.shape[0])

    @property
    def kv_heads(self) -> int:
        return int(self.keys.shape[1])

    @property
    def seq_len(self) -> int:
        return int(self.keys.shape[2])

    @property
    def head_dim(self) -> int:
        return int(self.keys.shape[3])


def capture_snapshots(
    cache: object,
    step: int,
    *,
    layer_indices: list[int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[KVSnapshot]:
    """Extract K/V tensors from a HuggingFace cache at a given decode step.

    Parameters
    ----------
    cache:
        A HuggingFace ``past_key_values`` object in any supported format:
        legacy tuple, ``DynamicCache``, or list of (K,V) pairs.
    step:
        Current decode step (0-indexed token position).
    layer_indices:
        Which layers to capture.  If ``None``, captures every layer.
    metadata:
        Extra key-value pairs attached to every snapshot.

    Returns
    -------
    list[KVSnapshot]
        One snapshot per captured layer.

    Notes
    -----
    GQA models (where kv_heads < attention_heads) are handled correctly:
    the captured shapes reflect the actual KV head count.
    """
    kv_tuple = _cache_to_tuple(cache)
    if not kv_tuple:
        raise ValueError("cache is empty or has an unrecognised format")

    total_layers = len(kv_tuple)
    indices = layer_indices if layer_indices is not None else list(range(total_layers))
    _validate_layer_indices(indices, total_layers)

    snapshots: list[KVSnapshot] = []
    for idx in indices:
        if idx >= len(kv_tuple):
            continue
        entry = kv_tuple[idx]
        if not isinstance(entry, (tuple, list)) or len(entry) < 2:
            continue
        k_tensor, v_tensor = entry[0], entry[1]
        k_np = _tensor_to_numpy(k_tensor)
        v_np = _tensor_to_numpy(v_tensor)
        snapshots.append(
            KVSnapshot(
                keys=k_np,
                values=v_np,
                layer_idx=idx,
                step=step,
                metadata=dict(metadata or {}),
            )
        )
    return snapshots


def save_snapshots(
    snapshots: list[KVSnapshot],
    output_dir: str | Path,
    *,
    compress: bool = True,
) -> list[Path]:
    """Save snapshots to disk as .npz files.

    Directory structure::

        {output_dir}/
          step_{step}_layer_{layer_idx}.npz
          index.json                  # aggregate metadata

    Parameters
    ----------
    snapshots:
        One or more snapshots to persist.
    output_dir:
        Target directory (created if needed).
    compress:
        If True, use ``np.savez_compressed`` (smaller files, slightly slower).

    Returns
    -------
    list[Path]
        Paths of every written file (including the index).

    Raises
    ------
    OSError
        If the output directory cannot be created or a file cannot be written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    index_entries: list[dict[str, Any]] = []
    save_fn = np.savez_compressed if compress else np.savez

    for snap in snapshots:
        fname = f"step_{snap.step}_layer_{snap.layer_idx}.npz"
        fpath = out / fname
        save_fn(
            fpath,
            keys=snap.keys,
            values=snap.values,
        )
        saved.append(fpath)
        index_entries.append(
            {
                "file": fname,
                "layer_idx": snap.layer_idx,
                "step": snap.step,
                "shape": list(snap.shape),
                "dtype": str(snap.keys.dtype),
                "batch_size": snap.batch_size,
                "kv_heads": snap.kv_heads,
                "seq_len": snap.seq_len,
                "head_dim": snap.head_dim,
                "metadata": snap.metadata,
            }
        )

    index_path = out / "index.json"
    index_path.write_text(json.dumps(index_entries, indent=2, sort_keys=True), encoding="utf-8")
    saved.append(index_path)
    return saved


def load_snapshots(input_dir: str | Path) -> list[KVSnapshot]:
    """Load all snapshots from a directory previously written by :func:`save_snapshots`.

    Parameters
    ----------
    input_dir:
        Directory containing ``.npz`` files and an ``index.json``.

    Returns
    -------
    list[KVSnapshot]
        Snapshots in the order they appear in the index.

    Raises
    ------
    FileNotFoundError
        If ``input_dir`` does not exist or contains no index.
    ValueError
        If the index is malformed.
    """
    in_dir = Path(input_dir)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"snapshot directory not found: {in_dir}")

    index_path = in_dir / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"snapshot index not found: {index_path}")

    entries = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"snapshot index must be a JSON array, got {type(entries).__name__}")

    snapshots: list[KVSnapshot] = []
    for entry in entries:
        fname = entry.get("file")
        if not fname:
            continue
        fpath = in_dir / fname
        if not fpath.is_file():
            continue
        try:
            data = np.load(fpath, allow_pickle=False)
        except (ValueError, OSError):
            continue
        layer_idx = entry.get("layer_idx", 0)
        step_val = entry.get("step", 0)
        metadata = dict(entry.get("metadata") or {})
        snapshots.append(
            KVSnapshot(
                keys=data["keys"],
                values=data["values"],
                layer_idx=int(layer_idx),
                step=int(step_val),
                metadata=metadata,
            )
        )
    return snapshots


def load_layer_snapshot(
    input_dir: str | Path,
    layer_idx: int,
) -> KVSnapshot | None:
    """Load the snapshot for a specific layer.

    This is a convenience wrapper around :func:`load_snapshots` for the common
    case where a latency benchmark operates on a single layer's K/V pair.

    Parameters
    ----------
    input_dir:
        Directory containing ``.npz`` files.
    layer_idx:
        Zero-based layer index to load.

    Returns
    -------
    KVSnapshot | None
        The matching snapshot, or ``None`` if no snapshot exists for that layer.
    """
    for snap in load_snapshots(input_dir):
        if snap.layer_idx == layer_idx:
            return snap
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cache_to_tuple(cache: object) -> tuple[tuple[object, ...], ...] | None:
    """Convert any supported HuggingFace cache format to a legacy tuple of (K,V) pairs."""
    if cache is None:
        return None
    if isinstance(cache, tuple):
        return cache
    if isinstance(cache, list):
        return tuple(cache)
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return tuple(
            (k, v) for k, v in zip(cache.key_cache, cache.value_cache)
        )
    if hasattr(cache, "layers"):
        layers = []
        for layer in cache.layers:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is not None and value is not None:
                layers.append((key, value))
        if layers:
            return tuple(layers)
    # DynamicCache (some versions) is iterable but not subscriptable
    try:
        items = tuple(cache)
        if items:
            return items
    except TypeError:
        pass
    raise RuntimeError(f"unsupported cache type: {type(cache)}")


def _tensor_to_numpy(tensor: object) -> np.ndarray:
    """Convert a PyTorch tensor to a detached float32 NumPy array.

    Callers should handle ``ImportError`` if torch is not available.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("snapshot capture requires torch") from exc

    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().float().numpy()
    if isinstance(tensor, np.ndarray):
        return tensor.astype(np.float32)
    raise TypeError(f"unsupported tensor type: {type(tensor)}")


def _validate_layer_indices(indices: list[int], total_layers: int) -> None:
    for idx in indices:
        if idx < 0 or idx >= total_layers:
            raise ValueError(
                f"layer index {idx} is out of range [0, {total_layers})"
            )
