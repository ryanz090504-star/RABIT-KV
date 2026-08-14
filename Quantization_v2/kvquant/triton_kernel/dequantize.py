"""Placeholder for future Triton on-demand dequantization kernels.

The current codebase does not implement in-register low-bit dequantization.
"""

from __future__ import annotations

_KERNELS_REGISTERED = False


def ensure_kernels() -> None:
    """No-op placeholder until real Triton JIT kernels are implemented."""
    global _KERNELS_REGISTERED
    if _KERNELS_REGISTERED:
        return
    try:
        import triton  # noqa: F401
    except ImportError:
        return
    # Kernel registration happens first time run_packed_attention is called.
    _KERNELS_REGISTERED = True
