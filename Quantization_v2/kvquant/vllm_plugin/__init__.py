"""vLLM plugin scaffold for custom kv_cache_dtype integration.

This package is not registered into vLLM yet. It documents the intended
integration points for future work and must not be used as deploy evidence.
"""

# Only import when vllm is available.
try:
    import vllm  # noqa: F401
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
