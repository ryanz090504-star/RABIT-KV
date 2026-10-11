"""
CPU-ONLY Modal app for the canonical-quality-v2 parity tests (no GPU, no model). Self-contained: no sibling-module
import at module level; repository paths are resolved ONLY in local context (modal.is_local()); remotely the needed
files are at /repo/<repo-relative path>.

Parity Attempt 3 TRANSPORT fix: the remote function builds the frozen suite's result dict, normalizes the one known
non-native field EXPLICITLY (torch_version = str(torch.__version__); python_version = str(...)), checks every value is
an EXACT JSON-native built-in (dict / list / str / int / float / bool / None -- no subclasses, sets, tuples, tensors or
custom objects; any violation FAILS), serializes with STRICT json.dumps (no default=), prints one short
CANONICAL_PARITY_REMOTE backup line, and RETURNS ONLY that plain str. No torch object crosses the RPC boundary.
The local entrypoint verifies isinstance(payload, str), parses, validates the schema, writes the exact UTF-8 text,
hashes it and re-reads / re-parses it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import modal

REMOTE_REPO = "/repo"
FILES = ["benchmarks/mlsys2027/canonical_rabit_quality.py", "benchmarks/mlsys2027/canonical_quality_parity_tests.py",
         "vllm-kvquant/vllm/v1/attention/ops/kvquant_k3.py", "benchmarks/quality/hotpotqa.py"]
RESULT_PATH_ENV = "CANONICAL_PARITY_RESULT_PATH"
GEOMETRY_KEYS = ("qwen2_5_7b", "llama3_1_8b")
SECTION_KEYS = ("T1_full_state", "T2_sequential_aging", "T3_hf_cache", "T4_old_harness", "summary")
JSON_NATIVE = (dict, list, str, int, float, bool, type(None))

app = modal.App("rabit-kv-canonical-quality-v2-parity")
image = (modal.Image.debian_slim(python_version="3.11")
         .run_commands("python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu")
         .pip_install("transformers==4.48.2"))
if modal.is_local():  # local repository layout is consulted ONLY when building the app locally
    from pathlib import Path

    _local_root = Path(__file__).resolve().parents[2]
    for _rel in FILES:
        image = image.add_local_file(str(_local_root / _rel), f"{REMOTE_REPO}/{_rel}", copy=True)


def assert_json_native(obj, path: str = "$") -> None:
    """EXACT built-in JSON types only (type(x) is ..., so str / int subclasses such as TorchVersion are rejected)."""
    t = type(obj)
    if t not in JSON_NATIVE:
        raise TypeError(f"non-JSON-native value at {path}: {t.__module__}.{t.__qualname__}")
    if t is dict:
        for k, v in obj.items():
            if type(k) is not str:
                raise TypeError(f"non-str key at {path}: {type(k).__qualname__}")
            assert_json_native(v, f"{path}.{k}")
    elif t is list:
        for i, v in enumerate(obj):
            assert_json_native(v, f"{path}[{i}]")
    elif t is float and obj != obj:  # NaN is not valid strict JSON
        raise TypeError(f"NaN at {path}")


def serialize_result(res: dict) -> str:
    """Strict serializer used remotely: validate exact types, then plain json.dumps (no default=)."""
    assert_json_native(res)
    return json.dumps(res, ensure_ascii=False, allow_nan=False)


def validate_payload(payload) -> dict:
    """Local-side checks on the RPC payload; returns the parsed result."""
    if type(payload) is not str:
        raise TypeError(f"payload is {type(payload).__qualname__}, expected str")
    res = json.loads(payload)
    for k in ("oracle", "lengths", "prefills", "geometries", "passed", "negative_control_passed", "torch_version",
              "python_version"):
        if k not in res:
            raise ValueError(f"schema: missing {k}")
    if sorted(res["geometries"]) != sorted(GEOMETRY_KEYS):
        raise ValueError(f"schema: geometries {sorted(res['geometries'])}")
    for g in GEOMETRY_KEYS:
        if sorted(res["geometries"][g]) != sorted(SECTION_KEYS):
            raise ValueError(f"schema: sections of {g}")
    assert_json_native(res)
    return res


def short_summary(res: dict) -> dict:
    return {"completed": True, "passed": res["passed"], "negative_control_passed": res["negative_control_passed"],
            **{f"{g}_summary": res["geometries"][g]["summary"] for g in GEOMETRY_KEYS}}


@app.function(image=image, cpu=8, memory=16384, timeout=1200)
def parity() -> str:
    from pathlib import Path

    import torch

    sys.path.insert(0, f"{REMOTE_REPO}/benchmarks/mlsys2027")
    import canonical_quality_parity_tests as t  # noqa: PLC0415  (frozen suite, unchanged)

    res = t.run(Path(REMOTE_REPO))
    # explicit normalization of the known non-native metadata fields
    res["torch_version"] = str(torch.__version__)
    res["python_version"] = str(sys.version.split()[0])
    payload = serialize_result(res)  # raises (attempt fails) on any remaining non-JSON-native value
    print("CANONICAL_PARITY_REMOTE=" + json.dumps(short_summary(res)), flush=True)
    return payload


@app.local_entrypoint()
def main():
    payload = parity.remote()
    res = validate_payload(payload)
    data = payload.encode("utf-8")
    path = os.environ[RESULT_PATH_ENV]
    with open(path, "wb") as fh:
        fh.write(data)
    with open(path, "rb") as fh:
        back = fh.read()
    if back != data or json.loads(back.decode("utf-8")) != res:
        raise SystemExit("local result file does not round-trip")
    print("CANONICAL_PARITY_LOCAL=" + json.dumps({"payload_type": "str", "bytes": len(data),
                                                  "sha256": hashlib.sha256(data).hexdigest(), **short_summary(res)}),
          flush=True)
    if not (res["passed"] and res["negative_control_passed"]):
        raise SystemExit(1)
