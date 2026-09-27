"""
Offline tests for the Stage3C component-profile tooling (no GPU, no torch, no
pytest required: `python benchmarks/mlsys2027/test_stage3c_profile_analysis.py`;
also collected by pytest).

Covers: profile-record parsing (valid / missing / forbidden / malformed),
aggregation, HOST/GPU attribution and percentage recomputation, the
pre-registered classification rule, per-request attribution of engine log lines,
and static checks of the engine-side hook (flag default OFF, patch targets exist
as call-time lookups in the frozen sources, triton_attn.py equal to the
benchmarked version modulo the profiling scope, tile32 unchanged, protected paths).
"""

from __future__ import annotations

import ast
import copy
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage3c_profile_analysis as pa  # noqa: E402

ROOT = HERE.parent.parent
OPS = ROOT / "vllm-kvquant" / "vllm" / "v1" / "attention" / "ops"
PROFILE_SRC = OPS / "rabit_kv2_stage3c_profile.py"
RABIT_SRC = OPS / "rabit_kv2.py"
TILE32_SRC = OPS / "rabit_kv2_stage3c_tile32.py"


# ------------------------------------------------------------------ synthetic records
def make_record(impl="reference", q=32, ctx=16384, recent_only=True, span_extra=0.5):
    """Engine-consistent v2 record: HOST window tree with exact exclusive arithmetic; GPU leaf pairs."""
    tile = impl == "tile32"
    n_open = q - 1 if recent_only else q
    calls = {"chunk_plan": 1, "apply_step": q, "closed_page": 1 if tile else q, "reduce": 1 if tile else q,
             "tail_prep.k_stats_codes": n_open, "tail_prep.k3_pack": n_open, "tail_prep.meta8g64": 4 * n_open,
             "tail_partial.open_recent": n_open, "tail": q, "tail_prep": n_open}
    if recent_only:
        calls["tail_partial.recent_only"] = 1
    if not tile:
        calls["reference_attention_call"] = q
    excl = {"chunk_plan": 0.9, "apply_step": 0.02 * q, "closed_page": 0.01 * q, "reduce": 0.01 * q,
            "tail_prep.k_stats_codes": 0.012 * q, "tail_prep.k3_pack": 0.012 * q, "tail_prep.meta8g64": 0.05 * q,
            "tail_partial.open_recent": 0.012 * q, "tail_partial.recent_only": 0.012, "tail": 0.01 * q,
            "tail_prep": 0.02 * q, "reference_attention_call": 0.03 * q}
    win = {}

    def build(key):
        kids = [k for k in calls if pa.parent_of(k, impl) == key]
        for k in kids:
            build(k)
        cw = sum(win[k]["inclusive_ms"] + win[k]["overhead_ms"] for k in kids)
        win[key] = {"calls": calls[key], "inclusive_ms": excl[key] + cw, "overhead_ms": 0.003 * calls[key],
                    "children_window_ms": cw, "exclusive_ms": excl[key]}

    tops = [k for k in calls if pa.parent_of(k, impl) == "chunk"]
    for k in tops:
        build(k)
    cw = sum(win[k]["inclusive_ms"] + win[k]["overhead_ms"] for k in tops)
    wall = cw + 0.05 * q
    win["chunk"] = {"calls": 1, "inclusive_ms": wall, "overhead_ms": 0.0, "children_window_ms": cw,
                    "exclusive_ms": wall - cw}
    gpu_ms = {"chunk_plan": 0.4, "closed_page": 0.1 * q, "reduce": 0.004 * q, "tail_prep.k_stats_codes": 0.004 * q,
              "tail_prep.k3_pack": 0.003 * q, "tail_prep.meta8g64": 0.012 * q, "tail_partial.open_recent": 0.01 * q,
              "tail_partial.recent_only": 0.005}
    leaves = {k: {"calls": calls[k], "gpu_ms": gpu_ms[k]} for k in pa.GPU_LEAVES if k in calls}
    span = sum(g["gpu_ms"] for g in leaves.values()) + span_extra * q
    return {"schema": pa.SCHEMA, "impl": impl, "q_len": q, "context_len": ctx, "gpu_nested": 0,
            "host": {"wall_ms": wall, "windows": win}, "gpu": {"span_ms": span, "leaves": leaves}}


def dumps(r):
    return json.dumps(r, sort_keys=True, separators=(",", ":"))


def expect_error(r_or_payload, fragment=""):
    payload = r_or_payload if isinstance(r_or_payload, str) else dumps(r_or_payload)
    try:
        pa.parse_record(payload)
    except pa.ProfileError as e:
        assert fragment in str(e), (fragment, str(e))
        return
    raise AssertionError(f"expected ProfileError ({fragment})")


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _reprop(r):
    """Recompute parents / root / wall after an edit so only the intended inconsistency remains."""
    win, impl = r["host"]["windows"], r["impl"]
    for p in ("tail_prep", "tail", "reference_attention_call", "chunk"):
        if p in win:
            cw = sum(w["inclusive_ms"] + w["overhead_ms"] for k, w in win.items() if pa.parent_of(k, impl) == p)
            win[p]["children_window_ms"] = cw
            if p == "chunk":
                win[p]["exclusive_ms"] = win[p]["inclusive_ms"] - cw
            else:
                win[p]["inclusive_ms"] = win[p]["exclusive_ms"] + cw
    return r


# ------------------------------------------------------------------ parser tests
def test_valid_records_parse_for_both_impls():
    for impl in pa.IMPLS:
        for q in (32, 512, 2048):
            for ro in (True, False):
                r = pa.parse_record(dumps(make_record(impl, q, recent_only=ro)))
                assert r["impl"] == impl and r["q_len"] == q


def test_optional_components_may_be_absent():
    att = pa.attribute(pa.aggregate([pa.parse_record(dumps(make_record("tile32", 32, recent_only=False)))]), 1)
    assert att["call_counts"].get("tail_partial.recent_only") is None
    assert att["GPU"]["components_ms"]["tail_partial_gpu"] > 0


def test_missing_required_category_rejected():
    for impl, key in (("reference", "tail_prep"), ("reference", "reduce"), ("reference", "reference_attention_call"),
                      ("tile32", "closed_page"), ("tile32", "apply_step"), ("tile32", "chunk")):
        r = make_record(impl, 32)
        del r["host"]["windows"][key]
        expect_error(r, "missing")


def test_forbidden_category_rejected():
    r = make_record("tile32", 32)
    r["host"]["windows"]["reference_attention_call"] = make_record("reference", 32)["host"]["windows"][
        "reference_attention_call"]
    expect_error(r, "not allowed")


def test_missing_or_mismatched_events_rejected():
    def mk():
        return make_record("reference", 32)

    r = mk(); del r["gpu"]["leaves"]["reduce"]; expect_error(r, "event pair missing")  # noqa: E702
    r = mk(); r["gpu"]["leaves"]["closed_page"]["calls"] -= 1; expect_error(r, "event pairs for")  # noqa: E702
    r = mk(); r["gpu"]["leaves"]["apply_step"] = {"calls": 32, "gpu_ms": 0.1}; expect_error(r, "unknown gpu")  # noqa
    r = mk(); del r["gpu"]["leaves"]["closed_page"]["gpu_ms"]; expect_error(r, "missing field")  # noqa: E702
    r = mk(); r["gpu"]["span_ms"] = 0.1; expect_error(r, "exceeds the Stage3C span")  # noqa: E702
    r = mk(); r["gpu_nested"] = 1; expect_error(r, "nested GPU")  # noqa: E702
    r = mk(); del r["gpu"]["span_ms"]; expect_error(r, "span_ms")  # noqa: E702
    r = mk(); del r["host"]["wall_ms"]; expect_error(r, "wall_ms")  # noqa: E702


def test_negative_exclusive_rejected_not_clamped():
    r = make_record("reference", 32)
    w = r["host"]["windows"]["tail"]
    w["children_window_ms"] = w["inclusive_ms"] + 0.5  # nested timers inconsistent
    w["exclusive_ms"] = w["inclusive_ms"] - w["children_window_ms"]
    expect_error(r, "negative exclusive")


def test_malformed_records_rejected():
    good = make_record("reference", 32)
    cases = []

    def mut(fn, frag, reprop=False):
        r = copy.deepcopy(good)
        fn(r)
        cases.append((_reprop(r) if reprop else r, frag))

    def W(r, k):
        return r["host"]["windows"][k]

    mut(lambda r: r.__setitem__("schema", "rabit2_stage3c_component_profile/v1"), "schema")
    mut(lambda r: r.__setitem__("impl", "tile64"), "impl")
    mut(lambda r: r.__setitem__("q_len", 1), "q_len")
    mut(lambda r: r["gpu"]["leaves"]["reduce"].__setitem__("gpu_ms", -1.0), "negative")
    mut(lambda r: r["gpu"]["leaves"]["reduce"].__setitem__("gpu_ms", float("nan")), "finite")
    mut(lambda r: W(r, "reduce").__setitem__("calls", 0), "calls")
    mut(lambda r: W(r, "reduce").__setitem__("calls", 1.5), "calls")
    mut(lambda r: W(r, "reduce").__delitem__("overhead_ms"), "missing field")
    mut(lambda r: r["host"]["windows"].__setitem__("mystery", W(r, "reduce")), "unknown host window")
    mut(lambda r: W(r, "tail").__setitem__("exclusive_ms", W(r, "tail")["exclusive_ms"] + 1.0), "exclusive_ms !=")
    mut(lambda r: W(r, "reduce").__setitem__("children_window_ms", 1.0), "exclusive_ms !=")
    mut(lambda r: W(r, "chunk").__setitem__("calls", 2), "root")
    mut(lambda r: W(r, "chunk").__setitem__("overhead_ms", 0.1), "root")
    mut(lambda r: r["host"].__setitem__("windows", []), "windows")
    mut(lambda r: r.__delitem__("gpu"), "domain missing")

    def leaf_children(r):  # a leaf with instrumented children (all other arithmetic kept consistent)
        w = W(r, "reduce")
        w["children_window_ms"] = 0.5
        w["inclusive_ms"] += 0.5
    mut(leaf_children, "leaf window with instrumented children", reprop=True)

    def parent_children_mismatch(r):  # parent's children_window no longer equals its children's windows
        w = W(r, "tail")
        w["children_window_ms"] += 5.0
        w["inclusive_ms"] += 5.0
    mut(parent_children_mismatch, "children_window_ms != sum")

    def wall_mismatch(r):
        r["host"]["wall_ms"] += 1.0
    mut(wall_mismatch, "root")
    for r, frag in cases:
        expect_error(r, frag)
    expect_error("{not json", "not JSON")
    expect_error("[1, 2]", "schema")


def test_find_records_rejects_trailing_garbage_and_truncation():
    good = dumps(make_record())
    line = f"(EngineCore pid=1) INFO 09-27 00:00:00 [rabit_kv2_stage3c_profile.py:1] {pa.TAG}={good}"
    assert pa.find_records([line, "unrelated"]) == [good]
    assert pa.find_records([line + "\r\n"]) == [good]
    for bad in (line + " extra", line[:-40], f"{pa.TAG} {good}"):
        try:
            recs = pa.find_records([bad])
            pa.parse_record(recs[0])
        except pa.ProfileError:
            continue
        raise AssertionError(f"accepted malformed line: {bad[-60:]!r}")


def test_aggregate_rejects_mixed_or_empty():
    a, b = pa.parse_record(dumps(make_record(q=32))), pa.parse_record(dumps(make_record(q=512)))
    for recs in ([], [a, b], [a, pa.parse_record(dumps(make_record("tile32", 32)))]):
        try:
            pa.aggregate(recs)
        except pa.ProfileError:
            continue
        raise AssertionError("aggregate accepted invalid input")


# ------------------------------------------------------------------ attribution / percentages
def _walk_keys(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(x, list):
        for v in x:
            yield from _walk_keys(v)


def test_domains_separate_and_residuals_recompute():
    for impl in pa.IMPLS:
        recs = [pa.parse_record(dumps(make_record(impl, 512))) for _ in range(32)]
        agg = pa.aggregate(recs)
        att = pa.attribute(agg, 32)
        G, Hd = att["GPU"], att["HOST"]
        assert close(G["total_stage3c_gpu_span_ms"], sum(r["gpu"]["span_ms"] for r in recs))
        assert close(Hd["total_stage3c_host_wall_ms"], sum(r["host"]["wall_ms"] for r in recs))
        assert close(G["unattributed_gpu_ms"], G["total_stage3c_gpu_span_ms"] - sum(G["components_ms"].values()))
        assert close(Hd["unattributed_host_ms"], Hd["total_stage3c_host_wall_ms"] - sum(Hd["components_ms"].values()))
        # HOST residual == root exclusive; GPU residual == span - all (never nested) event leaves
        assert close(Hd["unattributed_host_ms"], agg["host_windows"]["chunk"]["exclusive_ms"])
        assert close(G["unattributed_gpu_ms"],
                     agg["gpu_span_ms"] - sum(v["gpu_ms"] for v in agg["gpu_leaves"].values()))
        assert G["unattributed_gpu_ms"] > 0 and Hd["unattributed_host_ms"] > 0  # reported, not forced to zero
        for k, v in G["components_ms"].items():
            assert close(G["share_of_gpu_span"][k], v / G["total_stage3c_gpu_span_ms"])
        for k, v in Hd["components_ms"].items():
            assert close(Hd["share_of_host_wall"][k], v / Hd["total_stage3c_host_wall_ms"])
        assert close(G["share_of_gpu_span"]["unattributed_gpu"],
                     G["unattributed_gpu_ms"] / G["total_stage3c_gpu_span_ms"])
        assert close(sum(G["share_of_gpu_span"].values()), 1.0)
        assert close(sum(Hd["share_of_host_wall"].values()), 1.0)
        assert close(G["groups_ms_display_only"]["tail_prep_gpu"], G["components_ms"]["k_stats_codes_gpu"]
                     + G["components_ms"]["k3_pack_gpu"] + G["components_ms"]["meta8g64_gpu"])
        assert "tail_prep_gpu" not in G["sum_components"]
        assert att["records"] == att["records_expected"] == 32


def test_no_combined_host_gpu_metric_anywhere():
    att = pa.attribute(pa.aggregate([pa.parse_record(dumps(make_record("reference", 512)))]), 1)
    out = {**att, "classification": pa.classify(att)}
    assert set(att) == {"impl", "q_len", "context_len", "records", "records_expected", "GPU", "HOST", "call_counts",
                        "combined_boundaries", "diagnostic_only"}
    banned = ("serialized", "attributed", "host+gpu", "host_gpu", "gpu_host", "combined_share", "total_ms")
    assert not [k for k in _walk_keys(out) if any(b in k.lower().replace("unattributed", "") for b in banned)]
    assert all(k.endswith("_gpu") for k in att["GPU"]["components_ms"])
    assert all("gpu" not in k for k in att["HOST"]["components_ms"])
    # classification shares are single-domain sums only
    assert set(out["classification"]) >= {"gpu_rule", "host_rule"}


def test_component_sets_mutually_exclusive():
    used = [leaf for leaves in pa.GPU_COMPONENTS.values() for leaf in leaves]
    assert sorted(used) == sorted(pa.GPU_LEAVES) and len(used) == len(set(used))
    ex = [k for parts in pa.HOST_COMPONENTS.values() for k, f in parts if f == "exclusive_ms"]
    ov = [k for parts in pa.HOST_COMPONENTS.values() for k, f in parts if f == "overhead_ms"]
    non_root = sorted(set(pa.HOST_KNOWN) - {pa.ROOT})
    assert sorted(ex) == non_root and sorted(ov) == non_root  # each used exactly once; wrappers never inclusive
    assert all(f in ("exclusive_ms", "overhead_ms") for parts in pa.HOST_COMPONENTS.values() for _, f in parts)


def test_nested_wrappers_not_double_counted():
    r = pa.parse_record(dumps(make_record("reference", 64)))
    att = pa.attribute(pa.aggregate([r]), 1)
    w = r["host"]["windows"]
    assert close(att["HOST"]["components_ms"]["wrapper_exclusive_host.tail"], w["tail"]["exclusive_ms"])
    assert w["tail"]["inclusive_ms"] >= w["tail"]["exclusive_ms"] + w["tail_prep"]["inclusive_ms"]
    inclusive_sum = sum(v["inclusive_ms"] for k, v in w.items() if k != "chunk")
    assert inclusive_sum > r["host"]["wall_ms"]  # inclusive counting WOULD double count
    assert sum(att["HOST"]["components_ms"].values()) <= r["host"]["wall_ms"] + 1e-9


# ------------------------------------------------------------------ classification rule
def _att(gpu, host):
    g = {k: 0.0 for k in pa.GPU_COMPONENTS} | {"unattributed_gpu": 0.0} | gpu
    h = {k: 0.0 for k in pa.HOST_COMPONENTS} | {"unattributed_host": 0.0} | host
    return {"GPU": {"share_of_gpu_span": g}, "HOST": {"share_of_host_wall": h}}


def test_classification_rules_per_domain():
    c = pa.classify(_att({"k_stats_codes_gpu": 0.3, "meta8g64_gpu": 0.25}, {"apply_step_host": 0.6}))
    assert c["gpu_rule"] == "tail_prep_gpu" and c["host_rule"] == "python_state_exposure_host"
    assert c["gpu_action"] == "batch/fuse tail preparation first"
    c = pa.classify(_att({"tail_partial_gpu": 0.3, "reduce_gpu": 0.25},
                         {"launch_host.tail_prep": 0.3, "launch_host.reduce": 0.25}))
    assert c["gpu_rule"] == "tail_partial+reduce_gpu" and c["host_rule"] == "per_query_launch_host"
    c = pa.classify(_att({"closed_page_gpu": 0.4, "unattributed_gpu": 0.3}, {"unattributed_host": 0.4}))
    assert c["gpu_rule"] == "none" and c["host_rule"] == "none"
    c = pa.classify(_att({"k3_pack_gpu": 0.2, "tail_partial_gpu": 0.2, "reduce_gpu": 0.2}, {}))
    assert c["several_per_query_large"] and c["several_per_query_action"]


# ------------------------------------------------------------------ runner-side attribution of log lines
def test_profile_lines_attributed_to_open_request():
    import run_stage3c_profile_diagnostic as rp

    rec = dumps(make_record())
    pl = f"(EngineCore pid=9) INFO x [rabit_kv2_stage3c_profile.py:1] {pa.TAG}={rec}"
    lines = [pl,  # before any request -> outside
             'S3C_POINT_BEGIN={"i": 0, "role": "conditioning"}', pl, pl, 'S3C_POINT={"i": 0}',
             'S3C_POINT_BEGIN={"i": 1, "role": "measured"}', pl, 'S3C_POINT={"i": 1}']
    by = rp.profile_by_request(lines)
    assert len(by["outside"]) == 1 and len(by[0]) == 2 and len(by[1]) == 1
    try:
        rp.profile_by_request(['S3C_POINT_BEGIN={"i": 0}', pl + " junk"])
    except pa.ProfileError:
        pass
    else:
        raise AssertionError("malformed profile line accepted")


# ------------------------------------------------------------------ static engine-side checks
def _tree(p):
    return ast.parse(p.read_text(encoding="utf-8"))


def _fn(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_profile_flag_defaults_off_and_scope_is_noop_when_off():
    t = _tree(PROFILE_SRC)
    env = next(n.value.value for n in t.body if isinstance(n, ast.Assign)
               and getattr(n.targets[0], "id", None) == "COMPONENT_PROFILE_ENV")
    legacy = next(n.value.value for n in _tree(TILE32_SRC).body if isinstance(n, ast.Assign)
                  and getattr(n.targets[0], "id", None) == "STAGE3C_PROFILE_ENV")
    assert env == "VLLM_RABIT2_STAGE3C_COMPONENT_PROFILE" and env != legacy
    flag = _fn(t, "rabit2_stage3c_component_profiling")
    assert ast.unparse(flag.body[0]) == "return os.environ.get(COMPONENT_PROFILE_ENV, '0') == '1'"
    scope = _fn(t, "rabit2_stage3c_profile_scope")
    first = scope.body[1] if isinstance(scope.body[0], ast.Expr) else scope.body[0]
    assert ast.unparse(first) == ("if q_len <= 1 or not rabit2_stage3c_component_profiling():\n"
                                  "    return _NULL_SCOPE")
    null = next(n for n in t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "_NULL_SCOPE")
    assert ast.unparse(null.value) == "contextlib.nullcontext()"
    # Nothing at import time patches anything: every setattr lives inside _ChunkProfile methods.
    cls = next(n for n in t.body if isinstance(n, ast.ClassDef) and n.name == "_ChunkProfile")
    in_cls = {id(n) for n in ast.walk(cls)}
    for n in ast.walk(t):
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "setattr":
            assert id(n) in in_cls


def test_profiler_has_no_sleep_and_events_sync_only_inside_scope():
    src = PROFILE_SRC.read_text(encoding="utf-8")
    t = ast.parse(src)
    assert "_sleep" not in src
    calls = [n for n in ast.walk(t) if isinstance(n, ast.Call)]
    syncs = [n for n in calls if ast.unparse(n.func) == "torch.cuda.synchronize"]
    events = [n for n in calls if ast.unparse(n.func) == "torch.cuda.Event"]
    cls = next(n for n in t.body if isinstance(n, ast.ClassDef) and n.name == "_ChunkProfile")
    exit_fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__exit__")
    assert len(syncs) == 1 and syncs[0] in list(ast.walk(exit_fn))  # exactly one synchronize, at scope exit
    ev_fn = _fn(t, "_event")
    assert len(events) == 1 and events[0] in list(ast.walk(ev_fn))
    # _event() is only called from _ChunkProfile methods (never at import time or on the OFF path)
    users = [n for n in calls if ast.unparse(n.func) == "_event"]
    assert users and all(u in list(ast.walk(cls)) for u in users)
    pool = next(n for n in t.body if isinstance(n, ast.AnnAssign) and ast.unparse(n.target) == "_EVENT_POOL")
    assert ast.unparse(pool.value) == "[]"


def _top_names(tree):
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            names.update(t.id for t in n.targets if isinstance(t, ast.Name))
    return names


def test_patch_targets_exist_and_are_call_time_lookups():
    t = _tree(PROFILE_SRC)
    tables = {}
    for n in t.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "").startswith("_") and \
                isinstance(n.value, ast.Tuple) and n.targets[0].id in ("_RABIT_FUNCS", "_RABIT_KERNELS",
                                                                       "_TILE32_KERNELS", "_CHUNK_PLAN_METHODS"):
            tables[n.targets[0].id] = ast.literal_eval(n.value)
    rabit_src, t32_src = RABIT_SRC.read_text(encoding="utf-8"), TILE32_SRC.read_text(encoding="utf-8")
    rabit_names, t32_names = _top_names(ast.parse(rabit_src)), _top_names(ast.parse(t32_src))
    for name, _ in tables["_RABIT_FUNCS"]:
        assert name in rabit_names, name
    for name, _ in tables["_RABIT_KERNELS"]:
        assert name in rabit_names and f"{name}[" in rabit_src, name  # launched as NAME[grid](...)
    for name, _ in tables["_TILE32_KERNELS"]:
        assert name in t32_names and f"{name}[" in t32_src, name
    cp = next(n for n in ast.parse(rabit_src).body if isinstance(n, ast.ClassDef) and n.name == "Rabit2CausalChunkPlan")
    assert {m.name for m in cp.body if isinstance(m, ast.FunctionDef)} >= {"__init__", "apply_step"}
    # Final module bindings that the Stage3C path resolves at call time.
    last = {}
    for n in ast.parse(rabit_src).body:
        if isinstance(n, ast.Assign):
            for tg in n.targets:
                if isinstance(tg, ast.Name):
                    last[tg.id] = ast.unparse(n.value)
                elif isinstance(tg, ast.Attribute) and ast.unparse(tg) == "Rabit2CausalChunkPlan.__init__":
                    last["Rabit2CausalChunkPlan.__init__"] = ast.unparse(n.value)
    assert last["_rabit2_stage4b1_exactmeta_emit_tail_partial"] == "_rabit2_stage4d3_4_emit_tail_partial"
    assert last["rabit2_online_decode_attention_triton"] == "rabit2_online_decode_attention_triton_stage4b3_gqa4"
    assert last["Rabit2CausalChunkPlan.__init__"] == "_rabit2_stage4d2_old_chunkplan_init"
    # tile32 resolves the tail emitter and ChunkPlan through the module at call time (patchable).
    fwd = ast.unparse(_fn(ast.parse(t32_src), "rabit2_stage3c_forward_tile32"))
    assert "_r._rabit2_stage4b1_exactmeta_emit_tail_partial" in fwd and "_r.Rabit2CausalChunkPlan(" in fwd


def test_triton_attn_equals_benchmarked_version_modulo_scope():
    import run_stage3c_profile_diagnostic as rp

    base_ref = json.loads(rp.BENCH_MANIFEST.read_text(encoding="utf-8"))["provenance"]["git_head"]
    base = rp._git_show(base_ref, rp.TRITON_ATTN)
    cur = rp.TRITON_ATTN.read_text(encoding="utf-8")
    assert rp.triton_attn_unwrapped_equals(base, cur)
    assert not rp.triton_attn_unwrapped_equals(base, cur.replace("softmax_scale=self.scale,\n",
                                                                 "softmax_scale=1.0,\n", 1)
                                               .replace("softmax_scale=self.scale,\r\n", "softmax_scale=1.0,\r\n", 1))
    assert not rp.triton_attn_unwrapped_equals(base, cur.replace("with rabit2_stage3c_profile_scope(q_len, context_len)",
                                                                 "with rabit2_stage3c_profile_scope(q_len, 0)"))
    assert not rp.triton_attn_unwrapped_equals(base, base)  # the hook must be present exactly once


def test_frozen_sources_and_tile32_unchanged():
    import run_stage3c_profile_diagnostic as rp

    prov = json.loads(rp.BENCH_MANIFEST.read_text(encoding="utf-8"))["provenance"]
    assert rp.sha256(rp.RABIT_KV2) == rp.EXPECTED_RABIT_SHA256_LF == prov["rabit_kv2_sha256"]
    assert rp.sha256(rp.TILE32_MODULE) == prov["tile32_module_sha256"]
    assert rp.sha256(rp.TILE32_TESTS) == prov["tile32_tests_sha256"]
    for f in (rp.GATE, rp.WATCHDOG, rp.BENCH_WORKER, rp.BENCH_MODAL_APP):
        assert rp.run_git("status", "--short", "--", rp.rel(f)) == "", f


def test_equivalence_and_protected_paths():
    import run_stage3c_profile_diagnostic as rp

    eq = rp.verify_equivalence(json.loads(rp.BENCH_MANIFEST.read_text(encoding="utf-8")))
    assert eq["triton_attn_equal_benchmarked_modulo_profile_scope"] and eq["tile32_module_and_tests_unchanged"]
    rp.assert_protected_paths_clean("offline test")
    real = rp.run_git
    try:
        rp.run_git = lambda *a: " M results/mlsys2027/diagnostics/stage3c_tile32_benchmark/summary.json"
        try:
            rp.assert_protected_paths_clean("tamper")
        except RuntimeError:
            pass
        else:
            raise AssertionError("protected-path change not detected")
    finally:
        rp.run_git = real
    assert rp.rel(rp.BENCH_DIR) in [rp.rel(p) for p in rp.PROTECTED_PATHS]
    assert rp.rel(rp.RABIT_KV2) in [rp.rel(p) for p in rp.PROTECTED_PATHS]


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
