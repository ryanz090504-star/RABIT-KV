"""
Validate results/mlsys2027/multilingual_frontier/summary.json against the raw
Experiment 2 log (and, for canonical_reference only, the canonical log).

Checks:
  * raw log SHA-256 matches the log on disk;
  * every aggregate source row is verbatim the cited log line, for that method,
    and every raw aggregate value is a token of that row;
  * exactly 10 method/language aggregate rows, each with 1024 scored tokens;
  * exactly 80 per-sample rows (8 per method/language, numbered 1..8), each
    value matching its cited line; logged median PPL and avg KV MB re-derive
    from the per-sample values;
  * every compression ratio and PPL delta recomputes; computed deltas agree
    with the logged deltas; cross-language mean/worst recompute and match the
    log's summary table;
  * canonical_reference: source SHA-256 verified FIRST, then values compared
    to the canonical log, then exact equality with the rerun;
  * every number quoted in the notes matches the data, and each claim holds.

SHA-256s are over LF-normalized bytes (CRLF -> LF). Read-only; never modifies
any file. Runs locally; no Modal/GPU. Exits non-zero on any failure.

Usage:
    python benchmarks/mlsys2027/validate_exp2_summary.py
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "mlsys2027" / "multilingual_frontier"
LOG = OUT_DIR / "multilingual_ppl.log"
SUMMARY = OUT_DIR / "summary.json"
CANONICAL_LOG = ROOT / "results" / "quality" / "multilingual_ppl.log"

METHODS = ["bf16", "rabit8", "rabit4", "rabit3", "rabit2"]
LANGS = ["zh", "es"]
SAMPLES = 8
TOKENS = 1024
SAMPLE_LINE = re.compile(r"^  sample (\d+)/(\d+): PPL=([0-9.]+), KV=([0-9.]+) MB, ")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def in_row(value: float, tokens: list[str]) -> bool:
    for t in tokens:
        try:
            if float(t) == float(value):
                return True
        except ValueError:
            pass
    return False


def main() -> int:
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    fails: list[str] = []
    n = {"aggregate_rows": 0, "sample_rows": 0, "raw_values": 0, "derived_values": 0,
         "canonical_values": 0, "note_checks": 0}

    def fail(msg: str) -> None:
        fails.append(msg)

    # Raw log provenance.
    if s["raw_log"]["path"] != LOG.relative_to(ROOT).as_posix():
        fail(f"raw_log.path {s['raw_log']['path']} is not the Experiment 2 log")
    log_sha_ok = sha256(LOG) == s["raw_log"]["sha256"]
    if not log_sha_ok:
        fail("raw log SHA-256 differs from summary.json")
    log_lines = LOG.read_text(encoding="utf-8").splitlines()

    # Independent count of per-sample lines in the raw log.
    raw_sample_lines = [l for l in log_lines if SAMPLE_LINE.match(l)]
    if len(raw_sample_lines) != 2 * len(METHODS) * SAMPLES:
        fail(f"raw log has {len(raw_sample_lines)} per-sample lines, expected 80")

    # Per-language aggregate + per-sample checks.
    langs = s["languages"]
    if list(langs) != LANGS:
        fail(f"languages {list(langs)} != {LANGS}")
    for lang in LANGS:
        ms = langs[lang]["methods"]
        if list(ms) != METHODS:
            fail(f"{lang}: methods {list(ms)} != {METHODS}")
            continue
        bf = ms["bf16"]
        for m, e in ms.items():
            n["aggregate_rows"] += 1
            src = e["source"]
            actual = log_lines[src["line"] - 1].rstrip()
            if actual != src["row"]:
                fail(f"{lang}/{m}: source row differs from log line {src['line']}")
            tok = actual.split()
            if tok[0] != m or len(tok) != 7:
                fail(f"{lang}/{m}: source row is not this method's 7-column aggregate row")
            for key in ("ppl", "ppl_delta_pct_vs_bf16_logged", "median_sample_ppl",
                        "logical_kv_mb", "logged_compression", "scored_tokens"):
                n["raw_values"] += 1
                if not in_row(e[key], tok):
                    fail(f"{lang}/{m}.{key}={e[key]} not in source row")
            if e["scored_tokens"] != TOKENS:
                fail(f"{lang}/{m}: scored_tokens {e['scored_tokens']} != {TOKENS}")

            # Derived.
            n["derived_values"] += 2
            exp_ratio = round(bf["logical_kv_mb"] / e["logical_kv_mb"], 3)
            if e["logical_compression_vs_bf16"] != exp_ratio:
                fail(f"{lang}/{m}: compression {e['logical_compression_vs_bf16']} != {exp_ratio}")
            if abs(e["logged_compression"] - exp_ratio) > 0.0015:
                fail(f"{lang}/{m}: logged compression {e['logged_compression']} vs derived {exp_ratio}")
            exp_delta = round((e["ppl"] / bf["ppl"] - 1) * 100, 2)
            if e["ppl_delta_pct_vs_bf16_computed"] != exp_delta:
                fail(f"{lang}/{m}: computed delta {e['ppl_delta_pct_vs_bf16_computed']} != {exp_delta}")
            # Rounded-PPL recomputation vs the script's unrounded figure.
            if abs(exp_delta - e["ppl_delta_pct_vs_bf16_logged"]) > 0.006:
                fail(f"{lang}/{m}: computed delta {exp_delta} disagrees with logged {e['ppl_delta_pct_vs_bf16_logged']}")

            # Per-sample rows.
            smp = e["samples"]
            if [x["sample"] for x in smp] != list(range(1, SAMPLES + 1)):
                fail(f"{lang}/{m}: sample numbering {[x['sample'] for x in smp]}")
            for x in smp:
                n["sample_rows"] += 1
                mm = SAMPLE_LINE.match(log_lines[x["line"] - 1])
                if not mm or int(mm.group(1)) != x["sample"] or int(mm.group(2)) != SAMPLES:
                    fail(f"{lang}/{m} sample {x['sample']}: cited line {x['line']} is not that sample")
                    continue
                if float(mm.group(3)) != x["ppl"] or float(mm.group(4)) != x["logical_kv_mb"]:
                    fail(f"{lang}/{m} sample {x['sample']}: values differ from log line {x['line']}")
                # Confirm the sample belongs to this language/method block.
                header = next(
                    (log_lines[i] for i in range(x["line"] - 2, -1, -1) if log_lines[i].startswith("Running ")),
                    "",
                )
                if header != f"Running {lang}/{m}...":
                    fail(f"{lang}/{m} sample {x['sample']}: line {x['line']} is under '{header}'")
            # Re-derive logged median / avg KV from per-sample values.
            n["derived_values"] += 2
            med = statistics.median(x["ppl"] for x in smp)
            if abs(med - e["median_sample_ppl"]) > 0.0001 + 1e-9:
                fail(f"{lang}/{m}: median of samples {med:.5f} != logged {e['median_sample_ppl']}")
            avg_kv = statistics.mean(x["logical_kv_mb"] for x in smp)
            if abs(avg_kv - e["logical_kv_mb"]) > 0.0005 + 1e-9:
                fail(f"{lang}/{m}: mean sample KV {avg_kv:.4f} != logged {e['logical_kv_mb']}")

    if n["aggregate_rows"] != 10:
        fail(f"{n['aggregate_rows']} aggregate rows, expected 10")
    if n["sample_rows"] != 80:
        fail(f"{n['sample_rows']} per-sample rows in summary, expected 80")

    # Cross-language mean/worst.
    for m in METHODS[1:]:
        c = s["cross_language"][m]
        src = c["source"]
        actual = log_lines[src["line"] - 1].rstrip()
        tok = actual.split()
        if actual != src["row"] or tok[0] != m or len(tok) != 6:
            fail(f"cross_language/{m}: source row mismatch")
        for key in ("mean_ppl_delta_pct_logged", "worst_ppl_delta_pct_logged"):
            n["raw_values"] += 1
            if not in_row(c[key], tok):
                fail(f"cross_language/{m}.{key} not in source row")
        zh_d = langs["zh"]["methods"][m]["ppl_delta_pct_vs_bf16_logged"]
        es_d = langs["es"]["methods"][m]["ppl_delta_pct_vs_bf16_logged"]
        n["derived_values"] += 2
        if c["mean_ppl_delta_pct_computed"] != round((zh_d + es_d) / 2, 2):
            fail(f"cross_language/{m}: mean does not recompute")
        if c["worst_ppl_delta_pct_computed"] != max(zh_d, es_d):
            fail(f"cross_language/{m}: worst does not recompute")
        if abs(c["mean_ppl_delta_pct_computed"] - c["mean_ppl_delta_pct_logged"]) > 0.006:
            fail(f"cross_language/{m}: computed mean disagrees with logged summary")
        if c["worst_ppl_delta_pct_computed"] != c["worst_ppl_delta_pct_logged"]:
            fail(f"cross_language/{m}: computed worst disagrees with logged summary")

    # Canonical reference: SHA first, then values, then equality with rerun.
    cref = s["canonical_reference"]
    canon_sha_ok = (
        cref["source"] == CANONICAL_LOG.relative_to(ROOT).as_posix()
        and cref["source_sha256"] == sha256(CANONICAL_LOG)
    )
    if not canon_sha_ok:
        fail("canonical_reference source/SHA-256 mismatch; canonical values NOT trusted")
    elif "NOT newly generated" not in cref["note"]:
        fail("canonical_reference not labelled as NOT newly generated data")
    else:
        canon_lines = CANONICAL_LOG.read_text(encoding="utf-8").splitlines()
        canon_samples: dict = {}
        cur = None
        for line in canon_lines:
            mm = re.match(r"^Running (zh|es)/(bf16|rabit\d)\.\.\.$", line)
            if mm:
                cur = (mm.group(1), mm.group(2))
                canon_samples[cur] = []
                continue
            mm = SAMPLE_LINE.match(line)
            if mm and cur:
                canon_samples[cur].append(float(mm.group(3)))
        for lang in LANGS:
            for m in ("bf16", "rabit2"):
                c = cref["languages"][lang][m]
                actual = canon_lines[c["source"]["line"] - 1].rstrip()
                if actual != c["source"]["row"] or actual.split()[0] != m:
                    fail(f"canonical {lang}/{m}: source row mismatch")
                tok = actual.split()
                for key in ("ppl", "ppl_delta_pct_vs_bf16", "median_sample_ppl",
                            "logical_kv_mb", "logged_compression", "scored_tokens"):
                    n["canonical_values"] += 1
                    if not in_row(c[key], tok):
                        fail(f"canonical {lang}/{m}.{key} not in canonical row")
                if c["sample_ppls"] != canon_samples.get((lang, m)):
                    fail(f"canonical {lang}/{m}: sample PPLs differ from canonical log")
                r = langs[lang]["methods"][m]
                same = (
                    (c["ppl"], c["ppl_delta_pct_vs_bf16"], c["median_sample_ppl"], c["logical_kv_mb"],
                     c["logged_compression"], c["scored_tokens"], c["sample_ppls"])
                    == (r["ppl"], r["ppl_delta_pct_vs_bf16_logged"], r["median_sample_ppl"], r["logical_kv_mb"],
                        r["logged_compression"], r["scored_tokens"], [x["ppl"] for x in r["samples"]])
                )
                if not same:
                    fail(f"canonical {lang}/{m}: rerun is NOT exactly identical")
                if c["rerun_exactly_identical"] is not same:
                    fail(f"canonical {lang}/{m}: rerun_exactly_identical flag is wrong")
        if cref["rerun_bf16_rabit2_exactly_identical"] is not True:
            fail("canonical_reference: rerun_bf16_rabit2_exactly_identical is not true")

    # Notes: every quoted number matches the data, and each claim holds.
    zh, es = langs["zh"]["methods"], langs["es"]["methods"]
    d = lambda L, m: L[m]["ppl_delta_pct_vs_bf16_logged"]  # noqa: E731
    fmt = lambda v: f"+{v:.2f}%"  # noqa: E731
    notes = s["notes"]
    rabit3_note = next(x for x in notes if x.startswith("rabit3 is worse than rabit2"))
    lang_note = next(x for x in notes if x.startswith("Chinese has the larger delta"))
    claims = {
        "rabit3 worse than rabit2 in zh": d(zh, "rabit3") > d(zh, "rabit2"),
        "rabit3 worse than rabit2 in es": d(es, "rabit3") > d(es, "rabit2"),
        "rabit3 compresses less than rabit2 (zh)": zh["rabit3"]["logical_compression_vs_bf16"] < zh["rabit2"]["logical_compression_vs_bf16"],
        "rabit3 compresses less than rabit2 (es)": es["rabit3"]["logical_compression_vs_bf16"] < es["rabit2"]["logical_compression_vs_bf16"],
        "zh > es at rabit4": d(zh, "rabit4") > d(es, "rabit4"),
        "zh > es at rabit3": d(zh, "rabit3") > d(es, "rabit3"),
        "es > zh at rabit2": d(es, "rabit2") > d(zh, "rabit2"),
        "rabit3 note numbers": all(
            t in rabit3_note
            for t in (
                f"zh {fmt(d(zh, 'rabit3'))} vs {fmt(d(zh, 'rabit2'))}",
                f"es {fmt(d(es, 'rabit3'))} vs {fmt(d(es, 'rabit2'))}",
                f"{zh['rabit3']['logical_compression_vs_bf16']:.3f}x vs {zh['rabit2']['logical_compression_vs_bf16']:.3f}x",
            )
        ),
        "language note numbers": all(
            t in lang_note
            for t in (
                f"zh {fmt(d(zh, 'rabit4'))} vs es {fmt(d(es, 'rabit4'))}",
                f"zh {fmt(d(zh, 'rabit3'))} vs es {fmt(d(es, 'rabit3'))}",
                f"es {fmt(d(es, 'rabit2'))} vs zh {fmt(d(zh, 'rabit2'))}",
            )
        ),
        "8 samples/language stated": "8 articles per language" in " ".join(notes)
        and s["settings"]["samples_per_language"] == SAMPLES,
        "not a pure bit-width ablation stated": any("NOT a pure bit-width ablation" in x for x in notes),
        "rabit8/4/3 new evidence stated": any(x.startswith("rabit8, rabit4 and rabit3 are new") for x in notes),
        "bf16/rabit2 only canonical baselines stated": any(x.startswith("bf16 and rabit2 are the only canonical") for x in notes),
        "KV label": s["kv_memory_label"] == "LOGICAL fake-quant KV storage; not physical vLLM allocator capacity",
    }
    for label, ok in claims.items():
        n["note_checks"] += 1
        if not ok:
            fail(f"note check failed: {label}")

    print(f"raw log SHA-256 verified:        {log_sha_ok}")
    print(f"raw log per-sample lines:        {len(raw_sample_lines)}")
    print(f"aggregate rows verified:         {n['aggregate_rows']}")
    print(f"per-sample rows verified:        {n['sample_rows']}")
    print(f"raw values traced to log rows:   {n['raw_values']}")
    print(f"derived values recomputed:       {n['derived_values']}")
    print(f"canonical source SHA-256 ok:     {canon_sha_ok}")
    print(f"canonical values verified:       {n['canonical_values']}")
    print(f"note/claim checks:               {n['note_checks']}")
    print(f"FAILURES: {len(fails)}")
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
