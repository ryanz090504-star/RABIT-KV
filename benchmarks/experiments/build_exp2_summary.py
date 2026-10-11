"""
Build results/mlsys2027/multilingual_frontier/summary.json for MLSys 2027
Experiment 2 (multilingual operating-point quality-compression sweep).

Every rerun value is parsed ONLY from the raw Experiment 2 log,
results/mlsys2027/multilingual_frontier/multilingual_ppl.log:
  * per-language aggregate rows (source line + verbatim row recorded);
  * the 80 per-sample rows (2 languages x 5 methods x 8 samples);
  * the cross-language summary table (mean / worst delta).
Derived values (compression vs bf16, PPL delta vs bf16, mean/worst delta)
are computed from those raw values. The log's SHA-256 is recorded.

The only non-rerun data is the separate canonical_reference section, which
quotes the frozen bf16/rabit2 rows from results/quality/multilingual_ppl.log
(read-only, with that file's SHA-256) and records whether the rerun values
are exactly identical.

SHA-256s are over LF-normalized bytes (CRLF -> LF), so they match the
committed git content regardless of core.autocrlf.

Read-only with respect to the raw log and all canonical results; the only
file written is results/mlsys2027/multilingual_frontier/summary.json. Runs
locally; no Modal/GPU.

Usage:
    python benchmarks/mlsys2027/build_exp2_summary.py
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "mlsys2027" / "multilingual_frontier"
REL = "results/mlsys2027/multilingual_frontier"
LOG = OUT_DIR / "multilingual_ppl.log"
SUMMARY = OUT_DIR / "summary.json"
CANONICAL_LOG = ROOT / "results" / "quality" / "multilingual_ppl.log"

METHODS = ["bf16", "rabit8", "rabit4", "rabit3", "rabit2"]
LANGUAGES = {"zh": "CHINESE", "es": "SPANISH"}
KV_LABEL = "LOGICAL fake-quant KV storage; not physical vLLM allocator capacity"
SHA256_BASIS = (
    "SHA-256 of LF-normalized bytes (CRLF -> LF): equals the SHA-256 of the "
    "committed git content and is stable across core.autocrlf checkouts"
)

SECTION = re.compile(r"^RABIT-KV MULTILINGUAL CONTINUATION PPL \S+ (CHINESE|SPANISH)\s*$")
AGG_ROW = re.compile(r"^(bf16|rabit\d)\s")
RUNNING = re.compile(r"^Running (zh|es)/(bf16|rabit\d)\.\.\.$")
SAMPLE = re.compile(r"^  sample (\d+)/(\d+): PPL=([0-9.]+), KV=([0-9.]+) MB, ")
SUMMARY_HEADER = "MULTILINGUAL RELATIVE-QUALITY SUMMARY"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def parse_log(path: Path) -> dict:
    """Return {'aggregate': {lang: {method: row}}, 'samples': {lang: {method: [..]}},
    'summary': {method: row}, 'presets': {...}}, each row with line + verbatim text."""
    code_for = {v: k for k, v in LANGUAGES.items()}
    lines = path.read_text(encoding="utf-8").splitlines()
    aggregate: dict = {}
    samples: dict = {}
    summary: dict = {}
    section = None
    running = None
    in_summary = False
    for lineno, line in enumerate(lines, start=1):
        m = SECTION.match(line)
        if m:
            section, running = code_for[m.group(1)], None
            aggregate[section] = {}
            continue
        if line.startswith(SUMMARY_HEADER):
            section, running, in_summary = None, None, True
            continue
        m = RUNNING.match(line)
        if m:
            running = (m.group(1), m.group(2))
            samples.setdefault(running[0], {}).setdefault(running[1], [])
            continue
        m = SAMPLE.match(line)
        if m and running:
            samples[running[0]][running[1]].append(
                {
                    "sample": int(m.group(1)),
                    "of": int(m.group(2)),
                    "ppl": float(m.group(3)),
                    "logical_kv_mb": float(m.group(4)),
                    "line": lineno,
                }
            )
            continue
        if AGG_ROW.match(line):
            tok = line.split()
            if section and len(tok) == 7:
                aggregate[section][tok[0]] = {"line": lineno, "row": line.rstrip(), "tok": tok}
            elif in_summary and len(tok) == 6:
                summary[tok[0]] = {"line": lineno, "row": line.rstrip(), "tok": tok}

    i = lines.index("Configurations:")
    presets = {}
    for line in lines[i + 1 :]:
        if not line.startswith("  ") or ": " not in line:
            break
        key, value = line.strip().split(": ", 1)
        presets[key] = value
    return {"aggregate": aggregate, "samples": samples, "summary": summary, "presets": presets}


def source(lineno: int, row: str, log_rel: str) -> dict:
    return {"log": log_rel, "line": lineno, "row": row}


def build() -> dict:
    parsed = parse_log(LOG)
    agg, smp, summ = parsed["aggregate"], parsed["samples"], parsed["summary"]
    log_rel = rel(LOG)

    languages = {}
    for lang in LANGUAGES:
        rows = agg[lang]
        bf16_ppl = float(rows["bf16"]["tok"][1])
        bf16_mb = float(rows["bf16"]["tok"][4])
        methods = {}
        for m in METHODS:
            tok = rows[m]["tok"]
            ppl, mb = float(tok[1]), float(tok[4])
            e = {
                "ppl": ppl,
                "ppl_delta_pct_vs_bf16_logged": float(tok[2]),
                "ppl_delta_pct_vs_bf16_computed": round((ppl / bf16_ppl - 1) * 100, 2),
                "median_sample_ppl": float(tok[3]),
                "logical_kv_mb": mb,
                "logical_compression_vs_bf16": round(bf16_mb / mb, 3),
                "logged_compression": float(tok[5]),
                "scored_tokens": int(tok[6]),
                "samples": [
                    {"sample": s["sample"], "ppl": s["ppl"], "logical_kv_mb": s["logical_kv_mb"], "line": s["line"]}
                    for s in smp[lang][m]
                ],
                "source": source(rows[m]["line"], rows[m]["row"], log_rel),
            }
            methods[m] = e
        languages[lang] = {"language": LANGUAGES[lang].title(), "methods": methods}

    cross = {}
    for m in METHODS[1:]:
        tok = summ[m]["tok"]  # Method zh_d% es_d% Mean_d% Worst_d% Mean_comp
        zh_d = languages["zh"]["methods"][m]["ppl_delta_pct_vs_bf16_logged"]
        es_d = languages["es"]["methods"][m]["ppl_delta_pct_vs_bf16_logged"]
        cross[m] = {
            "mean_ppl_delta_pct_logged": float(tok[3]),
            "worst_ppl_delta_pct_logged": float(tok[4]),
            "mean_ppl_delta_pct_computed": round((zh_d + es_d) / 2, 2),
            "worst_ppl_delta_pct_computed": max(zh_d, es_d),
            "worst_language": "zh" if zh_d >= es_d else "es",
            "source": source(summ[m]["line"], summ[m]["row"], log_rel),
        }

    # Canonical reference (bf16/rabit2 only), read-only.
    canon = parse_log(CANONICAL_LOG)
    canon_rel = rel(CANONICAL_LOG)
    canonical = {}
    identical = True
    for lang in LANGUAGES:
        canonical[lang] = {}
        for m in ("bf16", "rabit2"):
            tok = canon["aggregate"][lang][m]["tok"]
            c = {
                "ppl": float(tok[1]),
                "ppl_delta_pct_vs_bf16": float(tok[2]),
                "median_sample_ppl": float(tok[3]),
                "logical_kv_mb": float(tok[4]),
                "logged_compression": float(tok[5]),
                "scored_tokens": int(tok[6]),
                "sample_ppls": [s["ppl"] for s in canon["samples"][lang][m]],
                "source": source(canon["aggregate"][lang][m]["line"], canon["aggregate"][lang][m]["row"], canon_rel),
            }
            rerun = languages[lang]["methods"][m]
            same = (
                c["ppl"] == rerun["ppl"]
                and c["ppl_delta_pct_vs_bf16"] == rerun["ppl_delta_pct_vs_bf16_logged"]
                and c["median_sample_ppl"] == rerun["median_sample_ppl"]
                and c["logical_kv_mb"] == rerun["logical_kv_mb"]
                and c["logged_compression"] == rerun["logged_compression"]
                and c["scored_tokens"] == rerun["scored_tokens"]
                and c["sample_ppls"] == [s["ppl"] for s in rerun["samples"]]
            )
            c["rerun_exactly_identical"] = same
            identical &= same
            canonical[lang][m] = c

    zh, es = languages["zh"]["methods"], languages["es"]["methods"]
    d = lambda L, m: L[m]["ppl_delta_pct_vs_bf16_logged"]  # noqa: E731
    fmt = lambda v: f"+{v:.2f}%"  # noqa: E731

    return {
        "experiment": "MLSys 2027 Experiment 2 -- multilingual operating-point quality-compression sweep",
        "description": (
            "Chinese/Spanish Wikipedia teacher-forced continuation perplexity across the "
            "bf16 baseline and the rabit8/rabit4/rabit3/rabit2 presets on "
            "Llama-3.1-8B-Instruct. This is an operating-point sweep, NOT a pure bit-width "
            "ablation: the presets differ in more than bit width (see presets_as_logged)."
        ),
        "derivation": (
            "All rerun values are parsed ONLY from the raw log recorded in raw_log; each "
            "aggregate row carries its source line and verbatim text, and each per-sample "
            "value its line number. logical_compression_vs_bf16 = bf16 logical KV MB / "
            "method logical KV MB (3 dp). ppl_delta_pct_vs_bf16_computed = (PPL / bf16 PPL "
            "- 1) x 100 (2 dp) from the rounded logged PPLs; the *_logged values are the "
            "script's own figures from unrounded internals. Cross-language mean/worst are "
            "computed from the per-language logged deltas and checked against the log's "
            "summary table. canonical_reference is the only non-rerun data."
        ),
        "generated_by": "benchmarks/mlsys2027/build_exp2_summary.py",
        "validated_by": "benchmarks/mlsys2027/validate_exp2_summary.py",
        "kv_memory_label": KV_LABEL,
        "raw_log": {"path": log_rel, "sha256": sha256(LOG), "sha256_basis": SHA256_BASIS},
        "run_reference": {
            "note": "Run metadata only (not used for any number); see manifest.json.",
            "manifest": f"{REL}/manifest.json",
            "regression_check": f"{REL}/regression_check.json",
        },
        "settings": {
            "model": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
            "languages": list(LANGUAGES),
            "dataset": "wikimedia/wikipedia 20231101.{zh,es}",
            "dataset_revision": "cf584d1dc131caa92a5cb910f41a8b7591b12732",
            "shuffle_seed": 20260804,
            "shuffle_buffer": 1000,
            "context_tokens": 1024,
            "eval_tokens_per_sample": 128,
            "samples_per_language": 8,
            "scored_tokens_per_language_method": 8 * 128,
            "note": "Settings as passed by the Experiment 2 runner; see manifest.json.",
        },
        "presets_as_logged": parsed["presets"],
        "languages": languages,
        "cross_language": cross,
        "canonical_reference": {
            "note": (
                "CANONICAL REFERENCE -- frozen bf16/rabit2 values quoted read-only from the "
                "canonical multilingual run. NOT newly generated data and NOT part of this rerun."
            ),
            "source": canon_rel,
            "source_sha256": sha256(CANONICAL_LOG),
            "sha256_basis": SHA256_BASIS,
            "methods_in_canonical_run": ["bf16", "rabit2"],
            "languages": canonical,
            "rerun_bf16_rabit2_exactly_identical": identical,
        },
        "notes": [
            "This is an operating-point quality-compression sweep, NOT a pure bit-width ablation: rabit8/rabit4/rabit3/rabit2 differ in more than bit width (see presets_as_logged).",
            "rabit8, rabit4 and rabit3 are new MLSys 2027 multilingual evidence; the canonical multilingual run contained only bf16 and rabit2.",
            "bf16 and rabit2 are the only canonical regression baselines; the rerun reproduces them exactly (see canonical_reference).",
            (
                f"rabit3 is worse than rabit2 in both languages (zh {fmt(d(zh, 'rabit3'))} vs {fmt(d(zh, 'rabit2'))}; "
                f"es {fmt(d(es, 'rabit3'))} vs {fmt(d(es, 'rabit2'))}) while also compressing less "
                f"({zh['rabit3']['logical_compression_vs_bf16']:.3f}x vs {zh['rabit2']['logical_compression_vs_bf16']:.3f}x). "
                "This is observational and must not be causally attributed before the K/V/G/R/META ablations."
            ),
            (
                f"Chinese has the larger delta at rabit4 (zh {fmt(d(zh, 'rabit4'))} vs es {fmt(d(es, 'rabit4'))}) and "
                f"rabit3 (zh {fmt(d(zh, 'rabit3'))} vs es {fmt(d(es, 'rabit3'))}), while Spanish is slightly larger "
                f"at rabit2 (es {fmt(d(es, 'rabit2'))} vs zh {fmt(d(zh, 'rabit2'))})."
            ),
            "Small sample count (8 articles per language): sample-level and median differences, and small aggregate differences, should not be interpreted as statistically significant.",
            f"All KV figures: {KV_LABEL}.",
        ],
    }


def main() -> int:
    summary = build()
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {rel(SUMMARY)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
