"""Builds audit_summary.json and llama_manifest.tsv from the raw audit artifacts in ../raw (stdlib only; offline).
Usage (from the repository root):
    python results/mlsys2027/canonical_quality_v2/llama_identity_audit/scripts/build_audit_summary.py
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parents[3]
REVISION = "359efdbb8af05b788a4ad4185215c6b8caa9052c"
SNAPSHOT = "models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master"
MOUNTED = "/model_cache/" + SNAPSHOT

audit = json.loads((HERE / "raw/llama_cache_audit.json").read_text(encoding="utf-8"))
gm = json.loads((HERE / "raw/git_manifests.json").read_text(encoding="utf-8"))
assert audit["llama_snapshots_listing"] == ["master"]
rows = audit["llama_snapshots"]["master"]
files = sorted([r["path"], r["size"], r["sha256"]] for r in rows)
tsv = "\n".join(f"{p}\t{s}\t{h}" for p, s, h in files)  # same definition as canonical_ppl_identity.manifest_sha256
manifest_sha = hashlib.sha256(tsv.encode()).hexdigest()
(HERE / "llama_manifest.tsv").write_bytes(tsv.encode())

cand = gm[REVISION]
gf = {f[0]: f for f in cand["files"]}
per_file = [{"path": p, "size": s, "sha256": h, "git_kind": gf[p][3], "git_blob_sha1": gf[p][4],
             "match": gf[p][1] == s and gf[p][2] == h} for p, s, h in files]
hist = {p: h for p, _, h in files}
by_commit = {c: {"tree": v["tree"], "files": len(v["files"]), "manifest_sha256": v["manifest_sha256"],
                 "equals_historical_manifest": v["manifest_sha256"] == manifest_sha,
                 "differing_files": sorted(f[0] for f in v["files"] if hist.get(f[0]) != f[2]),
                 "meta_parents_ct_date_subject": v["meta"]} for c, v in gm.items()}

shard1 = hist["model-00001-of-00004.safetensors"]
corroboration = []
for f in subprocess.run(["git", "grep", "-l", shard1, "--", "results"], cwd=ROOT, capture_output=True,
                        text=True, check=True).stdout.split():
    if "llama_identity_audit" in f:
        continue
    txt = (ROOT / f).read_text(encoding="utf-8", errors="replace")
    ts = sorted(re.findall(r"20\d\d-\d\d-\d\d[ T]\d\d:\d\d:\d\d", txt))
    corroboration.append({"file": f,
                          "line_of_shard1_hash": [i + 1 for i, ln in enumerate(txt.splitlines()) if shard1 in ln][0],
                          "files_whose_sha256_appears": sorted(p for p, h in hist.items() if h in txt),
                          "records_snapshot_dir": MOUNTED in txt, "first_timestamp_in_file": ts[0] if ts else None})
hashed_historically = sorted(set().union(*[set(c["files_whose_sha256_appears"]) for c in corroboration]))

summary = {
    "kind": "READ-ONLY identity recovery audit of the Llama model used by accepted Exp12 (no GPU, no model execution; "
            "NOT a quality result)",
    "audit_date_utc": "2026-10-03",
    "status": "accepted by review (2026-10-03)",
    "approved_wording": "Exp12 loaded model bytes byte-identical to ModelScope revision "
                        "359efdbb8af05b788a4ad4185215c6b8caa9052c, recovered by content matching of the preserved "
                        "historical cache.",
    "not_claimed": ["Exp12 explicitly recorded that revision",
                    "the historical master branch pointer was directly recovered"],
    "model_id": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    "immutable_revision": REVISION,
    "immutable_tree": cand["tree"],
    "repository": "https://www.modelscope.cn/LLM-Research/Meta-Llama-3.1-8B-Instruct.git",
    "exp12": {
        "started_utc": "2026-09-30T00:37:26.468843+00:00", "completed_utc": "2026-09-30T00:54:58.175753+00:00",
        "evidence": ["results/mlsys2027/variance/acceptance_record.json:12-13",
                     "results/mlsys2027/variance/manifest.json:5", "results/mlsys2027/variance/manifest.json:65"],
        "model_load_log_lines": {"results/mlsys2027/variance/continuation_ppl.log:24": "2026-09-30 00:37:41",
                                 "results/mlsys2027/variance/niah.log:21": "2026-09-30 00:38:47",
                                 "results/mlsys2027/variance/passage_retrieval.log:21": "2026-09-30 00:41:12",
                                 "results/mlsys2027/variance/hotpotqa.log:22": "2026-09-30 00:48:40",
                                 "results/mlsys2027/variance/qasper.log:22": "2026-09-30 00:53:13"},
        "recorded_identity": "only 'Downloading 18 files from LLM-Research/Meta-Llama-3.1-8B-Instruct@master' and "
                             "'(modelscope snapshot_download)'; no revision, hash, manifest or snapshot path",
        "code": "benchmarks/quality/continuation_ppl.py (raw sha256 4f1385a6..., pinned in variance/manifest.json:12) "
                "mounts Modal volume modelscope-llama31-cache at /model_cache and calls "
                "snapshot_download(model_name, cache_dir='/model_cache') with no revision (lines 42-52, 115-119)",
        "downloaded_or_reused": "reused the persistent cache: all 18 files and the snapshot directory carry 2026-07-10 "
                                "mtime and ctime; each Exp12 '18 files' step took about 20 s for 32 GB"},
    "historical_cache": {
        "modal_volume": "modelscope-llama31-cache", "volume_created": "2026-07-10 03:28:40-04:00",
        "snapshot_path_in_volume": SNAPSHOT, "snapshot_path_as_mounted": MOUNTED,
        "llama_snapshot_directories": audit["llama_snapshots_listing"],
        "revision_metadata_in_cache": "none (no refs directory, no metadata file; only zero-byte .lock files)",
        "file_mtime_utc_range": [min(r["mtime_utc"] for r in rows), max(r["mtime_utc"] for r in rows)],
        "symlinks": sum(r["is_link"] for r in rows)},
    "historical_manifest": {
        "definition": "sha256 of the sorted 'path<TAB>size<TAB>sha256' lines joined by LF, no trailing newline "
                      "(canonical_ppl_identity.manifest_sha256); llama_manifest.tsv holds exactly those bytes",
        "files": len(files), "total_bytes": sum(f[1] for f in files), "manifest_sha256": manifest_sha},
    "comparison_with_revision": {
        "source": "git objects of a bare clone (plain blobs hashed; LFS pointers' oid sha256 + size)",
        "matched": sum(x["match"] for x in per_file), "of": len(per_file),
        "extra_in_git_tree": sorted(set(gf) - set(hist)), "result": "EXACT MATCH (18/18)", "per_file": per_file,
        "commits_examined": len(by_commit),
        "commits_equal_to_historical_manifest": [c for c, v in by_commit.items() if v["equals_historical_manifest"]],
        "by_commit": by_commit},
    "historical_corroboration": {
        "note": "accepted runs that printed size + SHA-256 of files in the SAME snapshot directory at run time "
                "(before and after Exp12)",
        "files_hashed_historically": hashed_historically,
        "files_not_hashed_historically": sorted(set(hist) - set(hashed_historically)),
        "records": corroboration},
    "limitations": [
        "Exp12 recorded '@master', not an immutable revision.",
        "The branch pointer itself was not reconstructed: ModelScope exposes current refs and commit history only (no "
        "reflog); a since-deleted commit with an identical tree cannot be excluded.",
        "The historical cache was hashed later (2026-10-03), not at Exp12 time.",
        "The model-relevant files (weight shards, index, configs, tokenizer files) have stronger historical "
        "corroboration: accepted runs before and after Exp12 recorded the same hashes from the same directory. The "
        "other files (.gitattributes, LICENSE, README.md, USE_POLICY.md, original/*) rest on the unchanged 2026-07-10 "
        "timestamps.",
        "README.md is the only file separating 359efdbb from its parent d02f94ee; the model-relevant bytes are "
        "identical in both.",
        "Exp12's logs do not print the snapshot path; it follows from the hash-pinned script and from other runs that "
        "print the same path."],
    "provenance": {
        "hashing": "CPU-only Modal app llama-cache-identity-audit-readonly (ap-MckPw5RFdG9cFcXlUyARGF), debian_slim "
                   "python 3.11, cpu=8, no GPU, no torch, no model load; modal client 1.4.3",
        "volume_mount": "read-only (modal.Volume.from_name(...).read_only())",
        "write_probe": audit["write_probe"],
        "git": "bare partial clone (GIT_LFS_SKIP_SMUDGE=1, --filter=blob:limit=20m); nothing downloaded into the volume",
        "raw_artifacts": ["raw/llama_cache_audit.json", "raw/git_manifests.json", "raw/vol_listing.json",
                          "raw/audit_run.log"],
        "scripts_as_run": ["scripts/llama_cache_audit_modal.py", "scripts/git_manifests.py", "scripts/vol_list.py",
                           "scripts/compare.py"]},
    "accepted_evidence_modified": False,
}
assert manifest_sha == "85d9cffee6980348ad1c334d71f8731f6442553535848542457b68d85d70ce89"
assert summary["comparison_with_revision"]["matched"] == 18
assert summary["comparison_with_revision"]["commits_equal_to_historical_manifest"] == [REVISION]
(HERE / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                                         newline="\n")
print(manifest_sha, cand["tree"], len(corroboration), hashed_historically)
