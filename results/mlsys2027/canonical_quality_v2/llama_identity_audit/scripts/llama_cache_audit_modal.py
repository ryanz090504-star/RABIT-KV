"""
READ-ONLY audit of the historical Llama snapshot in the modelscope-llama31-cache Modal volume.
CPU only. No GPU, no download, no model load, no torch. The volume is mounted READ-ONLY (Volume.read_only()),
so nothing can be written or committed. Emits LLAMA_CACHE_AUDIT=<json>.
"""
import json

import modal

app = modal.App("llama-cache-identity-audit-readonly")
vol = modal.Volume.from_name("modelscope-llama31-cache").read_only()
image = modal.Image.debian_slim(python_version="3.11")

LLAMA_ROOT = "/model_cache/models/LLM-Research--Meta-Llama-3.1-8B-Instruct"


@app.function(image=image, cpu=8, memory=4096, timeout=3600, volumes={"/model_cache": vol})
def audit() -> dict:
    import datetime
    import hashlib
    import os
    from concurrent.futures import ThreadPoolExecutor

    def iso(t):
        return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat()

    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    # write probe: must FAIL on a read-only mount
    try:
        open("/model_cache/.audit_write_probe", "w").close()
        write_probe = "UNEXPECTED: write succeeded"
        os.remove("/model_cache/.audit_write_probe")
    except Exception as e:  # noqa: BLE001
        write_probe = f"write refused ({type(e).__name__}: {e})"

    # every entry in the whole volume (dirs, files, symlinks, hidden) -- bookkeeping / provenance view
    entries = []
    for root, dirs, files in os.walk("/model_cache", followlinks=False):
        for name in sorted(dirs) + sorted(files):
            p = os.path.join(root, name)
            st = os.lstat(p)
            entries.append({
                "path": os.path.relpath(p, "/model_cache").replace(os.sep, "/"),
                "kind": "link" if os.path.islink(p) else ("dir" if os.path.isdir(p) else "file"),
                "link_target": os.readlink(p) if os.path.islink(p) else None,
                "size": st.st_size, "mtime_utc": iso(st.st_mtime), "ctime_utc": iso(st.st_ctime),
            })

    # per-snapshot-directory manifest of the Llama repo
    snapshots = {}
    snap_root = os.path.join(LLAMA_ROOT, "snapshots")
    for snap in sorted(os.listdir(snap_root)):
        base = os.path.join(snap_root, snap)
        paths = []
        for root, _dirs, files in os.walk(base, followlinks=False):
            for name in files:
                paths.append(os.path.join(root, name))
        paths.sort()
        with ThreadPoolExecutor(max_workers=8) as ex:
            digests = list(ex.map(sha, paths))
        rows = []
        for p, d in zip(paths, digests):
            st = os.lstat(p)
            rows.append({"path": os.path.relpath(p, base).replace(os.sep, "/"), "size": st.st_size, "sha256": d,
                         "mtime_utc": iso(st.st_mtime), "is_link": os.path.islink(p)})
        snapshots[snap] = rows

    # small text bookkeeping files anywhere in the Llama repo dir / .lock (content, if any)
    small = {}
    for root, _dirs, files in list(os.walk(LLAMA_ROOT)) + list(os.walk("/model_cache/.lock")):
        for name in files:
            p = os.path.join(root, name)
            if "/snapshots/" in p.replace(os.sep, "/"):
                continue
            if os.path.getsize(p) <= 65536:
                with open(p, "rb") as f:
                    small[os.path.relpath(p, "/model_cache")] = f.read().decode("utf-8", "replace")

    res = {"write_probe": write_probe, "entries": entries, "llama_snapshots": snapshots,
           "non_snapshot_small_files": small,
           "llama_repo_dir_listing": sorted(os.listdir(LLAMA_ROOT)),
           "llama_snapshots_listing": sorted(os.listdir(snap_root))}
    print("LLAMA_CACHE_AUDIT=" + json.dumps(res, sort_keys=True), flush=True)
    return res


@app.local_entrypoint()
def main(out: str):
    res = audit.remote()
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print("written", out)
