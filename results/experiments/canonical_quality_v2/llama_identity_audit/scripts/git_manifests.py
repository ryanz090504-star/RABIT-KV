"""Per-commit content manifests derived purely from git objects of the ModelScope repo (bare clone, LFS pointers).
For a normal blob: sha256 of the blob bytes. For an LFS pointer: the pointer's `oid sha256:` and `size`.
Output: git_manifests.json  {commit: {"meta":..., "files": [[path,size,sha256,kind,git_blob_sha1],...], "manifest_sha256":...}}
"""
import hashlib
import json
import subprocess
import sys

GIT = ["git", "--git-dir", sys.argv[1]]


def run(*a, text=True):
    return subprocess.run(GIT + list(a), check=True, capture_output=True, text=text, encoding="utf-8" if text else None, errors="replace" if text else None).stdout


def manifest_sha256(files):
    return hashlib.sha256("\n".join(f"{p}\t{s}\t{h}" for p, s, h in sorted(files)).encode()).hexdigest()


cache = {}


def blob_info(sha1):
    if sha1 not in cache:
        data = run("cat-file", "blob", sha1, text=False)
        if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
            kv = dict(line.split(" ", 1) for line in data.decode().strip().splitlines())
            cache[sha1] = (int(kv["size"]), kv["oid"].split(":", 1)[1], "lfs")
        else:
            cache[sha1] = (len(data), hashlib.sha256(data).hexdigest(), "blob")
    return cache[sha1]


out = {}
refs = [l.split()[0] for l in run("for-each-ref", "--format=%(objectname) %(refname)").splitlines()]
commits = run("rev-list", "--all").split()
for c in commits:
    meta = run("log", "-1", "--format=%P|%ct|%cI|%s", c).strip()
    files = []
    for line in run("ls-tree", "-r", c).splitlines():
        info, path = line.split("\t", 1)
        sha1 = info.split()[2]
        size, sha, kind = blob_info(sha1)
        files.append([path, size, sha, kind, sha1])
    out[c] = {"meta": meta, "tree": run("rev-parse", c + "^{tree}").strip(), "files": files,
              "manifest_sha256": manifest_sha256([f[:3] for f in files])}
json.dump(out, open(sys.argv[2], "w"), indent=1)
for c in commits:
    print(c, out[c]["tree"], len(out[c]["files"]), out[c]["manifest_sha256"], "|", out[c]["meta"])
