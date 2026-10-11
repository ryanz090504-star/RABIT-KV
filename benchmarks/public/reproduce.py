"""Public orchestration only; frozen workers, protocols and gates are read-only inputs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading

REPOSITORY = Path(__file__).resolve().parents[2]
PINNED_COMMIT = "45fd03d8c10f2c14f96dcaedfcad25c4f56fe746"
SPECS = {
    "canonical-ppl": ("run_canonical_ppl", "canonical_ppl_protocol.json"),
    "long-context": ("run_canonical_longctx", "canonical_longctx_protocol.json"),
    "matched-serving": ("run_experiment13_external_baseline", "exp13_external_baseline_protocol.json"),
}
MODELS = ("llama3_1_8b", "qwen2_5_7b")
_POLICY = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def no_links(path):
    """Reject symlinks/junctions, including existing ancestors of a new destination."""
    path = Path(os.path.abspath(path))
    for candidate in (path, *path.parents):
        if candidate.exists() or candidate.is_symlink():
            attrs = candidate.lstat()
            if stat.S_ISLNK(attrs.st_mode) or getattr(attrs, "st_file_attributes", 0) & 0x400:
                raise ValueError(f"linked/reparse path is not allowed: {candidate}")


def output_path(value, source, repository=REPOSITORY):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("--output-dir must be absolute")
    no_links(path)
    path = path.resolve()
    if inside(path, source) or inside(path, repository):
        raise ValueError("output must be outside both source and public repositories")
    if path.exists():
        raise FileExistsError(f"refusing to reuse any existing destination: {path}")
    if not path.parent.is_dir():
        raise ValueError("output parent must already exist")
    return path


def confined(path, root):
    if isinstance(path, int):  # already-open descriptor, not a new pathname
        return
    no_links(path)
    if not inside(path, root):
        raise PermissionError(f"write outside registered output directory: {path}")


def check_command(argv, root, source, execute):
    """Allow only frozen static checks, read-only Git, and explicit Modal execution."""
    if not isinstance(argv, (list, tuple)) or not argv:
        raise PermissionError("shell/string commands are not supported")
    args = [os.fspath(x) for x in argv]
    binary = Path(args[0]).name.lower()
    if binary in ("git", "git.exe"):
        rest = args[1:]
        if rest[:2] == ["-c", "core.autocrlf=false"]:
            rest = rest[2:]
        if not rest:
            raise PermissionError("empty Git command")
        verb = rest[0]
        if verb == "archive":
            if "-o" not in rest or "--format=zip" not in rest or rest[-1] != "HEAD:vllm-kvquant":
                raise PermissionError("only the frozen vLLM snapshot archive is allowed")
            confined(rest[rest.index("-o") + 1], root)
            return
        allowed = {"status", "rev-parse", "diff", "show", "log", "ls-files", "cat-file", "merge-base"}
        if verb not in allowed or any(x in rest for x in ("--output", "-o")) or any(x.startswith("--output=") for x in rest):
            raise PermissionError(f"Git mutation/publication command denied: {verb}")
        return
    if Path(args[0]).resolve() == Path(sys.executable).resolve():
        if args[1:3] == ["-m", "modal"] and execute:
            return
        static = {source / "benchmarks/mlsys2027/test_canonical_ppl_harness.py",
                  source / "benchmarks/mlsys2027/test_canonical_longctx_harness.py"}
        if len(args) == 2 and Path(args[1]).resolve() in static:
            return
    raise PermissionError(f"unregistered child command denied: {args[0]}")


def audit(event, args):
    policy = _POLICY
    if policy is None:
        return
    root, source, execute = policy
    if event == "open":
        path, mode, flags = args
        if (mode and any(x in mode for x in "wax+")) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            confined(path, root)
    elif event in ("os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.truncate", "os.utime"):
        confined(args[0], root)
    elif event in ("os.rename", "os.replace"):
        confined(args[0], root)
        confined(args[1], root)
    elif event in ("os.symlink", "os.link", "os.system", "os.exec", "os.posix_spawn"):
        raise PermissionError(f"unregistered operation denied: {event}")
    elif event == "subprocess.Popen":
        check_command(args[1], root, source, execute)


sys.addaudithook(audit)


@contextmanager
def guard(root, source, execute=False):
    global _POLICY
    previous = _POLICY
    _POLICY = (Path(root).resolve(), Path(source).resolve(), execute)
    try:
        yield
    finally:
        _POLICY = previous


class OutputAlias(type(Path())):
    """Physical external output with the frozen protocol's historical logical name.

    Only relative_to(source_root) uses the logical alias. All filesystem operations
    use the physical path. This preserves protocol identity, not original locations.
    """
    def __new__(cls, physical, logical, source):
        obj = super().__new__(cls, physical)
        obj.logical = Path(logical)
        obj.source = Path(source).resolve()
        return obj

    def __init__(self, physical, logical, source):
        # pathlib moved parsing from __new__ to __init__ in Python 3.12.
        if sys.version_info >= (3, 12):
            super().__init__(physical)
        self.logical = Path(logical)
        self.source = Path(source).resolve()

    def relative_to(self, *other, **kwargs):
        if len(other) == 1 and Path(other[0]).resolve() == self.source:
            return self.logical.relative_to(self.source)
        return Path(str(self)).relative_to(*other, **kwargs)


def git(source, *args):
    return subprocess.run(["git", *args], cwd=source, capture_output=True,
                          text=True, check=True, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).stdout.strip()


def load_frozen(source, experiment):
    # Each CLI invocation selects one source tree; never combine modules from trees.
    here = source / "benchmarks/mlsys2027"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(here))
    name, filename = SPECS[experiment]
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != here / f"{name}.py":
        raise RuntimeError("frozen module loaded from the wrong source tree")
    original = (here / filename).read_bytes()
    parsed = json.loads(original)
    if module.load_protocol() != parsed or (here / filename).read_bytes() != original:
        raise RuntimeError("frozen protocol identity failed")
    return module, original


def bind_outputs(module, experiment, output, source):
    if experiment != "matched-serving":
        module.OUT_BASE = output / "artifacts"
        return {"OUT_BASE": str(module.OUT_BASE)}
    aliases = {}
    for name in ("OUT_DIR", "SESSION_LOG", "MANIFEST", "INTEGRITY", "SUMMARY"):
        old = getattr(module, name)
        if isinstance(old, OutputAlias):
            old = old.logical  # repeated offline bindings retain the original alias
        new = output / "artifacts" if name == "OUT_DIR" else output / "artifacts" / old.name
        # OUT_DIR need not preserve a logical name: protocol references its files.
        setattr(module, name, new if name == "OUT_DIR" else OutputAlias(new, old, source))
        aliases[name] = {"historical": str(old), "physical": str(new)}
    return aliases


def write_json(path, value):
    # Exclusive create: even an interrupted run is never overwritten.
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def archive_local(path):
    """Persist uninspected raw-file hashes before the second model, without Git."""
    files = {p.relative_to(path).as_posix(): digest(p) for p in path.rglob("*") if p.is_file()}
    write_json(path / "public_archive.json", {"status": "UNINSPECTED local archive", "sha256": files})


def require_quality_artifacts(path, attempt, model=None):
    """A successful exit alone is insufficient; require the frozen evidence record."""
    for name in ("result.json", "record.json", "session.log"):
        file = path / name
        if not file.is_file() or not file.stat().st_size:
            raise RuntimeError(f"required nonempty artifact missing: {file}")
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    record = json.loads((path / "record.json").read_text(encoding="utf-8"))
    if not isinstance(result, dict) or not isinstance(record, dict):
        raise RuntimeError("required result/record must be JSON objects")
    if (record.get("valid") is not True or record.get("process_ok") is not True
            or record.get("evaluation", {}).get("valid") is not True
            or record.get("cleanup", {}).get("verified") is not True
            or record.get("attempt") != attempt
            or (model is not None and record.get("model_key") != model)
            or record.get("result_sha256") != digest(path / "result.json")):
        raise RuntimeError("frozen result/record validity, identity or content binding failed")


def require_archive(path):
    record = json.loads((path / "public_archive.json").read_text(encoding="utf-8"))
    hashes = record.get("sha256", {})
    actual = {p.relative_to(path).as_posix(): digest(p) for p in path.rglob("*")
              if p.is_file() and p.name != "public_archive.json"}
    if not actual or hashes != actual:
        raise RuntimeError("local archival is missing, incomplete or no longer matches raw outputs")


def require_serving_artifacts(module):
    for path in (module.SESSION_LOG, module.MANIFEST, module.INTEGRITY, module.SUMMARY):
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"required nonempty serving artifact missing: {path}")
    manifest = json.loads(module.MANIFEST.read_text(encoding="utf-8"))
    integrity = json.loads(module.INTEGRITY.read_text(encoding="utf-8"))
    summary = json.loads(module.SUMMARY.read_text(encoding="utf-8"))
    if (manifest.get("status") != "reproduced/unreviewed" or integrity.get("passed") is not True
            or not isinstance(summary, dict) or not summary
            or manifest.get("session_sha256") != digest(module.SESSION_LOG)):
        raise RuntimeError("serving artifact validity or content binding failed")


def archive_cancelled(output):
    """Keep raw bytes and a separate invalid record, never rewrite frozen records."""
    target = output / "cancelled_archive.json"
    if not target.exists():
        files = {p.relative_to(output).as_posix(): digest(p)
                 for p in (output / "artifacts").rglob("*") if p.is_file()}
        write_json(target, {"valid": False, "cancelled": True, "status": "INVALID/cancelled",
                            "sha256": files})


class Cancellation:
    """Defer SIGINT unwinding so frozen quality runners reach their cleanup polls."""
    def __init__(self):
        self.cancelled = False
        self.processes = []
        self.kill_errors = []

    def request(self, *_):
        self.cancelled = True
        for process in self.processes:
            try:
                if process.poll() is None:
                    process.kill()
            except Exception as exc:
                self.kill_errors.append(str(exc))


class CancellableProcess:
    def __init__(self, process, cancellation):
        self.process = process
        self.cancellation = cancellation

    def __getattr__(self, name):
        return getattr(self.process, name)

    def wait(self, *args, **kwargs):
        try:
            code = self.process.wait(*args, **kwargs)
        except KeyboardInterrupt:
            self.cancellation.request()
            # Reap the local client, then return failure to the unchanged runner.
            # Its normal app discovery/poll_cleanup still executes, with no retry.
            self.process.wait(timeout=30)
            code = 130
        return 130 if self.cancellation.cancelled else code


@contextmanager
def cancellation_transport():
    cancellation = Cancellation()
    original_popen = subprocess.Popen
    previous_handler = None

    def popen(argv, *args, **kwargs):
        modal_run = isinstance(argv, (list, tuple)) and list(argv[1:4]) == ["-m", "modal", "run"]
        if modal_run and cancellation.cancelled:
            raise KeyboardInterrupt("cancelled; no new worker may start")
        process = original_popen(argv, *args, **kwargs)
        if modal_run:
            cancellation.processes.append(process)
            if cancellation.cancelled:  # SIGINT may arrive while Popen is spawning
                cancellation.request()
            return CancellableProcess(process, cancellation)
        return process

    subprocess.Popen = popen
    if threading.current_thread() is threading.main_thread():
        previous_handler = signal.signal(signal.SIGINT, cancellation.request)
    try:
        yield cancellation
    finally:
        subprocess.Popen = original_popen
        if previous_handler is not None:
            signal.signal(signal.SIGINT, previous_handler)


def run_two_model(run, archive, output, attempt, cancellation=None):
    registration = {"attempt": attempt, "order": list(MODELS), "policy": "no selective rerun",
                    "both_models_required": True, "status": "REGISTERED before execution"}
    write_json(output / "registration.json", registration)
    codes = dict.fromkeys(MODELS)
    errors = {}
    archived = dict.fromkeys(MODELS, False)
    cancelled = False
    for model in MODELS:
        if cancelled or (cancellation and cancellation.cancelled):
            cancelled = True
            break
        if model == MODELS[1] and (codes[MODELS[0]] != 0 or not archived[MODELS[0]]):
            break
        path = output / "artifacts" / model / f"attempt_{attempt}"
        try:
            codes[model] = run(model, attempt)
            if codes[model] == 0:
                require_quality_artifacts(path, attempt, model)
        except KeyboardInterrupt:
            cancelled = True
            if cancellation:
                cancellation.request()
            codes[model] = 130
            errors[model] = "KeyboardInterrupt; attempt cancelled"
        except (Exception, SystemExit) as exc:
            codes[model] = 1
            errors[model] = str(exc)
        cancelled = cancelled or bool(cancellation and cancellation.cancelled)
        if cancelled:
            codes[model] = 130
        if path.is_dir():
            try:
                archive(path)
                require_archive(path)
                archived[model] = True
            except KeyboardInterrupt:
                cancelled = True
                codes[model] = 130
                errors[model] = "KeyboardInterrupt during archival; attempt cancelled"
            except Exception as exc:
                codes[model] = 1
                errors[model] = f"local archival failed: {exc}"
    if cancelled:
        archive_cancelled(output)
    valid = not cancelled and all(c == 0 for c in codes.values()) and all(archived.values())
    write_json(output / "two_model_record.json", {
        **registration, "runner_exit_codes": codes, "models_not_run": [m for m, c in codes.items() if c is None],
        "errors": errors, "valid": valid, "archived": archived, "cancelled": cancelled,
        "status": "INVALID/cancelled whole attempt" if cancelled else
                  "valid, reproduced/unreviewed" if valid else "INVALID whole attempt; never pool or selectively rerun",
    })
    return 130 if cancelled else 0 if valid else 1


def run_serving(module):
    """Frozen preflight, measurement worker, parser, statistics and integrity gate."""
    provenance = module.preflight(False)
    module.OUT_DIR.mkdir(exist_ok=False)
    snapshot = module.build_snapshot()
    os.environ["EXP13_VLLM_SNAPSHOT"] = str(snapshot)
    code = module.e1.stream_command([
        sys.executable, "-m", "modal", "run", str(module.MODAL_APP), "--legs", module.legs_arg(),
        "--warmups", str(module.WARMUPS_PER_LEG), "--reps-per-leg", str(module.REPS_PER_LEG),
    ], module.SESSION_LOG)
    result = module.analyze(module.SESSION_LOG.read_text(encoding="utf-8", errors="replace"))
    write_json(module.INTEGRITY, result["integrity"])
    valid = code == 0 and result["integrity"]["passed"] and not module.protected_status()
    if valid:
        write_json(module.SUMMARY, result["summary"])
    write_json(module.MANIFEST, {"status": "reproduced/unreviewed" if valid else "INVALID",
                                "provenance": provenance, "modal_returncode": code,
                                "snapshot_sha256": digest(snapshot), "session_sha256": digest(module.SESSION_LOG)})
    return 0 if valid else 1


def child_guard(output, source):
    """Python subprocesses inherit the same write/command policy via sitecustomize."""
    bootstrap = output / "bootstrap"
    bootstrap.mkdir()
    shutil.copyfile(__file__, bootstrap / "public_guard.py")
    (bootstrap / "sitecustomize.py").write_text(
        "import os\n"
        "try:\n"
        "    import pathlib, public_guard\n"
        "    public_guard._POLICY = (pathlib.Path(os.environ['RABIT_PUBLIC_OUTPUT']), "
        "pathlib.Path(os.environ['RABIT_PUBLIC_SOURCE']), True)\n"
        "except BaseException:\n"
        "    os._exit(70)  # sitecustomize errors must never disable confinement\n", encoding="utf-8")
    os.environ.update(RABIT_PUBLIC_OUTPUT=str(output), RABIT_PUBLIC_SOURCE=str(source),
                      PYTHONPATH=str(bootstrap) + os.pathsep + os.environ.get("PYTHONPATH", ""))


def plan(source, experiment, output):
    if git(source, "rev-parse", "--show-toplevel") != str(source) and Path(git(source, "rev-parse", "--show-toplevel")).resolve() != source:
        raise ValueError("--source-root must be a Git repository root")
    head = git(source, "rev-parse", "HEAD")
    if head != PINNED_COMMIT:
        raise ValueError(f"source HEAD must be preserved commit {PINNED_COMMIT}")
    module, protocol = load_frozen(source, experiment)
    aliases = bind_outputs(module, experiment, output, source)
    if module.load_protocol() != json.loads(protocol):
        raise RuntimeError("output mapping changed the frozen protocol")
    return module, {"experiment": experiment, "source_root": str(source), "source_commit": head,
                    "protocol_sha256": hashlib.sha256(protocol).hexdigest(),
                    "output_dir": str(output), "output_aliases": aliases,
                    "source_status": git(source, "status", "--porcelain"),
                    "execution": "not executed; frozen preflight still required",
                    "publication": "no Git mutation/publication commands permitted"}


def require_ready_source(source):
    if git(source, "status", "--porcelain"):
        raise RuntimeError("frozen gates require a clean source checkout; use --source-root for a separate preserved checkout")
    if git(source, "rev-parse", "@{u}") != PINNED_COMMIT:
        raise RuntimeError("frozen gates require source HEAD to match an existing upstream ref; adapter will not push or invent one")


def fresh_attempt(module, experiment, source, attempt):
    if experiment == "canonical-ppl":
        base = source / "results/mlsys2027/canonical_quality_v2/continuation_ppl"
        paths = [base / model / f"attempt_{attempt}" for model in MODELS]
        paths.append(base / f"two_model_attempt_{attempt}")
    elif experiment == "long-context":
        paths = [source / "results/mlsys2027/canonical_quality_v2/long_context" / f"attempt_{attempt}"]
    else:
        paths = []
    if any(path.exists() for path in paths):
        raise FileExistsError("attempt number already belongs to preserved evidence; select a fresh attempt")
    if experiment == "long-context":
        # The unchanged worker appends crash-recovery rows to a shared remote
        # volume. A fresh local directory alone does not protect remote evidence.
        result = subprocess.run([
            sys.executable, "-m", "modal", "volume", "ls", module.ARTIFACT_VOLUME, "--json",
        ], capture_output=True, text=True, check=True)
        entries = json.loads(result.stdout)
        # Modal's table serializer uses filename (older clients: Filename),
        # not the SDK FileEntry.path attribute. Unknown/ambiguous schemas refuse.
        if not isinstance(entries, list):
            raise RuntimeError("cannot establish remote artifact namespace freshness")
        names = []
        for item in entries:
            if not isinstance(item, dict):
                raise RuntimeError("cannot establish remote artifact namespace freshness")
            values = [item[key] for key in ("filename", "Filename") if key in item]
            if not values or any(not isinstance(value, str) or not value.strip("/") for value in values) or len(set(values)) != 1:
                raise RuntimeError("cannot establish remote artifact namespace freshness")
            names.append(values[0].strip("/").split("/")[0])
        if f"attempt_{attempt}" in names:
            raise FileExistsError("remote artifact attempt already exists; refusing to append")


def execute(module, experiment, output, source, attempt, record):
    output.mkdir(exist_ok=False)
    with guard(output, source, execute=True), cancellation_transport() as cancellation:
        try:
            tmp = output / "tmp"
            tmp.mkdir()
            os.environ.update(TMPDIR=str(tmp), TEMP=str(tmp), TMP=str(tmp), PYTHONDONTWRITEBYTECODE="1",
                              GIT_OPTIONAL_LOCKS="0", XDG_CACHE_HOME=str(output / "cache"))
            tempfile.tempdir = str(tmp)
            child_guard(output, source)
            write_json(output / "plan.json", record)
            require_ready_source(source)
            if experiment == "long-context":
                module.preflight(attempt, execute=True)  # local frozen checks before remote volume access
            fresh_attempt(module, experiment, source, attempt)
            if experiment == "canonical-ppl":
                code = run_two_model(module.run, archive_local, output, attempt, cancellation)
            elif experiment == "long-context":
                code = module.run(attempt)
                if code == 0 and not cancellation.cancelled:
                    require_quality_artifacts(output / "artifacts" / f"attempt_{attempt}", attempt)
            else:
                code = run_serving(module)
                if code == 0 and not cancellation.cancelled:
                    require_serving_artifacts(module)
        except KeyboardInterrupt:
            cancellation.request()
            code = 130
        except (Exception, SystemExit) as exc:
            write_json(output / "failure.json", {"status": "INVALID; fail closed", "error": str(exc)})
            code = 1
        cancelled = cancellation.cancelled or code == 130
        if cancelled:
            archive_cancelled(output)
            write_json(output / "cancelled_record.json", {
                "experiment": experiment, "attempt": attempt, "valid": False, "cancelled": True,
                "status": "INVALID/cancelled; no retry or selective rerun",
                "local_client_kill_errors": cancellation.kill_errors,
                "cleanup": "frozen runner cleanup retained; consult artifact record/session",
            })
            code = 130
        write_json(output / "completion.json", {"returncode": code, "cancelled": cancelled,
                   "status": "INVALID/cancelled" if cancelled else "reproduced/unreviewed" if code == 0 else "INVALID"})
        return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=SPECS)
    parser.add_argument("--source-root", type=Path, default=REPOSITORY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--execute", action="store_true", help="explicitly authorize frozen compute; default is a read-only plan")
    args = parser.parse_args(argv)
    if args.attempt < 1:
        parser.error("attempt must be positive")
    try:
        source = args.source_root.resolve()
        if not source.is_dir():
            raise ValueError("source root does not exist")
        output = output_path(args.output_dir, source)
        module, record = plan(source, args.experiment, output)
        if not args.execute:
            print(json.dumps(record, indent=2))
            return 0
        return execute(module, args.experiment, output, source, args.attempt, record)
    except (Exception, SystemExit) as exc:
        print(f"public reproduction refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
