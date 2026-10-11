"""Offline adapter checks only: no GPU, Modal client, downloads or publication."""
import json
import io
from contextlib import ExitStack, redirect_stdout
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reproduce as adapter


def verify_preserved_source(source):
    """Only external pinned sources/byte-bound offline fixtures may supply workers."""
    source = Path(source)
    if not source.is_absolute() or not source.is_dir() or adapter.inside(source, adapter.REPOSITORY):
        raise ValueError("offline workers require an existing external preserved source")
    marker = source / '_public_offline_fixture.json'
    if marker.is_file():
        record = json.loads(marker.read_text(encoding='utf-8'))
        if record.get('source_commit') != adapter.PINNED_COMMIT:
            raise ValueError("offline fixture is not bound to the preserved commit")
        files = record.get('files', [])
        paths = [item['path'] for item in files]
        if len(paths) != 6730 or len(set(paths)) != len(paths):
            raise ValueError("offline fixture lacks complete original tracked-path coverage")
        for item in files:
            path = source / item['path']
            if not adapter.inside(path, source) or not path.is_file() or adapter.digest(path) != item['sha256_raw']:
                raise ValueError(f"offline fixture bytes do not match preservation manifest: {item['path']}")
    elif adapter.git(source, 'rev-parse', 'HEAD') != adapter.PINNED_COMMIT:
        raise ValueError("preserved source checkout HEAD differs from the pinned commit")
    return source.resolve()


_PRESERVED_SOURCE = None
def preserved_source():
    global _PRESERVED_SOURCE
    if _PRESERVED_SOURCE is None:
        value = os.environ.get('RABIT_PRESERVED_SOURCE')
        if not value:
            raise ValueError("set RABIT_PRESERVED_SOURCE to an external pinned checkout or complete offline fixture")
        _PRESERVED_SOURCE = verify_preserved_source(value)
    return _PRESERVED_SOURCE


class PublicAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        preserved_source()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("RABIT_TEST_TEMP"))
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name).resolve()
        self.source = self.parent / "source"
        self.source.mkdir()
        self.output = self.parent / "new-run"

    def test_output_requires_absolute_external_fresh_destination(self):
        for value in (Path("relative"), self.source / "run", adapter.REPOSITORY / "run"):
            with self.assertRaises(ValueError):
                adapter.output_path(value, self.source)
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            adapter.output_path(self.output, self.source)

    def test_nonexistent_parent_and_traversal_rejected(self):
        with self.assertRaises(ValueError):
            adapter.output_path(self.parent / "missing" / "run", self.source)
        self.output.mkdir()
        with self.assertRaises(PermissionError):
            adapter.confined(self.output / ".." / "escape", self.output)

    def test_linked_parent_rejected(self):
        from types import SimpleNamespace
        with patch.object(Path, "lstat", return_value=SimpleNamespace(st_mode=0, st_file_attributes=0x400)):
            with self.assertRaises(ValueError):
                adapter.output_path(self.output, self.source)

    def test_guard_confines_open_mkdir_and_rename(self):
        self.output.mkdir()
        with adapter.guard(self.output, self.source):
            (self.output / "valid.txt").write_text("valid")
            with self.assertRaises(PermissionError):
                (self.source / "bad.txt").write_text("bad")
            with self.assertRaises(PermissionError):
                (self.source / "bad-dir").mkdir()
            with self.assertRaises(PermissionError):
                (self.output / "valid.txt").rename(self.source / "bad.txt")
        self.assertFalse((self.source / "bad.txt").exists())

    def test_git_publication_and_shell_commands_denied(self):
        self.output.mkdir()
        for verb in ("add", "commit", "push", "fetch", "checkout", "reset", "config", "update-ref", "clean"):
            with self.assertRaises(PermissionError):
                adapter.check_command(["git", verb], self.output, self.source, True)
        for command in ("git push", ["powershell", "-Command", "git push"], [sys.executable, "-c", "print(1)"]):
            with self.assertRaises(PermissionError):
                adapter.check_command(command, self.output, self.source, True)
        for verb in ("status", "diff", "rev-parse", "cat-file"):
            adapter.check_command(["git", verb], self.output, self.source, False)
        with adapter.guard(self.output, self.source):
            with self.assertRaises(PermissionError):
                subprocess.run(["git", "push"], check=True)

    def test_modal_requires_explicit_execution(self):
        with self.assertRaises(PermissionError):
            adapter.check_command([sys.executable, "-m", "modal", "run", "worker.py"],
                                  self.output, self.source, False)

    def test_dirty_or_unpushed_source_fails_closed(self):
        for answers in ((" M source.py",), ("", "different-head")):
            with patch.object(adapter, "git", side_effect=answers):
                with self.assertRaises(RuntimeError):
                    adapter.require_ready_source(self.source)

    def test_serving_preflight_failure_never_launches_or_changes_gates(self):
        class Module:
            def preflight(self, dry_run):
                self.assertion = dry_run
                raise RuntimeError("required gate failed")
        module = Module()
        with self.assertRaises(RuntimeError):
            adapter.run_serving(module)
        self.assertFalse(module.assertion)

    def test_child_python_inherits_write_and_publication_guard(self):
        self.output.mkdir()
        with patch.dict(os.environ, {"PYTHONDONTWRITEBYTECODE": "1"}):
            adapter.child_guard(self.output, self.source)
            commands = [
                f"from pathlib import Path; Path({str(self.source / 'bad.txt')!r}).write_text('bad')",
                "import subprocess; subprocess.run(['git','push'])",
            ]
            for script in commands:
                result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("PermissionError", result.stderr)
        self.assertFalse((self.source / "bad.txt").exists())

    def test_execution_gate_failure_records_invalid_without_worker(self):
        from types import SimpleNamespace
        module = SimpleNamespace(run=unittest.mock.Mock())
        previous_temp = tempfile.tempdir
        try:
            with patch.dict(os.environ), patch.object(adapter, "require_ready_source", side_effect=RuntimeError("source not ready")):
                code = adapter.execute(module, "canonical-ppl", self.output, self.source, 123, {})
            self.assertEqual(code, 1)
            module.run.assert_not_called()
            self.assertEqual(json.loads((self.output / "completion.json").read_text())["status"], "INVALID")
        finally:
            tempfile.tempdir = previous_temp

    def test_default_cli_is_read_only_plan(self):
        with patch.object(adapter, "plan", return_value=(object(), {"execution": "not executed"})), \
                patch.object(adapter, "execute") as execute:
            self.assertEqual(adapter.main(["canonical-ppl", "--source-root", str(self.source),
                                           "--output-dir", str(self.output)]), 0)
            execute.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_exclusive_record_write_rejects_overwrite(self):
        self.output.mkdir()
        target = self.output / "record.json"
        adapter.write_json(target, {"first": True})
        with self.assertRaises(FileExistsError):
            adapter.write_json(target, {"second": True})
        self.assertEqual(json.loads(target.read_text()), {"first": True})

    def two_model(self, results, archive_failure=False):
        self.output.mkdir()
        called = []
        archived = []
        def run(model, attempt):
            self.assertTrue((self.output / "registration.json").is_file())
            if model == adapter.MODELS[1]:
                self.assertEqual(archived, [adapter.MODELS[0]])
            called.append(model)
            path = self.output / "artifacts" / model / f"attempt_{attempt}"
            path.mkdir(parents=True)
            (path / "raw.json").write_text('{}')
            (path / "result.json").write_text('{}')
            (path / "session.log").write_text('mock worker log\n')
            adapter.write_json(path / "record.json", {
                "model_key": model, "attempt": attempt, "valid": True, "process_ok": True,
                "evaluation": {"valid": True}, "cleanup": {"verified": True},
                "result_sha256": adapter.digest(path / "result.json"),
            })
            result = results[len(called) - 1]
            if isinstance(result, BaseException):
                raise result
            return result
        def archive(path):
            archived.append(path.parent.name)
            if archive_failure:
                raise OSError("archive unavailable")
            adapter.archive_local(path)
        with adapter.guard(self.output, self.source):
            code = adapter.run_two_model(run, archive, self.output, 123)
        return code, called, json.loads((self.output / "two_model_record.json").read_text())

    def test_two_model_success_registered_and_locally_archived(self):
        code, called, record = self.two_model([0, 0])
        self.assertEqual((code, called, record["valid"]), (0, list(adapter.MODELS), True))

    def test_first_model_failure_skips_second_and_invalidates_whole_attempt(self):
        code, called, record = self.two_model([1])
        self.assertEqual((code, called, record["valid"]), (1, [adapter.MODELS[0]], False))
        self.assertEqual(record["models_not_run"], [adapter.MODELS[1]])

    def test_second_model_failure_never_keeps_first_as_valid(self):
        code, called, record = self.two_model([0, 1])
        self.assertEqual((code, called, record["valid"]), (1, list(adapter.MODELS), False))

    def test_exception_preserves_invalid_registration_and_raw_archive(self):
        code, called, record = self.two_model([RuntimeError("failed validity gate")])
        self.assertEqual(code, 1)
        self.assertIn(adapter.MODELS[0], record["errors"])
        self.assertFalse(record["valid"])

    def test_archive_failure_stops_second_model(self):
        code, called, record = self.two_model([0], archive_failure=True)
        self.assertEqual((code, called, record["valid"]), (1, [adapter.MODELS[0]], False))

    def test_zero_exit_without_directory_or_required_artifacts_is_invalid(self):
        for missing in ("directory", "result.json", "record.json", "session.log"):
            with self.subTest(missing=missing):
                output = self.parent / missing.replace('.', '-')
                output.mkdir()
                calls = []
                def run(model, attempt):
                    calls.append(model)
                    if missing != "directory":
                        path = output / "artifacts" / model / f"attempt_{attempt}"
                        path.mkdir(parents=True)
                        for name in ("result.json", "record.json", "session.log"):
                            if name != missing:
                                (path / name).write_text('{}')
                    return 0
                code = adapter.run_two_model(run, adapter.archive_local, output, 123)
                record = json.loads((output / "two_model_record.json").read_text())
                self.assertEqual((code, calls, record["valid"]), (1, [adapter.MODELS[0]], False))

    def test_noop_archive_cannot_authorize_second_model(self):
        original = adapter.require_archive
        with patch.object(adapter, "archive_local", return_value=None):
            code, called, record = self.two_model([0])
        self.assertEqual((code, called, record["valid"]), (1, [adapter.MODELS[0]], False))
        self.assertIs(adapter.require_archive, original)

    def test_keyboard_interrupt_stops_sequence_and_archives_available_outputs(self):
        code, called, record = self.two_model([KeyboardInterrupt()])
        self.assertEqual((code, called, record["valid"], record["cancelled"]),
                         (130, [adapter.MODELS[0]], False, True))
        archive = json.loads((self.output / "cancelled_archive.json").read_text())
        self.assertTrue(archive["cancelled"])
        self.assertTrue(any(p.endswith('/raw.json') for p in archive["sha256"]))

    def test_second_model_keyboard_interrupt_invalidates_and_archives_both(self):
        code, called, record = self.two_model([0, KeyboardInterrupt()])
        self.assertEqual((code, called, record["valid"], record["cancelled"]),
                         (130, list(adapter.MODELS), False, True))
        archive = json.loads((self.output / "cancelled_archive.json").read_text())
        for model in adapter.MODELS:
            self.assertTrue(any(model in p for p in archive["sha256"]))

    def test_archive_interrupt_cancels_before_second_model(self):
        with patch.object(adapter, "archive_local", side_effect=KeyboardInterrupt()):
            code, called, record = self.two_model([0])
        self.assertEqual((code, called, record["cancelled"]), (130, [adapter.MODELS[0]], True))
        self.assertTrue((self.output / "cancelled_archive.json").is_file())

    def test_zero_exit_cannot_validate_missing_artifacts_in_any_execution_mode(self):
        from types import SimpleNamespace
        previous_temp = tempfile.tempdir
        try:
            for mode in adapter.SPECS:
                with self.subTest(mode=mode), patch.dict(os.environ):
                    output = self.parent / mode
                    module = SimpleNamespace(run=unittest.mock.Mock(return_value=0),
                                             preflight=unittest.mock.Mock(),
                                             SESSION_LOG=output / 'missing.log', MANIFEST=output / 'missing-manifest.json',
                                             SUMMARY=output / 'missing-summary.json', INTEGRITY=output / 'missing-integrity.json')
                    with patch.object(adapter, 'require_ready_source'), patch.object(adapter, 'fresh_attempt'), \
                            patch.object(adapter, 'run_serving', return_value=0):
                        code = adapter.execute(module, mode, output, self.source, 123, {})
                    self.assertEqual(code, 1)
                    self.assertEqual(json.loads((output / 'completion.json').read_text())['status'], 'INVALID')
        finally:
            tempfile.tempdir = previous_temp

    def test_keyboard_interrupt_during_setup_still_persists_cancelled_record(self):
        previous_temp = tempfile.tempdir
        try:
            with patch.dict(os.environ), patch.object(adapter, 'child_guard', side_effect=KeyboardInterrupt()):
                code = adapter.execute(object(), 'canonical-ppl', self.output, self.source, 123, {})
            self.assertEqual(code, 130)
            self.assertEqual(json.loads((self.output / 'completion.json').read_text())['status'], 'INVALID/cancelled')
            self.assertFalse(json.loads((self.output / 'cancelled_record.json').read_text())['valid'])
        finally:
            tempfile.tempdir = previous_temp

    def test_cancelled_transport_never_starts_another_worker(self):
        with patch.object(subprocess, 'Popen') as popen:
            with adapter.cancellation_transport() as cancellation:
                cancellation.request()
                with self.assertRaises(KeyboardInterrupt):
                    subprocess.Popen([sys.executable, '-m', 'modal', 'run', 'never.py'])
            popen.assert_not_called()

    def test_artifact_record_must_bind_valid_result_and_verified_cleanup(self):
        path = self.parent / 'evidence'
        path.mkdir()
        (path / 'result.json').write_text('{}')
        (path / 'session.log').write_text('mock log\n')
        valid = {'valid': True, 'process_ok': True, 'attempt': 123,
                 'model_key': adapter.MODELS[0], 'evaluation': {'valid': True},
                 'cleanup': {'verified': True}, 'result_sha256': adapter.digest(path / 'result.json')}
        mutations = ({'valid': False}, {'cleanup': {'verified': False}},
                     {'result_sha256': 'unbound'}, {'attempt': 124}, {'model_key': adapter.MODELS[1]})
        for mutation in mutations:
            (path / 'record.json').write_text(json.dumps({**valid, **mutation}))
            with self.assertRaises(RuntimeError):
                adapter.require_quality_artifacts(path, 123, adapter.MODELS[0])

    def test_incomplete_archive_is_not_successful_archival(self):
        path = self.parent / 'evidence'
        path.mkdir()
        (path / 'raw.json').write_text('{}')
        adapter.write_json(path / 'public_archive.json', {'sha256': {}})
        with self.assertRaises(RuntimeError):
            adapter.require_archive(path)

    def test_migrated_public_tree_cannot_supply_preserved_test_workers(self):
        with self.assertRaises(ValueError):
            verify_preserved_source(adapter.REPOSITORY)

    def test_external_fixture_requires_pinned_commit_and_full_coverage(self):
        marker = self.source / '_public_offline_fixture.json'
        for record in ({'source_commit': 'wrong'}, {'source_commit': adapter.PINNED_COMMIT, 'files': []}):
            marker.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                verify_preserved_source(self.source)

    def test_historical_attempt_number_refused(self):
        path = self.source / "results/mlsys2027/canonical_quality_v2/continuation_ppl/llama3_1_8b/attempt_2"
        path.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            adapter.fresh_attempt(object(), "canonical-ppl", self.source, 2)

    def test_remote_freshness_fails_closed_without_modal_or_on_collision(self):
        class Module:
            ARTIFACT_VOLUME = "frozen-volume"
        for payload in ('{}', '[{"path":"attempt_123/rows.jsonl"}]',
                        '[{"filename":"attempt_123"}]', '[{"Filename":"/attempt_123/rows.jsonl"}]',
                        '[{"filename":null}]', '[{"filename":""}]',
                        '[{"filename":"other", "Filename":"attempt_123"}]'):
            with patch.object(adapter.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, payload)):
                with self.assertRaises((RuntimeError, FileExistsError)):
                    adapter.fresh_attempt(Module(), "long-context", self.source, 123)
        for payload in ('[]', '[{"filename":"attempt_122"}]', '[{"Filename":"attempt_124"}]'):
            with patch.object(adapter.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, payload)):
                adapter.fresh_attempt(Module(), "long-context", self.source, 123)
        with patch.object(adapter.subprocess, "run", side_effect=OSError("unavailable")):
            with self.assertRaises(OSError):
                adapter.fresh_attempt(Module(), "long-context", self.source, 123)

    def test_output_alias_preserves_declaration_but_writes_externally(self):
        self.output.mkdir()
        logical = self.source / "results/old/manifest.json"
        mapped = adapter.OutputAlias(self.output / "manifest.json", logical, self.source)
        self.assertEqual(mapped.relative_to(self.source), Path("results/old/manifest.json"))
        with adapter.guard(self.output, self.source):
            mapped.write_text('{}')
        self.assertTrue((self.output / "manifest.json").is_file())
        self.assertFalse(logical.exists())

    def test_real_frozen_protocol_identity_for_all_three_modes(self):
        # Offline identity remains testable after the public adapter is committed;
        # it does not require executing against this checkout's newer HEAD.
        for experiment in adapter.SPECS:
            module, original = adapter.load_frozen(preserved_source(), experiment)
            adapter.bind_outputs(module, experiment, self.output, preserved_source())
            protocol = preserved_source() / "benchmarks/mlsys2027" / adapter.SPECS[experiment][1]
            self.assertEqual(module.load_protocol(), json.loads(original))
            self.assertEqual(protocol.read_bytes(), original)
            self.assertFalse(self.output.exists())

    def test_frozen_ppl_statistics_and_gates_replay_stored_raw_results(self):
        module, _ = adapter.load_frozen(preserved_source(), "canonical-ppl")
        base = preserved_source() / "results/mlsys2027/canonical_quality_v2/continuation_ppl"
        expected = json.loads((base / "two_model_attempt_2/summary.json").read_text())
        for model in adapter.MODELS:
            raw = json.loads((base / model / "attempt_2/result.json").read_text())
            evaluated = module.evaluate(raw, model, module.legacy_reference(model))
            self.assertTrue(evaluated["valid"])
            stats = evaluated["statistics"]
            for field in ("bf16_ppl", "rabit_ppl", "delta_pct"):
                self.assertAlmostEqual(stats[field], expected["models"][model][field], places=10)
            for field, position in (("ci_low", 0), ("ci_high", 1)):
                self.assertAlmostEqual(stats[field], expected["models"][model]["delta_pct_ci95_paired_bootstrap"][position], places=10)

    def test_frozen_long_context_statistics_and_gates_replay_stored_results(self):
        module, _ = adapter.load_frozen(preserved_source(), "long-context")
        base = preserved_source() / "results/mlsys2027/canonical_quality_v2/long_context/attempt_1"
        evaluated = module.evaluate(json.loads((base / "result.json").read_text(encoding="utf-8")))
        expected = json.loads((base / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(evaluated["valid"])
        for scorer in ("primary_legacy_scorer", "secondary_official_scorer"):
            for field in ("bf16", "rabit", "delta_points", "ci_low", "ci_high"):
                self.assertAlmostEqual(evaluated["statistics"]["hotpotqa"][scorer][field], expected["hotpotqa"][scorer][field], places=10)

    def test_frozen_serving_parser_and_statistics_replay_stored_session(self):
        module, _ = adapter.load_frozen(preserved_source(), "matched-serving")
        base = preserved_source() / "results/mlsys2027/external_baseline/exp13"
        evaluated = module.analyze((base / "modal_session.log").read_text(encoding="utf-8"))
        expected = json.loads((base / "matched_capacity_latency_summary.json").read_text())
        self.assertTrue(evaluated["integrity"]["passed"])
        self.assertEqual(evaluated["summary"]["latency_primary_cross_leg"], expected["latency_primary_cross_leg"])
        self.assertEqual(evaluated["summary"]["capacity_ratios"], expected["capacity_ratios"])


class FrozenRunnerExecutionTests(unittest.TestCase):
    """Actual frozen preflight/run/cleanup/evaluation, only transports simulated.

    Git cleanliness/static-process responses simulate a prepared source checkout.
    Modal subprocesses use preserved raw-result/log fixtures, never a remote client.
    Unknown commands fail instead of falling back to live execution.
    """
    @classmethod
    def setUpClass(cls):
        preserved_source()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("RABIT_TEST_TEMP"))
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name).resolve()

    def execute_frozen(self, mode, behavior="success"):
        module, _ = adapter.load_frozen(preserved_source(), mode)
        output = self.parent / f"{mode}-{behavior}"
        adapter.bind_outputs(module, mode, output, preserved_source())
        calls, clients, active = [], [], {}
        fixtures = preserved_source() / "results/mlsys2027"
        serving_log = (fixtures / "external_baseline/exp13/modal_session.log").read_text(encoding="utf-8")

        def run(argv, **kwargs):
            args = [os.fspath(a) for a in argv]
            calls.append(args)
            if args[0] == "git":
                verb = args[1] if args[1] != "-c" else args[3]
                if verb == "archive":
                    Path(args[args.index('-o') + 1]).write_bytes(b'offline mock snapshot')
                    value = ""
                elif verb in ("status", "diff", "merge-base"):
                    value = ""
                elif verb == "rev-parse":
                    value = (module.UPSTREAM_ATTN_UTILS_BLOB if mode == "matched-serving"
                             and args[-1].startswith('HEAD:') and args[-1] != 'HEAD:vllm-kvquant'
                             else adapter.PINNED_COMMIT)
                else:
                    raise AssertionError(f"unregistered offline Git operation: {args}")
            elif len(args) == 2 and args[1].endswith(('_ppl_harness.py', '_longctx_harness.py')):
                value = "1/1 passed\n"
            elif args[1:3] == ["-m", "modal"]:
                if args[3:5] == ["app", "list"]:
                    value = json.dumps(list(active.values()))
                elif args[3:5] == ["app", "stop"]:
                    active[args[-1]].update(State="stopped", Tasks="0")
                    value = ""
                elif args[3:5] == ["volume", "ls"]:
                    value = "[]"
                elif args[3:5] == ["volume", "get"]:
                    Path(args[-1]).write_text('offline incremental fixture\n', encoding='utf-8')
                    value = ""
                else:
                    raise AssertionError(f"unregistered offline Modal operation: {args}")
            else:
                raise AssertionError(f"live child process forbidden in offline execution: {args}")
            return subprocess.CompletedProcess(args, 0, value, "")

        class Client:
            def __init__(client, argv, **kwargs):
                args = [os.fspath(a) for a in argv]
                if args[1:4] != ["-m", "modal", "run"]:
                    raise AssertionError(f"live process forbidden: {args}")
                calls.append(args)
                clients.append(client)
                client.killed = False
                client.returncode = None
                client.waits = 0
                client.app = 'ap-' + str(len(clients)).zfill(24)
                if mode != "matched-serving":
                    active[client.app] = {"App ID": client.app, "Description": module.APP_NAME,
                                          "State": "stopped", "Tasks": "0"}
                    environment = kwargs['env']
                    if behavior != "missing-result":
                        if mode == "canonical-ppl":
                            model = args[args.index('--model-key') + 1]
                            raw = fixtures / 'canonical_quality_v2/continuation_ppl' / model / 'attempt_2/result.json'
                            destination = environment['CANONICAL_PPL_RESULT_PATH']
                        else:
                            raw = fixtures / 'canonical_quality_v2/long_context/attempt_1/result.json'
                            destination = environment['CANONICAL_LONGCTX_RESULT_PATH']
                        Path(destination).write_bytes(raw.read_bytes())
                    log = '' if behavior == "empty-log" else client.app + ' offline completed\n'
                else:
                    log = serving_log
                client.stdout = io.StringIO(log)
                if behavior == "cancel-stream":
                    class InterruptedStream:
                        def __iter__(stream):
                            yield 'partial offline log\n'
                            raise KeyboardInterrupt()
                    client.stdout = InterruptedStream()

            def poll(client):
                return client.returncode

            def kill(client):
                client.killed = True
                client.returncode = -9

            def wait(client, **kwargs):
                client.waits += 1
                if behavior == "cancel" and client.waits == 1:
                    raise KeyboardInterrupt()
                if behavior == "signal-cancel" and client.waits == 1:
                    signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
                client.returncode = -9 if client.killed else 0
                return client.returncode

        previous_temp = tempfile.tempdir
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ))
                os.environ.pop('VLLM_USE_V2_MODEL_RUNNER', None)
                stack.enter_context(patch.object(subprocess, 'run', side_effect=run))
                stack.enter_context(patch.object(subprocess, 'Popen', side_effect=Client))
                stack.enter_context(redirect_stdout(io.StringIO()))
                if mode != "matched-serving":
                    cleanup = stack.enter_context(patch.object(module, 'poll_cleanup', wraps=module.poll_cleanup))
                else:
                    cleanup = None
                code = adapter.execute(module, mode, output, preserved_source(), 12345, {"offline": True})
                cleanup_calls = cleanup.call_count if cleanup else None
        finally:
            tempfile.tempdir = previous_temp
        return code, output, clients, calls, cleanup_calls

    def test_actual_frozen_two_model_ppl_execution_with_mocked_modal(self):
        code, output, clients, calls, cleanup_calls = self.execute_frozen('canonical-ppl')
        self.assertEqual((code, len(clients), cleanup_calls), (0, 2, 2))
        record = json.loads((output / 'two_model_record.json').read_text())
        self.assertTrue(record['valid'])
        self.assertTrue(all(record['archived'].values()))
        for model in adapter.MODELS:
            path = output / 'artifacts' / model / 'attempt_12345'
            adapter.require_quality_artifacts(path, 12345, model)
            adapter.require_archive(path)
            frozen = json.loads((path / 'record.json').read_text())
            self.assertTrue(frozen['cleanup']['verified'])
            self.assertTrue(all(frozen['preflight'].values()))
        self.assertFalse(any(args[0] == 'git' and args[1] in ('add', 'commit', 'push') for args in calls))

    def test_actual_frozen_long_context_execution_with_mocked_modal(self):
        code, output, clients, _, cleanup_calls = self.execute_frozen('long-context')
        self.assertEqual((code, len(clients), cleanup_calls), (0, 1, 1))
        path = output / 'artifacts/attempt_12345'
        adapter.require_quality_artifacts(path, 12345)
        record = json.loads((path / 'record.json').read_text())
        self.assertTrue(record['cleanup']['verified'])
        self.assertTrue(record['incremental_rows_copied'])
        self.assertTrue(all(record['preflight'].values()))

    def test_actual_frozen_matched_serving_execution_with_mocked_modal(self):
        code, output, clients, calls, _ = self.execute_frozen('matched-serving')
        self.assertEqual((code, len(clients)), (0, 1))
        module, _ = adapter.load_frozen(preserved_source(), 'matched-serving')
        adapter.require_serving_artifacts(module)
        record = json.loads((output / 'artifacts/integrity_check.json').read_text())
        self.assertTrue(record['passed'])
        command = next(args for args in calls if args[1:4] == ['-m', 'modal', 'run'])
        self.assertEqual(command[command.index('--legs') + 1], module.legs_arg())
        self.assertEqual(command[command.index('--reps-per-leg') + 1], str(module.REPS_PER_LEG))

    def test_actual_frozen_zero_exit_missing_result_is_invalid(self):
        for mode in ('canonical-ppl', 'long-context'):
            with self.subTest(mode=mode):
                code, output, clients, _, cleanup_calls = self.execute_frozen(mode, 'missing-result')
                self.assertEqual((code, len(clients), cleanup_calls), (1, 1, 1))
                self.assertEqual(json.loads((output / 'completion.json').read_text())['status'], 'INVALID')

    def test_actual_frozen_zero_exit_empty_ppl_log_stops_second_model(self):
        code, output, clients, _, cleanup_calls = self.execute_frozen('canonical-ppl', 'empty-log')
        self.assertEqual((code, len(clients), cleanup_calls), (1, 1, 1))
        record = json.loads((output / 'two_model_record.json').read_text())
        self.assertFalse(record['valid'])
        self.assertEqual(record['models_not_run'], [adapter.MODELS[1]])

    def test_actual_frozen_interrupt_retains_cleanup_and_cancelled_records_all_modes(self):
        for mode in adapter.SPECS:
            with self.subTest(mode=mode):
                code, output, clients, _, cleanup_calls = self.execute_frozen(mode, 'cancel')
                self.assertEqual((code, len(clients)), (130, 1))
                self.assertTrue(clients[0].killed)
                self.assertGreaterEqual(clients[0].waits, 2)
                record = json.loads((output / 'cancelled_record.json').read_text())
                self.assertTrue(record['cancelled'])
                self.assertFalse(record['valid'])
                self.assertEqual(json.loads((output / 'completion.json').read_text())['status'], 'INVALID/cancelled')
                archive = json.loads((output / 'cancelled_archive.json').read_text())
                self.assertTrue(archive['sha256'])
                for name, expected in archive['sha256'].items():
                    self.assertEqual(adapter.digest(output / name), expected)
                if mode != 'matched-serving':
                    self.assertEqual(cleanup_calls, 1)
                    path = (output / 'artifacts' / adapter.MODELS[0] / 'attempt_12345'
                            if mode == 'canonical-ppl' else output / 'artifacts/attempt_12345')
                    frozen = json.loads((path / 'record.json').read_text())
                    self.assertTrue(frozen['cleanup']['verified'])
                    self.assertFalse(frozen['valid'])
                if mode == 'canonical-ppl':
                    attempt = json.loads((output / 'two_model_record.json').read_text())
                    self.assertTrue(attempt['cancelled'])
                    self.assertEqual(attempt['models_not_run'], [adapter.MODELS[1]])

    def test_sigint_handler_allows_actual_frozen_cleanup_and_stops_sequence(self):
        previous_handler = signal.getsignal(signal.SIGINT)
        code, output, clients, _, cleanup_calls = self.execute_frozen('canonical-ppl', 'signal-cancel')
        self.assertEqual((code, len(clients), cleanup_calls), (130, 1, 1))
        self.assertTrue(clients[0].killed)
        self.assertEqual(signal.getsignal(signal.SIGINT), previous_handler)
        record = json.loads((output / 'artifacts' / adapter.MODELS[0] / 'attempt_12345/record.json').read_text())
        self.assertTrue(record['cleanup']['verified'])
        self.assertFalse(record['valid'])

    def test_actual_serving_stream_interrupt_preserves_frozen_kill_and_reap(self):
        code, output, clients, _, _ = self.execute_frozen('matched-serving', 'cancel-stream')
        self.assertEqual((code, len(clients)), (130, 1))
        self.assertTrue(clients[0].killed)
        self.assertGreaterEqual(clients[0].waits, 1)
        self.assertEqual((output / 'artifacts/modal_session.log').read_text(), 'partial offline log\n')
        record = json.loads((output / 'cancelled_record.json').read_text())
        self.assertTrue(record['cancelled'])
        self.assertFalse(record['valid'])


if __name__ == "__main__":
    unittest.main(verbosity=2)
