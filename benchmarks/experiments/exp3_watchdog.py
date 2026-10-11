"""
MLSys 2027 Experiment 3 -- hard per-process watchdog (POSIX; runs INSIDE the
Modal container, imported by exp3_deployment_modal.py).

run_with_watchdog() launches a command in its OWN session / process group
(start_new_session=True), relays its combined stdout/stderr line by line with
a prefix, and enforces a hard wall-clock timeout:

  * on timeout: SIGTERM to the whole process group, a short grace period,
    then SIGKILL to the whole group (this includes vLLM EngineCore children,
    which inherit the group), reap the leader, and report the outcome;
  * on normal exit: any process still left in the group (e.g. an orphaned
    EngineCore) is also SIGKILLed and reported, so it cannot hold GPU memory
    into the next leg.

It never retries. The caller decides to abort the experiment on timeout.
No Modal or vLLM imports, so it can be tested on any Linux host.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def group_members(pgid: int) -> list[dict]:
    """Live (non-zombie) processes whose process group is `pgid`, via /proc."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text()
        except OSError:
            continue
        # comm may contain spaces/parentheses: split after the LAST ')'.
        rest = stat[stat.rfind(")") + 2:].split()
        state, pgrp = rest[0], int(rest[2])
        if pgrp == pgid and state != "Z":
            out.append({"pid": int(d.name), "state": state})
    return out


def _signal_group(pgid: int, sig: int) -> bool:
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False


def run_with_watchdog(cmd: list[str], prefix: str, timeout_s: float, label: str,
                      grace_s: float = 10.0) -> dict:
    """Run `cmd` under a hard timeout. Returns metadata; never raises on timeout
    (the caller must check result['timed_out'] and abort)."""
    if os.name != "posix":
        raise RuntimeError("run_with_watchdog requires POSIX process groups")
    t0 = time.monotonic()
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        start_new_session=True,
    )
    pgid = os.getpgid(p.pid)
    sid = os.getsid(p.pid)

    def relay() -> None:
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(f"{prefix}{line}")
            sys.stdout.flush()

    reader = threading.Thread(target=relay, daemon=True)
    reader.start()

    timed_out = False
    signals_sent: list[str] = []
    try:
        returncode = p.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        if _signal_group(pgid, signal.SIGTERM):
            signals_sent.append("SIGTERM")
        try:
            p.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            pass
        if _signal_group(pgid, signal.SIGKILL):
            signals_sent.append("SIGKILL")
        returncode = p.wait()  # reap the leader

    # Anything still alive in the group (timeout path or orphaned children
    # after a normal exit) is killed, then re-checked.
    leftover_before = group_members(pgid)
    if leftover_before:
        if _signal_group(pgid, signal.SIGKILL):
            signals_sent.append("SIGKILL(leftover)")
        deadline = time.monotonic() + grace_s
        while group_members(pgid) and time.monotonic() < deadline:
            time.sleep(0.2)
    leftover_after = group_members(pgid)
    reader.join(timeout=30)

    return {
        "label": label,
        "cmd": cmd,
        "pid": p.pid,
        "pgid": pgid,
        "own_session": sid == p.pid and pgid == p.pid,
        "timeout_s": timeout_s,
        "grace_s": grace_s,
        "elapsed_s": round(time.monotonic() - t0, 3),
        "timed_out": timed_out,
        "returncode": returncode,
        "signals_sent": signals_sent,
        "group_processes_after_leader_exit": leftover_before,
        "group_processes_remaining": leftover_after,
        "reader_finished": not reader.is_alive(),
    }
