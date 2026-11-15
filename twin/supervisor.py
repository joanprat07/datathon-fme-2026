"""Keeps the live loop running: starts it, and starts it again if it stops OR freezes.

    python -m twin.supervisor            # same as `python -m twin.live`, but restarted on a crash or a freeze
    python -m twin.supervisor --dry      # arguments are passed on to twin.live
    .\\start_live.ps1                     # the way to start it for the final: its own window, one copy only

A restart reloads the last snapshot (`python -m twin.build live`) and replays the organic events since
then, so answers resume in about the time the state takes to load. Stop it with Ctrl+C.

Freeze detection: twin.live rewrites a heartbeat file every ~5 s from its main loop. If the file is
older than STALE_S, or never appears within STARTUP_S, the whole process tree is killed and started
again. (On Windows the venv's python.exe is a launcher with the real interpreter as its child, so
the tree is killed with taskkill /T: killing only the launcher would leave a second loop running.)

Everything the loop prints is also appended to artifacts/logs/live_<date>.log, so its output can be
read even if this window is closed. While it runs, Windows is asked not to sleep (the request ends
with the process; closing the laptop lid can still suspend it).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time

from twin.rt import ARTIFACTS

STARTUP_S = 300      # loading the state + restoring from RawTree: normally ~30-60 s
STALE_S = 60         # the main loop beats every ~5 s; nothing legitimate blocks it for a minute


def keep_awake() -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        except Exception as e:
            print(f"[supervisor] could not ask Windows to stay awake: {e}", flush=True)


def kill_tree(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        subprocess.call(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        proc.kill()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass


def pump(proc: subprocess.Popen, log_path) -> None:
    """Copy the loop's output to this console and to the log file."""
    with open(log_path, "a", encoding="utf-8") as log:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            try:
                log.write(line)
                log.flush()
            except OSError:
                pass


def run_once(cmd: list[str], heartbeat: str, log_path) -> int:
    """Runs the loop until it exits (its exit code) or freezes (-1)."""
    try:
        os.remove(heartbeat)
    except OSError:
        pass
    env = dict(os.environ, TWIN_HEARTBEAT=heartbeat, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    reader = threading.Thread(target=pump, args=(proc, log_path), daemon=True)
    reader.start()
    started = time.time()
    try:
        while True:
            try:
                code = proc.wait(timeout=5)
                reader.join(timeout=5)
                return code
            except subprocess.TimeoutExpired:
                pass
            try:
                age = time.time() - os.path.getmtime(heartbeat)
                frozen = age > STALE_S
                why = f"no heartbeat for {age:.0f} s"
            except OSError:
                frozen = time.time() - started > STARTUP_S
                why = f"no heartbeat {STARTUP_S} s after starting"
            if frozen:
                print(f"[supervisor] {time.strftime('%H:%M:%S')} the loop looks frozen ({why}): killing it", flush=True)
                kill_tree(proc)
                return -1
    except KeyboardInterrupt:
        kill_tree(proc)
        raise


def main() -> None:
    cmd = [sys.executable, "-m", "twin.live", *sys.argv[1:]]
    heartbeat = os.path.join(tempfile.gettempdir(), f"twin_heartbeat_{os.getpid()}.txt")
    logs = ARTIFACTS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"live_{time.strftime('%Y%m%d')}.log"
    keep_awake()
    restarts, wait = 0, 0.5
    while True:
        started = time.time()
        print(f"[supervisor] {time.strftime('%H:%M:%S')} starting: {' '.join(cmd)} (restarts so far: {restarts}) "
              f"| log {log_path}", flush=True)
        try:
            code = run_once(cmd, heartbeat, log_path)
        except KeyboardInterrupt:
            print("[supervisor] stopped by the user", flush=True)
            return
        if code == 0:      # a test run (--seconds) that ended on its own
            return
        wait = 0.5 if time.time() - started > 120 else min(wait * 2, 20.0)   # crashing at start: wait longer each time
        restarts += 1
        print(f"[supervisor] {time.strftime('%H:%M:%S')} the loop stopped (exit code {code}); "
              f"starting again in {wait:.1f} s", flush=True)
        try:
            time.sleep(wait)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
