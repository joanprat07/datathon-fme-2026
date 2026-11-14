"""Keeps the live loop running: starts it and starts it again if it ever stops.

    python -m twin.supervisor            # same as `python -m twin.live`, but restarted on a crash
    python -m twin.supervisor --dry      # arguments are passed on to twin.live

A restart reloads the last snapshot (`python -m twin.build live`) and replays the organic events since
then, so answers resume in about the time the state takes to load. Stop it with Ctrl+C.
"""
from __future__ import annotations

import subprocess
import sys
import time


def main() -> None:
    cmd = [sys.executable, "-m", "twin.live", *sys.argv[1:]]
    restarts, wait = 0, 0.5
    while True:
        started = time.time()
        print(f"[supervisor] starting: {' '.join(cmd)} (restarts so far: {restarts})", flush=True)
        try:
            code = subprocess.call(cmd)
        except KeyboardInterrupt:
            print("[supervisor] stopped by the user", flush=True)
            return
        if code == 0:      # a test run (--seconds) that ended on its own
            return
        wait = 0.5 if time.time() - started > 120 else min(wait * 2, 20.0)   # crashing at start: wait longer each time
        restarts += 1
        print(f"[supervisor] the loop stopped (exit code {code}); starting again in {wait:.1f} s", flush=True)
        try:
            time.sleep(wait)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
