"""One reusable in-container supervisor; no SSH connection owns the job."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def main():
    spec = json.loads(Path("/edge/run_spec.json").read_text())
    output = Path("/results")
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    started = time.time()
    started_monotonic = time.monotonic()
    timeout = spec["timeout_seconds"]
    result = {"started_unix": started, "timed_out": False, "stop_requested": False}
    child = None
    try:
        child = subprocess.Popen(spec["command"], cwd="/workspace", start_new_session=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        def copy_output():
            with (output / "stdout.log").open("ab", buffering=0) as stream:
                while data := os.read(child.stdout.fileno(), 8192):
                    stream.write(data)
                    try:
                        os.write(sys.stdout.fileno(), data)
                    except (BrokenPipeError, OSError):
                        pass

        copier = threading.Thread(target=copy_output, daemon=True)
        copier.start()
        while child.poll() is None:
            expired = timeout is not None and time.monotonic() - started_monotonic >= timeout
            if stop.is_set() or expired:
                result.update(timed_out=expired, stop_requested=stop.is_set())
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                # Kill the entire group even if its parent exits before descendants.
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        os.killpg(child.pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.1)
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            stop.wait(0.1)
        rc = child.wait()
        copier.join(timeout=2)
        # Docker removes all remaining container processes on exit as well.
        result["exit_code"] = 124 if result["timed_out"] else (143 if result["stop_requested"] else (rc if rc >= 0 else 128 - rc))
    except Exception as exc:
        result.update(exit_code=1, error=f"{type(exc).__name__}: {exc}")
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    result["finished_unix"] = time.time()
    temporary = output / ".runner_result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output / "runner_result.json")
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
