"""Detached, versioned host worker for a Docker environment build."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def process_token(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def main():
    root = Path(sys.argv[1]).resolve()
    plan = json.loads((root / "plan.json").read_text())
    cancelled = False
    child = None

    def stop(*_):
        nonlocal cancelled
        cancelled = True
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save(root / "worker.json", {"pid": os.getpid(), "process_token": process_token(os.getpid()),
         "fingerprint": plan["fingerprint"], "started_utc": datetime.now(timezone.utc).isoformat()})
    result = {"started_unix": time.time(), "fingerprint": plan["fingerprint"]}
    try:
        # Pin the default local Docker builder without changing user-wide settings.
        env = {**os.environ, "BUILDX_BUILDER": "default"}
        child = subprocess.Popen(plan["build_argv"], stdin=subprocess.DEVNULL, env=env)
        stopped_at = None
        while child.poll() is None:
            if cancelled:
                if stopped_at is None:
                    child.terminate()
                    stopped_at = time.monotonic()
                elif time.monotonic() - stopped_at > 10:
                    child.kill()
            time.sleep(.1)
        result["exit_code"] = child.wait()
        result["status"] = "stopped" if cancelled else ("succeeded" if result["exit_code"] == 0 else "failed")
        if result["status"] == "succeeded":
            image_id = (root / "image.id").read_text().strip()
            info = json.loads(subprocess.check_output(["docker", "image", "inspect", image_id], text=True))[0]
            labels = info["Config"].get("Labels") or {}
            if labels.get("org.clas12.edge-mcp.environment-id") != plan["environment_id"] or labels.get("org.clas12.edge-mcp.fingerprint") != plan["fingerprint"]:
                raise ValueError("Built image labels do not match the environment plan")
            base = json.loads(subprocess.check_output(["docker", "image", "inspect", plan["base_image_id"]], text=True))[0]
            layers = base["RootFS"]["Layers"]
            if info["RootFS"]["Layers"][:len(layers)] != layers or info["Architecture"] != base["Architecture"]:
                raise ValueError("Built image does not extend the pinned base image")
            result["image_id"] = info["Id"]
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()
        result.update(status="stopped" if cancelled else "failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
    result["finished_unix"] = time.time()
    save(root / "result.json", result)
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
