"""Standard-library remote operations, sent through SSH, never installed as a service.

Loaded as a standard-library module by transport.py. The conditional telemetry
import also supports clients already running the earlier concatenated transport.
No caller-supplied shell command is built.
"""

import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

if __package__:
    from .telemetry import errors, fast_snapshot, slow_snapshot

WORK_ROOT = Path("/home/alessio/ML-work")
ARTIFACT_ROOT = WORK_ROOT / "result_deep_storage"
MANAGED_ROOT = ARTIFACT_ROOT / "edge_mcp"
LABEL = "org.clas12.edge-mcp"
MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_LOG_BYTES = 65536


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def execute(args, timeout=15, check=True):
    process = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if check and process.returncode:
        raise RuntimeError(f"{args[0]} exited {process.returncode}: {process.stderr.strip()[-2000:]}")
    return process


def atomic_json(path, value):
    fd, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def valid_id(job_id):
    if not isinstance(job_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", job_id):
        raise ValueError("Invalid job ID")
    return job_id


def below(path, root):
    return path == root or root in path.parents


def remote_path(value, root):
    path = Path(value)
    if not path.is_absolute() or any(c in str(path) for c in ["\x00", "\n", ",", ":"]):
        raise ValueError("Mount paths must be absolute and cannot contain NUL, newline, comma, or colon")
    resolved = path.resolve()
    if not below(resolved, root.resolve()):
        raise ValueError(f"Path must be inside {root}")
    return resolved


def job_directory(job_id):
    path = MANAGED_ROOT / "jobs" / valid_id(job_id)
    if path.is_symlink() or (path.exists() and path.resolve() != path.absolute()):
        raise ValueError("Managed job directory must not contain symlinks")
    return path


@contextmanager
def locked(name):
    MANAGED_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_root = MANAGED_ROOT / "locks"
    lock_root.mkdir(mode=0o700, exist_ok=True)
    with (lock_root / (name + ".lock")).open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another operation holds this job/host lock; retry with the same job ID") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def validate_spec(spec):
    allowed = {"job_id", "command", "image", "mounts", "environment", "expected_artifacts", "gpu", "cpus", "memory_gib", "pids_limit", "timeout_seconds", "ipc_mode", "network"}
    if set(spec) != allowed:
        raise ValueError("Run specification has missing or unknown fields")
    valid_id(spec["job_id"])
    argv = spec["command"]
    if not isinstance(argv, list) or not 1 <= len(argv) <= 256 or not argv[0] or any(not isinstance(x, str) or "\x00" in x for x in argv):
        raise ValueError("Command must be a nonempty argument array without NUL")
    if sum(len(x.encode()) for x in argv) > 65536:
        raise ValueError("Command exceeds 64 KiB")
    image = spec["image"]
    if not isinstance(image, str) or image.startswith("-") or not re.fullmatch(r"[A-Za-z0-9_./:@-]+", image):
        raise ValueError("Invalid image reference")
    if type(spec["gpu"]) is not bool or spec["ipc_mode"] not in ["host", "private"] or spec["network"] not in ["none", "bridge"]:
        raise ValueError("Invalid GPU, IPC, or network setting")
    for name in ["cpus", "memory_gib", "pids_limit", "timeout_seconds"]:
        value = spec[name]
        if value is None:
            continue
        if type(value) not in [int, float] or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {name}")
        if name != "cpus" and type(value) is not int:
            raise ValueError(f"{name} must be an integer")
    env = spec["environment"]
    if not isinstance(env, dict) or len(env) > 100 or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) or not isinstance(v, str) or "\x00" in v for k, v in env.items()):
        raise ValueError("Invalid environment variables")
    if sum(len(k.encode()) + len(v.encode()) for k, v in env.items()) > 65536:
        raise ValueError("Environment exceeds 64 KiB")
    if not isinstance(spec["expected_artifacts"], list) or len(spec["expected_artifacts"]) > 100:
        raise ValueError("Invalid expected artifacts")
    for name in spec["expected_artifacts"]:
        p = PurePosixPath(name)
        if not name or p.is_absolute() or ".." in p.parts or "\x00" in name or name in ["runner_result.json", "stdout.log"]:
            raise ValueError("Expected artifacts must be relative output paths; supervisor files are reserved")
    if not isinstance(spec["mounts"], list) or len(spec["mounts"]) > 24:
        raise ValueError("Invalid mounts")
    targets = [PurePosixPath(p) for p in ["/workspace", "/edge", "/results"]]
    normalized = []
    for mount in spec["mounts"]:
        if set(mount) != {"source", "target", "read_only"} or type(mount["read_only"]) is not bool:
            raise ValueError("Invalid mount specification")
        source = remote_path(mount["source"], WORK_ROOT if mount["read_only"] else ARTIFACT_ROOT)
        if not source.exists():
            raise FileNotFoundError(f"Mount source does not exist: {source}")
        target = PurePosixPath(mount["target"])
        if not target.is_absolute() or ".." in target.parts or any(c in str(target) for c in ["\x00", "\n", ",", ":"]):
            raise ValueError("Invalid container mount target")
        if any(below(target, p) or below(p, target) for p in targets):
            raise ValueError("Container mount targets must not overlap each other or /workspace, /edge, /results")
        if (below(source, MANAGED_ROOT.resolve()) or below(MANAGED_ROOT.resolve(), source)) and not mount["read_only"]:
            raise ValueError("Additional mounts cannot write integration job records")
        if not mount["read_only"] and not os.access(source, os.W_OK):
            raise PermissionError(f"Mount source is not writable: {source}")
        targets.append(target)
        normalized.append({**mount, "source": str(source), "target": str(target)})
    return {**spec, "mounts": normalized}


def make_plan(host, payload):
    spec = validate_spec(payload["spec"])
    source = payload["source"]
    if digest(source["files"]) != source["sha256"]:
        raise ValueError("Source manifest digest mismatch")
    runner = payload["runner"]
    if len(runner.encode()) > 65536:
        raise ValueError("Supervisor is too large")
    info = json.loads(execute(["docker", "image", "inspect", spec["image"]]).stdout)[0]
    image_id = info["Id"]
    plan = {"schema_version": 1, "host": host, "spec": spec, "image_id": image_id,
            "image_architecture": info["Architecture"], "source": {"sha256": source["sha256"], "file_count": len(source["files"]), "git": source.get("git")},
            "execution_user": {"uid": os.getuid(), "gid": os.getgid()},
            "runner_sha256": hashlib.sha256(runner.encode()).hexdigest()}
    plan["fingerprint"] = digest(plan)
    plan["container_name"] = "edge-mcp-" + spec["job_id"]
    plan["job_dir"] = str(job_directory(spec["job_id"]))
    plan["docker_create_argv"] = create_argv(plan)
    return plan


def create_argv(plan):
    spec = plan["spec"]
    root = Path(plan["job_dir"])
    args = ["docker", "create", "--pull=never", "--name", plan["container_name"],
            "--user", f"{plan['execution_user']['uid']}:{plan['execution_user']['gid']}",
            "--label", LABEL + ".managed=1", "--label", LABEL + ".job-id=" + spec["job_id"],
            "--label", LABEL + ".fingerprint=" + plan["fingerprint"],
            "--restart=no", "--init", "--stop-timeout=15", "--network", spec["network"],
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--ipc", spec["ipc_mode"],
            "--log-driver=json-file", "--log-opt=max-size=10m", "--log-opt=max-file=3",
            "--workdir=/workspace", "--env=PYTHONUNBUFFERED=1", "--env=PYTHONDONTWRITEBYTECODE=1",
            "--env=HOME=/tmp", "--env=MPLCONFIGDIR=/tmp/edge-mcp-matplotlib"]
    if spec["cpus"] is not None:
        args.extend(["--cpus", str(spec["cpus"])])
    if spec["memory_gib"] is not None:
        args.extend(["--memory", str(spec["memory_gib"]) + "g"])
    if spec["pids_limit"] is not None:
        args.extend(["--pids-limit", str(spec["pids_limit"])])
    if spec["ipc_mode"] == "private":
        args.append("--shm-size=1g")
    if spec["gpu"]:
        args.extend(["--gpus", "all"])
    for name, value in sorted(spec["environment"].items()):
        args.extend(["--env", name + "=" + value])
    mounts = [{"source": str(root / "source"), "target": "/workspace", "read_only": True},
              {"source": str(root / "runner"), "target": "/edge", "read_only": True},
              {"source": str(root / "output"), "target": "/results", "read_only": False}, *spec["mounts"]]
    for mount in mounts:
        value = f"type=bind,source={mount['source']},target={mount['target']}"
        if mount["read_only"]:
            value += ",readonly"
        args.extend(["--mount", value])
    args.extend([plan["image_id"], "python", "/edge/supervisor.py"])
    return args


def unpack_source(source, destination):
    data = base64.b64decode(source["archive"], validate=True)
    if len(data) > MAX_SOURCE_BYTES + 1024 * 1024:
        raise ValueError("Source archive is too large")
    manifest = source["files"]
    found, size = set(), 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for entry in archive:
            path = PurePosixPath(entry.name)
            if not entry.isfile() or path.is_absolute() or ".." in path.parts or entry.name in found or entry.name not in manifest:
                raise ValueError("Unsafe or unexpected source archive member")
            size += entry.size
            if size > MAX_SOURCE_BYTES or len(found) >= 20000:
                raise ValueError("Source archive exceeds its size/file limit")
            record = manifest[entry.name]
            if entry.size != record["size"] or record["mode"] not in [0o644, 0o755]:
                raise ValueError("Source archive metadata mismatch")
            stream = archive.extractfile(entry)
            contents = stream.read(MAX_SOURCE_BYTES + 1)
            if hashlib.sha256(contents).hexdigest() != record["sha256"]:
                raise ValueError("Source file checksum mismatch")
            target = destination / str(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
            target.chmod(record["mode"])
            found.add(entry.name)
    if found != set(manifest):
        raise ValueError("Source archive is missing manifest files")


def public_plan(plan):
    return {**plan, "docker_create_argv": create_argv(plan), "output_dir": str(Path(plan["job_dir"]) / "output")}


def prepare_run(host, payload, preview=False):
    plan = make_plan(host, payload)
    if preview:
        return {"prepared": False, "preview": True, **public_plan(plan)}
    with locked("job-" + plan["spec"]["job_id"]):
        root = Path(plan["job_dir"])
        if root.exists():
            existing = load_plan(host, plan["spec"]["job_id"])
            if existing["fingerprint"] != plan["fingerprint"]:
                raise ValueError("Job ID already belongs to a different immutable plan; choose a new job ID")
            return {"prepared": True, "already_prepared": True, **public_plan(existing)}
        root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".prepare-", dir=root.parent))
        try:
            for directory in ["source", "runner", "output"]:
                (temporary / directory).mkdir()
            unpack_source(payload["source"], temporary / "source")
            (temporary / "runner/supervisor.py").write_text(payload["runner"])
            atomic_json(temporary / "runner/run_spec.json", plan["spec"])
            atomic_json(temporary / "source_manifest.json", payload["source"]["files"])
            atomic_json(temporary / "plan.json", {**plan, "prepared_at": now()})
            temporary.rename(root)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return {"prepared": True, "already_prepared": False, **public_plan(plan)}


def load_plan(host, job_id):
    root = job_directory(job_id)
    plan = json.loads((root / "plan.json").read_text())
    if plan["host"] != host or plan["spec"]["job_id"] != job_id or Path(plan["job_dir"]) != root:
        raise ValueError("Job record identity mismatch")
    core = {k: v for k, v in plan.items() if k not in ["fingerprint", "container_name", "job_dir", "prepared_at", "docker_create_argv"]}
    if digest(core) != plan["fingerprint"]:
        raise ValueError("Job plan checksum mismatch")
    return plan


def inspect_managed(plan):
    process = execute(["docker", "container", "inspect", plan["container_name"]], check=False)
    if process.returncode:
        if "No such container" in process.stderr or "No such object" in process.stderr:
            return None
        raise RuntimeError("Docker inspect failed: " + process.stderr[-1000:])
    info = json.loads(process.stdout)[0]
    labels = info["Config"].get("Labels") or {}
    required = {LABEL + ".managed": "1", LABEL + ".job-id": plan["spec"]["job_id"], LABEL + ".fingerprint": plan["fingerprint"]}
    if any(labels.get(k) != v for k, v in required.items()) or info["Image"] != plan["image_id"]:
        raise ValueError("Container ownership/image mismatch; refusing to operate on it")
    return info


def optional_json(path):
    if not path.exists():
        return None
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Job metadata file is too large")
    return json.loads(path.read_text())


def artifact_checks(root, expected):
    output = root / "output"
    result = []
    for name in expected:
        path = output / name
        safe = below(path.resolve(), output.resolve())
        present = safe and path.is_file()
        result.append({"path": name, "present": present, "size": path.stat().st_size if present else None})
    return result


def run_status(host, job_id, plan=None):
    plan = plan or load_plan(host, job_id)
    root = Path(plan["job_dir"])
    saved = optional_json(root / "state.json") or {}
    result = optional_json(root / "output/runner_result.json")
    info = inspect_managed(plan)
    artifacts = artifact_checks(root, plan["spec"]["expected_artifacts"])
    complete = all(a["present"] for a in artifacts)
    state, exit_code = "prepared", None
    if info:
        docker_state = info["State"]
        state = docker_state["Status"]
        if state in ["exited", "dead"]:
            exit_code = docker_state["ExitCode"]
            if result and result.get("timed_out"):
                state = "timed_out"
            elif saved.get("stop_requested_at"):
                state = "stopped"
            elif exit_code != 0 or (result and result.get("exit_code") != 0):
                state = "failed"
            elif result and complete:
                state = "succeeded"
            else:
                state = "incomplete"
    elif saved.get("launch_requested_at"):
        state = "unknown"  # Never mistake a removed container for a completed job.
    config = info.get("HostConfig", {}) if info else {}
    effective_limits = None if not info else {
        "cpus": config.get("NanoCpus", 0) / 1e9 or None,
        "memory_bytes": config.get("Memory") or None,
        "pids_limit": config.get("PidsLimit") if (config.get("PidsLimit") or 0) > 0 else None,
        "runtime_deadline_seconds": plan["spec"]["timeout_seconds"],
        "network_mode": config.get("NetworkMode"),
    }
    return {"host": host, "job_id": job_id, "status": state, "container_id": info["Id"] if info else None,
            "container_name": plan["container_name"], "docker_state": info["State"] if info else None,
            "exit_code": exit_code, "runner_result": result, "artifacts": artifacts,
            "artifacts_complete": complete, "scientific_validation": "not_assessed",
            "fingerprint": plan["fingerprint"], "output_dir": str(root / "output"),
            "effective_limits": effective_limits, "control_record": saved, "observed_at": now()}


def verify_snapshot(plan):
    root = Path(plan["job_dir"])
    manifest = json.loads((root / "source_manifest.json").read_text())
    if digest(manifest) != plan["source"]["sha256"]:
        raise ValueError("Source manifest was modified after preparation")
    source_root = root / "source"
    paths = list(source_root.rglob("*"))
    if any(path.is_symlink() for path in paths) or {str(p.relative_to(source_root)) for p in paths if p.is_file()} != set(manifest):
        raise ValueError("Source snapshot file inventory was modified")
    for name, entry in manifest.items():
        path = root / "source" / name
        mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
        if mode != entry["mode"] or path.is_symlink() or not below(path.resolve(), (root / "source").resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Source snapshot was modified: {name}")
    if hashlib.sha256((root / "runner/supervisor.py").read_bytes()).hexdigest() != plan["runner_sha256"]:
        raise ValueError("Supervisor was modified after preparation")
    if json.loads((root / "runner/run_spec.json").read_text()) != plan["spec"]:
        raise ValueError("Runner specification was modified after preparation")


def check_write_conflicts(plan):
    requested = [Path(m["source"]) for m in plan["spec"]["mounts"] if not m["read_only"]]
    if not requested:
        return
    ids = execute(["docker", "ps", "-aq", "--filter", "label=" + LABEL + ".managed=1"]).stdout.split()
    for cid in ids:
        info = json.loads(execute(["docker", "inspect", cid]).stdout)[0]
        if info["Name"].lstrip("/") == plan["container_name"] or info["State"]["Status"] not in ["created", "running", "paused", "restarting"]:
            continue
        for mount in info.get("Mounts", []):
            if mount.get("RW"):
                other = Path(mount["Source"]).resolve()
                if any(below(p, other) or below(other, p) for p in requested):
                    raise ValueError(f"Writable mount overlaps active managed job {info['Name']}")


def launch_run(host, job_id):
    valid_id(job_id)
    # Serialize create/start and conflict checks across all integration clients.
    with locked("host-launch"), locked("job-" + job_id):
        plan = load_plan(host, job_id)
        info = inspect_managed(plan)
        if info and info["State"]["Status"] != "created":
            return {"already_launched": True, **run_status(host, job_id, plan)}
        state_path = Path(plan["job_dir"]) / "state.json"
        state = optional_json(state_path) or {}
        if info is None and state.get("launch_requested_at"):
            raise ValueError("A previous launch was recorded but its container is missing; refusing to recreate an ambiguous job. Inspect this ID before preparing an intentional new attempt.")
        if plan.get("docker_create_argv") != create_argv(plan):
            raise ValueError("Execution policy differs from the prepared Docker command; prepare a new job")
        verify_snapshot(plan)
        # Revalidate mount availability and paths; do not silently resolve a moved symlink.
        if validate_spec(plan["spec"]) != plan["spec"]:
            raise ValueError("Mount resolution changed after preparation")
        check_write_conflicts(plan)
        state.setdefault("launch_requested_at", now())
        atomic_json(state_path, state)
        if info is None:
            execute(create_argv(plan), timeout=20)
        # Recovery from an interrupted create/start reuses the existing container.
        execute(["docker", "start", plan["container_name"]], timeout=20)
        state["started_at"] = now()
        atomic_json(state_path, state)
        return {"already_launched": False, **run_status(host, job_id, plan)}


def stop_run(host, job_id):
    with locked("job-" + valid_id(job_id)):
        plan = load_plan(host, job_id)
        info = inspect_managed(plan)
        if info is None or info["State"]["Status"] in ["exited", "dead"]:
            return {"already_stopped": True, **run_status(host, job_id, plan)}
        state_path = Path(plan["job_dir"]) / "state.json"
        state = optional_json(state_path) or {}
        state["stop_requested_at"] = now()
        atomic_json(state_path, state)
        execute(["docker", "stop", "--time=15", plan["container_name"]], timeout=22)
        return {"already_stopped": False, **run_status(host, job_id, plan)}


def bounded_tail(path, tail):
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - MAX_LOG_BYTES))
        data = stream.read(MAX_LOG_BYTES)
    lines = data.decode("utf-8", errors="replace").splitlines()
    text = "\n".join(lines[-tail:]).encode("utf-8")[-MAX_LOG_BYTES:].decode("utf-8", errors="ignore")
    return {"text": text, "truncated": size > MAX_LOG_BYTES or len(lines) > tail}


def read_logs(host, payload):
    plan = load_plan(host, payload["job_id"])
    tail = payload.get("tail", 100)
    if type(tail) is not int or not 1 <= tail <= 1000:
        raise ValueError("tail must be between 1 and 1000")
    source = payload.get("source", "stdout")
    if source == "stdout":
        path = Path(plan["job_dir"]) / "output/stdout.log"
        if path.is_symlink():
            raise ValueError("Log file must not be a symlink")
        data = bounded_tail(path, tail) if path.exists() else {"text": "", "truncated": False}
    elif source == "docker":
        info = inspect_managed(plan)
        if info is None:
            return {"text": "", "truncated": False, "source": source}
        with tempfile.TemporaryFile() as stream:
            process = subprocess.run(["docker", "logs", "--timestamps", "--tail", str(tail), info["Id"]], stdout=stream, stderr=subprocess.STDOUT, timeout=15)
            if process.returncode:
                raise RuntimeError("Docker log retrieval failed")
            size = stream.tell()
            stream.seek(max(0, size - MAX_LOG_BYTES))
            text = stream.read(MAX_LOG_BYTES).decode("utf-8", errors="replace").encode("utf-8")[-MAX_LOG_BYTES:].decode("utf-8", errors="ignore")
            data = {"text": text, "truncated": size > MAX_LOG_BYTES}
    else:
        raise ValueError("Log source must be stdout or docker")
    return {"host": host, "job_id": payload["job_id"], "source": source, **data}


def read_artifact(host, payload):
    plan = load_plan(host, payload["job_id"])
    output = Path(plan["job_dir"]) / "output"
    name = PurePosixPath(payload["path"])
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("Artifact path must be relative to the job output directory")
    path = output / str(name)
    if not below(path.resolve(), output.resolve()) or not path.is_file():
        raise ValueError("Artifact is missing or outside the job output directory")
    if path.stat().st_size > MAX_LOG_BYTES:
        raise ValueError("Artifact exceeds 64 KiB; read a summary or retrieve it through SSH")
    return {"host": host, "job_id": payload["job_id"], "path": str(name), "text": path.read_text()}


def list_runs(host, limit):
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    root = MANAGED_ROOT / "jobs"
    entries = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) if root.exists() else []
    result = []
    for path in entries:
        if path.name.startswith(".") or not path.is_dir():
            continue
        try:
            result.append(run_status(host, path.name))
        except Exception as exc:
            result.append({"host": host, "job_id": path.name, "status": "record_error", "error": f"{type(exc).__name__}: {exc}"})
        if len(result) >= limit:
            break
    return {"host": host, "runs": result}


def dispatch(request):
    host, action, payload = request["host"], request["action"], request["payload"]
    if host not in ["a02c", "a518"] or os.uname().nodename.split(".")[0] != "edgexpert-" + host:
        raise ValueError("SSH host identity does not match selected host")
    if action == "inspect_host":
        errors.clear()
        data = fast_snapshot()
        if payload.get("details", True):
            data.update(slow_snapshot())
        data.update(host=host, errors=list(errors), observed_at=now())
        return data
    if action in ["preview_run", "prepare_run"]:
        return prepare_run(host, payload, preview=action == "preview_run")
    if action == "get_run_status":
        return run_status(host, payload["job_id"])
    if action == "launch_run":
        return launch_run(host, payload["job_id"])
    if action == "stop_run":
        return stop_run(host, payload["job_id"])
    if action == "read_run_logs":
        return read_logs(host, payload)
    if action == "read_run_artifact":
        return read_artifact(host, payload)
    if action == "list_runs":
        return list_runs(host, payload.get("limit", 20))
    from .remote_explore import dispatch_explore
    return dispatch_explore(host, action, payload)


def serve_request(request):
    try:
        return {"ok": True, "result": dispatch(request)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    print(json.dumps(serve_request(json.load(sys.stdin))))
