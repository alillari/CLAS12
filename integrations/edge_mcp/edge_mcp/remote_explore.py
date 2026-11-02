"""Exploratory primitives; standard library only on the remote SSH host."""

import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time

from . import remote as core
from .snapshot import source_snapshot

MAX_SCAN = 5000


def checked_path(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("Use an absolute remote path")
    path = path.resolve(strict=True)
    if not core.below(path, core.WORK_ROOT.resolve()):
        raise ValueError(f"Inspection/source paths must be inside {core.WORK_ROOT}")
    return path


def integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def walk_files(root, recursive, include_hidden, state):
    pending = [root]
    deadline = time.monotonic() + 5
    while pending:
        directory = pending.pop()
        # Bound discovery even for exceptionally large dataset directories.
        entries = []
        with os.scandir(directory) as stream:
            for item in stream:
                state["scanned"] += 1
                if state["scanned"] > MAX_SCAN or time.monotonic() > deadline:
                    state["scan_truncated"] = True
                    break
                entries.append(item)
        directories = []
        for item in sorted(entries, key=lambda x: x.name):
            if item.name == ".git" or (not include_hidden and item.name.startswith(".")):
                continue
            path = Path(item.path)
            yield path
            if recursive and item.is_dir(follow_symlinks=False):
                directories.append(path)
        pending.extend(reversed(directories))
        if state["scan_truncated"]:
            return


def list_remote_files(payload):
    root = checked_path(payload["path"])
    if not root.is_dir():
        raise ValueError("Listing requires a directory")
    limit = integer(payload.get("limit", 100), 1, 500, "limit")
    offset = integer(payload.get("offset", 0), 0, MAX_SCAN, "offset")
    state = {"scanned": 0, "scan_truncated": False}
    entries, matched = [], 0
    pattern = payload.get("pattern", "*")
    more = False
    for path in walk_files(root, payload.get("recursive", False), payload.get("include_hidden", False), state):
        relative = str(path.relative_to(root))
        if not fnmatch.fnmatch(relative, pattern):
            continue
        matched += 1
        if matched <= offset:
            continue
        if len(entries) == limit:
            more = True
            break
        stat = path.lstat()
        kind = "symlink" if path.is_symlink() else ("directory" if path.is_dir() else "file")
        entries.append({"path": str(path), "relative_path": relative, "type": kind, "size": stat.st_size,
                        "mtime_ns": str(stat.st_mtime_ns)})
    return {"path": str(root), "entries": entries, "next_offset": offset + len(entries) if more else None,
            "truncated": more or state["scan_truncated"], **state,
            "note": "Refine the directory/pattern if the scan budget is exhausted; symlink directories are not followed."}


def read_remote_file(payload):
    path = checked_path(payload["path"])
    if not path.is_file():
        raise ValueError("Read requires a regular file")
    offset = integer(payload.get("offset", 0), 0, 2**63 - 1, "offset")
    maximum = integer(payload.get("max_bytes", 65536), 1, 65536, "max_bytes")
    with path.open("rb") as stream:
        stream.seek(offset)
        data = stream.read(maximum)
    if b"\x00" in data:
        raise ValueError("Binary file: use a container test/probe to inspect its format")
    size = path.stat().st_size
    text = data.decode("utf-8", errors="replace").encode("utf-8")[:65536].decode("utf-8", errors="ignore")
    return {"path": str(path), "size": size, "offset": offset, "bytes_read": len(data), "text": text,
            "next_offset": offset + len(data) if offset + len(data) < size else None,
            "truncated": offset + len(data) < size}


def search_remote_files(payload):
    root = checked_path(payload["path"])
    query = payload["query"]
    if not isinstance(query, str) or not query or len(query.encode()) > 4096:
        raise ValueError("query must be a nonempty literal string up to 4096 bytes")
    limit = integer(payload.get("limit", 50), 1, 200, "limit")
    state = {"scanned": 0, "scan_truncated": False}
    paths = [root] if root.is_file() else walk_files(root, True, payload.get("include_hidden", False), state)
    matches, searched, skipped, scanned_bytes, output_bytes = [], 0, 0, 0, 0
    deadline = time.monotonic() + 8
    truncated = False
    for path in paths:
        pattern = payload.get("pattern", "*")
        relative = path.name if root.is_file() else str(path.relative_to(root))
        if path.is_symlink() or not path.is_file() or not (fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(relative, pattern.removeprefix("**/"))):
            continue
        if scanned_bytes >= 8 * 1024 * 1024 or time.monotonic() > deadline:
            truncated = True
            break
        size = path.stat().st_size
        maximum = min(size, 1024 * 1024, 8 * 1024 * 1024 - scanned_bytes)
        with path.open("rb") as stream:
            data = stream.read(maximum)
        scanned_bytes += len(data)
        if b"\x00" in data:
            skipped += 1
            continue
        searched += 1
        if len(data) < size:
            truncated = True
        for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
            if query in line:
                match = {"path": str(path), "line": number, "text": line[:1000]}
                output_bytes += len(json.dumps(match).encode())
                if output_bytes > 49152:
                    truncated = True
                    break
                matches.append(match)
                if len(matches) >= limit:
                    truncated = True
                    break
        if len(matches) >= limit or output_bytes > 49152:
            break
    return {"path": str(root), "query": query, "matches": matches, "files_searched": searched,
            "binary_files_skipped": skipped, "bytes_scanned": scanned_bytes,
            "truncated": truncated or state["scan_truncated"], **state}


def inspect_image(image):
    if not isinstance(image, str) or image.startswith("-") or not re.fullmatch(r"[A-Za-z0-9_./:@-]+", image):
        raise ValueError("Invalid image reference")
    info = json.loads(core.execute(["docker", "image", "inspect", image]).stdout)[0]
    config = info["Config"]
    return {"image_id": info["Id"], "tags": info.get("RepoTags"), "digests": info.get("RepoDigests"),
            "architecture": info["Architecture"], "os": info["Os"], "created": info["Created"], "size": info["Size"],
            "user": config.get("User"), "working_dir": config.get("WorkingDir"),
            "entrypoint": config.get("Entrypoint"), "command": config.get("Cmd"),
            "environment_names": [x.split("=", 1)[0] for x in config.get("Env", [])],
            "managed_labels": {k: v for k, v in (config.get("Labels") or {}).items() if k.startswith(core.LABEL)}}


def list_images(limit):
    integer(limit, 1, 200, "limit")
    rows = [json.loads(line) for line in core.execute(["docker", "image", "ls", "--no-trunc", "--format", "{{json .}}"]).stdout.splitlines()]
    return {"images": [{k: r.get(k) for k in ["ID", "Repository", "Tag", "Size", "CreatedAt"]} for r in rows[:limit]], "truncated": len(rows) > limit}


def prepare_remote_run(host, payload):
    root = checked_path(payload["remote_source_path"])
    if not root.is_dir():
        raise ValueError("remote_source_path must be a directory")
    return core.prepare_run(host, {"spec": payload["spec"], "source": source_snapshot(str(root)), "runner": payload["runner"]})


def environment_root(environment_id):
    root = core.MANAGED_ROOT / "environments" / core.valid_id(environment_id)
    if root.exists() and root.resolve() != root.absolute():
        raise ValueError("Environment record path must not contain symlinks")
    return root


def environment_plan(host, payload):
    environment_id = core.valid_id(payload["environment_id"])
    instructions = payload["instructions"]
    if not isinstance(instructions, str) or not instructions.strip() or len(instructions.encode()) > 65536 or "\x00" in instructions:
        raise ValueError("instructions must contain a Dockerfile body up to 64 KiB")
    # FROM is generated from the resolved local base, including on continued lines.
    logical_instructions = instructions.replace("\\\r\n", "").replace("\\\n", "")
    if re.search(r"(?im)^\s*(FROM|ONBUILD)\s", logical_instructions):
        raise ValueError("Supply Dockerfile instructions after FROM; multi-stage FROM/ONBUILD is not supported")
    base = inspect_image(payload.get("base_image", "pp_collision:alessio-latest"))
    source = payload.get("source")
    if payload.get("remote_context_path"):
        source = source_snapshot(str(checked_path(payload["remote_context_path"])))
    if source is None or core.digest(source["files"]) != source["sha256"]:
        raise ValueError("Invalid build context manifest")
    # A Dockerfile supplied in the context must not silently replace the generated one.
    if "Dockerfile.edge-mcp" in source["files"]:
        raise ValueError("Dockerfile.edge-mcp is reserved")
    network = payload.get("network", "default")
    if network not in ["default", "none"]:
        raise ValueError("Build network must be default or none")
    base_tag = "clas12-explore-base:" + base["image_id"].split(":", 1)[1]
    dockerfile = "FROM " + base_tag + "\n" + instructions.rstrip() + "\n"
    plan = {"schema_version": 1, "host": host, "environment_id": environment_id,
            "base_image_id": base["image_id"], "base_tag": base_tag,
            "image_tag": "clas12-explore/" + environment_id + ":local", "dockerfile": dockerfile,
            "network": network, "source": {"sha256": source["sha256"], "file_count": len(source["files"]), "git": source.get("git")},
            "worker_sha256": hashlib.sha256(payload["worker"].encode()).hexdigest()}
    plan["fingerprint"] = core.digest(plan)
    plan["root"] = str(environment_root(environment_id))
    context = str(Path(plan["root"]) / "context")
    plan["build_argv"] = ["docker", "build", "--pull=false", "--progress=plain", "--network", network,
         "--iidfile", str(Path(plan["root"]) / "image.id"), "--tag", plan["image_tag"],
         "--label", core.LABEL + ".environment-id=" + environment_id,
         "--label", core.LABEL + ".fingerprint=" + plan["fingerprint"],
         "--file", str(Path(context) / "Dockerfile.edge-mcp"), context]
    return plan, source


def load_environment(host, environment_id):
    root = environment_root(environment_id)
    plan = json.loads((root / "plan.json").read_text())
    if plan["host"] != host or plan["environment_id"] != environment_id or plan["root"] != str(root):
        raise ValueError("Environment identity mismatch")
    core_fields = {k: v for k, v in plan.items() if k not in ["fingerprint", "root", "build_argv", "prepared_at"]}
    if core.digest(core_fields) != plan["fingerprint"]:
        raise ValueError("Environment plan checksum mismatch")
    return plan


def matching_worker(root, plan):
    worker = core.optional_json(root / "worker.json")
    if not worker or worker.get("fingerprint") != plan["fingerprint"]:
        return None
    try:
        pid = worker["pid"]
        process = Path(f"/proc/{pid}")
        token = process.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
        args = process.joinpath("cmdline").read_bytes().split(b"\x00")
        if token == worker["process_token"] and str(root / "worker.py").encode() in args:
            return worker
    except (FileNotFoundError, ProcessLookupError):
        pass
    return None


def get_environment_build(host, payload):
    plan = load_environment(host, payload["environment_id"])
    root = Path(plan["root"])
    result = core.optional_json(root / "result.json")
    worker = matching_worker(root, plan)
    launch = core.optional_json(root / "launch.json")
    if result:
        if result.get("fingerprint") != plan["fingerprint"]:
            raise ValueError("Build result identity mismatch")
        status = result["status"]
    else:
        status = "building" if worker else ("unknown" if launch else "prepared")
    logs = core.bounded_tail(root / "build.log", 100) if (root / "build.log").exists() else {"text": "", "truncated": False}
    return {"host": host, "environment_id": plan["environment_id"], "status": status,
            "image_tag": plan["image_tag"], "image_id": result.get("image_id") if result else None,
            "base_image_id": plan["base_image_id"], "fingerprint": plan["fingerprint"],
            "root": str(root), "worker": worker, "result": result, "log_tail": logs["text"], "log_truncated": logs["truncated"]}


def build_environment(host, payload):
    plan, source = environment_plan(host, payload)
    root = Path(plan["root"])
    with core.locked("environment-" + plan["environment_id"]):
        if root.exists():
            existing = load_environment(host, plan["environment_id"])
            if existing["fingerprint"] != plan["fingerprint"]:
                raise ValueError("Environment ID already belongs to a different immutable build plan")
            if (root / "launch.json").exists():
                return {"already_started": True, **get_environment_build(host, payload)}
            if existing["build_argv"] != plan["build_argv"]:
                raise ValueError("Build command changed after preparation")
        else:
            root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".build-", dir=root.parent))
            try:
                (temporary / "context").mkdir()
                core.unpack_source(source, temporary / "context")
                (temporary / "context/Dockerfile.edge-mcp").write_text(plan["dockerfile"])
                (temporary / "worker.py").write_text(payload["worker"])
                core.atomic_json(temporary / "source_manifest.json", source["files"])
                core.atomic_json(temporary / "plan.json", {**plan, "prepared_at": core.now()})
                temporary.rename(root)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        manifest = json.loads((root / "source_manifest.json").read_text())
        if core.digest(manifest) != plan["source"]["sha256"]:
            raise ValueError("Build context manifest changed after preparation")
        paths = list((root / "context").rglob("*"))
        actual = {str(p.relative_to(root / "context")) for p in paths if p.is_file()}
        if any(p.is_symlink() for p in paths) or actual != set(manifest) | {"Dockerfile.edge-mcp"}:
            raise ValueError("Build context file inventory changed after preparation")
        for name, entry in manifest.items():
            path = root / "context" / name
            if path.is_symlink() or not core.below(path.resolve(), (root / "context").resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("Build context changed after preparation")
        if (root / "context/Dockerfile.edge-mcp").read_text() != plan["dockerfile"]:
            raise ValueError("Generated Dockerfile changed after preparation")
        if hashlib.sha256((root / "worker.py").read_bytes()).hexdigest() != plan["worker_sha256"]:
            raise ValueError("Build worker changed after preparation")
        # Neither an existing user image tag nor another managed image is overwritten.
        target = core.execute(["docker", "image", "inspect", plan["image_tag"]], check=False)
        if target.returncode == 0:
            raise ValueError("Target image tag already exists; refusing to overwrite it")
        alias = core.execute(["docker", "image", "inspect", plan["base_tag"]], check=False)
        if alias.returncode == 0:
            if json.loads(alias.stdout)[0]["Id"] != plan["base_image_id"]:
                raise ValueError("Content-addressed base tag points to a different image")
        else:
            core.execute(["docker", "image", "tag", plan["base_image_id"], plan["base_tag"]])
        core.atomic_json(root / "launch.json", {"launch_requested_at": core.now(), "fingerprint": plan["fingerprint"]})
        with (root / "build.log").open("ab", buffering=0) as log:
            process = subprocess.Popen([sys.executable, str(root / "worker.py"), str(root)],
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True, close_fds=True)
        # Give the worker a brief chance to publish its identity; no build wait.
        for _ in range(20):
            if (root / "worker.json").exists() or process.poll() is not None:
                break
            time.sleep(.025)
        return {"already_started": False, **get_environment_build(host, payload)}


def stop_environment_build(host, payload):
    environment_id = core.valid_id(payload["environment_id"])
    with core.locked("environment-" + environment_id):
        plan = load_environment(host, environment_id)
        root = Path(plan["root"])
        result = core.optional_json(root / "result.json")
        if result:
            return {"already_stopped": True, **get_environment_build(host, payload)}
        worker = matching_worker(root, plan)
        if not worker:
            raise ValueError("No matching live build worker; refusing to signal an unrelated process")
        descriptor = os.pidfd_open(worker["pid"])
        try:
            if matching_worker(root, plan) != worker:
                raise ValueError("Build worker identity changed before cancellation")
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        finally:
            os.close(descriptor)
        return {"stop_requested": True, **get_environment_build(host, payload)}


def dispatch_explore(host, action, payload):
    reads = {"list_remote_files": list_remote_files, "read_remote_file": read_remote_file,
             "search_remote_files": search_remote_files}
    if action in reads:
        return {"host": host, **reads[action](payload)}
    if action == "inspect_image":
        return {"host": host, **inspect_image(payload["image"])}
    if action == "list_images":
        return {"host": host, **list_images(payload.get("limit", 50))}
    if action == "prepare_remote_run":
        return prepare_remote_run(host, payload)
    if action == "build_environment":
        return build_environment(host, payload)
    if action == "get_environment_build":
        return get_environment_build(host, payload)
    if action == "stop_environment_build":
        return stop_environment_build(host, payload)
    raise ValueError("Unknown operation")
