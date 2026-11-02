"""Reusable Python API; also used by the CLI and MCP tools."""

from pathlib import Path
import time

from .models import HOSTS, RunSpec, TestSpec
from .snapshot import source_snapshot
from .transport import EdgeError, SSHTransport


class EdgeClient:
    def __init__(self, transport=None):
        self.transport = transport or SSHTransport()

    def close(self):
        self.transport.close()

    @staticmethod
    def _retry_lock(operation):
        """Retry only explicit busy-lock responses, with the same immutable ID."""
        deadline = time.monotonic() + 5
        while True:
            try:
                return operation()
            except EdgeError as exc:
                if "Another operation holds this job/host lock" not in str(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(.1)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def list_hosts(self):
        return {"hosts": [{"host": host, "ssh_alias": alias} for host, alias in HOSTS.items()]}

    def inspect_host(self, host, details=True):
        return self.transport.call(host, "inspect_host", {"details": details})

    def prepare_run(self, host, spec: RunSpec | dict, source_path=None, preview=False):
        spec = spec if isinstance(spec, RunSpec) else RunSpec.model_validate(spec)
        source = source_snapshot(source_path)
        if preview:
            source.pop("archive")
        payload = {"spec": spec.model_dump(), "source": source,
                   "runner": Path(__file__).with_name("supervisor.py").read_text()}
        return self.transport.call(host, "preview_run" if preview else "prepare_run", payload, timeout=90)

    def launch_run(self, host, job_id):
        return self.transport.call(host, "launch_run", {"job_id": job_id})

    def get_run_status(self, host, job_id):
        return self.transport.call(host, "get_run_status", {"job_id": job_id})

    def read_run_logs(self, host, job_id, tail=100, source="stdout"):
        return self.transport.call(host, "read_run_logs", {"job_id": job_id, "tail": tail, "source": source})

    def read_run_artifact(self, host, job_id, path):
        return self.transport.call(host, "read_run_artifact", {"job_id": job_id, "path": path})

    def list_runs(self, host, limit=20):
        return self.transport.call(host, "list_runs", {"limit": limit})

    def stop_run(self, host, job_id):
        return self.transport.call(host, "stop_run", {"job_id": job_id})

    def list_remote_files(self, host, path, pattern="*", recursive=False, include_hidden=False, limit=100, offset=0):
        return self.transport.call(host, "list_remote_files", dict(path=path, pattern=pattern, recursive=recursive,
                                    include_hidden=include_hidden, limit=limit, offset=offset))

    def read_remote_file(self, host, path, offset=0, max_bytes=65536):
        return self.transport.call(host, "read_remote_file", dict(path=path, offset=offset, max_bytes=max_bytes))

    def search_remote_files(self, host, path, query, pattern="*", limit=50, include_hidden=False):
        return self.transport.call(host, "search_remote_files", dict(path=path, query=query, pattern=pattern,
                                    limit=limit, include_hidden=include_hidden))

    def list_images(self, host, limit=50):
        return self.transport.call(host, "list_images", {"limit": limit})

    def inspect_image(self, host, image="pp_collision:alessio-latest"):
        return self.transport.call(host, "inspect_image", {"image": image})

    def run_tests(self, host, spec, source_path=None, remote_source_path=None, wait_seconds=10):
        if source_path is not None and remote_source_path is not None:
            raise ValueError("Choose local source_path or remote_source_path, not both")
        if type(wait_seconds) is not int or not 0 <= wait_seconds <= 30:
            raise ValueError("wait_seconds must be between 0 and 30; longer tests continue detached")
        spec = spec if isinstance(spec, TestSpec) else TestSpec.model_validate(spec)
        if remote_source_path:
            self._retry_lock(lambda: self.transport.call(host, "prepare_remote_run", {"spec": spec.model_dump(),
                 "remote_source_path": remote_source_path,
                 "runner": Path(__file__).with_name("supervisor.py").read_text()}, timeout=90))
        else:
            self._retry_lock(lambda: self.prepare_run(host, spec, source_path))
        status = self._retry_lock(lambda: self.launch_run(host, spec.job_id))
        deadline = time.monotonic() + wait_seconds
        while status["status"] in ["created", "running", "restarting", "paused"] and time.monotonic() < deadline:
            time.sleep(.25)
            status = self.get_run_status(host, spec.job_id)
        logs = self.read_run_logs(host, spec.job_id)
        return {**status, "stdout_tail": logs["text"], "log_truncated": logs["truncated"]}

    def build_environment(self, host, environment_id, instructions, base_image="pp_collision:alessio-latest",
                          source_path=None, remote_context_path=None, network="default"):
        if source_path is not None and remote_context_path is not None:
            raise ValueError("Choose local source_path or remote_context_path, not both")
        payload = {"environment_id": environment_id, "instructions": instructions, "base_image": base_image,
                   "network": network, "remote_context_path": remote_context_path,
                   "worker": Path(__file__).with_name("build_worker.py").read_text()}
        if remote_context_path is None:
            payload["source"] = source_snapshot(source_path)
        return self.transport.call(host, "build_environment", payload, timeout=90)

    def get_environment_build(self, host, environment_id):
        return self.transport.call(host, "get_environment_build", {"environment_id": environment_id})

    def stop_environment_build(self, host, environment_id):
        return self.transport.call(host, "stop_environment_build", {"environment_id": environment_id})
