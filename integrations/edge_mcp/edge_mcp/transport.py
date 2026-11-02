"""Bounded SSH transport. Requests travel as JSON, never shell fragments."""

import json
from pathlib import Path
import shlex
import subprocess
import tempfile

from .models import HOSTS


class EdgeError(RuntimeError):
    pass


class SSHTransport:
    def __init__(self):
        self._sockets = tempfile.TemporaryDirectory(prefix="edge-mcp-ssh-")
        self._used = set()
        self._options = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ClearAllForwardings=yes",
                         "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=8",
                         "-o", "ConnectionAttempts=1", "-o", "ServerAliveInterval=5",
                         "-o", "ServerAliveCountMax=2", "-o", "ControlMaster=auto",
                         "-o", "ControlPersist=60", "-o", f"ControlPath={self._sockets.name}/%C"]

    def call(self, host: str, action: str, payload: dict | None = None, timeout: int = 55) -> dict:
        if host not in HOSTS:
            raise ValueError(f"Unknown host {host!r}; choose a02c or a518")
        root = Path(__file__).parent
        # Ship modules and request over stdin. Only this small loader appears in
        # process listings, rather than a base64 copy of the entire agent.
        bootstrap = (
            "import json,sys,types; b=json.load(sys.stdin); "
            "p=types.ModuleType('edge_mcp_wire'); p.__path__=[]; sys.modules[p.__name__]=p; "
            "exec(\"for name, source in b['sources'].items():\\n"
            " m=types.ModuleType('edge_mcp_wire.'+name); m.__package__='edge_mcp_wire'; "
            "sys.modules[m.__name__]=m; exec(compile(source,'<'+m.__name__+'>','exec'),m.__dict__)\"); "
            "print(json.dumps(sys.modules['edge_mcp_wire.remote'].serve_request(b['request'])))"
        )
        command = "python3 -c " + shlex.quote(bootstrap)
        request = {"host": host, "action": action, "payload": payload or {}}
        bundle = {"sources": {name: (root / (name + ".py")).read_text()
                              for name in ["snapshot", "telemetry", "remote", "remote_explore"]},
                  "request": request}
        self._used.add(host)
        try:
            result = subprocess.run(self._options + [HOSTS[host], command],
                                    input=json.dumps(bundle), capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise EdgeError(f"SSH request timed out for {host}/{action}. A mutation may have completed; inspect the same job ID before retrying.") from exc
        if result.returncode:
            raise EdgeError(f"SSH failed for {host}/{action}: {result.stderr.strip()[-2000:]}")
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise EdgeError(f"Invalid remote response from {host}; inspect the job before retrying a mutation") from exc
        if not response.get("ok"):
            raise EdgeError(response.get("error", "Remote operation failed"))
        return response["result"]

    def close(self):
        for host in self._used:
            try:
                subprocess.run(self._options + ["-O", "exit", HOSTS[host]], capture_output=True, timeout=3)
            except subprocess.TimeoutExpired:
                pass
        self._used.clear()
        self._sockets.cleanup()
