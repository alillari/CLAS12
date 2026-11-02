"""Real MCP -> SSH -> Docker lifecycle checks on explicitly selected hosts.

Run with the integration venv. Starts small, bounded containers and preserves
their records/logs. Never launches a scientific campaign or touches other jobs.
"""

import argparse
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time
import traceback

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from edge_mcp.client import EdgeClient

ROOT = Path(__file__).resolve().parents[1]
TOOLS = {"list_hosts", "inspect_host", "preview_run", "prepare_run", "launch_run", "get_run_status", "list_runs", "read_run_logs", "read_run_artifact", "stop_run"}


@asynccontextmanager
async def connection(log_path):
    with log_path.open("a") as errors:
        parameters = StdioServerParameters(command=sys.executable, args=["-m", "edge_mcp.server"])
        async with stdio_client(parameters, errlog=errors) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=100)) as session:
                await session.initialize()
                listing = await session.list_tools()
                assert TOOLS <= {tool.name for tool in listing.tools}
                assert all(tool.outputSchema is not None for tool in listing.tools)
                assert next(t for t in listing.tools if t.name == "inspect_host").annotations.readOnlyHint
                assert not next(t for t in listing.tools if t.name == "launch_run").annotations.readOnlyHint
                yield session


async def call(session, tool, arguments, expect_error=False):
    response = await session.call_tool(tool, arguments)
    if expect_error:
        assert response.isError, f"{tool} unexpectedly accepted invalid input"
        return {"rejected": True, "message": " ".join(c.text for c in response.content if c.type == "text")[:500]}
    if response.isError:
        raise RuntimeError(f"{tool}: " + " ".join(c.text for c in response.content if c.type == "text"))
    assert response.structuredContent is not None, f"{tool} lacks structured output"
    return response.structuredContent


async def wait_status(session, host, job_id, terminal=True, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = await call(session, "get_run_status", {"host": host, "job_id": job_id})
        if not terminal or status["status"] not in ["created", "running", "restarting", "paused"]:
            return status
        await asyncio.sleep(.5)
    raise AssertionError(f"{host}/{job_id} did not finish within {timeout}s")


def make_spec(job_id, mode, **updates):
    spec = {"job_id": job_id, "command": ["python", "smoke_job.py", "--mode", mode],
            "gpu": mode == "gpu", "cpus": 1, "memory_gib": 8 if mode == "gpu" else 1,
            "timeout_seconds": 120 if mode in ["gpu", "wait"] else 30, "ipc_mode": "private",
            "pids_limit": 512, "network": "none"}
    if mode == "gpu":
        spec.update(expected_artifacts=["gpu.json"], environment={"EDGE_LITERAL": "$(literal); `literal` $HOME"},
                    mounts=[{"source": "/home/alessio/ML-work/pretrained-FMs", "target": "/checkpoints", "read_only": True}])
    elif mode == "missing":
        spec["expected_artifacts"] = ["missing.json"]
    elif mode == "wait":
        spec["expected_artifacts"] = ["ready.json"]
    spec.update(updates)
    return spec


async def host_checks(host, run_id, report_dir):
    data = {"host": host, "jobs": {}, "checks": []}
    known_ids = []
    log_path = report_dir / (host + "-mcp.stderr.log")
    source_path = str(ROOT / "examples")
    try:
        async with connection(log_path) as session:
            hosts = await call(session, "list_hosts", {})
            assert {h["host"] for h in hosts["hosts"]} == {"a02c", "a518"}
            telemetry = await call(session, "inspect_host", {"host": host})
            assert not telemetry["errors"], telemetry["errors"]
            data["telemetry"] = {k: telemetry[k] for k in ["hostname", "cores", "gpus", "observed_at"]}
            data["checks"].append("MCP discovery, structured outputs, monitoring")
            gpu_id = f"{run_id}-{host}-gpu"
            spec = make_spec(gpu_id, "gpu")
            args = {"host": host, "spec": spec, "source_path": source_path}
            preview = await call(session, "preview_run", args)
            assert not preview["prepared"] and preview["preview"]
            await call(session, "get_run_status", {"host": host, "job_id": gpu_id}, expect_error=True)
            plan = await call(session, "prepare_run", args)
            known_ids.append(gpu_id)
            again = await call(session, "prepare_run", args)
            assert again["already_prepared"] and again["fingerprint"] == plan["fingerprint"]
            data["image_id"] = plan["image_id"]
            data["source_sha256"] = plan["source"]["sha256"]
            data["checks"].append("preview writes nothing; preparation is idempotent")
            changed = {**args, "spec": {**spec, "cpus": 2}}
            await call(session, "prepare_run", changed, expect_error=True)
            data["checks"].append("job ID cannot be reused for a changed plan")
            first = await call(session, "launch_run", {"host": host, "job_id": gpu_id})
            duplicate = await call(session, "launch_run", {"host": host, "job_id": gpu_id})
            assert duplicate["already_launched"] and duplicate["container_id"] == first["container_id"]
            status = await wait_status(session, host, gpu_id)
            assert status["status"] == "succeeded", status
            gpu = json.loads((await call(session, "read_run_artifact", {"host": host, "job_id": gpu_id, "path": "gpu.json"}))["text"])
            assert gpu["cuda"] and gpu["finite_gradients"] and gpu["checkpoint_mount_readable"]
            assert gpu["literal_environment"] == spec["environment"]["EDGE_LITERAL"]
            assert (await call(session, "read_run_logs", {"host": host, "job_id": gpu_id}))["text"]
            assert (await call(session, "read_run_logs", {"host": host, "job_id": gpu_id, "source": "docker"}))["text"]
            data["jobs"]["gpu"] = {"job_id": gpu_id, "status": status, "gpu_result": gpu}
            data["checks"].append("real CUDA forward/backward, read-only input mount, literal environment, persistent artifact/stdout/Docker logs")
            data["checks"].append("duplicate launch returns same container")

            wait_id = f"{run_id}-{host}-stop"
            await call(session, "prepare_run", {"host": host, "spec": make_spec(wait_id, "wait"), "source_path": source_path})
            known_ids.append(wait_id)
            waiting = await call(session, "launch_run", {"host": host, "job_id": wait_id})
            for _ in range(30):
                logs = await call(session, "read_run_logs", {"host": host, "job_id": wait_id})
                if "DESCENDANT-READY" in logs["text"]:
                    break
                await asyncio.sleep(.2)
            else:
                raise AssertionError("Waiting job/descendant did not start")
        # Exiting the context terminates the MCP subprocess and its SSH sockets.
        async with connection(log_path) as session:
            after = await call(session, "get_run_status", {"host": host, "job_id": wait_id})
            assert after["status"] == "running" and after["container_id"] == waiting["container_id"]
            duplicate = await call(session, "launch_run", {"host": host, "job_id": wait_id})
            assert duplicate["already_launched"] and duplicate["container_id"] == waiting["container_id"]
            data["checks"].append("job survives MCP/SSH disconnect and reconnect without duplication")
            stopped = await call(session, "stop_run", {"host": host, "job_id": wait_id})
            assert stopped["status"] == "stopped", stopped
            assert stopped["runner_result"]["stop_requested"]
            assert (await call(session, "stop_run", {"host": host, "job_id": wait_id}))["already_stopped"]
            assert (await call(session, "launch_run", {"host": host, "job_id": wait_id}))["already_launched"]
            data["jobs"]["stop"] = {"job_id": wait_id, "status": stopped}
            data["checks"].append("graceful stop with TERM-ignoring descendant; repeated stop and launch do not restart")
            data["stop_unknown"] = await call(session, "stop_run", {"host": host, "job_id": "not-a-managed-job"}, expect_error=True)

            for mode, expected in [("fail", "failed"), ("missing", "incomplete"), ("timeout", "timed_out")]:
                job_id = f"{run_id}-{host}-{mode}"
                spec = make_spec(job_id, mode, **({"timeout_seconds": 2} if mode == "timeout" else {}))
                await call(session, "prepare_run", {"host": host, "spec": spec, "source_path": source_path})
                known_ids.append(job_id)
                await call(session, "launch_run", {"host": host, "job_id": job_id})
                status = await wait_status(session, host, job_id, timeout=60)
                assert status["status"] == expected, status
                data["jobs"][mode] = {"job_id": job_id, "status": status}
                if mode == "fail":
                    logs = await call(session, "read_run_logs", {"host": host, "job_id": job_id})
                    assert logs["truncated"] and logs["text"].endswith("FAILURE-END")
                    assert len(logs["text"].encode()) <= 65536
                    assert status["exit_code"] == 7
            data["checks"].append("nonzero exit, missing artifact, runtime deadline and bounded oversized logs")
            # This command exits immediately. The harness still has an external
            # watchdog/cleanup, while the job itself uses the real uncapped defaults.
            job_id = f"{run_id}-{host}-defaults"
            spec = make_spec(job_id, "missing", expected_artifacts=[], cpus=None,
                             memory_gib=None, pids_limit=None, timeout_seconds=None, network="bridge")
            await call(session, "prepare_run", {"host": host, "spec": spec, "source_path": source_path})
            known_ids.append(job_id)
            await call(session, "launch_run", {"host": host, "job_id": job_id})
            status = await wait_status(session, host, job_id, timeout=60)
            assert status["status"] == "succeeded", status
            limits = status["effective_limits"]
            assert all(limits[k] is None for k in ["cpus", "memory_bytes", "pids_limit", "runtime_deadline_seconds"]), limits
            assert limits["network_mode"] == "bridge", limits
            data["jobs"]["defaults"] = {"job_id": job_id, "status": status}
            data["checks"].append("Docker inspection confirms no default CPU/RAM/PID cap or runtime deadline")
            listing = await call(session, "list_runs", {"host": host, "limit": 100})
            selected = [r for r in listing["runs"] if r["job_id"] in known_ids]
            assert len(selected) == len(known_ids)
            assert not any(r["status"] in ["running", "created", "restarting", "paused"] for r in selected)
            data["checks"].append("all test containers are terminal; records survive client restarts")
        data["passed"] = True
        print(f"{host}: passed {len(data['checks'])} lifecycle checks", flush=True)
    except BaseException as exc:
        data["passed"] = False
        data["error"] = "".join(traceback.format_exception(exc))
        print(f"{host}: FAILED: {exc}", flush=True)
    finally:
        # Cleanup only IDs created by this invocation, including on assertion errors.
        def cleanup():
            with EdgeClient() as client:
                errors = []
                for job_id in known_ids:
                    try:
                        status = client.get_run_status(host, job_id)
                        if status["status"] in ["running", "created", "paused", "restarting"]:
                            client.stop_run(host, job_id)
                    except Exception as exc:
                        errors.append({"job_id": job_id, "error": str(exc)})
                return errors
        data["cleanup_errors"] = await asyncio.to_thread(cleanup)
        if data["cleanup_errors"]:
            data["passed"] = False
    return data


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", nargs="+", choices=["a02c", "a518"], required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-id", default="mcp-e2e-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    results = await asyncio.gather(*(host_checks(h, args.run_id, args.report.parent) for h in args.hosts))
    report = {"run_id": args.run_id, "finished_utc": datetime.now(timezone.utc).isoformat(),
              "passed": all(r["passed"] for r in results), "hosts": results}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
