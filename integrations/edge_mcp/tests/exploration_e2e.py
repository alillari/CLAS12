"""Real exploratory MCP workflow; builds only separately tagged test images."""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback

from e2e import call, connection
from edge_mcp.client import EdgeClient

TRAINING_JOB = "mamba3-clusters-w512-d12-k5-causal-20261027"
NEW_TOOLS = {"list_remote_files", "read_remote_file", "search_remote_files", "list_images", "inspect_image",
             "run_tests", "build_environment", "get_environment_build", "stop_environment_build"}


async def wait_build(session, host, environment_id, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = await call(session, "get_environment_build", {"host": host, "environment_id": environment_id})
        if result["status"] in ["succeeded", "failed", "stopped", "unknown"]:
            return result
        await asyncio.sleep(.5)
    raise AssertionError(f"Build did not finish: {environment_id}")


async def run_host(host, run_id, report_dir):
    report = {"host": host, "checks": [], "builds": {}, "tests": {}}
    job_ids, environment_ids = [], []
    log = report_dir / (host + "-exploration.stderr.log")
    try:
        with tempfile.TemporaryDirectory(prefix="edge-explore-context-") as temporary:
            context = Path(temporary)
            package = context / "probe_package"
            package.mkdir()
            (package / "setup.py").write_text("from setuptools import setup\nsetup(name='edge-explore-probe',version='0.0.1',py_modules=['edge_explore_probe'])\n")
            (package / "edge_explore_probe.py").write_text("VALUE = 17\n")
            (context / "binary.dat").write_bytes(b"a\x00b")
            tests = context / "local_tests"
            tests.mkdir()
            test_file = tests / "check.py"
            test_file.write_text("import edge_explore_probe\nassert edge_explore_probe.VALUE == 18, 'intentional failure'\n")
            instructions = "COPY probe_package /tmp/edge-probe\nRUN python -m pip install --no-deps --no-build-isolation /tmp/edge-probe\nENV EDGE_EXPLORATION_PROBE=ready"
            environment_id = f"{run_id}-{host}-env"
            async with connection(log) as session:
                tools = {t.name for t in (await session.list_tools()).tools}
                assert NEW_TOOLS <= tools
                base = await call(session, "inspect_image", {"host": host})
                report["base_image_before"] = base["image_id"]
                images = await call(session, "list_images", {"host": host})
                assert any(i["Repository"] == "pp_collision" for i in images["images"])
                primary = "/home/alessio/ML-work/CLAS12/requirements.txt"
                original = await call(session, "read_remote_file", {"host": host, "path": primary})
                report["primary_file_sha256_before"] = hashlib.sha256(original["text"].encode()).hexdigest()
                if host == "a02c":
                    report["training_before"] = await call(session, "get_run_status", {"host": host, "job_id": TRAINING_JOB})
                environment_ids.append(environment_id)
                built = await call(session, "build_environment", {"host": host, "environment_id": environment_id,
                                      "instructions": instructions, "source_path": str(context), "network": "none"})
                built = await wait_build(session, host, environment_id)
                assert built["status"] == "succeeded", built
                report["builds"]["environment"] = built
                derived = await call(session, "inspect_image", {"host": host, "image": built["image_id"]})
                assert derived["managed_labels"]["org.clas12.edge-mcp.environment-id"] == environment_id
                repeated = await call(session, "build_environment", {"host": host, "environment_id": environment_id,
                                      "instructions": instructions, "source_path": str(context), "network": "none"})
                assert repeated["already_started"] and repeated["image_id"] == built["image_id"]
                await call(session, "build_environment", {"host": host, "environment_id": environment_id,
                              "instructions": instructions + "\nRUN true", "source_path": str(context), "network": "none"}, expect_error=True)
                report["checks"].append("build/install local package into a derived image; reuse ID; reject changed plan")

                remote_context = built["root"] + "/context"
                listed = await call(session, "list_remote_files", {"host": host, "path": remote_context, "recursive": True, "pattern": "*.py", "limit": 1})
                assert listed["truncated"] and listed["next_offset"] is not None
                again = await call(session, "list_remote_files", {"host": host, "path": remote_context, "recursive": True, "pattern": "*.py", "offset": listed["next_offset"]})
                assert again["entries"]
                module = remote_context + "/probe_package/edge_explore_probe.py"
                part = await call(session, "read_remote_file", {"host": host, "path": module, "max_bytes": 5})
                rest = await call(session, "read_remote_file", {"host": host, "path": module, "offset": part["next_offset"]})
                assert part["text"] + rest["text"] == "VALUE = 17\n"
                found = await call(session, "search_remote_files", {"host": host, "path": remote_context, "query": "VALUE = 17", "pattern": "**/*.py"})
                assert any(m["path"] == module and m["line"] == 1 for m in found["matches"])
                await call(session, "read_remote_file", {"host": host, "path": "/etc/passwd"}, expect_error=True)
                await call(session, "read_remote_file", {"host": host, "path": remote_context + "/binary.dat"}, expect_error=True)
                report["checks"].append("direct listing/search/byte pagination; outside-root and binary rejection")

                common = {"image": built["image_id"], "command": ["python", "check.py"], "gpu": False,
                          "cpus": 1, "memory_gib": 2, "timeout_seconds": 30, "network": "none"}
                failed_id = f"{run_id}-{host}-test-fail"
                job_ids.append(failed_id)
                failed = await call(session, "run_tests", {"host": host, "spec": {**common, "job_id": failed_id}, "source_path": str(tests), "wait_seconds": 10})
                assert failed["status"] == "failed" and "intentional failure" in failed["stdout_tail"], failed
                report["tests"]["failed"] = failed
                test_file.write_text("import edge_explore_probe,os\nfrom pathlib import Path\nassert edge_explore_probe.VALUE == 17\nassert os.environ['EDGE_EXPLORATION_PROBE'] == 'ready'\nPath('/results/escape').symlink_to('/etc/passwd')\nprint('EDIT-TEST-PASS',flush=True)\n")
                passed_id = f"{run_id}-{host}-test-pass"
                job_ids.append(passed_id)
                passed = await call(session, "run_tests", {"host": host, "spec": {**common, "job_id": passed_id}, "source_path": str(tests), "wait_seconds": 10})
                assert passed["status"] == "succeeded" and "EDIT-TEST-PASS" in passed["stdout_tail"], passed
                duplicate = await call(session, "run_tests", {"host": host, "spec": {**common, "job_id": passed_id}, "source_path": str(tests), "wait_seconds": 0})
                assert duplicate["container_id"] == passed["container_id"]
                await call(session, "read_remote_file", {"host": host, "path": passed["output_dir"] + "/escape"}, expect_error=True)
                report["tests"]["passed"] = passed
                report["checks"].append("test fails, local edit fixes it in built image, repeat returns same container")

                remote_id = f"{run_id}-{host}-remote-test"
                job_ids.append(remote_id)
                remote = await call(session, "run_tests", {"host": host, "spec": {**common, "job_id": remote_id,
                       "command": ["python", "-c", "from pathlib import Path; assert Path('probe_package/edge_explore_probe.py').read_text() == 'VALUE = 17\\n'; print('REMOTE-SNAPSHOT-PASS')"]},
                       "remote_source_path": remote_context, "wait_seconds": 10})
                assert remote["status"] == "succeeded", remote
                report["tests"]["remote_source"] = remote
                report["checks"].append("test a remote source snapshot without editing its checkout/context")

                concurrent_ids = [f"{run_id}-{host}-parallel-{i}" for i in range(2)]
                job_ids.extend(concurrent_ids)
                concurrent = await asyncio.gather(*(call(session, "run_tests", {"host": host,
                     "spec": {**common, "job_id": job_id, "command": ["python", "-c", "print('PARALLEL-PASS')"]},
                     "wait_seconds": 10}) for job_id in concurrent_ids))
                assert all(r["status"] == "succeeded" for r in concurrent), concurrent
                assert concurrent[0]["container_id"] != concurrent[1]["container_id"]
                report["checks"].append("concurrent probes handle host launch-lock contention without duplicate jobs")

                detached_id = f"{run_id}-{host}-detached-test"
                job_ids.append(detached_id)
                detached = await call(session, "run_tests", {"host": host, "spec": {**common, "job_id": detached_id,
                      "command": ["python", "-u", "-c", "import time; print('DETACHED-READY'); time.sleep(2)"]}, "wait_seconds": 0})
                cancel_id = f"{run_id}-{host}-cancel-build"
                environment_ids.append(cancel_id)
                started_build = await call(session, "build_environment", {"host": host, "environment_id": cancel_id,
                    "instructions": "RUN python -u -c \"import time; print('BUILD-READY'); time.sleep(90)\"", "network": "none"})
            # Terminate MCP/SSH connections while a test and build are detached.
            async with connection(log) as session:
                for _ in range(20):
                    done = await call(session, "get_run_status", {"host": host, "job_id": detached_id})
                    if done["status"] == "succeeded":
                        break
                    await asyncio.sleep(.3)
                assert done["status"] == "succeeded" and done["container_id"] == detached["container_id"], done
                build = await call(session, "get_environment_build", {"host": host, "environment_id": cancel_id})
                assert build["status"] == "building", build
                await call(session, "stop_environment_build", {"host": host, "environment_id": cancel_id})
                cancelled = await wait_build(session, host, cancel_id)
                assert cancelled["status"] == "stopped" and cancelled["image_id"] is None, cancelled
                report["builds"]["cancelled"] = cancelled
                report["checks"].append("test/build survive MCP disconnect; only owned build is cancelled")

                bad_id = f"{run_id}-{host}-failed-build"
                environment_ids.append(bad_id)
                bad_args = {"host": host, "environment_id": bad_id, "instructions": "RUN python -c \"raise SystemExit(7)\"", "network": "none"}
                await call(session, "build_environment", bad_args)
                bad = await wait_build(session, host, bad_id)
                assert bad["status"] == "failed" and bad["result"]["exit_code"] != 0, bad
                assert (await call(session, "build_environment", bad_args))["already_started"]
                report["builds"]["failed"] = bad
                report["checks"].append("failed builds retain logs and are not silently restarted")
                final_base = await call(session, "inspect_image", {"host": host})
                assert final_base["image_id"] == report["base_image_before"]
                final_file = await call(session, "read_remote_file", {"host": host, "path": primary})
                assert hashlib.sha256(final_file["text"].encode()).hexdigest() == report["primary_file_sha256_before"]
                if host == "a02c":
                    report["training_after"] = await call(session, "get_run_status", {"host": host, "job_id": TRAINING_JOB})
                    assert report["training_after"]["container_id"] == report["training_before"]["container_id"]
                    assert report["training_after"]["status"] in ["running", "succeeded"]
                report["checks"].append("original image tag, primary source and existing training are preserved")
            report["passed"] = True
    except BaseException as exc:
        report["passed"] = False
        report["error"] = "".join(traceback.format_exception(exc))
    finally:
        def cleanup():
            errors = []
            with EdgeClient() as client:
                for job_id in job_ids:
                    try:
                        status = client.get_run_status(host, job_id)
                        if status["status"] in ["created", "running", "paused", "restarting"]:
                            client.stop_run(host, job_id)
                    except Exception as exc:
                        errors.append(str(exc))
                for environment_id in environment_ids:
                    try:
                        status = client.get_environment_build(host, environment_id)
                        if status["status"] == "building":
                            client.stop_environment_build(host, environment_id)
                    except Exception as exc:
                        errors.append(str(exc))
            return errors
        report["cleanup_errors"] = await asyncio.to_thread(cleanup)
        report["passed"] = report.get("passed", False) and not report["cleanup_errors"]
        print(f"{host}: {'passed' if report['passed'] else 'FAILED'} {len(report['checks'])} exploration checks", flush=True)
    return report


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", nargs="+", required=True, choices=["a02c", "a518"])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-id", default="explore-e2e-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    results = await asyncio.gather(*(run_host(h, args.run_id, args.report.parent) for h in args.hosts))
    report = {"run_id": args.run_id, "passed": all(r["passed"] for r in results), "hosts": results}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
