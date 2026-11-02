"""Failure-oriented tests; no network access or Docker daemon is required."""

import base64
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from edge_mcp import remote
from edge_mcp.models import RunSpec
from edge_mcp.snapshot import source_snapshot


class FakeDocker:
    def __init__(self):
        self.containers = {}
        self.starts = self.stops = 0

    def __call__(self, argv, timeout=15, check=True):
        out, err, rc = "", "", 0
        if argv[:3] == ["docker", "image", "inspect"]:
            out = json.dumps([{"Id": "sha256:" + "b" * 64, "Architecture": "arm64"}])
        elif argv[:3] == ["docker", "container", "inspect"]:
            if argv[3] in self.containers:
                out = json.dumps([self.containers[argv[3]]])
            else:
                rc, err = 1, "No such container"
        elif argv[:2] == ["docker", "create"]:
            name = argv[argv.index("--name") + 1]
            labels = dict(argv[i + 1].split("=", 1) for i, value in enumerate(argv) if value == "--label")
            self.containers[name] = {"Id": "c" * 64, "Name": "/" + name,
                                     "Image": argv[-3], "Config": {"Labels": labels},
                                     "State": {"Status": "created", "ExitCode": 0}, "Mounts": []}
            out = "c" * 64
        elif argv[:2] == ["docker", "start"]:
            self.starts += 1
            self.containers[argv[2]]["State"]["Status"] = "running"
        elif argv[:2] == ["docker", "stop"]:
            self.stops += 1
            self.containers[argv[-1]]["State"] = {"Status": "exited", "ExitCode": 143}
        elif argv[:3] == ["docker", "ps", "-aq"]:
            out = "\n".join(self.containers)
        elif argv[:2] == ["docker", "inspect"]:
            out = json.dumps([self.containers[argv[2]]])
        else:
            raise AssertionError(f"Unexpected Docker operation: {argv}")
        return subprocess.CompletedProcess(argv, rc, out, err)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name) / "work"
        self.artifacts = self.work / "result_deep_storage"
        self.managed = self.artifacts / "edge_mcp"
        self.work.mkdir()
        self.artifacts.mkdir()
        for name, value in [("WORK_ROOT", self.work), ("ARTIFACT_ROOT", self.artifacts), ("MANAGED_ROOT", self.managed)]:
            p = patch.object(remote, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.docker = FakeDocker()
        p = patch.object(remote, "execute", self.docker)
        p.start()
        self.addCleanup(p.stop)
        self.source = self.work / "local"
        self.source.mkdir()
        (self.source / "main.py").write_text("print('hello')\n")
        self.spec = RunSpec(job_id="test-one", command=["python", "main.py"], expected_artifacts=["summary.json"])

    def payload(self, spec=None):
        return {"spec": (spec or self.spec).model_dump(), "source": source_snapshot(str(self.source)), "runner": "# supervisor\n"}

    def prepared(self):
        return remote.prepare_run("a02c", self.payload())

    def result(self, exit_code=0, **extra):
        path = self.managed / "jobs/test-one/output/runner_result.json"
        path.write_text(json.dumps({"exit_code": exit_code, "timed_out": False, **extra}))

    def test_preview_does_not_create_job_directories(self):
        plan = remote.prepare_run("a02c", self.payload(), preview=True)
        self.assertFalse(self.managed.exists())
        self.assertIn("--pull=never", plan["docker_create_argv"])
        self.assertIn("--user", plan["docker_create_argv"])
        self.assertEqual(plan["docker_create_argv"][-3], "sha256:" + "b" * 64)

    def test_default_run_has_no_integration_resource_caps_or_deadline(self):
        plan = remote.prepare_run("a02c", self.payload(), preview=True)
        argv = plan["docker_create_argv"]
        for field in ["cpus", "memory_gib", "pids_limit", "timeout_seconds"]:
            self.assertIsNone(plan["spec"][field])
        for flag in ["--cpus", "--memory", "--pids-limit"]:
            self.assertNotIn(flag, argv)
        self.assertEqual(plan["spec"]["network"], "bridge")

    def test_optional_resource_limits_are_applied_and_multweek_deadlines_are_accepted(self):
        spec = RunSpec(job_id="long-pretraining", command=["python", "train.py"],
                       timeout_seconds=60 * 86400, cpus=2, memory_gib=8, pids_limit=512)
        plan = remote.prepare_run("a02c", self.payload(spec), preview=True)
        self.assertEqual(plan["spec"]["timeout_seconds"], 60 * 86400)
        argv = plan["docker_create_argv"]
        self.assertEqual(argv[argv.index("--cpus") + 1], "2.0")
        self.assertEqual(argv[argv.index("--memory") + 1], "8g")
        self.assertEqual(argv[argv.index("--pids-limit") + 1], "512")

    def test_resource_validation_has_no_fixed_host_ceiling(self):
        spec = RunSpec(job_id="explicit-large-limits", command=["true"], cpus=64,
                       memory_gib=256, pids_limit=8192, timeout_seconds=90 * 86400)
        self.assertEqual(remote.validate_spec(spec.model_dump()), spec.model_dump())

    def test_same_prepare_is_idempotent_but_changed_source_is_rejected(self):
        self.prepared()
        self.assertTrue(self.prepared()["already_prepared"])
        (self.source / "main.py").write_text("print('changed')")
        with self.assertRaisesRegex(ValueError, "different immutable plan"):
            self.prepared()

    def test_launch_retry_and_completed_job_never_start_twice(self):
        self.prepared()
        first = remote.launch_run("a02c", "test-one")
        self.assertEqual(first["status"], "running")
        self.assertTrue(remote.launch_run("a02c", "test-one")["already_launched"])
        self.docker.containers["edge-mcp-test-one"]["State"] = {"Status": "exited", "ExitCode": 0}
        self.result()
        (self.managed / "jobs/test-one/output/summary.json").write_text("{}")
        self.assertEqual(remote.launch_run("a02c", "test-one")["status"], "succeeded")
        self.assertEqual(self.docker.starts, 1)

    def test_missing_result_or_declared_artifact_is_incomplete(self):
        self.prepared()
        remote.launch_run("a02c", "test-one")
        self.docker.containers["edge-mcp-test-one"]["State"] = {"Status": "exited", "ExitCode": 0}
        self.assertEqual(remote.run_status("a02c", "test-one")["status"], "incomplete")
        self.result()
        self.assertEqual(remote.run_status("a02c", "test-one")["status"], "incomplete")

    def test_failed_and_timed_out_jobs_cannot_be_reported_as_success(self):
        self.prepared()
        remote.launch_run("a02c", "test-one")
        self.docker.containers["edge-mcp-test-one"]["State"] = {"Status": "exited", "ExitCode": 7}
        self.result(exit_code=7)
        self.assertEqual(remote.run_status("a02c", "test-one")["status"], "failed")
        self.result(exit_code=124, timed_out=True)
        self.assertEqual(remote.run_status("a02c", "test-one")["status"], "timed_out")

    def test_deleted_container_is_unknown_even_with_success_result(self):
        self.prepared()
        remote.launch_run("a02c", "test-one")
        self.result()
        self.docker.containers.clear()
        self.assertEqual(remote.run_status("a02c", "test-one")["status"], "unknown")
        with self.assertRaisesRegex(ValueError, "refusing to recreate"):
            remote.launch_run("a02c", "test-one")
        self.assertEqual(self.docker.starts, 1)

    def test_interrupted_create_can_be_started_without_recreation(self):
        plan = self.prepared()
        self.docker(remote.create_argv(plan))
        status = remote.launch_run("a02c", "test-one")
        self.assertEqual(status["status"], "running")
        self.assertEqual(len(self.docker.containers), 1)
        self.assertEqual(self.docker.starts, 1)

    def test_stop_is_idempotent_and_does_not_restart(self):
        self.prepared()
        remote.launch_run("a02c", "test-one")
        self.assertEqual(remote.stop_run("a02c", "test-one")["status"], "stopped")
        self.assertTrue(remote.stop_run("a02c", "test-one")["already_stopped"])
        self.assertTrue(remote.launch_run("a02c", "test-one")["already_launched"])
        self.assertEqual((self.docker.starts, self.docker.stops), (1, 1))

    def test_stop_refuses_container_with_wrong_ownership(self):
        self.prepared()
        remote.launch_run("a02c", "test-one")
        self.docker.containers["edge-mcp-test-one"]["Config"]["Labels"].clear()
        with self.assertRaisesRegex(ValueError, "ownership"):
            remote.stop_run("a02c", "test-one")
        self.assertEqual(self.docker.stops, 0)

    def test_snapshot_and_runner_tampering_block_launch(self):
        plan = self.prepared()
        root = Path(plan["job_dir"])
        (root / "source/main.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "snapshot was modified"):
            remote.launch_run("a02c", "test-one")
        self.assertEqual(self.docker.starts, 0)

    def test_new_source_file_or_changed_executable_bit_blocks_launch(self):
        plan = self.prepared()
        path = Path(plan["job_dir"]) / "source/added.py"
        path.write_text("new")
        with self.assertRaisesRegex(ValueError, "inventory"):
            remote.launch_run("a02c", "test-one")
        path.unlink()
        (path.parent / "main.py").chmod(0o755)
        with self.assertRaisesRegex(ValueError, "snapshot was modified"):
            remote.launch_run("a02c", "test-one")

    def test_host_launch_is_a_valid_job_id(self):
        spec = self.spec.model_copy(update={"job_id": "host-launch"})
        remote.prepare_run("a02c", self.payload(spec))
        self.assertEqual(remote.launch_run("a02c", "host-launch")["status"], "running")

    def test_overlapping_output_mounts_block_a_second_active_job(self):
        output = self.artifacts / "shared-campaign"
        output.mkdir()
        spec = self.spec.model_dump()
        spec["mounts"] = [{"source": str(output), "target": "/campaign", "read_only": False}]
        first = remote.prepare_run("a02c", {**self.payload(), "spec": spec})
        remote.launch_run("a02c", "test-one")
        self.docker.containers[first["container_name"]]["Mounts"] = [{"Source": str(output), "RW": True}]
        spec["job_id"] = "test-two"
        remote.prepare_run("a02c", {**self.payload(), "spec": spec})
        with self.assertRaisesRegex(ValueError, "overlaps active"):
            remote.launch_run("a02c", "test-two")
        self.assertEqual(self.docker.starts, 1)

    def test_changed_execution_policy_blocks_launch(self):
        plan = self.prepared()
        with patch.object(remote, "create_argv", return_value=["different-policy"]):
            with self.assertRaisesRegex(ValueError, "Execution policy"):
                remote.launch_run("a02c", "test-one")

    def test_mounts_cannot_write_source_or_managed_records(self):
        for source in [self.work, self.artifacts, self.managed]:
            source.mkdir(exist_ok=True)
            spec = self.spec.model_dump()
            spec["mounts"] = [{"source": str(source), "target": "/external", "read_only": False}]
            with self.assertRaises((ValueError, FileNotFoundError)):
                remote.validate_spec(spec)

    def test_overlapping_container_mounts_are_rejected(self):
        for target in ["/", "/workspace", "/results/child", "/edge"]:
            spec = self.spec.model_dump()
            spec["mounts"] = [{"source": str(self.source), "target": target, "read_only": True}]
            with self.assertRaisesRegex(ValueError, "overlap"):
                remote.validate_spec(spec)

    def test_archive_traversal_and_links_are_rejected_before_extraction(self):
        for name, kind in [("../escape", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)]:
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w:gz") as archive:
                info = tarfile.TarInfo(name)
                info.type, info.linkname = kind, "/etc/passwd"
                archive.addfile(info)
            value = {"archive": base64.b64encode(stream.getvalue()).decode(), "files": {name: {"size": 0, "sha256": "", "mode": 0o644}}}
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                remote.unpack_source(value, self.source)
        self.assertFalse((self.work / "escape").exists())

    def test_artifact_escape_and_log_limit(self):
        plan = self.prepared()
        output = Path(plan["output_dir"])
        (output / "escape").symlink_to(self.source / "main.py")
        with self.assertRaises(ValueError):
            remote.read_artifact("a02c", {"job_id": "test-one", "path": "escape"})
        (output / "stdout.log").write_text("x" * 100000)
        result = remote.read_logs("a02c", {"job_id": "test-one", "tail": 5})
        self.assertTrue(result["truncated"])
        self.assertLessEqual(len(result["text"].encode()), remote.MAX_LOG_BYTES)
        (output / "stdout.log").write_bytes(b"\xff" * 100000)
        result = remote.read_logs("a02c", {"job_id": "test-one", "tail": 5})
        self.assertLessEqual(len(result["text"].encode()), remote.MAX_LOG_BYTES)

    def test_job_lock_rejects_overlapping_operations(self):
        with remote.locked("test-one"):
            with self.assertRaisesRegex(RuntimeError, "lock"):
                with remote.locked("test-one"):
                    self.fail("Second lock was acquired")

    def test_git_snapshot_includes_modified_and_untracked_files(self):
        subprocess.run(["git", "init", "-q", str(self.source)], check=True)
        (self.source / ".gitignore").write_text("ignored\n")
        subprocess.run(["git", "-C", str(self.source), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.source), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "initial"], check=True)
        (self.source / "main.py").write_text("modified")
        (self.source / "new.py").write_text("new")
        (self.source / "ignored").write_text("ignore")
        snap = source_snapshot(str(self.source))
        self.assertEqual(set(snap["files"]), {".gitignore", "main.py", "new.py"})
        self.assertTrue(snap["git"]["dirty"])
        self.assertEqual(snap["sha256"], source_snapshot(str(self.source))["sha256"])

    def test_snapshot_refuses_external_symlinks(self):
        (self.source / "external").symlink_to("/etc/passwd")
        with self.assertRaisesRegex(ValueError, "symlink"):
            source_snapshot(str(self.source))

    def test_literal_shell_metacharacters_remain_command_arguments(self):
        value = "$(touch /tmp/should-not-exist); `id` $HOME"
        spec = self.spec.model_dump()
        spec["command"] = ["python", "-c", "import sys; print(sys.argv[1])", value]
        plan = remote.prepare_run("a02c", {**self.payload(), "spec": spec})
        saved = json.loads((Path(plan["job_dir"]) / "runner/run_spec.json").read_text())
        self.assertEqual(saved["command"][-1], value)


if __name__ == "__main__":
    unittest.main()
