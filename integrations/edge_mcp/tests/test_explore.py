"""Inspection and build ownership/recovery tests without remote access."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from edge_mcp import remote as core, remote_explore as explore
from edge_mcp.client import EdgeClient
from edge_mcp.models import TestSpec
from edge_mcp.snapshot import source_snapshot
from edge_mcp.transport import EdgeError, SSHTransport


class ExploreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name) / "work"
        self.work.mkdir()
        self.managed = self.work / "result_deep_storage/edge_mcp"
        for name, value in [("WORK_ROOT", self.work), ("ARTIFACT_ROOT", self.work / "result_deep_storage"), ("MANAGED_ROOT", self.managed)]:
            handle = patch.object(core, name, value)
            handle.start()
            self.addCleanup(handle.stop)
        self.image_id = "sha256:" + "b" * 64
        self.images = {"pp_collision:alessio-latest": self.image_id, self.image_id: self.image_id}
        self.operations = []
        handle = patch.object(core, "execute", self.engine)
        handle.start()
        self.addCleanup(handle.stop)

    def engine(self, args, timeout=15, check=True):
        self.operations.append(args)
        if args[:3] == ["docker", "image", "inspect"]:
            if args[3] not in self.images:
                return subprocess.CompletedProcess(args, 1, "", "No such image")
            info = {"Id": self.images[args[3]], "Architecture": "arm64", "Os": "linux", "Created": "today", "Size": 123,
                    "Config": {"User": "", "Env": ["SECRET=value"], "Labels": {}}, "RepoTags": [args[3]]}
            return subprocess.CompletedProcess(args, 0, json.dumps([info]), "")
        if args[:3] == ["docker", "image", "tag"]:
            self.images[args[4]] = self.images[args[3]]
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(args)

    def payload(self, **updates):
        return {"environment_id": "test-env", "instructions": "RUN python -V", "base_image": "pp_collision:alessio-latest",
                "source": source_snapshot(None), "worker": "# worker\n", **updates}

    def fake_worker(self, args, **kwargs):
        self.worker_args, self.worker_options = args, kwargs
        root = Path(args[-1])
        plan = json.loads((root / "plan.json").read_text())
        (root / "worker.json").write_text(json.dumps({"pid": 123, "process_token": "test", "fingerprint": plan["fingerprint"]}))
        return Mock(poll=Mock(return_value=None))

    def test_file_pagination_and_literal_metacharacters(self):
        name = self.work / "$(never-execute); `id`.txt"
        name.write_text("abcdef")
        first = explore.read_remote_file({"path": str(name), "max_bytes": 3})
        self.assertEqual(first["text"], "abc")
        self.assertEqual(first["next_offset"], 3)
        self.assertEqual(explore.read_remote_file({"path": str(name), "offset": 3})["text"], "def")
        self.assertEqual(self.operations, [])

    def test_reads_reject_outside_paths_symlink_escape_and_binary(self):
        outside = Path(self.temp.name) / "outside"
        outside.write_text("private")
        (self.work / "escape").symlink_to(outside)
        for path in [outside, self.work / "escape"]:
            with self.assertRaisesRegex(ValueError, "inside"):
                explore.read_remote_file({"path": str(path)})
        binary = self.work / "binary"
        binary.write_bytes(b"a\x00b")
        with self.assertRaisesRegex(ValueError, "Binary"):
            explore.read_remote_file({"path": str(binary)})

    def test_listing_paginates_and_does_not_follow_symlink_directories(self):
        for i in range(5):
            (self.work / f"{i}.py").write_text("needle")
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        (outside / "hidden.py").write_text("not visible")
        (self.work / "escape").symlink_to(outside, target_is_directory=True)
        result = explore.list_remote_files({"path": str(self.work), "pattern": "*.py", "recursive": True, "limit": 2})
        self.assertEqual([e["relative_path"] for e in result["entries"]], ["0.py", "1.py"])
        again = explore.list_remote_files({"path": str(self.work), "pattern": "*.py", "recursive": True, "offset": result["next_offset"]})
        self.assertEqual(len(again["entries"]), 3)

    def test_search_is_literal_and_reports_truncation(self):
        path = self.work / "test.py"
        path.write_text("a+b [value]\nother\na+b [value]\n")
        result = explore.search_remote_files({"path": str(self.work), "query": "a+b [value]", "pattern": "**/*.py", "limit": 1})
        self.assertEqual(result["matches"][0]["line"], 1)
        self.assertTrue(result["truncated"])

    def test_inspection_does_not_create_managed_directories(self):
        (self.work / "config.yaml").write_text("key: value")
        explore.list_remote_files({"path": str(self.work)})
        explore.search_remote_files({"path": str(self.work), "query": "key"})
        self.assertFalse(self.managed.exists())

    def test_image_inspection_excludes_environment_values(self):
        result = explore.inspect_image("pp_collision:alessio-latest")
        self.assertEqual(result["environment_names"], ["SECRET"])
        self.assertNotIn("value", json.dumps(result))

    def test_build_plan_pins_base_and_uses_owned_tag(self):
        plan, _ = explore.environment_plan("a518", self.payload())
        self.assertEqual(plan["base_image_id"], self.image_id)
        self.assertTrue(plan["dockerfile"].startswith("FROM clas12-explore-base:"))
        self.assertEqual(plan["image_tag"], "clas12-explore/test-env:local")
        self.assertIn("--pull=false", plan["build_argv"])
        self.assertNotIn("--push", plan["build_argv"])

    def test_build_disallows_replacing_base_stage_and_reserved_context_file(self):
        with self.assertRaisesRegex(ValueError, "FROM"):
            explore.environment_plan("a518", self.payload(instructions="FROM other-image"))
        with self.assertRaisesRegex(ValueError, "FROM"):
            explore.environment_plan("a518", self.payload(instructions="FROM\\\n other-image"))
        source = self.work / "context"
        source.mkdir()
        (source / "Dockerfile.edge-mcp").write_text("FROM bad")
        with self.assertRaisesRegex(ValueError, "reserved"):
            explore.environment_plan("a518", self.payload(source=source_snapshot(str(source))))

    def test_duplicate_build_reuses_one_detached_worker(self):
        with patch.object(explore.subprocess, "Popen", side_effect=self.fake_worker) as spawn:
            explore.build_environment("a518", self.payload())
            repeated = explore.build_environment("a518", self.payload())
        self.assertTrue(repeated["already_started"])
        self.assertEqual(spawn.call_count, 1)
        self.assertTrue(self.worker_options["start_new_session"])
        self.assertEqual(self.worker_options["stdin"], subprocess.DEVNULL)
        self.assertEqual(self.images["pp_collision:alessio-latest"], self.image_id)
        with self.assertRaisesRegex(ValueError, "different immutable"):
            explore.build_environment("a518", self.payload(instructions="RUN different"))

    def test_existing_target_image_tag_is_never_overwritten(self):
        self.images["clas12-explore/test-env:local"] = self.image_id
        with patch.object(explore.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "overwrite"):
                explore.build_environment("a518", self.payload())
        spawn.assert_not_called()

    def test_failed_build_id_is_not_automatically_restarted(self):
        with patch.object(explore.subprocess, "Popen", side_effect=self.fake_worker):
            explore.build_environment("a518", self.payload())
        root = explore.environment_root("test-env")
        plan = explore.load_environment("a518", "test-env")
        (root / "result.json").write_text(json.dumps({"fingerprint": plan["fingerprint"], "status": "failed", "exit_code": 7}))
        with patch.object(explore.subprocess, "Popen") as spawn:
            result = explore.build_environment("a518", self.payload())
        self.assertEqual(result["status"], "failed")
        spawn.assert_not_called()

    def test_cancellation_refuses_stale_or_unowned_process(self):
        with patch.object(explore.subprocess, "Popen", side_effect=self.fake_worker):
            explore.build_environment("a518", self.payload())
        with patch.object(explore.os, "pidfd_open", create=True) as open_pid:
            with self.assertRaisesRegex(ValueError, "matching live"):
                explore.stop_environment_build("a518", {"environment_id": "test-env"})
        open_pid.assert_not_called()

    def test_remote_source_testing_stages_copy_without_editing_checkout(self):
        source = self.work / "checkout"
        source.mkdir()
        path = source / "test.py"
        path.write_text("print('test')")
        spec = TestSpec(job_id="test-remote", command=["python", "test.py"])
        plan = explore.prepare_remote_run("a518", {"spec": spec.model_dump(), "remote_source_path": str(source), "runner": "# runner"})
        self.assertEqual((Path(plan["job_dir"]) / "source/test.py").read_text(), path.read_text())
        self.assertEqual(path.read_text(), "print('test')")

    def test_transport_keeps_request_and_source_out_of_ssh_command(self):
        transport = SSHTransport()
        response = subprocess.CompletedProcess([], 0, json.dumps({"ok": True, "result": {}}), "")
        with patch("edge_mcp.transport.subprocess.run", return_value=response) as run:
            path = str(self.work / "$(do-not-run)")
            transport.call("a518", "read_remote_file", {"path": path})
            args, options = run.call_args
            self.assertNotIn(path, args[0][-1])
            self.assertLess(len(args[0][-1]), 1500)
            self.assertEqual(json.loads(options["input"])["request"]["payload"]["path"], path)
            transport.close()

    def test_test_runner_retries_busy_host_lock_with_same_job_id(self):
        transport = Mock()
        busy = EdgeError("RuntimeError: Another operation holds this job/host lock; retry with the same job ID")
        transport.call.side_effect = [{}, busy, busy, {"status": "succeeded", "container_id": "same"}, {"text": "PASS", "truncated": False}]
        client = EdgeClient(transport)
        result = client.run_tests("a518", {"job_id": "concurrent-test", "command": ["true"]}, wait_seconds=0)
        self.assertEqual(result["container_id"], "same")
        launches = [call.args[2] for call in transport.call.call_args_list if call.args[1] == "launch_run"]
        self.assertEqual(launches, [{"job_id": "concurrent-test"}] * 3)


if __name__ == "__main__":
    unittest.main()
