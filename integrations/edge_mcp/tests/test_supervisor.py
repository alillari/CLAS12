"""Exercise real supervisor control flow with simulated multiweek elapsed time."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from edge_mcp import supervisor


class SupervisorDeadlineTests(unittest.TestCase):
    def run_supervisor(self, timeout, elapsed):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "run_spec.json"
            spec_path.write_text(json.dumps({"command": ["unused"], "timeout_seconds": timeout}))
            output = root / "output"
            output.mkdir()
            read_fd, write_fd = os.pipe()
            os.close(write_fd)

            class Child:
                pid = 12345

                def __init__(self):
                    self.stdout = os.fdopen(read_fd, "rb")
                    self.polls = iter([None, 0])

                def poll(self):
                    return next(self.polls, 0)

                def wait(self):
                    return 0

            child = Child()
            ticks = iter([0, elapsed])

            def path(value):
                return spec_path if value == "/edge/run_spec.json" else output

            try:
                with patch.object(supervisor, "Path", side_effect=path), \
                     patch.object(supervisor.subprocess, "Popen", return_value=child), \
                     patch.object(supervisor.signal, "signal"), \
                     patch.object(supervisor.time, "monotonic", side_effect=lambda: next(ticks, elapsed)), \
                     patch.object(supervisor.os, "killpg", side_effect=ProcessLookupError) as killed:
                    code = supervisor.main()
                result = json.loads((output / "runner_result.json").read_text())
                return code, result, killed.call_count
            finally:
                child.stdout.close()

    def test_no_deadline_does_not_kill_a_five_week_run(self):
        code, result, kills = self.run_supervisor(None, 35 * 86400)
        self.assertEqual(code, 0)
        self.assertFalse(result["timed_out"])
        self.assertEqual(kills, 0)

    def test_explicit_four_week_deadline_allows_three_weeks(self):
        code, result, kills = self.run_supervisor(28 * 86400, 21 * 86400)
        self.assertEqual(code, 0)
        self.assertFalse(result["timed_out"])
        self.assertEqual(kills, 0)

    def test_explicit_deadline_still_stops_an_overdue_job(self):
        code, result, kills = self.run_supervisor(28 * 86400, 29 * 86400)
        self.assertEqual(code, 124)
        self.assertTrue(result["timed_out"])
        self.assertGreater(kills, 0)


if __name__ == "__main__":
    unittest.main()
