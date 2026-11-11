"""Evaluation policy metadata, cache invalidation and campaign comparison guards."""

import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fm4npp.datasets.track_eligibility import primary_track_eligibility_contract
from train.downstream.track_finding_contract import (
    FINDING_METRICS_VERSION, finding_evaluation_metadata,
)

with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "campaign"), *sys.path]):
    from train.downstream.campaign import run_track_finding_campaign as runner


def current_summary():
    return {"finding_metrics_version": FINDING_METRICS_VERSION,
            "track_eligibility_contract": primary_track_eligibility_contract(),
            "metrics": {"track_efficiency_global": .75}}


def incompatible_summaries():
    missing = current_summary()
    missing.pop("track_eligibility_contract")
    yield "missing policy", missing
    for key, value in (("policy", "old_policy"), ("min_clusters", 3),
                       ("count_support", "cropped_event"), ("background_label", 0)):
        wrong = current_summary()
        wrong["track_eligibility_contract"][key] = value
        yield key, wrong
    extra = current_summary()
    extra["track_eligibility_contract"]["max_clusters"] = 12
    yield "unsupported selection", extra
    old_metrics = current_summary()
    old_metrics["finding_metrics_version"] = FINDING_METRICS_VERSION - 1
    yield "metric version", old_metrics


class FindingEvaluationContractTest(unittest.TestCase):
    def make_run(self, root, name, summary=None, headline=None):
        directory = root / name
        directory.mkdir()
        if summary is not None:
            (directory / "summary.json").write_text(json.dumps(summary))
        if headline is not None:
            (directory / "campaign_headline_metrics.jsonl").write_text(json.dumps(headline) + "\n")
        return dict(run_id=name, evaluation_dir=str(directory), status="eval_done",
                    embed_dim=64, num_layers_backbone=2, pretrain_events=100,
                    adapter_checkpoint="unused.pth", model_yaml="unused.yaml",
                    model_config="unused", eval_stdout=str(root / f"{name}.log"))

    def eval_args(self, force=False):
        return SimpleNamespace(skip_eval=False, force_eval=force, cuda_device="0",
                               cuda_preflight_attempts=1, cuda_preflight_retry_delay_seconds=0)

    def test_metadata_requires_verified_dataset_policy(self):
        for contract in (None, {"policy": "primary_mc_min4_v1"},
                         {**primary_track_eligibility_contract(), "min_clusters": 3}):
            with self.subTest(contract=contract), self.assertRaisesRegex(ValueError, "track_eligibility_contract"):
                finding_evaluation_metadata(SimpleNamespace(track_eligibility_contract=contract))
        with self.assertRaisesRegex(ValueError, "track_eligibility_contract"):
            finding_evaluation_metadata(SimpleNamespace())

    def test_valid_cache_is_reused_without_launch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = self.make_run(root, "one", current_summary(), current_summary())
            with patch.object(runner, "run_with_cuda_preflight") as launch:
                runner.eval_if_needed(self.eval_args(), {}, root / "status.yaml", run)
            launch.assert_not_called()
            self.assertFalse((root / "status.yaml").exists())

    def test_stale_cache_is_reevaluated_and_new_contract_is_checked(self):
        cases = [(name, summary, None, False) for name, summary in incompatible_summaries()]
        cases += [("stale headline", current_summary(), {"finding_metrics_version": 2}, False),
                  ("force valid cache", current_summary(), None, True)]
        for name, old, headline, force in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                run = self.make_run(root, "one", old, headline)
                directory = Path(run["evaluation_dir"])
                def evaluate(*args, **kwargs):
                    (directory / "summary.json").write_text(json.dumps(current_summary()))
                    (directory / "campaign_headline_metrics.jsonl").write_text(json.dumps(current_summary()) + "\n")
                    return 0, root / "eval.log", root / "preflight.log", 1
                with patch.object(runner, "eval_command", return_value=["fixture-eval"]), \
                     patch.object(runner, "command_env", return_value={}), \
                     patch.object(runner, "run_with_cuda_preflight", side_effect=evaluate) as launch:
                    runner.eval_if_needed(self.eval_args(force), {}, root / "status.yaml", run)
                launch.assert_called_once()
                self.assertEqual(runner.load_status(root / "status.yaml")["runs"]["one"]["status"], "eval_done")

    def test_successful_subprocess_cannot_publish_an_unmarked_summary(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = self.make_run(root, "one", {"finding_metrics_version": FINDING_METRICS_VERSION})
            with patch.object(runner, "eval_command", return_value=["fixture-eval"]), \
                 patch.object(runner, "command_env", return_value={}), \
                 patch.object(runner, "run_with_cuda_preflight",
                              return_value=(0, root / "eval.log", root / "preflight.log", 1)):
                with self.assertRaisesRegex(ValueError, "track_eligibility_contract"):
                    runner.eval_if_needed(self.eval_args(), {}, root / "status.yaml", run)
            record = runner.load_status(root / "status.yaml")["runs"]["one"]
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["reason"], "incompatible_evaluation_contract")

    def test_incompatible_collation_preserves_existing_outputs(self):
        cases = [(name, summary, None) for name, summary in incompatible_summaries()]
        cases.append(("stale headline", current_summary(), {"finding_metrics_version": 2}))
        for name, summary, headline in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                good = self.make_run(root, "good", current_summary())
                bad = self.make_run(root, "bad", summary, headline)
                output = root / "summary"
                output.mkdir()
                for filename in ("run_table.csv", "campaign_headline_metrics.jsonl"):
                    (output / filename).write_text("previous valid output\n")
                with self.assertRaisesRegex(ValueError, "Reevaluate"):
                    runner.collate_summary({"campaign_dir": str(root), "runs": [good, bad]})
                for filename in ("run_table.csv", "campaign_headline_metrics.jsonl"):
                    self.assertEqual((output / filename).read_text(), "previous valid output\n")

    def test_matching_contracts_collate_with_explicit_policy_columns(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs = [self.make_run(root, name, current_summary(), current_summary())
                    for name in ("one", "two")]
            runner.collate_summary({"campaign_dir": str(root), "runs": runs})
            with (root / "summary/run_table.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["track_efficiency_global"], "0.75")
                self.assertEqual(row["track_eligibility_policy"], "primary_mc_min4_v1")
                self.assertEqual(row["track_eligibility_min_clusters"], "4")
                self.assertEqual(row["track_eligibility_count_support"], "full_event_owned_clusters")
                self.assertEqual(row["track_eligibility_background_label"], "-1")
            records = [json.loads(line) for line in
                       (root / "summary/campaign_headline_metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertTrue(all(row["track_eligibility_contract"] == primary_track_eligibility_contract()
                                for row in records))


if __name__ == "__main__":
    unittest.main()
