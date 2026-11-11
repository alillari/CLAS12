"""Exercise both public evaluators with real CPU heads and tiny HDF5 data."""
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from ruamel.yaml import YAML

from fm4npp.models.mamba_config import architecture_metadata
from fm4npp.hilbert import DEFAULT_HILBERT_GRID
from fm4npp.utils import YParams
from train.downstream.eval.evaluate_track_regression import build_head, DownstreamTrainer
from train.downstream.model import RegressionTargetNormalizer
from train.downstream.physics_checkpoints import resolve_config
from train.downstream.scripts.compute_regression_target_stats import compute_stats
from train.downstream.tests.test_cvt_downstream import event, publish_fixture

ROOT = Path(__file__).resolve().parents[3]


class MomentumEvaluationUnitsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        data = cls.root / "data"
        data.mkdir()
        publish_fixture(data, {split: [[event(i)]] for i, split in enumerate(("train", "validation", "test"))})
        cls.stats = compute_stats(data, "train", 1, 100, 2, 10, task="mom",
                                  reader_type="hdf5", segment_min_clusters=2)
        (cls.root / "stats.json").write_text(json.dumps(cls.stats))
        params = YParams(str(ROOT / "scripts/configs/clas12_adapter_recipes.yaml"), "hdf5_momentum_5k")
        for key, value in dict(data_root=str(data), data_root_test=str(data), stat_dir=str(data / "stats"),
                artifact_root=str(cls.root), checkpoint_dir=str(cls.root / "checkpoints"),
                regression_target_stats=str(cls.root / "stats.json"), pretrained_ckpt=None,
                backbone_version="transformer", adapter_version="transformer", embed_dim=32,
                base_dim=32, adapter_embed_dim=32, num_layers_backbone=1, num_adapter_layers=1,
                num_embedder_layers=0, num_data_workers=0, segment_min_clusters=2,
                limit_size=2, limit_test_size=2, hilbert_angular_grid=DEFAULT_HILBERT_GRID,
                physics_checkpoint={"momentum_scale_to_gev": 1.0}).items():
            params[key] = value
        cls.params = params.params
        cls.dump(cls.root / "model.yaml", {"units": cls.params})
        with patch("torch.cuda.is_available", return_value=False):
            trainer = DownstreamTrainer(params, SimpleNamespace(root_dir=str(cls.root / "runtime"),
                global_log_dir="logs", config="units", run_num="0"))
            cls.head = build_head(trainer)
        with torch.no_grad():
            for parameter in cls.head.parameters():
                parameter.zero_()

    @staticmethod
    def dump(path, data):
        with path.open("w") as stream:
            YAML().dump(data, stream)

    def checkpoint(self, kind):
        gev = kind == "gev"
        self.head.target_normalizer = RegressionTargetNormalizer(
            3, [1., 2., 3.] if gev else [1000., 2000., 3000.], [1., 1., 1.])
        saved = dict(self.params, reader_type="hdf5" if gev else "ragged_mmap")
        saved.pop("physics_checkpoint")
        payload = dict(model_state_dict=self.head.state_dict(), regression_task="mom",
            regression_target_columns=self.stats["columns"], use_pretrained_backbone=False,
            pooling=self.head.pooling, adapter_architecture=architecture_metadata(self.head),
            hilbert_angular_grid=self.params["hilbert_angular_grid"], params={"params": saved})
        if kind == "gev":
            payload["data_contract"] = self.stats["data_contract"]
            payload["physics_checkpoint_config"] = resolve_config("mom", {"momentum_scale_to_gev": 1.0})
        elif kind == "mev":
            payload["physics_checkpoint_config"] = resolve_config("mom", {"momentum_scale_to_gev": .001})
        elif kind == "legacy_params":
            saved["physics_checkpoint"] = {"momentum_scale_to_gev": .001}
        path = self.root / f"{kind}.pth"
        torch.save(payload, path)
        return path

    def evaluate(self, kind, entrypoint):
        checkpoint = self.checkpoint(kind)
        output = self.root / f"{entrypoint}_{kind}"
        if entrypoint == "physics":
            analysis = YAML(typ="safe").load(ROOT / "train/downstream/eval/track_regression_analysis_adapteronly.yaml")["analysis"]
            analysis.update(model_yaml=str(self.root / "model.yaml"), model_config="units",
                checkpoint=str(checkpoint), output_dir=str(output), training_log=None,
                max_samples=2, batch_size=2, num_workers=0, physics_histogram_bins=10)
            path = self.root / f"analysis_{kind}.yaml"
            self.dump(path, {"analysis": analysis})
            command = ["train/downstream/eval/evaluate_track_regression.py", "--analysis-config", str(path)]
        else:
            command = ["train/downstream/eval_momentum_regression.py", "--yaml_config", str(self.root / "model.yaml"),
                "--config", "units", "--checkpoint", str(checkpoint), "--root_dir", str(output),
                "--eventnumber", "2", "--eval_batch_size", "2"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MPLBACKEND="Agg",
                   MPLCONFIGDIR=str(self.root / "matplotlib"), PYTHONPATH=str(ROOT))
        result = subprocess.run([sys.executable, *command], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
        return result, output

    def assert_rejected(self, kind, message):
        for entrypoint in ("physics", "compatibility"):
            with self.subTest(entrypoint=entrypoint):
                result, output = self.evaluate(kind, entrypoint)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(message, result.stdout)
                self.assertFalse(list(output.rglob("predictions.csv.gz")))
                self.assertFalse(list(output.rglob("physics_checkpoint_summary.json")))

    def test_both_evaluators_reject_explicit_mev_checkpoint(self):
        self.assert_rejected("mev", "Checkpoint momentum unit mismatch")

    def test_both_evaluators_check_legacy_saved_params_without_physics_metadata(self):
        self.assert_rejected("legacy_params", "Checkpoint momentum unit mismatch")

    def test_both_evaluators_reject_unknown_checkpoint_units(self):
        self.assert_rejected("unknown", "Checkpoint momentum units are unknown")

    def test_both_evaluators_preserve_gev_truth_and_predictions(self):
        for entrypoint in ("physics", "compatibility"):
            with self.subTest(entrypoint=entrypoint):
                result, output = self.evaluate("gev", entrypoint)
                self.assertEqual(result.returncode, 0, result.stdout)
                summary_path, = output.rglob("physics_checkpoint_summary.json")
                summary = json.loads(summary_path.read_text())
                self.assertEqual(summary["config"]["momentum_scale_to_gev"], 1.)
                self.assertEqual(summary["checkpoint"]["n_valid_truth"], 2)
                expected = (np.sqrt(14.) / np.linalg.norm([.2, .3, .4]) - 1.) / 2.
                self.assertAlmostEqual(summary["checkpoint"]["inclusive"]["q50"], expected, places=5)
                if entrypoint == "physics":
                    with gzip.open(output / "predictions.csv.gz", "rt") as stream:
                        row = next(csv.DictReader(stream))
                    for prefix in ("true", "adapter"):
                        np.testing.assert_allclose([float(row[f"{prefix}_{axis}_gev"]) for axis in ("px", "py", "pz")], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
