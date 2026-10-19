"""CPU-only checkpoint and mounted-manifest contracts for layer-weight audits."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
import yaml

from train.downstream.probes import audit_layer_weights as audit


class LayerWeightAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def checkpoint(self, weights, legacy=False, depth=2):
        path = self.root / "adapter.pth"
        torch.save({
            "model_state_dict": {
                "module.weighted_avg_weights": torch.tensor(weights),
                "module.input_proj.1.weight": torch.zeros(8, 4),
            },
            "params": {"params": {"num_layers_backbone": depth, "embed_dim": 4}},
            "best_ARI": 0.9,
        }, path, _use_new_zipfile_serialization=not legacy)
        return path

    def test_softmax_ddp_nested_params_projection_and_legacy(self):
        for legacy in (False, True):
            rows, summary = audit.audit_checkpoint(self.checkpoint([0., 1.], legacy))
            self.assertAlmostEqual(sum(r["softmax_weight"] for r in rows), 1., places=6)
            self.assertEqual(summary["top_layer"], 1)
            self.assertEqual(summary["top_layer_1based"], 2)
            self.assertEqual(summary["projection_input_dim"], 4)
            self.assertEqual(summary["projection_output_dim"], 8)
            self.assertEqual(summary["best_ari"], 0.9)

    def test_empty_and_invalid_mixers(self):
        self.assertIsNone(audit.audit_checkpoint(self.checkpoint([], depth=0)))
        for weights, depth in (([float('nan'), 0.], 2), ([0., 1.], 3)):
            with self.assertRaises(ValueError):
                audit.audit_checkpoint(self.checkpoint(weights, depth=depth))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            audit.audit_checkpoint(self.checkpoint([0., 1.]), {"embed_dim": 5})

    def manifest(self):
        self.checkpoint([1., 1.])
        (self.root / "artifacts.json").write_text(json.dumps({"usepretrain": True}))
        manifest = {"campaign_dir": "/remote/campaign", "runs": [
            {"run_id": "raw", "use_pretrained_backbone": False},
            {"run_id": "pretrained", "use_pretrained_backbone": True,
             "adapter_checkpoint": "/remote/campaign/adapter.pth",
             "artifact_summary": "/remote/campaign/artifacts.json",
             "embed_dim": 4, "num_layers_backbone": 2, "eventnumber": 100},
        ]}
        (self.root / "manifest.yaml").write_text(yaml.safe_dump(manifest))
        return manifest

    def test_mounted_campaign_excludes_unused_mixer(self):
        self.manifest()
        selected, skipped = audit.campaign_checkpoints(self.root)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][0], self.root / "adapter.pth")
        self.assertEqual(skipped, [{"run_id": "raw", "reason": "adapter_only_mixer_unused"}])
        _, summary = audit.audit_checkpoint(*selected[0])
        self.assertAlmostEqual(summary["effective_layers"], 2.)

    def test_missing_checkpoint_and_contradictory_training_metadata(self):
        self.manifest()
        (self.root / "artifacts.json").write_text(json.dumps({"usepretrain": False}))
        with self.assertRaisesRegex(ValueError, "contradicts"):
            audit.campaign_checkpoints(self.root)
        (self.root / "adapter.pth").unlink()
        with self.assertRaises(FileNotFoundError):
            audit.campaign_checkpoints(self.root)

    def test_completed_only_excludes_partial_checkpoints(self):
        manifest = self.manifest()
        manifest["runs"][1]["evaluation_dir"] = "/remote/campaign/evaluation"
        (self.root / "manifest.yaml").write_text(yaml.safe_dump(manifest))
        selected, skipped = audit.campaign_checkpoints(self.root, completed_only=True)
        self.assertEqual(selected, [])
        self.assertIn({"run_id": "pretrained", "reason": "evaluation_not_complete"}, skipped)
        (self.root / "evaluation").mkdir()
        (self.root / "evaluation/summary.json").write_text("{}")
        (self.root / "artifacts.json").write_text(json.dumps({
            "usepretrain": True, "final_step": 30000,
            "checkpoint": "/remote/campaign/adapter.pth",
        }))
        selected, skipped = audit.campaign_checkpoints(self.root, completed_only=True)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][1]["final_step"], 30000)


if __name__ == "__main__":
    unittest.main()
