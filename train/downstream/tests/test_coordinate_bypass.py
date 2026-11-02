import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from train.downstream.eval.evaluate_track_regression import build_head
from train.downstream.model import MambaTrackRegressionHead
from train.downstream.regression_utils import validate_coordinate_bypass
from train.downstream.track_regression_experiment import TrackRegressionExperimentConfig, resolve_params
from train.downstream.track_regression_trainer import DownstreamTrainer


ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "scripts/configs/mamba_clas12_momentum_adapteronly.yaml"


class Params(dict):
    def __getattr__(self, key):
        if key == "params":
            return self
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 128)

    def forward(self, points, return_z=False):
        hidden = self.projection(points)
        return None, [hidden, hidden.tanh()], None


class CoordinateBypassTest(unittest.TestCase):
    def head(self, enabled=True, **kwargs):
        options = dict(input_dim=128, embed_dim=64, num_layers=1, d_state=4,
                       num_embedder_layers=0, num_feature_layers=2,
                       embed_method="pos_only", pooling="attention", dropout=0.,
                       coordinate_bypass=enabled, return_embedding=True)
        options.update(kwargs)
        # Exercise the repository's actual PyTorch fallback on CPU even when
        # CUDA-only fused extensions happen to be installed in the environment.
        with patch("fm4npp.models.mamba2.RMSNormGated", None):
            head = MambaTrackRegressionHead(**options)
        for layer in head.mamba_layers:
            layer[1].use_mem_eff_path = False
        return head

    def inputs(self):
        torch.manual_seed(7)
        return torch.rand(2, 4, 3), torch.randn(2, 2, 4, 128), torch.ones(2, 4, dtype=torch.bool)

    def trainer(self, enabled=True):
        trainer = DownstreamTrainer.__new__(DownstreamTrainer)
        trainer.params = Params(coordinate_bypass=enabled, task="mom", embed_dim=128,
                                num_layers_backbone=2, num_embedder_layers=0,
                                num_output_classes=3, embed_method="pos_only", pe_method="nerf",
                                pooling="attention")
        trainer.device = "cpu"
        trainer.down_model = self.head(enabled)
        trainer.model = Backbone().eval()
        trainer.down_optimizer = torch.optim.AdamW(trainer.down_model.parameters(), lr=.001)
        trainer.down_scheduler = torch.optim.lr_scheduler.StepLR(trainer.down_optimizer, 1)
        trainer.grad_clip_value = 1.
        trainer.regression_target_stats = dict(task="mom", mean=[0.]*3, std=[1.]*3,
                                              columns=["px", "py", "pz"], path="synthetic.json",
                                              angular_indices=[], phi_pairs=[])
        trainer.regression_loss = "mae"
        trainer.physics_config = None
        trainer.checkpoint_selection = "validation_loss"
        trainer.best_loss = float("inf")
        trainer.log_to_screen = False
        trainer.down_results = {"val": []}
        return trainer

    def test_shared_seeded_parameters_and_legacy_state_keys_are_preserved(self):
        torch.manual_seed(3)
        baseline = self.head(False)
        baseline_random_state = torch.get_rng_state()
        torch.manual_seed(3)
        bypass = self.head(True)
        torch.testing.assert_close(torch.get_rng_state(), baseline_random_state, rtol=0, atol=0)
        self.assertFalse(any(k.startswith("coordinate_proj") for k in baseline.state_dict()))
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, bypass.state_dict()[key], rtol=0, atol=0)

    def test_frozen_mode_ignores_coordinates_only_when_bypass_is_disabled(self):
        points, features, mask = self.inputs()
        for enabled in (False, True):
            head = self.head(enabled).eval()
            before = head(points, features, mask, pretrain=True)["pred_regression"]
            after = head(points + .3, features, mask, pretrain=True)["pred_regression"]
            self.assertEqual(torch.equal(before, after), not enabled)
            other_features = head(points, features * -1, mask, pretrain=True)["pred_regression"]
            self.assertFalse(torch.allclose(before, other_features))

    def test_bypass_uses_the_complete_adapteronly_input_path(self):
        points, features, mask = self.inputs()
        for embed_method, width in (("pos_only", 3), ("pos_plus_aux", 7)):
            with self.subTest(embed_method=embed_method):
                points = torch.rand(2, 4, width)
                baseline = self.head(False, embed_method=embed_method).eval()
                bypass = self.head(True, embed_method=embed_method).eval()
                bypass.load_state_dict(baseline.state_dict(), strict=False)
                # Isolate the coordinate branch and compare it to the actual
                # AdapterOnly mode at the shared refinement-layer boundary.
                bypass.coordinate_proj.load_state_dict(baseline.input_proj.state_dict())
                with torch.no_grad():
                    bypass.input_proj[1].weight.zero_()
                    bypass.input_proj[1].bias.zero_()
                expected = baseline(points, padding_mask=mask)["embedding_post_projection"]
                actual = bypass(points, features, mask, pretrain=True)["embedding_post_projection"]
                torch.testing.assert_close(actual, expected)

    def test_padding_cannot_change_predictions_or_poison_gradients(self):
        points, features, mask = self.inputs()
        for pooling in ("mean", "attention"):
            with self.subTest(pooling=pooling):
                head = self.head(pooling=pooling).eval()
                expected = head(points, features, mask, pretrain=True)["pred_regression"]
                padded_points = torch.cat((points, torch.full((2, 2, 3), float("nan"))), dim=1)
                padded_features = torch.cat((features, torch.full((2, 2, 2, 128), float("nan"))), dim=2)
                padded_mask = torch.cat((mask, torch.zeros(2, 2, dtype=torch.bool)), dim=1)
                actual = head(padded_points, padded_features, padded_mask, pretrain=True)["pred_regression"]
                torch.testing.assert_close(actual, expected)
                actual.square().mean().backward()
                for parameter in head.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_bad_alignment_and_missing_pretraining_fail(self):
        points, features, mask = self.inputs()
        head = self.head()
        for bad_features in (None, features[:, :, :-1], features[:1], features[..., :-1]):
            with self.assertRaisesRegex(ValueError, "row-aligned"):
                head(points, bad_features, mask, pretrain=True)
        with self.assertRaisesRegex(ValueError, "pretrain=True"):
            head(points)
        with self.assertRaisesRegex(ValueError, "boolean"):
            head(points, features, mask.float(), pretrain=True)
        with self.assertRaisesRegex(ValueError, "expects"):
            head(points[..., :2], features, mask, pretrain=True)

    def test_sentinel_padding_can_be_inferred(self):
        points, features, mask = self.inputs()
        mask[:, -1] = False
        points[:, -1] = -100
        head = self.head().eval()
        explicit = head(points, features, mask, pretrain=True)["pred_regression"]
        implicit = head(points, features, pretrain=True)["pred_regression"]
        torch.testing.assert_close(explicit, implicit)

    def test_real_optimizer_step_and_validation_keep_backbone_frozen(self):
        points, _, _ = self.inputs()
        trainer = self.trainer()
        batch = dict(points=points, reg_target=torch.ones(2, 4, 7))
        before = {name: p.detach().clone() for name, p in trainer.model.named_parameters()}
        initial_projection = trainer.down_model.coordinate_proj[1].weight.detach().clone()
        loss = trainer._train_one_batch(batch, pretrain=True)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(initial_projection, trainer.down_model.coordinate_proj[1].weight))
        for module in (trainer.down_model.input_proj, trainer.down_model.coordinate_proj):
            self.assertGreater(module[1].weight.grad.abs().sum().item(), 0)
        for name, parameter in trainer.model.named_parameters():
            self.assertIsNone(parameter.grad)
            torch.testing.assert_close(before[name], parameter, rtol=0, atol=0)
        trainer.val_data_loader = [batch]
        self.assertTrue(torch.isfinite(torch.tensor(trainer.validate_end_to_end_one_epoch(True))))

    def test_checkpoint_roundtrip_and_architecture_mismatches(self):
        points, features, mask = self.inputs()
        with tempfile.TemporaryDirectory() as tmp:
            for enabled in (False, True):
                with self.subTest(enabled=enabled):
                    source = self.trainer(enabled)
                    source.params["checkpoint_dir"] = tmp
                    path = source._save_checkpoint("adapter.pth", 1, .5)
                    payload = torch.load(path, weights_only=False)
                    self.assertIs(payload["coordinate_bypass"], enabled)
                    target = self.trainer(enabled)
                    target.load_checkpoint(path, inference=True)
                    source.down_model.eval()
                    target.down_model.eval()
                    expected = source.down_model(points, features, mask, pretrain=True)["pred_regression"]
                    actual = target.down_model(points, features, mask, pretrain=True)["pred_regression"]
                    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                    with self.assertRaisesRegex(ValueError, "coordinate_bypass"):
                        self.trainer(not enabled).load_checkpoint(path, inference=True)
                    # A historical baseline has no new metadata field.
                    payload.pop("coordinate_bypass")
                    torch.save(payload, path)
                    target.load_checkpoint(path, inference=True)
                    if enabled:
                        del payload["model_state_dict"]["coordinate_proj.1.weight"]
                        torch.save(payload, path)
                        with self.assertRaisesRegex(RuntimeError, "Failed to load adapter"):
                            target.load_checkpoint(path, inference=True)

    def test_evaluator_builds_both_architectures(self):
        for enabled in (False, True):
            trainer = self.trainer(enabled)
            self.assertEqual(build_head(trainer).coordinate_bypass, enabled)

    def test_resolved_recipes_are_matched_and_require_pretraining(self):
        def config(name, **kwargs):
            return TrackRegressionExperimentConfig(yaml_config=str(RECIPE), config=name, **kwargs)
        with self.assertRaisesRegex(ValueError, "--usepretrain"):
            resolve_params(config("clas12_momentum_pretrained_coordinate_bypass"))
        with tempfile.NamedTemporaryFile() as checkpoint:
            baseline = resolve_params(config("clas12_momentum_pretrained", usepretrain=True,
                                             pretrained_ckpt=checkpoint.name)).params
            bypass = resolve_params(config("clas12_momentum_pretrained_coordinate_bypass", usepretrain=True,
                                           pretrained_ckpt=checkpoint.name)).params
        differences = {key for key in baseline if baseline[key] != bypass[key]}
        self.assertEqual(differences, {"coordinate_bypass", "downstream_dir", "checkpoint_dir",
                                      "log_file_name", "checkpoint_file_name"})

    def test_invalid_flag_values_are_rejected(self):
        for value in ("false", "true", 1, None):
            with self.assertRaisesRegex(ValueError, "boolean"):
                validate_coordinate_bypass(dict(coordinate_bypass=value))
        with self.assertRaisesRegex(ValueError, "pos_only"):
            validate_coordinate_bypass(dict(coordinate_bypass=True, embed_method="add"))


if __name__ == "__main__":
    unittest.main()
