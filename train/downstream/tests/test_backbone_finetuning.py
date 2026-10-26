import copy
import tempfile
import unittest
from pathlib import Path
import torch
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from train.downstream.backbone_finetuning import (
    configure_backbone, backbone_features, validate_backbone_finetuning,
    restore_tuned_backbone, ScaledGroupScheduler,
)
from train.downstream.lr_schedulers import CosineAnnealingWarmupThenHold
from train.downstream.tests import test_validation_loss_training as validation_tests
from train.downstream.model import RegressionTargetNormalizer


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = torch.nn.Linear(3, 3)
        self.mamba_layers = torch.nn.ModuleList([torch.nn.Linear(3, 3) for _ in range(4)])
        self.norm = torch.nn.LayerNorm(3)
        self.output_layer = torch.nn.Linear(3, 3)

    def forward(self, x, return_z=False):
        x = self.embedder(x)
        features = []
        for block in self.mamba_layers:
            x = x + block(x)
            features.append(x)
        return None, features, None


class FeatureHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.target_normalizer = RegressionTargetNormalizer(3)

    def forward(self, points, feature=None, **kwargs):
        return {"pred_regression": self.linear(feature.mean(0).mean(1))}


class BackboneFinetuningTest(unittest.TestCase):
    def test_only_last_two_blocks_receive_gradients_and_updates(self):
        for count in (0, 2):
            torch.manual_seed(5)
            model = Backbone()
            trainable = configure_backbone(model, count)
            head = FeatureHead()
            before = copy.deepcopy(model.state_dict())
            optimizer = torch.optim.AdamW(list(head.parameters()) + trainable, lr=.001)
            x = torch.randn(3, 5, 3)
            features = backbone_features(model, x, tune=bool(count))
            prediction = head(x, feature=features)["pred_regression"]
            prediction.square().mean().backward()
            optimizer.step()
            for name, p in model.named_parameters():
                expected = count > 0 and name.startswith(("mamba_layers.2.", "mamba_layers.3."))
                self.assertEqual(p.grad is not None, expected, name)
                self.assertEqual(not torch.equal(before[name], p), expected, name)
            self.assertFalse(model.training)

    def test_real_trainer_batch_and_checkpoint_restore_tuned_backbone(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = validation_tests.ValidationLossTrainingTest().trainer(directory)
            trainer.model = Backbone()
            trainer.down_model = FeatureHead()
            trainer.backbone_unfreeze_last_n = 2
            trainer.params["backbone_unfreeze_last_n"] = 2
            trainable = configure_backbone(trainer.model, 2)
            trainer.down_optimizer = torch.optim.AdamW([
                {"params": trainer.down_model.parameters()}, {"params": trainable}], lr=.001)
            trainer.down_scheduler = torch.optim.lr_scheduler.StepLR(trainer.down_optimizer, 1)
            trainer.grad_clip_value = .1
            trainer._representation_input = lambda points, _: points
            initial = copy.deepcopy(trainer.model.state_dict())
            batch = {"points": torch.randn(3, 5, 3), "reg_target": torch.randn(3, 5, 8)}
            loss = trainer._train_one_batch(batch, pretrain=True)
            self.assertTrue(torch.isfinite(torch.tensor(loss)))
            self.assertTrue(torch.equal(initial["mamba_layers.0.weight"], trainer.model.mamba_layers[0].weight))
            self.assertFalse(torch.equal(initial["mamba_layers.3.weight"], trainer.model.mamba_layers[3].weight))
            x = batch["points"]
            expected = trainer.down_model(x, feature=backbone_features(trainer.model, x))["pred_regression"].detach()
            path = trainer._save_checkpoint("tuned.pth", 0, loss)
            saved = torch.load(path, weights_only=False)
            self.assertIn("backbone_state_dict", saved)
            with torch.no_grad():
                trainer.model.mamba_layers[3].weight.add_(10)
                trainer.down_model.linear.weight.add_(10)
            trainer.load_checkpoint(path, inference=True)
            actual = trainer.down_model(x, feature=backbone_features(trainer.model, x))["pred_regression"].detach()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            trainer.load_checkpoint(path, inference=False)
            with self.assertRaisesRegex(ValueError, "contract"):
                restore_tuned_backbone(trainer.model, saved, 0)
            del saved["backbone_state_dict"]
            with self.assertRaisesRegex(ValueError, "contract"):
                restore_tuned_backbone(trainer.model, saved, 2)
            restore_tuned_backbone(trainer.model, {}, 0)  # old frozen checkpoints

    def test_lr_ratios_survive_cycles_hold_and_resume(self):
        for mode in ("restart", "hold"):
            def make():
                optimizer = torch.optim.SGD([
                    {"params": [torch.nn.Parameter(torch.ones(1))]},
                    {"params": [torch.nn.Parameter(torch.ones(1))]}], lr=.1)
                if mode == "restart":
                    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=10,
                        warmup_steps=2, max_lr=.1, min_lr=.01)
                else:
                    scheduler = CosineAnnealingWarmupThenHold(optimizer, anneal_steps=10,
                        warmup_steps=2, max_lr=.1, min_lr=.01)
                return optimizer, ScaledGroupScheduler(scheduler, [1., .1])
            opt, sched = make()
            for step in range(25):
                a, b = sched.get_last_lr()
                self.assertAlmostEqual(b, a * .1)
                if step == 7:
                    opt2, sched2 = make()
                    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
                    sched2.load_state_dict(copy.deepcopy(sched.state_dict()))
                    self.assertEqual(sched2.get_last_lr(), sched.get_last_lr())
                if step >= 7:
                    self.assertEqual(sched2.get_last_lr(), sched.get_last_lr())
                    sched2.step()
                sched.step()

    def test_bad_configuration_fails_early(self):
        base = {"num_layers_backbone": 12, "mambaversion": "mamba1"}
        self.assertEqual(validate_backbone_finetuning(base, False), 0)
        for update, pretrained in [({"backbone_unfreeze_last_n": 2}, False),
                ({"backbone_unfreeze_last_n": 13}, True),
                ({"backbone_unfreeze_last_n": True}, True),
                ({"backbone_unfreeze_last_n": -1}, True),
                ({"backbone_lr_scale": float("nan")}, True),
                ({"backbone_unfreeze_last_n": 2, "mambaversion": "linformer"}, True)]:
            with self.assertRaises(ValueError):
                validate_backbone_finetuning({**base, **update}, pretrained)
