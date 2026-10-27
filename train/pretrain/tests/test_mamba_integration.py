"""Compatibility with the current pretraining integration and historical optimizers."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import yaml

from fm4npp.models.mamba_config import build_backbone, optimizer_parameters
from fm4npp.models.mambagpt import Mamba1GPT
from train.pretrain.nppmamba import train_multi_gpu_mamba1 as pretrain
from train.pretrain.nppmamba.optimizer_state import parameter_names, restore_optimizer

try:
    from fm4npp.models.mamba3 import Mamba3SISO
    HAS_MAMBA3 = True
except ImportError:
    HAS_MAMBA3 = False

ROOT = Path(__file__).resolve().parents[3]


def params(version="mamba2", **overrides):
    result = dict(backbone_version=version, embed_dim=64, num_layers_backbone=1,
                  d_state=16, klen=1, dropout=0., embed_method="pos_only", pe_method="nerf",
                  stat_dir="/unused", use_mem_eff_path=False if version == "mamba2" else None,
                  min_lr=.001, max_lr=.002, warmup_steps=0, total_steps=4,
                  rep_aaai=False, loss_reweight=False, ablate_loss_scale=False,
                  grad_clip_value=1., n_eval_steps=1, save_checkpoint=True)
    result.update(overrides)
    return SimpleNamespace(**result)


def make_trainer(directory, version="mamba2", **overrides):
    channels = 7 if overrides.get("embed_method") == "pos_plus_aux" else 3
    batch = (torch.rand(2, 5, channels), None, torch.rand(2, 5, 3))
    args = SimpleNamespace(root_dir=directory, config=version, run_num="test")
    with patch.object(pretrain, "get_data_loader", return_value=([batch], None, [batch], None)), \
            patch.object(pretrain, "pickle_load", return_value=[0., 1.]), \
            patch("fm4npp.models.mamba2.RMSNormGated", None):
        return pretrain.Trainer(params(version, **overrides), args)


class PretrainingIntegrationTest(unittest.TestCase):
    def test_current_recipe_defaults_keep_mamba1_weights_and_shapes(self):
        for file, key in (("mamba_v602_sweep.yaml", "scale_w64_d12_v602"),
                          ("mamba_v701_aux_sweep.yaml", "scale_w64_d12_v701aux"),
                          ("mamba_v701_pitch_sweep.yaml", "scale_w64_d12_v701pitch")):
            with self.subTest(recipe=key):
                config = yaml.safe_load((ROOT / "scripts/configs" / file).read_text())[key]
                config["num_layers_backbone"] = 1
                torch.manual_seed(81)
                old = Mamba1GPT(embed_dim=config["embed_dim"], num_layers=1,
                                d_state=config["d_state"], klen=config["klen"],
                                dropout=config["dropout"], embed_method=config["embed_method"],
                                pe_method=config["pe_method"])
                torch.manual_seed(81)
                new = build_backbone(config, default="mamba1")
                self.assertEqual(old.state_dict().keys(), new.state_dict().keys())
                for name, value in old.state_dict().items():
                    torch.testing.assert_close(new.state_dict()[name], value, rtol=0, atol=0)
                self.assertEqual(new.backbone_version, "mamba1")
                width = 3 if config["embed_method"] == "pos_only" else 7
                embedding, _ = new.embedder(torch.rand(2, 5, width))
                self.assertEqual(embedding.shape, (2, 5, 64))

    def test_mup_training_validation_limit_and_named_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(directory, use_mup=True, mup_base_width=128,
                                   max_val_batches=1, weight_decay=.02)
            self.assertEqual(len(trainer.optimizer.param_groups), 2)
            self.assertEqual(trainer._mup_matrix_group_indices, [1])
            trainer.valid_data_loader *= 2
            trainer.train_one_epoch()
            rate = trainer.scheduler.get_lr()[0]
            self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], rate)
            self.assertAlmostEqual(trainer.optimizer.param_groups[1]["lr"], rate / .5)
            # Main's max_val_batches denominator must stay capped at one batch.
            with torch.no_grad():
                x, _, target = trainer.valid_data_loader[0]
                expected = (trainer.model(x.to(trainer.device)) - target.to(trainer.device)).square().mean()
            torch.testing.assert_close(trainer.logs["val_loss"].squeeze(), expected)
            resumed = make_trainer(directory, use_mup=True, mup_base_width=128,
                                   max_val_batches=1, weight_decay=.02)
            self.assertEqual(resumed.iters, trainer.iters)
            self.assertEqual(resumed.scheduler.state_dict(), trainer.scheduler.state_dict())
            self.assertEqual(parameter_names(resumed.model, resumed.optimizer),
                             parameter_names(trainer.model, trainer.optimizer))

    def test_offline_wandb_logging_is_preserved(self):
        tracking = Mock()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ), \
                patch.object(pretrain, "_WANDB_AVAILABLE", True), \
                patch.object(pretrain, "wandb", tracking, create=True):
            trainer = make_trainer(directory, use_wandb=True)
            trainer.train_one_epoch()
            self.assertEqual(os.environ["WANDB_MODE"], "offline")
            tracking.init.assert_called_once()
            self.assertEqual(tracking.init.call_args.kwargs["dir"], str(Path(directory) / "wandb"))
            keys = {key for call in tracking.log.call_args_list for key in call.args[0]}
            self.assertTrue({"train/loss", "train/lr", "val/loss"} <= keys)

    @unittest.skipUnless(HAS_MAMBA3, "Mamba3 is optional")
    def test_mamba3_mup_scales_matrix_groups_and_keeps_no_decay(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(directory, "mamba3", use_mup=True, mup_base_width=128)
            self.assertEqual(len(trainer.optimizer.param_groups), 3)
            self.assertEqual(trainer._mup_matrix_group_indices, [2])
            trainer.scheduler.step()
            rate = trainer.scheduler.get_lr()[0]
            trainer._scale_mup_matrix_lr()
            for index, group in enumerate(trainer.optimizer.param_groups):
                expected = rate / .5 if index in trainer._mup_matrix_group_indices else rate
                self.assertAlmostEqual(group["lr"], expected)
                if any(getattr(p, "_no_weight_decay", False) for p in group["params"]):
                    self.assertEqual(group["weight_decay"], 0.)


class LegacyOptimizerResumeTest(unittest.TestCase):
    def check_layout(self, version, layout):
        model = build_backbone(params(version), default="mamba1")
        all_params = list(model.parameters())
        trainable = [p for p in all_params if p.requires_grad]
        rank = [{"params": [p for p in trainable if p.ndim <= 1]},
                {"params": [p for p in trainable if p.ndim > 1]}]
        old_groups = (all_params if layout == "single" else rank if layout == "rank" else
                      optimizer_parameters(model))
        old = torch.optim.AdamW(old_groups, lr=.001, weight_decay=.1)
        sum(p.square().sum() for p in trainable).backward()
        old.step()
        saved = old.state_dict()
        new_groups = optimizer_parameters(model, rank[:1]) + optimizer_parameters(model, rank[1:])
        new = torch.optim.AdamW(new_groups, lr=.002, weight_decay=.1)
        restore_optimizer(model, new, {"optimizer_state_dict": saved})
        for p in trainable:
            torch.testing.assert_close(new.state[p]["exp_avg"], old.state[p]["exp_avg"], rtol=0, atol=0)
            torch.testing.assert_close(new.state[p]["exp_avg_sq"], old.state[p]["exp_avg_sq"], rtol=0, atol=0)
        self.assertEqual({id(p) for g in new.param_groups for p in g["params"]}, {id(p) for p in trainable})

    def test_legacy_single_and_current_rank_groups(self):
        for layout in ("single", "rank"):
            with self.subTest(layout=layout):
                self.check_layout("mamba1", layout)

    @unittest.skipUnless(HAS_MAMBA3, "Mamba3 is optional")
    def test_original_mamba3_no_decay_groups(self):
        self.check_layout("mamba3", "no_decay")

    def test_invalid_named_parameter_join_is_rejected(self):
        model = build_backbone(params("mamba2"))
        optimizer = torch.optim.AdamW(model.parameters())
        names = parameter_names(model, optimizer)
        names[0][0] = names[0][1]
        with self.assertRaisesRegex(ValueError, "duplicated"):
            restore_optimizer(model, optimizer, {"optimizer_state_dict": optimizer.state_dict(),
                                                  "optimizer_param_names": names})


@unittest.skipUnless(torch.cuda.is_available() and HAS_MAMBA3,
                     "CUDA and Mamba3 are required for trainer execution")
class PretrainingCUDATest(unittest.TestCase):
    def test_mamba3_mup_center_and_aux_training_validation_and_resume(self):
        for method in ("pos_only", "pos_plus_aux"):
            with self.subTest(embed_method=method), tempfile.TemporaryDirectory() as directory:
                options = dict(use_mup=True, mup_base_width=128, embed_method=method)
                trainer = make_trainer(directory, "mamba3", **options)
                trainer.train_one_epoch()
                self.assertTrue(torch.isfinite(trainer.logs["val_loss"]).all())
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                    for p in trainer.model.parameters()))
                expected = {name: tensor.clone() for name, tensor in trainer.model.state_dict().items()}
                resumed = make_trainer(directory, "mamba3", **options)
                self.assertEqual(resumed.iters, 1)
                for name, tensor in expected.items():
                    torch.testing.assert_close(resumed.model.state_dict()[name], tensor, rtol=0, atol=0)
                resumed.train_one_epoch()
                self.assertEqual(resumed.iters, 2)
                rate = resumed.scheduler.get_lr()[0]
                for index, group in enumerate(resumed.optimizer.param_groups):
                    self.assertAlmostEqual(group["lr"], rate / .5 if index in resumed._mup_matrix_group_indices else rate)


if __name__ == "__main__":
    unittest.main()
