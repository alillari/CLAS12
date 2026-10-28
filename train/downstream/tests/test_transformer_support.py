"""Causal-prefix, bidirectional-context, padding, and transfer contracts."""

import copy
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

from fm4npp.models.mamba_config import (
    adapter_options, architecture_metadata, build_backbone,
    validate_checkpoint_architecture,
)
from fm4npp.models.transformer import SequenceAttention, TransformerBlock
from train.downstream.model import MambaHead, MambaTrackRegressionHead, AttentionHead
from train.downstream.trackinghead import MambaAttentionHead


def config(encoding="rope", **overrides):
    result = dict(backbone_version="transformer", embed_dim=32, num_layers_backbone=2,
                  num_heads_backbone=4, backbone_sequence_encoding=encoding,
                  dropout=0., embed_method="pos_only", pe_method="nerf", klen=1)
    result.update(overrides)
    return result


def head(cls=MambaTrackRegressionHead, encoding="rope", **overrides):
    options = dict(input_dim=32, embed_dim=16, num_layers=1,
                   num_embedder_layers=1, num_feature_layers=2,
                   embed_method="pos_only", dropout=0., num_output_dim=3,
                   **adapter_options(dict(adapter_version="transformer",
                                          adapter_num_heads=4,
                                          adapter_sequence_encoding=encoding)))
    if cls is MambaAttentionHead:
        options.update(num_output_dim=16, num_prototypes=3, ffn_dim=32,
                       num_self_attn_layers=1)
    options.update(overrides)
    return cls(**options)


class TransformerSemanticsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(72)

    def test_backbone_prefix_outputs_features_and_gradients_are_causal(self):
        for encoding in ("none", "rope"):
            with self.subTest(encoding=encoding):
                model = build_backbone(config(encoding)).eval()
                x = torch.rand(2, 9, 3, requires_grad=True)
                original = model(x)
                changed = x.detach().clone()
                changed[:, 4:] += 3.
                torch.testing.assert_close(model(changed)[:, :4], original[:, :4])
                torch.testing.assert_close(model(x[:, :4]), original[:, :4])
                _, full_features, _ = model(x, return_z=True)
                _, prefix_features, _ = model(x[:, :4], return_z=True)
                for full, prefix in zip(full_features, prefix_features):
                    torch.testing.assert_close(full[:, :4], prefix)
                original[:, :4].square().sum().backward()
                self.assertEqual(x.grad[:, 4:].abs().max().item(), 0.)
                self.assertGreater(x.grad[:, :4].abs().max().item(), 0.)

    def test_bidirectional_attention_uses_later_tokens_for_both_encodings(self):
        for encoding in ("none", "rope"):
            model = TransformerBlock(16, 4, causal=False, sequence_encoding=encoding).eval()
            x = torch.rand(1, 6, 16, requires_grad=True)
            model(x)[:, 0].square().sum().backward()
            self.assertGreater(x.grad[:, -1].abs().max().item(), 1e-6)

    def test_nope_is_permutation_equivariant_but_rope_encodes_order(self):
        x = torch.randn(2, 7, 16)
        permutation = torch.tensor([3, 0, 6, 2, 1, 5, 4])
        for encoding in ("none", "rope"):
            model = TransformerBlock(16, 4, sequence_encoding=encoding).eval()
            error = (model(x[:, permutation]) - model(x)[:, permutation]).abs().max().item()
            if encoding == "none":
                self.assertLess(error, 2e-6)
            else:
                self.assertGreater(error, 1e-4)

    def test_rotary_rotation_and_relative_position_identity(self):
        attention = SequenceAttention(8, 2, causal=True)
        x = torch.tensor([[[[1., 0., 1., 0.]]]])
        rotated = attention._rotate(x, torch.tensor([[1]]))
        expected = torch.tensor([math.cos(1.), math.sin(1.), math.cos(.01), math.sin(.01)]).reshape(1, 1, 1, 4)
        torch.testing.assert_close(rotated, expected)
        q, k = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)
        positions = torch.tensor([[0, 2, 7]])
        before = attention._rotate(q, positions) @ attention._rotate(k, positions).transpose(-2, -1)
        shifted = attention._rotate(q, positions + 11) @ attention._rotate(k, positions + 11).transpose(-2, -1)
        torch.testing.assert_close(before, shifted)

    def test_padding_is_excluded_including_left_internal_and_all_padding(self):
        for causal in (False, True):
            for encoding in ("none", "rope"):
                with self.subTest(causal=causal, encoding=encoding):
                    model = TransformerBlock(16, 4, causal=causal, sequence_encoding=encoding).eval()
                    x = torch.randn(1, 4, 16)
                    mask = torch.tensor([[False, True, True, False, True, True, False]])
                    padded = torch.full((1, 7, 16), float("nan"))
                    padded[:, mask[0]] = x
                    result = model(padded, mask)
                    torch.testing.assert_close(result[:, mask[0]], model(x))
                    self.assertEqual(result[:, ~mask[0]].abs().max().item(), 0.)
                    empty = torch.full((2, 5, 16), float("nan"), requires_grad=True)
                    output = model(empty, torch.zeros(2, 5, dtype=torch.bool))
                    self.assertEqual(output.abs().max().item(), 0.)
                    output.sum().backward()
                    self.assertTrue(torch.isfinite(empty.grad).all())
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                        for p in model.parameters()))

    def test_backbone_infers_padding_and_supports_all_token_embeddings(self):
        for method, channels in (("pos_only", 3), ("pos_plus_aux", 7), ("add", 4), ("concat", 4)):
            for encoding in ("none", "rope"):
                model = build_backbone(config(encoding, embed_method=method)).eval()
                x = torch.rand(2, 5, channels)
                padded = torch.cat((x, torch.full((2, 3, channels), -100.)), dim=1)
                torch.testing.assert_close(model(x), model(padded)[:, :5])
                self.assertEqual(model(padded)[:, 5:].abs().max().item(), 0.)
                self.assertEqual(model(padded).shape, (2, 8, 3))

    def test_eval_disables_attention_dropout(self):
        model = TransformerBlock(16, 4, dropout=.5).eval()
        x = torch.rand(2, 5, 16)
        torch.testing.assert_close(model(x), model(x), rtol=0, atol=0)

    def test_invalid_options_and_zero_layer_adapters_fail_clearly(self):
        for kwargs in (dict(num_heads=3), dict(sequence_encoding="absolute"),
                       dict(rope_theta=0), dict(mlp_ratio=0), dict(dropout=1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TransformerBlock(16, **kwargs)
        with self.assertRaisesRegex(ValueError, "even"):
            TransformerBlock(12, 4, sequence_encoding="rope")
        TransformerBlock(12, 4, sequence_encoding="none")
        for cls in (MambaHead, AttentionHead, MambaAttentionHead, MambaTrackRegressionHead):
            with self.assertRaisesRegex(ValueError, "at least one"):
                head(cls, num_layers=0)


class TransformerCheckpointTest(unittest.TestCase):
    @staticmethod
    def payload(model, kind):
        key = "model_state" if kind == "backbone" else "model_state_dict"
        return {key: model.state_dict(), f"{kind}_architecture": architecture_metadata(model)}

    def test_reload_and_encoding_mismatch_for_backbone_and_adapter(self):
        for kind, factory in (("backbone", lambda enc: build_backbone(config(enc))),
                               ("adapter", lambda enc: head(encoding=enc))):
            for encoding in ("none", "rope"):
                model = factory(encoding)
                saved = self.payload(model, kind)
                clone = factory(encoding)
                validate_checkpoint_architecture(saved, clone, kind)
                clone.load_state_dict(next(v for k, v in saved.items() if k.startswith("model_state")), strict=True)
                with self.assertRaisesRegex(ValueError, "architecture does not match"):
                    validate_checkpoint_architecture(saved, factory("rope" if encoding == "none" else "none"), kind)
                unnamed = {k: v for k, v in saved.items() if not k.endswith("_architecture")}
                with self.assertRaisesRegex(ValueError, "require architecture metadata"):
                    validate_checkpoint_architecture(unnamed, clone, kind)

    def test_other_same_shape_settings_and_frozen_backbone_are_checked(self):
        model = build_backbone(config())
        for override in (dict(num_heads_backbone=8), dict(backbone_rope_theta=1000.),
                         dict(dropout=.1), dict(pe_method="ff")):
            with self.assertRaisesRegex(ValueError, "architecture does not match"):
                validate_checkpoint_architecture(self.payload(model, "backbone"),
                                                 build_backbone(config(**override)), "backbone")
        adapter = head()
        saved = {**self.payload(adapter, "adapter"), "backbone_architecture": architecture_metadata(model)}
        with self.assertRaisesRegex(ValueError, "different backbone"):
            validate_checkpoint_architecture(saved, adapter, "adapter",
                                             backbone=build_backbone(config("none")))

    def test_legacy_mamba_checkpoint_rejects_cross_family_loads(self):
        mamba = build_backbone(config(backbone_version="mamba2", use_mem_eff_path=False))
        transformer = build_backbone(config())
        for source, target in ((mamba, transformer), (transformer, mamba)):
            saved = self.payload(source, "backbone")
            with self.assertRaises(ValueError):
                validate_checkpoint_architecture(saved, target, "backbone")
            saved.pop("backbone_architecture")
            with self.assertRaises(ValueError):
                validate_checkpoint_architecture(saved, target, "backbone")


class TransformerTransferTest(unittest.TestCase):
    def test_example_backbones_match_the_pretraining_checkpoint_architecture(self):
        root = Path(__file__).resolve().parents[3] / "scripts/configs"
        pretraining = yaml.safe_load((root / "transformer_pretrain.yaml").read_text())
        references = {encoding: architecture_metadata(build_backbone(pretraining[f"transformer_{name}_v602"]))
                      for encoding, name in (("rope", "rope"), ("none", "nope"))}
        for filename in ("mamba_clas12_track_regression_pretrained.yaml",
                         "mamba_clas12_track_regression_adapteronly.yaml",
                         "mamba_clas12_track_finding_adapteronly.yaml",
                         "mamba_clas12_point_classification_adapteronly.yaml"):
            for key, params in yaml.safe_load((root / filename).read_text()).items():
                if "_transformer_" in key:
                    self.assertEqual(architecture_metadata(build_backbone(params)),
                                     references[params["backbone_sequence_encoding"]], key)

    def test_all_task_heads_train_adapteronly_and_frozen_feature_paths(self):
        for encoding in ("none", "rope"):
            backbone = build_backbone(config(encoding)).eval()
            x = torch.rand(2, 7, 3)
            mask = torch.ones(2, 7, dtype=torch.bool)
            mask[1, -2:] = False
            x[~mask] = -100
            with torch.no_grad():
                _, features, _ = backbone(x, return_z=True)
                features = torch.stack(features)
            for cls, key in ((MambaHead, "pred_logits"), (AttentionHead, "pred_logits"),
                             (MambaTrackRegressionHead, "pred_regression"),
                             (MambaAttentionHead, "mask_probs")):
                for pretrain in (False, True):
                    with self.subTest(encoding=encoding, cls=cls.__name__, pretrain=pretrain):
                        model = head(cls, encoding)
                        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                        output = model(x, features if pretrain else None, padding_mask=mask, pretrain=pretrain)[key]
                        self.assertTrue(torch.isfinite(output).all())
                        output.square().mean().backward()
                        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                            for p in model.parameters()))
                        optimizer.step()
                        clone = head(cls, encoding).eval()
                        clone.load_state_dict(model.state_dict(), strict=True)
                        model.eval()
                        torch.testing.assert_close(
                            clone(x, features, padding_mask=mask, pretrain=pretrain)[key],
                            model(x, features, padding_mask=mask, pretrain=pretrain)[key])

    def test_auxiliary_tokens_and_masked_values_do_not_leak_into_valid_outputs(self):
        for encoding in ("none", "rope"):
            for aux_dim in (4, 5):
                x = torch.rand(2, 7, 3 + aux_dim)
                mask = torch.ones(2, 7, dtype=torch.bool)
                mask[:, -2:] = False
                altered = x.clone()
                altered[~mask] = float("nan")
                for cls, key in ((MambaHead, "pred_logits"),
                                 (MambaTrackRegressionHead, "pred_regression"),
                                 (MambaAttentionHead, "class_probs")):
                    model = head(cls, encoding, embed_method="pos_plus_aux", aux_dim=aux_dim).eval()
                    first = model(x, padding_mask=mask)[key]
                    second = model(altered, padding_mask=mask)[key]
                    if cls is MambaHead:
                        first, second = first[mask], second[mask]
                    torch.testing.assert_close(first, second)
                    second.square().sum().backward()
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                        for p in model.parameters()))

    def test_backbone_and_adapter_families_are_independent(self):
        with patch("fm4npp.models.mamba2.RMSNormGated", None):
            for backbone_family, adapter_family in (("mamba2", "transformer"), ("transformer", "mamba2")):
                backbone = build_backbone(config(backbone_version=backbone_family, use_mem_eff_path=False))
                options = adapter_options(dict(adapter_version=adapter_family, adapter_use_mem_eff_path=False))
                adapter = MambaTrackRegressionHead(input_dim=32, embed_dim=32, num_layers=1,
                                                   num_embedder_layers=1, num_feature_layers=2,
                                                   embed_method="pos_only", **options)
                x = torch.rand(2, 6, 3)
                with torch.no_grad():
                    _, features, _ = backbone(x, return_z=True)
                prediction = adapter(x, torch.stack(features), pretrain=True)["pred_regression"]
                prediction.square().mean().backward()
                self.assertTrue(torch.isfinite(prediction).all())

    def test_regression_and_tracking_evaluators_build_selected_adapter(self):
        from train.downstream.eval.evaluate_track_regression import build_head
        from train.downstream.eval.evaluate_track_finding import init_down_model
        params = SimpleNamespace(**config(), adapter_version="transformer", adapter_sequence_encoding="none",
                                 num_output_classes=3, num_embedder_layers=1, num_adapter_layers=2,
                                 num_prototypes=3, max_gt_classes=3, adapter_embed_dim=16, max_lr=.001)
        trainer = SimpleNamespace(params=params, device="cpu",
                                  regression_target_stats={"mean": [0.] * 3, "std": [1.] * 3})
        for model in (build_head(trainer), init_down_model(params, "cpu")):
            self.assertEqual(model.embed_dim, 16)
            self.assertEqual(len(model.mamba_layers), 2)
            metadata = architecture_metadata(model)
            self.assertEqual(metadata["version"], "transformer")
            self.assertTrue(metadata["blocks"])
            self.assertTrue(all(c["sequence_encoding"] == "none" and not c["causal"]
                                for c in metadata["blocks"].values()))

    def test_real_regression_training_step_and_tracking_matching_loss(self):
        from train.downstream.tests.test_validation_loss_training import ValidationLossTrainingTest
        from train.downstream.tests.test_track_finding_validation_history import fixture_trainer
        for encoding in ("none", "rope"):
            for pretrain in (False, True):
                with tempfile.TemporaryDirectory() as directory:
                    trainer = ValidationLossTrainingTest().trainer(directory)
                    trainer.model = build_backbone(config(encoding)).eval()
                    trainer.down_model = head(encoding=encoding)
                    trainer.down_optimizer = torch.optim.AdamW(trainer.down_model.parameters(), lr=.001)
                    trainer.grad_clip_value = 1.
                    batch = dict(points=torch.rand(2, 6, 3), reg_target=torch.rand(2, 6, 3))
                    loss = trainer._train_one_batch(batch, pretrain=pretrain)
                    self.assertTrue(math.isfinite(loss))
                    self.assertTrue(trainer.down_optimizer.state)
                with tempfile.TemporaryDirectory() as directory:
                    trainer = fixture_trainer(directory)
                    trainer.model = build_backbone(config(encoding)).eval()
                    trainer.down_model = head(MambaAttentionHead, encoding=encoding)
                    loss, *_ = trainer._compute_batch_loss(trainer.val_data_loader[0], pretrain=pretrain, train=True)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                        for p in trainer.down_model.parameters()))


class TransformerPretrainingTest(unittest.TestCase):
    def test_trainer_train_validate_and_optimizer_resume_for_center_and_aux(self):
        from train.pretrain.tests.test_mamba_integration import make_trainer
        for encoding in ("none", "rope"):
            for method in ("pos_only", "pos_plus_aux"):
                with self.subTest(encoding=encoding, method=method), tempfile.TemporaryDirectory() as directory:
                    kwargs = dict(backbone_sequence_encoding=encoding, embed_method=method,
                                  use_mup=True, mup_base_width=128, max_val_batches=1)
                    trainer = make_trainer(directory, "transformer", **kwargs)
                    trainer.train_one_epoch()
                    self.assertTrue(torch.isfinite(trainer.logs["val_loss"]).all())
                    saved = copy.deepcopy(trainer.model.state_dict())
                    optimizer = copy.deepcopy(trainer.optimizer.state_dict())
                    resumed = make_trainer(directory, "transformer", **kwargs)
                    self.assertEqual(resumed.iters, trainer.iters)
                    self.assertEqual(resumed.scheduler.state_dict(), trainer.scheduler.state_dict())
                    for name, tensor in resumed.model.state_dict().items():
                        torch.testing.assert_close(tensor, saved[name], rtol=0, atol=0)
                    for key, state in resumed.optimizer.state_dict()["state"].items():
                        for item, value in state.items():
                            torch.testing.assert_close(value, optimizer["state"][key][item], rtol=0, atol=0)
                    resumed.train_one_epoch()
                    self.assertEqual(resumed.iters, 2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is optional; CPU tests cover transformer semantics")
class TransformerCUDATest(unittest.TestCase):
    def test_cuda_autocast_and_masked_transfer(self):
        for encoding in ("none", "rope"):
            model = build_backbone(config(encoding, embed_dim=256)).cuda()
            adapter = head(encoding=encoding, input_dim=256, embed_dim=64).cuda()
            x = torch.rand(2, 65, 3, device="cuda")
            x[1, -7:] = -100
            mask = x[..., 0] != -100
            optimizer = torch.optim.AdamW(list(model.parameters()) + list(adapter.parameters()))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x)
                _, features, _ = model(x, return_z=True)
                task = adapter(x, torch.stack(features), padding_mask=mask, pretrain=True)["pred_regression"]
                loss = pred[mask].float().square().mean() + task.float().square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                for p in list(model.parameters()) + list(adapter.parameters())))
            optimizer.step()


if __name__ == "__main__":
    unittest.main()
