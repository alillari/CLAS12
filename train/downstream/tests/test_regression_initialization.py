import copy
import tempfile
import unittest
from unittest.mock import patch

import torch

from train.downstream.model import MambaTrackRegressionHead
from train.downstream.regression_initialization import initialize_regression_adapter
from train.downstream.tests import test_validation_loss_training as training_tests


def head():
    return MambaTrackRegressionHead(
        input_dim=64, embed_dim=64, num_layers=1, num_embedder_layers=0,
        num_feature_layers=3, d_state=16, embed_method="pos_only", pooling="attention",
    )


class RegressionInitializationTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for fused Mamba2 execution')
    def test_cuda_updates_and_checkpoint_roundtrip(self):
        for mode in ('constructor', 'legacy'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                model = head().cuda()
                initialize_regression_adapter(model, mode)
                points = torch.randn(2, 8, 3, device='cuda')
                mask = torch.ones(2, 8, dtype=torch.bool, device='cuda')
                mask[1, 5:] = False
                points[~mask] = -100
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(points, padding_mask=mask)['pred_regression']
                    self.assertTrue(torch.isfinite(prediction).all())
                    prediction.square().mean().backward()
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                        for p in model.parameters()))
                    optimizer.step()
                path = tmp + '/head.pth'
                torch.save(model.state_dict(), path)
                restored = head().cuda()
                restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
                model.eval()
                restored.eval()
                with torch.no_grad():
                    torch.testing.assert_close(
                        restored(points, padding_mask=mask)['pred_regression'],
                        model(points, padding_mask=mask)['pred_regression'], rtol=0, atol=0,
                    )

    def test_default_preserves_ssm_and_matches_legacy_other_weights_and_rng(self):
        original = head()
        native, legacy = copy.deepcopy(original), copy.deepcopy(original)
        torch.manual_seed(123)
        initialize_regression_adapter(native)
        native_rng = torch.get_rng_state()
        torch.manual_seed(123)
        initialize_regression_adapter(legacy, "legacy")
        self.assertTrue(torch.equal(native_rng, torch.get_rng_state()))
        before = dict(original.named_parameters())
        legacy_params = dict(legacy.named_parameters())
        changed = []
        for name, param in native.named_parameters():
            is_ssm = name.rsplit(".", 1)[-1] in {"dt_bias", "A_log", "D"}
            expected = before[name] if is_ssm else legacy_params[name]
            torch.testing.assert_close(param, expected, rtol=0, atol=0)
            if is_ssm:
                self.assertFalse(torch.equal(param, legacy_params[name]))
                changed.append(name)
        self.assertEqual(len(changed), 3)
        self.assertTrue(torch.equal(legacy.mamba_layers[0][1].dt_bias,
                                    torch.zeros_like(legacy.mamba_layers[0][1].dt_bias)))

    def test_invalid_mode_fails_before_mutation(self):
        model = head()
        before = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state()
        with self.assertRaisesRegex(ValueError, "adapter_ssm_initialization"):
            initialize_regression_adapter(model, "typo")
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_protected_initialization_is_retained(self):
        for flag in ("_preserve_mamba_init", "_preserve_transformer_init"):
            model = head()
            param = model.mamba_layers[0][1].in_proj.weight
            setattr(param, flag, True)
            before = param.detach().clone()
            initialize_regression_adapter(model)
            torch.testing.assert_close(param, before, rtol=0, atol=0)

    def test_unrelated_parameters_named_like_ssm_keep_legacy_behavior(self):
        model = torch.nn.Module()
        for name in ("dt_bias", "A_log", "D"):
            model.register_parameter(name, torch.nn.Parameter(torch.ones(3)))
        legacy = copy.deepcopy(model)
        torch.manual_seed(7)
        initialize_regression_adapter(model)
        torch.manual_seed(7)
        initialize_regression_adapter(legacy, "legacy")
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, legacy.state_dict()[name], rtol=0, atol=0)

    def test_trainer_default_and_legacy_checkpoint_reload_keep_trained_weights(self):
        for mode in (None, "legacy"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                trainer = training_tests.ValidationLossTrainingTest().trainer(tmp)
                trainer.params.update(embed_dim=64, num_layers_backbone=2,
                    num_output_classes=3, num_embedder_layers=0,
                    embed_method='pos_only', pe_method='nerf', dropout=0.,
                    max_lr=.001, min_lr=.0001, warmup_steps=0,
                    max_optimizer_steps=1, scheduler_first_cycle_steps=2,
                    log_file_name='initialization.log')
                if mode is not None:
                    trainer.params['adapter_ssm_initialization'] = mode
                constructed = []

                def capture_head(*args, **kwargs):
                    model = MambaTrackRegressionHead(*args, **kwargs)
                    constructed.append(copy.deepcopy(model.state_dict()))
                    return model

                trainer_module = __import__(trainer.__module__, fromlist=['DownstreamTrainer'])
                with patch.object(trainer_module, 'MambaTrackRegressionHead', capture_head), \
                        patch.object(trainer, '_train_by_optimizer_step'), \
                        patch.object(trainer, '_finish_validation'):
                    trainer.train(pretrain=False)
                self.assertEqual(trainer.adapter_ssm_initialization, mode or 'constructor')
                key = 'mamba_layers.0.1.dt_bias'
                if mode is None:
                    torch.testing.assert_close(trainer.down_model.state_dict()[key],
                                               constructed[0][key], rtol=0, atol=0)
                else:
                    self.assertTrue((trainer.down_model.state_dict()[key] == 0).all())
                with torch.no_grad():
                    trainer.down_model.mamba_layers[0][1].dt_bias.fill_(1.234)
                path = trainer._save_checkpoint('initialized.pth', 0, 0.)
                saved = torch.load(path, weights_only=False)
                self.assertEqual(saved['adapter_ssm_initialization'], mode or 'constructor')
                # Unmarked checkpoints came from the legacy initializer; loading
                # their learned parameters must not apply the new initialization.
                if mode == 'legacy':
                    del saved['adapter_ssm_initialization']
                    torch.save(saved, path)
                for inference in (False, True):
                    with torch.no_grad():
                        trainer.down_model.mamba_layers[0][1].dt_bias.zero_()
                    trainer.load_checkpoint(path, inference=inference, use_pretrained_backbone=False)
                    for name, value in trainer.down_model.state_dict().items():
                        torch.testing.assert_close(value, saved['model_state_dict'][name], rtol=0, atol=0)
                    self.assertEqual(trainer.adapter_ssm_initialization, mode or 'constructor')


if __name__ == "__main__":
    unittest.main()
