import unittest

import torch

from train.downstream.regression_features import (
    regression_backbone_features, regression_encoding_config, regression_precision,
    validate_regression_encoding,
)


class RecordingEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x, return_z=False):
        self.inputs.append(x.clone())
        return None, [x, x*2], None


class Scale(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x*self.factor


class ResidualEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba_layers = torch.nn.ModuleList([Scale(1e-9), Scale(.25)])

    def forward(self, x, return_z=False):
        states = []
        for layer in self.mamba_layers:
            x = x+layer(x)
            states.append(x)
        return (None, states, None) if return_z else x


class RegressionFeaturesTest(unittest.TestCase):
    def test_defaults_preserve_values_shapes_and_single_call(self):
        model = RecordingEncoder()
        points = torch.randn(3, 5, 3)
        output = regression_backbone_features(model, points, {})
        self.assertEqual(len(model.inputs), 1)
        torch.testing.assert_close(model.inputs[0], points, rtol=0, atol=0)
        torch.testing.assert_close(output, torch.stack([points, points*2]), rtol=0, atol=0)

    def test_fixed_shapes_split_large_batch_and_preserve_original_order(self):
        model = RecordingEncoder()
        points = torch.arange(5*3*3, dtype=torch.float32).reshape(5, 3, 3)
        output = regression_backbone_features(model, points,
            dict(backbone_sequence_padding=8, backbone_batch_padding=2))
        self.assertEqual([tuple(x.shape) for x in model.inputs], [(2, 8, 3)]*3)
        for x in model.inputs:
            self.assertTrue((x[:, 3:] == -100).all())
        self.assertTrue((model.inputs[-1][1] == -100).all())
        torch.testing.assert_close(output, torch.stack([points, points*2]), rtol=0, atol=0)

    def test_overlong_sequence_fails_before_encoder(self):
        model = RecordingEncoder()
        with self.assertRaisesRegex(ValueError, 'exceeds backbone_sequence_padding'):
            regression_backbone_features(model, torch.zeros(2, 9, 3), dict(backbone_sequence_padding=8))
        self.assertEqual(model.inputs, [])

    def test_checkpoint_requires_same_encoding_and_legacy_means_default(self):
        validate_regression_encoding({}, {})
        params = dict(backbone_sequence_padding=128, backbone_batch_padding=128,
                      regression_matmul_precision='highest')
        checkpoint = dict(backbone_encoding=regression_encoding_config(params))
        validate_regression_encoding(checkpoint, params)
        for saved, requested in [(checkpoint, {}), ({}, params),
                                 (checkpoint, dict(params, backbone_batch_padding=64))]:
            with self.assertRaisesRegex(ValueError, 'backbone encoding mismatch'):
                validate_regression_encoding(saved, requested)

    def test_invalid_config_rejected(self):
        for options in [dict(backbone_sequence_padding=-1), dict(backbone_batch_padding=True),
                        dict(backbone_batch_padding=1.5), dict(regression_matmul_precision='fast'),
                        dict(backbone_feature_kind='unknown')]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                regression_encoding_config(options)

    def test_residual_capture_keeps_updates_rounded_away_by_addition(self):
        model = ResidualEncoder()
        points = torch.ones(3, 5, 3)
        before = model(points)
        actual = regression_backbone_features(model, points,
            dict(backbone_feature_kind='residual_updates', backbone_sequence_padding=8,
                 backbone_batch_padding=2))
        torch.testing.assert_close(actual, torch.stack([points*1e-9, points*.25]), rtol=0, atol=0)
        states = regression_backbone_features(model, points, {})
        self.assertTrue((states[0]-points == 0).all())
        torch.testing.assert_close(model(points), before, rtol=0, atol=0)
        self.assertTrue(all(not layer._forward_hooks for layer in model.mamba_layers))

    def test_feature_policy_mismatch_and_prior_encoding_metadata(self):
        prior = dict(sequence_padding=0, batch_padding=0, matmul_precision='default')
        validate_regression_encoding(dict(backbone_encoding=prior), {})
        update_params = dict(backbone_feature_kind='residual_updates')
        for saved, requested in [(dict(backbone_encoding=prior), update_params),
                ({}, update_params),
                (dict(backbone_encoding=regression_encoding_config(update_params)), {})]:
            with self.assertRaisesRegex(ValueError, 'backbone encoding mismatch'):
                validate_regression_encoding(saved, requested)

    def test_legacy_policy_fills_only_missing_feature_metadata(self):
        params = dict(backbone_feature_kind='residual_updates',
                      legacy_backbone_feature_kind='residual_updates')
        validate_regression_encoding({}, params)
        prior = dict(sequence_padding=8, batch_padding=0, matmul_precision='default')
        checkpoint = dict(backbone_encoding=prior)
        validate_regression_encoding(checkpoint, dict(params, backbone_sequence_padding=8))
        self.assertNotIn('feature_kind', prior)
        with self.assertRaisesRegex(ValueError, 'backbone encoding mismatch'):
            validate_regression_encoding(checkpoint, params)
        explicit = dict(backbone_encoding=regression_encoding_config({}))
        with self.assertRaisesRegex(ValueError, 'backbone encoding mismatch'):
            validate_regression_encoding(explicit, params)

    def test_opt_in_precision_restores_global_settings_after_exception(self):
        matmul = torch.backends.cuda.matmul
        cudnn = torch.backends.cudnn

        def state():
            return (torch.get_float32_matmul_precision(), matmul.allow_tf32,
                    cudnn.allow_tf32, getattr(matmul, 'allow_bf16_reduced_precision_reduction', None))

        original = state()
        try:
            torch.set_float32_matmul_precision('medium')
            cudnn.allow_tf32 = True
            if original[3] is not None:
                matmul.allow_bf16_reduced_precision_reduction = True
            expected = state()
            options = dict(regression_matmul_precision='highest')
            with self.assertRaisesRegex(RuntimeError, 'evaluation failed'):
                with regression_precision(options):
                    self.assertEqual(state(), ('highest', False, False,
                                               False if original[3] is not None else None))
                    with regression_precision(options):
                        self.assertEqual(torch.get_float32_matmul_precision(), 'highest')
                    self.assertEqual(torch.get_float32_matmul_precision(), 'highest')
                    raise RuntimeError('evaluation failed')
            with regression_precision({}):
                self.assertEqual(state(), expected)
        finally:
            matmul.allow_tf32 = original[1]
            torch.set_float32_matmul_precision(original[0])
            cudnn.allow_tf32 = original[2]
            if original[3] is not None:
                matmul.allow_bf16_reduced_precision_reduction = original[3]

    def test_unsupported_residual_encoder_rejected_before_forward(self):
        model = RecordingEncoder()
        with self.assertRaisesRegex(ValueError, 'mamba_layers'):
            regression_backbone_features(model, torch.ones(1, 2, 3),
                                         dict(backbone_feature_kind='residual_updates'))
        self.assertEqual(model.inputs, [])

    def test_residual_hooks_removed_after_forward_failure(self):
        class BrokenEncoder(ResidualEncoder):
            def forward(self, x, return_z=False):
                self.mamba_layers[0](x)
                raise RuntimeError('probe failure')
        model = BrokenEncoder()
        with self.assertRaisesRegex(RuntimeError, 'probe failure'):
            regression_backbone_features(model, torch.ones(1, 2, 3),
                                         dict(backbone_feature_kind='residual_updates'))
        self.assertTrue(all(not layer._forward_hooks for layer in model.mamba_layers))


if __name__ == '__main__':
    unittest.main()
