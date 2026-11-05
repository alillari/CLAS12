import unittest
from types import SimpleNamespace

import numpy as np
import torch

from train.downstream.loss import simple_point_loss
from train.downstream.noise_metrics import (
    at_threshold, calibrate_thresholds, inverse_frequency_weights,
    retention_threshold, training_class_counts,
)


class NoiseBalanceTest(unittest.TestCase):
    def test_equal_class_loss_and_gradient_mass_with_padding(self):
        labels = torch.tensor([[0, 1, 1, 1, 1, -100]])
        logits = torch.zeros(1, 6, 2, requires_grad=True)
        weights = torch.tensor(inverse_frequency_weights([1, 4]), dtype=torch.float32)
        loss = simple_point_loss({'pred_logits': logits}, {'labels': labels}, labels >= 0,
                                 class_weights=weights)['loss']
        loss.backward()
        self.assertAlmostEqual(loss.item(), np.log(2), places=6)
        self.assertAlmostEqual(logits.grad[0, 0].abs().sum().item(),
                               logits.grad[0, 1:5].abs().sum().item(), places=6)
        self.assertEqual(logits.grad[0, 5].abs().sum().item(), 0.)

    def test_counts_use_only_accepted_events(self):
        dataset = SimpleNamespace(chunk_training=False, idxlist=[2, 0],
                                  memmap_noise_target=[np.array([0, 1]), np.array([9]),
                                                       np.array([1, 1, 0])])
        np.testing.assert_array_equal(training_class_counts(dataset), [2, 3])
        for counts in ([0, 1], [1, 0], [1, 2, 3]):
            with self.assertRaises(ValueError):
                inverse_frequency_weights(counts)

    def test_retention_threshold_preserves_ties(self):
        labels = np.array([0, 0, 0, 0, 1, 1])
        scores = np.array([.1, .2, .2, .9, .8, .95])
        threshold = retention_threshold(labels, scores, .5)
        self.assertEqual(threshold, .2)
        metrics = at_threshold(labels, scores, threshold)
        self.assertEqual(metrics['signal_retention'], .75)
        self.assertEqual(metrics['noise_recall'], 1.)

    def test_threshold_selected_on_validation_is_fixed_on_test(self):
        labels = np.array([0, 0, 1, 1])
        coat = np.array([0, 0, 1, 1])
        points = calibrate_thresholds(labels, [.1, .3, .8, .9], coat)
        threshold = points['retain_99pct']['threshold']
        test = at_threshold(labels, [.2, .4, .8, .9], threshold)
        self.assertEqual(threshold, .3)
        self.assertEqual(test['signal_retention'], .5)


if __name__ == '__main__':
    unittest.main()
