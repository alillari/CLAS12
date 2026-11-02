"""Check physical band separation and strict causal target identity."""

import unittest

import torch

from fm4npp.hilbert import CLAS12_LAYER_RADII, assign_clas12_layer
from dev_scripts.mamba3_cluster_pretraining.targets import outward_band_targets


class OutwardTargetTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.points = torch.tensor([[.3, .4, CLAS12_LAYER_RADII[0]],
                                    [.3, .4, CLAS12_LAYER_RADII[0]],
                                    [.3, .4, CLAS12_LAYER_RADII[1]],
                                    [.9, .9, CLAS12_LAYER_RADII[2]],
                                    [.3, .4, CLAS12_LAYER_RADII[3]],
                                    [.3, .4, CLAS12_LAYER_RADII[5]]])

    def test_same_band_is_excluded_even_when_it_is_identical(self):
        targets, indices = outward_band_targets(self.points, 5)
        self.assertEqual(indices[0, 0].item(), 2)
        self.assertNotIn(indices[0, 0].item(), [0, 1])
        torch.testing.assert_close(targets[0, :3], self.points[2])

    def test_each_slot_uses_its_own_minimum_band_offset(self):
        _, indices = outward_band_targets(self.points, 5)
        bands = assign_clas12_layer(self.points[:, 2])
        for query in range(len(self.points)):
            for slot in range(5):
                candidates = [j for j in range(len(self.points)) if bands[j] >= bands[query] + slot + 1]
                if not candidates:
                    self.assertEqual(indices[query, slot].item(), -1)
                    continue
                expected = min(candidates, key=lambda j: float(torch.linalg.vector_norm(self.points[query] - self.points[j])))
                self.assertEqual(indices[query, slot].item(), expected)
                self.assertGreater(expected, query)

    def test_outermost_band_is_masked_and_repeated_slots_are_permitted(self):
        targets, indices = outward_band_targets(self.points, 5)
        self.assertTrue((indices[-1] == -1).all())
        self.assertTrue((targets[-1] == -100).all())
        self.assertEqual(indices[0, 1].item(), indices[0, 2].item())

    def test_unsorted_bands_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "serialized"):
            outward_band_targets(self.points.flip(0), 5)


if __name__ == "__main__":
    unittest.main()
