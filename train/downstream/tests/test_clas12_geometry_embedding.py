import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from fm4npp.datasets.dataset import MyCollator
from fm4npp.models.embed import (
    CLAS12GeometryContextEmbedder,
    EmbedderPosPlusAux,
    EmbedderPosOnly,
    clas12_xyz_to_normalized_etaphr,
)
from fm4npp.utils import cartesian_to_polar_batched


class CLAS12GeometryEmbeddingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.embedding = CLAS12GeometryContextEmbedder(
            embed_dim=16, pitch_mean_cm=0.04, pitch_std_cm=0.01
        )
        self.center = torch.randn(2, 3, 16)
        self.context = torch.tensor(
            [
                [[1, 1, 0, 0, 0], [1, 6, 0, 0, 0], [2, 1, 0, 0, 0]],
                [[2, 6, 0, 0, 0], [1, 3, 0, 0, 0], [0, 0, 0, 0, 0]],
            ],
            dtype=torch.long,
        )
        self.geometry = torch.tensor(
            [
                [
                    [8.0, 1.0, -4.0, 9.0, 2.0, -3.0, 0, 0, 0, 1.0, 0.033],
                    [11.0, 3.0, -2.0, 12.0, 4.0, -1.0, 0, 0, 0, 1.0, 0.086],
                    [6.5, -1.0, 5.0, 7.5, -2.0, 6.0, 0, 0, 0, 1.0, 0.017],
                ],
                [
                    [7.0, 2.0, 1.0, 8.0, 3.0, 2.0, 0, 0, 0, 1.0, 0.018],
                    [10.0, -1.0, 3.0, 11.0, -2.0, 4.0, 0, 0, 0, 1.0, 0.040],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.000],
                ],
            ],
            dtype=torch.float32,
        )
        self.mask = torch.tensor([[True, True, True], [True, True, False]])

    def test_endpoint_swap_is_exactly_invariant(self):
        original, _ = self.embedding(self.center, self.context, self.geometry, self.mask)
        swapped = self.geometry.clone()
        swapped[..., 0:3], swapped[..., 3:6] = (
            self.geometry[..., 3:6].clone(), self.geometry[..., 0:3].clone()
        )
        exchanged, _ = self.embedding(self.center, self.context, swapped, self.mask)
        torch.testing.assert_close(original, exchanged, rtol=0.0, atol=0.0)

    def test_endpoint_phi_is_shared_nonlinear_map_and_receives_gradients(self):
        self.assertIsInstance(self.embedding.endpoint_phi, torch.nn.Sequential)
        self.assertIsInstance(self.embedding.endpoint_phi[1], torch.nn.SiLU)
        token, _ = self.embedding(self.center, self.context, self.geometry, self.mask)
        token[self.mask].square().mean().backward()
        for layer_index in (0, 2):
            gradient = self.embedding.endpoint_phi[layer_index].weight.grad
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_detector_layer_pairs_are_distinct_and_padding_is_zero(self):
        mapped = self.embedding.detector_layer_index(self.context, self.mask)
        self.assertEqual(mapped[0].tolist(), [1, 6, 7])
        self.assertEqual(mapped[1].tolist(), [12, 3, 0])
        token, _ = self.embedding(self.center, self.context, self.geometry, self.mask)
        torch.testing.assert_close(token[1, 2], torch.zeros_like(token[1, 2]))

    def test_coordinate_conversion_and_pitch_are_finite(self):
        endpoint_coordinates = clas12_xyz_to_normalized_etaphr(self.geometry[..., 0:3])
        legacy = cartesian_to_polar_batched(self.geometry[..., 0:3])
        legacy[..., 0] = (legacy[..., 0] + 2.5) / 4.0
        legacy[..., 1] = (legacy[..., 1] + torch.pi) / (2.0 * torch.pi)
        legacy[..., 2] = (legacy[..., 2] - 6.0) / 17.0
        torch.testing.assert_close(endpoint_coordinates[self.mask], legacy[self.mask])
        self.assertTrue(torch.isfinite(endpoint_coordinates).all())
        pitch = (self.geometry[..., 10] - 0.04) / 0.01
        self.assertTrue(torch.isfinite(pitch[self.mask]).all())
        self.assertGreater(float(pitch[self.mask].max()), 0.0)
        self.assertLess(float(pitch[self.mask].min()), 0.0)

    def test_output_shape_and_branch_norms(self):
        token, norms = self.embedding(self.center, self.context, self.geometry, self.mask)
        self.assertEqual(tuple(token.shape), (2, 3, 16))
        self.assertEqual(set(norms), {"center", "endpoint", "detector_layer", "pitch", "sum"})
        self.assertTrue(all(torch.isfinite(value) for value in norms.values()))

    def test_legacy_center_only_embedder_contract_is_unchanged(self):
        embedder = EmbedderPosOnly(pe_method="nerf", embed_dim=16)
        points = torch.rand(2, 3, 3)
        output, positional = embedder(points)
        torch.testing.assert_close(output, positional)
        self.assertEqual(tuple(output.shape), (2, 3, 16))

    def test_pos_plus_aux_embedding_has_a_strict_token_width_contract(self):
        embedder = EmbedderPosPlusAux(
            pe_method="nerf", embed_dim=16, pos_dim=3, aux_dim=4
        )
        tokens = torch.rand(2, 3, 7)
        output, positional = embedder(tokens)
        self.assertEqual(tuple(output.shape), (2, 3, 16))
        self.assertEqual(tuple(positional.shape), (2, 3, 16))
        with self.assertRaisesRegex(ValueError, "expected 7 token columns"):
            embedder(tokens[..., :-1])

    def test_collator_keeps_typed_context_and_zero_pads_it(self):
        sample = {
            "points": torch.ones(2, 3),
            "knearest_points": torch.ones(2, 3),
            "target": torch.ones(2, dtype=torch.long),
            "reg_target": torch.ones(2, 7),
            "pid_target": torch.ones(2, dtype=torch.long),
            "noise_target": torch.ones(2, dtype=torch.long),
            "token_context": torch.ones(2, 5, dtype=torch.long),
            "geometry_context": torch.ones(2, 11),
        }
        second = {key: value[:1].clone() for key, value in sample.items()}
        batch = MyCollator()([sample, second])
        self.assertEqual(batch["token_context"].dtype, torch.long)
        self.assertEqual(tuple(batch["geometry_context"].shape), (2, 2, 11))
        self.assertEqual(batch["token_context"][1, 1].sum().item(), 0)
        self.assertEqual(batch["geometry_context"][1, 1].sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
