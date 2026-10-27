import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from fm4npp.datasets.dataset import MyCollator
from fm4npp.models.embed import (
    EmbedderPosPlusAux,
    EmbedderPosOnly,
)


class CLAS12InputContractTest(unittest.TestCase):






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
