"""Tests of the default within-band Hilbert grid and historical compatibility."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from mmap_ninja import RaggedMmap

from fm4npp.datasets.dataset_pretrain import get_data_loader, get_val_loader
from fm4npp.datasets import dataset as downstream_data
from fm4npp.hilbert import assign_clas12_layer, clas12_band_hilbert_order, clas12_hilbert_cells, encode, validate_checkpoint_hilbert_grid
from fm4npp.utils import YParams

RECIPE = Path(__file__).resolve().parents[3] / "scripts/configs/mamba_v602_sweep.yaml"


class HilbertGridTest(unittest.TestCase):
    def test_unmarked_checkpoints_require_explicit_legacy_policy(self):
        legacy = SimpleNamespace(hilbert_angular_grid="legacy_v1")
        validate_checkpoint_hilbert_grid({}, legacy)
        validate_checkpoint_hilbert_grid({"hilbert_angular_grid": "unit_v2"}, SimpleNamespace())
        with self.assertRaisesRegex(ValueError, "Checkpoint Hilbert grid"):
            validate_checkpoint_hilbert_grid({}, SimpleNamespace())
        with self.assertRaisesRegex(ValueError, "Checkpoint Hilbert grid"):
            validate_checkpoint_hilbert_grid({"hilbert_angular_grid": "unit_v2"}, legacy)
        validate_checkpoint_hilbert_grid({}, SimpleNamespace(serialization="radius"))

    def test_cells_and_legacy_permutation(self):
        phi = torch.tensor([0., .20, .80, .999, 1.])
        eta = torch.tensor([0., .40, .70, 1., .50])
        radii = torch.tensor([.9, .1, .1, .1, .9])
        historical_cells = (torch.stack((phi, eta), -1) * 1e4).long().clamp_max(1023)
        historical_keys = assign_clas12_layer(radii) * 2**48 + encode(historical_cells, 2, 10)
        torch.testing.assert_close(clas12_hilbert_cells(phi, eta, angular_grid="legacy_v1"), historical_cells)
        torch.testing.assert_close(clas12_band_hilbert_order(phi, eta, radii, angular_grid="legacy_v1"), historical_keys.argsort())

        expected = torch.tensor([[0, 0], [204, 409], [819, 716], [1022, 1023], [1023, 512]])
        torch.testing.assert_close(clas12_hilbert_cells(phi, eta), expected)
        actual = clas12_band_hilbert_order(phi, eta, radii, angular_grid="unit_v2")
        keys = assign_clas12_layer(radii) * 2**48 + encode(expected, 2, 10)
        torch.testing.assert_close(actual, keys.argsort())
        edges = torch.tensor([-.1, 0., .25, .999, 1., 1.1])
        torch.testing.assert_close(clas12_hilbert_cells(edges, edges, 2, angular_grid="unit_v2")[:, 0],
                                   torch.tensor([0, 0, 1, 3, 3, 3]))
        with self.assertRaisesRegex(ValueError, "Unknown Hilbert"):
            clas12_hilbert_cells(phi, eta, angular_grid="typo")
        with self.assertRaisesRegex(ValueError, "finite"):
            clas12_hilbert_cells(torch.tensor([float("nan")]), torch.tensor([.5]))
        with self.assertRaisesRegex(ValueError, "custom scaler"):
            clas12_hilbert_cells(phi, eta, scaler=100)
        singleton = clas12_band_hilbert_order(phi[:1], eta[:1], radii[:1])
        self.assertEqual(singleton.tolist(), [0])

    def test_profile_changes_only_grid_and_run_identity(self):
        old = YParams(RECIPE, "scale_w64_d12_v602_legacy_v1").params.copy()
        new = YParams(RECIPE, "scale_w64_d12_v602_unit_v2").params.copy()
        self.assertEqual(new.pop("hilbert_angular_grid"), "unit_v2")
        self.assertEqual(old.pop("hilbert_angular_grid"), "legacy_v1")
        self.assertNotEqual(old.pop("save_version"), new.pop("save_version"))
        self.assertEqual(old, new)

    def test_mmap_train_and_validation_reorder_rows_and_targets_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Several angularly separated clusters within each band; each cluster
            # has its own label so the row permutation is visible.
            radii = np.repeat([6.5, 9.3, 12.2], 8)
            phi = np.tile([1.2, .5, 2., -.7, -1.3, 2.6, -.1, -1.8], 3)
            xyz = np.column_stack([radii * np.cos(phi), radii * np.sin(phi), np.arange(24) / 5])
            for split, label_base in (("pretrain", 0), ("test", 100)):
                for field, values in (("features", xyz.astype("f4")), ("seg_target", np.arange(24) + label_base),
                                      ("reg_target", np.zeros((24, 3), dtype="f4"))):
                    RaggedMmap.from_lists(root / f"{field}_{split}", [values])

            samples = []
            for profile in ("scale_w64_d12_v602_legacy_v1", "scale_w64_d12_v602"):
                params = YParams(RECIPE, profile)
                params.data_root = str(root)
                train, _, val, _ = get_data_loader(params, distributed=False)
                samples.append((train.dataset[0], val.dataset[0]))
                for a, b in zip(val.dataset[0], get_val_loader(params, False).dataset[0]):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)

                params.data_root_test = str(root)
                params.return_dict = True
                params.return_reg_test = False
                params.adapter_sample_mode = 'track_legacy'
                params.num_data_workers = 0
                down_train, _, down_val, _ = downstream_data.get_data_loader(params, False)
                for expected, actual in zip((train, val), (down_train, down_val)):
                    reference = expected.dataset[0]
                    sample = actual.dataset[0]
                    torch.testing.assert_close(sample['points'], reference[0], rtol=0, atol=0)
                    torch.testing.assert_close(sample['target'], reference[1], rtol=0, atol=0)
                standalone = downstream_data.get_val_loader(params, False).dataset[0]
                for key in ('points', 'target'):
                    torch.testing.assert_close(standalone[key], down_val.dataset[0][key], rtol=0, atol=0)

            for i, (control, experiment) in enumerate(zip(*samples)):
                x, labels, targets = control
                new_x, new_labels, new_targets = experiment
                self.assertFalse(torch.equal(labels, new_labels))
                self.assertEqual(int(labels.min()), 100 * i)
                torch.testing.assert_close(assign_clas12_layer(x[:, 2]), assign_clas12_layer(new_x[:, 2]))
                # Restore both outputs to the original cluster identity, including
                # each token's precomputed outward-neighbor target.
                old_idx, new_idx = labels.argsort(), new_labels.argsort()
                for a, b in zip(control, experiment):
                    torch.testing.assert_close(a[old_idx], b[new_idx], atol=0, rtol=0)
