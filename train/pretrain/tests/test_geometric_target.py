"""knn_rule='geometric' (Test 3): kNNN = 1 target from the CVT geometry."""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fm4npp import geometric_target as gt
from fm4npp.datasets.dataset_pretrain import TPCBatchDataset
from fm4npp.hilbert import CLAS12_BAND_BOUNDARIES, assign_clas12_layer
from train.pretrain.tests.test_cvt_hdf5 import event as fixture_event, publish_fixture


def event(n=150, seed=0):
    g = np.random.default_rng(seed)
    edges = np.array([0.0, *CLAS12_BAND_BOUNDARIES, 1.0])
    band = g.integers(0, 12, n)
    r = (edges[band] + (edges[band + 1] - edges[band]) * g.uniform(0.3, 0.7, n)) * 17 + 6
    phi = g.uniform(-math.pi, math.pi, n)
    z = g.uniform(-15, 15, n)
    return np.column_stack((r * np.cos(phi), r * np.sin(phi), z))


def reference(xyz):
    """Plain-loop version of the rule, for checking the vectorized one."""
    sz, srphi = gt.layer_sigmas()
    r = np.hypot(xyz[:, 0], xyz[:, 1]); lay = assign_clas12_layer(torch.tensor((r - 6) / 17)).numpy()
    phi = np.arctan2(xyz[:, 1], xyz[:, 0]); th = np.arctan2(r, xyz[:, 2] - gt.TARGET_CENTER)
    out = []
    for i in range(len(xyz)):
        best = (99, math.inf, -1); best_any = (math.inf, -1)
        for j in range(len(xyz)):
            if lay[j] <= lay[i]:
                continue
            dphi = math.atan2(math.sin(phi[j] - phi[i]), math.cos(phi[j] - phi[i]))
            sb = abs(r[j] - r[i]) / (2 * gt.R_MIN_CURV) / math.sqrt(3)
            sv = gt.TARGET_LEN / math.sqrt(12) * math.sin(th[i]) ** 2 * abs(1 / r[i] - 1 / r[j])
            s_i = srphi[lay[i]] / r[i]; s_j = srphi[lay[j]] / r[j]
            t_i = sz[lay[i]] * math.sin(th[i]) ** 2 / r[i]; t_j = sz[lay[j]] * math.sin(th[j]) ** 2 / r[j]
            c = dphi ** 2 / (s_i ** 2 + s_j ** 2 + sb ** 2) + (th[j] - th[i]) ** 2 / (t_i ** 2 + t_j ** 2 + sv ** 2)
            if c < best_any[0]:
                best_any = (c, j)
            if c < gt.CHI2_CUT and (lay[j], c) < best[:2]:
                best = (lay[j], c, j)
        out.append(best[2] if best[2] >= 0 else best_any[1])
    return np.array(out)


class GeometricTargetTest(unittest.TestCase):
    def test_matches_reference(self):
        for seed in range(3):
            xyz = event(seed=seed)
            self.assertTrue(np.array_equal(gt.geometric_target_indices(torch.from_numpy(xyz)).numpy(),
                                           reference(xyz)), seed)

    def test_target_is_further_out_or_none(self):
        xyz = torch.from_numpy(event(seed=4))
        lay = assign_clas12_layer((torch.hypot(xyz[:, 0], xyz[:, 1]) - 6) / 17)
        t = gt.geometric_target_indices(xyz)
        last = lay == lay.max()
        self.assertTrue(bool((t[last] == -1).all()))
        self.assertTrue(bool((lay[t[~last]] > lay[~last]).all()))

    def test_sigmas_from_geometry(self):
        sz, srphi = gt.layer_sigmas()
        self.assertAlmostEqual(sz[0] * 10, 333.37 / math.sqrt(12), places=1)       # SVT strip length
        self.assertAlmostEqual(srphi[7] * 10, 0.487 / math.sqrt(12), places=3)     # BMT Z layer 2 pitch
        self.assertLess(sz[6], 0.02)                                                # BMT C measures z

    def test_seam_first_compatible_layer_and_global_fallback(self):
        def positions(radii, angles):
            r = torch.tensor(radii, dtype=torch.float64)
            phi = torch.tensor(angles, dtype=torch.float64)
            return torch.stack((r * phi.cos(), r * phi.sin(), r * 0 + gt.TARGET_CENTER), -1)

        xyz = positions([7, 9, 11], [math.pi - .01, -math.pi + .01, math.pi - .01])
        layer = torch.tensor([0, 1, 2])
        self.assertEqual(gt.geometric_target_indices(xyz, layer).tolist(), [1, 2, -1])
        distant = positions([7, 9, 11], [0, math.pi, math.pi])
        self.assertEqual(gt.geometric_target_indices(distant, layer)[0].item(), 2)
        self.assertEqual(gt.geometric_target_indices(xyz[:0]).tolist(), [])

    def test_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            xyz = event(80, seed=5).astype(np.float32)
            ev = fixture_event(len(xyz), 0)
            ev["clusters/xyz"] = xyz
            publish_fixture(Path(tmp), {"train": [[ev]], "validation": [[ev]]})
            mk = lambda **kw: TPCBatchDataset(Path(tmp), reader_type="hdf5", split="train", train=False, **kw)
            with self.assertRaises(ValueError):
                mk(num_pred_points=1, knn_rule="nearest")
            with self.assertRaises(ValueError):
                mk(num_pred_points=2, knn_rule="geometric")
            ds = mk(num_pred_points=1, coord_system="theta", knn_rule="geometric", hilbert_angular_grid="unit_v2")
            x, _, kn = ds[0]
            # map each returned row back to its input cluster, then check every target
            phys = ds.apply_unnorm(x.double())
            c0, phi, r = phys.unbind(-1)
            back = torch.stack([r * torch.cos(phi), r * torch.sin(phi), r / torch.tan(c0)], -1)
            orig = torch.cdist(back, torch.from_numpy(xyz).double()).argmin(1)       # row -> input index
            self.assertEqual(len(set(orig.tolist())), len(xyz))
            row_of = torch.empty_like(orig); row_of[orig] = torch.arange(len(orig))
            t = gt.geometric_target_indices(torch.from_numpy(xyz).double())          # on the input clusters
            for i in range(len(x)):
                j = t[orig[i]]
                want = torch.full((3,), -100.0) if j < 0 else x[row_of[j]]
                self.assertTrue(torch.equal(kn[i], want.to(kn.dtype)), i)
            plain = mk(num_pred_points=1, coord_system="theta", hilbert_angular_grid="unit_v2")[0][2]
            self.assertFalse(torch.equal(kn, plain))


if __name__ == "__main__":
    unittest.main()
