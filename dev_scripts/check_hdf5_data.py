"""
Pre-flight check of the published HDF5 dataset through the exact pretraining
loader and config the Optuna search uses.

  1. manifest: format, status, splits, parts and event counts
  2. one part's fields: names, shapes, dtypes; offsets consistent
  3. raw coordinates on a sample: clusters/event, radius, eta, phi ranges vs the
     fixed normalization (eta [-2.5, 1.5], r [6, 23] cm), and layer assignment
  4. loader output: input/target shapes, kNNN slot fill per slot, no NaNs

Run from the repo root:  python3 dev_scripts/check_hdf5_data.py [n_events, default 300]
"""
import json, sys
from pathlib import Path
import numpy as np
import h5py
import torch

sys.path.insert(0, ".")
from fm4npp.datasets.cvt_hdf5 import CVTClusterEvents
from fm4npp.datasets.dataset_pretrain import TPCBatchDataset
from fm4npp.hilbert import assign_clas12_layer

ROOT = Path("/workspace/PP_collision/data/cvt_cluster_hdf5_20261103_002")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
ETA_LIM, R_LIM = (-2.5, 1.5), (6.0, 23.0)

# 1. manifest
m = json.loads((ROOT / "manifest.json").read_text())
print("=== 1. manifest ===")
print({k: v for k, v in m.items() if not isinstance(v, (list, dict))})
for split, parts in m["splits"].items():
    on_disk = sum((ROOT / p["path"]).exists() for p in parts)
    print(f"  {split}: {len(parts)} parts ({on_disk} on disk), {sum(p['events'] for p in parts):,} events")

# 2. fields in one part
print("\n=== 2. fields in", m["splits"]["train"][0]["path"], "===")
with h5py.File(ROOT / m["splits"]["train"][0]["path"], "r") as f:
    print("  attrs:", dict(f.attrs))
    f.visititems(lambda n, o: print(f"  {n:45s} {str(o.shape):>16s} {o.dtype}")
                 if isinstance(o, h5py.Dataset) else None)
    off = f["clusters/offsets"][:]
    print(f"  clusters/offsets: {len(off) - 1} events, monotonic={bool(np.all(np.diff(off) >= 0))}, "
          f"last={off[-1]} vs xyz rows={f['clusters/xyz'].shape[0]}")

# 3. raw coordinates on a sample
print(f"\n=== 3. raw coordinates, {N} train events ===")
ev = CVTClusterEvents(ROOT, "train")
idx = np.linspace(0, len(ev) - 1, N).astype(int)
xyz = [ev.read(int(i), ["clusters.xyz"])["clusters.xyz"] for i in idx]
n = np.array([len(a) for a in xyz]); a = np.concatenate(xyz).astype(np.float64)
rho = np.hypot(a[:, 0], a[:, 1]); eta = np.arcsinh(a[:, 2] / rho); phi = np.arctan2(a[:, 1], a[:, 0])
print(f"  clusters/event: mean {n.mean():.1f}, min {n.min()}, median {np.median(n):.0f}, max {n.max()}")
for name, v, lo, hi in (("r (cm)", rho, *R_LIM), ("eta", eta, *ETA_LIM), ("phi", phi, -np.pi, np.pi)):
    out = np.mean((v < lo) | (v > hi)) * 100
    print(f"  {name:7s} min {v.min():8.3f}  max {v.max():8.3f}   outside [{lo}, {hi}]: {out:.3f}%")
print(f"  NaN/inf in xyz: {int((~np.isfinite(a)).sum())}")
try:
    bands = assign_clas12_layer(torch.tensor((rho - R_LIM[0]) / (R_LIM[1] - R_LIM[0])))
    print("  clusters per layer band:", np.bincount(bands.numpy(), minlength=12).tolist())
except Exception as e:
    print(f"  LAYER ASSIGNMENT FAILED: {type(e).__name__}: {e}")
ev.close()

# 4. loader output with the Optuna config
print(f"\n=== 4. loader (positions only, sin/cos target, kNNN=5, unit_v2) ===")
for split, train in (("train", True), ("validation", False)):
    ds = TPCBatchDataset(str(ROOT), reader_type="hdf5", split=split, train=train, limit_data=False,
                         num_pred_points=5, voxelize=False, space_filling_order=True,
                         hilbert_angular_grid="unit_v2")
    print(f"  {split}: {len(ds):,} events")
    fill = np.zeros(5); total = 0; bad = 0
    for i in np.linspace(0, len(ds) - 1, min(N, 100)).astype(int):
        g, t, k = ds[int(i)]
        bad += int(~torch.isfinite(g).all()) + int(~torch.isfinite(k[k != -100]).all())
        k = k.reshape(k.shape[0], 5, 3)
        fill += (k[..., 0] != -100).sum(0).numpy(); total += k.shape[0]
    print(f"    input {tuple(g.shape)}, target {tuple(k.shape)}; inputs in [{g.min():.3f}, {g.max():.3f}]")
    print(f"    % of clusters with a target, slot 1..5: {np.round(100 * fill / total, 1).tolist()}; non-finite: {bad}")
print("\nDone.")