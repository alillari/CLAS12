"""
Profile where TPCBatchDataset.__getitem__ spends time on the v4 data.
Times a batch of events and breaks down the cost, so we know whether the
Hilbert ordering (or something else) is the actual bottleneck before
building a caching optimization.

Run inside container:  python3 dev_scripts/profile_getitem.py
"""
import sys, time
sys.path.insert(0, "/workspace/PP_collision")
import numpy as np
import torch
from fm4npp.datasets.dataset_pretrain import TPCBatchDataset

ds = TPCBatchDataset(
    data_root='data/mmap_v4', version='dataset_name', split='pretrain',
    reader_type='ragged_npy', num_pred_points=1, group_size=1, normalize=True,
    nleave=1e6, chunk_training=False, train=True, order='clas12_band_hilbert_order',
)
print(f"dataset len: {len(ds)}")

N = 200  # number of events to time

# --- time full __getitem__ ---
t0 = time.time()
for i in range(N):
    _ = ds[i]
t_full = time.time() - t0
print(f"\nFull __getitem__: {t_full:.3f}s for {N} events = {1000*t_full/N:.2f} ms/event")
print(f"  -> at batch_size 128, ~{128*t_full/N:.2f}s per batch just for data prep")
print(f"  -> 8 workers in parallel would cut that ~8x if CPU-bound and enough cores")

# --- now time just the raw read (bypass ordering) to isolate ordering cost ---
mm = ds.memmap_feature
t0 = time.time()
for i in range(N):
    _ = mm[i]
t_read = time.time() - t0
print(f"\nRaw event read only: {t_read:.3f}s = {1000*t_read/N:.2f} ms/event")

print(f"\n=> Ordering + norm + kNN overhead: {1000*(t_full - t_read)/N:.2f} ms/event "
      f"({100*(t_full-t_read)/t_full:.0f}% of total)")
print("\nIf that overhead is the bulk of the time, precomputing the ordering to disk")
print("once (instead of every epoch) is the fix.")