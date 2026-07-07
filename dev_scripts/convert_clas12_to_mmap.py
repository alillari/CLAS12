"""
CLAS12 CSV -> RaggedMmap converter (small training run).

Reads the two raw CLAS12 CSVs (BST crosses + BMT combined points), keeps only
(event, x0, y0, z0, trkID) from each, merges them per-event into one combined
point cloud, splits events 90/10 into pretrain/test, and writes the six
RaggedMmap directories the dataset pipeline expects:

    <out_dir>/features_pretrain      per-event (n_hits, 3) = [x0, y0, z0]
    <out_dir>/seg_target_pretrain    per-event (n_hits,)   = trkID
    <out_dir>/reg_target_pretrain    per-event (n_hits, 7) = zeros (stub)
    <out_dir>/features_test
    <out_dir>/seg_target_test
    <out_dir>/reg_target_test

Notes:
  - Energy is intentionally absent (CLAS12 has none); features are 3 columns.
  - trkID is carried through as seg_target but never read by the pretraining
    loss. trkID = -1 (noise/unmatched) hits are KEPT — they are real detector
    hits and the model should see realistic noisy events.
  - This goes straight CSV -> mmap (no intermediate .npz). The .npz ideal
    format is for the future large-scale (millions of events) converter; for
    this small existing-data run it adds nothing.

Usage (inside the container):
    python3 convert_clas12_to_mmap.py
Edit the paths in the CONFIG block below to match your files.
"""

import os
import numpy as np
import pandas as pd
from mmap_ninja import RaggedMmap

# ============================================================
# CONFIG — edit these paths
# ============================================================

CSV_FILE_BST = "/workspace/PP_collision/data/THREE_HADRONS_bst_crosses.csv"
CSV_FILE_BMT = "/workspace/PP_collision/data/THREE_HADRONS_bmt_combined_points.csv"

OUT_DIR = "/workspace/PP_collision/data/mmap"

TEST_FRACTION = 0.10   # 90/10 pretrain/test split (pretraining train/val, NOT downstream)
RANDOM_SEED = 42

N_REG_COLS = 7         # reg_target stub width (matches particle_reg_cols)

# ============================================================
# LOAD + KEEP ONLY NEEDED COLUMNS
# ============================================================

KEEP = ["event", "x0", "y0", "z0", "trkID"]

print("Loading CSVs...")
df_bst = pd.read_csv(CSV_FILE_BST)[KEEP]
df_bmt = pd.read_csv(CSV_FILE_BMT)[KEEP]
print(f"  BST hits: {len(df_bst)}")
print(f"  BMT hits: {len(df_bmt)}")

# Merge both detectors into one combined set of hits.
df = pd.concat([df_bst, df_bmt], ignore_index=True)
print(f"  Combined hits: {len(df)}")

# ============================================================
# GROUP BY EVENT -> per-event arrays
# ============================================================

# Sort by event so groups are contiguous/deterministic; within an event the
# hit order here does NOT matter (the dataset re-sorts by radius then applies
# the Hilbert band ordering at load time).
all_events = sorted(df["event"].unique())
print(f"  Total events: {len(all_events)}")

features_by_event = {}
seg_by_event = {}
for ev, group in df.groupby("event"):
    xyz = group[["x0", "y0", "z0"]].to_numpy(dtype=np.float32)   # (n_hits, 3)
    trk = group["trkID"].to_numpy(dtype=np.float32)              # (n_hits,)
    features_by_event[ev] = xyz
    seg_by_event[ev] = trk

# ============================================================
# 90/10 SPLIT (by event, so an event is wholly in one split)
# ============================================================

rng = np.random.RandomState(RANDOM_SEED)
events_shuffled = list(all_events)
rng.shuffle(events_shuffled)

n_test = max(1, int(round(len(events_shuffled) * TEST_FRACTION)))
test_events = set(events_shuffled[:n_test])
pretrain_events = events_shuffled[n_test:]

print(f"  Pretrain events: {len(pretrain_events)}")
print(f"  Test events:     {len(test_events)}")

# ============================================================
# WRITE RAGGEDMMAP DIRECTORIES
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)

def gen_features(event_list):
    for ev in event_list:
        yield features_by_event[ev]

def gen_seg(event_list):
    for ev in event_list:
        yield seg_by_event[ev]

def gen_reg(event_list):
    # reg_target is a zero stub, per-event (n_hits, 7), never read by the loss.
    for ev in event_list:
        n_hits = features_by_event[ev].shape[0]
        yield np.zeros((n_hits, N_REG_COLS), dtype=np.float32)

def write_split(split_name, event_list):
    print(f"\nWriting split '{split_name}' ({len(event_list)} events)...")
    RaggedMmap.from_generator(
        out_dir=os.path.join(OUT_DIR, f"features_{split_name}"),
        sample_generator=gen_features(event_list),
        batch_size=1024,
        verbose=True,
    )
    RaggedMmap.from_generator(
        out_dir=os.path.join(OUT_DIR, f"seg_target_{split_name}"),
        sample_generator=gen_seg(event_list),
        batch_size=1024,
        verbose=True,
    )
    RaggedMmap.from_generator(
        out_dir=os.path.join(OUT_DIR, f"reg_target_{split_name}"),
        sample_generator=gen_reg(event_list),
        batch_size=1024,
        verbose=True,
    )

write_split("pretrain", pretrain_events)
write_split("test", list(test_events))

print("\nDone. Wrote 6 RaggedMmap directories to:", OUT_DIR)
print("Sanity check: re-loading features_pretrain[0]...")
check = RaggedMmap(os.path.join(OUT_DIR, "features_pretrain"))
print("  first event shape:", np.array(check[0]).shape, "(should be (n_hits, 3))")
print("  first event sample row:", np.array(check[0])[0])
