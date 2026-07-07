"""
CLAS12 single-track regrouping converter.

Same raw CSVs as the main converter, but instead of one sequence per collision
EVENT (3 mixed particles), this emits one sequence per TRACK: it groups by
(event, trkID) and drops noise hits (trkID == -1). So a 3-track collision event
becomes 3 separate single-track sequences.

This isolates "learning a single clean trajectory" from "disentangling
overlapping tracks" — a probe complementary to the straight-track baseline.

Output: the usual 6 RaggedMmap dirs, in a SEPARATE directory so it doesn't
clobber the real or straight data.

Run inside the container:
    python3 convert_clas12_single_track.py
"""

import os
import numpy as np
import pandas as pd
from mmap_ninja import RaggedMmap

# ============================================================
# CONFIG
# ============================================================

CSV_FILE_BST = "/workspace/PP_collision/data/THREE_HADRONS_bst_crosses.csv"
CSV_FILE_BMT = "/workspace/PP_collision/data/THREE_HADRONS_bmt_combined_points.csv"

OUT_DIR = "/workspace/PP_collision/data/mmap_single_track"

TEST_FRACTION = 0.10
RANDOM_SEED = 42
N_REG_COLS = 7

# ============================================================
# LOAD + KEEP NEEDED COLUMNS
# ============================================================

KEEP = ["event", "x0", "y0", "z0", "trkID"]
print("Loading CSVs...")
df = pd.concat([
    pd.read_csv(CSV_FILE_BST)[KEEP],
    pd.read_csv(CSV_FILE_BMT)[KEEP],
], ignore_index=True)
print(f"  total hits (before dropping noise): {len(df)}")

# Drop noise hits (trkID == -1) — clean single-track sequences only.
df = df[df["trkID"] != -1].copy()
print(f"  hits after dropping trkID==-1: {len(df)}")

# ============================================================
# GROUP BY (event, trkID) -> one sequence per track
# ============================================================

# Each (event, trkID) pair is one single-track sequence. This keeps event 0's
# track 1 separate from event 5's track 1, since we group on the pair.
track_sequences = []   # list of (n_hits, 4) arrays [x0,y0,z0,trkID]
for (ev, trk), group in df.groupby(["event", "trkID"]):
    xyz_trk = group[["x0", "y0", "z0", "trkID"]].to_numpy(dtype=np.float32)
    track_sequences.append(xyz_trk)

n_tracks = len(track_sequences)
hit_counts = np.array([s.shape[0] for s in track_sequences])
print(f"  total single-track sequences: {n_tracks}")
print(f"  hits per track: mean={hit_counts.mean():.2f}, "
      f"min={hit_counts.min()}, max={hit_counts.max()}")
print(f"  hits-per-track distribution: "
      f"{dict(zip(*np.unique(hit_counts, return_counts=True)))}")

# ============================================================
# 90/10 SPLIT (by track)
# ============================================================

rng = np.random.RandomState(RANDOM_SEED)
idx = np.arange(n_tracks)
rng.shuffle(idx)
n_test = max(1, int(round(n_tracks * TEST_FRACTION)))
test_idx = sorted(idx[:n_test].tolist())
pretrain_idx = sorted(idx[n_test:].tolist())
print(f"  pretrain tracks: {len(pretrain_idx)}, test tracks: {len(test_idx)}")

# ============================================================
# WRITE RAGGEDMMAP
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)

def gen_features(ids):
    for i in ids:
        yield track_sequences[i][:, :3]        # (n_hits, 3) = x,y,z

def gen_seg(ids):
    for i in ids:
        yield track_sequences[i][:, 3]         # (n_hits,) = trkID

def gen_reg(ids):
    for i in ids:
        yield np.zeros((track_sequences[i].shape[0], N_REG_COLS), dtype=np.float32)

def write_split(name, ids):
    print(f"Writing '{name}' ({len(ids)} tracks)...")
    RaggedMmap.from_generator(out_dir=os.path.join(OUT_DIR, f"features_{name}"),
                              sample_generator=gen_features(ids), batch_size=1024, verbose=True)
    RaggedMmap.from_generator(out_dir=os.path.join(OUT_DIR, f"seg_target_{name}"),
                              sample_generator=gen_seg(ids), batch_size=1024, verbose=True)
    RaggedMmap.from_generator(out_dir=os.path.join(OUT_DIR, f"reg_target_{name}"),
                              sample_generator=gen_reg(ids), batch_size=1024, verbose=True)

write_split("pretrain", pretrain_idx)
write_split("test", test_idx)

print("\nDone. Wrote to:", OUT_DIR)
check = RaggedMmap(os.path.join(OUT_DIR, "features_pretrain"))
ev0 = np.array(check[0])
print("first track shape:", ev0.shape, "(should be (n_hits, 3), n_hits ~5-6)")
print("first track r values:", np.sqrt(ev0[:,0]**2 + ev0[:,1]**2).round(3))
print("  ^ should be DISTINCT increasing radii (one track crossing layers, no repeats)")