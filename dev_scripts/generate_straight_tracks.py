"""
Synthetic straight-track CLAS12 baseline generator.

Builds events that match real CLAS12 MC structure in every way EXCEPT the one
variable under test: tracks are perfectly straight radial lines from the origin
(zero magnetic-field bending). Used to test whether the low real-data
pretraining loss reflects genuine learned structure or just the easy/predictable
component of the task.

Each event: exactly 3 straight tracks, one hit per track at each of the 6 fixed
layer radii = 18 hits/event. No misses, no noise (purest baseline).

A straight radial track from the origin has CONSTANT phi and CONSTANT eta across
all 6 layers — the hit at each layer is just (that direction) scaled to that
layer's radius. So the next-nearest-neighbor is trivially "same angle, next
radius out" — the maximally predictable case.

Output: same 6 RaggedMmap directories the dataset pipeline expects, written to a
SEPARATE directory so it doesn't clobber the real data.

Run inside the container:
    python3 generate_straight_tracks.py
"""

import os
import numpy as np
from mmap_ninja import RaggedMmap

# ============================================================
# CONFIG
# ============================================================

OUT_DIR = "/workspace/PP_collision/data/mmap_straight"   # SEPARATE from real data/mmap

N_EVENTS = 8858            # match real pretrain event count (post-filter) for comparable loss scale
N_TRACKS_PER_EVENT = 3     # MC truth: exactly 3 particles
TEST_FRACTION = 0.10
RANDOM_SEED = 1234
N_REG_COLS = 7

# The 6 real CLAS12 layer radii (raw units), same constants as the ordering code.
LAYER_RADII = np.array([
    6.52944,    # BST 1
    9.28923,    # BST 2
    12.03261,   # BST 3
    14.76460,   # BMT 1
    19.26460,   # BMT 2
    22.26460,   # BMT 3
], dtype=np.float64)

# Angular coverage to sample track directions from. Defaults below are the real
# measured ranges; UPDATE from characterize_real_events.py output if different.
ETA_MIN, ETA_MAX = -1.908, 1.209
PHI_MIN, PHI_MAX = -np.pi, np.pi

# ============================================================
# GENERATION
# ============================================================

rng = np.random.RandomState(RANDOM_SEED)

def eta_to_theta(eta):
    # eta = -ln(tan(theta/2))  ->  theta = 2*arctan(exp(-eta))
    return 2.0 * np.arctan(np.exp(-eta))

def make_event():
    """One event: 3 straight radial tracks, 18 hits, columns [x,y,z,trkID]."""
    rows = []
    for trk in range(1, N_TRACKS_PER_EVENT + 1):
        # Random straight-line direction (constant phi, eta along the track).
        phi = rng.uniform(PHI_MIN, PHI_MAX)
        eta = rng.uniform(ETA_MIN, ETA_MAX)
        theta = eta_to_theta(eta)
        # Unit direction in 3D. r = transverse radius = sin(theta); to place a hit
        # AT a given layer transverse-radius R_layer, scale so sqrt(x^2+y^2)=R_layer.
        # x = R_layer*cos(phi), y = R_layer*sin(phi), z = R_layer / tan(theta).
        for R in LAYER_RADII:
            x = R * np.cos(phi)
            y = R * np.sin(phi)
            z = R / np.tan(theta)
            rows.append([x, y, z, float(trk)])
    return np.array(rows, dtype=np.float32)   # (18, 4)

print(f"Generating {N_EVENTS} straight-track events...")
events = [make_event() for _ in range(N_EVENTS)]

# split
idx = np.arange(N_EVENTS)
rng.shuffle(idx)
n_test = max(1, int(round(N_EVENTS * TEST_FRACTION)))
test_idx = set(idx[:n_test].tolist())
pretrain_idx = [i for i in range(N_EVENTS) if i not in test_idx]
test_idx = sorted(test_idx)
print(f"  pretrain: {len(pretrain_idx)}, test: {len(test_idx)}")

# ============================================================
# WRITE RAGGEDMMAP
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)

def gen_features(ids):
    for i in ids:
        yield events[i][:, :3]            # (18, 3) = x,y,z

def gen_seg(ids):
    for i in ids:
        yield events[i][:, 3]             # (18,) = trkID

def gen_reg(ids):
    for i in ids:
        yield np.zeros((events[i].shape[0], N_REG_COLS), dtype=np.float32)

def write_split(name, ids):
    print(f"Writing '{name}' ({len(ids)} events)...")
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
print("first event shape:", ev0.shape, "(should be (18, 3))")
print("first event r values:", np.sqrt(ev0[:,0]**2 + ev0[:,1]**2).round(3))
print("  ^ should be the 6 layer radii, each appearing 3x (once per track)")