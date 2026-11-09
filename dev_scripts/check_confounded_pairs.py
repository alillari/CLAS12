"""
Direct test of whether the track-contrastive linear map separates
GENUINELY CONFOUNDED pairs (different tracks, but spatially close together
in the raw FM feature space) -- as opposed to the aggregate same/diff ratio,
which averages over ALL pairs and can look good purely from already-easy
pairs getting further apart, while hard/overlapping cases stay unresolved.

Method:
  1. In RAW feature space, find all different-track pairs, rank by distance.
  2. Take the bottom X% (closest = most confounded) as the "hard" set.
  3. Check where THOSE SAME pairs land in the MAPPED space -- did they
     actually get pushed apart, or are they still stuck together?
  4. Compare against same-track pairs' mapped distances as a reference scale.

Run: same width/event args as viz_pca_both.py, uses the same model + map.
    python3 dev_scripts/check_confounded_pairs.py --width 1536 --n-events 10
"""
import os, sys, glob, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/workspace/PP_collision")
from fm4npp.utils import YParams
from fm4npp.datasets.dataset_pretrain import TPCBatchDataset
from fm4npp.models.mambagpt import Mamba1GPT

SWEEP_DIR = "/workspace/PP_collision/checkpoints/Unbiased_Cluster_Full_Sweep_2026-08-13"
RUN_NUM = "sweep"
DEPTH = 12
GEN_YAML = f"{SWEEP_DIR}/_grid_generated.yaml"
TRAIN_EVENTS = 3000
PROBE_EPOCHS = 10
PROJ_DIM = 32
MARGIN = 1.0
NOISE_LABEL = -1
HARD_PCT = 25  # bottom X% closest different-track pairs = "confounded"

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, default=1536)
ap.add_argument("--event", type=int, default=0)
ap.add_argument("--n-events", type=int, default=10)
args = ap.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M_LABEL = {64:'m1',128:'m2',256:'m3',512:'m4',1024:'m5',1536:'m6'}


def find_cfg(w):
    h = glob.glob(f"{SWEEP_DIR}/scale_w{w}_d{DEPTH}_n*")
    return os.path.basename(h[0]) if h else None

cfg = find_cfg(args.width)
prm = YParams(os.path.abspath(GEN_YAML), cfg)
model = Mamba1GPT(embed_dim=prm.embed_dim, num_layers=prm.num_layers_backbone,
                  d_state=prm.d_state, klen=prm.klen, dropout=prm.dropout,
                  embed_method=prm.embed_method, pe_method=prm.pe_method).to(device)
ck = torch.load(f"{SWEEP_DIR}/{cfg}/{RUN_NUM}/training_checkpoints/ckpt_best.tar",
                map_location=device, weights_only=False)
st = ck.get("model_state", ck.get("model_state_dict", ck))
model.load_state_dict({k.replace("module.", ""): v for k, v in st.items()}, strict=False)
model.eval()
for p in model.parameters():
    p.requires_grad = False
print(f"Loaded {M_LABEL.get(args.width,'w'+str(args.width))} (w{args.width})")

_feat = {}
model.norm.register_forward_hook(lambda m,i,o: _feat.__setitem__("x", o.detach()))
def fm_feat(sp):
    with torch.no_grad():
        model(sp.unsqueeze(0).to(device))
    return _feat["x"].squeeze(0).cpu()

def get_ds(split):
    return TPCBatchDataset(
        data_root=prm.data_root, version=getattr(prm,"data_version","dataset_name"),
        split=split, reader_type=getattr(prm,"reader_type","ragged_mmap"),
        num_pred_points=prm.klen, group_size=prm.group_size, normalize=True,
        nleave=prm.nleave, chunk_training=False, train=(split=="pretrain"),
        order=prm.order)

# ---- retrain the SAME contrastive map (reproducing viz_pca_both.py exactly) ----
ds_tr = get_ds("pretrain")
print(f"Collecting {TRAIN_EVENTS} train events...")
F_tr, T_tr = [], []
for i in range(min(TRAIN_EVENTS, len(ds_tr))):
    sp, seg, _ = ds_tr[i]
    F_tr.append(fm_feat(sp)); T_tr.append(seg.long())

lin_trk = nn.Linear(prm.embed_dim, PROJ_DIM, bias=False).to(device)
opt = torch.optim.Adam(lin_trk.parameters(), lr=1e-3)
print("Training track-contrastive linear map...")
for ep in range(PROBE_EPOCHS):
    for f, t in zip(F_tr, T_tr):
        f, t = f.to(device), t.to(device)
        e = lin_trk(f); d = torch.cdist(e, e)
        eye = torch.eye(len(t), device=device)
        is_noise = (t == NOISE_LABEL)
        noise_pair = is_noise[:, None] | is_noise[None, :]
        same = (t[:, None] == t[None, :]).float() * (1 - eye) * (~noise_pair).float()
        diff = (1 - same - eye).clamp(min=0)
        pos = (same*d.pow(2)).sum()/same.sum().clamp(min=1)
        neg = (diff*torch.relu(MARGIN-d).pow(2)).sum()/diff.sum().clamp(min=1)
        loss = pos+neg; opt.zero_grad(); loss.backward(); opt.step()
print("  done")

# ---- test events: gather RAW + MAPPED features ----
ds_te = get_ds("test")
RAW, PRJ, LAB = [], [], []
for k in range(args.n_events):
    sp, seg, _ = ds_te[args.event+k]
    f = fm_feat(sp)
    with torch.no_grad():
        e = lin_trk(f.to(device)).cpu()
    seg_np = seg.long().numpy()
    offset = np.where(seg_np == NOISE_LABEL, NOISE_LABEL, seg_np + 100*k)
    RAW.append(f.numpy()); PRJ.append(e.numpy()); LAB.append(offset)
RAW = np.concatenate(RAW); PRJ = np.concatenate(PRJ); LAB = np.concatenate(LAB)

# ---- restrict to real tracks only ----
real = LAB != NOISE_LABEL
RAW, PRJ, LAB = RAW[real], PRJ[real], LAB[real]
n = len(LAB)
print(f"\n{n} real (non-noise) points across {args.n_events} events, "
      f"{len(np.unique(LAB))} distinct tracks")

D_raw = np.linalg.norm(RAW[:,None,:]-RAW[None,:,:], axis=-1)
D_map = np.linalg.norm(PRJ[:,None,:]-PRJ[None,:,:], axis=-1)
same_mask = (LAB[:,None]==LAB[None,:]) & ~np.eye(n, dtype=bool)
diff_mask = (LAB[:,None]!=LAB[None,:])

same_map_mean = D_map[same_mask].mean()
print(f"\nsame-track pairs: mean mapped distance = {same_map_mean:.4f}  (reference scale)")

# ============================================================
# [SUPERCLUSTER-AWARE] The naive "bottom 25% closest raw pairs" bucket can
# be contaminated by BETWEEN-supercluster pairs that happen to be adjacent,
# rather than genuinely confounded WITHIN-supercluster different-track pairs
# (e.g. 4 different tracks visually merged into one PCA blob). To directly
# test "do tracks sharing one visual blob get separated", we explicitly find
# the superclusters (KMeans on the same PCA-of-RAW projection used for the
# plot) and restrict the analysis to WITHIN-supercluster pairs only --
# between-supercluster pairs are discarded entirely from this check.
# ============================================================
from sklearn.cluster import KMeans

def pca2(X):
    Xc = X - X.mean(0, keepdims=True)
    _,_,Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T

RAW_2D = pca2(RAW)
N_SUPERCLUSTERS = 3
km = KMeans(n_clusters=N_SUPERCLUSTERS, n_init=10, random_state=42).fit(RAW_2D)
supercluster = km.labels_

print(f"\n=== superclusters found (KMeans, k={N_SUPERCLUSTERS}, on PCA-of-raw, matching the visual plot) ===")
for c in range(N_SUPERCLUSTERS):
    tracks_in_c = np.unique(LAB[supercluster == c])
    print(f"  supercluster {c}: {(supercluster==c).sum()} points, "
          f"{len(tracks_in_c)} distinct tracks {list(tracks_in_c)}")

within_super = supercluster[:,None] == supercluster[None,:]
diff_within = diff_mask & within_super
diff_between = diff_mask & ~within_super

print(f"\n=== WITHIN-supercluster different-track pairs only, n={diff_within.sum()} ===")
print(f"  (tracks sharing one visual blob -- the genuinely confounded case)")
if diff_within.sum() > 0:
    print(f"  raw distance   : mean={D_raw[diff_within].mean():.4f}")
    print(f"  mapped distance: mean={D_map[diff_within].mean():.4f}")
    print(f"  ratio to same-track mapped distance: {D_map[diff_within].mean()/same_map_mean:.3f}")
    print(f"  (1.0 = still confused with each other despite being trained; >>1 = successfully separated)")
else:
    print("  none found -- superclusters may be pure single-track (check counts above)")

print(f"\n=== BETWEEN-supercluster different-track pairs, n={diff_between.sum()} ===")
print(f"  (already visually separated blobs -- the 'easy' comparisons)")
if diff_between.sum() > 0:
    print(f"  raw distance   : mean={D_raw[diff_between].mean():.4f}")
    print(f"  mapped distance: mean={D_map[diff_between].mean():.4f}")
    print(f"  ratio to same-track mapped distance: {D_map[diff_between].mean()/same_map_mean:.3f}")

print(f"\n=== overall (for comparison to the aggregate ratio reported before) ===")
print(f"  all diff-track mapped distance: mean={D_map[diff_mask].mean():.4f}")
print(f"  same/diff ratio: {same_map_mean/D_map[diff_mask].mean():.3f}")