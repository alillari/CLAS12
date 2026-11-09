"""
FM4NPP Figure-8 style PCA visualization: coloured by TRACK ID -> reproduces
the paper's Fig 8: raw FM features (mixed) vs after a linear map (per-track
clusters). Linear map trained with a contrastive loss on seg_target.

[NOISE HANDLING] seg_target == -1 marks unmatched/noise hits. These are NOT
a real class (two noise hits aren't "the same track" just because they share
the -1 label), so they are EXCLUDED from the positive (pull-together) pairs,
but kept in the negative (push-apart) pairs against everything else,
including other noise. This teaches the map "noise doesn't belong with any
real track" without falsely clustering unrelated noise hits together. In the
plot, noise gets a fixed gray 'x' marker (not part of the rotating track
color scheme), consistent across all pooled events.

Backbone stays frozen throughout. The map is TRAINED (as in the paper), not
an unsupervised transform.

Run (from /workspace/PP_collision):
    python3 dev_scripts/viz_pca_both.py --width 1536
    python3 dev_scripts/viz_pca_both.py --width 1536 --event 5 --n-events 6
"""
import os, sys, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace/PP_collision")
from fm4npp.utils import YParams
from fm4npp.datasets.dataset_pretrain import (
    TPCBatchDataset, cartesian_to_polar_batched, clas12_band_hilbert_order)
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

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, default=1536)
ap.add_argument("--event", type=int, default=0)
ap.add_argument("--n-events", type=int, default=4)
args = ap.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M_LABEL = {64:'m1',128:'m2',256:'m3',512:'m4',1024:'m5',1536:'m6'}


def find_cfg(w):
    h = glob.glob(f"{SWEEP_DIR}/scale_w{w}_d{DEPTH}_n*")
    return os.path.basename(h[0]) if h else None

cfg = find_cfg(args.width)
if cfg is None:
    raise SystemExit(f"no run for width {args.width}")
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

# ---- gather training features + labels ----
ds_tr = get_ds("pretrain")
print(f"Collecting {TRAIN_EVENTS} train events...")
F_tr, T_tr = [], []
for i in range(min(TRAIN_EVENTS, len(ds_tr))):
    sp, seg, _ = ds_tr[i]
    F_tr.append(fm_feat(sp)); T_tr.append(seg.long())

# ---- (1) contrastive linear map for TRACK separation ----
lin_trk = nn.Linear(prm.embed_dim, PROJ_DIM, bias=False).to(device)
opt = torch.optim.Adam(lin_trk.parameters(), lr=1e-3)
print("Training track-contrastive linear map (noise excluded from positive pairs)...")
for ep in range(PROBE_EPOCHS):
    tot = 0.0
    for f, t in zip(F_tr, T_tr):
        f, t = f.to(device), t.to(device)
        e = lin_trk(f); d = torch.cdist(e, e)
        eye = torch.eye(len(t), device=device)
        is_noise = (t == NOISE_LABEL)
        noise_pair = is_noise[:, None] | is_noise[None, :]   # either point is noise
        # [NOISE FIX] same-track (positive) pairs require matching, non-noise labels.
        # Two noise hits sharing label -1 are NOT treated as the same class.
        same = (t[:, None] == t[None, :]).float() * (1 - eye) * (~noise_pair).float()
        diff = 1 - same - eye   # everything else (incl. noise-vs-anything) is negative
        diff = diff.clamp(min=0)
        pos = (same*d.pow(2)).sum()/same.sum().clamp(min=1)
        neg = (diff*torch.relu(MARGIN-d).pow(2)).sum()/diff.sum().clamp(min=1)
        loss = pos+neg; opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    print(f"  epoch {ep+1}: {tot/len(F_tr):.4f}")

# ---- visualise held-out test events (TRACK ID only -- no momentum map) ----
ds_te = get_ds("test")
RAW, PRJt, LAB = [], [], []
for k in range(args.n_events):
    sp, seg, _ = ds_te[args.event+k]
    f = fm_feat(sp)
    with torch.no_grad():
        et = lin_trk(f.to(device)).cpu()
    RAW.append(f.numpy()); PRJt.append(et.numpy())
    seg_np = seg.long().numpy()
    # [NOISE FIX] keep noise as a SINGLE consistent label (-1) across all pooled
    # events, rather than offsetting it per-event like real tracks (+100*k) --
    # noise from different events isn't meaningfully "different classes", and
    # giving it one fixed identity lets the plot render it as one consistent
    # gray marker instead of scattering it across many rotating colors.
    offset_labels = np.where(seg_np == NOISE_LABEL, NOISE_LABEL, seg_np + 100*k)
    LAB.append(offset_labels)
RAW=np.concatenate(RAW); PRJt=np.concatenate(PRJt)
LAB=np.concatenate(LAB)

def pca2(X, label=""):
    if not np.isfinite(X).all():
        n_bad = (~np.isfinite(X)).sum()
        raise ValueError(
            f"pca2({label}): input contains {n_bad} non-finite (NaN/Inf) values "
            f"out of {X.size} -- SVD cannot run on this. Check the training loss "
            f"printed above for where it diverged."
        )
    Xc = X - X.mean(0, keepdims=True)
    _,_,Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T

def sep_ratio(X, lab):
    """same/diff distance ratio, computed over REAL tracks only (noise excluded --
    it isn't a coherent class, so including it would distort the metric)."""
    real_mask = lab != NOISE_LABEL
    X, lab = X[real_mask], lab[real_mask]
    D = np.linalg.norm(X[:,None,:]-X[None,:,:], axis=-1)
    same = (lab[:,None]==lab[None,:]) & ~np.eye(len(lab),dtype=bool)
    diff = (lab[:,None]!=lab[None,:])
    return D[same].mean()/D[diff].mean()

# --- FIGURE 1: track id ---
A, Bt = pca2(RAW, "RAW"), pca2(PRJt, "PRJt (track map)")
r_raw, r_prj = sep_ratio(RAW,LAB), sep_ratio(PRJt,LAB)

# distinct colors for real tracks (60-color palette + marker cycling, so this
# stays readable even with many pooled events/tracks); noise gets one fixed
# gray 'x' marker, drawn separately from the rotating scheme.
real_uniq = np.unique(LAB[LAB != NOISE_LABEL])
K = len(real_uniq)
remap = {u:i for i,u in enumerate(real_uniq)}
base_colors = (list(plt.get_cmap("tab20").colors)
               + list(plt.get_cmap("tab20b").colors)
               + list(plt.get_cmap("tab20c").colors))
markers = ["o","s","^","D","v","P","X","*","h","<",">"]
def color_of(i): return base_colors[i % len(base_colors)]
def marker_of(i): return markers[(i // len(base_colors)) % len(markers)]

fig,axes = plt.subplots(1,2,figsize=(14,6))
for ax,X,title,ratio in [(axes[0],A,"PCA of raw FM features",r_raw),
                          (axes[1],Bt,"PCA after track linear map",r_prj)]:
    is_noise = LAB == NOISE_LABEL
    if is_noise.any():
        ax.scatter(X[is_noise,0], X[is_noise,1], s=35, color="gray", marker="x",
                   alpha=0.5, linewidths=1.0, label="noise" if ax is axes[0] else None)
    for u in real_uniq:
        i = remap[u]
        m = LAB==u
        ax.scatter(X[m,0],X[m,1],s=70,color=color_of(i),marker=marker_of(i),
                   edgecolors="black",linewidths=0.4, alpha=0.9)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    ax.set_title(f"{title}\nsame/diff track dist ratio = {ratio:.3f} (real tracks only)")
    ax.grid(alpha=0.2)
axes[0].legend(fontsize=8, loc="best")
fig.suptitle(f"{M_LABEL.get(args.width,'w'+str(args.width))} frozen backbone — coloured by TRACK ID "
             f"({K} real tracks + noise, across {args.n_events} event(s))", y=1.02)
out1 = f"{SWEEP_DIR}/pca_trackid_w{args.width}.png"
fig.tight_layout(); fig.savefig(out1,dpi=150,bbox_inches="tight")
print(f"\ntracks shown: {K}  (noise points: {int((LAB==NOISE_LABEL).sum())})")
print(f"same/diff track ratio (real tracks only): raw {r_raw:.3f} -> mapped {r_prj:.3f}")
print(f"Saved: {out1}")