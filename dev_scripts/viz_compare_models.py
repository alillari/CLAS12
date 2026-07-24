"""
3D + top-down comparison of model predictions for ONE event.

Replaces the obsolete viz_compare_4models.py. Fitted to the current setup:
  - reads models from a SWEEP_DIR (e.g. LR_Full_Test_1_2026-07-20)
  - regression only (band-classification decode removed)
  - mmap_v4 data via the ragged_npy reader
  - radius-based kNNN objective: each prediction is the nearest hit in a
    band FURTHER OUT than the query hit, so predictions should land on the
    next ring outward -- the rings make that visually checkable.

  solid blue          = real hits
  transparent colors  = each model's predicted next-neighbour points

Run (from /workspace/PP_collision):
    python3 dev_scripts/viz_compare_models.py --event 0
    python3 dev_scripts/viz_compare_models.py --event 3 --widths 64 256 1536
"""
import os, sys, glob, re, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa

sys.path.insert(0, "/workspace/PP_collision")
from fm4npp.utils import YParams
from fm4npp.datasets.dataset_pretrain import TPCBatchDataset
from fm4npp.models.mambagpt import Mamba1GPT
from fm4npp.hilbert import CLAS12_LAYER_RADII_RAW

# ============================================================
SWEEP_DIR = "/workspace/PP_collision/checkpoints/LR_Full_Test_1_2026-07-20"
RUN_NUM   = "sweep"
DEPTH     = 12
# ============================================================

GEN_YAML = f"{SWEEP_DIR}/_grid_generated.yaml"
RADII = CLAS12_LAYER_RADII_RAW
COLOR_HIT = "tab:blue"
POINT_SIZE = 60
M_LABEL = {64:'m1', 128:'m2', 256:'m3', 512:'m4', 1024:'m5', 1536:'m6'}
PALETTE = {64:"tab:red", 128:"tab:orange", 256:"tab:green",
           512:"tab:purple", 1024:"tab:brown", 1536:"tab:pink"}

p = argparse.ArgumentParser()
p.add_argument("--event", type=int, default=0)
p.add_argument("--widths", type=int, nargs="*", default=[64, 256, 1536],
               help="widths to overlay (default: 64 256 1536 = m1/m3/m6)")
args = p.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_cfg(width):
    """Find the config name scale_w{width}_d{DEPTH}_n* in the sweep folder."""
    hits = glob.glob(f"{SWEEP_DIR}/scale_w{width}_d{DEPTH}_n*")
    return os.path.basename(hits[0]) if hits else None


def ckpt_path(cfg):
    return f"{SWEEP_DIR}/{cfg}/{RUN_NUM}/training_checkpoints/ckpt_best.tar"


def load_model(cfg):
    prm = YParams(os.path.abspath(GEN_YAML), cfg)
    m = Mamba1GPT(embed_dim=prm.embed_dim, num_layers=prm.num_layers_backbone,
                  d_state=prm.d_state, klen=prm.klen, dropout=prm.dropout,
                  embed_method=prm.embed_method, pe_method=prm.pe_method).to(device)
    ck = torch.load(ckpt_path(cfg), map_location=device, weights_only=False)
    st = ck.get("model_state", ck.get("model_state_dict", ck))
    m.load_state_dict({k.replace("module.", ""): v for k, v in st.items()}, strict=False)
    m.eval()
    return m, prm


# --- dataset (any cell's params work; they share data settings) ---
first_cfg = next((find_cfg(w) for w in args.widths if find_cfg(w)), None)
if first_cfg is None:
    raise SystemExit(f"No runs found in {SWEEP_DIR} for widths {args.widths}")
bp = YParams(os.path.abspath(GEN_YAML), first_cfg)
ds = TPCBatchDataset(data_root=bp.data_root,
                     version=getattr(bp, "data_version", "dataset_name"),
                     split="test", reader_type=getattr(bp, "reader_type", "ragged_mmap"),
                     num_pred_points=bp.klen, group_size=bp.group_size,
                     normalize=True, nleave=bp.nleave,
                     chunk_training=bp.chunk_training, train=False, order=bp.order)


def unnorm(a): return ds.apply_unnorm(a.clone())
def to_xyz(a):
    eta, phi, r = a[:, 0], a[:, 1], a[:, 2]
    x = r * torch.cos(phi); y = r * torch.sin(phi)
    th = 2.0 * torch.atan(torch.exp(-eta)); z = r / torch.tan(th)
    return torch.stack([x, y, z], dim=1)


sp_t, seg, knearest = ds[args.event]
sp = sp_t.unsqueeze(0).to(device)
N = sp.shape[1]
actual = knearest.reshape(N, 3)
valid = (actual != -100).all(dim=1)
hits_xyz = to_xyz(unnorm(sp_t)).numpy()
act_xyz = to_xyz(unnorm(actual[valid])).numpy()
print(f"Event {args.event}: {N} hits, {int(valid.sum())} with a valid next-neighbour target "
      f"({N - int(valid.sum())} outermost-band hits have none)")

pred_by_model = {}
for w in args.widths:
    cfg = find_cfg(w)
    if cfg is None:
        print(f"SKIP w{w}: no run folder in sweep"); continue
    if not os.path.exists(ckpt_path(cfg)):
        print(f"SKIP w{w}: no checkpoint yet ({cfg})"); continue
    model, prm = load_model(cfg)
    with torch.no_grad():
        out = model(sp).squeeze(0).cpu()[valid]
    pred_real = unnorm(out[:, :3])
    err = np.linalg.norm(to_xyz(pred_real).numpy() - act_xyz, axis=1).mean()
    pred_by_model[w] = (to_xyz(pred_real).numpy(), PALETTE.get(w, "gray"))
    print(f"  {M_LABEL.get(w,'w'+str(w))} (w{w}): {len(pred_real)} predictions, "
          f"mean 3D error {err:.3f} cm")

if not pred_by_model:
    raise SystemExit("No models could be loaded.")

# --- figure ---
fig = plt.figure(figsize=(18, 8))
ax = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122)

allz = np.concatenate([hits_xyz[:, 2]] + [v[0][:, 2] for v in pred_by_model.values()])
ax.plot([0, 0], [0, 0], [allz.min(), allz.max()], color="black", lw=1.5, ls="--")
theta = np.linspace(0, 2 * np.pi, 200)
for rad in RADII:
    rx, ry = rad * np.cos(theta), rad * np.sin(theta)
    ax.plot(rx, ry, np.zeros_like(theta), color="black", lw=0.8, alpha=0.3)
    ax2.plot(rx, ry, color="black", lw=0.8, alpha=0.3)
ax2.plot(0, 0, marker="o", color="black", markersize=4)


def scat(xyz, color, alpha, edge=False):
    kw = dict(c=color, s=POINT_SIZE, alpha=alpha, depthshade=False)
    if edge: kw.update(edgecolors="black", linewidths=0.4)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], **kw)
    kw2 = dict(c=color, s=POINT_SIZE, alpha=alpha)
    if edge: kw2.update(edgecolors="black", linewidths=0.4)
    ax2.scatter(xyz[:, 0], xyz[:, 1], **kw2)


scat(hits_xyz, COLOR_HIT, 1.0, edge=True)
for w, (xyz, color) in pred_by_model.items():
    scat(xyz, color, 0.35)

ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax2.set_xlabel("x"); ax2.set_ylabel("y")
ax2.set_title("Top-down (x-y)"); ax2.set_aspect("equal", adjustable="datalim")
widths_s = ", ".join(M_LABEL.get(w, f"w{w}") for w in pred_by_model)
ax.set_title(f"Event {args.event}: real hits vs predictions (k=1, radius-based target)\n{widths_s}")

legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_HIT,
                 markersize=9, label="real hits")]
for w, (xyz, color) in pred_by_model.items():
    legend.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                         markersize=9, alpha=0.6,
                         label=f"{M_LABEL.get(w, 'w'+str(w))} (w{w}) prediction"))
legend.append(Line2D([0], [0], color="black", lw=1.5, ls="--", label="z-axis ray"))
fig.legend(handles=legend, bbox_to_anchor=(1.0, 1.0), loc="upper left")

out = f"{SWEEP_DIR}/predictions_event{args.event}.png"
plt.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")