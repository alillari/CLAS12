"""
3D + top-down comparison for ONE 3-particle event, all FOUR models:
  m3 regression, m6 regression, m3 band-classification, m6 band-classification

  - solid blue        = real hits
  - transparent colors = each model's predicted next-neighbor points

Same project style: 3D view + top-down x-y, 6 layer rings, dashed z-axis ray.

Run INSIDE container (after the two pip installs):
    python3 viz_compare_4models.py --event 0
"""
import os, sys, argparse
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

YAML = "/workspace/PP_collision/scripts/configs/mamba_pretrain.yaml"
OUT_DIR = "/workspace/PP_collision/sweep_analysis"
RADII = CLAS12_LAYER_RADII_RAW
CKPT_ROOT = "/workspace/PP_collision/checkpoints"
RUN_NUM = "sweep"

MODELS = [
    ("m3 regression",  "sweep_real_large_k1", False, "tab:red"),
    ("m6 regression",  "sweep_real_m6_k1",    False, "tab:orange"),
    ("m3 band-class",  "band_real_large_k1",  True,  "tab:purple"),
    ("m6 band-class",  "band_real_m6_k1",     True,  "tab:green"),
]
COLOR_HIT = "tab:blue"
POINT_SIZE = 60

p = argparse.ArgumentParser()
p.add_argument("--event", type=int, default=0)
args = p.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ckpt_path(cfg): return f"{CKPT_ROOT}/{cfg}/{RUN_NUM}/training_checkpoints/ckpt_best.tar"

def load_model(cfg, is_band):
    prm = YParams(os.path.abspath(YAML), cfg)
    m = Mamba1GPT(embed_dim=prm.embed_dim, num_layers=prm.num_layers_backbone, d_state=prm.d_state,
                  klen=prm.klen, dropout=prm.dropout, embed_method=prm.embed_method, pe_method=prm.pe_method,
                  band_classification=getattr(prm, 'band_classification', False),
                  n_bands=getattr(prm, 'n_bands', 6)).to(device)
    ck = torch.load(ckpt_path(cfg), map_location=device, weights_only=False)
    st = ck.get("model_state", ck.get("model_state_dict", ck))
    m.load_state_dict({k.replace("module.", ""): v for k, v in st.items()}, strict=False)
    m.eval()
    return m

base_params = YParams(os.path.abspath(YAML), "sweep_real_large_k1")
ds = TPCBatchDataset(data_root=base_params.data_root, version=getattr(base_params, "data_version", "dataset_name"),
                     split="test", num_pred_points=base_params.klen, group_size=base_params.group_size,
                     normalize=True, nleave=base_params.nleave, chunk_training=base_params.chunk_training,
                     train=False, order=base_params.order)

def unnorm(a): return ds.apply_unnorm(a.clone())
def to_xyz(a):
    eta, phi, r = a[:,0], a[:,1], a[:,2]
    x = r*torch.cos(phi); y = r*torch.sin(phi)
    th = 2.0*torch.atan(torch.exp(-eta)); z = r/torch.tan(th)
    return torch.stack([x,y,z], dim=1)

sp_t, seg, knearest = ds[args.event]
sp = sp_t.unsqueeze(0).to(device)
N = sp.shape[1]
actual = knearest.reshape(N, 3)
valid = (actual != -100).all(dim=1)
hits_xyz = to_xyz(unnorm(sp_t)).numpy()

pred_xyz_by_model = {}
for label, cfg, is_band, color in MODELS:
    if not os.path.exists(ckpt_path(cfg)):
        print(f"SKIP {label}: no checkpoint at {ckpt_path(cfg)}")
        continue
    model = load_model(cfg, is_band)
    with torch.no_grad():
        out = model(sp).squeeze(0).cpu()[valid]
    if is_band:
        ep, logits = out[:, :2], out[:, 2:]
        bidx = logits.argmax(dim=-1).numpy()
        r_real = torch.tensor([RADII[i] for i in bidx], dtype=torch.float32)
        eta_r = ds.minmax_unnormalize(ep[:,0], ds.eta_lim['max'], ds.eta_lim['min'])
        phi_r = ds.minmax_unnormalize(ep[:,1], ds.phi_lim['max'], ds.phi_lim['min'])
        pred_real = torch.stack([eta_r, phi_r, r_real], dim=1)
    else:
        pred_real = unnorm(out)
    pred_xyz_by_model[label] = (to_xyz(pred_real).numpy(), color)
    print(f"{label}: {len(pred_real)} predicted points")

fig = plt.figure(figsize=(18, 8))
ax  = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122)

allz_list = [hits_xyz[:,2]] + [v[0][:,2] for v in pred_xyz_by_model.values()]
allz = np.concatenate(allz_list)
ax.plot([0,0],[0,0],[allz.min(), allz.max()], color="black", lw=1.5, ls="--")
theta = np.linspace(0, 2*np.pi, 200)
for rad in RADII:
    rx, ry = rad*np.cos(theta), rad*np.sin(theta)
    ax.plot(rx, ry, np.zeros_like(theta), color="black", lw=0.8, alpha=0.3)
    ax2.plot(rx, ry, color="black", lw=0.8, alpha=0.3)
ax2.plot(0, 0, marker="o", color="black", markersize=4)

def scat(xyz, color, alpha, edge=False):
    kw = dict(c=color, s=POINT_SIZE, alpha=alpha, depthshade=False)
    if edge: kw.update(edgecolors="black", linewidths=0.4)
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], **kw)
    kw2 = dict(c=color, s=POINT_SIZE, alpha=alpha)
    if edge: kw2.update(edgecolors="black", linewidths=0.4)
    ax2.scatter(xyz[:,0], xyz[:,1], **kw2)

scat(hits_xyz, COLOR_HIT, 1.0, edge=True)
for label, (xyz, color) in pred_xyz_by_model.items():
    scat(xyz, color, 0.35)

ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax2.set_xlabel("x"); ax2.set_ylabel("y")
ax2.set_title("Top-down (x-y)"); ax2.set_aspect("equal", adjustable="datalim")
ax.set_title(f"3-particle event {args.event}: real hits vs 4 models (k=1)\n"
             "m3 vs m6, regression vs band-classification")

legend_elements = [Line2D([0],[0], marker="o", color="w", markerfacecolor=COLOR_HIT, markersize=9, label="real hits")]
for label, (xyz, color) in pred_xyz_by_model.items():
    legend_elements.append(Line2D([0],[0], marker="o", color="w", markerfacecolor=color, markersize=9, alpha=0.6, label=label))
legend_elements.append(Line2D([0],[0], color="black", lw=1.5, ls="--", label="z-axis ray"))
fig.legend(handles=legend_elements, bbox_to_anchor=(1.0, 1.0), loc="upper left")

os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR, f"compare_4models_event{args.event}.png")
plt.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
