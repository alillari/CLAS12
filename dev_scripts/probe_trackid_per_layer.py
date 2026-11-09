"""
Per-layer track-ID separation comparison: old vs Chinchilla-corrected
checkpoints, at EVERY backbone depth (not just the final layer).

Uses return_z=True (with this session's already-applied cumulative-state
fix -- feature[i] = accumulated hidden state after layer i, not the
isolated delta) to get all 12 layers in one forward pass per event, then
trains a separate small contrastive linear probe per layer on the cached
features. Tests whether the "Chinchilla checkpoints separate tracks worse"
finding (seen at the final layer) is concentrated at later layers
specifically (consistent with late-layer specialization to the kNNN
pretext task) or holds uniformly across depth.

Defaults to m4/m5 (w512/w1024) -- the two widths that showed a real,
non-noise-sized effect in the final-layer-only comparison. Edit WIDTHS
below to extend to others.

Run: python3 dev_scripts/probe_trackid_per_layer.py
"""
import os, glob, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/workspace/PP_collision")
from fm4npp.models.mambagpt import Mamba1GPT
from fm4npp.datasets.dataset_pretrain import TPCBatchDataset

# ============================================================
OLD_SWEEP_DIR = "/workspace/PP_collision/checkpoints/Unbiased_Cluster_Full_Sweep_2026-08-13"
OLD_RUN_NUM = "sweep"
OLD_CONFIG_PATTERN = "scale_w{w}_d12_n39553933"
CHINCHILLA_ROOT = "/workspace/PP_collision/checkpoints"
CHINCHILLA_RUN_NUM = "sweep"
CHINCHILLA_CONFIG_PATTERN = "scale_w{w}_d12_chinchilla"

DEPTH = 12
WIDTHS = [512, 1024]  # m4, m5 -- the widths that showed a real effect. Edit to extend.
M_LABEL = {64: 'm1', 128: 'm2', 256: 'm3', 512: 'm4', 1024: 'm5'}

TRAIN_EVENTS = 1500  # per-layer probes are cheaper individually, but there are 12x as many now
PROBE_EPOCHS = 10
PROJ_DIM = 32
MARGIN = 1.0
NOISE_LABEL = -1
N_TEST_EVENTS = 10
OUT_PLOT = "/workspace/PP_collision/fig_trackid_per_layer.png"
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_chinchilla_sweep_dir():
    candidates = sorted(glob.glob(f"{CHINCHILLA_ROOT}/Chinchilla_Sweep_*"))
    if not candidates:
        raise SystemExit("No Chinchilla_Sweep_* folder found.")
    return candidates[-1]


def load_model(ckpt_path, embed_dim):
    model = Mamba1GPT(embed_dim=embed_dim, num_layers=DEPTH, d_state=16, klen=1,
                      dropout=0.1, embed_method='pos_only', pe_method='nerf').to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    st = ck.get("model_state", ck.get("model_state_dict", ck))
    model.load_state_dict({k.replace("module.", ""): v for k, v in st.items()}, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def get_ds(split):
    return TPCBatchDataset(
        data_root="/workspace/PP_collision/data/mmap_cluster_full",
        version="dataset_name", split=split, reader_type="ragged_mmap",
        num_pred_points=1, group_size=8, normalize=True, nleave=1e6,
        chunk_training=False, train=(split == "pretrain"), order="REP")


def extract_all_layers(model, sp):
    """One forward pass, return_z=True -- all 12 layers at once (cumulative
    state, post-fix). Returns a list of 12 tensors, each (N_points, embed_dim)."""
    with torch.no_grad():
        _, layers, _ = model(sp.unsqueeze(0).to(device), return_z=True)
    return [l.squeeze(0).cpu() for l in layers]


def cache_dataset(model, ds, n_events, depth):
    """Cache per-layer features + labels for n_events, ONE backbone pass each."""
    F_layers = [[] for _ in range(depth)]
    T_list = []
    for i in range(min(n_events, len(ds))):
        sp, seg, _ = ds[i]
        layers = extract_all_layers(model, sp)
        for li in range(depth):
            F_layers[li].append(layers[li])
        T_list.append(seg.long())
    return F_layers, T_list


def probe_one_layer(F_tr_layer, T_tr, F_te_layer, T_te_offset, embed_dim):
    """Train a contrastive linear map on ONE layer's cached features, return
    the same/diff distance ratio on held-out data."""
    lin = nn.Linear(embed_dim, PROJ_DIM, bias=False).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
    for ep in range(PROBE_EPOCHS):
        for f, t in zip(F_tr_layer, T_tr):
            f, t = f.to(device), t.to(device)
            e = lin(f); d = torch.cdist(e, e)
            eye = torch.eye(len(t), device=device)
            is_noise = (t == NOISE_LABEL)
            noise_pair = is_noise[:, None] | is_noise[None, :]
            same = (t[:, None] == t[None, :]).float() * (1 - eye) * (~noise_pair).float()
            diff = (1 - same - eye).clamp(min=0)
            pos = (same * d.pow(2)).sum() / same.sum().clamp(min=1)
            neg = (diff * torch.relu(MARGIN - d).pow(2)).sum() / diff.sum().clamp(min=1)
            loss = pos + neg; opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        PRJ = [lin(f.to(device)).cpu().numpy() for f in F_te_layer]
    PRJ = np.concatenate(PRJ)
    LAB = np.concatenate(T_te_offset)

    real = LAB != NOISE_LABEL
    PRJ, LAB = PRJ[real], LAB[real]
    D = np.linalg.norm(PRJ[:, None, :] - PRJ[None, :, :], axis=-1)
    n = len(LAB)
    same_mask = (LAB[:, None] == LAB[None, :]) & ~np.eye(n, dtype=bool)
    diff_mask = (LAB[:, None] != LAB[None, :])
    return D[same_mask].mean() / D[diff_mask].mean()


def run_all_layers(ckpt_path, embed_dim, label):
    model = load_model(ckpt_path, embed_dim)
    ds_tr = get_ds("pretrain")
    print(f"    caching train features ({TRAIN_EVENTS} events, all {DEPTH} layers)...")
    F_tr_layers, T_tr = cache_dataset(model, ds_tr, TRAIN_EVENTS, DEPTH)

    ds_te = get_ds("test")
    print(f"    caching test features ({N_TEST_EVENTS} events, all {DEPTH} layers)...")
    F_te_layers_raw, T_te_raw = cache_dataset(model, ds_te, N_TEST_EVENTS, DEPTH)
    T_te_offset = [np.where(t.numpy() == NOISE_LABEL, NOISE_LABEL, t.numpy() + 100 * k)
                   for k, t in enumerate(T_te_raw)]

    ratios = []
    for li in range(DEPTH):
        ratio = probe_one_layer(F_tr_layers[li], T_tr, F_te_layers_raw[li], T_te_offset, embed_dim)
        print(f"    layer {li:>2}: ratio = {ratio:.4f}")
        ratios.append(ratio)

    del model; torch.cuda.empty_cache()
    return ratios


def find_ckpt(sweep_dir, config_pattern, w, run_num):
    cfg = config_pattern.format(w=w)
    path = f"{sweep_dir}/{cfg}/{run_num}/training_checkpoints/ckpt_best.tar"
    return path if os.path.exists(path) else None


chinchilla_dir = find_chinchilla_sweep_dir()
print(f"Chinchilla sweep dir: {chinchilla_dir}\n")

all_results = {}
for w in WIDTHS:
    old_ckpt = find_ckpt(OLD_SWEEP_DIR, OLD_CONFIG_PATTERN, w, OLD_RUN_NUM)
    chin_ckpt = find_ckpt(chinchilla_dir, CHINCHILLA_CONFIG_PATTERN, w, CHINCHILLA_RUN_NUM)
    if old_ckpt is None or chin_ckpt is None:
        print(f"{M_LABEL.get(w,'w'+str(w))}: missing checkpoint(s), skipped")
        continue

    print(f"\n=== {M_LABEL[w]} (w{w}) -- OLD checkpoint ===")
    old_ratios = run_all_layers(old_ckpt, w, "old")

    print(f"\n=== {M_LABEL[w]} (w{w}) -- CHINCHILLA checkpoint ===")
    chin_ratios = run_all_layers(chin_ckpt, w, "chinchilla")

    all_results[w] = dict(old=old_ratios, chinchilla=chin_ratios)

print("\n" + "=" * 70)
for w, res in all_results.items():
    print(f"\n{M_LABEL[w]} (w{w}):")
    print(f"{'layer':>6} {'old':>10} {'chinchilla':>12}  better?")
    for li in range(DEPTH):
        o, c = res["old"][li], res["chinchilla"][li]
        print(f"{li:>6} {o:>10.4f} {c:>12.4f}  {'chinchilla' if c < o else 'old'}")

if all_results:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(all_results), figsize=(7 * len(all_results), 5.5), sharey=True)
    if len(all_results) == 1:
        axes = [axes]
    for ax, (w, res) in zip(axes, all_results.items()):
        layers = list(range(DEPTH))
        ax.plot(layers, res["old"], "o-", color="#2a9d8f", label="Old (flat 100k)")
        ax.plot(layers, res["chinchilla"], "o-", color="#e76f51", label="Chinchilla-corrected")
        ax.set_xlabel("Backbone layer")
        ax.set_title(f"{M_LABEL[w]} (w{w})")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("same/diff track distance ratio (lower = better)")
    fig.suptitle("Per-layer track-ID separation: old vs Chinchilla-corrected", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT_PLOT}")