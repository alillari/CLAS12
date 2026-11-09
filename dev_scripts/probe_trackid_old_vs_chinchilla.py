"""
Track-ID separation quality comparison: old (flat-100k) vs Chinchilla-corrected
checkpoints, across all widths completed so far (m1-m5; m6 still training).

Same methodology as the earlier viz_pca_both.py / check_confounded_pairs.py
work this session: freeze the backbone, train a small linear map with a
track-contrastive loss (noise excluded from positive pairs, kept in negative
pairs), measure the same-track / different-track mapped-distance ratio.
Lower ratio = better separation = backbone features are more track-informative.

This directly tests whether Chinchilla-corrected pretraining produces BETTER
downstream representations, not just a lower pretraining loss number.

Run: python3 dev_scripts/probe_trackid_old_vs_chinchilla.py
"""
import os, glob, sys, argparse
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
# [NEW] cross-based pretraining (mmap_v4, older TPC-cross representation) --
# added as a third series to give a preliminary read on how much track-ID
# information survives training under the OLDER cross representation,
# compared to both cluster sweeps. Different dataset AND reader_type from
# the cluster sweeps (ragged_npy, not ragged_mmap) -- see get_ds() below.
CROSS_SWEEP_DIR = "/workspace/PP_collision/checkpoints/LR_Full_Test_1_2026-07-20"
CROSS_RUN_NUM = "sweep"
CROSS_CONFIG_PATTERN = "scale_w{w}_d12_n5483352"
CROSS_DATA_ROOT = "/workspace/PP_collision/data/mmap_v4"

DEPTH = 12
WIDTHS = [64, 128, 256, 512, 1024, 1536]  # full sweep, m6 chinchilla now complete
M_LABEL = {64: 'm1', 128: 'm2', 256: 'm3', 512: 'm4', 1024: 'm5', 1536: 'm6'}

TRAIN_EVENTS = 3000
PROBE_EPOCHS = 10
PROJ_DIM = 32
MARGIN = 1.0
NOISE_LABEL = -1
N_TEST_EVENTS = 10
OUT_PLOT = "/workspace/PP_collision/fig_trackid_old_vs_chinchilla.png"
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


def get_feat_extractor(model):
    feat = {}
    handle = model.norm.register_forward_hook(lambda m, i, o: feat.__setitem__("x", o.detach()))
    def fm_feat(sp):
        with torch.no_grad():
            model(sp.unsqueeze(0).to(device))
        return feat["x"].squeeze(0).cpu()
    return fm_feat, handle


def get_ds(split, data_root="/workspace/PP_collision/data/mmap_cluster_full",
          reader_type="ragged_mmap"):
    return TPCBatchDataset(
        data_root=data_root,
        version="dataset_name", split=split, reader_type=reader_type,
        num_pred_points=1, group_size=8, normalize=True, nleave=1e6,
        chunk_training=False, train=(split == "pretrain"), order="REP")


def probe_same_diff_ratio(model, embed_dim, data_root="/workspace/PP_collision/data/mmap_cluster_full", reader_type="ragged_mmap"):
    fm_feat, handle = get_feat_extractor(model)
    try:
        ds_tr = get_ds("pretrain", data_root=data_root, reader_type=reader_type)
        F_tr, T_tr = [], []
        for i in range(min(TRAIN_EVENTS, len(ds_tr))):
            sp, seg, _ = ds_tr[i]
            F_tr.append(fm_feat(sp)); T_tr.append(seg.long())

        lin_trk = nn.Linear(embed_dim, PROJ_DIM, bias=False).to(device)
        opt = torch.optim.Adam(lin_trk.parameters(), lr=1e-3)
        for ep in range(PROBE_EPOCHS):
            for f, t in zip(F_tr, T_tr):
                f, t = f.to(device), t.to(device)
                e = lin_trk(f); d = torch.cdist(e, e)
                eye = torch.eye(len(t), device=device)
                is_noise = (t == NOISE_LABEL)
                noise_pair = is_noise[:, None] | is_noise[None, :]
                same = (t[:, None] == t[None, :]).float() * (1 - eye) * (~noise_pair).float()
                diff = (1 - same - eye).clamp(min=0)
                pos = (same * d.pow(2)).sum() / same.sum().clamp(min=1)
                neg = (diff * torch.relu(MARGIN - d).pow(2)).sum() / diff.sum().clamp(min=1)
                loss = pos + neg; opt.zero_grad(); loss.backward(); opt.step()

        ds_te = get_ds("test", data_root=data_root, reader_type=reader_type)
        PRJ, LAB = [], []
        for k in range(N_TEST_EVENTS):
            sp, seg, _ = ds_te[k]
            f = fm_feat(sp)
            with torch.no_grad():
                e = lin_trk(f.to(device)).cpu()
            seg_np = seg.long().numpy()
            offset = np.where(seg_np == NOISE_LABEL, NOISE_LABEL, seg_np + 100 * k)
            PRJ.append(e.numpy()); LAB.append(offset)
        PRJ = np.concatenate(PRJ); LAB = np.concatenate(LAB)

        real = LAB != NOISE_LABEL
        PRJ, LAB = PRJ[real], LAB[real]
        D = np.linalg.norm(PRJ[:, None, :] - PRJ[None, :, :], axis=-1)
        n = len(LAB)
        same_mask = (LAB[:, None] == LAB[None, :]) & ~np.eye(n, dtype=bool)
        diff_mask = (LAB[:, None] != LAB[None, :])
        same_mean = D[same_mask].mean()
        diff_mean = D[diff_mask].mean()
        return same_mean / diff_mean
    finally:
        handle.remove()


def find_ckpt(sweep_dir, config_pattern, w, run_num):
    cfg = config_pattern.format(w=w)
    path = f"{sweep_dir}/{cfg}/{run_num}/training_checkpoints/ckpt_best.tar"
    return path if os.path.exists(path) else None


chinchilla_dir = find_chinchilla_sweep_dir()
print(f"Chinchilla sweep dir: {chinchilla_dir}\n")

results = []
for w in WIDTHS:
    old_ckpt = find_ckpt(OLD_SWEEP_DIR, OLD_CONFIG_PATTERN, w, OLD_RUN_NUM)
    chin_ckpt = find_ckpt(chinchilla_dir, CHINCHILLA_CONFIG_PATTERN, w, CHINCHILLA_RUN_NUM)
    cross_ckpt = find_ckpt(CROSS_SWEEP_DIR, CROSS_CONFIG_PATTERN, w, CROSS_RUN_NUM)

    if old_ckpt is None or chin_ckpt is None:
        print(f"{M_LABEL[w]} (w{w}): missing cluster checkpoint(s) -- skipped "
              f"(old={'ok' if old_ckpt else 'MISSING'}, chinchilla={'ok' if chin_ckpt else 'MISSING'})")
        continue

    print(f"{M_LABEL[w]} (w{w}): probing OLD (cluster) checkpoint...")
    old_model = load_model(old_ckpt, w)
    old_ratio = probe_same_diff_ratio(old_model, w)
    del old_model; torch.cuda.empty_cache()

    print(f"{M_LABEL[w]} (w{w}): probing CHINCHILLA (cluster) checkpoint...")
    chin_model = load_model(chin_ckpt, w)
    chin_ratio = probe_same_diff_ratio(chin_model, w)
    del chin_model; torch.cuda.empty_cache()

    cross_ratio = None
    if cross_ckpt is not None:
        print(f"{M_LABEL[w]} (w{w}): probing CROSS (mmap_v4) checkpoint...")
        cross_model = load_model(cross_ckpt, w)
        cross_ratio = probe_same_diff_ratio(cross_model, w, data_root=CROSS_DATA_ROOT, reader_type="ragged_npy")
        del cross_model; torch.cuda.empty_cache()
    else:
        print(f"{M_LABEL[w]} (w{w}): no cross checkpoint found -- skipping that series for this width")

    cross_str = f"  cross={cross_ratio:.4f}" if cross_ratio is not None else "  cross=N/A"
    print(f"{M_LABEL[w]} (w{w}): old={old_ratio:.4f}  chinchilla={chin_ratio:.4f}{cross_str}\n")
    results.append(dict(width=w, m=M_LABEL[w], old=old_ratio, chinchilla=chin_ratio, cross=cross_ratio))

print("\n" + "=" * 80)
print(f"{'model':>6} {'old ratio':>12} {'chinchilla ratio':>18} {'cross ratio':>14}  {'best?':>10}")
for r in results:
    vals = {"old": r["old"], "chinchilla": r["chinchilla"]}
    if r["cross"] is not None:
        vals["cross"] = r["cross"]
    best = min(vals, key=vals.get)
    cross_disp = f"{r['cross']:>14.4f}" if r["cross"] is not None else f"{'N/A':>14}"
    print(f"{r['m']:>6} {r['old']:>12.4f} {r['chinchilla']:>18.4f} {cross_disp}  {best:>10}")

if results:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["m"] for r in results]
    old_vals = [r["old"] for r in results]
    chin_vals = [r["chinchilla"] for r in results]
    cross_vals = [r["cross"] for r in results]  # may contain None
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width, old_vals, width, label="Old (flat 100k)", color="#2a9d8f")
    ax.bar(x, chin_vals, width, label="Chinchilla-corrected", color="#e76f51")
    # only plot cross bars where data actually exists
    cross_x = [xi for xi, cv in zip(x, cross_vals) if cv is not None]
    cross_y = [cv for cv in cross_vals if cv is not None]
    if cross_y:
        ax.bar([xi + width for xi in cross_x], cross_y, width, label="Cross (mmap_v4, preliminary)", color="#8ab17d")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("same/diff track distance ratio (lower = better separation)")
    ax.set_title("Track-ID separation quality: cluster (old/Chinchilla) vs cross checkpoints")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    print(f"\nSaved: {OUT_PLOT}")