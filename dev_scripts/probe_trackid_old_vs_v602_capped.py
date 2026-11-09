"""
Track-ID separation quality comparison: old (off-target contaminated)
vs. v6_02 (target-restricted) checkpoints, both flat-100k sweeps, all
six widths. Same contrastive linear-probe methodology used throughout
this session (same/diff mapped-distance ratio; noise excluded from
positive pairs, kept in negative pairs). Lower ratio = better separation.

[UPDATED] Points-per-track are capped at MAX_POINTS_PER_TRACK before the
distance computation, to control for a confirmed structural difference in
track segmentation between the two datasets (old caps at 12 points/track,
v6_02 allows up to 23) that could otherwise inflate v6_02's same-track
distance independent of actual representation quality.

Run: python3 dev_scripts/probe_trackid_old_vs_v602_capped.py
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
OLD_DATA_ROOT = "/workspace/PP_collision/data/mmap_cluster_full"

V602_ROOT = "/workspace/PP_collision/checkpoints"
V602_RUN_NUM = "sweep"
V602_CONFIG_PATTERN = "scale_w{w}_d12_v602"
V602_DATA_ROOT = "/workspace/PP_collision/mmap_canonical_loose_truthseg_event_v6_02"

DEPTH = 12
WIDTHS = [64, 128, 256, 512, 1024, 1536]
M_LABEL = {64: 'm1', 128: 'm2', 256: 'm3', 512: 'm4', 1024: 'm5', 1536: 'm6'}

TRAIN_EVENTS = 3000
PROBE_EPOCHS = 10
PROJ_DIM = 32
MARGIN = 1.0
NOISE_LABEL = -1
N_TEST_EVENTS = 10
MAX_POINTS_PER_TRACK = 10  # cap for fair same/diff comparison -- see fix note below
OUT_PLOT = "/workspace/PP_collision/fig_trackid_old_vs_v602_capped.png"
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_v602_sweep_dir():
    candidates = sorted(glob.glob(f"{V602_ROOT}/Canonical_Loose_v6_02_Full_Sweep_*"))
    if not candidates:
        raise SystemExit("No Canonical_Loose_v6_02_Full_Sweep_* folder found.")
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


def get_ds(split, data_root):
    return TPCBatchDataset(
        data_root=data_root,
        version="dataset_name", split=split, reader_type="ragged_mmap",
        num_pred_points=1, group_size=8, normalize=True, nleave=1e6,
        chunk_training=False, train=(split == "pretrain"), order="REP")


def probe_same_diff_ratio(model, embed_dim, data_root):
    fm_feat, handle = get_feat_extractor(model)
    try:
        ds_tr = get_ds("pretrain", data_root)
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

        ds_te = get_ds("test", data_root)
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

        # [FIX] Cap points-per-track before computing distances. Datasets
        # with different track-segmentation rules can have very different
        # max track sizes (confirmed: old dataset caps at 12 points/track,
        # v6_02 allows up to 23) -- a longer track naturally spans more
        # physical distance, inflating same-track distance for that
        # dataset regardless of actual representation quality. Randomly
        # subsampling every track down to the same fixed cap, for BOTH
        # datasets, removes this confound from the reported metric without
        # touching model input or probe training at all.
        rng = np.random.RandomState(42)
        keep_idx = []
        for track_id in np.unique(LAB):
            idxs = np.where(LAB == track_id)[0]
            if len(idxs) > MAX_POINTS_PER_TRACK:
                idxs = rng.choice(idxs, size=MAX_POINTS_PER_TRACK, replace=False)
            keep_idx.append(idxs)
        keep_idx = np.concatenate(keep_idx)
        PRJ, LAB = PRJ[keep_idx], LAB[keep_idx]

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


v602_dir = find_v602_sweep_dir()
print(f"v6_02 sweep dir: {v602_dir}\n")

results = []
for w in WIDTHS:
    old_ckpt = find_ckpt(OLD_SWEEP_DIR, OLD_CONFIG_PATTERN, w, OLD_RUN_NUM)
    v602_ckpt = find_ckpt(v602_dir, V602_CONFIG_PATTERN, w, V602_RUN_NUM)

    if old_ckpt is None or v602_ckpt is None:
        print(f"{M_LABEL[w]} (w{w}): missing checkpoint(s) -- skipped "
              f"(old={'ok' if old_ckpt else 'MISSING'}, v6_02={'ok' if v602_ckpt else 'MISSING'})")
        continue

    print(f"{M_LABEL[w]} (w{w}): probing OLD checkpoint...")
    old_model = load_model(old_ckpt, w)
    old_ratio = probe_same_diff_ratio(old_model, w, OLD_DATA_ROOT)
    del old_model; torch.cuda.empty_cache()

    print(f"{M_LABEL[w]} (w{w}): probing V6_02 checkpoint...")
    v602_model = load_model(v602_ckpt, w)
    v602_ratio = probe_same_diff_ratio(v602_model, w, V602_DATA_ROOT)
    del v602_model; torch.cuda.empty_cache()

    print(f"{M_LABEL[w]} (w{w}): old={old_ratio:.4f}  v6_02={v602_ratio:.4f}  "
          f"({'v6_02 better' if v602_ratio < old_ratio else 'old better'})\n")
    results.append(dict(width=w, m=M_LABEL[w], old=old_ratio, v602=v602_ratio))

print("\n" + "=" * 70)
print(f"{'model':>6} {'old ratio':>12} {'v6_02 ratio':>14}  {'better?':>10}")
for r in results:
    better = "v6_02" if r["v602"] < r["old"] else "old"
    print(f"{r['m']:>6} {r['old']:>12.4f} {r['v602']:>14.4f}  {better:>10}")

if results:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["m"] for r in results]
    old_vals = [r["old"] for r in results]
    v602_vals = [r["v602"] for r in results]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(x - width/2, old_vals, width, label="Old (off-target contaminated)", color="#2a9d8f")
    ax.bar(x + width/2, v602_vals, width, label="v6_02 (target-restricted)", color="#e76f51")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("same/diff track distance ratio (lower = better separation)")
    ax.set_title(f"Track-ID separation (points/track capped at {MAX_POINTS_PER_TRACK}): old vs v6_02")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    print(f"\nSaved: {OUT_PLOT}")