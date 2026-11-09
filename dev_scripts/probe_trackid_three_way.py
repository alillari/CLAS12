"""
Three-way track-ID separation comparison: OLD vs V6_02 vs V7_01+aux, with
disk caching keyed by each checkpoint's own mtime -- so already-probed,
finished checkpoints (OLD, V6_02) are instant on repeat runs, while a
checkpoint that's still updating (V7_01+aux, mid-training) automatically
gets re-probed only when it actually changes.

Cache file: trackid_probe_cache.json (in the working directory). Safe to
delete any time to force a full re-probe of everything.

Run: python3 dev_scripts/probe_trackid_three_way.py
Force re-probe everything (ignore cache): python3 dev_scripts/probe_trackid_three_way.py --no-cache
"""
import os, glob, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/workspace/PP_collision")
from fm4npp.models.mambagpt import Mamba1GPT
from fm4npp.datasets.dataset_pretrain import TPCBatchDataset

# ============================================================
DEPTH = 12
WIDTHS = [64, 128, 256, 512, 1024, 1536]
M_LABEL = {64: 'm1', 128: 'm2', 256: 'm3', 512: 'm4', 1024: 'm5', 1536: 'm6'}

TRAIN_EVENTS = 3000
PROBE_EPOCHS = 10
PROJ_DIM = 32
MARGIN = 1.0
NOISE_LABEL = -1
N_TEST_EVENTS = 10
OUT_PLOT = "/workspace/PP_collision/fig_trackid_three_way.png"
CACHE_FILE = "/workspace/PP_collision/trackid_probe_cache_learnedweight.json"

# Per-series config: (label, sweep_dir_or_None-for-autoglob, glob_pattern-if-autoglob,
# config_pattern, run_num, data_root, reader_type, use_aux_features, color)
SERIES = [
    dict(label="OLD (off-target contaminated)",
        sweep_dir="/workspace/PP_collision/checkpoints/Unbiased_Cluster_Full_Sweep_2026-08-13",
        sweep_glob=None, config_pattern="scale_w{w}_d12_n39553933", run_num="sweep",
        data_root="/workspace/PP_collision/data/mmap_cluster_full", reader_type="ragged_mmap",
        use_aux_features=False, color="#0072B2"),
    dict(label="V6_02 (target-restricted)",
        sweep_dir=None, sweep_glob="Canonical_Loose_v6_02_Full_Sweep_*",
        config_pattern="scale_w{w}_d12_v602", run_num="sweep",
        data_root="/workspace/PP_collision/mmap_canonical_loose_truthseg_event_v6_02", reader_type="ragged_mmap",
        use_aux_features=False, color="#E69F00"),
    dict(label="V7_01 + aux (sx,sy,sz,length)",
        sweep_dir=None, sweep_glob="v7_01_Aux_Full_Sweep_*",
        config_pattern="scale_w{w}_d12_v701aux", run_num="sweep",
        data_root="/workspace/PP_collision/data/mmap_canonical_loose_truthseg_event_v7_01", reader_type="ragged_mmap",
        use_aux_features=True, aux_extra_feature="length", color="#009E73"),
    dict(label="V7_01 + aux (sx,sy,sz,pitch)",
        sweep_dir=None, sweep_glob="v7_01_Pitch_Full_Sweep_*",
        config_pattern="scale_w{w}_d12_v701pitch", run_num="sweep",
        data_root="/workspace/PP_collision/data/mmap_canonical_loose_truthseg_event_v7_01", reader_type="ragged_mmap",
        use_aux_features=True, aux_extra_feature="pitch", color="#CC79A7"),
]
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def find_sweep_dir(root, pattern, label):
    candidates = sorted(glob.glob(f"{root}/{pattern}"))
    if not candidates:
        print(f"  [{label}] no matching sweep folder found -- skipped")
        return None
    return candidates[-1]


def find_ckpt(sweep_dir, config_pattern, w, run_num):
    cfg = config_pattern.format(w=w)
    path = f"{sweep_dir}/{cfg}/{run_num}/training_checkpoints/ckpt_best.tar"
    return path if os.path.exists(path) else None


def load_model(ckpt_path, embed_dim, use_aux_features):
    kwargs = dict(embed_dim=embed_dim, num_layers=DEPTH, d_state=16, klen=1,
                  dropout=0.1, pe_method='nerf')
    if use_aux_features:
        kwargs.update(embed_method='pos_plus_aux', pos_dim=3, aux_dim=4)
    else:
        kwargs.update(embed_method='pos_only')
    model = Mamba1GPT(**kwargs).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    st = ck.get("model_state", ck.get("model_state_dict", ck))
    model.load_state_dict({k.replace("module.", ""): v for k, v in st.items()}, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def extract_all_layers(model, sp, depth=DEPTH):
    """[FIX] was: a single forward hook on model.norm, final layer only.
    Every probe this session inherited that shortcut from the very first
    script without ever revisiting it, even as results stayed inconclusive
    across three separate interventions -- that inconsistency should have
    been the signal to check the measurement tool, not just the backbone.
    return_z=True gives all 12 layers (cumulative state, post-fix) in one
    forward pass, same as probe_trackid_per_layer.py."""
    with torch.no_grad():
        _, layers, _ = model(sp.unsqueeze(0).to(device), return_z=True)
    return [l.squeeze(0).cpu() for l in layers]


def get_ds(split, data_root, reader_type, use_aux_features, aux_extra_feature="length"):
    return TPCBatchDataset(
        data_root=data_root, version="dataset_name", split=split, reader_type=reader_type,
        num_pred_points=1, group_size=8, normalize=True, nleave=1e6,
        chunk_training=False, train=(split == "pretrain"), order="REP",
        use_aux_features=use_aux_features, aux_extra_feature=aux_extra_feature)


def probe_learned_combination(model, embed_dim, data_root, reader_type, use_aux_features, aux_extra_feature="length"):
    """[REVISED] Instead of training 12 SEPARATE per-layer probes (12x the
    cost), trains ONE probe jointly: a learned softmax-weighted combination
    across all 12 layers, feeding a single shared linear projection --
    directly mirroring the real adapter's own weighted_avg_weights
    mechanism, rather than testing "which single layer wins" (which isn't
    even how the actual adapter combines layers). Same feature-extraction
    cost as before (one return_z pass per event, all layers), but the
    TRAINING loop itself is the same size as the original single-layer
    probe -- the joint combination adds negligible cost.
    Returns (ratio, learned_layer_weights: list[12]).
    """
    ds_tr = get_ds("pretrain", data_root, reader_type, use_aux_features, aux_extra_feature)
    F_tr = []  # list of (12, embed_dim) tensors, one per event's mean-pooled layers
    T_tr = []
    for i in range(min(TRAIN_EVENTS, len(ds_tr))):
        sp, seg, _ = ds_tr[i]
        layers = extract_all_layers(model, sp)          # list of 12 (N_points, embed_dim)
        F_tr.append(torch.stack(layers, dim=0))          # (12, N_points, embed_dim)
        T_tr.append(seg.long())

    ds_te = get_ds("test", data_root, reader_type, use_aux_features, aux_extra_feature)
    F_te = []
    T_te_raw = []
    for k in range(N_TEST_EVENTS):
        sp, seg, _ = ds_te[k]
        layers = extract_all_layers(model, sp)
        F_te.append(torch.stack(layers, dim=0))
        T_te_raw.append(seg.long())
    T_te_offset = [np.where(t.numpy() == NOISE_LABEL, NOISE_LABEL, t.numpy() + 100 * k)
                   for k, t in enumerate(T_te_raw)]

    # learned components: layer_weights mirrors the real adapter's
    # weighted_avg_weights (raw scores -> softmax), lin is the shared
    # projection applied to the resulting combined feature.
    layer_weights = nn.Parameter(torch.zeros(DEPTH, device=device))
    lin = nn.Linear(embed_dim, PROJ_DIM, bias=False).to(device)
    opt = torch.optim.Adam(list(lin.parameters()) + [layer_weights], lr=1e-3)

    for ep in range(PROBE_EPOCHS):
        for f_layers, t in zip(F_tr, T_tr):  # f_layers: (12, N_points, embed_dim)
            f_layers, t = f_layers.to(device), t.to(device)
            w = torch.softmax(layer_weights, dim=0)               # (12,)
            combined = (f_layers * w.view(DEPTH, 1, 1)).sum(dim=0)  # (N_points, embed_dim)
            e = lin(combined); d = torch.cdist(e, e)
            eye = torch.eye(len(t), device=device)
            is_noise = (t == NOISE_LABEL)
            noise_pair = is_noise[:, None] | is_noise[None, :]
            same = (t[:, None] == t[None, :]).float() * (1 - eye) * (~noise_pair).float()
            diff = (1 - same - eye).clamp(min=0)
            pos = (same * d.pow(2)).sum() / same.sum().clamp(min=1)
            neg = (diff * torch.relu(MARGIN - d).pow(2)).sum() / diff.sum().clamp(min=1)
            loss = pos + neg; opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        w = torch.softmax(layer_weights, dim=0)
        PRJ = []
        for f_layers in F_te:
            f_layers = f_layers.to(device)
            combined = (f_layers * w.view(DEPTH, 1, 1)).sum(dim=0)
            PRJ.append(lin(combined).cpu().numpy())
    PRJ = np.concatenate(PRJ)
    LAB = np.concatenate(T_te_offset)
    real = LAB != NOISE_LABEL
    PRJ, LAB = PRJ[real], LAB[real]
    D = np.linalg.norm(PRJ[:, None, :] - PRJ[None, :, :], axis=-1)
    n = len(LAB)
    same_mask = (LAB[:, None] == LAB[None, :]) & ~np.eye(n, dtype=bool)
    diff_mask = (LAB[:, None] != LAB[None, :])
    ratio = float(D[same_mask].mean() / D[diff_mask].mean())
    final_weights = [float(x) for x in torch.softmax(layer_weights, dim=0).detach().cpu()]
    return ratio, final_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true", help="ignore cache, re-probe everything")
    args = parser.parse_args()

    cache = {} if args.no_cache else load_cache()

    # resolve sweep dirs
    for s in SERIES:
        if s["sweep_dir"] is None:
            s["sweep_dir"] = find_sweep_dir("/workspace/PP_collision/checkpoints", s["sweep_glob"], s["label"])

    results = {s["label"]: [] for s in SERIES}

    for s in SERIES:
        if s["sweep_dir"] is None:
            continue
        for w in WIDTHS:
            ckpt = find_ckpt(s["sweep_dir"], s["config_pattern"], w, s["run_num"])
            if ckpt is None:
                print(f"[{s['label']}] {M_LABEL[w]} (w{w}): checkpoint not found -- skipped")
                continue

            mtime = os.path.getmtime(ckpt)
            cache_key = f"{s['label']}|{w}"
            cached = cache.get(cache_key)

            if cached is not None and cached.get("mtime") == mtime and "layer_weights" in cached:
                ratio, layer_weights = cached["ratio"], cached["layer_weights"]
                top_layer = int(np.argmax(layer_weights))
                print(f"[{s['label']}] {M_LABEL[w]} (w{w}): CACHE HIT, ratio={ratio:.4f}, "
                      f"top-weighted layer={top_layer} (w={layer_weights[top_layer]:.3f})")
            else:
                print(f"[{s['label']}] {M_LABEL[w]} (w{w}): probing with learned layer combination "
                      f"(cache miss, checkpoint changed, or cache predates this version)...")
                model = load_model(ckpt, w, s["use_aux_features"])
                ratio, layer_weights = probe_learned_combination(
                    model, w, s["data_root"], s["reader_type"], s["use_aux_features"],
                    s.get("aux_extra_feature", "length"))
                del model; torch.cuda.empty_cache()
                cache[cache_key] = {"mtime": mtime, "ratio": ratio, "layer_weights": layer_weights}
                save_cache(cache)  # save after every probe, not just at the end -- don't lose work on a crash
                top_layer = int(np.argmax(layer_weights))
                print(f"[{s['label']}] {M_LABEL[w]} (w{w}): done, ratio={ratio:.4f}, "
                      f"top-weighted layer={top_layer} (w={layer_weights[top_layer]:.3f})")
                print(f"    layer weights: " + " ".join(f"{x:.3f}" for x in layer_weights))

            results[s["label"]].append(dict(width=w, ratio=ratio, layer_weights=layer_weights))

    print("\n" + "=" * 80)
    print(f"{'model':>6}", end="")
    for s in SERIES:
        print(f" {s['label'][:20]:>22}", end="")
    print()
    for w in WIDTHS:
        print(f"{M_LABEL[w]:>6}", end="")
        for s in SERIES:
            row = next((r for r in results[s["label"]] if r["width"] == w), None)
            val = f"{row['ratio']:.4f}" if row else "N/A"
            print(f" {val:>22}", end="")
        print()
    print()
    print("ratio computed from a LEARNED softmax-weighted combination of all 12 layers,")
    print("trained jointly -- mirrors the real adapter weighted_avg_weights mechanism")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(WIDTHS))
    n_series = len(SERIES)
    bar_width = 0.8 / n_series

    for i, s in enumerate(SERIES):
        vals = []
        for w in WIDTHS:
            row = next((r for r in results[s["label"]] if r["width"] == w), None)
            vals.append(row["ratio"] if row else np.nan)
        offset = (i - (n_series - 1) / 2) * bar_width
        ax.bar(x + offset, vals, bar_width, label=s["label"], color=s["color"])

    ax.set_xticks(x); ax.set_xticklabels([M_LABEL[w] for w in WIDTHS])
    ax.set_ylabel("same/diff track distance ratio (lower = better separation)")
    ax.set_title("Track-ID separation: OLD vs V6_02 vs V7_01+aux")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    print(f"\nSaved: {OUT_PLOT}")
    print(f"Cache: {CACHE_FILE}")


if __name__ == "__main__":
    main()