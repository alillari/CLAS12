"""
Track-finding comparison: 4 pretrained backbones + AdapterOnly + coatjava,
now correctly matching the label-count-swept run structure (was previously
built for a stale single-run-per-width design).

Real path pattern, per run: {ROOT_DIR}/{cfg}/checkpoints/
    {cfg}_nerf_tracking_head_d{lc}_label{lc}.log
(limit_size AND run_num are both driven by the same labeled-event count lc).
AdapterOnly uses a fixed config name (no width suffix) -- confirmed from
run_table.csv's own row count that it never varied by width.

Produces two views from the same data:
  1. Fig7-style: ARI vs labeled events, one line per backbone, at a fixed
     representative width (m6).
  2. Fig6-style: ARI vs width, one line per backbone, at the largest label
     count (100,000).

Run: python3 dev_scripts/plot_track_finding_4way.py
"""
import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TREE = "/workspace/PP_collision"
ROOT_DIR = f"{TREE}/train/downstream/downstream_log"

WIDTHS = [64, 128, 256, 512, 1024, 1536]
LABEL_COUNTS = [100, 1000, 10000, 100000]
M_LABEL = {64: 'm1', 128: 'm2', 256: 'm3', 512: 'm4', 1024: 'm5', 1536: 'm6'}
FIXED_WIDTH_FOR_FIG7 = 1536   # m6
MAX_LABELS_FOR_FIG6 = 100000

COATJAVA_ARI = 0.9115  # fixed, non-ML, from run_table.csv

BACKBONES = [
    ("OLD (off-target contaminated)", "old", "#0072B2"),
    ("V6_02 (target-restricted)", "v602", "#E69F00"),
    ("V7_01 + aux length", "v701length", "#009E73"),
    ("V7_01 + aux pitch", "v701pitch", "#CC79A7"),
]


def pretrained_log_path(bname, w, lc):
    cfg = f"clas12_track_finding_pretrained_{bname}_w{w}"
    return f"{ROOT_DIR}/{cfg}/checkpoints/{cfg}_nerf_tracking_head_d{lc}_label{lc}.log"


def adapteronly_log_path(lc):
    cfg = "clas12_track_finding_adapteronly"
    return f"{ROOT_DIR}/{cfg}/checkpoints/{cfg}_nerf_tracking_head_d{lc}_label{lc}.log"


def best_ari_from_log(path):
    if not os.path.exists(path):
        return None
    best = None
    with open(path) as f:
        header = f.readline().strip().split("\t")
        if "ARI" not in header:
            return None
        ari_idx = header.index("ARI")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= ari_idx:
                continue
            try:
                val = float(parts[ari_idx])
            except ValueError:
                continue
            if best is None or val > best:
                best = val
    return best


print(f"Looking for logs under: {ROOT_DIR}\n")

data = {bname: {} for _, bname, _ in BACKBONES}
for label, bname, color in BACKBONES:
    for w in WIDTHS:
        for lc in LABEL_COUNTS:
            path = pretrained_log_path(bname, w, lc)
            ari = best_ari_from_log(path)
            if ari is not None:
                data[bname][(w, lc)] = ari
                print(f"  [{label}] {M_LABEL[w]} / {lc} labels: Best ARI={ari:.4f}")

adapteronly_data = {}
for lc in LABEL_COUNTS:
    path = adapteronly_log_path(lc)
    ari = best_ari_from_log(path)
    if ari is not None:
        adapteronly_data[lc] = ari
        print(f"  [AdapterOnly] {lc} labels: Best ARI={ari:.4f}")

if not any(data[bname] for _, bname, _ in BACKBONES) and not adapteronly_data:
    print("\nNo data found yet for ANY run -- nothing to plot.")
else:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    for label, bname, color in BACKBONES:
        xs, ys = [], []
        for lc in LABEL_COUNTS:
            if (FIXED_WIDTH_FOR_FIG7, lc) in data[bname]:
                xs.append(lc); ys.append(data[bname][(FIXED_WIDTH_FOR_FIG7, lc)])
        if xs:
            ax1.plot(xs, ys, "o-", color=color, label=label, lw=1.8)
    if adapteronly_data:
        xs = sorted(adapteronly_data.keys())
        ys = [adapteronly_data[lc] for lc in xs]
        ax1.plot(xs, ys, "k--", label="AdapterOnly", lw=1.8)
    ax1.axhline(COATJAVA_ARI, color="#888888", linestyle=":", lw=2, label="coatjava")
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of labeled events")
    ax1.set_ylabel("Best ARI")
    ax1.set_title(f"ARI vs. labeled data (width={M_LABEL[FIXED_WIDTH_FOR_FIG7]})")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)

    for label, bname, color in BACKBONES:
        xs, ys = [], []
        for w in WIDTHS:
            if (w, MAX_LABELS_FOR_FIG6) in data[bname]:
                xs.append(w); ys.append(data[bname][(w, MAX_LABELS_FOR_FIG6)])
        if xs:
            ax2.plot(xs, ys, "o-", color=color, label=label, lw=1.8)
    if MAX_LABELS_FOR_FIG6 in adapteronly_data:
        ax2.axhline(adapteronly_data[MAX_LABELS_FOR_FIG6], color="black", linestyle="--",
                   label=f"AdapterOnly (@ {MAX_LABELS_FOR_FIG6:,} labels)")
    ax2.axhline(COATJAVA_ARI, color="#888888", linestyle=":", lw=2, label="coatjava")
    ax2.set_xscale("log")
    ax2.set_xlabel("Model Width")
    ax2.set_ylabel("Best ARI")
    ax2.set_title(f"ARI vs. width (@ {MAX_LABELS_FOR_FIG6:,} labeled events)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out = f"{TREE}/fig_track_finding_4way.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")