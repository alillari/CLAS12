"""
Pitch vs pitch+sincos-loss track-finding comparison -- single plot, ARI vs
labeled events, one line per (backbone, width) combination (6 total: 2
backbones x 3 widths). Auto-discovers whatever data actually exists on
disk -- add more widths to WIDTHS later and just rerun those specific new
combinations; nothing already on disk needs rerunning.

Run: python3 dev_scripts/plot_track_finding_pitch_sincos.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TREE = "/workspace/PP_collision"
ROOT_DIR = f"{TREE}/train/downstream/downstream_log"

WIDTHS = [64, 512, 1536]
LABEL_COUNTS = [100, 1000, 10000, 100000]
M_LABEL = {64: 'm1', 128: 'm2', 256: 'm3', 512: 'm4', 1024: 'm5', 1536: 'm6'}

COATJAVA_ARI = 0.9115  # fixed, non-ML, from run_table.csv

BACKBONES = [
    ("V7_01 + pitch", "v701pitch", "#CC79A7"),
    ("V7_01 + pitch + sincos loss", "v701pitchsincos", "#0072B2"),
]
LINESTYLES = {64: "-", 512: "--", 1536: ":"}
MARKERS = {64: "o", 512: "s", 1536: "^"}


def log_path(bname, w, lc):
    cfg = f"clas12_track_finding_pretrained_{bname}_w{w}"
    return f"{ROOT_DIR}/{cfg}/checkpoints/{cfg}_nerf_tracking_head_d{lc}_label{lc}.log"


def best_ari_from_log(path):
    # [FIX] was: parsing "ARI", but the trainer's own "Best ARI" printout
    # tracks ARI_2 (confirmed directly from track_finding_trainer.py:
    # checkpoint selection uses avg_ari_2, not avg_ari). Match what the
    # trainer itself considers "best".
    if not os.path.exists(path):
        return None
    best = None
    with open(path) as f:
        header = f.readline().strip().split("\t")
        if "ARI_2" not in header:
            return None
        ari_idx = header.index("ARI_2")
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

fig, ax = plt.subplots(figsize=(9.5, 7))
any_data = False

for label, bname, color in BACKBONES:
    for w in WIDTHS:
        xs, ys = [], []
        for lc in LABEL_COUNTS:
            ari = best_ari_from_log(log_path(bname, w, lc))
            if ari is not None:
                xs.append(lc); ys.append(ari)
                print(f"  [{label}] {M_LABEL.get(w, f'w{w}')} / {lc} labels: Best ARI={ari:.4f}")
        if xs:
            any_data = True
            ax.plot(xs, ys, linestyle=LINESTYLES[w], marker=MARKERS[w], color=color,
                    label=f"{label} ({M_LABEL.get(w, f'w{w}')})", lw=1.8, markersize=6)

if not any_data:
    print("\nNo data found yet for ANY run -- nothing to plot.")
else:
    ax.axhline(COATJAVA_ARI, color="#888888", linestyle=(0, (1, 1)), lw=2, label="coatjava")
    ax.set_xscale("log")
    ax.set_xlabel("Number of labeled events")
    ax.set_ylabel("Best ARI")
    ax.set_title("Track-finding: pitch vs pitch+sincos loss, all widths")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    out = f"{TREE}/fig_track_finding_pitch_sincos.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")