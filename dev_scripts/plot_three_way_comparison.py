"""
Three-way model scaling comparison, all flat 100k steps:
  1. OLD    -- off-target contaminated dataset, position-only input
  2. V602   -- target-restricted dataset, position-only input
  3. V701AUX -- target-restricted dataset + aux features (sx,sy,sz,length)

Handles in-progress runs: hollow circle = width hasn't reached 100k yet;
filled = complete. Widths with no data at all yet are skipped, not
crashed on. Auto-finds the v602/v701aux sweep folders by glob.

Run: python3 dev_scripts/plot_three_way_comparison.py
"""
import os, glob, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace/PP_collision")
from fm4npp.models.mambagpt import Mamba1GPT

# ============================================================
OLD_SWEEP_DIR = "/workspace/PP_collision/checkpoints/Unbiased_Cluster_Full_Sweep_2026-08-13"
OLD_CONFIG_PATTERN = "scale_w{w}_d12_n39553933"

V602_ROOT = "/workspace/PP_collision/checkpoints"
V602_GLOB = "Canonical_Loose_v6_02_Full_Sweep_*"
V602_CONFIG_PATTERN = "scale_w{w}_d12_v602"

V701AUX_ROOT = "/workspace/PP_collision/checkpoints"
V701AUX_GLOB = "v7_01_Aux_Full_Sweep_*"
V701AUX_CONFIG_PATTERN = "scale_w{w}_d12_v701aux"

RUN_NUM = "sweep"
DEPTH = 12
TOTAL_STEPS = 100_000  # flat, same for every width, all three sweeps
WIDTHS = [64, 128, 256, 512, 1024, 1536]
M_LABEL = {64: 'm1', 128: 'm2', 256: 'm3', 512: 'm4', 1024: 'm5', 1536: 'm6'}
FIT_COMPLETED_ONLY = True
OUT = "/workspace/PP_collision/fig_three_way_comparison.png"

# [ERROR BARS -- opt-in via --error-bars] Unscientific, purely visual: shows
# min/max TRAIN loss (not val -- val is only logged every 1000 steps in
# these sweeps, too coarse; train is logged every step) sampled at 100-step
# increments over the last ERROR_BAR_WINDOW steps of each run. Meant as a
# rough "how much is loss still wobbling near the end" indicator, not a
# real confidence interval -- anchored visually at the already-plotted val
# loss point, even though it's a different metric.
ERROR_BAR_WINDOW = 10_000
ERROR_BAR_STRIDE = 100

# Okabe-Ito colorblind-safe palette -- chosen specifically to be maximally
# distinguishable from each other (unlike the previous teal/orange/navy set,
# where two colors were too close in the same family).
SERIES = [
    ("OLD (off-target contaminated)", OLD_SWEEP_DIR, OLD_CONFIG_PATTERN, "#0072B2"),   # blue
    ("V6_02 (target-restricted)", None, V602_CONFIG_PATTERN, "#E69F00"),               # orange
    ("V7_01 + aux (sx,sy,sz,length)", None, V701AUX_CONFIG_PATTERN, "#009E73"),        # green
]
# ============================================================


def find_sweep_dir(root, pattern, label):
    candidates = sorted(glob.glob(f"{root}/{pattern}"))
    if not candidates:
        print(f"  [{label}] no matching sweep folder found ({pattern}) -- series will be empty")
        return None
    return candidates[-1]


def param_count(width, depth, klen=1):
    m = Mamba1GPT(embed_dim=width, num_layers=depth, d_state=16, klen=klen,
                  dropout=0.1, embed_method='pos_only', pe_method='nerf')
    return sum(p.numel() for p in m.parameters())


def read_run(run_dir, config_name, run_num):
    csv = f"{run_dir}/{config_name}/{run_num}/config_{config_name}_run_{run_num}.csv"
    if not os.path.exists(csv):
        return None, None
    df = pd.read_csv(csv, header=None, names=["split", "step", "loss", "lr"])
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df = df.dropna(subset=["step", "loss"])
    v = df[df["split"] == "val"].sort_values("step")
    if len(v) == 0:
        return None, None
    return float(v["loss"].min()), int(v["step"].iloc[-1])


def train_loss_minmax_last_n(run_dir, config_name, run_num, last_step,
                             window=ERROR_BAR_WINDOW, stride=ERROR_BAR_STRIDE):
    """Min/max TRAIN loss, sampled every `stride` steps, over the last
    `window` steps up to `last_step`. Returns (min, max) or (None, None) if
    unavailable. Purely for the opt-in --error-bars visual indicator."""
    csv = f"{run_dir}/{config_name}/{run_num}/config_{config_name}_run_{run_num}.csv"
    if not os.path.exists(csv):
        return None, None
    df = pd.read_csv(csv, header=None, names=["split", "step", "loss", "lr"])
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df = df.dropna(subset=["step", "loss"])
    t = df[df["split"] == "train"].sort_values("step")
    if len(t) == 0:
        return None, None
    window_start = max(0, last_step - window)
    t_window = t[(t["step"] >= window_start) & (t["step"] <= last_step)]
    if len(t_window) == 0:
        return None, None
    # sample every `stride` steps within the window (train is logged every
    # step, so this just subsamples for the requested granularity)
    t_sampled = t_window[t_window["step"] % stride == 0]
    if len(t_sampled) == 0:
        t_sampled = t_window  # fallback: window too short for the stride, use everything available
    return float(t_sampled["loss"].min()), float(t_sampled["loss"].max())


def collect(sweep_dir, config_pattern, run_num, label, error_bars=False):
    if sweep_dir is None:
        return []
    rows = []
    for w in WIDTHS:
        cfg = config_pattern.format(w=w)
        loss, last_step = read_run(sweep_dir, cfg, run_num)
        if loss is None:
            print(f"  [{label}] w{w}: not started yet -- skipped")
            continue
        done = last_step is not None and last_step >= TOTAL_STEPS
        row = dict(width=w, loss=loss, last_step=last_step, total=TOTAL_STEPS,
                  done=done, params=param_count(w, DEPTH))
        if error_bars:
            tmin, tmax = train_loss_minmax_last_n(sweep_dir, cfg, run_num, last_step)
            row["train_min"] = tmin
            row["train_max"] = tmax
        rows.append(row)
    return rows


parser = argparse.ArgumentParser()
parser.add_argument("--error-bars", action="store_true",
                    help="show unscientific error bars: min/max TRAIN loss "
                         f"(sampled every {ERROR_BAR_STRIDE} steps) over the last "
                         f"{ERROR_BAR_WINDOW} steps of each run. Off by default.")
args = parser.parse_args()

# resolve the auto-found sweep dirs
resolved_series = []
for label, sweep_dir, config_pattern, color in SERIES:
    if sweep_dir is None:
        if "V602" in label or "V6_02" in label:
            sweep_dir = find_sweep_dir(V602_ROOT, V602_GLOB, label)
        else:
            sweep_dir = find_sweep_dir(V701AUX_ROOT, V701AUX_GLOB, label)
    resolved_series.append((label, sweep_dir, config_pattern, color))
    print(f"{label}: {sweep_dir}")

print()
all_rows = {}
for label, sweep_dir, config_pattern, color in resolved_series:
    rows = collect(sweep_dir, config_pattern, RUN_NUM, label, error_bars=args.error_bars)
    rows.sort(key=lambda r: r["width"])
    all_rows[label] = rows
    print(f"\n{label}:")
    print(f"{'width':>6} {'m':>3} {'params':>12} {'val loss':>12} {'steps':>18}  status")
    for r in rows:
        step_s = f"{r['last_step']:,}/{r['total']:,}" if r["last_step"] else "-"
        print(f"{r['width']:>6} {M_LABEL.get(r['width'],'?'):>3} {r['params']:>12,} "
              f"{r['loss']:>12.6g} {step_s:>18}  {'complete' if r['done'] else 'IN PROGRESS'}")


def fit_and_plot(ax, rows, color, label):
    if not rows:
        print(f"\n{label}: no data yet, skipping.")
        return
    fit_rows = [r for r in rows if r["done"]] if FIT_COMPLETED_ONLY else rows
    if len(fit_rows) < 2:
        print(f"\n{label}: not enough completed runs to fit ({len(fit_rows)} done).")
        fit_rows = rows if len(rows) >= 2 else None

    done = [r for r in rows if r["done"]]
    prog = [r for r in rows if not r["done"]]
    if done:
        ax.scatter([r["params"] for r in done], [r["loss"] for r in done],
                   s=75, color=color, zorder=5, edgecolors="black",
                   linewidths=0.7, label=f"{label} (completed)")
    if prog:
        ax.scatter([r["params"] for r in prog], [r["loss"] for r in prog],
                   s=90, facecolors="none", edgecolors=color,
                   linewidths=2.0, zorder=5, label=f"{label} (in progress)")

    # [ERROR BARS -- opt-in] unscientific, purely visual: min/max TRAIN loss
    # near the end of training, anchored at the plotted VAL loss point even
    # though it's a different metric. Clipped so yerr is never negative
    # (val and train can differ enough that the point falls outside the
    # train min/max range).
    rows_with_bars = [r for r in rows if r.get("train_min") is not None]
    if rows_with_bars:
        xs = [r["params"] for r in rows_with_bars]
        ys = [r["loss"] for r in rows_with_bars]
        lower = [max(0.0, r["loss"] - r["train_min"]) for r in rows_with_bars]
        upper = [max(0.0, r["train_max"] - r["loss"]) for r in rows_with_bars]
        ax.errorbar(xs, ys, yerr=[lower, upper], fmt="none", ecolor=color,
                    elinewidth=1.0, capsize=3, alpha=0.6, zorder=4)

    for r in rows:
        ax.annotate(M_LABEL.get(r["width"], f"w{r['width']}"),
                    (r["params"], r["loss"]), textcoords="offset points",
                    xytext=(6, 6), fontsize=8, color=color)

    if fit_rows and len(fit_rows) >= 2:
        P = np.array([r["params"] for r in fit_rows], float)
        L = np.array([r["loss"] for r in fit_rows], float)
        b, c = np.polyfit(np.log10(P), np.log10(L), 1)
        a = 10 ** (-c / b)
        allP = np.array([r["params"] for r in rows], float)
        fx = np.logspace(np.log10(allP.min()) * 0.98, np.log10(allP.max()) * 1.02, 200)
        fy = (fx / a) ** b
        ax.plot(fx, fy, "--", color=color, lw=1.4,
                label=fr"{label}: $L=(M/{a:.3g})^{{{b:.3f}}}$")
        print(f"{label} fit (on {len(fit_rows)} completed points): L = (M / {a:.4g})^{b:.4f}")


fig, ax = plt.subplots(figsize=(9.5, 7))
for label, sweep_dir, config_pattern, color in resolved_series:
    fit_and_plot(ax, all_rows[label], color, label)

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Model Parameters (M)")
ax.set_ylabel("MSE (L)")
title = "Model scaling: OLD vs. V6_02 vs. V7_01+aux, all flat 100k steps"
if args.error_bars:
    title += f"\n(error bars: train loss min/max, last {ERROR_BAR_WINDOW:,} steps, every {ERROR_BAR_STRIDE} -- not a real CI)"
ax.set_title(title, fontsize=10)
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=7.5, loc="best")
fig.tight_layout()
out_path = OUT.replace(".png", "_errorbars.png") if args.error_bars else OUT
fig.savefig(out_path, dpi=150)
print(f"\nSaved: {out_path}")