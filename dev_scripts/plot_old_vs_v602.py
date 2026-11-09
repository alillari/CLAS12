"""
Model scaling comparison: original sweep (Unbiased_Cluster_Full_Sweep,
old dataset with off-target contamination) vs. the new v6_02 sweep
(target-restricted dataset). Both flat 100k steps -- k=1 only.

Handles in-progress runs: hollow circle = width hasn't reached 100k yet;
filled = complete. Widths with no data at all yet are skipped, not
crashed on.

Run: python3 dev_scripts/plot_old_vs_v602.py
"""
import os, glob, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace/PP_collision")
from fm4npp.models.mambagpt import Mamba1GPT

# ============================================================
OLD_SWEEP_DIR = "/workspace/PP_collision/checkpoints/Unbiased_Cluster_Full_Sweep_2026-08-13"
OLD_RUN_NUM = "sweep"
OLD_CONFIG_PATTERN = "scale_w{w}_d12_n39553933"

V602_ROOT = "/workspace/PP_collision/checkpoints"
V602_RUN_NUM = "sweep"
V602_CONFIG_PATTERN = "scale_w{w}_d12_v602"

DEPTH = 12
TOTAL_STEPS = 100_000  # flat, same for every width, both sweeps
WIDTHS = [64, 128, 256, 512, 1024, 1536]
M_LABEL = {64: 'm1', 128: 'm2', 256: 'm3', 512: 'm4', 1024: 'm5', 1536: 'm6'}
FIT_COMPLETED_ONLY = True
OUT = "/workspace/PP_collision/fig_old_vs_v602.png"
# ============================================================


def find_v602_sweep_dir():
    candidates = sorted(glob.glob(f"{V602_ROOT}/Canonical_Loose_v6_02_Full_Sweep_*"))
    if not candidates:
        raise SystemExit("No Canonical_Loose_v6_02_Full_Sweep_* folder found.")
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


def collect(sweep_dir, config_pattern, run_num, label):
    rows = []
    for w in WIDTHS:
        cfg = config_pattern.format(w=w)
        loss, last_step = read_run(sweep_dir, cfg, run_num)
        if loss is None:
            print(f"  [{label}] w{w}: not started yet -- skipped")
            continue
        done = last_step is not None and last_step >= TOTAL_STEPS
        rows.append(dict(width=w, loss=loss, last_step=last_step, total=TOTAL_STEPS,
                         done=done, params=param_count(w, DEPTH)))
    return rows


v602_dir = find_v602_sweep_dir()
print(f"v6_02 sweep dir: {v602_dir}\n")

rows_old = collect(OLD_SWEEP_DIR, OLD_CONFIG_PATTERN, OLD_RUN_NUM, "OLD")
rows_v602 = collect(v602_dir, V602_CONFIG_PATTERN, V602_RUN_NUM, "V6_02")

for label, rows in [("OLD (off-target contaminated)", rows_old), ("V6_02 (target-restricted)", rows_v602)]:
    rows.sort(key=lambda r: r["width"])
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


fig, ax = plt.subplots(figsize=(8, 6.5))
fit_and_plot(ax, rows_old, "#2a9d8f", "Old (off-target contaminated)")
fit_and_plot(ax, rows_v602, "#e76f51", "v6_02 (target-restricted)")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Model Parameters (M)")
ax.set_ylabel("MSE (L)")
ax.set_title("Model scaling: old (off-target contaminated) vs. v6_02 (target-restricted), flat 100k steps")
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=8, loc="best")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"\nSaved: {OUT}")