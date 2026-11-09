"""
Model scaling plot, Chinchilla-corrected sweep only. Same style/conventions as
the other scaling plots (plot_old_vs_v602.py etc.): hollow circle = width hasn't
reached its step budget yet, filled = complete; power-law fit on completed points.

Run:            python3 dev_scripts/plot_chinchilla_only.py
Preview only:   python3 dev_scripts/plot_chinchilla_only.py --recorded
                (recorded final val losses + param counts, no torch/CSVs needed)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

USE_RECORDED = "--recorded" in sys.argv

# ============================================================
SWEEP_DIR = "/workspace/PP_collision/checkpoints/Chinchilla_Sweep_2026-08-28"
CONFIG_PATTERN = "scale_w{w}_d12_chinchilla"
RUN_NUM = "sweep"
DEPTH = 12
WIDTHS = [64, 128, 256, 512, 1024, 1536]
M_LABEL = {64: 'm1', 128: 'm2', 256: 'm3', 512: 'm4', 1024: 'm5', 1536: 'm6'}
# Chinchilla per-width step budgets
TOTAL_STEPS = {64: 100_000, 128: 100_000, 256: 100_000,
               512: 200_000, 1024: 800_000, 1536: 1_600_000}
FIT_COMPLETED_ONLY = True
COLOR = "#e76f51"
OUT = "/workspace/PP_collision/fig_chinchilla_only.png"

# only used with --recorded
REC_PARAMS = {64: 396_739, 128: 1_407_875, 256: 5_273_347,
              512: 20_377_091, 1024: 80_075_779, 1536: 179_096_067}
REC_LOSS = {64: 0.000994, 128: 0.000761, 256: 0.000663,
            512: 0.000573, 1024: 0.000533, 1536: 0.000464}
# ============================================================


def param_count(width, depth, klen=1):
    if USE_RECORDED:
        return REC_PARAMS[width]
    sys.path.insert(0, "/workspace/PP_collision")
    from fm4npp.models.mambagpt import Mamba1GPT
    m = Mamba1GPT(embed_dim=width, num_layers=depth, d_state=16, klen=klen,
                  dropout=0.1, embed_method='pos_only', pe_method='nerf')
    return sum(p.numel() for p in m.parameters())


def read_run(run_dir, config_name, run_num):
    csv = f"{run_dir}/{config_name}/{run_num}/config_{config_name}_run_{run_num}.csv"
    if not os.path.exists(csv):
        return None, None
    import pandas as pd
    df = pd.read_csv(csv, header=None, names=["split", "step", "loss", "lr"])
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df = df.dropna(subset=["step", "loss"])
    v = df[df["split"] == "val"].sort_values("step")
    if len(v) == 0:
        return None, None
    return float(v["loss"].min()), int(v["step"].iloc[-1])


rows = []
for w in WIDTHS:
    if USE_RECORDED:
        loss, last_step = REC_LOSS[w], TOTAL_STEPS[w]
    else:
        loss, last_step = read_run(SWEEP_DIR, CONFIG_PATTERN.format(w=w), RUN_NUM)
    if loss is None:
        print(f"  w{w}: not started yet -- skipped")
        continue
    done = last_step is not None and last_step >= TOTAL_STEPS[w]
    rows.append(dict(width=w, loss=loss, last_step=last_step, total=TOTAL_STEPS[w],
                     done=done, params=param_count(w, DEPTH)))
rows.sort(key=lambda r: r["width"])

print(f"{'width':>6} {'m':>3} {'params':>12} {'val loss':>12} {'steps':>18}  status")
for r in rows:
    step_s = f"{r['last_step']:,}/{r['total']:,}" if r["last_step"] else "-"
    print(f"{r['width']:>6} {M_LABEL[r['width']]:>3} {r['params']:>12,} {r['loss']:>12.6g} "
          f"{step_s:>18}  {'complete' if r['done'] else 'IN PROGRESS'}")


def fit_and_plot(ax, rows, color, label):
    fit_rows = [r for r in rows if r["done"]] if FIT_COMPLETED_ONLY else rows
    if len(fit_rows) < 2:
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
        ax.annotate(M_LABEL[r["width"]], (r["params"], r["loss"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8, color=color)

    if fit_rows and len(fit_rows) >= 2:
        P = np.array([r["params"] for r in fit_rows], float)
        L = np.array([r["loss"] for r in fit_rows], float)
        b, c = np.polyfit(np.log10(P), np.log10(L), 1)
        a = 10 ** (-c / b)
        allP = np.array([r["params"] for r in rows], float)
        fx = np.logspace(np.log10(allP.min()) * 0.98, np.log10(allP.max()) * 1.02, 200)
        ax.plot(fx, (fx / a) ** b, "--", color=color, lw=1.4,
                label=fr"{label}: $L=(M/{a:.3g})^{{{b:.3f}}}$")
        print(f"{label} fit (on {len(fit_rows)} completed points): L = (M / {a:.4g})^{b:.4f}")


fig, ax = plt.subplots(figsize=(8, 6.5))
fit_and_plot(ax, rows, COLOR, "Chinchilla-corrected")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Model Parameters (M)")
ax.set_ylabel("MSE (L)")
ax.set_title("Model scaling: Chinchilla-corrected sweep (per-width step budgets)")
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=8, loc="best")
fig.tight_layout()

out = "/mnt/user-data/outputs/fig_chinchilla_only.png" if USE_RECORDED else OUT
fig.savefig(out, dpi=150)
print(f"\nSaved: {out}")