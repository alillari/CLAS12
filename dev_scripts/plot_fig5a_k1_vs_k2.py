"""
Figure 5(a) style plot comparing kNNN=1 vs kNNN=2: validation MSE vs model
parameters (log-log), both on the unbiased cluster-point dataset, m1-m6,
proper muP HP transfer for each. Hollow markers = run not yet complete.

Run: python3 dev_scripts/plot_fig5a_k1_vs_k2.py     (from /workspace/PP_collision)
"""
import os, glob, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace/PP_collision")
from fm4npp.models.mambagpt import Mamba1GPT

# ============================================================
SWEEP_K1 = "/workspace/PP_collision/checkpoints/Unbiased_Cluster_Full_Sweep_2026-08-13"
SWEEP_K2 = "/workspace/PP_collision/checkpoints/Unbiased_Cluster_k2_Full_Sweep_2026-08-20"
RUN_NUM  = "sweep"
DEPTH    = 12
FIT_COMPLETED_ONLY = True
OUT = "/workspace/PP_collision/fig5a_k1_vs_k2.png"
# ============================================================

M_LABEL = {64:'m1', 128:'m2', 256:'m3', 512:'m4', 1024:'m5', 1536:'m6'}


def read_run(run_dir):
    name = os.path.basename(run_dir)
    csv = f"{run_dir}/{RUN_NUM}/config_{name}_run_{RUN_NUM}.csv"
    if not os.path.exists(csv):
        return None, None
    df = pd.read_csv(csv, header=None, names=["split", "step", "loss", "lr"])
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df = df.dropna(subset=["step", "loss"])
    v = df[df["split"] == "val"].sort_values("step")
    if len(v) == 0:
        return None, None
    return float(v["loss"].iloc[-1]), int(v["step"].iloc[-1])


def total_steps_from_yaml(sweep_dir):
    y = f"{sweep_dir}/_grid_generated.yaml"
    if not os.path.exists(y):
        return None
    m = re.findall(r"total_steps:\s*(\d+)", open(y).read())
    return int(m[-1]) if m else None  # last match = sweep block, not *default anchor


def param_count(width, depth, klen):
    m = Mamba1GPT(embed_dim=width, num_layers=depth, d_state=16, klen=klen,
                  dropout=0.1, embed_method='pos_only', pe_method='nerf')
    return sum(p.numel() for p in m.parameters())


def collect(sweep_dir, klen):
    rows = []
    pat = re.compile(rf"scale_w(\d+)_d{DEPTH}_n(\d+)$")
    tot = total_steps_from_yaml(sweep_dir)
    for d in sorted(glob.glob(f"{sweep_dir}/scale_w*_d{DEPTH}_n*")):
        m = pat.match(os.path.basename(d))
        if not m:
            continue
        w = int(m.group(1))
        loss, last_step = read_run(d)
        if loss is None:
            print(f"  [k={klen}] w{w}: no val data -- skipped")
            continue
        done = (tot is not None and last_step is not None and last_step >= tot)
        params = param_count(w, DEPTH, klen)
        rows.append(dict(width=w, loss=loss, last_step=last_step, total=tot,
                         done=done, params=params))
    return rows


rows_k1 = collect(SWEEP_K1, klen=1)
rows_k2 = collect(SWEEP_K2, klen=2)

for label, rows in [("k=1", rows_k1), ("k=2", rows_k2)]:
    rows.sort(key=lambda r: r["width"])
    print(f"\n{label}:")
    print(f"{'width':>6} {'m':>3} {'params':>12} {'val loss':>12} {'steps':>16}  status")
    for r in rows:
        step_s = f"{r['last_step']}/{r['total']}" if r["last_step"] and r["total"] else "-"
        print(f"{r['width']:>6} {M_LABEL.get(r['width'],'?'):>3} {r['params']:>12,} "
              f"{r['loss']:>12.6g} {step_s:>16}  {'complete' if r['done'] else 'IN PROGRESS'}")


def fit_and_plot(ax, rows, color, label):
    fit_rows = [r for r in rows if r["done"]] if FIT_COMPLETED_ONLY else rows
    if len(fit_rows) < 2:
        print(f"\n{label}: not enough completed runs to fit; fitting all points.")
        fit_rows = rows
    P = np.array([r["params"] for r in fit_rows], float)
    L = np.array([r["loss"] for r in fit_rows], float)
    b, c = np.polyfit(np.log10(P), np.log10(L), 1)
    a = 10 ** (-c / b)

    allP = np.array([r["params"] for r in rows], float)
    fx = np.logspace(np.log10(allP.min()) * 0.98, np.log10(allP.max()) * 1.02, 200)
    fy = (fx / a) ** b
    ax.plot(fx, fy, "--", color=color, lw=1.4,
            label=fr"{label}: $L=(M/{a:.3g})^{{{b:.3f}}}$")

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
    print(f"{label} fit (on {len(fit_rows)} points): L = (M / {a:.5g})^{b:.5f}")
    return a, b


fig, ax = plt.subplots(figsize=(8, 6.5))
fit_and_plot(ax, rows_k1, "#2a9d8f", "kNNN=1")
fit_and_plot(ax, rows_k2, "#e76f51", "kNNN=2")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Model Parameters (M)")
ax.set_ylabel("MSE (L)")
ax.set_title("Model scaling: kNNN=1 vs kNNN=2  (cluster data, HP transfer)")
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=8, loc="best")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"\nSaved: {OUT}")