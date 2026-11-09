"""
Loss comparison for the sincos A/B sweep (m1-m5, no m6). Same conventions
as every other scaling plot this session: hollow circle = width hasn't
reached its step budget yet, filled = complete; power-law fit on completed
points only.

Built so klen>1 runs can be added later just by appending to SERIES below --
each entry already carries its own klen label and config-naming pattern, no
restructuring needed when that day comes. For now only klen=1 entries exist
since that's the only thing actually running.

Run: python3 dev_scripts/plot_sincos_ab_sweep.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace/PP_collision")
from fm4npp.models.mambagpt import Mamba1GPT

# ============================================================
DEPTH = 12
WIDTHS = [64, 128, 256, 512, 1024]  # m1-m5, m6 deliberately excluded
M_LABEL = {64: 'm1', 128: 'm2', 256: 'm3', 512: 'm4', 1024: 'm5', 1536: 'm6'}
TOTAL_STEPS = 100_000  # flat, same for every width/series right now
RUN_NUM = "sweep"
FIT_COMPLETED_ONLY = True
OUT = "/workspace/PP_collision/fig_sincos_ab_sweep.png"

# Each entry: (label, color, sweep_dir, config_pattern_fn(width), embed_method
# for param counting, pos_dim, aux_dim). To add a klen=2/3 series later,
# just append another entry here -- everything else (fitting, plotting,
# in-progress handling) already works generically off this list.
SERIES = [
    # [FIX] was pointing at the YAML's own checkpoint_dir field
    # (v7_01_SincosA_Full_Sweep_.../) -- but the runner's --root_dir CLI
    # flag takes precedence over that, so the real files sit directly
    # under checkpoints/scale_w{w}_d12_sincosA/sweep/, confirmed via find.
    dict(label="Sincos A (klen=1, no aux)", color="#0072B2",
        sweep_dir="/workspace/PP_collision/checkpoints",
        config_pattern=lambda w: f"scale_w{w}_d12_sincosA",
        embed_method="pos_only", pos_dim=3, aux_dim=4, klen=1),
    dict(label="Sincos B (klen=1, aux pitch)", color="#CC79A7",
        sweep_dir="/workspace/PP_collision/checkpoints",
        config_pattern=lambda w: f"scale_w{w}_d12_sincosB",
        embed_method="pos_plus_aux", pos_dim=3, aux_dim=4, klen=1),
    dict(label="Sincos A (klen=5, no aux)", color="#56B4E9",
        sweep_dir="/workspace/PP_collision/checkpoints",
        config_pattern=lambda w: f"scale_w{w}_d12_sincosA_klen5",
        embed_method="pos_only", pos_dim=3, aux_dim=4, klen=5),
    dict(label="Sincos B (klen=5, aux pitch)", color="#D55E00",
        sweep_dir="/workspace/PP_collision/checkpoints",
        config_pattern=lambda w: f"scale_w{w}_d12_sincosB_klen5",
        embed_method="pos_plus_aux", pos_dim=3, aux_dim=4, klen=5),
]
# ============================================================


def param_count(width, depth, embed_method, pos_dim, aux_dim, klen):
    kwargs = dict(embed_dim=width, num_layers=depth, d_state=16, klen=klen,
                  dropout=0.1, embed_method=embed_method, pe_method='nerf',
                  use_sincos_target=True)  # all series here use the sincos head
    if embed_method == 'pos_plus_aux':
        kwargs.update(pos_dim=pos_dim, aux_dim=aux_dim)
    m = Mamba1GPT(**kwargs)
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


def collect(series):
    rows = []
    for w in WIDTHS:
        cfg = series["config_pattern"](w)
        loss, last_step = read_run(series["sweep_dir"], cfg, RUN_NUM)
        if loss is None:
            print(f"  [{series['label']}] {M_LABEL[w]}: not started yet -- skipped")
            continue
        done = last_step is not None and last_step >= TOTAL_STEPS
        params = param_count(w, DEPTH, series["embed_method"], series["pos_dim"],
                             series["aux_dim"], series["klen"])
        rows.append(dict(width=w, loss=loss, last_step=last_step, total=TOTAL_STEPS,
                         done=done, params=params))
    return rows


def fit_and_plot(ax, rows, color, label):
    if not rows:
        print(f"\n{label}: no data yet, skipping.")
        return
    rows = sorted(rows, key=lambda r: r["width"])
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


fig, ax = plt.subplots(figsize=(9, 7))
for series in SERIES:
    print(f"\n{series['label']}:")
    rows = collect(series)
    for r in rows:
        step_s = f"{r['last_step']:,}/{r['total']:,}"
        print(f"  {M_LABEL[r['width']]:>3}  params={r['params']:>12,}  loss={r['loss']:.6g}  "
              f"steps={step_s}  {'complete' if r['done'] else 'IN PROGRESS'}")
    fit_and_plot(ax, rows, series["color"], series["label"])

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Model Parameters (M)")
ax.set_ylabel("MSE (L)")
ax.set_title("Sincos A vs. B, kNNN=1 vs. 5 (m1-m5, flat 100k steps)\n(loss definitions differ across series -- compare scaling trends, not absolute values)", fontsize=10)
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=8, loc="best")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"\nSaved: {OUT}")