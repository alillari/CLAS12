"""Presentation exports for stored Cartesian MAE, pooled RMSE and macro R-squared."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from campaign_util import require_entrance_evaluation

from momentum_presentation import (
    PAPER, TEXT, VARIABLES, headline_panel, make_axes, model_catalog, number,
    prepare_headlines, save_figure, short, unique_baseline, variants, write_csv,
)

COMPONENTS = (("px_gev", r"$p_x$"), ("py_gev", r"$p_y$"), ("pz_gev", r"$p_z$"))
METRICS = (
    ("mae", "MAE", "Mean component MAE ↓", "MAE [GeV]"),
    ("rmse", "RMSE", "Pooled component RMSE ↓", "RMSE [GeV]"),
    ("r2", r"$R^2$", r"Mean component $R^2$ ↑", r"$R^2$"),
)


def prepare_component_rows(raw, runs):
    """Select physical Cartesian errors, preserving undefined R2 as an error."""
    groups = defaultdict(dict)
    variables = {v for v, _ in COMPONENTS}
    for row in raw:
        if (row.get("record_type"), row.get("space")) != ("ml_error", "component"):
            continue
        run_id = row.get("run_name") or row.get("run_num")
        if run_id not in runs or row.get("method") not in ("adapter", "cvt") or row.get("variable") not in variables:
            continue
        key = (run_id, row["method"])
        if row["variable"] in groups[key]:
            raise ValueError(f"Duplicate Cartesian metric: {key}, {row['variable']}")
        if row.get("unit") != "GeV":
            raise ValueError(f"Expected GeV Cartesian errors for {key}")
        clean = {**row, **runs[run_id]}
        for field in ("mae", "rmse", "r2", "n"):
            clean[field] = number(row.get(field))
            if clean[field] is None:
                raise ValueError(f"Missing finite Cartesian {field}: {key}, {row['variable']}")
        if clean["mae"] < 0 or clean["rmse"] < 0 or clean["n"] <= 0:
            raise ValueError(f"Invalid Cartesian error/count: {key}, {row['variable']}")
        groups[key][row["variable"]] = clean
    missing = set(runs) - {run_id for run_id, method in groups if method == "adapter"}
    if missing:
        raise ValueError(f"No Cartesian evaluation metrics for {sorted(missing)}")
    components, aggregate = [], []
    for (run_id, method), group in sorted(groups.items()):
        if set(group) != variables:
            raise ValueError(f"Incomplete Cartesian components for {run_id}, {method}")
        rows = [group[v] for v, _ in COMPONENTS]
        if len({r['n'] for r in rows}) != 1:
            raise ValueError(f"Unequal component counts for {run_id}, {method}; cannot use equal-count pooling")
        components.extend(rows)
        aggregate.append({
            **rows[0], "variable": "cartesian", "space": "component_aggregate",
            "label": "Cartesian momentum", "mae": sum(r["mae"] for r in rows) / 3,
            "rmse": math.sqrt(sum(r["rmse"] ** 2 for r in rows) / 3),
            "r2": sum(r["r2"] for r in rows) / 3,
        })
    truths = {r.get("comparison_truth") for r in components}
    if len(truths) > 1:
        raise ValueError(f"Different comparison truths in Cartesian results: {truths}")
    for variable in variables | {"cartesian"}:
        unique_baseline([r for r in components + aggregate if r["method"] == "cvt" and r["variable"] == variable],
                        ("mae", "rmse", "r2", "n", "comparison_truth"))
    seen = set()
    for row in aggregate:
        if row["method"] == "adapter":
            key = (row["backbone_run_id"], row["labeled_events"])
            if key in seen:
                raise ValueError(f"Duplicate model/budget {key}; select a single recipe/seed")
            seen.add(key)
    return components, aggregate


def render_panels(rows, panels, catalog, ordered, out, stem, subtitle, variant, scaling, *, singles=False):
    """Each panel carries its own metric field and physical quantity."""
    for vertical, layout in ((False, "1x3"), (True, "3x1")):
        fig, axes = make_axes(3, vertical)
        for ax, (variable, field) in zip(axes, panels):
            headline_panel(ax, rows, variable, catalog, ordered, variant, scaling, field=field)
        save_figure(fig, axes, out, f"{stem}_{variant}_{layout}", subtitle)
    if singles:
        for variable, field in panels:
            fig, axes = make_axes(1)
            headline_panel(axes[0], rows, variable, catalog, ordered, variant, scaling, field=field)
            save_figure(fig, axes, out / "individual", f"{field}_{variable[0]}_{variant}", subtitle)


def make_ml_presentation(headline_path: Path, manifest, out: Path, *, manifest_path=None):
    raw = [json.loads(line) for line in headline_path.read_text().splitlines() if line.strip()]
    for row in raw:
        require_entrance_evaluation(row, headline_path)
    ids = {r.get("run_name") or r.get("run_num") for r in raw
           if (r.get("record_type"), r.get("method"), r.get("space")) == ("ml_error", "adapter", "component")}
    if not ids:
        raise ValueError(f"No Cartesian adapter metrics found in {headline_path}")
    catalog, runs, ordered = model_catalog(manifest, ids)
    components, aggregate = prepare_component_rows(raw, runs)
    physical = prepare_headlines(raw, runs)
    for row in physical:
        row["rmse"] = number(row.get("rmse"))
        if row["rmse"] is None or row["rmse"] < 0:
            raise ValueError(f"Missing physical RMSE for {row['run_id']}, {row['variable']}")
    for variable, _, _ in VARIABLES:
        unique_baseline([r for r in physical if r["method"] == "cvt" and r["variable"] == variable],
                        ("rmse", "n", "comparison_truth"))
    scaling = len({r["labeled_events"] for r in aggregate}) > 1
    counts = sorted({int(r["n"]) for r in components + physical})
    sample_note = f"{short(counts[0])} evaluated tracks" if len(counts) == 1 else "Evaluation counts vary; see source CSV"
    if not scaling:
        sample_note = f"{short(aggregate[0]['labeled_events'])} labeled events · " + sample_note
    context = f"Cartesian momentum · {sample_note}"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "component_plot_data.csv", components)
    write_csv(out / "aggregate_plot_data.csv", aggregate)
    write_csv(out / "kinematic_plot_data.csv", physical)
    write_csv(out / "model_labels.csv", list(catalog.values()))
    summary_panels = [(("cartesian", title, ylabel), field) for field, _, title, ylabel in METRICS]
    rc = {"font.size": 12, "axes.labelsize": 13, "text.color": TEXT, "axes.labelcolor": TEXT,
          "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": PAPER}
    with plt.rc_context(rc):
        for variant in variants(catalog, ordered):
            if not scaling and variant == "other_pretrained_envelope":
                continue
            render_panels(aggregate, summary_panels, catalog, ordered, out, "regression_metrics",
                          context, variant, scaling, singles=variant in ("all_models", "focus"))
            if variant not in ("all_models", "focus"):
                continue
            for field, label, _, _ in METRICS:
                unit = "" if field == "r2" else " [GeV]"
                direction = "Higher is better" if field == "r2" else "Lower is better"
                panels = [((variable, f"{component} {label}", label + unit), field)
                          for variable, component in COMPONENTS]
                render_panels(components, panels, catalog, ordered, out, f"{field}_components",
                              f"{sample_note} · {direction}", variant, scaling, singles=True)
            panels = [((variable, title.replace("error", "RMSE"), ylabel.replace("MAE", "RMSE")), "rmse")
                      for variable, title, ylabel in VARIABLES]
            render_panels(physical, panels, catalog, ordered, out, "rmse_kinematic",
                          f"{sample_note} · Lower is better", variant, scaling, singles=True)
            # Full-range R2 remains the main panel; a clearly labeled zoom resolves
            # the high-label regime without hiding low-label failures in that panel.
            high_labels = [r for r in aggregate if r["labeled_events"] >= 10000]
            if len({r["labeled_events"] for r in high_labels}) >= 2:
                fig, axes = make_axes(1)
                headline_panel(axes[0], high_labels, summary_panels[2][0], catalog, ordered,
                               variant, True, field="r2")
                save_figure(fig, axes, out / "individual", f"r2_cartesian_10k_plus_{variant}",
                            f"10k+ labeled events · {sample_note} · Higher is better")
    sources = [headline_path] + ([manifest_path] if manifest_path and manifest_path.exists() else [])
    (out / "provenance.json").write_text(json.dumps({
        "sources": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
        "runs": sorted(ids), "evaluation_counts": counts,
        "metrics": {"mae": "Arithmetic mean of px, py, pz MAEs (GeV)",
                    "rmse": "sqrt((RMSE_px^2 + RMSE_py^2 + RMSE_pz^2)/3) (GeV); equal counts required",
                    "r2": "Arithmetic mean of stored px, py, pz R2 values (dimensionless)"},
        "r2_axis": "Linear, including negative values; no clipping to [0,1]",
        "baseline": "CVT::Tracks p/theta with CVT::Trajectory entrance phi; identical duplicates shown once",
        "missing_metrics": "Kinematic R2 was not stored and is not inferred from RMSE",
    }, indent=2) + "\n")
    (out / "README.md").write_text(
        "# MAE, RMSE and R² presentation figures\n\n"
        "Start with regression_metrics_all_models_1x3.pdf or regression_metrics_focus_1x3.pdf. "
        "These show Cartesian aggregate MAE, RMSE and R² versus training labels. "
        "Vertical layouts, component panels and individual/ exports are included as PDF and 300-dpi PNG.\n\n"
        "- MAE: arithmetic mean of the px, py, pz MAEs, in GeV. Lower is better.\n"
        "- RMSE: sqrt((RMSE_px² + RMSE_py² + RMSE_pz²)/3), in GeV. Lower is better. "
        "This pools squared scalar errors before taking the root; it is not the mean component RMSE "
        "or the per-track 3D vector-error RMSE. Component counts must be equal.\n"
        "- R²: arithmetic mean of the stored px, py, pz R² values. Higher is better. "
        "This is a macro average, not a pooled R². R² axes are linear and preserve negative values.\n\n"
        "Positive MAE/RMSE scaling axes are logarithmic. The 10k+ R² figures explicitly zoom the "
        "high-label regime; the main figures retain the full label range. "
        "Additional rmse_kinematic panels show p [GeV], theta [degrees] and wrapped phi [degrees] separately. "
        "R² for these kinematic quantities was not stored and is not fabricated.\n\n"
        "COATJAVA uses CVT::Tracks p/theta and CVT::Trajectory entrance phi. Duplicate identical baselines are plotted once. "
        "The other-backbone envelope is a min/max range, not an uncertainty interval. "
        "No seed uncertainties or confidence intervals are inferred. "
        "CSV files and provenance.json retain the plotted numbers, sample counts, model identities and source hashes. "
        "No training or reevaluation is needed.\n"
    )
    print(f"Wrote MAE/RMSE/R2 presentation figures to {out}")
