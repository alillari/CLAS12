"""Slide-ready momentum figures from stored metrics; no model or data loading."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
from campaign_util import require_entrance_evaluation

PAPER, NAVY, RUST = "#F7F6F2", "#18344A", "#B85C3B"
TEXT, MUTED, GRID = "#202428", "#6B7177", "#E8E6E1"
OTHER = ("#C3C8C6", "#B3BAB7", "#A2ABA7", "#929C98", "#828E89")
VARIABLES = (
    ("p_gev", "Momentum error", r"MAE$(p)$ [GeV]"),
    ("theta_deg", "Polar-angle error", r"MAE$(\theta)$ [deg]"),
    ("phi_deg", "Azimuthal-angle error", r"MAE$(\phi)$ [deg]"),
)
FIT_METRICS = (
    ("delta_p_over_p", "Momentum resolution", r"$\sigma(\Delta p/p)$ [%]", r"$\mu(\Delta p/p)$ [%]", 100.),
    ("delta_theta", "Polar-angle resolution", r"$\sigma(\Delta\theta)$ [deg]", r"$\mu(\Delta\theta)$ [deg]", 1.),
)


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def short(value, _position=None):
    return f"{value / 1e6:g}M" if value >= 1e6 else f"{value / 1e3:g}k" if value >= 1e3 else f"{value:g}"


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)


def model_catalog(manifest, run_ids):
    """Keep source identities, including multiple pretraining recipes at a width."""
    catalog, runs = {}, {}
    for run_id in sorted(run_ids):
        if run_id not in manifest:
            raise ValueError(f"Presentation requires manifest metadata for {run_id}")
        row = manifest[run_id]
        name = row.get("backbone_run_id") or run_id
        adapter = row.get("model_family") == "adapteronly" or name == "adapteronly"
        budget = number(row.get("labeled_events", row.get("eventnumber")))
        width = number(row.get("embed_dim", row.get("base_dim")))
        if budget is None or budget <= 0 or width is None:
            raise ValueError(f"Missing positive label budget or backbone width for {run_id}")
        metadata = dict(backbone_run_id=name, embed_dim=int(width),
                        num_layers_backbone=int(row.get("num_layers_backbone", 0)),
                        pretrain_events=row.get("pretrain_events"), adapter_only=adapter)
        if name in catalog and catalog[name] != metadata:
            raise ValueError(f"Conflicting backbone metadata for {name}")
        catalog[name] = metadata
        runs[run_id] = dict(run_id=run_id, backbone_run_id=name, labeled_events=int(budget))
    ordered = sorted((k for k in catalog if not catalog[k]["adapter_only"]),
                     key=lambda k: (catalog[k]["embed_dim"], catalog[k]["num_layers_backbone"], k))
    for index, name in enumerate(ordered, 1):
        catalog[name]["plot_label"] = f"m{index}"
    for name in catalog:
        if catalog[name]["adapter_only"]:
            catalog[name]["plot_label"] = "Adapter only"
    return catalog, runs, ordered


def unique_baseline(rows, keys):
    """Deduplicate baselines, tolerating rounding only in continuous metrics."""
    if not rows:
        return None
    first = rows[0]
    metric_keys = {"mae", "rmse", "r2", "fit_mean", "fit_sigma",
                   "fit_mean_error", "fit_sigma_error"}

    def matches(key, left, right):
        if left == right:
            return True
        if key not in metric_keys:
            return False
        left, right = number(left), number(right)
        return (left is not None and right is not None
                and math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-8))

    for row in rows[1:]:
        if any(not matches(k, row.get(k), first.get(k)) for k in keys):
            raise ValueError("COATJAVA values/populations differ across runs; plot separate campaigns or subsets.")
    return first


def prepare_headlines(raw, runs):
    rows, seen = [], set()
    for row in raw:
        if (row.get("record_type"), row.get("space")) != ("ml_error", "kinematic"):
            continue
        if row.get("method") not in ("adapter", "cvt") or row.get("variable") not in {v[0] for v in VARIABLES}:
            continue
        run_id = row.get("run_name") or row.get("run_num")
        if run_id not in runs:
            continue
        value = number(row.get("mae"))
        if value is None or value < 0:
            continue
        key = (run_id, row["method"], row["variable"])
        if key in seen:
            raise ValueError(f"Duplicate headline metric: {key}")
        seen.add(key)
        expected_unit = "GeV" if row["variable"] == "p_gev" else "deg"
        if row.get("unit") != expected_unit:
            raise ValueError(f"Unexpected physical unit for {key}: {row.get('unit')}")
        rows.append({**runs[run_id], **row, "mae": value})
    truths = {r.get("comparison_truth") for r in rows}
    if len(truths) > 1:
        raise ValueError(f"Cannot combine different comparison truths: {truths}")
    for variable, _, _ in VARIABLES:
        missing = set(runs) - {r["run_id"] for r in rows
                               if r["method"] == "adapter" and r["variable"] == variable}
        if missing:
            raise ValueError(f"Missing physical {variable} MAE for {sorted(missing)}")
        unique_baseline([r for r in rows if r["method"] == "cvt" and r["variable"] == variable],
                        ("mae", "n", "comparison_truth"))
    # Do not silently connect replicates or distinct adapter recipes as one trace.
    points = set()
    for row in rows:
        if row["method"] == "adapter":
            key = (row["backbone_run_id"], row["labeled_events"], row["variable"])
            if key in points:
                raise ValueError(f"Multiple runs at {key}; select a single recipe/seed before plotting.")
            points.add(key)
    return rows


def select_runs(runs, labeled_events=None):
    by_model = defaultdict(set)
    for row in runs.values():
        by_model[row["backbone_run_id"]].add(row["labeled_events"])
    common = set.intersection(*by_model.values())
    if labeled_events is None:
        if not common:
            raise ValueError("No common label budget; select a comparable campaign subset.")
        labeled_events = max(common)
    if labeled_events not in common:
        raise ValueError(f"Budget {labeled_events} not available for every model; common budgets: {sorted(common)}")
    selected = [r for r in runs.values() if r["labeled_events"] == labeled_events]
    if len({r["backbone_run_id"] for r in selected}) != len(selected):
        raise ValueError("Multiple runs per model at the requested budget; select a single recipe/seed.")
    return labeled_events, {r["run_id"] for r in selected}


def prepare_fits(raw, selected, runs, metric):
    rows, seen = [], set()
    for row in raw:
        run_id = row.get("run_id")
        if run_id not in selected or row.get("method") not in ("adapter", "cvt"):
            continue
        result = {**row, **runs[run_id], "metric": metric}
        for key in ("bin_center_gev", "bin_low_gev", "bin_high_gev", "fit_mean", "fit_sigma",
                    "fit_mean_error", "fit_sigma_error", "n"):
            result[key] = number(row.get(key))
        if any(result[k] is None for k in ("bin_center_gev", "bin_low_gev", "bin_high_gev")):
            raise ValueError(f"Missing momentum bin in {metric}: {run_id}")
        key = (run_id, row["method"], result["bin_low_gev"], result["bin_high_gev"])
        if key in seen:
            raise ValueError(f"Duplicate fit bin: {key}")
        seen.add(key)
        result["plotted"] = (row.get("fit_status") == "ok" and result["fit_sigma"] is not None
                             and result["fit_sigma"] > 0 and result["fit_mean"] is not None)
        rows.append(result)
    baselines = defaultdict(list)
    for row in rows:
        if row["method"] == "cvt":
            baselines[(row["bin_low_gev"], row["bin_high_gev"])].append(row)
    for group in baselines.values():
        unique_baseline(group, ("fit_mean", "fit_sigma", "fit_mean_error", "fit_sigma_error", "n", "fit_status"))
    return rows


def style_axis(ax):
    ax.set_facecolor(PAPER)
    ax.grid(axis="y", color=GRID, linewidth=.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=TEXT, labelsize=11)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_axisbelow(True)


def variants(catalog, ordered):
    result = ["all_models"]
    if ordered and len(catalog) > 1:
        result.append("focus")
    if len(ordered) >= 3:
        result.extend(("other_pretrained_lines", "other_pretrained_envelope"))
    return result


def traces(catalog, ordered, variant):
    largest = ordered[-1] if ordered else None
    names = [k for k, v in catalog.items() if v["adapter_only"]] + ordered
    for name in names:
        row = catalog[name]
        is_other = not row["adapter_only"] and name != largest
        if variant == "focus" and is_other:
            continue
        if variant == "other_pretrained_envelope" and is_other:
            continue
        color = NAVY if row["adapter_only"] else RUST if name == largest else OTHER[ordered.index(name) % len(OTHER)]
        label = row["plot_label"]
        if name == largest:
            label += " (largest backbone)"
        if variant == "other_pretrained_lines" and is_other:
            color, label = "#AAB2AE", "Other pretrained backbones"
        yield name, label, color, 2.7 if name == largest or row["adapter_only"] else 1.8


def envelope(ax, points, ordered):
    by_x = defaultdict(dict)
    for model in ordered[:-1]:
        for x, y, _ in points.get(model, []):
            values = by_x[x]
            if math.isfinite(y):
                values[model] = y
    # Require every other backbone at each point; missing bins break the band.
    xs = sorted(by_x)
    lo = [min(by_x[x].values()) if len(by_x[x]) == len(ordered) - 1 else np.nan for x in xs]
    hi = [max(by_x[x].values()) if len(by_x[x]) == len(ordered) - 1 else np.nan for x in xs]
    if xs:
        ax.fill_between(xs, lo, hi, color="#B5BDB9", alpha=.55,
                        label="Other backbones (min–max)", zorder=1)


def draw_points(ax, points, catalog, ordered, variant, errors=False):
    if variant == "other_pretrained_envelope":
        envelope(ax, points, ordered)
    for model, label, color, lw in traces(catalog, ordered, variant):
        values = sorted(points.get(model, []))
        if not values:
            continue
        xs, ys, es = zip(*values)
        if errors:
            ax.errorbar(xs, ys, yerr=es, label=label, color=color, linewidth=lw,
                        marker="o", markersize=5, elinewidth=1, capsize=2, zorder=3)
        else:
            ax.plot(xs, ys, label=label, color=color, linewidth=lw, marker="o", markersize=5, zorder=3)


def save_figure(fig, axes, out, stem, subtitle):
    handles = {}
    for ax in axes:
        hs, ls = ax.get_legend_handles_labels()
        handles.update(zip(ls, hs))
    fig.suptitle(subtitle, fontsize=12, color=MUTED)
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=min(4, len(handles)),
               frameon=False, fontsize=10, labelcolor=TEXT)
    out.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{extension}", dpi=300, facecolor=PAPER)
    plt.close(fig)


def make_axes(count, vertical=False):
    size = (7.2, 3.5 * count + .7) if vertical else (5 * count, 4.8)
    if count == 1:
        size = (7.2, 4.9)
    fig, axes = plt.subplots(count if vertical else 1, 1 if vertical else count,
                             figsize=size, squeeze=False, constrained_layout=True)
    fig.patch.set_facecolor(PAPER)
    for ax in axes.flat:
        style_axis(ax)
    return fig, list(axes.flat)


def headline_panel(ax, rows, variable, catalog, ordered, variant, scaling, field="mae"):
    key, title, ylabel = variable
    subset = [r for r in rows if r["variable"] == key]
    points = defaultdict(list)
    names = [k for k, v in catalog.items() if v["adapter_only"]] + ordered
    for row in subset:
        if row["method"] == "adapter":
            x = row["labeled_events"] if scaling else names.index(row["backbone_run_id"])
            points[row["backbone_run_id"]].append((x, row[field], 0))
    draw_points(ax, points, catalog, ordered, variant)
    baseline = next((r for r in subset if r["method"] == "cvt"), None)
    if baseline:
        ax.axhline(baseline[field], color=TEXT, ls="--", lw=1.8, label="COATJAVA", zorder=2)
    if scaling:
        ax.set_xscale("log")
        budgets = sorted({r["labeled_events"] for r in rows})
        ticks = [x for x in budgets if math.isclose(math.log10(x) % 1, 0, abs_tol=1e-10)]
        ax.set_xticks(ticks if len(ticks) >= 2 else budgets[::max(1, len(budgets)//5)])
        ax.xaxis.set_major_formatter(FuncFormatter(short))
        ax.set_xlabel("Labeled events")
        if field != "r2" and all(r[field] > 0 for r in subset):
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_xticks(range(len(names)), [catalog[k]["plot_label"] for k in names])
        ax.set_xlim(-.5, len(names) - .5)
        if field != "r2":
            ax.set_ylim(bottom=0)
        ax.set_xlabel("Model")
    ax.set_title(title, weight="semibold", fontsize=14, pad=10)
    ax.set_ylabel(ylabel)


def fit_panel(ax, rows, metric, catalog, ordered, variant, field):
    _, title, sigma_label, mean_label, scale = metric
    points, baseline = defaultdict(list), {}
    # A NaN remains at failed/sparse bins so lines do not bridge missing fits.
    for row in rows:
        y = row[field] * scale if row["plotted"] else np.nan
        error = row.get(f"{field}_error")
        error = scale * error if row["plotted"] and error is not None and error >= 0 else np.nan
        point = (row["bin_center_gev"], y, error)
        if row["method"] == "adapter":
            points[row["backbone_run_id"]].append(point)
        else:
            baseline[row["bin_center_gev"]] = point
    centers = sorted({row["bin_center_gev"] for row in rows})
    for model, model_points in points.items():
        by_center = {point[0]: point for point in model_points}
        points[model] = [by_center.get(x, (x, np.nan, np.nan)) for x in centers]
    if baseline:
        baseline = {x: baseline.get(x, (x, np.nan, np.nan)) for x in centers}
    draw_points(ax, points, catalog, ordered, variant, errors=True)
    if baseline:
        xs, ys, es = zip(*sorted(baseline.values()))
        ax.errorbar(xs, ys, yerr=es, color=TEXT, ls="--", lw=1.8, marker="s", ms=4,
                    elinewidth=1, capsize=2, label="COATJAVA")
    ax.set_xlabel(r"True momentum $p$ [GeV]")
    ax.set_ylabel(sigma_label if field == "fit_sigma" else mean_label)
    ax.set_title(title if field == "fit_sigma" else title.replace("resolution", "bias"),
                 weight="semibold", fontsize=14, pad=10)
    if field == "fit_sigma":
        ax.set_ylim(bottom=0)
    else:
        ax.axhline(0, color=MUTED, lw=.8, zorder=1)


def make_presentation(headline_path: Path, manifest, out: Path, *, labeled_events=None, manifest_path=None):
    raw = [json.loads(line) for line in headline_path.read_text().splitlines() if line.strip()]
    for row in raw:
        require_entrance_evaluation(row, headline_path)
    ids = {r.get("run_name") or r.get("run_num") for r in raw
           if r.get("method") == "adapter" and r.get("record_type") == "ml_error" and r.get("space") == "kinematic"}
    if not ids:
        raise ValueError(f"No physical adapter metrics in {headline_path}")
    catalog, runs, ordered = model_catalog(manifest, ids)
    rows = prepare_headlines(raw, runs)
    budget, selected = select_runs(runs, labeled_events)
    fits, sources, warnings = {}, [headline_path], []
    if manifest_path and manifest_path.exists():
        sources.append(manifest_path)
    for metric in FIT_METRICS:
        path = headline_path.parent / f"{metric[0]}_fits.csv"
        if path.exists():
            sources.append(path)
            with path.open(newline="") as stream:
                raw_fits = list(csv.DictReader(stream))
            for row in raw_fits:
                require_entrance_evaluation(row, path)
            fit_rows = prepare_fits(raw_fits, selected, runs, metric[0])
            available = {r["run_id"] for r in fit_rows if r["method"] == "adapter" and r["plotted"]}
            if available == selected:
                fits[metric[0]] = fit_rows
            else:
                warnings.append(f"Skipped {metric[0]}: valid fits missing for {sorted(selected - available)}")
        else:
            warnings.append(f"Skipped {metric[0]}: {path.name} is unavailable; no reevaluation was run.")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "model_labels.csv", list(catalog.values()))
    write_csv(out / "selected_runs.csv", [runs[k] for k in sorted(selected)])
    write_csv(out / "headline_plot_data.csv", rows)
    write_csv(out / "resolution_plot_data.csv", [r for rs in fits.values() for r in rs])
    scaling = len({r["labeled_events"] for r in runs.values()}) > 1
    headline_rows = rows if scaling else [r for r in rows if r["run_id"] in selected]
    counts = sorted({int(r["n"]) for r in rows if r.get("n") is not None})
    sample_note = f"{short(counts[0])} evaluated tracks" if len(counts) == 1 else "Evaluation counts vary; see plot data"
    context = f"{sample_note} · Lower is better"
    if not scaling:
        context = f"{short(budget)} labeled events · " + context
    rc = {"font.size": 12, "axes.labelsize": 13, "text.color": TEXT, "axes.labelcolor": TEXT,
          "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": PAPER}
    with plt.rc_context(rc):
        for variant in variants(catalog, ordered):
            for vertical, layout in ((False, "1x3"), (True, "3x1")):
                # A fixed-budget categorical comparison cannot use a continuous envelope.
                if not scaling and variant == "other_pretrained_envelope":
                    continue
                fig, axes = make_axes(3, vertical)
                for ax, variable in zip(axes, VARIABLES):
                    headline_panel(ax, headline_rows, variable, catalog, ordered, variant, scaling)
                save_figure(fig, axes, out, f"momentum_{'scaling' if scaling else 'errors'}_{variant}_{layout}", context)
            if variant in ("all_models", "focus"):
                for variable in VARIABLES:
                    fig, axes = make_axes(1)
                    headline_panel(axes[0], headline_rows, variable, catalog, ordered, variant, scaling)
                    save_figure(fig, axes, out / "individual", f"mae_{variable[0]}_{variant}", context)
            if not fits:
                continue
            metrics = [m for m in FIT_METRICS if m[0] in fits]
            fit_context = f"{short(budget)} labeled events · Gaussian core fits"
            for field, kind in (("fit_sigma", "resolution"), ("fit_mean", "bias")):
                for vertical in (False, True):
                    if len(metrics) == 1 and vertical:
                        continue
                    layout = f"{len(metrics)}x1" if vertical else f"1x{len(metrics)}"
                    fig, axes = make_axes(len(metrics), vertical)
                    for ax, metric in zip(axes, metrics):
                        fit_panel(ax, fits[metric[0]], metric, catalog, ordered, variant, field)
                    save_figure(fig, axes, out, f"momentum_{kind}_{variant}_{layout}", fit_context)
                if variant in ("all_models", "focus"):
                    for metric in metrics:
                        fig, axes = make_axes(1)
                        fit_panel(axes[0], fits[metric[0]], metric, catalog, ordered, variant, field)
                        save_figure(fig, axes, out / "individual", f"{kind}_{metric[0]}_{variant}", fit_context)
    provenance = {
        "sources": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
        "selected_labeled_events": budget, "selected_runs": sorted(selected), "evaluation_counts": counts,
        "comparison_truth": sorted({r.get("comparison_truth", "") for r in rows}), "warnings": warnings,
        "baseline_policy": "Exact duplicate CVT results shown once; differing values or counts rejected.",
        "envelope_policy": "Min/max across other backbones at common x; not statistical uncertainty.",
        "fit_error_policy": "Stored Gaussian-fit parameter errors; sigma itself is residual width, not uncertainty.",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (out / "README.md").write_text(
        "# Momentum presentation figures\n\n"
        "Vector PDF and 300-dpi PNG figures in the track-finding DNP palette. "
        "Use the horizontal panels at the top of a slide; individual/ contains standalone panels.\n\n"
        f"Resolution and bias panels use {short(budget)} labeled events, selected by budget, not performance. "
        f"{sample_note}. Physical errors are shown separately in GeV and degrees; phi residuals are wrapped by the evaluator. "
        "Scaling panels use logarithmic axes when errors are positive.\n\n"
        "COATJAVA uses CVT::Tracks p/theta and CVT::Trajectory entrance phi. Repeated identical baselines are shown once, "
        "without averaging or multiplying sample counts. Missing/failed fit bins break the curve. "
        "Error bars are stored fit-parameter errors. Gaussian core widths are not full-distribution RMS values. "
        "The other-backbone envelope is a min/max range, not an uncertainty interval.\n\n"
        "model_labels.csv records the actual backbone IDs, widths, depths and pretraining sizes. "
        "m-labels are local to this campaign. selected_runs.csv and the plot-data CSVs identify every plotted source. "
        "provenance.json records source paths and SHA256 checksums. No training or evaluation was performed.\n\n"
        + ("\n".join(warnings) + "\n" if warnings else "")
    )
    for warning in warnings:
        print(f"Presentation: {warning}")
    print(f"Wrote momentum presentation figures to {out}")
