#!/usr/bin/env python3
"""Plot campaign-level CLAS12 track-regression scaling traces."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from campaign_util import read_yaml


LABEL_RE = re.compile(r"(?:^|_)label(?P<label>\d+)(?:_|$)")
COMPONENTS = ("px_gev", "py_gev", "pz_gev")
PHYSICS_PLOT_METHODS = ("adapter", "cvt")
PERCENT_SCALE = 100.0
COMPONENT_LABELS = {
    "px_gev": "px",
    "py_gev": "py",
    "pz_gev": "pz",
}
PLOT_KEY_LABELS = {
    "backbone_width_label": "backbone",
    "embed_dim": "embed_dim",
    "labeled_events": "labeled_events",
    "pretrain_events": "pretrain_events",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", help="Campaign directory containing manifest.yaml and summary/.")
    parser.add_argument("--manifest", help="Optional explicit manifest path.")
    parser.add_argument("--headline-jsonl", help="Optional explicit campaign_headline_metrics.jsonl path.")
    parser.add_argument("--output-dir", help="Optional explicit plot output directory.")
    parser.add_argument(
        "--plot-suite",
        choices=("standard", "momentum-resolution", "all"),
        default="standard",
        help="Plot suite to generate. 'standard' preserves existing campaign plots.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path | None, Path, Path | None, Path]:
    campaign_dir = Path(args.campaign_dir).resolve() if args.campaign_dir else None
    if args.headline_jsonl:
        headline_jsonl = Path(args.headline_jsonl).resolve()
    elif campaign_dir is not None:
        headline_jsonl = campaign_dir / "summary" / "campaign_headline_metrics.jsonl"
    else:
        raise ValueError("Provide either --campaign-dir or --headline-jsonl")

    if args.manifest:
        manifest = Path(args.manifest).resolve()
    elif campaign_dir is not None:
        manifest = campaign_dir / "manifest.yaml"
    else:
        manifest = None

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    elif campaign_dir is not None:
        output_dir = campaign_dir / "summary" / "plots"
    else:
        output_dir = headline_jsonl.parent / "plots"
    return campaign_dir, headline_jsonl, manifest, output_dir


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
    return rows


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def parse_backbone_metadata(run_id: str) -> dict[str, Any]:
    from campaign_util import parse_run_name

    label_match = LABEL_RE.search(run_id)
    labeled_events = int(label_match.group("label")) if label_match else None
    backbone_run_id = LABEL_RE.sub("", run_id)
    metadata = parse_run_name(backbone_run_id)
    metadata.update({
        "run_id": run_id,
        "backbone_run_id": backbone_run_id,
        "labeled_events": labeled_events,
    })
    return metadata


def manifest_lookup(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    manifest = read_yaml(path)
    lookup = {}
    for run in manifest.get("runs", []):
        lookup[run["run_id"]] = dict(run)
    return lookup


def finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def finite_int(value: Any) -> int | None:
    number = finite_float(value)
    return int(number) if number is not None else None


def plot_key_label(key: str) -> str:
    return PLOT_KEY_LABELS.get(key, key)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parameter_counter():
    try:
        from fm4npp.models.mambagpt import Mamba1GPT, MambaGPT
        from fm4npp.utils import count_parameters
    except Exception:
        return None

    cache: dict[tuple[Any, ...], int] = {}

    def count(row: dict[str, Any]) -> int | None:
        key = (
            row.get("model_family", "mamba1"),
            int(row["embed_dim"]),
            int(row.get("num_layers_backbone", 12)),
            int(row.get("d_state", 16)),
            int(row.get("d_conv", 4)),
            int(row.get("expand", 2)),
            int(row.get("klen", 1)),
            row.get("embed_method", "pos_only"),
            row.get("pe_method", "nerf"),
        )
        if key in cache:
            return cache[key]
        model_family, embed_dim, layers, d_state, d_conv, expand, klen, embed_method, pe_method = key
        cls = Mamba1GPT if model_family == "mamba1" else MambaGPT
        with contextlib.redirect_stdout(io.StringIO()):
            model = cls(
                embed_dim=embed_dim,
                num_layers=layers,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                klen=klen,
                dropout=0.1,
                embed_method=embed_method,
                pe_method=pe_method,
            )
        cache[key] = int(count_parameters(model))
        return cache[key]

    return count


def build_plot_rows(metric_rows: list[dict[str, Any]], manifest_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    adapter_native: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    adapter_components: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in metric_rows:
        run_id = row.get("run_name") or row.get("run_num")
        if not run_id:
            continue
        if (
            row.get("record_type") == "ml_error"
            and row.get("method") == "adapter"
            and row.get("space") == "native_target"
        ):
            adapter_native[run_id][row["variable"]] = row
        elif (
            row.get("record_type") == "ml_error"
            and row.get("method") == "adapter"
            and row.get("space") == "component"
            and row.get("variable") in COMPONENTS
        ):
            adapter_components[run_id][row["variable"]] = row

    count_params = parameter_counter()
    rows = []
    for run_id in sorted(set(adapter_native) | set(adapter_components)):
        metric_rows_for_run = adapter_native.get(run_id) or adapter_components.get(run_id, {})
        metric_space = "native_target" if run_id in adapter_native else "component"
        try:
            metadata = parse_backbone_metadata(run_id)
        except ValueError:
            metadata = {"run_id": run_id, "backbone_run_id": run_id}
        manifest_row = manifest_rows.get(run_id, {})
        merged = {**metadata, **manifest_row}
        if merged.get("labeled_events") is None:
            merged["labeled_events"] = merged.get("eventnumber")
        if "embed_dim" not in merged and "base_dim" in merged:
            merged["embed_dim"] = merged["base_dim"]

        row = {
            "run_id": run_id,
            "backbone_run_id": merged.get("backbone_run_id", metadata.get("backbone_run_id")),
            "embed_dim": int(merged["embed_dim"]) if merged.get("embed_dim") is not None else None,
            "num_layers_backbone": int(merged.get("num_layers_backbone", 12)),
            "pretrain_events": int(merged["pretrain_events"]) if merged.get("pretrain_events") is not None else None,
            "labeled_events": int(merged["labeled_events"]) if merged.get("labeled_events") is not None else None,
            "model_family": merged.get("model_family", "mamba1"),
            "campaign_metric_space": metric_space,
        }
        row["backbone_width_label"] = (
            "adapter-only"
            if row["model_family"] == "adapteronly"
            else row["embed_dim"]
        )
        mae_values = []
        rmse_values = []
        r2_values = []
        for variable, metric in sorted(metric_rows_for_run.items()):
            mae = finite_float(metric.get("mae"))
            rmse = finite_float(metric.get("rmse"))
            r2 = finite_float(metric.get("r2"))
            row[f"adapter_mae_{variable}"] = mae
            row[f"adapter_rmse_{variable}"] = rmse
            row[f"adapter_r2_{variable}"] = r2
            if mae is not None:
                mae_values.append(mae)
            if rmse is not None:
                rmse_values.append(rmse)
            if r2 is not None:
                r2_values.append(r2)
        row["adapter_mae_mean"] = sum(mae_values) / len(mae_values) if mae_values else None
        row["adapter_rmse_mean"] = sum(rmse_values) / len(rmse_values) if rmse_values else None
        row["adapter_r2_mean"] = sum(r2_values) / len(r2_values) if r2_values else None
        if row.get("model_family") == "adapteronly":
            row["backbone_n_params"] = 0
        elif count_params is not None and row["embed_dim"] is not None:
            try:
                row["backbone_n_params"] = count_params({**merged, **row})
            except Exception:
                row["backbone_n_params"] = None
        else:
            row["backbone_n_params"] = None
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def best_by_slice(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best = {}
    for row in rows:
        value = row.get("adapter_rmse_mean")
        if value is None:
            continue
        key = (row.get("backbone_run_id"), row.get("labeled_events"))
        if key not in best or value < best[key]["adapter_rmse_mean"]:
            best[key] = row
    return list(best.values())


def safe_label(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def track_count_label(value: Any) -> str:
    count = finite_int(value)
    if count is None:
        return "unknown tracks"
    if count >= 1000 and count % 1000 == 0:
        return f"{count // 1000}k tracks"
    if count >= 1000:
        return f"{count / 1000:.1f}k tracks"
    return f"{count} tracks"


def is_adapter_only_row(row: dict[str, Any]) -> bool:
    return row.get("model_family") == "adapteronly" or row.get("backbone_run_id") == "adapteronly"


def pretrain_events_is_meaningful(rows: list[dict[str, Any]]) -> bool:
    """Return true only for real pretrained-data sweeps.

    AdapterOnly rows conventionally carry pretrain_events=0, but that is not a
    backbone pretraining size and should not create a separate facet by itself.
    """
    pretrained_event_counts = {
        int(row["pretrain_events"])
        for row in rows
        if not is_adapter_only_row(row)
        and row.get("pretrain_events") is not None
        and int(row["pretrain_events"]) > 0
    }
    return len(pretrained_event_counts) > 1


def plot_faceted_lines(
    rows: list[dict[str, Any]],
    output_path: Path,
    x_key: str,
    y_key: str,
    line_key: str,
    facet_key: str | None,
    title: str,
    ylabel: str,
    xlabel: str,
    x_log: bool = True,
    reference_y: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filtered = [
        row for row in rows
        if row.get(x_key) is not None and row.get(y_key) is not None
    ]
    if not filtered:
        return

    if facet_key is None:
        facet_values = [None]
    else:
        facet_values = sorted(
            {row.get(facet_key) for row in filtered},
            key=lambda value: (value is None, safe_label(value)),
        )

    n_facets = len(facet_values)
    ncols = min(3, n_facets)
    nrows = int(math.ceil(n_facets / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols, 4.0 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()

    for axis, facet_value in zip(axes_flat, facet_values):
        facet_rows = [
            row for row in filtered
            if facet_key is None or row.get(facet_key) == facet_value
        ]
        grouped = defaultdict(list)
        for row in facet_rows:
            grouped[row.get(line_key)].append(row)
        for line_value, line_rows in sorted(grouped.items(), key=lambda item: safe_label(item[0])):
            line_rows = sorted(line_rows, key=lambda row: row[x_key])
            axis.plot(
                [row[x_key] for row in line_rows],
                [row[y_key] for row in line_rows],
                marker="o",
                label=f"{plot_key_label(line_key)}={safe_label(line_value)}",
            )
        if reference_y is not None:
            axis.axhline(reference_y, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
        if x_log:
            axis.set_xscale("log")
        axis.grid(alpha=0.25)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        if facet_key is not None:
            axis.set_title(f"{plot_key_label(facet_key)}={safe_label(facet_value)}")
        axis.legend(fontsize=8)

    for axis in axes_flat[n_facets:]:
        axis.axis("off")
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def make_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    size_x = "backbone_n_params" if any(row.get("backbone_n_params") for row in rows) else "embed_dim"
    size_xlabel = "Backbone trainable parameters" if size_x == "backbone_n_params" else "Backbone embed dim"
    pretrain_axis_is_meaningful = pretrain_events_is_meaningful(rows)
    pretrain_facet_key = "pretrain_events" if pretrain_axis_is_meaningful else None
    native_metric_space = any(row.get("campaign_metric_space") == "native_target" for row in rows)
    metric_variables = sorted({
        key.removeprefix("adapter_mae_")
        for row in rows
        for key, value in row.items()
        if key.startswith("adapter_mae_")
        and key != "adapter_mae_mean"
        and value is not None
    })
    metric_mean_label = (
        "native targets" if native_metric_space else "px, py, pz"
    )
    unit_label = "" if native_metric_space else " [GeV]"

    for metric, ylabel, better in (
        ("mae", f"MAE{unit_label}", "lower"),
        ("rmse", f"RMSE{unit_label}", "lower"),
        ("r2", "R2", "higher"),
    ):
        mean_key = f"adapter_{metric}_mean"
        plot_faceted_lines(
            rows,
            output_dir / f"{metric}_mean_vs_labeled_events_by_width.png",
            x_key="labeled_events",
            y_key=mean_key,
            line_key="backbone_width_label",
            facet_key=pretrain_facet_key,
            title=f"Adapter mean {ylabel} vs labeled adapter data ({better} is better)",
            ylabel=f"Mean {ylabel} across {metric_mean_label}",
            xlabel="Adapter labeled events",
        )
        plot_faceted_lines(
            rows,
            output_dir / f"{metric}_mean_vs_backbone_params_by_labeled_events.png",
            x_key=size_x,
            y_key=mean_key,
            line_key="labeled_events",
            facet_key=pretrain_facet_key,
            title=f"Adapter mean {ylabel} vs backbone size ({better} is better)",
            ylabel=f"Mean {ylabel} across {metric_mean_label}",
            xlabel=size_xlabel,
        )
        if pretrain_axis_is_meaningful:
            plot_faceted_lines(
                rows,
                output_dir / f"{metric}_mean_vs_pretrain_events_by_labeled_events.png",
                x_key="pretrain_events",
                y_key=mean_key,
                line_key="labeled_events",
                facet_key="backbone_width_label",
                title=f"Adapter mean {ylabel} vs backbone pretraining data ({better} is better)",
                ylabel=f"Mean {ylabel} across {metric_mean_label}",
                xlabel="Backbone pretraining events",
            )
        for variable in metric_variables:
            component_key = f"adapter_{metric}_{variable}"
            component_label = COMPONENT_LABELS.get(variable, variable)
            plot_faceted_lines(
                rows,
                output_dir / f"{metric}_{variable}_vs_labeled_events_by_width.png",
                x_key="labeled_events",
                y_key=component_key,
                line_key="backbone_width_label",
                facet_key=pretrain_facet_key,
                title=f"Adapter {component_label} {ylabel} vs labeled adapter data ({better} is better)",
                ylabel=f"{component_label} {ylabel}",
                xlabel="Adapter labeled events",
            )
            plot_faceted_lines(
                rows,
                output_dir / f"{metric}_{variable}_vs_backbone_params_by_labeled_events.png",
                x_key=size_x,
                y_key=component_key,
                line_key="labeled_events",
                facet_key=pretrain_facet_key,
                title=f"Adapter {component_label} {ylabel} vs backbone size ({better} is better)",
                ylabel=f"{component_label} {ylabel}",
                xlabel=size_xlabel,
            )
            if pretrain_axis_is_meaningful:
                plot_faceted_lines(
                    rows,
                    output_dir / f"{metric}_{variable}_vs_pretrain_events_by_labeled_events.png",
                    x_key="pretrain_events",
                    y_key=component_key,
                    line_key="labeled_events",
                    facet_key="backbone_width_label",
                    title=f"Adapter {component_label} {ylabel} vs backbone pretraining data ({better} is better)",
                    ylabel=f"{component_label} {ylabel}",
                    xlabel="Backbone pretraining events",
                )


def normalize_delta_p_over_p_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        fit_mean = finite_float(row.get("fit_mean"))
        fit_sigma = finite_float(row.get("fit_sigma"))
        if fit_mean is None or fit_sigma is None:
            continue
        method = row.get("method")
        model_family = row.get("model_family")
        use_pretrained = bool_value(row.get("use_pretrained_backbone"))
        if method == "cvt":
            family = "conventional"
            trace_label = "CVT::Tracks"
        elif model_family == "adapteronly" or not use_pretrained:
            family = "adapter_only"
            trace_label = (
                f"AdapterOnly label={safe_label(row.get('labeled_events'))} "
                f"run={safe_label(row.get('run_id'))}"
            )
        else:
            family = "pretrained_adapter"
            trace_label = (
                f"Pretrained+Adapter w={safe_label(row.get('embed_dim'))} "
                f"pretrain={safe_label(row.get('pretrain_events'))} "
                f"label={safe_label(row.get('labeled_events'))} "
                f"run={safe_label(row.get('run_id'))}"
            )
        normalized.append({
            **row,
            "family": family,
            "trace_label": trace_label,
            "bin_center_gev": finite_float(row.get("bin_center_gev")),
            "bin_low_gev": finite_float(row.get("bin_low_gev")),
            "bin_high_gev": finite_float(row.get("bin_high_gev")),
            "fit_mean": fit_mean,
            "fit_sigma": fit_sigma,
            "n": finite_int(row.get("n")),
            "labeled_events": finite_int(row.get("labeled_events")),
            "embed_dim": finite_int(row.get("embed_dim")),
            "pretrain_events": finite_int(row.get("pretrain_events")),
        })
    return [row for row in normalized if row["bin_center_gev"] is not None]


def normalize_delta_theta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return normalize_delta_p_over_p_rows(rows)


def per_run_delta_p_over_p_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_run_rows = []
    for row in rows:
        method = row.get("method")
        if method not in PHYSICS_PLOT_METHODS:
            continue
        label = "Adapter" if method == "adapter" else "CVT::Tracks"
        per_run_rows.append({**row, "trace_label": label})
    return per_run_rows


def per_run_delta_theta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return per_run_delta_p_over_p_rows(rows)


def collapse_conventional_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conventional = [row for row in rows if row["family"] == "conventional"]
    others = [row for row in rows if row["family"] != "conventional"]
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in conventional:
        grouped[row["bin_center_gev"]].append(row)

    collapsed = []
    for center, group_rows in sorted(grouped.items()):
        weights = np.asarray([row["n"] or 1 for row in group_rows], dtype=float)
        means = np.asarray([row["fit_mean"] for row in group_rows], dtype=float)
        sigmas = np.asarray([row["fit_sigma"] for row in group_rows], dtype=float)
        first = group_rows[0]
        collapsed.append({
            **first,
            "run_id": "conventional_aggregate",
            "trace_label": "CVT::Tracks",
            "bin_center_gev": center,
            "fit_mean": float(np.average(means, weights=weights)),
            "fit_sigma": float(np.average(sigmas, weights=weights)),
            "n": int(np.sum(weights)),
        })
    return others + collapsed


def grouped_delta_p_over_p_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["trace_label"]].append(row)
    return {
        label: sorted(trace_rows, key=lambda row: row["bin_center_gev"])
        for label, trace_rows in grouped.items()
    }


def plot_delta_p_over_p_errorbars(rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    grouped = grouped_delta_p_over_p_rows(rows)

    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    linestyles = {
        "conventional": "--",
        "adapter_only": "-",
        "pretrained_adapter": "-.",
    }
    for label, trace_rows in sorted(grouped.items()):
        ax.errorbar(
            [row["bin_center_gev"] for row in trace_rows],
            [PERCENT_SCALE * row["fit_mean"] for row in trace_rows],
            yerr=[PERCENT_SCALE * row["fit_sigma"] for row in trace_rows],
            marker="o",
            capsize=3,
            linewidth=1.4,
            linestyle=linestyles.get(trace_rows[0]["family"], "-"),
            label=label,
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set(
        xlabel="True p [GeV]",
        ylabel=r"Gaussian mean of $(p_{reco} - p_{true}) / p_{true}$ [%]",
        title=title,
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_delta_p_over_p_scalar(
    rows: list[dict[str, Any]],
    output_path: Path,
    y_key: str,
    title: str,
    ylabel: str,
    zero_line: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    grouped = grouped_delta_p_over_p_rows(rows)
    linestyles = {
        "conventional": "--",
        "adapter_only": "-",
        "pretrained_adapter": "-.",
    }
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    for label, trace_rows in sorted(grouped.items()):
        trace_rows = [row for row in trace_rows if row.get(y_key) is not None]
        if not trace_rows:
            continue
        ax.plot(
            [row["bin_center_gev"] for row in trace_rows],
            [PERCENT_SCALE * row[y_key] for row in trace_rows],
            marker="o",
            linewidth=1.4,
            linestyle=linestyles.get(trace_rows[0]["family"], "-"),
            label=label,
        )
    if zero_line:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set(xlabel="True p [GeV]", ylabel=ylabel, title=title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_delta_p_over_p_set(rows: list[dict[str, Any]], output_dir: Path, stem: str, title: str) -> None:
    plot_delta_p_over_p_errorbars(
        rows,
        output_dir / f"{stem}.png",
        title,
    )
    plot_delta_p_over_p_scalar(
        rows,
        output_dir / f"{stem}_mean.png",
        "fit_mean",
        f"{title}: Gaussian mean",
        r"Gaussian mean of $(p_{reco} - p_{true}) / p_{true}$ [%]",
        zero_line=True,
    )
    plot_delta_p_over_p_scalar(
        rows,
        output_dir / f"{stem}_sigma.png",
        "fit_sigma",
        f"{title}: Gaussian sigma",
        r"Gaussian sigma of $(p_{reco} - p_{true}) / p_{true}$ [%]",
        zero_line=False,
    )


def plot_delta_theta_errorbars(rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    grouped = grouped_delta_p_over_p_rows(rows)

    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    linestyles = {
        "conventional": "--",
        "adapter_only": "-",
        "pretrained_adapter": "-.",
    }
    for label, trace_rows in sorted(grouped.items()):
        ax.errorbar(
            [row["bin_center_gev"] for row in trace_rows],
            [row["fit_mean"] for row in trace_rows],
            yerr=[row["fit_sigma"] for row in trace_rows],
            marker="o",
            capsize=3,
            linewidth=1.4,
            linestyle=linestyles.get(trace_rows[0]["family"], "-"),
            label=label,
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set(
        xlabel="True p [GeV]",
        ylabel=r"Gaussian mean of $\theta_{reco} - \theta_{true}$ [deg]",
        title=title,
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_delta_theta_scalar(
    rows: list[dict[str, Any]],
    output_path: Path,
    y_key: str,
    title: str,
    ylabel: str,
    zero_line: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    grouped = grouped_delta_p_over_p_rows(rows)
    linestyles = {
        "conventional": "--",
        "adapter_only": "-",
        "pretrained_adapter": "-.",
    }
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    for label, trace_rows in sorted(grouped.items()):
        trace_rows = [row for row in trace_rows if row.get(y_key) is not None]
        if not trace_rows:
            continue
        ax.plot(
            [row["bin_center_gev"] for row in trace_rows],
            [row[y_key] for row in trace_rows],
            marker="o",
            linewidth=1.4,
            linestyle=linestyles.get(trace_rows[0]["family"], "-"),
            label=label,
        )
    if zero_line:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set(xlabel="True p [GeV]", ylabel=ylabel, title=title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_delta_theta_set(rows: list[dict[str, Any]], output_dir: Path, stem: str, title: str) -> None:
    plot_delta_theta_errorbars(
        rows,
        output_dir / f"{stem}.png",
        title,
    )
    plot_delta_theta_scalar(
        rows,
        output_dir / f"{stem}_mean.png",
        "fit_mean",
        f"{title}: Gaussian mean",
        r"Gaussian mean of $\theta_{reco} - \theta_{true}$ [deg]",
        zero_line=True,
    )
    plot_delta_theta_scalar(
        rows,
        output_dir / f"{stem}_sigma.png",
        "fit_sigma",
        f"{title}: Gaussian sigma",
        r"Gaussian sigma of $\theta_{reco} - \theta_{true}$ [deg]",
        zero_line=False,
    )


def select_presentation_adapter_run(rows: list[dict[str, Any]]) -> str | None:
    adapter_rows = [row for row in rows if row.get("method") == "adapter"]
    if not adapter_rows:
        return None
    adapter_only = [row for row in adapter_rows if row["family"] == "adapter_only"]
    candidates = adapter_only or adapter_rows
    return max(
        candidates,
        key=lambda row: (
            row.get("labeled_events") or -1,
            row.get("embed_dim") or -1,
            row.get("pretrain_events") or -1,
            safe_label(row.get("run_id")),
        ),
    ).get("run_id")


def plot_presentation_momentum_resolution(rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_id = select_presentation_adapter_run(rows)
    if run_id is None:
        return
    run_rows = [row for row in rows if row.get("run_id") == run_id]
    adapter_rows = sorted(
        [row for row in run_rows if row.get("method") == "adapter" and row.get("fit_sigma") is not None],
        key=lambda row: row["bin_center_gev"],
    )
    cvt_rows = sorted(
        [row for row in run_rows if row.get("method") == "cvt" and row.get("fit_sigma") is not None],
        key=lambda row: row["bin_center_gev"],
    )
    if not adapter_rows or not cvt_rows:
        return

    adapter_name = "AdapterOnly" if adapter_rows[0]["family"] == "adapter_only" else "Adapter"
    adapter_label = f"{adapter_name}, {track_count_label(adapter_rows[0].get('labeled_events'))}"
    colors = {
        "adapter": "#0072B2",
        "cvt": "#D55E00",
    }
    with plt.rc_context({
        "font.size": 20,
        "axes.labelsize": 24,
        "axes.titlesize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "lines.linewidth": 3.2,
        "lines.markersize": 9.5,
    }):
        fig, ax = plt.subplots(figsize=(10.5, 7.0), constrained_layout=True)
        ax.plot(
            [row["bin_center_gev"] for row in adapter_rows],
            [PERCENT_SCALE * row["fit_sigma"] for row in adapter_rows],
            marker="o",
            linestyle="-",
            linewidth=3.2,
            markersize=9.5,
            color=colors["adapter"],
            label=adapter_label,
        )
        ax.plot(
            [row["bin_center_gev"] for row in cvt_rows],
            [PERCENT_SCALE * row["fit_sigma"] for row in cvt_rows],
            marker="s",
            linestyle="-",
            linewidth=3.2,
            markersize=9.5,
            color=colors["cvt"],
            label="Conventional CVT reconstruction",
        )
        ax.set_xlabel("p [GeV]")
        ax.set_ylabel("σ(Δp/p) [%]")
        ax.minorticks_off()
        ax.grid(True, which="major", alpha=0.28, linewidth=1.1)
        ax.legend(frameon=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220)
        plt.close(fig)


def plot_presentation_theta_resolution(rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_id = select_presentation_adapter_run(rows)
    if run_id is None:
        return
    run_rows = [row for row in rows if row.get("run_id") == run_id]
    adapter_rows = sorted(
        [row for row in run_rows if row.get("method") == "adapter" and row.get("fit_sigma") is not None],
        key=lambda row: row["bin_center_gev"],
    )
    cvt_rows = sorted(
        [row for row in run_rows if row.get("method") == "cvt" and row.get("fit_sigma") is not None],
        key=lambda row: row["bin_center_gev"],
    )
    if not adapter_rows or not cvt_rows:
        return

    adapter_name = "AdapterOnly" if adapter_rows[0]["family"] == "adapter_only" else "Adapter"
    adapter_label = f"{adapter_name}, {track_count_label(adapter_rows[0].get('labeled_events'))}"
    colors = {
        "adapter": "#0072B2",
        "cvt": "#D55E00",
    }
    with plt.rc_context({
        "font.size": 20,
        "axes.labelsize": 24,
        "axes.titlesize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "lines.linewidth": 3.2,
        "lines.markersize": 9.5,
    }):
        fig, ax = plt.subplots(figsize=(10.5, 7.0), constrained_layout=True)
        ax.plot(
            [row["bin_center_gev"] for row in adapter_rows],
            [row["fit_sigma"] for row in adapter_rows],
            marker="o",
            linestyle="-",
            linewidth=3.2,
            markersize=9.5,
            color=colors["adapter"],
            label=adapter_label,
        )
        ax.plot(
            [row["bin_center_gev"] for row in cvt_rows],
            [row["fit_sigma"] for row in cvt_rows],
            marker="s",
            linestyle="-",
            linewidth=3.2,
            markersize=9.5,
            color=colors["cvt"],
            label="Conventional CVT reconstruction",
        )
        ax.set_xlabel("p [GeV]")
        ax.set_ylabel("σ(Δθ) [deg]")
        ax.minorticks_off()
        ax.grid(True, which="major", alpha=0.28, linewidth=1.1)
        ax.legend(frameon=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220)
        plt.close(fig)


def make_momentum_resolution_plots(delta_rows: list[dict[str, Any]], output_dir: Path) -> None:
    run_rows = normalize_delta_p_over_p_rows(delta_rows)
    rows = collapse_conventional_rows(run_rows)
    if not rows:
        return
    plot_dir = output_dir / "momentum_resolution"
    plot_presentation_momentum_resolution(
        run_rows,
        plot_dir / "presentation_sigma_delta_p_over_p.png",
    )
    groups = [
        ("conventional", "Conventional CVT::Tracks", {"conventional"}),
        ("adapter_only", "AdapterOnly", {"adapter_only"}),
        ("pretrained_adapter", "Pretrained+Adapter", {"pretrained_adapter"}),
        ("all", "Momentum resolution: all methods", {"conventional", "adapter_only", "pretrained_adapter"}),
        ("conventional_adapter_only", "Momentum resolution: CVT::Tracks and AdapterOnly", {"conventional", "adapter_only"}),
        ("conventional_pretrained_adapter", "Momentum resolution: CVT::Tracks and Pretrained+Adapter", {"conventional", "pretrained_adapter"}),
    ]
    for filename, title, families in groups:
        group_rows = [row for row in rows if row["family"] in families]
        if group_rows:
            plot_delta_p_over_p_set(
                group_rows,
                plot_dir,
                f"delta_p_over_p_{filename}",
                title,
            )

    run_plot_dir = plot_dir / "runs"
    by_run = defaultdict(list)
    for row in per_run_delta_p_over_p_rows(run_rows):
        by_run[row["run_id"]].append(row)
    for run_id, run_plot_rows in sorted(by_run.items()):
        if not run_plot_rows:
            continue
        plot_delta_p_over_p_set(
            run_plot_rows,
            run_plot_dir / safe_label(run_id),
            "delta_p_over_p",
            f"Momentum resolution: {safe_label(run_id)}",
        )


def make_theta_resolution_plots(delta_rows: list[dict[str, Any]], output_dir: Path) -> None:
    run_rows = normalize_delta_theta_rows(delta_rows)
    rows = collapse_conventional_rows(run_rows)
    if not rows:
        return
    plot_dir = output_dir / "momentum_resolution"
    plot_presentation_theta_resolution(
        run_rows,
        plot_dir / "presentation_sigma_delta_theta.png",
    )
    groups = [
        ("conventional", "Polar-angle resolution: Conventional CVT::Tracks", {"conventional"}),
        ("adapter_only", "Polar-angle resolution: AdapterOnly", {"adapter_only"}),
        ("pretrained_adapter", "Polar-angle resolution: Pretrained+Adapter", {"pretrained_adapter"}),
        ("all", "Polar-angle resolution: all methods", {"conventional", "adapter_only", "pretrained_adapter"}),
        ("conventional_adapter_only", "Polar-angle resolution: CVT::Tracks and AdapterOnly", {"conventional", "adapter_only"}),
        ("conventional_pretrained_adapter", "Polar-angle resolution: CVT::Tracks and Pretrained+Adapter", {"conventional", "pretrained_adapter"}),
    ]
    for filename, title, families in groups:
        group_rows = [row for row in rows if row["family"] in families]
        if group_rows:
            plot_delta_theta_set(
                group_rows,
                plot_dir,
                f"delta_theta_{filename}",
                title,
            )

    run_plot_dir = plot_dir / "runs"
    by_run = defaultdict(list)
    for row in per_run_delta_theta_rows(run_rows):
        by_run[row["run_id"]].append(row)
    for run_id, run_plot_rows in sorted(by_run.items()):
        if not run_plot_rows:
            continue
        plot_delta_theta_set(
            run_plot_rows,
            run_plot_dir / safe_label(run_id),
            "delta_theta",
            f"Polar-angle resolution: {safe_label(run_id)}",
        )


def main() -> None:
    args = parse_args()
    campaign_dir, headline_jsonl, manifest_path, output_dir = resolve_paths(args)
    if args.plot_suite in ("standard", "all") and not headline_jsonl.exists():
        raise FileNotFoundError(f"Campaign headline JSONL does not exist: {headline_jsonl}")
    manifest_rows = manifest_lookup(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_suite in ("standard", "all"):
        metric_rows = read_jsonl(headline_jsonl)
        rows = build_plot_rows(metric_rows, manifest_rows)
        if not rows:
            raise RuntimeError(f"No completed adapter ml_error rows found in {headline_jsonl}")
        write_csv(output_dir / "plot_data.csv", rows)
        write_csv(output_dir / "best_by_slice.csv", best_by_slice(rows))
        make_plots(rows, output_dir)

    if args.plot_suite in ("momentum-resolution", "all"):
        if campaign_dir is None:
            delta_path = headline_jsonl.parent / "delta_p_over_p_fits.csv"
            delta_theta_path = headline_jsonl.parent / "delta_theta_fits.csv"
        else:
            delta_path = campaign_dir / "summary" / "delta_p_over_p_fits.csv"
            delta_theta_path = campaign_dir / "summary" / "delta_theta_fits.csv"
        if not delta_path.exists():
            raise FileNotFoundError(
                f"Delta-p/p fit CSV does not exist: {delta_path}. "
                "Run campaign collation after reevaluating runs with this feature."
            )
        make_momentum_resolution_plots(read_csv_rows(delta_path), output_dir)
        if delta_theta_path.exists():
            make_theta_resolution_plots(read_csv_rows(delta_theta_path), output_dir)
    print(f"Wrote campaign plots and CSVs to {output_dir}")


if __name__ == "__main__":
    main()
