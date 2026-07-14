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

from campaign_util import read_yaml


LABEL_RE = re.compile(r"(?:^|_)label(?P<label>\d+)(?:_|$)")
COMPONENTS = ("px_gev", "py_gev", "pz_gev")
COMPONENT_LABELS = {
    "px_gev": "px",
    "py_gev": "py",
    "pz_gev": "pz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", help="Campaign directory containing manifest.yaml and summary/.")
    parser.add_argument("--manifest", help="Optional explicit manifest path.")
    parser.add_argument("--headline-jsonl", help="Optional explicit campaign_headline_metrics.jsonl path.")
    parser.add_argument("--output-dir", help="Optional explicit plot output directory.")
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
    adapter_components: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in metric_rows:
        run_id = row.get("run_name") or row.get("run_num")
        if not run_id:
            continue
        if (
            row.get("record_type") == "ml_error"
            and row.get("method") == "adapter"
            and row.get("space") == "component"
            and row.get("variable") in COMPONENTS
        ):
            adapter_components[run_id][row["variable"]] = row

    count_params = parameter_counter()
    rows = []
    for run_id, component_metrics in sorted(adapter_components.items()):
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
        }
        rmse_values = []
        r2_values = []
        for component in COMPONENTS:
            metric = component_metrics.get(component, {})
            rmse = finite_float(metric.get("rmse"))
            r2 = finite_float(metric.get("r2"))
            row[f"adapter_rmse_{component}"] = rmse
            row[f"adapter_r2_{component}"] = r2
            if rmse is not None:
                rmse_values.append(rmse)
            if r2 is not None:
                r2_values.append(r2)
        row["adapter_rmse_mean"] = sum(rmse_values) / len(rmse_values) if rmse_values else None
        row["adapter_r2_mean"] = sum(r2_values) / len(r2_values) if r2_values else None
        if count_params is not None and row["embed_dim"] is not None:
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
        facet_values = sorted({row.get(facet_key) for row in filtered}, key=lambda value: (value is None, value))

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
                label=f"{line_key}={safe_label(line_value)}",
            )
        if reference_y is not None:
            axis.axhline(reference_y, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
        if x_log:
            axis.set_xscale("log")
        axis.grid(alpha=0.25)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        if facet_key is not None:
            axis.set_title(f"{facet_key}={safe_label(facet_value)}")
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

    for metric, ylabel, better in (
        ("rmse", "RMSE [GeV]", "lower"),
        ("r2", "R2", "higher"),
    ):
        mean_key = f"adapter_{metric}_mean"
        plot_faceted_lines(
            rows,
            output_dir / f"{metric}_mean_vs_labeled_events_by_width.png",
            x_key="labeled_events",
            y_key=mean_key,
            line_key="embed_dim",
            facet_key="pretrain_events",
            title=f"Adapter mean {ylabel} vs labeled adapter data ({better} is better)",
            ylabel=f"Mean {ylabel} across px, py, pz",
            xlabel="Adapter labeled events",
        )
        plot_faceted_lines(
            rows,
            output_dir / f"{metric}_mean_vs_backbone_params_by_labeled_events.png",
            x_key=size_x,
            y_key=mean_key,
            line_key="labeled_events",
            facet_key="pretrain_events",
            title=f"Adapter mean {ylabel} vs backbone size ({better} is better)",
            ylabel=f"Mean {ylabel} across px, py, pz",
            xlabel=size_xlabel,
        )
        plot_faceted_lines(
            rows,
            output_dir / f"{metric}_mean_vs_pretrain_events_by_labeled_events.png",
            x_key="pretrain_events",
            y_key=mean_key,
            line_key="labeled_events",
            facet_key="embed_dim",
            title=f"Adapter mean {ylabel} vs backbone pretraining data ({better} is better)",
            ylabel=f"Mean {ylabel} across px, py, pz",
            xlabel="Backbone pretraining events",
        )
        for component in COMPONENTS:
            component_key = f"adapter_{metric}_{component}"
            component_label = COMPONENT_LABELS[component]
            plot_faceted_lines(
                rows,
                output_dir / f"{metric}_{component}_vs_labeled_events_by_width.png",
                x_key="labeled_events",
                y_key=component_key,
                line_key="embed_dim",
                facet_key="pretrain_events",
                title=f"Adapter {component_label} {ylabel} vs labeled adapter data ({better} is better)",
                ylabel=f"{component_label} {ylabel}",
                xlabel="Adapter labeled events",
            )
            plot_faceted_lines(
                rows,
                output_dir / f"{metric}_{component}_vs_backbone_params_by_labeled_events.png",
                x_key=size_x,
                y_key=component_key,
                line_key="labeled_events",
                facet_key="pretrain_events",
                title=f"Adapter {component_label} {ylabel} vs backbone size ({better} is better)",
                ylabel=f"{component_label} {ylabel}",
                xlabel=size_xlabel,
            )
            plot_faceted_lines(
                rows,
                output_dir / f"{metric}_{component}_vs_pretrain_events_by_labeled_events.png",
                x_key="pretrain_events",
                y_key=component_key,
                line_key="labeled_events",
                facet_key="embed_dim",
                title=f"Adapter {component_label} {ylabel} vs backbone pretraining data ({better} is better)",
                ylabel=f"{component_label} {ylabel}",
                xlabel="Backbone pretraining events",
            )


def main() -> None:
    args = parse_args()
    _, headline_jsonl, manifest_path, output_dir = resolve_paths(args)
    if not headline_jsonl.exists():
        raise FileNotFoundError(f"Campaign headline JSONL does not exist: {headline_jsonl}")
    metric_rows = read_jsonl(headline_jsonl)
    manifest_rows = manifest_lookup(manifest_path)
    rows = build_plot_rows(metric_rows, manifest_rows)
    if not rows:
        raise RuntimeError(f"No completed adapter component ml_error rows found in {headline_jsonl}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "plot_data.csv", rows)
    write_csv(output_dir / "best_by_slice.csv", best_by_slice(rows))
    make_plots(rows, output_dir)
    print(f"Wrote campaign plots and CSVs to {output_dir}")


if __name__ == "__main__":
    main()
