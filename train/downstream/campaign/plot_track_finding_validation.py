#!/usr/bin/env python3
"""Plot a single training trajectory and report ARI/background trade-offs."""

import argparse
import csv
import json
import math
from pathlib import Path


METRICS = {
    "ari_signal": "Signal ARI",
    "ari_with_background": "Inclusive ARI",
    "track_purity_global": "Global track purity",
    "track_efficiency_global": "Global track efficiency",
    "matched_iou_mean": "Matched IoU",
    "matched_purity_mean": "Matched cluster purity",
    "background_rejection": "Background rejection",
    "background_contamination": "Background contamination",
    "signal_loss_to_background": "Signal loss to background",
    "fake_rate": "Fake-track rate",
    "split_rate": "Split rate",
    "merge_rate": "Merge rate",
}
LOWER_IS_BETTER = {"background_contamination", "signal_loss_to_background", "fake_rate", "split_rate", "merge_rate"}


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def read_history(path, option=2):
    path = Path(path)
    if path.is_dir():
        path = path / "metrics.jsonl"
    with path.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()] if path.suffix == ".jsonl" else list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Validation history is empty")
    prefix = f"option{option}_"
    for row in rows:
        if number(row.get(prefix + "ari_signal")) is None:
            raise ValueError("History lacks full validation metrics; legacy ARI-only logs cannot recover them")
    # Joining differing validation samples/operating points can manufacture a trend.
    for key in ("history_id", "assignment_threshold", "match_iou_threshold", "match_min_purity", "match_min_efficiency",
                prefix + "n_events", prefix + "n_points", prefix + "n_true_tracks"):
        if len({str(row.get(key)) for row in rows}) != 1:
            raise ValueError(f"Cannot compare validation points with differing {key}")
    rows.sort(key=lambda row: int(row["validation_index"]))
    if len({int(row["validation_index"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate validation indices; supply one training history")
    return rows


def trend_summary(rows, option=2, min_delta=1e-4):
    prefix = f"option{option}_"
    best = {}
    for metric in METRICS:
        available = [row for row in rows if number(row.get(prefix + metric)) is not None]
        if available:
            selector = min if metric in LOWER_IS_BETTER else max
            row = selector(available, key=lambda item: number(item[prefix + metric]))
            best[metric] = {key: row.get(key) for key in ("validation_index", "step", "epoch", "checkpoint")}
            best[metric]["value"] = number(row[prefix + metric])
    intervals = []
    for before, after in zip(rows, rows[1:]):
        delta_ari = number(after[prefix + "ari_signal"]) - number(before[prefix + "ari_signal"])
        if delta_ari <= min_delta:
            continue
        degraded = {}
        for metric in ("track_purity_global", "background_rejection", "matched_purity_mean",
                       "fake_rate", "background_contamination", "signal_loss_to_background"):
            previous, current = number(before.get(prefix + metric)), number(after.get(prefix + metric))
            if previous is None or current is None:
                continue
            delta = current - previous
            if (delta if metric in LOWER_IS_BETTER else -delta) > min_delta:
                degraded[metric] = delta
        if degraded:
            intervals.append({"from_step": int(before["step"]), "to_step": int(after["step"]),
                              "delta_signal_ari": delta_ari, "degraded_metric_deltas": degraded,
                              "checkpoint": after.get("checkpoint")})
    return {"assignment_option": option, "assignment_threshold": number(rows[0].get("assignment_threshold")),
            "validation_points": len(rows), "min_delta": min_delta,
            "interpretation": "Descriptive validation comparisons within one training trajectory; not independent test results or proof of causality.",
            "best_by_metric": best, "ari_rising_degradation_intervals": intervals}


def plot_history(rows, output_dir, option=2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"option{option}_"
    steps = np.asarray([int(row["step"]) for row in rows])

    def values(metric):
        return np.asarray([number(row.get(prefix + metric)) if number(row.get(prefix + metric)) is not None else np.nan for row in rows])

    selected = [index for index, row in enumerate(rows) if str(row.get("selected_best")).lower() == "true"]
    selected = selected[-1] if selected else None
    groups = (("ari_signal", "ari_with_background"),
              ("track_purity_global", "track_efficiency_global"),
              ("background_rejection", "signal_loss_to_background"),
              ("matched_iou_mean", "matched_purity_mean"),
              ("fake_rate", "split_rate", "merge_rate"),
              ("background_contamination",))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, metrics in zip(axes.flat, groups):
        for metric in metrics:
            ax.plot(steps, values(metric), marker=".", label=METRICS[metric])
        if selected is not None:
            ax.axvline(steps[selected], color="black", linestyle=":", alpha=.5)
        ax.set_xlabel("Optimizer step")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    threshold = number(rows[0].get("assignment_threshold"))
    fig.suptitle(f"Validation trajectory · assignment option {option} · threshold {threshold:g}\nDotted line: final checkpoint selected during training")
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"metrics_vs_step.{extension}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    ari = values("ari_signal")
    for ax, metric in zip(axes, ("track_purity_global", "background_rejection", "fake_rate")):
        y = values(metric)
        ax.plot(ari, y, color="0.7", linewidth=1, zorder=0)
        points = ax.scatter(ari, y, c=steps, cmap="viridis")
        if selected is not None:
            ax.scatter(ari[selected], y[selected], marker="*", s=150, c="orange", edgecolors="black", zorder=3)
        ax.set_xlabel("Signal ARI")
        ax.set_ylabel(METRICS[metric])
        ax.grid(alpha=.2)
    fig.colorbar(points, ax=axes, label="Optimizer step", shrink=.8)
    fig.suptitle("Validation trade-offs · connected in training order · star: selected checkpoint")
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"background_vs_signal_ari.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True, type=Path, help="One run's metrics.jsonl/CSV or history directory")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--assignment-option", type=int, choices=(1, 2), default=2)
    parser.add_argument("--after-step", type=int, default=0, help="Restrict to a later training interval")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Absolute change required to flag a trade-off interval")
    args = parser.parse_args()
    if args.min_delta < 0:
        parser.error("--min-delta must be non-negative")
    rows = read_history(args.history, args.assignment_option)
    rows = [row for row in rows if int(row["step"]) >= args.after_step]
    if not rows:
        parser.error("No validation records in the requested step range")
    out = args.output_dir or ((args.history if args.history.is_dir() else args.history.parent) / "plots")
    plot_history(rows, out, args.assignment_option)
    summary = trend_summary(rows, args.assignment_option, args.min_delta)
    (out / "trend_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Wrote plots and descriptive trade-off intervals to {out}")


if __name__ == "__main__":
    main()
