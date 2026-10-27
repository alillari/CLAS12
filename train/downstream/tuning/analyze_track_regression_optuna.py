#!/usr/bin/env python3
"""Analyze a completed AdapterOnly track-regression Optuna study."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from train.downstream.tuning.build_track_regression_seed_ablation import (  # noqa: E402
    complete_trials,
)

LOGGER = logging.getLogger("optuna-study-analysis")
LOG_SCALE_HINTS = {"lr", "weight_decay", "decay", "clip"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", required=True, help="Optuna storage URL, e.g. sqlite:////path/study.db")
    parser.add_argument("--study-name", required=True, help="Optuna study name")
    parser.add_argument(
        "--study-dir",
        help="Directory containing per-trial outputs. Defaults to sibling directory named like --study-name.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for plots and tables. Defaults to <study-dir>/analysis/optuna_diagnostics.",
    )
    parser.add_argument("--min-step-trials", type=int, default=2, help="Minimum trials required to draw step statistics")
    parser.add_argument(
        "--parallel-filter",
        choices=["all", "top-fraction", "absolute", "relative"],
        default="top-fraction",
        help="Completed-trial filter for parallel-coordinate and interaction plots.",
    )
    parser.add_argument("--top-fraction", type=float, default=0.2, help="Retain this best fraction for top-fraction plots")
    parser.add_argument("--absolute-threshold", type=float, help="Retain trials with objective at or below this value")
    parser.add_argument(
        "--relative-pct",
        type=float,
        default=5.0,
        help="Retain trials within this percent of the best objective for relative filtering",
    )
    parser.add_argument("--min-filtered-trials", type=int, default=4, help="Minimum filtered trials needed for filtered plots")
    parser.add_argument("--importance-seed", type=int, default=42)
    parser.add_argument("--top-interaction-params", type=int, default=4)
    parser.add_argument(
        "--interaction-params",
        help="Comma-separated parameter list for pairwise interactions. Defaults to top importance parameters.",
    )
    parser.add_argument(
        "--checkpoint-fractions",
        default="0.2,0.4,0.6,0.8",
        help="Comma-separated fractions of the inferred step budget for rank-stability plots",
    )
    parser.add_argument(
        "--checkpoint-tolerance-fraction",
        type=float,
        default=0.05,
        help="Nearest-step tolerance as a fraction of inferred budget for checkpoint plots",
    )
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def configure_runtime(output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    })
    return plt


def resolve_study_dir(storage: str, study_name: str, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if storage.startswith("sqlite:///"):
        db_path = Path(storage.removeprefix("sqlite:///")).expanduser().resolve()
        candidate = db_path.parent / study_name
        if candidate.exists():
            return candidate
    return None


def default_output_dir(study_dir: Path | None, storage: str, study_name: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if study_dir is not None:
        return study_dir / "analysis" / "optuna_diagnostics"
    if storage.startswith("sqlite:///"):
        return Path(storage.removeprefix("sqlite:///")).expanduser().resolve().parent / study_name / "analysis" / "optuna_diagnostics"
    return Path("optuna_diagnostics") / study_name


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
    tmp.replace(path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def direction_is_minimize(study: Any) -> bool:
    import optuna

    return study.direction == optuna.study.StudyDirection.MINIMIZE


def objective_sort_key(study: Any, trial: Any) -> float:
    value = float(trial.value)
    return value if direction_is_minimize(study) else -value


def best_value(values: np.ndarray, minimize: bool) -> float:
    return float(np.nanmin(values) if minimize else np.nanmax(values))


def better_or_equal(values: np.ndarray, threshold: float, minimize: bool) -> np.ndarray:
    return values <= threshold if minimize else values >= threshold


def completed_trial_rows(study: Any, trials: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for trial in trials:
        row = {
            "number": trial.number,
            "value": float(trial.value),
            "best_step": trial.user_attrs.get("best_step"),
            "best_epoch": trial.user_attrs.get("best_epoch"),
            "trial_dir": trial.user_attrs.get("trial_dir"),
            "log_file": trial.user_attrs.get("log_file"),
            "artifact_summary": trial.user_attrs.get("artifact_summary"),
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        rows.append(row)
    return sorted(rows, key=lambda row: objective_sort_key(study, next(t for t in trials if t.number == row["number"])))


def parameter_names(trials: list[Any]) -> list[str]:
    ordered: list[str] = []
    for trial in trials:
        for key in trial.params:
            if key not in ordered:
                ordered.append(key)
    return ordered


def is_log_distribution(trials: list[Any], param: str) -> bool:
    for trial in trials:
        distribution = trial.distributions.get(param)
        if distribution is not None and bool(getattr(distribution, "log", False)):
            return True
    lowered = param.lower()
    return any(hint in lowered for hint in LOG_SCALE_HINTS)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def candidate_trial_dirs(trial: Any, study_dir: Path | None) -> list[Path]:
    paths: list[Path] = []
    attr_dir = trial.user_attrs.get("trial_dir")
    if attr_dir:
        paths.append(Path(attr_dir))
    if study_dir is not None:
        paths.append(study_dir / f"trial_{trial.number:06d}")
    unique: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def source_log_path(trial: Any, study_dir: Path | None) -> Path | None:
    attr_path = trial.user_attrs.get("log_file")
    candidates = [Path(attr_path)] if attr_path else []
    for trial_dir in candidate_trial_dirs(trial, study_dir):
        candidates.extend(sorted((trial_dir / "checkpoints").glob("*.log")))
    for path in candidates:
        if path.is_file():
            return path
    return None


def source_config_path(trial: Any, study_dir: Path | None) -> Path | None:
    for trial_dir in candidate_trial_dirs(trial, study_dir):
        path = trial_dir / "config" / "resolved_config.json"
        if path.is_file():
            return path
    return None


def source_artifact_path(trial: Any, study_dir: Path | None) -> Path | None:
    attr_path = trial.user_attrs.get("artifact_summary")
    candidates = [Path(attr_path)] if attr_path else []
    for trial_dir in candidate_trial_dirs(trial, study_dir):
        candidates.append(trial_dir / "train" / "artifacts.json")
        candidates.append(trial_dir / "trial_result.json")
    for path in candidates:
        if path.is_file():
            return path
    return None


def parse_training_log(path: Path) -> dict[int, float]:
    rows: dict[int, float] = {}
    try:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            for row in reader:
                step = finite_float(row.get("Step") or row.get("step"))
                val_loss = finite_float(row.get("Val_Loss") or row.get("val_loss") or row.get("Validation_Loss"))
                if step is not None and val_loss is not None:
                    rows[int(step)] = val_loss
    except OSError:
        return {}
    return rows


def validation_history(trial: Any, study_dir: Path | None) -> dict[int, float]:
    if trial.intermediate_values:
        return {
            int(step): float(value)
            for step, value in trial.intermediate_values.items()
            if finite_float(value) is not None
        }
    log_path = source_log_path(trial, study_dir)
    return parse_training_log(log_path) if log_path is not None else {}


def infer_max_step_budget(trials: list[Any], study_dir: Path | None) -> int | None:
    values: list[int] = []
    keys = ("max_optimizer_steps", "scheduler_first_cycle_steps", "final_step")
    for trial in trials:
        for attr in ("best_step",):
            value = finite_float(trial.user_attrs.get(attr))
            if value is not None:
                values.append(int(value))
        for path_getter in (source_config_path, source_artifact_path):
            path = path_getter(trial, study_dir)
            if path is None:
                continue
            data = load_json(path)
            if not data:
                continue
            for key in keys:
                value = finite_float(data.get(key))
                if value is not None:
                    values.append(int(value))
    return max(values) if values else None


def actual_max_reported_steps(histories: dict[int, dict[int, float]]) -> dict[str, Any]:
    maxima = [max(history) for history in histories.values() if history]
    if not maxima:
        return {"found": False}
    arr = np.asarray(maxima, dtype=float)
    return {
        "found": True,
        "min": int(np.min(arr)),
        "median": float(np.median(arr)),
        "max": int(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def aggregate_validation_histories(histories: dict[int, dict[int, float]], min_trials: int) -> list[dict[str, Any]]:
    by_step: dict[int, list[float]] = defaultdict(list)
    for history in histories.values():
        for step, value in history.items():
            by_step[int(step)].append(float(value))
    rows: list[dict[str, Any]] = []
    for step in sorted(by_step):
        values = np.asarray(by_step[step], dtype=float)
        count = int(values.size)
        rows.append({
            "step": int(step),
            "n_trials": count,
            "draw": count >= min_trials,
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        })
    return rows


def plot_validation_aggregate(rows: list[dict[str, Any]], output_dir: Path, study_name: str, min_trials: int, dpi: int) -> Path | None:
    plot_rows = [row for row in rows if row["draw"]]
    if not plot_rows:
        LOGGER.warning("Skipping aggregate validation plot: no step has at least %s contributing trials", min_trials)
        return None
    plt = import_pyplot()
    steps = np.asarray([row["step"] for row in plot_rows], dtype=float)
    median = np.asarray([row["median"] for row in plot_rows], dtype=float)
    mean = np.asarray([row["mean"] for row in plot_rows], dtype=float)
    p10 = np.asarray([row["p10"] for row in plot_rows], dtype=float)
    p25 = np.asarray([row["p25"] for row in plot_rows], dtype=float)
    p75 = np.asarray([row["p75"] for row in plot_rows], dtype=float)
    p90 = np.asarray([row["p90"] for row in plot_rows], dtype=float)
    counts = np.asarray([row["n_trials"] for row in plot_rows], dtype=int)

    fig, (ax, count_ax) = plt.subplots(
        2, 1, figsize=(9, 6.3), sharex=True, height_ratios=(4, 1), constrained_layout=True
    )
    ax.fill_between(steps, p10, p90, color="#4C78A8", alpha=0.12, label="10th-90th percentile")
    ax.fill_between(steps, p25, p75, color="#4C78A8", alpha=0.26, label="25th-75th percentile")
    ax.plot(steps, median, color="#1F4E79", linewidth=2.2, marker="o", markersize=3.5, label="Median")
    ax.plot(steps, mean, color="#E45756", linewidth=1.6, linestyle="--", marker="s", markersize=3, label="Mean")
    ax.set_ylabel("Validation loss")
    ax.set_title(f"{study_name}: aggregate validation loss by reported optimizer step")
    ax.legend(loc="best", fontsize=9)
    count_ax.bar(steps, counts, width=max(1.0, np.min(np.diff(steps)) * 0.65) if len(steps) > 1 else 1.0, color="#72B7B2")
    count_ax.set_ylabel("Trials")
    count_ax.set_xlabel("Optimizer step")
    count_ax.set_ylim(0, max(counts) * 1.18)
    path = output_dir / "aggregate_validation_loss_by_step.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_optimization_history(study: Any, trials: list[Any], output_dir: Path, study_name: str, budget: int | None, dpi: int) -> Path | None:
    if not trials:
        LOGGER.warning("Skipping optimization history: no completed trials")
        return None
    plt = import_pyplot()
    ordered = sorted(trials, key=lambda trial: trial.number)
    numbers = np.asarray([trial.number for trial in ordered], dtype=int)
    values = np.asarray([float(trial.value) for trial in ordered], dtype=float)
    minimize = direction_is_minimize(study)
    running = np.minimum.accumulate(values) if minimize else np.maximum.accumulate(values)
    best_idx = int(np.argmin(values) if minimize else np.argmax(values))
    budget_text = f", max step budget {budget}" if budget is not None else ""

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    ax.scatter(numbers, values, s=34, color="#4C78A8", alpha=0.8, label="Completed trial")
    ax.plot(numbers, running, color="#E45756", linewidth=2, label="Best so far")
    ax.scatter([numbers[best_idx]], [values[best_idx]], s=110, color="#F2A541", edgecolor="black", zorder=5, label="Best trial")
    ax.annotate(
        f"trial {numbers[best_idx]}\n{values[best_idx]:.6g}",
        xy=(numbers[best_idx], values[best_idx]),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Objective")
    ax.set_title(f"{study_name}: optimization history ({len(trials)} completed{budget_text})")
    ax.legend(loc="best", fontsize=9)
    path = output_dir / "optimization_history.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_objective_distribution(study: Any, trials: list[Any], output_dir: Path, study_name: str, dpi: int) -> Path | None:
    if not trials:
        LOGGER.warning("Skipping objective distribution: no completed trials")
        return None
    plt = import_pyplot()
    values = np.asarray([float(trial.value) for trial in trials], dtype=float)
    minimize = direction_is_minimize(study)
    best = best_value(values, minimize)
    median = float(np.median(values))
    mean = float(np.mean(values))
    q25, q75 = np.percentile(values, [25, 75])

    fig, ax = plt.subplots(figsize=(5.8, 5.4), constrained_layout=True)
    parts = ax.violinplot([values], positions=[1], widths=0.68, showextrema=False, showmedians=False)
    for body in parts["bodies"]:
        body.set_facecolor("#4C78A8")
        body.set_edgecolor("#1F4E79")
        body.set_alpha(0.25)
    jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) <= 80 else np.zeros(len(values))
    ax.scatter(1 + jitter, values, s=20 if len(values) <= 80 else 10, alpha=0.55, color="#4C78A8", label="Trial")
    ax.plot([0.78, 1.22], [median, median], color="black", linewidth=2.0, label="Median")
    ax.scatter([1], [mean], marker="D", s=55, color="#E45756", label="Mean", zorder=4)
    ax.scatter([1], [best], marker="*", s=155, color="#F2A541", edgecolor="black", label="Best", zorder=5)
    ax.plot([1, 1], [q25, q75], color="black", linewidth=7, alpha=0.3, solid_capstyle="butt", label="IQR")
    ax.set_xlim(0.55, 1.45)
    ax.set_xticks([1])
    ax.set_xticklabels(["Completed trials"])
    ax.set_ylabel("Best validation loss")
    ax.set_title(f"{study_name}: final-objective distribution ({len(trials)} completed)")
    ax.legend(loc="best", fontsize=9)
    path = output_dir / "objective_distribution.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def compute_importance(study: Any, output_dir: Path, seed: int) -> tuple[dict[str, float], str | None]:
    import optuna

    evaluator_name = "FanovaImportanceEvaluator"
    try:
        evaluator = optuna.importance.FanovaImportanceEvaluator(seed=seed)
        importance = optuna.importance.get_param_importances(
            study,
            evaluator=evaluator,
            params=None,
            target=lambda trial: float(trial.value),
        )
    except Exception as exc:
        LOGGER.warning("Parameter importance failed with %s: %s", evaluator_name, exc)
        write_json(output_dir / "parameter_importance_error.json", {
            "evaluator": evaluator_name,
            "seed": seed,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        return {}, evaluator_name
    rows = [{"parameter": key, "importance": float(value), "evaluator": evaluator_name, "seed": seed} for key, value in importance.items()]
    write_csv(output_dir / "parameter_importance.csv", rows, ["parameter", "importance", "evaluator", "seed"])
    return {key: float(value) for key, value in importance.items()}, evaluator_name


def plot_importance(importance: dict[str, float], evaluator_name: str | None, output_dir: Path, study_name: str, dpi: int) -> Path | None:
    if not importance:
        return None
    plt = import_pyplot()
    items = list(reversed(sorted(importance.items(), key=lambda item: item[1])))
    labels = [item[0] for item in items]
    values = np.asarray([item[1] for item in items], dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, max(4.0, 0.42 * len(labels) + 1.8)), constrained_layout=True)
    ax.barh(labels, values, color="#4C78A8")
    ax.set_xlabel("Importance")
    ax.set_title(f"{study_name}: parameter importance ({evaluator_name})")
    path = output_dir / "parameter_importance.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def trial_param_array(trials: list[Any], param: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for trial in trials:
        value = finite_float(trial.params.get(param))
        if value is None or trial.value is None:
            continue
        xs.append(value)
        ys.append(float(trial.value))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def binned_medians(x: np.ndarray, y: np.ndarray, log_x: bool, bins: int = 8) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < 6 or float(np.nanmin(x)) == float(np.nanmax(x)):
        return np.asarray([]), np.asarray([])
    basis = np.log10(x) if log_x else x
    edges = np.linspace(float(np.min(basis)), float(np.max(basis)), min(bins, max(3, len(x) // 3)) + 1)
    centers: list[float] = []
    medians: list[float] = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (basis >= low) & (basis <= high if high == edges[-1] else basis < high)
        if np.count_nonzero(mask) < 2:
            continue
        center = (low + high) / 2
        centers.append(10**center if log_x else center)
        medians.append(float(np.median(y[mask])))
    return np.asarray(centers), np.asarray(medians)


def plot_parameter_slices(study: Any, trials: list[Any], params: list[str], output_dir: Path, study_name: str, dpi: int) -> list[Path]:
    plt = import_pyplot()
    paths: list[Path] = []
    minimize = direction_is_minimize(study)
    best = best_value(np.asarray([float(t.value) for t in trials], dtype=float), minimize)
    rows: list[dict[str, Any]] = []
    for param in params:
        x, y = trial_param_array(trials, param)
        if len(x) == 0:
            continue
        log_x = is_log_distribution(trials, param) and np.all(x > 0)
        for xv, yv in zip(x, y):
            rows.append({"parameter": param, "param_value": float(xv), "objective": float(yv), "log_axis": log_x})
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        ax.scatter(x, y, s=30, color="#4C78A8", alpha=0.72)
        centers, medians = binned_medians(x, y, log_x)
        if len(centers) > 0:
            ax.plot(centers, medians, color="#E45756", linewidth=2, marker="o", label="Binned median")
            ax.legend(loc="best", fontsize=9)
        ax.axhline(best, color="#F2A541", linestyle=":", linewidth=1.5, label="Best observed")
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(param)
        ax.set_ylabel("Best validation loss")
        ax.set_title(f"{study_name}: {param} versus objective ({len(x)} completed)")
        path = output_dir / "parameter_slices" / f"slice_{param}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)
    write_csv(output_dir / "parameter_slice_data.csv", rows, ["parameter", "param_value", "objective", "log_axis"])
    return paths


def filter_trials(study: Any, trials: list[Any], mode: str, top_fraction: float, absolute_threshold: float | None, relative_pct: float) -> tuple[list[Any], str, str]:
    if mode == "all":
        return list(trials), "all completed trials", "all"
    if not trials:
        return [], "no completed trials", mode
    sorted_trials = sorted(trials, key=lambda trial: objective_sort_key(study, trial))
    values = np.asarray([float(trial.value) for trial in trials], dtype=float)
    minimize = direction_is_minimize(study)
    best = best_value(values, minimize)
    if mode == "top-fraction":
        keep = max(1, int(math.ceil(len(sorted_trials) * max(0.0, min(1.0, top_fraction)))))
        return sorted_trials[:keep], f"top {top_fraction:.0%} completed trials", f"top{int(round(top_fraction * 100))}pct"
    if mode == "absolute":
        if absolute_threshold is None:
            LOGGER.warning("No --absolute-threshold supplied; falling back to all completed trials")
            return list(trials), "all completed trials (missing absolute threshold)", "all"
        mask = better_or_equal(values, absolute_threshold, minimize)
        selected = [trial for trial, keep in zip(trials, mask) if keep]
        return selected, f"objective {'<=' if minimize else '>='} {absolute_threshold:.6g}", f"abs_{absolute_threshold:.6g}"
    if mode == "relative":
        if minimize:
            threshold = best * (1.0 + relative_pct / 100.0) if best >= 0 else best * (1.0 - relative_pct / 100.0)
        else:
            threshold = best * (1.0 - relative_pct / 100.0) if best >= 0 else best * (1.0 + relative_pct / 100.0)
        mask = better_or_equal(values, threshold, minimize)
        selected = [trial for trial, keep in zip(trials, mask) if keep]
        return selected, f"within {relative_pct:.3g}% of best", f"within{relative_pct:g}pct"
    raise ValueError(f"Unknown filter mode: {mode}")


def scaled_param_values(trials: list[Any], params: list[str]) -> tuple[dict[str, np.ndarray], dict[str, tuple[float, float, bool]]]:
    scaled: dict[str, np.ndarray] = {}
    ranges: dict[str, tuple[float, float, bool]] = {}
    for param in params:
        values = np.asarray([finite_float(trial.params.get(param)) for trial in trials], dtype=object)
        numeric = np.asarray([float(value) if value is not None else np.nan for value in values], dtype=float)
        log_axis = is_log_distribution(trials, param) and np.all(numeric[np.isfinite(numeric)] > 0)
        basis = np.log10(numeric) if log_axis else numeric
        finite = basis[np.isfinite(basis)]
        if finite.size == 0:
            continue
        low = float(np.min(finite))
        high = float(np.max(finite))
        ranges[param] = ((10**low if log_axis else low), (10**high if log_axis else high), log_axis)
        scaled[param] = np.full_like(basis, 0.5) if high == low else (basis - low) / (high - low)
    return scaled, ranges


def plot_parallel_coordinates(study: Any, trials: list[Any], params: list[str], output_dir: Path, study_name: str, filter_label: str, filter_slug: str, dpi: int) -> Path | None:
    if len(trials) < 2 or len(params) < 2:
        LOGGER.warning("Skipping parallel-coordinate plot: insufficient filtered trials or parameters")
        return None
    plt = import_pyplot()
    scaled, ranges = scaled_param_values(trials, params)
    params = [param for param in params if param in scaled]
    if len(params) < 2:
        LOGGER.warning("Skipping parallel-coordinate plot: fewer than two numeric parameter axes")
        return None
    write_csv(
        output_dir / f"parallel_coordinates_{filter_slug}_ranges.csv",
        [
            {
                "parameter": param,
                "low": ranges[param][0],
                "high": ranges[param][1],
                "log_axis": ranges[param][2],
                "filter": filter_label,
                "n_trials": len(trials),
            }
            for param in params
        ],
        ["parameter", "low", "high", "log_axis", "filter", "n_trials"],
    )
    values = np.asarray([float(trial.value) for trial in trials], dtype=float)
    order = np.argsort(values if direction_is_minimize(study) else -values)
    cmap = plt.get_cmap("viridis_r" if direction_is_minimize(study) else "viridis")
    norm = plt.Normalize(vmin=float(np.min(values)), vmax=float(np.max(values)))
    fig, ax = plt.subplots(figsize=(max(8.5, 1.25 * len(params)), 5.8), constrained_layout=True)
    xs = np.arange(len(params))
    for idx in order[::-1]:
        ys = [scaled[param][idx] for param in params]
        if np.any(~np.isfinite(ys)):
            continue
        ax.plot(xs, ys, color=cmap(norm(values[idx])), alpha=0.58, linewidth=1.3)
    ax.set_xticks(xs)
    ax.set_xticklabels(params, rotation=30, ha="right")
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["low", "mid", "high"])
    ax.set_ylabel("Scaled parameter value")
    ax.set_title(f"{study_name}: parallel coordinates, {filter_label} (n={len(trials)})")
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    colorbar.set_label("Objective")
    path = output_dir / f"parallel_coordinates_{filter_slug}_n{len(trials)}.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def selected_interaction_params(params: list[str], importance: dict[str, float], explicit: str | None, top_n: int) -> list[str]:
    if explicit:
        requested = [item.strip() for item in explicit.split(",") if item.strip()]
        return [param for param in requested if param in params]
    if importance:
        ranked = [param for param, _ in sorted(importance.items(), key=lambda item: item[1], reverse=True)]
        selected = [param for param in ranked if param in params][:top_n]
        if len(selected) >= 2:
            return selected
    return params[:top_n]


def plot_pairwise_interactions(study: Any, trials: list[Any], params: list[str], output_dir: Path, study_name: str, filter_label: str, filter_slug: str, dpi: int) -> list[Path]:
    if len(trials) < 4 or len(params) < 2:
        LOGGER.warning("Skipping pairwise interactions: insufficient filtered trials or parameters")
        return []
    plt = import_pyplot()
    values = np.asarray([float(trial.value) for trial in trials], dtype=float)
    cmap = plt.get_cmap("viridis_r" if direction_is_minimize(study) else "viridis")
    paths: list[Path] = []
    rows: list[dict[str, Any]] = []
    for i, x_param in enumerate(params):
        for y_param in params[i + 1:]:
            x, obj_x = trial_param_array(trials, x_param)
            y, obj_y = trial_param_array(trials, y_param)
            if len(x) != len(y) or len(x) < 4 or not np.allclose(obj_x, obj_y):
                joined = [
                    (finite_float(t.params.get(x_param)), finite_float(t.params.get(y_param)), float(t.value))
                    for t in trials
                    if finite_float(t.params.get(x_param)) is not None and finite_float(t.params.get(y_param)) is not None
                ]
                if len(joined) < 4:
                    continue
                x = np.asarray([item[0] for item in joined], dtype=float)
                y = np.asarray([item[1] for item in joined], dtype=float)
                obj_x = np.asarray([item[2] for item in joined], dtype=float)
            x_log = is_log_distribution(trials, x_param) and np.all(x > 0)
            y_log = is_log_distribution(trials, y_param) and np.all(y > 0)
            for xv, yv, ov in zip(x, y, obj_x):
                rows.append({"x_parameter": x_param, "y_parameter": y_param, "x_value": float(xv), "y_value": float(yv), "objective": float(ov), "filter": filter_label})
            fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
            sc = ax.scatter(x, y, c=obj_x, cmap=cmap, s=42, alpha=0.82, edgecolor="white", linewidth=0.35)
            if x_log:
                ax.set_xscale("log")
            if y_log:
                ax.set_yscale("log")
            ax.set_xlabel(x_param)
            ax.set_ylabel(y_param)
            ax.set_title(f"{study_name}: {x_param} vs {y_param}, {filter_label} (n={len(x)})")
            colorbar = fig.colorbar(sc, ax=ax, pad=0.02)
            colorbar.set_label("Objective")
            path = output_dir / "pairwise_interactions" / f"interaction_{x_param}__{y_param}_{filter_slug}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            paths.append(path)
    write_csv(output_dir / "pairwise_interaction_data.csv", rows, ["x_parameter", "y_parameter", "x_value", "y_value", "objective", "filter"])
    return paths


def nearest_checkpoint_value(history: dict[int, float], target_step: int, tolerance: int) -> tuple[int, float] | None:
    if not history:
        return None
    nearest_step = min(history, key=lambda step: abs(step - target_step))
    if abs(nearest_step - target_step) > tolerance:
        return None
    return nearest_step, float(history[nearest_step])


def spearman_rank(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    try:
        from scipy.stats import spearmanr

        result = spearmanr(x, y)
        corr = finite_float(result.statistic)
        return corr
    except Exception:
        x_rank = np.argsort(np.argsort(x)).astype(float)
        y_rank = np.argsort(np.argsort(y)).astype(float)
        corr = np.corrcoef(x_rank, y_rank)[0, 1]
        return float(corr) if math.isfinite(corr) else None


def plot_checkpoint_rank_stability(
    trials: list[Any],
    histories: dict[int, dict[int, float]],
    budget: int | None,
    fractions: list[float],
    tolerance_fraction: float,
    output_dir: Path,
    study_name: str,
    dpi: int,
) -> list[Path]:
    if budget is None or not histories:
        LOGGER.warning("Skipping checkpoint rank stability: no step budget or validation histories")
        return []
    plt = import_pyplot()
    tolerance = max(1, int(round(budget * tolerance_fraction)))
    rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    for fraction in fractions:
        target = int(round(budget * fraction))
        points: list[tuple[int, int, float, float]] = []
        for trial in trials:
            history = histories.get(trial.number, {})
            match = nearest_checkpoint_value(history, target, tolerance)
            if match is None or trial.value is None:
                continue
            step, checkpoint_value = match
            points.append((trial.number, step, checkpoint_value, float(trial.value)))
        if len(points) < 3:
            LOGGER.warning("Skipping checkpoint %.0f%%: only %s valid trials", fraction * 100, len(points))
            continue
        checkpoint = np.asarray([item[2] for item in points], dtype=float)
        final = np.asarray([item[3] for item in points], dtype=float)
        corr = spearman_rank(checkpoint, final)
        for trial_number, step, checkpoint_value, final_value in points:
            rows.append({
                "target_fraction": fraction,
                "target_step": target,
                "matched_step": step,
                "trial_number": trial_number,
                "checkpoint_value": checkpoint_value,
                "final_objective": final_value,
                "spearman": corr,
                "n_trials": len(points),
                "tolerance_steps": tolerance,
            })
        fig, ax = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
        ax.scatter(checkpoint, final, s=34, color="#4C78A8", alpha=0.78)
        label = "not defined" if corr is None else f"{corr:.3f}"
        ax.set_xlabel(f"Validation loss near step {target}")
        ax.set_ylabel("Final/best objective")
        ax.set_title(f"{study_name}: rank stability at {fraction:.0%} budget\nSpearman={label}, n={len(points)}")
        path = output_dir / "checkpoint_rank_stability" / f"checkpoint_rank_stability_{int(round(fraction * 100)):03d}pct.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)
    write_csv(
        output_dir / "checkpoint_rank_stability.csv",
        rows,
        ["target_fraction", "target_step", "matched_step", "trial_number", "checkpoint_value", "final_objective", "spearman", "n_trials", "tolerance_steps"],
    )
    return paths


def study_summary(study: Any, trials: list[Any], histories: dict[int, dict[int, float]], budget: int | None) -> dict[str, Any]:
    import optuna

    state_counts = Counter(trial.state.name.lower() for trial in study.trials)
    values = np.asarray([float(trial.value) for trial in trials], dtype=float)
    minimize = direction_is_minimize(study)
    best_trial = min(trials, key=lambda trial: objective_sort_key(study, trial)) if trials else None
    summary: dict[str, Any] = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "total_trials": len(study.trials),
        "completed_trials": state_counts.get(optuna.trial.TrialState.COMPLETE.name.lower(), 0),
        "pruned_trials": state_counts.get(optuna.trial.TrialState.PRUNED.name.lower(), 0),
        "failed_trials": state_counts.get(optuna.trial.TrialState.FAIL.name.lower(), 0),
        "max_training_step_budget": budget,
        "actual_max_reported_step_distribution": actual_max_reported_steps(histories),
    }
    if values.size:
        q10, q25, q75, q90 = np.percentile(values, [10, 25, 75, 90])
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        best = best_value(values, minimize)
        near_threshold = best * (1.01 if minimize and best >= 0 else 0.99 if minimize else 0.99 if best >= 0 else 1.01)
        near_mask = better_or_equal(values, near_threshold, minimize)
        summary.update({
            "best_objective": best,
            "mean_objective": float(np.mean(values)),
            "median_objective": median,
            "std_objective": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "p25_objective": float(q25),
            "p75_objective": float(q75),
            "iqr_objective": float(q75 - q25),
            "mad_objective": mad,
            "p10_objective": float(q10),
            "p90_objective": float(q90),
            "fraction_completed_within_1pct_of_best": float(np.mean(near_mask)),
            "best_trial_number": int(best_trial.number) if best_trial else None,
            "best_trial_params": dict(best_trial.params) if best_trial else {},
        })
    return summary


def parse_fractions(value: str) -> list[float]:
    fractions = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        fraction = float(item)
        if not 0 < fraction < 1:
            raise ValueError("--checkpoint-fractions values must be between 0 and 1")
        fractions.append(fraction)
    return fractions


def main() -> None:
    args = parse_args()
    study_dir = resolve_study_dir(args.storage, args.study_name, args.study_dir)
    output_dir = default_output_dir(study_dir, args.storage, args.study_name, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_runtime(output_dir)

    import optuna

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    trials = complete_trials(study)
    params = parameter_names(trials)
    LOGGER.info("Loaded study %s with %s total trials and %s completed trials", args.study_name, len(study.trials), len(trials))
    if study_dir is not None:
        LOGGER.info("Using study directory %s", study_dir)
    LOGGER.info("Writing diagnostics to %s", output_dir)

    histories = {trial.number: validation_history(trial, study_dir) for trial in trials}
    histories = {number: history for number, history in histories.items() if history}
    budget = infer_max_step_budget(trials, study_dir)

    output_files: list[str] = []
    trial_rows = completed_trial_rows(study, trials)
    write_csv(output_dir / "completed_trials.csv", trial_rows)
    output_files.append(str(output_dir / "completed_trials.csv"))

    summary = study_summary(study, trials, histories, budget)
    write_json(output_dir / "study_summary.json", summary)
    output_files.append(str(output_dir / "study_summary.json"))

    aggregate_rows = aggregate_validation_histories(histories, args.min_step_trials)
    write_csv(output_dir / "aggregate_validation_loss_by_step.csv", aggregate_rows)
    output_files.append(str(output_dir / "aggregate_validation_loss_by_step.csv"))

    plotters = [
        plot_validation_aggregate(aggregate_rows, output_dir, args.study_name, args.min_step_trials, args.dpi),
        plot_optimization_history(study, trials, output_dir, args.study_name, budget, args.dpi),
        plot_objective_distribution(study, trials, output_dir, args.study_name, args.dpi),
    ]
    output_files.extend(str(path) for path in plotters if path is not None)

    importance, evaluator_name = compute_importance(study, output_dir, args.importance_seed)
    if importance:
        output_files.append(str(output_dir / "parameter_importance.csv"))
    importance_plot = plot_importance(importance, evaluator_name, output_dir, args.study_name, args.dpi)
    if importance_plot is not None:
        output_files.append(str(importance_plot))

    output_files.extend(str(path) for path in plot_parameter_slices(study, trials, params, output_dir, args.study_name, args.dpi))

    filtered_trials, filter_label, filter_slug = filter_trials(
        study, trials, args.parallel_filter, args.top_fraction, args.absolute_threshold, args.relative_pct
    )
    if len(filtered_trials) < args.min_filtered_trials:
        LOGGER.warning(
            "Filtered plot selection '%s' retained only %s trials; falling back to all completed trials",
            filter_label,
            len(filtered_trials),
        )
        filtered_trials, filter_label, filter_slug = list(trials), "all completed trials", "all"

    parallel_path = plot_parallel_coordinates(study, filtered_trials, params, output_dir, args.study_name, filter_label, filter_slug, args.dpi)
    if parallel_path is not None:
        output_files.append(str(parallel_path))

    interaction_params = selected_interaction_params(params, importance, args.interaction_params, args.top_interaction_params)
    output_files.extend(str(path) for path in plot_pairwise_interactions(
        study, filtered_trials, interaction_params, output_dir, args.study_name, filter_label, filter_slug, args.dpi
    ))

    fractions = parse_fractions(args.checkpoint_fractions)
    output_files.extend(str(path) for path in plot_checkpoint_rank_stability(
        trials, histories, budget, fractions, args.checkpoint_tolerance_fraction, output_dir, args.study_name, args.dpi
    ))

    write_json(output_dir / "generated_outputs.json", {
        "study_name": args.study_name,
        "storage": args.storage,
        "study_dir": str(study_dir) if study_dir is not None else None,
        "output_dir": str(output_dir),
        "files": output_files,
        "warnings": {
            "optuna_intermediate_values_found": any(trial.intermediate_values for trial in trials),
            "validation_histories_found": len(histories),
        },
    })
    print(f"Wrote {len(output_files)} diagnostics files to {output_dir}")


if __name__ == "__main__":
    main()
