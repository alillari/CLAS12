#!/usr/bin/env python3
"""Fit frozen-backbone linear probes for CLAS12 track-regression targets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
DOWNSTREAM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DOWNSTREAM_DIR))

from model import RegressionTargetNormalizer  # noqa: E402
from track_regression_experiment import (  # noqa: E402
    TrackRegressionExperimentConfig,
    resolve_params,
    set_global_seed,
)
from track_regression_trainer import DownstreamTrainer  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None


class _TargetNormalizerHolder(nn.Module):
    def __init__(self, mean: list[float], std: list[float]):
        super().__init__()
        self.target_normalizer = RegressionTargetNormalizer(
            len(mean), mean=mean, std=std
        )


@dataclass
class SufficientStats:
    xtx: torch.Tensor
    xty: torch.Tensor
    y_sum: torch.Tensor
    y2_sum: torch.Tensor
    count: int = 0

    @classmethod
    def empty(cls, num_layers: int, feature_dim: int, output_dim: int) -> "SufficientStats":
        augmented_dim = feature_dim + 1
        return cls(
            xtx=torch.zeros(num_layers, augmented_dim, augmented_dim, dtype=torch.float64),
            xty=torch.zeros(num_layers, augmented_dim, output_dim, dtype=torch.float64),
            y_sum=torch.zeros(output_dim, dtype=torch.float64),
            y2_sum=torch.zeros(output_dim, dtype=torch.float64),
            count=0,
        )


@dataclass
class EvalStats:
    sse: torch.Tensor
    sae: torch.Tensor
    y_sum: torch.Tensor
    y2_sum: torch.Tensor
    count: int = 0

    @classmethod
    def empty(cls, num_layers: int, output_dim: int) -> "EvalStats":
        return cls(
            sse=torch.zeros(num_layers, output_dim, dtype=torch.float64),
            sae=torch.zeros(num_layers, output_dim, dtype=torch.float64),
            y_sum=torch.zeros(output_dim, dtype=torch.float64),
            y2_sum=torch.zeros(output_dim, dtype=torch.float64),
            count=0,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml_config", required=True, help="Track-regression YAML config")
    parser.add_argument("--config", required=True, help="YAML model config key")
    parser.add_argument("--pretrained_ckpt", required=True, help="Backbone checkpoint")
    parser.add_argument("--output_dir", required=True, help="Directory for probe outputs")
    parser.add_argument("--run_num", default="0", help="Synthetic run number for loader setup")
    parser.add_argument("--eventnumber", default=50000, type=int, help="Training event cap")
    parser.add_argument("--batch_size", default=64, type=int, help="Train/eval batch size")
    parser.add_argument("--num_workers", default=None, type=int, help="Override data-loader workers")
    parser.add_argument("--max_train_batches", default=None, type=int)
    parser.add_argument("--max_eval_batches", default=None, type=int)
    parser.add_argument("--ridge_alpha", default=1.0e-4, type=float)
    parser.add_argument("--seed", default=12345, type=int)
    parser.add_argument(
        "--shuffled_control_events",
        default=20000,
        type=int,
        help="Maximum train events kept for the shuffled-label control; 0 disables it.",
    )
    parser.add_argument(
        "--random_backbone_control",
        action="store_true",
        help="Also fit/evaluate probes on a randomly initialized backbone.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def setup_trainer(
    args: argparse.Namespace,
    output_dir: Path,
    pretrained_ckpt: str | None,
    run_suffix: str,
) -> DownstreamTrainer:
    params = resolve_params(
        TrackRegressionExperimentConfig(
            yaml_config=args.yaml_config,
            config=args.config,
            run_num=f"{args.run_num}_{run_suffix}",
            root_dir=str(output_dir / "runtime"),
            global_log_dir="globallogs",
            eventnumber=args.eventnumber,
            usepretrain=pretrained_ckpt is not None,
            train_batch_size=args.batch_size,
            pretrained_ckpt=pretrained_ckpt,
            seed=args.seed,
        )
    )
    if args.num_workers is not None:
        params.num_data_workers = int(args.num_workers)
    params.valid_batch_size = int(args.batch_size)

    trainer_args = SimpleNamespace(
        root_dir=str(output_dir / "runtime"),
        global_log_dir="globallogs",
        config=args.config,
        run_num=f"{args.run_num}_{run_suffix}",
    )
    trainer = DownstreamTrainer(params, trainer_args)
    trainer.launch()
    trainer.model.eval()
    trainer.down_model = _TargetNormalizerHolder(
        trainer.regression_target_stats["mean"],
        trainer.regression_target_stats["std"],
    ).to(trainer.device)
    return trainer


def masked_mean_layers(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool stacked layer features from (L, B, N, D) to (L, B, D)."""
    weights = mask.to(dtype=features.dtype).unsqueeze(0).unsqueeze(-1)
    denom = weights.sum(dim=2).clamp_min(1.0)
    return (features * weights).sum(dim=2) / denom


def event_valid(target_valid: torch.Tensor) -> torch.Tensor:
    if target_valid.ndim == 1:
        return target_valid
    return target_valid.all(dim=-1)


def add_intercept(x: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(x.shape[0], 1, dtype=x.dtype)
    return torch.cat([x, ones], dim=1)


def add_intercept_layers(x_layers: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(
        x_layers.shape[0],
        x_layers.shape[1],
        1,
        dtype=x_layers.dtype,
        device=x_layers.device,
    )
    return torch.cat([x_layers, ones], dim=2)


def update_fit_stats(stats: SufficientStats, x_layers: torch.Tensor, y: torch.Tensor) -> None:
    x_layers = x_layers.to(dtype=torch.float64, device="cpu")
    y = y.to(dtype=torch.float64, device="cpu")
    stats.count += int(y.shape[0])
    stats.y_sum += y.sum(dim=0)
    stats.y2_sum += (y * y).sum(dim=0)
    x_aug = add_intercept_layers(x_layers)
    stats.xtx += torch.einsum("lbi,lbj->lij", x_aug, x_aug)
    stats.xty += torch.einsum("lbi,bo->lio", x_aug, y)


def solve_ridge(stats: SufficientStats, alpha: float) -> torch.Tensor:
    coeffs = []
    penalty = torch.eye(stats.xtx.shape[-1], dtype=torch.float64)
    penalty[-1, -1] = 0.0
    for layer_idx in range(stats.xtx.shape[0]):
        lhs = stats.xtx[layer_idx] + float(alpha) * penalty
        rhs = stats.xty[layer_idx]
        try:
            coeff = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            coeff = torch.linalg.lstsq(lhs, rhs).solution
        coeffs.append(coeff)
    return torch.stack(coeffs)


def update_eval_stats(stats: EvalStats, coeffs: torch.Tensor, x_layers: torch.Tensor, y: torch.Tensor) -> None:
    x_layers = x_layers.to(dtype=torch.float64, device="cpu")
    y = y.to(dtype=torch.float64, device="cpu")
    coeffs = coeffs.to(dtype=torch.float64, device="cpu")
    stats.count += int(y.shape[0])
    stats.y_sum += y.sum(dim=0)
    stats.y2_sum += (y * y).sum(dim=0)
    pred = torch.einsum("lbi,lio->lbo", add_intercept_layers(x_layers), coeffs)
    residual = pred - y.unsqueeze(0)
    stats.sse += (residual * residual).sum(dim=1)
    stats.sae += residual.abs().sum(dim=1)


def iter_probe_batches(
    trainer: DownstreamTrainer,
    loader: Iterable,
    max_batches: int | None,
    desc: str,
):
    device = trainer.device
    total = max_batches
    if total is None:
        try:
            total = len(loader)
        except TypeError:
            total = None
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, total=total, desc=desc, unit="batch")
    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            if max_batches is not None and batch_idx >= max_batches:
                break
            points = batch["points"].to(device)
            batch_size, channels = points.size(0), points.size(-1)
            points = points.reshape(batch_size, -1, channels)
            mask = points[..., 0] != -100
            targets = trainer.build_regression_targets(
                batch["reg_target"].to(device), mask
            )
            valid = event_valid(targets["target_valid"])
            if not bool(valid.any()):
                continue
            _, embeddings, _ = trainer.model(points, return_z=True)
            pooled = masked_mean_layers(torch.stack(embeddings), mask)
            yield pooled[:, valid].detach(), targets["target"][valid].detach()


def fit_probe(
    trainer: DownstreamTrainer,
    max_batches: int | None,
    ridge_alpha: float,
    collect_control_events: int,
    desc: str,
):
    stats = None
    collected_x = []
    collected_y = []
    collected_count = 0
    for x_layers, y in iter_probe_batches(
        trainer,
        trainer.train_data_loader,
        max_batches,
        desc=desc,
    ):
        if stats is None:
            stats = SufficientStats.empty(x_layers.shape[0], x_layers.shape[-1], y.shape[-1])
        update_fit_stats(stats, x_layers, y)
        if collect_control_events > 0 and collected_count < collect_control_events:
            take = min(int(y.shape[0]), collect_control_events - collected_count)
            collected_x.append(x_layers[:, :take].cpu().float())
            collected_y.append(y[:take].cpu().float())
            collected_count += take
    if stats is None or stats.count == 0:
        raise RuntimeError("No valid training events were available for the probe")
    coeffs = solve_ridge(stats, ridge_alpha)
    train_mean = stats.y_sum / max(stats.count, 1)
    control_x = torch.cat(collected_x, dim=1) if collected_x else None
    control_y = torch.cat(collected_y, dim=0) if collected_y else None
    return coeffs, stats, train_mean, control_x, control_y


def evaluate_probe(
    trainer: DownstreamTrainer,
    coeffs: torch.Tensor,
    max_batches: int | None,
    desc: str,
) -> EvalStats:
    eval_stats = None
    for x_layers, y in iter_probe_batches(
        trainer,
        trainer.val_data_loader,
        max_batches,
        desc=desc,
    ):
        if eval_stats is None:
            eval_stats = EvalStats.empty(coeffs.shape[0], y.shape[-1])
        update_eval_stats(eval_stats, coeffs, x_layers, y)
    if eval_stats is None or eval_stats.count == 0:
        raise RuntimeError("No valid evaluation events were available for the probe")
    return eval_stats


def evaluate_probe_many(
    trainer: DownstreamTrainer,
    coeff_sets: dict[str, torch.Tensor],
    max_batches: int | None,
    desc: str,
) -> dict[str, EvalStats]:
    eval_stats: dict[str, EvalStats] = {}
    for x_layers, y in iter_probe_batches(
        trainer,
        trainer.val_data_loader,
        max_batches,
        desc=desc,
    ):
        for name, coeffs in coeff_sets.items():
            if name not in eval_stats:
                eval_stats[name] = EvalStats.empty(coeffs.shape[0], y.shape[-1])
            update_eval_stats(eval_stats[name], coeffs, x_layers, y)
    if not eval_stats:
        raise RuntimeError("No valid evaluation events were available for the probe")
    return eval_stats


def fit_shuffled_control(
    x_layers: torch.Tensor | None,
    y: torch.Tensor | None,
    ridge_alpha: float,
    seed: int,
) -> torch.Tensor | None:
    if x_layers is None or y is None or y.shape[0] < 2:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permuted_y = y[torch.randperm(y.shape[0], generator=generator)]
    stats = SufficientStats.empty(x_layers.shape[0], x_layers.shape[-1], y.shape[-1])
    update_fit_stats(stats, x_layers, permuted_y)
    return solve_ridge(stats, ridge_alpha)


def summarize_eval(
    name: str,
    coeffs: torch.Tensor,
    eval_stats: EvalStats,
    train_mean: torch.Tensor,
    train_stats: SufficientStats,
) -> tuple[list[dict], dict]:
    output_dim = eval_stats.sse.shape[-1]
    train_var = train_stats.y2_sum / max(train_stats.count, 1) - train_mean.square()
    eval_mean = eval_stats.y_sum / max(eval_stats.count, 1)
    eval_var = eval_stats.y2_sum / max(eval_stats.count, 1) - eval_mean.square()
    mean_sse = ((eval_stats.y2_sum - 2 * train_mean * eval_stats.y_sum) + eval_stats.count * train_mean.square()).clamp_min(0)
    gaussian_expected_sse = eval_stats.count * (
        eval_var + train_var + (eval_mean - train_mean).square()
    ).clamp_min(0)

    rows = []
    for layer_idx in range(eval_stats.sse.shape[0]):
        layer_sse = eval_stats.sse[layer_idx]
        layer_sae = eval_stats.sae[layer_idx]
        mse_components = layer_sse / max(eval_stats.count, 1)
        mae_components = layer_sae / max(eval_stats.count, 1)
        mean_mse_components = mean_sse / max(eval_stats.count, 1)
        r2_components = 1.0 - layer_sse / mean_sse.clamp_min(1.0e-12)
        rows.append(
            {
                "probe": name,
                "layer": layer_idx,
                "count": eval_stats.count,
                "mse": float(mse_components.mean().item()),
                "mae": float(mae_components.mean().item()),
                "r2": float(r2_components.mean().item()),
                "mse_px": float(mse_components[0].item()) if output_dim > 0 else None,
                "mse_py": float(mse_components[1].item()) if output_dim > 1 else None,
                "mse_pz": float(mse_components[2].item()) if output_dim > 2 else None,
                "mae_px": float(mae_components[0].item()) if output_dim > 0 else None,
                "mae_py": float(mae_components[1].item()) if output_dim > 1 else None,
                "mae_pz": float(mae_components[2].item()) if output_dim > 2 else None,
                "r2_px": float(r2_components[0].item()) if output_dim > 0 else None,
                "r2_py": float(r2_components[1].item()) if output_dim > 1 else None,
                "r2_pz": float(r2_components[2].item()) if output_dim > 2 else None,
                "mean_baseline_mse": float(mean_mse_components.mean().item()),
                "gaussian_random_expected_mse": float((gaussian_expected_sse / max(eval_stats.count, 1)).mean().item()),
            }
        )
    best = max(rows, key=lambda row: row["r2"])
    return rows, {
        "probe": name,
        "best_layer": best["layer"],
        "best_r2": best["r2"],
        "best_mse": best["mse"],
        "count": eval_stats.count,
        "target_train_mean": train_mean.tolist(),
        "target_train_var": train_var.tolist(),
        "target_eval_mean": eval_mean.tolist(),
        "target_eval_var": eval_var.tolist(),
        "mean_baseline_mse": rows[0]["mean_baseline_mse"] if rows else None,
        "gaussian_random_expected_mse": rows[0]["gaussian_random_expected_mse"] if rows else None,
    }


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_layer_metrics(path: Path, rows: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting dependency is optional
        print(f"Skipping plot generation: {exc}")
        return

    probes = sorted({row["probe"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for probe in probes:
        probe_rows = [row for row in rows if row["probe"] == probe]
        probe_rows.sort(key=lambda row: row["layer"])
        layers = [row["layer"] for row in probe_rows]
        axes[0].plot(layers, [row["r2"] for row in probe_rows], marker="o", label=probe)
        axes[1].plot(layers, [row["mse"] for row in probe_rows], marker="o", label=probe)
    axes[0].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Backbone layer")
    axes[0].set_ylabel("R2 vs train-mean baseline")
    axes[1].set_xlabel("Backbone layer")
    axes[1].set_ylabel("Normalized MSE")
    axes[0].legend()
    axes[1].legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_one_probe(
    args: argparse.Namespace,
    output_dir: Path,
    name: str,
    pretrained_ckpt: str | None,
) -> tuple[list[dict], dict, torch.Tensor]:
    start_time = time.monotonic()
    print(f"[{name}] initializing trainer", flush=True)
    trainer = setup_trainer(args, output_dir, pretrained_ckpt, name)
    try:
        print(f"[{name}] extracting train features", flush=True)
        coeffs, train_stats, train_mean, control_x, control_y = fit_probe(
            trainer,
            max_batches=args.max_train_batches,
            ridge_alpha=args.ridge_alpha,
            collect_control_events=args.shuffled_control_events,
            desc=f"{name} train",
        )
        coeff_sets = {name: coeffs}
        print(f"[{name}] fit complete on {train_stats.count} train events", flush=True)
        shuffled_coeffs = fit_shuffled_control(
            control_x,
            control_y,
            ridge_alpha=args.ridge_alpha,
            seed=args.seed + 17,
        )
        if shuffled_coeffs is not None:
            coeff_sets[f"{name}_shuffled_labels"] = shuffled_coeffs
        print(f"[{name}] evaluating {len(coeff_sets)} coefficient set(s)", flush=True)
        eval_stats_by_name = evaluate_probe_many(
            trainer,
            coeff_sets,
            max_batches=args.max_eval_batches,
            desc=f"{name} eval",
        )
        eval_stats = eval_stats_by_name[name]
        rows, summary = summarize_eval(name, coeffs, eval_stats, train_mean, train_stats)

        if shuffled_coeffs is not None:
            shuffled_rows, shuffled_summary = summarize_eval(
                f"{name}_shuffled_labels",
                shuffled_coeffs,
                eval_stats_by_name[f"{name}_shuffled_labels"],
                train_mean,
                train_stats,
            )
            rows.extend(shuffled_rows)
            summary["shuffled_label_control"] = shuffled_summary
    finally:
        trainer.cleanup()
    print(f"[{name}] finished in {time.monotonic() - start_time:.1f} seconds", flush=True)
    return rows, summary, coeffs


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)
    set_global_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    all_rows = []
    summaries = {}
    coefficients = {}

    pretrained_rows, pretrained_summary, pretrained_coeffs = run_one_probe(
        args, output_dir, "pretrained_backbone", args.pretrained_ckpt
    )
    all_rows.extend(pretrained_rows)
    summaries["pretrained_backbone"] = pretrained_summary
    coefficients["pretrained_backbone"] = pretrained_coeffs.float()

    if args.random_backbone_control:
        random_rows, random_summary, random_coeffs = run_one_probe(
            args, output_dir, "random_backbone", None
        )
        all_rows.extend(random_rows)
        summaries["random_backbone"] = random_summary
        coefficients["random_backbone"] = random_coeffs.float()

    write_metrics_csv(output_dir / "per_layer_metrics.csv", all_rows)
    plot_layer_metrics(output_dir / "layer_metric_curves.png", all_rows)
    torch.save(coefficients, output_dir / "linear_probe_weights.pt")
    summary = {
        "yaml_config": os.path.abspath(args.yaml_config),
        "config": args.config,
        "pretrained_ckpt": os.path.abspath(args.pretrained_ckpt),
        "eventnumber": args.eventnumber,
        "batch_size": args.batch_size,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "ridge_alpha": args.ridge_alpha,
        "seed": args.seed,
        "summaries": summaries,
    }
    with (output_dir / "probe_summary.json").open("w") as stream:
        json.dump(json_safe(summary), stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
