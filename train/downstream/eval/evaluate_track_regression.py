#!/usr/bin/env python3
"""Physics-oriented evaluation for the CLAS12 track-regression adapter."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from mmap_ninja import RaggedMmap
from ruamel.yaml import YAML
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
DOWNSTREAM_DIR = HERE.parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(DOWNSTREAM_DIR))
sys.path.insert(0, str(REPO_ROOT))

from fm4npp.utils import YParams  # noqa: E402
from model import MambaTrackRegressionHead  # noqa: E402
from track_regression_trainer import DownstreamTrainer  # noqa: E402


AUX_LAYOUT = (
    "mc_particle_px", "mc_particle_py", "mc_particle_pz", "mc_particle_p",
    "cvt_px", "cvt_py", "cvt_pz", "cvt_p",
    "cvtrec_px", "cvtrec_py", "cvtrec_pz", "cvtrec_p",
    "rec_particle_px", "rec_particle_py", "rec_particle_pz", "rec_particle_p",
)
METHODS = ("adapter", "cvt", "cvtrec", "rec_particle")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-config",
        default=str(HERE / "track_regression_analysis.yaml"),
        help="Analysis YAML file",
    )
    parser.add_argument("--checkpoint", help="Override the checkpoint in the YAML")
    parser.add_argument("--output-dir", help="Override the output directory")
    parser.add_argument("--max-samples", type=int, help="Override the sample limit")
    return parser.parse_args()


def resolve_path(value, base_dir):
    if value is None:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_analysis_config(args):
    config_path = Path(args.analysis_config).resolve()
    with config_path.open() as stream:
        config = YAML(typ="safe").load(stream)["analysis"]
    config["model_yaml"] = resolve_path(config["model_yaml"], config_path.parent)
    config["checkpoint"] = resolve_path(
        args.checkpoint or config["checkpoint"], config_path.parent
    )
    config["output_dir"] = resolve_path(
        args.output_dir or config["output_dir"], config_path.parent
    )
    if args.max_samples is not None:
        config["max_samples"] = args.max_samples
    config["analysis_config"] = config_path
    return config


def build_head(trainer):
    params = trainer.params
    return MambaTrackRegressionHead(
        input_dim=params.embed_dim,
        num_layers=1,
        num_output_dim=params.num_output_classes,
        d_state=64,
        d_conv=4,
        expand=2,
        num_feature_layers=params.num_layers_backbone,
        num_embedder_layers=params.num_embedder_layers,
        pooling=getattr(params, "pooling", "mean"),
        embed_method=params.embed_method,
        pe_method=params.pe_method,
        target_mean=trainer.regression_target_stats["mean"],
        target_std=trainer.regression_target_stats["std"],
    ).to(trainer.device)


def first_valid_row(array):
    array = np.asarray(array)
    if array.ndim == 1:
        return array
    valid = np.all(np.isfinite(array), axis=1)
    return array[np.flatnonzero(valid)[0]] if valid.any() else array[0]


def vector_kinematics(vector):
    px, py, pz = np.moveaxis(np.asarray(vector), -1, 0)
    pt = np.hypot(px, py)
    p = np.sqrt(px * px + py * py + pz * pz)
    theta = np.degrees(np.arctan2(pt, pz))
    phi = np.degrees(np.arctan2(py, px))
    return p, pt, theta, phi


def opening_angle_deg(prediction, truth):
    dot = np.sum(prediction * truth, axis=1)
    denom = np.linalg.norm(prediction, axis=1) * np.linalg.norm(truth, axis=1)
    cosine = np.clip(np.divide(dot, denom, out=np.ones_like(dot), where=denom > 0), -1, 1)
    return np.degrees(np.arccos(cosine))


def safe_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def central_width(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    low, high = np.quantile(values, [0.16, 0.84])
    return 0.5 * (high - low)


def calculate_metrics(truth, prediction, tolerances):
    valid = np.all(np.isfinite(truth), axis=1) & np.all(np.isfinite(prediction), axis=1)
    truth = truth[valid]
    prediction = prediction[valid]
    if not len(truth):
        return {"n": 0}

    residual = prediction - truth
    truth_p, truth_pt, _, _ = vector_kinematics(truth)
    pred_p, pred_pt, _, _ = vector_kinematics(prediction)
    nonzero = truth_p > 1e-12
    rel = (pred_p[nonzero] - truth_p[nonzero]) / truth_p[nonzero]
    p_residual = pred_p - truth_p

    component = {}
    for index, name in enumerate(("px", "py", "pz")):
        y = truth[:, index]
        r = residual[:, index]
        denominator = np.sum((y - y.mean()) ** 2)
        component[name] = {
            "bias_gev": safe_float(np.mean(r)),
            "mae_gev": safe_float(np.mean(np.abs(r))),
            "rmse_gev": safe_float(np.sqrt(np.mean(r ** 2))),
            "r2": safe_float(1.0 - np.sum(r ** 2) / denominator) if denominator > 0 else None,
        }

    result = {
        "n": int(len(truth)),
        "components": component,
        "momentum": {
            "bias_gev": safe_float(np.mean(p_residual)),
            "mae_gev": safe_float(np.mean(np.abs(p_residual))),
            "rmse_gev": safe_float(np.sqrt(np.mean(p_residual ** 2))),
            "relative_bias": safe_float(np.mean(rel)),
            "relative_median": safe_float(np.median(rel)),
            "relative_resolution_68": safe_float(central_width(rel)),
            "relative_tail_fraction_10pct": safe_float(np.mean(np.abs(rel) > 0.10)),
            "relative_tail_fraction_25pct": safe_float(np.mean(np.abs(rel) > 0.25)),
        },
        "pt": {
            "mae_gev": safe_float(np.mean(np.abs(pred_pt - truth_pt))),
            "rmse_gev": safe_float(np.sqrt(np.mean((pred_pt - truth_pt) ** 2))),
        },
        "direction": {
            "opening_angle_mean_deg": safe_float(np.mean(opening_angle_deg(prediction, truth))),
            "opening_angle_median_deg": safe_float(np.median(opening_angle_deg(prediction, truth))),
            "opening_angle_68_deg": safe_float(np.quantile(opening_angle_deg(prediction, truth), 0.68)),
        },
    }
    result["momentum"]["fraction_within_relative_tolerance"] = {
        str(tolerance): safe_float(np.mean(np.abs(rel) <= tolerance))
        for tolerance in tolerances
    }
    return result


def compact_metrics(truth, prediction):
    truth_p = np.linalg.norm(truth, axis=1)
    pred_p = np.linalg.norm(prediction, axis=1)
    valid = np.all(np.isfinite(prediction), axis=1) & (truth_p > 1e-12)
    rel = (pred_p[valid] - truth_p[valid]) / truth_p[valid]
    return {
        "n": int(valid.sum()),
        "relative_bias": safe_float(np.mean(rel)) if valid.any() else None,
        "relative_resolution_68": safe_float(central_width(rel)) if valid.any() else None,
        "relative_rmse": safe_float(np.sqrt(np.mean(rel ** 2))) if valid.any() else None,
        "tail_fraction_10pct": safe_float(np.mean(np.abs(rel) > 0.10)) if valid.any() else None,
    }


def add_binned_rows(rows, truth, predictions, values, group_name, edges, labels=None):
    if labels is None:
        groups = [(f"[{edges[i]}, {edges[i + 1]})", (values >= edges[i]) & (values < edges[i + 1]))
                  for i in range(len(edges) - 1)]
    else:
        groups = [(label, values == key) for key, label in labels.items()]
    for label, selection in groups:
        for method, prediction in predictions.items():
            metric = compact_metrics(truth[selection], prediction[selection])
            rows.append({"group": group_name, "bin": label, "method": method, **metric})


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(path, records):
    with gzip.open(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def pdg_charge(pdg):
    """Return charge sign for the charged species represented in this dataset."""
    if pdg == 0:
        return 0
    return -int(np.sign(pdg)) if abs(pdg) == 11 else int(np.sign(pdg))


def attach_sample_metadata(records, path):
    """Attach JSONL metadata by the original RaggedMmap row index."""
    by_index = {record["real_index"]: record for record in records}
    remaining = set(by_index)
    with Path(path).open() as stream:
        for index, line in enumerate(stream):
            if index in remaining:
                metadata = json.loads(line)
                pdg = int(metadata.get("pid", 0))
                by_index[index].update({
                    "source_file": metadata.get("source_file", ""),
                    "event": metadata.get("event", ""),
                    "reco_track_id": metadata.get("trkid", ""),
                    "truth_track_id": metadata.get("truth_tid", ""),
                    "pdg": pdg,
                    "charge": pdg_charge(pdg),
                })
                remaining.remove(index)
                if not remaining:
                    break
    if remaining:
        raise ValueError(
            f"Sample metadata {path} is missing {len(remaining)} evaluated indices"
        )


def make_plots(
    output_dir, truth, predictions, true_p, true_theta, momentum_bins, theta_bins
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, prediction in predictions.items():
        pred_p = np.linalg.norm(prediction, axis=1)
        valid = true_p > 1e-12
        rel = (pred_p[valid] - true_p[valid]) / true_p[valid]
        ax.hist(rel, bins=120, range=(-0.5, 0.5), histtype="step", density=True, label=method)
    ax.set(xlabel="(p estimate - p truth) / p truth", ylabel="Density", title="Momentum residuals")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "momentum_residuals.png", dpi=160)
    plt.close(fig)

    for values, name, xlabel, edges in (
        (true_p, "resolution_vs_p", "True momentum [GeV]", momentum_bins),
        (true_theta, "resolution_vs_theta", "True theta [deg]", theta_bins),
    ):
        fig, ax = plt.subplots(figsize=(7, 5))
        centers = 0.5 * (edges[:-1] + edges[1:])
        for method, prediction in predictions.items():
            pred_p = np.linalg.norm(prediction, axis=1)
            rel = np.divide(pred_p - true_p, true_p, out=np.full_like(true_p, np.nan), where=true_p > 0)
            widths = [central_width(rel[(values >= lo) & (values < hi)]) for lo, hi in zip(edges[:-1], edges[1:])]
            ax.plot(centers, widths, marker="o", label=method)
        ax.set(xlabel=xlabel, ylabel="Central 68% relative resolution", title=name.replace("_", " ").title())
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{name}.png", dpi=160)
        plt.close(fig)


def main():
    cli = parse_args()
    config = load_analysis_config(cli)
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Track-regression evaluation requires CUDA because the installed "
            "Mamba/causal-convolution implementation has no CPU forward kernel"
        )

    params = YParams(str(config["model_yaml"]), config["model_config"])
    params["batch_size"] = int(config["batch_size"])
    params["valid_batch_size"] = int(config["batch_size"])
    params["num_data_workers"] = int(config["num_workers"])
    params["num_embedder_layers"] = 0
    params["pretrained_ckpt"] = (
        str(resolve_path(config.get("pretrained_checkpoint"), config["analysis_config"].parent))
        if config.get("use_pretrained_backbone") else None
    )

    trainer_args = SimpleNamespace(
        root_dir=str(output_dir / "runtime"),
        global_log_dir="logs",
        config=config["model_config"],
        run_num=str(config["run_num"]),
    )
    trainer = DownstreamTrainer(params, trainer_args)
    trainer.launch()
    if trainer.world_size != 1:
        raise RuntimeError("This baseline analysis currently supports single-process evaluation only")
    trainer.down_model = build_head(trainer)
    trainer.load_checkpoint(str(config["checkpoint"]), inference=True)
    trainer.down_model.eval()
    trainer.model.eval()

    dataset = trainer.val_data_loader.dataset
    aux = RaggedMmap(str(Path(params.data_root_test) / "aux_target_test"))
    target_scale = float(config["target_momentum_scale_to_gev"])
    aux_scale = float(config["auxiliary_momentum_scale_to_gev"])
    max_samples = int(config["max_samples"])
    use_pretrained = bool(config["use_pretrained_backbone"])

    records = []
    cursor = 0
    with torch.no_grad():
        for batch in tqdm(trainer.val_data_loader, desc="Evaluating tracks"):
            remaining = max_samples - len(records)
            if remaining <= 0:
                break
            points = batch["points"].to(trainer.device)
            batch_size, channels = points.size(0), points.size(-1)
            points = points.reshape(batch_size, -1, channels)
            mask = points[..., 0] != -100
            regression = batch["reg_target"].to(trainer.device)

            if use_pretrained:
                _, embeddings, _ = trainer.model(points, return_z=True)
                prediction = trainer.down_model(
                    points, torch.stack(embeddings), pretrain=True, padding_mask=mask
                )["pred_regression"]
            else:
                prediction = trainer.down_model(points, feature=None, padding_mask=mask)["pred_regression"]

            normalized_truth = trainer.build_regression_targets(regression, mask)["target"]
            prediction = trainer.down_model.target_normalizer.denormalize(prediction).cpu().numpy()
            truth = trainer.down_model.target_normalizer.denormalize(normalized_truth).cpu().numpy()

            take = min(batch_size, remaining)
            for local_index in range(take):
                dataset_index = cursor + local_index
                real_index = int(dataset.idxlist[dataset_index])
                aux_row = first_valid_row(aux[real_index]).astype(float) * aux_scale
                pid_values = np.asarray(dataset.memmap_pid_target[real_index]).reshape(-1)
                pid_class = int(pid_values[0]) if len(pid_values) else -1
                true_vector = truth[local_index] * target_scale
                pred_vector = prediction[local_index] * target_scale
                true_p, true_pt, true_theta, true_phi = vector_kinematics(true_vector)
                pred_p, pred_pt, pred_theta, pred_phi = vector_kinematics(pred_vector)
                record = {
                    "real_index": real_index,
                    "n_hits": int(mask[local_index].sum().item()),
                    "pid_class": pid_class,
                    "pid_label": str(config["pid_labels"].get(pid_class, "unknown")),
                    "true_px_gev": true_vector[0], "true_py_gev": true_vector[1], "true_pz_gev": true_vector[2],
                    "true_p_gev": true_p, "true_pt_gev": true_pt, "true_theta_deg": true_theta, "true_phi_deg": true_phi,
                    "adapter_px_gev": pred_vector[0], "adapter_py_gev": pred_vector[1], "adapter_pz_gev": pred_vector[2],
                    "adapter_p_gev": pred_p, "adapter_pt_gev": pred_pt, "adapter_theta_deg": pred_theta, "adapter_phi_deg": pred_phi,
                }
                record.update({name + "_gev": value for name, value in zip(AUX_LAYOUT, aux_row)})
                records.append(record)
            cursor += batch_size

    if not records:
        raise RuntimeError("Evaluation produced no records")

    if config.get("include_sample_metadata", True):
        metadata_path = config.get("sample_metadata_jsonl")
        if metadata_path is None:
            metadata_path = Path(params.data_root_test) / "samples_test.jsonl"
        else:
            metadata_path = resolve_path(
                metadata_path, config["analysis_config"].parent
            )
        attach_sample_metadata(records, metadata_path)

    write_predictions(output_dir / "predictions.csv.gz", records)
    truth = np.asarray([[r["true_px_gev"], r["true_py_gev"], r["true_pz_gev"]] for r in records])
    predictions = {
        method: np.asarray([[r[f"{method}_px_gev"], r[f"{method}_py_gev"], r[f"{method}_pz_gev"]] for r in records])
        for method in METHODS
    }
    true_p, _, true_theta, _ = vector_kinematics(truth)
    tolerances = [float(value) for value in config["relative_error_tolerances"]]
    summary = {
        "checkpoint": str(config["checkpoint"]),
        "model_config": config["model_config"],
        "target_definition": "MC momentum at innermost matched CVT hit",
        "units": "GeV and degrees",
        "methods": {method: calculate_metrics(truth, prediction, tolerances) for method, prediction in predictions.items()},
    }
    adapter_resolution = summary["methods"]["adapter"]["momentum"]["relative_resolution_68"]
    summary["adapter_to_cvt_resolution_ratio"] = (
        adapter_resolution / summary["methods"]["cvt"]["momentum"]["relative_resolution_68"]
    )
    with (output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)

    binned_rows = []
    momentum_bins = np.asarray(config["momentum_bins_gev"], dtype=float)
    theta_bins = np.asarray(config["theta_bins_deg"], dtype=float)
    add_binned_rows(binned_rows, truth, predictions, true_p, "true_p_gev", momentum_bins)
    add_binned_rows(binned_rows, truth, predictions, true_theta, "true_theta_deg", theta_bins)
    pid = np.asarray([r["pid_class"] for r in records])
    pid_labels = {int(key): str(value) for key, value in config["pid_labels"].items()}
    add_binned_rows(binned_rows, truth, predictions, pid, "pid", None, labels=pid_labels)
    if config.get("include_sample_metadata", True):
        charge = np.asarray([r["charge"] for r in records])
        add_binned_rows(
            binned_rows, truth, predictions, charge, "charge", None,
            labels={-1: "negative", 1: "positive"},
        )
    write_csv(output_dir / "binned_metrics.csv", binned_rows)
    make_plots(
        output_dir, truth, predictions, true_p, true_theta,
        momentum_bins, theta_bins,
    )

    trainer.cleanup()
    print(f"Wrote evaluation results to {output_dir}")


if __name__ == "__main__":
    main()
