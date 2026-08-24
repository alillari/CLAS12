#!/usr/bin/env python3
"""Physics-oriented evaluation for the CLAS12 track-regression adapter."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
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
from regression_utils import target_to_cartesian_numpy  # noqa: E402
from track_regression_trainer import DownstreamTrainer  # noqa: E402


AUX_LAYOUT = (
    "mc_particle_px", "mc_particle_py", "mc_particle_pz", "mc_particle_p",
    "cvt_px", "cvt_py", "cvt_pz", "cvt_p",
    "cvtrec_px", "cvtrec_py", "cvtrec_pz", "cvtrec_p",
    "rec_particle_px", "rec_particle_py", "rec_particle_pz", "rec_particle_p",
)
METHODS = ("adapter", "cvt", "cvtrec")
PHYSICS_PLOT_METHODS = ("adapter", "cvt")
KINEMATIC_LABELS = {
    "p_gev": r"p [GeV]",
    "pt_gev": r"pT [GeV]",
    "theta_deg": r"theta [deg]",
    "phi_deg": r"phi [deg]",
    "eta": r"eta",
}
COMPONENT_LABELS = {
    "px_gev": r"px [GeV]",
    "py_gev": r"py [GeV]",
    "pz_gev": r"pz [GeV]",
}
ML_METRIC_LABELS = {
    "mae": "MAE",
    "rmse": "RMSE",
    "median_absolute_error": "Median absolute error",
    "p95_absolute_error": "95th percentile absolute error",
}
ENV_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")
DEFAULT_DELTA_P_OVER_P_BINS_GEV = [
    round(0.25 + 0.25 * index, 10)
    for index in range(12)
]


class ConfigFormatMap(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-config",
        default=str(HERE / "track_regression_analysis_adapteronly.yaml"),
        help="Analysis YAML file",
    )
    parser.add_argument("--checkpoint", help="Override the checkpoint in the YAML")
    parser.add_argument("--output-dir", help="Override the output directory")
    parser.add_argument("--max-samples", type=int, help="Override the sample limit")
    parser.add_argument("--model-yaml", help="Override the model YAML in the analysis config")
    parser.add_argument("--model-config", help="Override the model config name in the analysis config")
    parser.add_argument("--run-name", help="Override run_name metadata")
    parser.add_argument("--analysis-tag", help="Override analysis_tag metadata")
    parser.add_argument("--training-log", help="Override the training log path")
    parser.add_argument("--pretrained-checkpoint", help="Override pretrained backbone checkpoint metadata/path")
    parser.add_argument(
        "--use-pretrained-backbone",
        choices=("true", "false"),
        help="Override whether evaluation should run the frozen pretrained backbone",
    )
    return parser.parse_args()


def expand_env_defaults(value):
    def replace(match):
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default if default is not None else match.group(0))

    return ENV_DEFAULT_RE.sub(replace, value)


def expand_config_strings(config):
    expanded = {}
    for key, value in config.items():
        if isinstance(value, str):
            value = os.path.expanduser(os.path.expandvars(expand_env_defaults(value)))
        expanded[key] = value

    for _ in range(3):
        format_map = ConfigFormatMap(expanded)
        next_config = {}
        changed = False
        for key, value in expanded.items():
            if isinstance(value, str):
                formatted = value.format_map(format_map)
                changed = changed or formatted != value
                value = formatted
            next_config[key] = value
        expanded = next_config
        if not changed:
            break
    return expanded


def resolve_path(value, base_dir):
    if value is None:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(expand_env_defaults(str(value)))))
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_analysis_config(args):
    config_path = Path(args.analysis_config).resolve()
    with config_path.open() as stream:
        config = expand_config_strings(YAML(typ="safe").load(stream)["analysis"])
    config.setdefault("comparison_truth", "mctrue_swingback_doca")
    config.setdefault("swingback_enabled", True)
    config.setdefault("swingback_r_hit_cm", 6.5)
    config.setdefault("swingback_magnetic_field_t", 5.0)
    config.setdefault("swingback_polarity", 1)
    config.setdefault("charge_source", "metadata_or_positive")
    config.setdefault("fallback_charge", 1)
    config.setdefault("write_unswung_diagnostics", True)
    config.setdefault(
        "delta_p_over_p_bins_gev",
        DEFAULT_DELTA_P_OVER_P_BINS_GEV,
    )
    config.setdefault("delta_p_over_p_min_bin_entries", 200)
    config.setdefault("delta_p_over_p_min_populated_histogram_bins", 8)
    config.setdefault("delta_p_over_p_histogram_bins", 40)
    config.setdefault("delta_p_over_p_fit_quantile", 0.98)
    for key, attr in (
        ("model_config", "model_config"),
        ("run_name", "run_name"),
        ("analysis_tag", "analysis_tag"),
    ):
        value = getattr(args, attr)
        if value is not None:
            config[key] = value
    if args.use_pretrained_backbone is not None:
        config["use_pretrained_backbone"] = args.use_pretrained_backbone == "true"
    config["model_yaml"] = resolve_path(config["model_yaml"], config_path.parent)
    if args.model_yaml is not None:
        config["model_yaml"] = resolve_path(args.model_yaml, config_path.parent)
    config["checkpoint"] = resolve_path(
        args.checkpoint or config["checkpoint"], config_path.parent
    )
    config["output_dir"] = resolve_path(
        args.output_dir or config["output_dir"], config_path.parent
    )
    config["training_log"] = resolve_path(
        args.training_log or config.get("training_log"), config_path.parent
    )
    if args.pretrained_checkpoint is not None:
        config["pretrained_checkpoint"] = str(
            resolve_path(args.pretrained_checkpoint, config_path.parent)
        )
        if args.use_pretrained_backbone is None:
            config["use_pretrained_backbone"] = True
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


def wrap_phi_rad(phi):
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def swingback_phi_to_doca(
    px_gev,
    py_gev,
    charge,
    r_hit_cm=6.5,
    magnetic_field_t=5.0,
    polarity=1,
):
    """Swim transverse MC::True momentum direction from CVT entrance to DOCA."""
    px_gev = np.asarray(px_gev, dtype=float)
    py_gev = np.asarray(py_gev, dtype=float)
    charge = np.asarray(charge, dtype=float)
    pt = np.hypot(px_gev, py_gev)
    phi_hit = np.arctan2(py_gev, px_gev)

    valid = (
        (pt > 0)
        & (charge != 0)
        & np.isfinite(pt)
        & np.isfinite(charge)
        & np.isfinite(phi_hit)
    )
    phi_doca = np.full_like(phi_hit, np.nan, dtype=float)

    radius_cm = np.full_like(pt, np.nan, dtype=float)
    radius_cm[valid] = pt[valid] / (0.3 * float(magnetic_field_t)) * 100.0

    arg = np.full_like(pt, np.nan, dtype=float)
    arg[valid] = float(r_hit_cm) / (2.0 * radius_cm[valid])
    arg = np.clip(arg, -1.0, 1.0)

    dphi = int(polarity) * np.sign(charge) * 2.0 * np.arcsin(arg)
    phi_doca[valid] = wrap_phi_rad(phi_hit[valid] - dphi[valid])
    return phi_doca


def swingback_vector_to_doca(vector_gev, charge, config):
    vector_gev = np.asarray(vector_gev, dtype=float)
    px, py, pz = np.moveaxis(vector_gev, -1, 0)
    pt = np.hypot(px, py)
    phi_doca = swingback_phi_to_doca(
        px,
        py,
        charge,
        r_hit_cm=float(config["swingback_r_hit_cm"]),
        magnetic_field_t=float(config["swingback_magnetic_field_t"]),
        polarity=int(config["swingback_polarity"]),
    )
    return np.stack((pt * np.cos(phi_doca), pt * np.sin(phi_doca), pz), axis=-1)


def kinematic_variables(vector):
    p, pt, theta, phi = vector_kinematics(vector)
    px, py, pz = np.moveaxis(np.asarray(vector), -1, 0)
    eta = np.arcsinh(
        np.divide(pz, pt, out=np.full_like(pz, np.nan), where=pt > 1e-12)
    )
    return {
        "p_gev": p,
        "pt_gev": pt,
        "theta_deg": theta,
        "phi_deg": phi,
        "eta": eta,
    }


def wrapped_angle_residual_deg(estimate, truth):
    return (estimate - truth + 180.0) % 360.0 - 180.0


def kinematic_residuals(truth, prediction):
    truth_variables = kinematic_variables(truth)
    predicted_variables = kinematic_variables(prediction)
    absolute = {}
    for name in KINEMATIC_LABELS:
        if name == "phi_deg":
            residual = wrapped_angle_residual_deg(
                predicted_variables[name], truth_variables[name]
            )
        else:
            residual = predicted_variables[name] - truth_variables[name]
        absolute[name] = residual
    return truth_variables, absolute


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


def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def fit_gaussian_residuals(values, histogram_bins, fit_quantile, min_populated_bins):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None, None, None, None, "empty"

    tail = 0.5 * (1.0 - float(fit_quantile))
    if 0.0 < tail < 0.5 and len(values) > 2:
        low, high = np.quantile(values, [tail, 1.0 - tail])
        fit_values = values[(values >= low) & (values <= high)]
    else:
        fit_values = values
    if len(fit_values) < 3:
        return None, None, None, None, "too_few_after_trim"

    moment_mean = float(np.mean(fit_values))
    moment_sigma = float(np.std(fit_values, ddof=1)) if len(fit_values) > 1 else 0.0
    if not math.isfinite(moment_sigma) or moment_sigma <= 0:
        return safe_float(moment_mean), None, None, None, "zero_width"

    counts, edges = np.histogram(fit_values, bins=int(histogram_bins))
    centers = 0.5 * (edges[:-1] + edges[1:])
    populated = counts > 0
    if int(np.sum(populated)) < int(min_populated_bins):
        return None, None, None, None, "skipped_sparse_histogram"

    try:
        from scipy.optimize import curve_fit

        p0 = [float(counts.max()), moment_mean, moment_sigma]
        bounds = ([0.0, float(edges[0]), 1e-12], [np.inf, float(edges[-1]), np.inf])
        params, covariance = curve_fit(
            gaussian,
            centers[populated],
            counts[populated],
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )
        errors = np.sqrt(np.diag(covariance)) if covariance.size else np.full(3, np.nan)
        _, mean, sigma = params
        _, mean_error, sigma_error = errors
        return (
            safe_float(mean),
            safe_float(abs(sigma)),
            safe_float(mean_error) if math.isfinite(float(mean_error)) else None,
            safe_float(sigma_error) if math.isfinite(float(sigma_error)) else None,
            "ok",
        )
    except Exception:
        return (
            safe_float(moment_mean),
            safe_float(moment_sigma),
            None,
            None,
            "moment_fallback_fit_failed",
        )


def calculate_delta_p_over_p_fit_rows(truth, predictions, bins, config):
    true_p, _, _, _ = vector_kinematics(truth)
    edges = np.asarray(bins, dtype=float)
    min_entries = int(config["delta_p_over_p_min_bin_entries"])
    histogram_bins = int(config["delta_p_over_p_histogram_bins"])
    min_populated_bins = int(config["delta_p_over_p_min_populated_histogram_bins"])
    fit_quantile = float(config["delta_p_over_p_fit_quantile"])
    rows = []

    for index in range(len(edges) - 1):
        low = float(edges[index])
        high = float(edges[index + 1])
        label = f"[{low}, {high})"
        selection = (true_p >= low) & (true_p < high) & (true_p > 1e-12)
        for method in PHYSICS_PLOT_METHODS:
            pred_p, _, _, _ = vector_kinematics(predictions[method])
            valid = selection & np.isfinite(true_p) & np.isfinite(pred_p)
            residual = (pred_p[valid] - true_p[valid]) / true_p[valid]
            if len(residual) < min_entries:
                fit_mean = fit_sigma = fit_mean_error = fit_sigma_error = None
                fit_status = "skipped_sparse"
            else:
                fit_mean, fit_sigma, fit_mean_error, fit_sigma_error, fit_status = (
                    fit_gaussian_residuals(
                        residual, histogram_bins, fit_quantile, min_populated_bins
                    )
                )
            rows.append({
                "group": "truth_p_gev",
                "bin": label,
                "bin_low_gev": low,
                "bin_high_gev": high,
                "bin_center_gev": 0.5 * (low + high),
                "method": method,
                "n": int(len(residual)),
                "fit_mean": fit_mean,
                "fit_sigma": fit_sigma,
                "fit_mean_error": fit_mean_error,
                "fit_sigma_error": fit_sigma_error,
                "fit_status": fit_status,
            })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    with Path(path).open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


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


def resolve_charge(records, config):
    source = str(config.get("charge_source", "metadata_or_positive"))
    fallback = int(config.get("fallback_charge", 1))
    if fallback not in (-1, 0, 1):
        raise ValueError(f"fallback_charge must be -1, 0, or 1; got {fallback}")

    charges = []
    missing = 0
    for record in records:
        value = record.get("charge")
        try:
            value = float(value)
            finite = np.isfinite(value)
        except (TypeError, ValueError):
            finite = False
        if not finite:
            missing += 1
            value = fallback
        charges.append(int(np.sign(value)))
    charges = np.asarray(charges, dtype=int)

    if source == "positive":
        charges = np.ones(len(records), dtype=int)
        missing = 0
    elif source == "metadata":
        if missing:
            raise ValueError(
                f"charge_source=metadata but {missing} records do not have metadata charge"
            )
    elif source == "metadata_or_positive":
        pass
    elif source == "predicted_pid":
        raise NotImplementedError(
            "charge_source=predicted_pid is reserved for future PID-prediction evaluation"
        )
    else:
        raise ValueError(
            "charge_source must be one of: positive, metadata, metadata_or_positive, "
            f"predicted_pid; got {source!r}"
        )

    return charges, {
        "source": source,
        "fallback_charge": fallback,
        "records_missing_metadata_charge": int(missing),
        "n_positive": int(np.sum(charges > 0)),
        "n_negative": int(np.sum(charges < 0)),
        "n_neutral": int(np.sum(charges == 0)),
    }


def attach_vector_kinematics(record, prefix, vector):
    p, pt, theta, phi = vector_kinematics(vector)
    eta = kinematic_variables(vector)["eta"]
    record.update({
        f"{prefix}_px_gev": vector[0],
        f"{prefix}_py_gev": vector[1],
        f"{prefix}_pz_gev": vector[2],
        f"{prefix}_p_gev": p,
        f"{prefix}_pt_gev": pt,
        f"{prefix}_theta_deg": theta,
        f"{prefix}_phi_deg": phi,
        f"{prefix}_eta": eta,
    })


def make_swingback_diagnostics(raw_truth, doca_truth, charge, config):
    result = {
        "enabled": bool(config.get("swingback_enabled", True)),
        "comparison_truth": config.get("comparison_truth", "mctrue_swingback_doca"),
        "r_hit_cm": safe_float(config["swingback_r_hit_cm"]),
        "magnetic_field_t": safe_float(config["swingback_magnetic_field_t"]),
        "polarity": int(config["swingback_polarity"]),
        "n": int(len(raw_truth)),
        "write_unswung_diagnostics": bool(
            config.get("write_unswung_diagnostics", True)
        ),
        "charge_counts": {
            "positive": int(np.sum(charge > 0)),
            "negative": int(np.sum(charge < 0)),
            "neutral": int(np.sum(charge == 0)),
        },
    }
    if not result["write_unswung_diagnostics"]:
        return result

    raw_p, raw_pt, _, raw_phi = vector_kinematics(raw_truth)
    doca_p, doca_pt, _, doca_phi = vector_kinematics(doca_truth)
    dphi = wrapped_angle_residual_deg(doca_phi, raw_phi)
    vector_delta = np.linalg.norm(doca_truth - raw_truth, axis=1)
    finite_dphi = dphi[np.isfinite(dphi)]
    finite_delta = vector_delta[np.isfinite(vector_delta)]
    result.update({
        "pt_preserved_max_abs_gev": safe_float(np.nanmax(np.abs(doca_pt - raw_pt))),
        "p_preserved_max_abs_gev": safe_float(np.nanmax(np.abs(doca_p - raw_p))),
        "pz_preserved_max_abs_gev": safe_float(
            np.nanmax(np.abs(doca_truth[:, 2] - raw_truth[:, 2]))
        ),
        "doca_phi_minus_inner_phi_deg": scalar_error_metrics(finite_dphi),
        "vector_delta_gev": scalar_error_metrics(finite_delta),
    })
    return result


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


def audit_cvtrec_rec_particle(records):
    """Check the expected CVTRec::Tracks -> REC::Particle momentum copy."""
    cvtrec = np.asarray([
        [record[f"cvtrec_{component}_gev"] for component in ("px", "py", "pz")]
        for record in records
    ])
    rec_particle = np.asarray([
        [record[f"rec_particle_{component}_gev"] for component in ("px", "py", "pz")]
        for record in records
    ])
    finite = np.all(np.isfinite(cvtrec), axis=1) & np.all(
        np.isfinite(rec_particle), axis=1
    )
    cvtrec = cvtrec[finite]
    rec_particle = rec_particle[finite]
    if not len(cvtrec):
        return {"n_finite_pairs": 0, "consistent": False}
    difference = rec_particle - cvtrec
    exact = np.all(difference == 0, axis=1)
    close = np.all(np.isclose(rec_particle, cvtrec, rtol=1e-5, atol=1e-7), axis=1)
    cvtrec_p = np.linalg.norm(cvtrec, axis=1)
    rec_p = np.linalg.norm(rec_particle, axis=1)
    correlation = np.corrcoef(cvtrec_p, rec_p)[0, 1] if len(cvtrec) > 1 else np.nan
    return {
        "n_finite_pairs": int(len(cvtrec)),
        "exact_match_fraction": safe_float(np.mean(exact)),
        "close_match_fraction": safe_float(np.mean(close)),
        "median_vector_difference_gev": safe_float(
            np.median(np.linalg.norm(difference, axis=1))
        ),
        "momentum_magnitude_correlation": safe_float(correlation),
        "consistent": bool(np.all(close)),
        "action": (
            "REC::Particle excluded from performance comparisons unless this audit is consistent"
        ),
    }


def infer_training_log_path(config):
    if config.get("training_log") is not None:
        return config["training_log"]
    checkpoint = config.get("checkpoint")
    if checkpoint is None:
        return None
    checkpoint = Path(checkpoint)
    if checkpoint.name.endswith("_checkpoint.pth"):
        return checkpoint.with_name(checkpoint.name.replace("_checkpoint.pth", ".log"))
    return checkpoint.with_suffix(".log")


def read_training_history(path):
    if path is None or not Path(path).exists():
        return [], {
            "path": str(path) if path is not None else None,
            "found": False,
            "n_epochs": 0,
        }

    rows = []
    with Path(path).open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            try:
                rows.append({
                    "epoch": int(row["Epoch"]),
                    "train_loss": float(row["Train_Loss"]),
                    "val_loss": float(row["Val_Loss"]),
                    "time_sec": float(row["Time"]),
                })
            except (KeyError, TypeError, ValueError):
                continue

    summary = {
        "path": str(path),
        "found": True,
        "n_epochs": len(rows),
    }
    if rows:
        val_losses = np.asarray([row["val_loss"] for row in rows], dtype=float)
        train_losses = np.asarray([row["train_loss"] for row in rows], dtype=float)
        best_index = int(np.argmin(val_losses))
        summary.update({
            "best_epoch": rows[best_index]["epoch"],
            "best_val_loss": safe_float(val_losses[best_index]),
            "final_train_loss": safe_float(train_losses[-1]),
            "final_val_loss": safe_float(val_losses[-1]),
            "final_val_train_gap": safe_float(val_losses[-1] - train_losses[-1]),
        })
    return rows, summary


def write_training_history(path, rows):
    write_csv(path, rows)


def scalar_error_metrics(errors):
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    if not len(errors):
        return {
            "n": 0,
            "bias": None,
            "mae": None,
            "rmse": None,
            "median_absolute_error": None,
            "p95_absolute_error": None,
        }
    absolute = np.abs(errors)
    return {
        "n": int(len(errors)),
        "bias": safe_float(np.mean(errors)),
        "mae": safe_float(np.mean(absolute)),
        "rmse": safe_float(np.sqrt(np.mean(errors ** 2))),
        "median_absolute_error": safe_float(np.median(absolute)),
        "p95_absolute_error": safe_float(np.quantile(absolute, 0.95)),
    }


def ml_variable_arrays(truth, prediction):
    truth_kinematics = kinematic_variables(truth)
    prediction_kinematics = kinematic_variables(prediction)
    truth_components = {
        "px_gev": truth[:, 0],
        "py_gev": truth[:, 1],
        "pz_gev": truth[:, 2],
    }
    prediction_components = {
        "px_gev": prediction[:, 0],
        "py_gev": prediction[:, 1],
        "pz_gev": prediction[:, 2],
    }
    return truth_components, prediction_components, truth_kinematics, prediction_kinematics


def calculate_ml_metrics(truth, predictions):
    rows = []
    nested = {}
    for method, prediction in predictions.items():
        truth_components, prediction_components, truth_kinematics, prediction_kinematics = (
            ml_variable_arrays(truth, prediction)
        )
        nested[method] = {}

        for variable, label in COMPONENT_LABELS.items():
            errors = prediction_components[variable] - truth_components[variable]
            metrics = scalar_error_metrics(errors)
            rows.append({
                "method": method,
                "space": "component",
                "variable": variable,
                "label": label,
                "unit": "GeV",
                **metrics,
            })
            nested[method][variable] = metrics

        for variable, label in KINEMATIC_LABELS.items():
            if variable == "phi_deg":
                errors = wrapped_angle_residual_deg(
                    prediction_kinematics[variable], truth_kinematics[variable]
                )
            else:
                errors = prediction_kinematics[variable] - truth_kinematics[variable]
            unit = "deg" if variable.endswith("_deg") else "GeV" if variable.endswith("_gev") else ""
            metrics = scalar_error_metrics(errors)
            rows.append({
                "method": method,
                "space": "kinematic",
                "variable": variable,
                "label": label,
                "unit": unit,
                **metrics,
            })
            nested[method][variable] = metrics

        vector_error = np.linalg.norm(prediction - truth, axis=1)
        metrics = scalar_error_metrics(vector_error)
        # Vector norm error is already non-negative, so bias is not physically
        # meaningful. Keep the column populated in the CSV, but make this clear
        # in the nested JSON.
        metrics_for_json = dict(metrics)
        metrics_for_json["bias"] = None
        rows.append({
            "method": method,
            "space": "vector",
            "variable": "vector_error_gev",
            "label": r"||p estimate - p truth|| [GeV]",
            "unit": "GeV",
            **metrics_for_json,
        })
        nested[method]["vector_error_gev"] = metrics_for_json
    return rows, nested


def campaign_metadata(config, summary):
    metadata = {
        "run_name": config.get("run_name"),
        "analysis_tag": config.get("analysis_tag"),
        "run_num": config.get("run_num"),
        "model_config": config.get("model_config"),
        "checkpoint": str(config.get("checkpoint")),
        "output_dir": str(config.get("output_dir")),
        "analysis_config": str(config.get("analysis_config")),
        "comparison_truth": summary.get("comparison_truth"),
        "comparison_truth_definition": summary.get("comparison_truth_definition"),
        "max_samples": config.get("max_samples"),
        "use_pretrained_backbone": config.get("use_pretrained_backbone"),
        "pretrained_checkpoint": (
            str(config.get("pretrained_checkpoint"))
            if config.get("pretrained_checkpoint") is not None else None
        ),
    }
    training = summary.get("training_history", {})
    for key in (
        "path", "found", "n_epochs", "best_epoch", "best_val_loss",
        "final_train_loss", "final_val_loss", "final_val_train_gap",
    ):
        metadata[f"training_{key}"] = training.get(key)
    return metadata


def build_campaign_headline_rows(config, summary, ml_metric_rows):
    metadata = campaign_metadata(config, summary)
    rows = []

    component_r2 = {}
    for method, metrics in summary["methods"].items():
        for component, values in metrics.get("components", {}).items():
            component_r2[(method, f"{component}_gev")] = values.get("r2")

    for metric_row in ml_metric_rows:
        rows.append({
            **metadata,
            "record_type": "ml_error",
            "method": metric_row["method"],
            "space": metric_row["space"],
            "variable": metric_row["variable"],
            "label": metric_row["label"],
            "unit": metric_row["unit"],
            "n": metric_row["n"],
            "bias": metric_row["bias"],
            "mae": metric_row["mae"],
            "rmse": metric_row["rmse"],
            "r2": component_r2.get((metric_row["method"], metric_row["variable"])),
            "median_absolute_error": metric_row["median_absolute_error"],
            "p95_absolute_error": metric_row["p95_absolute_error"],
        })

    for method, metrics in summary["methods"].items():
        momentum = metrics.get("momentum", {})
        tolerance_fractions = {
            f"fraction_within_relative_tolerance_{str(tolerance).replace('.', 'p')}": value
            for tolerance, value in momentum.get(
                "fraction_within_relative_tolerance", {}
            ).items()
        }
        rows.append({
            **metadata,
            "record_type": "physics_summary",
            "method": method,
            "space": "kinematic",
            "variable": "p_gev",
            "label": r"p [GeV]",
            "unit": "GeV",
            "n": metrics.get("n"),
            "bias": momentum.get("bias_gev"),
            "mae": momentum.get("mae_gev"),
            "rmse": momentum.get("rmse_gev"),
            "r2": None,
            "relative_bias": momentum.get("relative_bias"),
            "relative_median": momentum.get("relative_median"),
            "relative_resolution_68": momentum.get("relative_resolution_68"),
            "relative_tail_fraction_10pct": momentum.get("relative_tail_fraction_10pct"),
            "relative_tail_fraction_25pct": momentum.get("relative_tail_fraction_25pct"),
            **tolerance_fractions,
        })

        pt = metrics.get("pt", {})
        rows.append({
            **metadata,
            "record_type": "physics_summary",
            "method": method,
            "space": "kinematic",
            "variable": "pt_gev",
            "label": r"pT [GeV]",
            "unit": "GeV",
            "n": metrics.get("n"),
            "bias": None,
            "mae": pt.get("mae_gev"),
            "rmse": pt.get("rmse_gev"),
            "r2": None,
        })

        direction = metrics.get("direction", {})
        rows.append({
            **metadata,
            "record_type": "physics_summary",
            "method": method,
            "space": "direction",
            "variable": "opening_angle_deg",
            "label": "opening angle [deg]",
            "unit": "deg",
            "n": metrics.get("n"),
            "opening_angle_mean": direction.get("opening_angle_mean_deg"),
            "opening_angle_median": direction.get("opening_angle_median_deg"),
            "opening_angle_68": direction.get("opening_angle_68_deg"),
        })

    rows.append({
        **metadata,
        "record_type": "comparison_summary",
        "method": "adapter_over_cvt",
        "space": "kinematic",
        "variable": "p_gev",
        "label": "adapter/cvt relative p resolution",
        "unit": "",
        "adapter_to_cvt_resolution_ratio": summary.get(
            "adapter_to_cvt_resolution_ratio"
        ),
    })
    return rows


def robust_symmetric_limit(values, quantile):
    values = np.concatenate([np.ravel(value) for value in values])
    values = np.abs(values[np.isfinite(values)])
    if not len(values):
        return 1.0
    limit = float(np.quantile(values, quantile))
    return limit if limit > 0 else 1.0


def robust_range(values, quantile):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return (-1.0, 1.0)
    tail = 0.5 * (1.0 - quantile)
    low, high = np.quantile(values, [tail, 1.0 - tail])
    if high <= low:
        return (float(low) - 0.5, float(high) + 0.5)
    return float(low), float(high)


def make_physics_residual_plots(output_dir, truth, predictions, config):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    plot_dir = output_dir / "plots" / "physics_2d"
    plot_dir.mkdir(parents=True, exist_ok=True)
    bins = int(config["physics_histogram_bins"])
    quantile = float(config["physics_plot_quantile"])
    method_results = {
        method: kinematic_residuals(truth, predictions[method])
        for method in PHYSICS_PLOT_METHODS
    }
    truth_variables = method_results["adapter"][0]

    x_ranges = {
        name: ((-180.0, 180.0) if name == "phi_deg" else
               (0.0, 180.0) if name == "theta_deg" else
               robust_range(values, quantile))
        for name, values in truth_variables.items()
    }

    y_limits = {
        name: robust_symmetric_limit(
            [method_results[method][1][name]
             for method in PHYSICS_PLOT_METHODS],
            quantile,
        )
        for name in KINEMATIC_LABELS
    }
    for residual_name in KINEMATIC_LABELS:
        histograms = {}
        maximum_count = 1
        for method in PHYSICS_PLOT_METHODS:
            residual = method_results[method][1][residual_name]
            for truth_name, truth_values in truth_variables.items():
                valid = np.isfinite(truth_values) & np.isfinite(residual)
                x_edges = np.linspace(*x_ranges[truth_name], bins + 1)
                y_edges = np.linspace(
                    -y_limits[residual_name], y_limits[residual_name], bins + 1
                )
                counts, _, _ = np.histogram2d(
                    truth_values[valid], residual[valid], bins=[x_edges, y_edges]
                )
                histograms[(method, truth_name)] = (counts, x_edges, y_edges)
                maximum_count = max(maximum_count, int(counts.max()))

        fig, axes = plt.subplots(2, 5, figsize=(24, 9), constrained_layout=True)
        image = None
        for row, method in enumerate(PHYSICS_PLOT_METHODS):
            for column, truth_name in enumerate(KINEMATIC_LABELS):
                axis = axes[row, column]
                counts, x_edges, y_edges = histograms[(method, truth_name)]
                image = axis.pcolormesh(
                    x_edges, y_edges, counts.T,
                    norm=LogNorm(vmin=1, vmax=max(2, maximum_count)), cmap="viridis",
                    shading="auto",
                )
                axis.axhline(0.0, color="white", linewidth=0.8, alpha=0.8)
                axis.set_xlabel(f"True {KINEMATIC_LABELS[truth_name]}")
                if column == 0:
                    title_method = "Adapter" if method == "adapter" else "CVT::Tracks"
                    axis.set_ylabel(
                        f"{title_method}\nReco - true {KINEMATIC_LABELS[residual_name]}"
                    )
        fig.suptitle(
            f"Adapter and CVT::Tracks: {KINEMATIC_LABELS[residual_name]} residual",
            fontsize=16,
        )
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Tracks per bin")
        fig.savefig(plot_dir / f"comparison_residual_{residual_name}.png", dpi=170)
        plt.close(fig)

    make_direct_error_comparison_plots(
        plot_dir, method_results, bins=bins, quantile=quantile
    )


def make_direct_error_comparison_plots(plot_dir, method_results, bins, quantile):
    """Compare per-track absolute errors; below the diagonal favors Adapter."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    adapter_residuals = method_results["adapter"][1]
    cvt_residuals = method_results["cvt"][1]
    for name in KINEMATIC_LABELS:
        adapter_error = np.abs(adapter_residuals[name])
        cvt_error = np.abs(cvt_residuals[name])
        valid = np.isfinite(adapter_error) & np.isfinite(cvt_error)
        adapter_error = adapter_error[valid]
        cvt_error = cvt_error[valid]
        limit = float(np.quantile(np.concatenate([adapter_error, cvt_error]), quantile))
        if limit <= 0:
            limit = 1.0
        adapter_better = float(np.mean(adapter_error < cvt_error))

        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        image = ax.hist2d(
            cvt_error, adapter_error, bins=bins, range=[[0, limit], [0, limit]],
            norm=LogNorm(vmin=1), cmap="viridis",
        )[3]
        ax.plot([0, limit], [0, limit], color="red", linewidth=1.3,
                label="Equal absolute error")
        ax.set(
            xlabel=f"CVT::Tracks absolute error in {KINEMATIC_LABELS[name]}",
            ylabel=f"Adapter absolute error in {KINEMATIC_LABELS[name]}",
            title=f"Direct per-track comparison: {KINEMATIC_LABELS[name]}",
            xlim=(0, limit), ylim=(0, limit),
        )
        ax.text(
            0.03, 0.95,
            f"Adapter has smaller error: {100 * adapter_better:.1f}%",
            transform=ax.transAxes, va="top", color="white",
            bbox={"facecolor": "black", "alpha": 0.6, "edgecolor": "none"},
        )
        ax.legend(loc="lower right")
        fig.colorbar(image, ax=ax, label="Tracks per bin")
        fig.savefig(plot_dir / f"direct_error_adapter_vs_cvt_{name}.png", dpi=170)
        plt.close(fig)


def make_training_curve_plot(output_dir, training_history):
    if not training_history:
        return

    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots" / "ml"
    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.asarray([row["epoch"] for row in training_history], dtype=int)
    train_loss = np.asarray([row["train_loss"] for row in training_history], dtype=float)
    val_loss = np.asarray([row["val_loss"] for row in training_history], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    ax.plot(epochs, train_loss, marker="o", markersize=3, label="Train loss")
    ax.plot(epochs, val_loss, marker="o", markersize=3, label="Validation loss")
    best = int(np.nanargmin(val_loss))
    ax.axvline(epochs[best], color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(
        epochs[best], val_loss[best],
        f" best val\n epoch {epochs[best]}",
        va="bottom", ha="left",
    )
    ax.set(
        xlabel="Epoch",
        ylabel="Loss",
        title="Training and validation loss",
    )
    if np.all(train_loss > 0) and np.all(val_loss > 0):
        ax.set_yscale("log")
    ax.legend()
    fig.savefig(plot_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def make_ml_error_bar_plot(plot_dir, metric_rows, space, filename, title):
    import matplotlib.pyplot as plt

    rows = [row for row in metric_rows if row["space"] == space]
    if not rows:
        return
    methods = [method for method in METHODS if any(row["method"] == method for row in rows)]
    variables = []
    labels = {}
    for row in rows:
        if row["variable"] not in variables:
            variables.append(row["variable"])
            labels[row["variable"]] = row["label"]

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 8), constrained_layout=True, sharex=True
    )
    axes = axes.ravel()
    x = np.arange(len(variables))
    width = 0.8 / max(1, len(methods))
    for axis, metric in zip(axes, ML_METRIC_LABELS):
        for offset, method in enumerate(methods):
            values = []
            for variable in variables:
                match = next(
                    (
                        row for row in rows
                        if row["method"] == method and row["variable"] == variable
                    ),
                    None,
                )
                values.append(np.nan if match is None or match[metric] is None else match[metric])
            axis.bar(
                x + (offset - 0.5 * (len(methods) - 1)) * width,
                values,
                width=width,
                label=method,
            )
        axis.set_title(ML_METRIC_LABELS[metric])
        axis.grid(axis="y", alpha=0.25)
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels([labels[variable] for variable in variables], rotation=25, ha="right")
    axes[0].legend()
    fig.suptitle(title)
    fig.savefig(plot_dir / filename, dpi=160)
    plt.close(fig)


def make_absolute_error_cdf_plot(plot_dir, truth, predictions):
    import matplotlib.pyplot as plt

    variables = {
        **COMPONENT_LABELS,
        **KINEMATIC_LABELS,
        "vector_error_gev": r"||p estimate - p truth|| [GeV]",
    }
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    axes = axes.ravel()
    for axis, (variable, label) in zip(axes, variables.items()):
        for method in METHODS:
            prediction = predictions[method]
            truth_components, prediction_components, truth_kinematics, prediction_kinematics = (
                ml_variable_arrays(truth, prediction)
            )
            if variable in COMPONENT_LABELS:
                errors = prediction_components[variable] - truth_components[variable]
            elif variable in KINEMATIC_LABELS:
                if variable == "phi_deg":
                    errors = wrapped_angle_residual_deg(
                        prediction_kinematics[variable], truth_kinematics[variable]
                    )
                else:
                    errors = prediction_kinematics[variable] - truth_kinematics[variable]
            else:
                errors = np.linalg.norm(prediction - truth, axis=1)
            absolute = np.sort(np.abs(errors[np.isfinite(errors)]))
            if len(absolute):
                cdf = np.arange(1, len(absolute) + 1) / len(absolute)
                axis.plot(absolute, cdf, label=method)
        axis.set_title(label)
        axis.set_xlabel("Absolute error")
        axis.set_ylabel("Cumulative fraction")
        axis.grid(alpha=0.25)
    for axis in axes[len(variables):]:
        axis.axis("off")
    axes[0].legend()
    fig.suptitle("Absolute error cumulative distributions")
    fig.savefig(plot_dir / "absolute_error_cdf.png", dpi=160)
    plt.close(fig)


def make_ml_plots(output_dir, truth, predictions, metric_rows, training_history):
    plot_dir = output_dir / "plots" / "ml"
    plot_dir.mkdir(parents=True, exist_ok=True)
    make_training_curve_plot(output_dir, training_history)
    make_ml_error_bar_plot(
        plot_dir, metric_rows, "component",
        "ml_error_bars_components.png",
        "Final evaluation errors: Cartesian momentum components",
    )
    make_ml_error_bar_plot(
        plot_dir, metric_rows, "kinematic",
        "ml_error_bars_kinematics.png",
        "Final evaluation errors: derived kinematics",
    )
    make_absolute_error_cdf_plot(plot_dir, truth, predictions)


def make_delta_p_over_p_plot(output_dir, rows):
    import matplotlib.pyplot as plt

    percent = 100.0
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    methods = [method for method in PHYSICS_PLOT_METHODS if any(row["method"] == method for row in rows)]
    if not methods:
        return

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    labels = {"adapter": "Adapter", "cvt": "CVT::Tracks"}
    for method in methods:
        method_rows = [
            row for row in rows
            if row["method"] == method
            and row.get("fit_mean") is not None
            and row.get("fit_sigma") is not None
        ]
        if not method_rows:
            continue
        method_rows = sorted(method_rows, key=lambda row: row["bin_center_gev"])
        x = np.asarray([row["bin_center_gev"] for row in method_rows], dtype=float)
        y = percent * np.asarray([row["fit_mean"] for row in method_rows], dtype=float)
        yerr = percent * np.asarray([row["fit_sigma"] for row in method_rows], dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=labels.get(method, method),
        )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set(
        xlabel="True p [GeV]",
        ylabel=r"Gaussian mean of $(p_{reco} - p_{true}) / p_{true}$ [%]",
        title=r"Momentum bias and resolution from fitted $\Delta p / p$",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(plot_dir / "delta_p_over_p_vs_true_p.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for method in methods:
        method_rows = [
            row for row in rows
            if row["method"] == method
            and row.get("fit_mean") is not None
        ]
        if not method_rows:
            continue
        method_rows = sorted(method_rows, key=lambda row: row["bin_center_gev"])
        ax.plot(
            [row["bin_center_gev"] for row in method_rows],
            [percent * row["fit_mean"] for row in method_rows],
            marker="o",
            linewidth=1.5,
            label=labels.get(method, method),
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set(
        xlabel="True p [GeV]",
        ylabel=r"Gaussian mean of $(p_{reco} - p_{true}) / p_{true}$ [%]",
        title=r"Momentum bias from fitted $\Delta p / p$",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(plot_dir / "delta_p_over_p_mean_vs_true_p.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for method in methods:
        method_rows = [
            row for row in rows
            if row["method"] == method
            and row.get("fit_sigma") is not None
        ]
        if not method_rows:
            continue
        method_rows = sorted(method_rows, key=lambda row: row["bin_center_gev"])
        ax.plot(
            [row["bin_center_gev"] for row in method_rows],
            [percent * row["fit_sigma"] for row in method_rows],
            marker="o",
            linewidth=1.5,
            label=labels.get(method, method),
        )
    ax.set(
        xlabel="True p [GeV]",
        ylabel=r"Gaussian sigma of $(p_{reco} - p_{true}) / p_{true}$ [%]",
        title=r"Momentum resolution from fitted $\Delta p / p$",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(plot_dir / "delta_p_over_p_sigma_vs_true_p.png", dpi=160)
    plt.close(fig)


def make_plots(
    output_dir, truth, predictions, true_p, true_theta, momentum_bins,
    theta_bins, config,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Remove plots from earlier analysis definitions so rerunning in the same
    # output directory cannot leave obsolete normalized/68%-width figures.
    for obsolete in (
        plot_dir / "resolution_vs_p.png",
        plot_dir / "resolution_vs_theta.png",
    ):
        obsolete.unlink(missing_ok=True)
    physics_dir = plot_dir / "physics_2d"
    if physics_dir.exists():
        for obsolete in physics_dir.glob("*_normalized_*.png"):
            obsolete.unlink()
        for obsolete in physics_dir.glob("*_absolute_*.png"):
            obsolete.unlink()

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

    make_physics_residual_plots(output_dir, truth, predictions, config)


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
    regression_task = trainer.regression_target_stats["task"]

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
            prediction_native = trainer.down_model.target_normalizer.denormalize(prediction).cpu().numpy()
            truth_native = trainer.down_model.target_normalizer.denormalize(normalized_truth).cpu().numpy()
            prediction = target_to_cartesian_numpy(prediction_native, regression_task)
            truth = target_to_cartesian_numpy(truth_native, regression_task)

            take = min(batch_size, remaining)
            for local_index in range(take):
                dataset_index = cursor + local_index
                real_index = int(dataset.idxlist[dataset_index])
                aux_row = first_valid_row(aux[real_index]).astype(float) * aux_scale
                pid_values = np.asarray(dataset.memmap_pid_target[real_index]).reshape(-1)
                pid_class = int(pid_values[0]) if len(pid_values) else -1
                true_native = truth_native[local_index]
                pred_native = prediction_native[local_index]
                true_vector = truth[local_index] * target_scale
                pred_vector = prediction[local_index] * target_scale
                true_p, true_pt, true_theta, true_phi = vector_kinematics(true_vector)
                pred_p, pred_pt, pred_theta, pred_phi = vector_kinematics(pred_vector)
                true_eta = kinematic_variables(true_vector)["eta"]
                pred_eta = kinematic_variables(pred_vector)["eta"]
                record = {
                    "real_index": real_index,
                    "n_hits": int(mask[local_index].sum().item()),
                    "pid_class": pid_class,
                    "pid_label": str(config["pid_labels"].get(pid_class, "unknown")),
                    "true_px_gev": true_vector[0], "true_py_gev": true_vector[1], "true_pz_gev": true_vector[2],
                    "true_p_gev": true_p, "true_pt_gev": true_pt, "true_theta_deg": true_theta, "true_phi_deg": true_phi,
                    "true_eta": true_eta,
                    "adapter_px_gev": pred_vector[0], "adapter_py_gev": pred_vector[1], "adapter_pz_gev": pred_vector[2],
                    "adapter_p_gev": pred_p, "adapter_pt_gev": pred_pt, "adapter_theta_deg": pred_theta, "adapter_phi_deg": pred_phi,
                    "adapter_eta": pred_eta,
                }
                if regression_task == "p_phi_eta":
                    record.update({
                        "true_native_p_gev": true_native[0] * target_scale,
                        "true_native_phi_rad": true_native[1],
                        "true_native_phi_deg": np.degrees(true_native[1]),
                        "true_native_eta": true_native[2],
                        "adapter_native_p_gev": pred_native[0] * target_scale,
                        "adapter_native_phi_rad": pred_native[1],
                        "adapter_native_phi_deg": np.degrees(pred_native[1]),
                        "adapter_native_eta": pred_native[2],
                    })
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

    raw_truth = np.asarray([
        [r["true_px_gev"], r["true_py_gev"], r["true_pz_gev"]]
        for r in records
    ])
    charge, charge_summary = resolve_charge(records, config)
    for record, value in zip(records, charge):
        record["comparison_charge"] = int(value)

    raw_adapter = np.asarray([
        [r["adapter_px_gev"], r["adapter_py_gev"], r["adapter_pz_gev"]]
        for r in records
    ])

    if bool(config.get("swingback_enabled", True)):
        truth_doca = swingback_vector_to_doca(raw_truth, charge, config)
        adapter_doca = swingback_vector_to_doca(raw_adapter, charge, config)
    else:
        truth_doca = raw_truth.copy()
        adapter_doca = raw_adapter.copy()
    for record, vector in zip(records, truth_doca):
        attach_vector_kinematics(record, "truth_doca", vector)
    for record, vector in zip(records, adapter_doca):
        attach_vector_kinematics(record, "adapter_doca", vector)

    write_predictions(output_dir / "predictions.csv.gz", records)
    comparison_truth = str(config.get("comparison_truth", "mctrue_swingback_doca"))
    if comparison_truth == "mctrue_swingback_doca":
        truth = truth_doca
        truth_definition = (
            "MC::True at innermost matched CVT hit, transversely swung back to DOCA"
        )
        truth_prefix = "truth_doca"
    elif comparison_truth == "mctrue_inner_hit":
        truth = raw_truth
        truth_definition = "MC::True momentum at innermost matched CVT hit"
        truth_prefix = "true"
    else:
        raise ValueError(
            "comparison_truth must be mctrue_swingback_doca or mctrue_inner_hit; "
            f"got {comparison_truth!r}"
        )
    predictions = {
        "adapter": adapter_doca,
        "cvt": np.asarray([
            [r[f"cvt_{component}_gev"] for component in ("px", "py", "pz")]
            for r in records
        ]),
        "cvtrec": np.asarray([
            [r[f"cvtrec_{component}_gev"] for component in ("px", "py", "pz")]
            for r in records
        ]),
    }
    ml_metric_rows, ml_metric_summary = calculate_ml_metrics(truth, predictions)
    training_log_path = infer_training_log_path(config)
    training_history, training_summary = read_training_history(training_log_path)
    true_p, _, true_theta, _ = vector_kinematics(truth)
    tolerances = [float(value) for value in config["relative_error_tolerances"]]
    summary = {
        "checkpoint": str(config["checkpoint"]),
        "model_config": config["model_config"],
        "training_target_task": regression_task,
        "training_target_columns": trainer.regression_target_stats["columns"],
        "training_target_definition": (
            "Derived (p, phi, eta) from MC::True Cartesian momentum at innermost matched CVT hit"
            if regression_task == "p_phi_eta"
            else "MC::True momentum at innermost matched CVT hit"
        ),
        "comparison_truth": comparison_truth,
        "comparison_truth_definition": truth_definition,
        "adapter_comparison_definition": (
            "Adapter output swung back to DOCA with the same charge and geometry "
            "as comparison truth"
            if bool(config.get("swingback_enabled", True))
            else "Raw adapter output; swingback disabled"
        ),
        "raw_truth_definition": "MC::True momentum at innermost matched CVT hit",
        "swingback": make_swingback_diagnostics(
            raw_truth, truth_doca, charge, config
        ),
        "charge_resolution": charge_summary,
        "reference_hierarchy": {
            "fair_baseline": "CVT::Tracks first fit without PID-dependent energy-loss correction",
            "pid_corrected_reference": "CVTRec::Tracks second fit with PID-dependent energy-loss correction",
            "rec_particle_expectation": "REC::Particle should copy CVTRec::Tracks, subject to the consistency audit below",
            "generator_reference": "MC::Particle is generator momentum before transport energy loss",
        },
        "cvtrec_rec_particle_consistency": audit_cvtrec_rec_particle(records),
        "units": "GeV and degrees",
        "methods": {method: calculate_metrics(truth, prediction, tolerances) for method, prediction in predictions.items()},
        "ml_metrics": ml_metric_summary,
        "training_history": training_summary,
    }
    adapter_resolution = summary["methods"]["adapter"]["momentum"]["relative_resolution_68"]
    summary["adapter_to_cvt_resolution_ratio"] = (
        adapter_resolution / summary["methods"]["cvt"]["momentum"]["relative_resolution_68"]
    )
    with (output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    with (output_dir / "ml_metrics_summary.json").open("w") as stream:
        json.dump(ml_metric_summary, stream, indent=2, allow_nan=False)
    write_csv(output_dir / "ml_metrics.csv", ml_metric_rows)
    write_jsonl(
        output_dir / "campaign_headline_metrics.jsonl",
        build_campaign_headline_rows(config, summary, ml_metric_rows),
    )
    write_training_history(output_dir / "training_history.csv", training_history)

    binned_rows = []
    momentum_bins = np.asarray(config["momentum_bins_gev"], dtype=float)
    theta_bins = np.asarray(config["theta_bins_deg"], dtype=float)
    add_binned_rows(
        binned_rows, truth, predictions, true_p, f"{truth_prefix}_p_gev", momentum_bins
    )
    add_binned_rows(
        binned_rows, truth, predictions, true_theta, f"{truth_prefix}_theta_deg", theta_bins
    )
    pid = np.asarray([r["pid_class"] for r in records])
    pid_labels = {int(key): str(value) for key, value in config["pid_labels"].items()}
    add_binned_rows(binned_rows, truth, predictions, pid, "pid", None, labels=pid_labels)
    add_binned_rows(
        binned_rows, truth, predictions, charge, "comparison_charge", None,
        labels={-1: "negative", 0: "neutral", 1: "positive"},
    )
    write_csv(output_dir / "binned_metrics.csv", binned_rows)
    delta_p_over_p_bins = np.asarray(config["delta_p_over_p_bins_gev"], dtype=float)
    delta_p_over_p_rows = calculate_delta_p_over_p_fit_rows(
        truth, predictions, delta_p_over_p_bins, config
    )
    write_csv(output_dir / "delta_p_over_p_fits.csv", delta_p_over_p_rows)
    make_plots(
        output_dir, truth, predictions, true_p, true_theta,
        momentum_bins, theta_bins, config,
    )
    make_delta_p_over_p_plot(output_dir, delta_p_over_p_rows)
    make_ml_plots(output_dir, truth, predictions, ml_metric_rows, training_history)

    trainer.cleanup()
    print(f"Wrote evaluation results to {output_dir}")


if __name__ == "__main__":
    main()
