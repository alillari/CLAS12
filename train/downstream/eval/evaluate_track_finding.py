#!/usr/bin/env python3
"""Evaluation for the CLAS12 event-level track-finding adapter."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from ruamel.yaml import YAML
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DOWNSTREAM_DIR = HERE.parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(DOWNSTREAM_DIR))
sys.path.insert(0, str(REPO_ROOT))

from fm4npp.utils import YParams  # noqa: E402
from loss import assign_points_to_masks  # noqa: E402
from trackinghead import MambaAttentionHead  # noqa: E402
from track_finding_metrics import (  # noqa: E402
    MatchConfig,
    compare_metric_summaries,
    event_track_metrics,
    finite_or_none,
    summarize_event_metrics,
    track_momentum_by_label,
)
from track_finding_trainer import DownstreamTrainer  # noqa: E402


ENV_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def expand_env_defaults(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            return os.environ.get(name, default or "")
        return ENV_DEFAULT_RE.sub(repl, value)
    if isinstance(value, list):
        return [expand_env_defaults(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env_defaults(val) for key, val in value.items()}
    return value


def format_placeholders(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        previous = None
        current = value
        while previous != current:
            previous = current
            current = current.format(**context)
        return current
    if isinstance(value, list):
        return [format_placeholders(item, context) for item in value]
    if isinstance(value, dict):
        return {key: format_placeholders(val, context) for key, val in value.items()}
    return value


def read_analysis(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        data = YAML(typ="safe").load(stream) or {}
    analysis = expand_env_defaults(data.get("analysis", data))
    context = dict(analysis)
    analysis = format_placeholders(analysis, context)
    return analysis


def json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(json_safe(payload), stream, indent=2, allow_nan=False)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    if not fields:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: json_safe(row.get(key)) for key in fields} for row in rows])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-config", required=True, help="Track-finding analysis YAML.")
    parser.add_argument("--checkpoint", help="Override adapter checkpoint path.")
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument("--max-samples", type=int, help="Override maximum evaluated events.")
    parser.add_argument("--save-per-point-predictions", action="store_true")
    parser.add_argument(
        "--assignment-thresholds",
        type=float,
        nargs="+",
        help=(
            "Run an inference-only assignment-threshold sweep. Requires --output-dir "
            "and must include 0.0 as the reference operating point."
        ),
    )
    parser.add_argument(
        "--calibration-modulus",
        type=int,
        default=10,
        help="Hash partition denominator; one remainder is held for threshold selection.",
    )
    parser.add_argument(
        "--calibration-remainder",
        type=int,
        default=0,
        help="Hash remainder assigned to the calibration partition.",
    )
    parser.add_argument(
        "--min-efficiency-retention",
        type=float,
        default=0.99,
        help="Minimum calibration global-efficiency retention relative to threshold 0.0.",
    )
    return parser.parse_args()


def init_down_model(params, device):
    return MambaAttentionHead(
        input_dim=params.embed_dim,
        embed_dim=params.embed_dim,
        num_layers=int(getattr(params, "num_adapter_layers", 0)),
        num_embedder_layers=int(getattr(params, "num_embedder_layers", 0)),
        d_state=int(getattr(params, "adapter_d_state", getattr(params, "d_state", 64))),
        d_conv=int(getattr(params, "adapter_d_conv", getattr(params, "d_conv", 4))),
        expand=int(getattr(params, "adapter_expand", getattr(params, "expand", 2))),
        num_feature_layers=params.num_layers_backbone,
        num_output_dim=params.embed_dim,
        num_prototypes=int(getattr(params, "num_prototypes", params.max_gt_classes)),
        num_heads=int(getattr(params, "num_heads_decoder", 4)),
        ffn_dim=int(getattr(params, "ffn_dim", 512)),
        num_self_attn_layers=int(getattr(params, "num_self_attn_layers", 2)),
        softmax_mask=bool(getattr(params, "softmax_mask", False)),
        do_masked_attn=bool(getattr(params, "do_masked_attn", True)),
        embed_method=getattr(params, "embed_method", "add"),
        pe_method=getattr(params, "pe_method", "nerf"),
        dropout=float(getattr(params, "downstream_dropout", 0.0)),
    ).to(device)


def load_checkpoint(trainer: DownstreamTrainer, checkpoint_path: Path) -> None:
    trainer.down_model = init_down_model(trainer.params, trainer.device)
    trainer.down_optimizer = torch.optim.AdamW(trainer.down_model.parameters(), lr=trainer.params.max_lr)
    trainer.down_scheduler = None
    trainer.load_checkpoint(str(checkpoint_path), inference=True)
    trainer.down_model.eval()
    trainer.model.eval()


def event_momentum_summary(labels: np.ndarray, reg: np.ndarray | None, scale: float) -> dict[int, dict[str, float]]:
    if reg is None:
        return {}
    return track_momentum_by_label(labels, reg, momentum_scale=scale)


def bin_track_rows(
    track_rows: list[dict[str, Any]],
    bins: list[float],
    variable: str,
) -> list[dict[str, Any]]:
    rows = []
    if not bins:
        return rows
    for low, high in zip(bins[:-1], bins[1:]):
        selected = [
            row for row in track_rows
            if row.get(variable) is not None and low <= float(row[variable]) < high
        ]
        n_true = len(selected)
        matched = [row for row in selected if row.get("matched")]
        rows.append({
            "variable": variable,
            "bin": f"[{low}, {high})",
            "n_true_tracks": n_true,
            "n_matched_tracks": len(matched),
            "track_efficiency": (len(matched) / n_true) if n_true else None,
            "matched_purity_mean": (
                float(np.mean([row["match_purity"] for row in matched if row.get("match_purity") is not None]))
                if matched else None
            ),
            "matched_efficiency_mean": (
                float(np.mean([row["match_efficiency"] for row in matched if row.get("match_efficiency") is not None]))
                if matched else None
            ),
            "event_ari_signal_mean": (
                float(np.mean([row["event_ari_signal"] for row in selected if row.get("event_ari_signal") is not None]))
                if selected else None
            ),
        })
    return rows


def evaluate_event_method(
    method: str,
    event_id: int,
    batch_index: int,
    sample_index: int,
    truth: np.ndarray,
    pred: np.ndarray,
    pred_signal: np.ndarray,
    valid: np.ndarray,
    reg: np.ndarray | None,
    match_config: MatchConfig,
    momentum_scale: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    result = event_track_metrics(
        truth,
        pred,
        valid_mask=valid,
        pred_signal_mask=pred_signal,
        config=match_config,
    )
    event_row = {
        key: value for key, value in result.items() if key != "matches"
    }
    event_row.update({
        "method": method,
        "event_id": event_id,
        "batch_index": batch_index,
        "sample_index": sample_index,
    })
    match_rows = [
        {"method": method, "event_id": event_id, **match}
        for match in result["matches"]
    ]

    momentum = event_momentum_summary(truth, reg, momentum_scale)
    matches_by_truth = {int(match["true_id"]): match for match in result["matches"]}
    truth_track_rows = []
    for true_id, true_info in momentum.items():
        match = matches_by_truth.get(int(true_id))
        truth_track_rows.append({
            "method": method,
            "event_id": event_id,
            "true_id": int(true_id),
            "p_gev": finite_or_none(true_info.get("p_gev")),
            "pt_gev": finite_or_none(true_info.get("pt_gev")),
            "matched": match is not None,
            "match_iou": finite_or_none(match.get("iou") if match else None),
            "match_purity": finite_or_none(match.get("purity") if match else None),
            "match_efficiency": finite_or_none(match.get("efficiency") if match else None),
            "event_ari_signal": finite_or_none(result.get("ari_signal")),
            "event_track_efficiency": finite_or_none(result.get("track_efficiency")),
            "event_track_purity": finite_or_none(result.get("track_purity")),
        })
    return event_row, match_rows, truth_track_rows


EVENT_MEAN_METRICS = (
    "ari_signal",
    "ari_with_background",
    "track_efficiency",
    "track_purity",
    "matched_iou_mean",
    "matched_purity_mean",
    "matched_efficiency_mean",
    "fake_rate",
    "miss_rate",
    "split_rate",
    "merge_rate",
    "background_rejection",
    "background_contamination",
    "signal_loss_to_background",
)
EVENT_COUNT_METRICS = (
    "n_points",
    "n_signal_points",
    "n_background_points",
    "n_true_tracks",
    "n_pred_tracks",
    "n_matched_tracks",
)


class EventMetricAccumulator:
    """Streaming equivalent of ``summarize_event_metrics`` for threshold sweeps."""

    def __init__(self) -> None:
        self.n_events = 0
        self.counts = {key: 0 for key in EVENT_COUNT_METRICS}
        self.sums = {key: 0.0 for key in EVENT_MEAN_METRICS}
        self.observations = {key: 0 for key in EVENT_MEAN_METRICS}

    def add(self, row: dict[str, Any]) -> None:
        self.n_events += 1
        for key in EVENT_COUNT_METRICS:
            self.counts[key] += int(row.get(key, 0))
        for key in EVENT_MEAN_METRICS:
            value = row.get(key)
            if value is not None:
                self.sums[key] += float(value)
                self.observations[key] += 1

    def summary(self) -> dict[str, Any]:
        out = {"n_events": self.n_events, **self.counts}
        for key in EVENT_MEAN_METRICS:
            n = self.observations[key]
            out[key] = finite_or_none(self.sums[key] / n) if n else None
        out["track_efficiency_global"] = (
            self.counts["n_matched_tracks"] / self.counts["n_true_tracks"]
            if self.counts["n_true_tracks"] else None
        )
        out["track_purity_global"] = (
            self.counts["n_matched_tracks"] / self.counts["n_pred_tracks"]
            if self.counts["n_pred_tracks"] else None
        )
        return out


def threshold_partition(event_id: int, modulus: int, calibration_remainder: int) -> str:
    digest = hashlib.blake2b(str(int(event_id)).encode(), digest_size=8).digest()
    bucket = int.from_bytes(digest, byteorder="little") % int(modulus)
    return "calibration" if bucket == int(calibration_remainder) else "heldout"


def normalize_thresholds(values: list[float]) -> list[float]:
    thresholds = sorted({float(value) for value in values})
    if not thresholds or thresholds[0] < 0.0:
        raise ValueError("assignment thresholds must be non-negative")
    if 0.0 not in thresholds:
        raise ValueError("assignment threshold sweep must include 0.0 as the reference point")
    return thresholds


def select_threshold(
    summaries: dict[float, dict[str, Any]],
    min_efficiency_retention: float,
) -> tuple[float, float]:
    baseline = summaries[0.0].get("track_efficiency_global")
    if baseline is None:
        raise ValueError("threshold-zero calibration result has no global track efficiency")
    efficiency_floor = float(baseline) * float(min_efficiency_retention)
    eligible = [
        threshold for threshold, summary in summaries.items()
        if summary.get("ari_with_background") is not None
        and summary.get("track_efficiency_global") is not None
        and float(summary["track_efficiency_global"]) >= efficiency_floor
    ]
    if not eligible:
        raise ValueError("no threshold retained the requested calibration track efficiency")
    selected = max(
        eligible,
        key=lambda threshold: (
            float(summaries[threshold]["ari_with_background"]),
            -float(threshold),
        ),
    )
    return selected, efficiency_floor


def apply_assignment_threshold(inferred: dict[str, torch.Tensor], threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a new threshold to one already-computed assignment result."""
    scores = inferred["scores"]
    keep = scores > float(threshold)
    assignments = torch.where(keep, inferred["assignments"], torch.full_like(inferred["assignments"], -1))
    classes = torch.where(keep, inferred["classes"], torch.zeros_like(inferred["classes"]))
    return assignments, classes


def run_threshold_sweep(
    *,
    args: argparse.Namespace,
    analysis: dict[str, Any],
    trainer: DownstreamTrainer,
    output_dir: Path,
    match_config: MatchConfig,
    assignment_option: int,
    evaluate_coatjava: bool,
) -> None:
    if not args.output_dir:
        raise ValueError("--assignment-thresholds requires an explicit --output-dir")
    if args.save_per_point_predictions:
        raise ValueError("--save-per-point-predictions is not supported for a threshold sweep")
    if args.calibration_modulus < 2:
        raise ValueError("--calibration-modulus must be at least 2")
    if not 0 <= args.calibration_remainder < args.calibration_modulus:
        raise ValueError("--calibration-remainder must be in [0, calibration_modulus)")
    if not 0.0 < args.min_efficiency_retention <= 1.0:
        raise ValueError("--min-efficiency-retention must be in (0, 1]")

    thresholds = normalize_thresholds(args.assignment_thresholds)
    partitions = ("calibration", "heldout")
    adapter_accumulators = {
        threshold: {partition: EventMetricAccumulator() for partition in partitions}
        for threshold in thresholds
    }
    coatjava_accumulators = {
        partition: EventMetricAccumulator() for partition in partitions
    }
    max_samples = int(analysis.get("max_samples", 10000))
    event_offset = 0

    with torch.no_grad():
        for batch in tqdm(trainer.val_data_loader, desc="threshold sweep"):
            if event_offset >= max_samples:
                break
            grouped, labels, _knearest, _reg = trainer._unpack_batch(batch)
            coatjava_batch = batch.get("coatjava_seg_pred") if isinstance(batch, dict) else None
            if evaluate_coatjava and coatjava_batch is None:
                raise RuntimeError(
                    "evaluate_coatjava=true requires a collated coatjava_seg_pred sidecar"
                )
            grouped = grouped.to(trainer.device)
            labels_device = labels.to(trainer.device)
            batch_size, channels = grouped.size(0), grouped.size(-1)
            grouped = grouped.reshape(batch_size, -1, channels)
            valid_mask = grouped[..., 0] != -100
            if bool(analysis.get("use_pretrained_backbone", False)):
                _, pre_embed, _ = trainer.model(grouped, return_z=True)
                feature = torch.stack(pre_embed)
                pred_dict = trainer.down_model(
                    grouped, feature, pretrain=True, padding_mask=valid_mask
                )
            else:
                pred_dict = trainer.down_model(grouped, feature=None, padding_mask=valid_mask)
            outputs = {
                "pred_probs": pred_dict["class_probs"],
                "pred_masks": pred_dict["mask_probs"].permute(0, 2, 1),
            }
            inferred = assign_points_to_masks(outputs, option=assignment_option, threshold=0.0)
            thresholded = {
                threshold: apply_assignment_threshold(inferred, threshold)
                for threshold in thresholds
            }

            for sample_index in range(batch_size):
                event_id = event_offset + sample_index
                if event_id >= max_samples:
                    break
                partition = threshold_partition(
                    event_id,
                    args.calibration_modulus,
                    args.calibration_remainder,
                )
                valid = valid_mask[sample_index].detach().cpu().numpy().astype(bool)
                truth = labels_device[sample_index].detach().cpu().numpy()
                for threshold, (assignments, classes) in thresholded.items():
                    result = event_track_metrics(
                        truth,
                        assignments[sample_index].detach().cpu().numpy(),
                        valid_mask=valid,
                        pred_signal_mask=(classes[sample_index] != 0).detach().cpu().numpy(),
                        config=match_config,
                    )
                    adapter_accumulators[threshold][partition].add(result)
                if evaluate_coatjava:
                    coatjava_pred = coatjava_batch[sample_index].detach().cpu().numpy()
                    coatjava_background = int(analysis.get("coatjava_background_label", -1))
                    coatjava_result = event_track_metrics(
                        truth,
                        coatjava_pred,
                        valid_mask=valid,
                        pred_signal_mask=(coatjava_pred != coatjava_background) & valid,
                        config=match_config,
                    )
                    coatjava_accumulators[partition].add(coatjava_result)
            event_offset += batch_size

    adapter_summaries = {
        threshold: {
            partition: accumulator.summary()
            for partition, accumulator in partition_map.items()
        }
        for threshold, partition_map in adapter_accumulators.items()
    }
    coatjava_summaries = {
        partition: accumulator.summary()
        for partition, accumulator in coatjava_accumulators.items()
    }
    calibration_summaries = {
        threshold: partition_map["calibration"]
        for threshold, partition_map in adapter_summaries.items()
    }
    selected_threshold, efficiency_floor = select_threshold(
        calibration_summaries,
        args.min_efficiency_retention,
    )

    rows = []
    for threshold in thresholds:
        for partition in partitions:
            adapter_summary = adapter_summaries[threshold][partition]
            coatjava_summary = coatjava_summaries.get(partition) if evaluate_coatjava else None
            comparison = (
                compare_metric_summaries(adapter_summary, coatjava_summary)
                if coatjava_summary is not None else {}
            )
            row = {
                "threshold": threshold,
                "partition": partition,
                "selected": threshold == selected_threshold,
                "method": "adapter",
                **adapter_summary,
            }
            row.update({f"adapter_minus_coatjava_{key}": value for key, value in comparison.items()})
            rows.append(row)
            if coatjava_summary is not None:
                rows.append({
                    "threshold": threshold,
                    "partition": partition,
                    "selected": threshold == selected_threshold,
                    "method": "coatjava",
                    **coatjava_summary,
                })

    selected_adapter = adapter_summaries[selected_threshold]["heldout"]
    selected_coatjava = coatjava_summaries.get("heldout") if evaluate_coatjava else None
    selected_comparison = (
        compare_metric_summaries(selected_adapter, selected_coatjava)
        if selected_coatjava is not None else {}
    )
    provenance = {
        "run_name": analysis.get("run_name"),
        "analysis_tag": analysis.get("analysis_tag"),
        "checkpoint": str(Path(analysis["checkpoint"]).resolve()),
        "max_samples": max_samples,
        "assignment_option": assignment_option,
        "thresholds": thresholds,
        "calibration_modulus": args.calibration_modulus,
        "calibration_remainder": args.calibration_remainder,
        "selection_metric": "ari_with_background",
        "min_efficiency_retention": args.min_efficiency_retention,
        "calibration_efficiency_floor": efficiency_floor,
        "selected_threshold": selected_threshold,
        "evaluate_coatjava": evaluate_coatjava,
    }
    write_csv(output_dir / "threshold_metrics.csv", rows)
    write_json(output_dir / "threshold_sweep.json", {
        **provenance,
        "adapter": adapter_summaries,
        "coatjava": coatjava_summaries if evaluate_coatjava else None,
    })
    write_json(output_dir / "selected_heldout_summary.json", {
        **provenance,
        "partition": "heldout",
        "metrics": selected_adapter,
        "baselines": ({"coatjava": selected_coatjava} if selected_coatjava is not None else {}),
        "comparisons": (
            {"adapter_minus_coatjava": selected_comparison}
            if selected_comparison else {}
        ),
    })
    print(f"Wrote threshold-sweep outputs to {output_dir}")


def main() -> None:
    args = parse_args()
    analysis_path = Path(args.analysis_config).resolve()
    analysis = read_analysis(analysis_path)
    if args.checkpoint:
        analysis["checkpoint"] = args.checkpoint
    if args.output_dir:
        analysis["output_dir"] = args.output_dir
    if args.max_samples is not None:
        analysis["max_samples"] = args.max_samples

    params = YParams(os.path.abspath(analysis["model_yaml"]), analysis["model_config"])
    evaluate_coatjava = bool(analysis.get("evaluate_coatjava", True))
    params.limit_data = True
    params.limit_size = int(analysis.get("max_samples", 10000))
    params.limit_test_data = True
    params.limit_test_size = int(analysis.get("max_samples", 10000))
    params.drop_last_test = False
    params.batch_size = int(analysis.get("batch_size", getattr(params, "batch_size", 1)))
    params.valid_batch_size = params.batch_size
    params.local_batch_size = params.batch_size
    params.local_valid_batch_size = params.batch_size
    params.num_data_workers = int(analysis.get("num_workers", getattr(params, "num_data_workers", 0)))
    params.return_dict = True
    params.return_reg_test = True
    params.adapter_sample_mode = "track_legacy"
    params.return_coatjava_seg_pred_test = evaluate_coatjava
    params.require_coatjava_seg_pred_test = evaluate_coatjava
    params.pretrained_ckpt = (
        analysis.get("pretrained_checkpoint")
        if bool(analysis.get("use_pretrained_backbone", False))
        else None
    )

    trainer_args = SimpleNamespace(
        root_dir=str(Path(analysis.get("output_dir", ".")).resolve() / "runtime"),
        global_log_dir="globallogs",
        config=analysis["model_config"],
        run_num=str(analysis.get("run_num", analysis.get("run_name", "eval"))),
    )
    trainer = DownstreamTrainer(params, trainer_args)
    output_dir = Path(analysis["output_dir"]).resolve()
    checkpoint = Path(analysis["checkpoint"]).resolve()
    event_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    per_track_metric_rows: list[dict[str, Any]] = []
    try:
        trainer.launch()
        load_checkpoint(trainer, checkpoint)
        match_config = MatchConfig(
            background_label=int(analysis.get("background_label", getattr(params, "background_label", -1))),
            iou_threshold=float(analysis.get("match_iou_threshold", getattr(params, "match_iou_threshold", 0.5))),
            min_purity=float(analysis.get("match_min_purity", getattr(params, "match_min_purity", 0.5))),
            min_efficiency=float(analysis.get("match_min_efficiency", getattr(params, "match_min_efficiency", 0.5))),
        )
        assignment_option = int(analysis.get("assignment_option", getattr(params, "assignment_option", 2)))
        assignment_threshold = float(analysis.get("assignment_threshold", getattr(params, "assignment_threshold", 0.0)))
        momentum_scale = float(analysis.get("target_momentum_scale_to_gev", 0.001))
        save_points = bool(analysis.get("save_per_point_predictions", False)) or args.save_per_point_predictions

        if args.assignment_thresholds is not None:
            run_threshold_sweep(
                args=args,
                analysis=analysis,
                trainer=trainer,
                output_dir=output_dir,
                match_config=match_config,
                assignment_option=assignment_option,
                evaluate_coatjava=evaluate_coatjava,
            )
            return

        event_offset = 0
        with torch.no_grad():
            for batch_index, batch in enumerate(tqdm(trainer.val_data_loader)):
                if batch_index >= int(analysis.get("max_samples", 10000)):
                    break
                grouped, labels, _knearest, reg = trainer._unpack_batch(batch)
                coatjava_batch = batch.get("coatjava_seg_pred") if isinstance(batch, dict) else None
                if evaluate_coatjava and coatjava_batch is None:
                    raise RuntimeError(
                        "evaluate_coatjava=true requires a collated coatjava_seg_pred sidecar"
                    )
                grouped = grouped.to(trainer.device)
                labels_device = labels.to(trainer.device)
                b, c = grouped.size(0), grouped.size(-1)
                grouped = grouped.reshape(b, -1, c)
                valid_mask = grouped[..., 0] != -100
                if bool(analysis.get("use_pretrained_backbone", False)):
                    _, pre_embed, _ = trainer.model(grouped, return_z=True)
                    feature = torch.stack(pre_embed)
                    pred_dict = trainer.down_model(grouped, feature, pretrain=True, padding_mask=valid_mask)
                else:
                    pred_dict = trainer.down_model(grouped, feature=None, padding_mask=valid_mask)
                outputs = {
                    "pred_probs": pred_dict["class_probs"],
                    "pred_masks": pred_dict["mask_probs"].permute(0, 2, 1),
                }
                inferred = assign_points_to_masks(
                    outputs,
                    option=assignment_option,
                    threshold=assignment_threshold,
                )
                for sample_index in range(b):
                    valid = valid_mask[sample_index].detach().cpu().numpy().astype(bool)
                    truth = labels_device[sample_index].detach().cpu().numpy()
                    pred = inferred["assignments"][sample_index].detach().cpu().numpy()
                    pred_signal = (inferred["classes"][sample_index] != 0).detach().cpu().numpy()
                    reg_np = None
                    if reg is not None:
                        reg_np = reg[sample_index].detach().cpu().numpy()
                    event_id = event_offset + sample_index
                    method_predictions = {"adapter": (pred, pred_signal)}
                    if evaluate_coatjava:
                        coatjava_pred = coatjava_batch[sample_index].detach().cpu().numpy()
                        coatjava_background = int(analysis.get("coatjava_background_label", -1))
                        method_predictions["coatjava"] = (
                            coatjava_pred,
                            (coatjava_pred != coatjava_background) & valid,
                        )

                    for method, (method_pred, method_pred_signal) in method_predictions.items():
                        event_row, method_matches, method_truth_tracks = evaluate_event_method(
                            method=method,
                            event_id=event_id,
                            batch_index=batch_index,
                            sample_index=sample_index,
                            truth=truth,
                            pred=method_pred,
                            pred_signal=method_pred_signal,
                            valid=valid,
                            reg=reg_np,
                            match_config=match_config,
                            momentum_scale=momentum_scale,
                        )
                        event_rows.append(event_row)
                        track_rows.extend(method_matches)
                        per_track_metric_rows.extend(method_truth_tracks)

                    if save_points:
                        coords = grouped[sample_index].detach().cpu().numpy()
                        for method, (method_pred, method_pred_signal) in method_predictions.items():
                            for point_idx in np.where(valid)[0]:
                                point_rows.append({
                                    "method": method,
                                    "event_id": event_id,
                                    "point_idx": int(point_idx),
                                    "truth_label": int(truth[point_idx]),
                                    "pred_label": int(method_pred[point_idx]),
                                    "pred_signal": bool(method_pred_signal[point_idx]),
                                    "eta": float(coords[point_idx, 0]),
                                    "phi": float(coords[point_idx, 1]) if coords.shape[1] > 1 else None,
                                    "r": float(coords[point_idx, 2]) if coords.shape[1] > 2 else None,
                                })
                event_offset += b

        methods = sorted({row["method"] for row in event_rows})
        method_summaries = {
            method: summarize_event_metrics([
                {key: value for key, value in row.items() if key != "method"}
                for row in event_rows if row["method"] == method
            ])
            for method in methods
        }
        global_summary = method_summaries["adapter"]
        coatjava_summary = method_summaries.get("coatjava")
        metric_deltas = (
            compare_metric_summaries(global_summary, coatjava_summary)
            if coatjava_summary is not None else {}
        )
        p_bins = [float(x) for x in analysis.get("momentum_bins_gev", [])]
        pt_bins = [float(x) for x in analysis.get("pt_bins_gev", p_bins)]
        binned_rows = []
        for method in methods:
            method_track_rows = [row for row in per_track_metric_rows if row["method"] == method]
            method_bins = bin_track_rows(method_track_rows, p_bins, "p_gev")
            method_bins.extend(bin_track_rows(method_track_rows, pt_bins, "pt_gev"))
            for row in method_bins:
                row["method"] = method
            binned_rows.extend(method_bins)

        summary = {
            "run_name": analysis.get("run_name"),
            "analysis_tag": analysis.get("analysis_tag"),
            "model_yaml": os.path.abspath(analysis["model_yaml"]),
            "model_config": analysis["model_config"],
            "checkpoint": str(checkpoint),
            "use_pretrained_backbone": bool(analysis.get("use_pretrained_backbone", False)),
            "max_samples": int(analysis.get("max_samples", 10000)),
            "background_label": match_config.background_label,
            "match_iou_threshold": match_config.iou_threshold,
            "match_min_purity": match_config.min_purity,
            "match_min_efficiency": match_config.min_efficiency,
            "metrics": global_summary,
            "baselines": ({"coatjava": coatjava_summary} if coatjava_summary is not None else {}),
            "comparisons": ({"adapter_minus_coatjava": metric_deltas} if metric_deltas else {}),
            "evaluate_coatjava": evaluate_coatjava,
            "coatjava_background_label": int(analysis.get("coatjava_background_label", -1)),
            "momentum_binned_metrics_available": bool(per_track_metric_rows),
        }
        write_json(output_dir / "summary.json", summary)
        write_csv(output_dir / "per_event_metrics.csv", event_rows)
        write_csv(
            output_dir / "per_track_matches.csv",
            track_rows,
            fields=["method", "event_id", "pred_id", "true_id", "iou", "purity", "efficiency"],
        )
        write_csv(
            output_dir / "per_truth_track_metrics.csv",
            per_track_metric_rows,
            fields=[
                "method",
                "event_id",
                "true_id",
                "p_gev",
                "pt_gev",
                "matched",
                "match_iou",
                "match_purity",
                "match_efficiency",
                "event_ari_signal",
                "event_track_efficiency",
                "event_track_purity",
            ],
        )
        write_csv(output_dir / "binned_metrics.csv", binned_rows)
        if save_points:
            write_csv(output_dir / "per_point_predictions.csv", point_rows)

        headline_path = output_dir / "campaign_headline_metrics.jsonl"
        with headline_path.open("w") as stream:
            for method, method_summary in method_summaries.items():
                for key, value in method_summary.items():
                    if isinstance(value, (int, float)) or value is None:
                        stream.write(json.dumps(json_safe({
                            "run_name": analysis.get("run_name"),
                            "record_type": "track_finding_metric",
                            "method": method,
                            "metric": key,
                            "value": value,
                            "use_pretrained_backbone": bool(analysis.get("use_pretrained_backbone", False)),
                        }), allow_nan=False) + "\n")
            for key, value in metric_deltas.items():
                stream.write(json.dumps(json_safe({
                    "run_name": analysis.get("run_name"),
                    "record_type": "track_finding_comparison",
                    "method": "adapter_minus_coatjava",
                    "metric": key,
                    "value": value,
                    "use_pretrained_backbone": bool(analysis.get("use_pretrained_backbone", False)),
                }), allow_nan=False) + "\n")
        print(f"Wrote track-finding evaluation to {output_dir}")
    finally:
        trainer.cleanup()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
