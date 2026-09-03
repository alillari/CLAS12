#!/usr/bin/env python3
"""Evaluation for the CLAS12 event-level track-finding adapter."""

from __future__ import annotations

import argparse
import csv
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
    params.limit_data = True
    params.limit_size = int(analysis.get("max_samples", 10000))
    params.batch_size = int(analysis.get("batch_size", getattr(params, "batch_size", 1)))
    params.valid_batch_size = params.batch_size
    params.local_batch_size = params.batch_size
    params.local_valid_batch_size = params.batch_size
    params.num_data_workers = int(analysis.get("num_workers", getattr(params, "num_data_workers", 0)))
    params.return_dict = True
    params.return_reg_test = True
    params.adapter_sample_mode = "track_legacy"
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

        with torch.no_grad():
            for batch_index, batch in enumerate(tqdm(trainer.val_data_loader)):
                if batch_index >= int(analysis.get("max_samples", 10000)):
                    break
                grouped, labels, _knearest, reg = trainer._unpack_batch(batch)
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
                    event_id = batch_index * b + sample_index
                    row = event_track_metrics(
                        truth,
                        pred,
                        valid_mask=valid,
                        pred_signal_mask=pred_signal,
                        config=match_config,
                    )
                    row.update({"event_id": event_id, "batch_index": batch_index, "sample_index": sample_index})
                    event_rows.append({key: val for key, val in row.items() if key != "matches"})
                    for match in row["matches"]:
                        track_rows.append({"event_id": event_id, **match})

                    momentum = event_momentum_summary(truth, reg_np, momentum_scale)
                    matches_by_truth = {int(match["true_id"]): match for match in row["matches"]}
                    for true_id, true_info in momentum.items():
                        match = matches_by_truth.get(int(true_id))
                        per_track_metric_rows.append({
                            "event_id": event_id,
                            "true_id": int(true_id),
                            "p_gev": finite_or_none(true_info.get("p_gev")),
                            "pt_gev": finite_or_none(true_info.get("pt_gev")),
                            "matched": match is not None,
                            "match_iou": finite_or_none(match.get("iou") if match else None),
                            "match_purity": finite_or_none(match.get("purity") if match else None),
                            "match_efficiency": finite_or_none(match.get("efficiency") if match else None),
                            "event_ari_signal": finite_or_none(row.get("ari_signal")),
                            "event_track_efficiency": finite_or_none(row.get("track_efficiency")),
                            "event_track_purity": finite_or_none(row.get("track_purity")),
                        })

                    if save_points:
                        coords = grouped[sample_index].detach().cpu().numpy()
                        for point_idx in np.where(valid)[0]:
                            point_rows.append({
                                "event_id": event_id,
                                "point_idx": int(point_idx),
                                "truth_label": int(truth[point_idx]),
                                "pred_label": int(pred[point_idx]),
                                "pred_signal": bool(pred_signal[point_idx]),
                                "eta": float(coords[point_idx, 0]),
                                "phi": float(coords[point_idx, 1]) if coords.shape[1] > 1 else None,
                                "r": float(coords[point_idx, 2]) if coords.shape[1] > 2 else None,
                            })

        global_summary = summarize_event_metrics(event_rows)
        p_bins = [float(x) for x in analysis.get("momentum_bins_gev", [])]
        pt_bins = [float(x) for x in analysis.get("pt_bins_gev", p_bins)]
        binned_rows = bin_track_rows(per_track_metric_rows, p_bins, "p_gev")
        binned_rows.extend(bin_track_rows(per_track_metric_rows, pt_bins, "pt_gev"))

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
            "momentum_binned_metrics_available": bool(per_track_metric_rows),
        }
        write_json(output_dir / "summary.json", summary)
        write_csv(output_dir / "per_event_metrics.csv", event_rows)
        write_csv(
            output_dir / "per_track_matches.csv",
            track_rows,
            fields=["event_id", "pred_id", "true_id", "iou", "purity", "efficiency"],
        )
        write_csv(
            output_dir / "per_truth_track_metrics.csv",
            per_track_metric_rows,
            fields=[
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
            for key, value in global_summary.items():
                if isinstance(value, (int, float)) or value is None:
                    stream.write(json.dumps(json_safe({
                        "run_name": analysis.get("run_name"),
                        "record_type": "track_finding_metric",
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
