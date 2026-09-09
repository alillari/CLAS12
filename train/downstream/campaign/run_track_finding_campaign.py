#!/usr/bin/env python3
"""Run a local CLAS12 track-finding adapter campaign."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from mmap_ninja import RaggedMmap

from campaign_util import (
    command_env,
    format_command,
    load_base_model_config,
    load_status,
    normalize_manifest_paths,
    read_json,
    read_yaml,
    run_current_status,
    run_logged_command,
    update_status,
    write_yaml,
)


DONE_STATUSES = {"eval_done"}
TRAIN_DONE_STATUSES = {"train_done", "running_eval", "eval_done"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to campaign manifest.yaml")
    parser.add_argument(
        "--cuda-device",
        default="0",
        help="CUDA device exposed to each subprocess via CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--only", action="append", help="Run only this run_id. Can be repeated.")
    parser.add_argument("--limit", type=int, help="Maximum number of selected runs to process.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without writing configs or running jobs.")
    parser.add_argument("--force-train", action="store_true", help="Train even if the adapter checkpoint/status already exists.")
    parser.add_argument("--force-eval", action="store_true", help="Evaluate even if evaluation outputs/status already exist.")
    parser.add_argument("--skip-eval", action="store_true", help="Train selected runs but do not evaluate.")
    parser.add_argument("--collate-only", action="store_true", help="Only rebuild campaign summary files from existing evaluations.")
    parser.add_argument("--collate-evaluation-suffix", help="Read each run's sibling evaluation directory with this suffix.")
    parser.add_argument("--summary-name", default="summary", help="Campaign child directory for collated outputs.")
    parser.add_argument("--status", action="store_true", help="Print campaign progress from status.yaml and expected outputs.")
    parser.add_argument("--preflight-only", action="store_true", help="Validate and summarize the event dataset, then exit.")
    return parser.parse_args()


def selected_runs(manifest: dict, only: list[str] | None, limit: int | None) -> list[dict]:
    runs = list(manifest.get("runs", []))
    if only:
        wanted = set(only)
        runs = [run for run in runs if run["run_id"] in wanted]
        missing = wanted - {run["run_id"] for run in runs}
        if missing:
            raise KeyError(f"Run id(s) not found in manifest: {', '.join(sorted(missing))}")
    return runs[:limit] if limit is not None else runs


def validate_run_inputs(run: dict[str, Any]) -> None:
    if not run.get("use_pretrained_backbone", True):
        return
    checkpoint = Path(run["pretrained_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint does not exist: {checkpoint}")


def ensure_run_dirs(run: dict[str, Any]) -> None:
    for key in ("config_dir", "train_dir", "checkpoint_dir", "evaluation_dir"):
        Path(run[key]).mkdir(parents=True, exist_ok=True)


def render_model_yaml(manifest: dict[str, Any], run: dict[str, Any]) -> None:
    params = load_base_model_config(Path(run.get("base_model_yaml", manifest["base_model_yaml"])), run.get("base_model_config", "clas12_track_finding_adapteronly"))
    for key in ("preflight_max_events", "min_multitrack_fraction"):
        params.pop(key, None)
    params.update({
        "artifact_root": str(Path(manifest["artifact_root"]).resolve()),
        "downstream_dir": str(Path(run["run_dir"]).resolve()),
        "checkpoint_dir": str(Path(run["checkpoint_dir"]).resolve()),
        "base_dim": int(run["base_dim"]),
        "embed_dim": int(run["embed_dim"]),
        "num_layers_backbone": int(run["num_layers_backbone"]),
        "model_version": run["model_config"],
        "batch_size": int(run["train_batch_size"]),
        "local_batch_size": int(run["train_batch_size"]),
        "valid_batch_size": int(run["train_batch_size"]),
        "local_valid_batch_size": int(run["train_batch_size"]),
        "limit_size": int(run["eventnumber"]),
    })
    params.update(manifest.get("training_overrides", {}))
    params.update(run.get("training_overrides", {}))
    write_yaml(Path(run["model_yaml"]), {run["model_config"]: params})


def render_analysis_yaml(manifest: dict[str, Any], run: dict[str, Any]) -> None:
    data = read_yaml(Path(manifest["base_analysis_yaml"]))
    analysis = dict(data["analysis"])
    analysis.update({
        "artifact_root": str(Path(manifest["artifact_root"]).resolve()),
        "campaign_name": manifest["campaign_name"],
        "campaign_root": str(Path(manifest["campaign_dir"]).resolve()),
        "run_name": run["run_id"],
        "analysis_tag": f"{manifest['campaign_name']}_{run['run_id']}",
        "model_yaml": str(Path(run["model_yaml"]).resolve()),
        "model_config": run["model_config"],
        "checkpoint": str(Path(run["adapter_checkpoint"]).resolve()),
        "training_log": str(Path(run["training_log"]).resolve()),
        "output_dir": str(Path(run["evaluation_dir"]).resolve()),
        "run_num": run["run_id"],
        "batch_size": int(run["train_batch_size"]),
        "max_samples": int(run["max_samples"]),
        "use_pretrained_backbone": bool(run.get("use_pretrained_backbone", True)),
        "pretrained_checkpoint": (
            str(Path(run["pretrained_checkpoint"]).resolve())
            if run.get("pretrained_checkpoint") else None
        ),
    })
    analysis.update(manifest.get("analysis_overrides", {}))
    analysis.update(run.get("analysis_overrides", {}))
    write_yaml(Path(run["analysis_yaml"]), {"analysis": analysis})


def train_command(run: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    cmd = [
        sys.executable,
        "train/downstream/train_track_finding.py",
        "--yaml_config", str(Path(run["model_yaml"]).resolve()),
        "--config", run["model_config"],
        "--run_num", run["run_id"],
        "--root_dir", str(Path(run["train_dir"]).resolve()),
        "--global_log_dir", str(Path(manifest["campaign_dir"]).resolve() / "logs" / "global"),
        "--eventnumber", str(int(run["eventnumber"])),
        "--train_batch_size", str(int(run["train_batch_size"])),
        "--checkpoint_dir", str(Path(run["checkpoint_dir"]).resolve()),
        "--log_file_name", f"{run['run_id']}.log",
        "--checkpoint_file_name", f"{run['run_id']}_adapter_checkpoint.pth",
        "--artifact_summary", str(Path(run["artifact_summary"]).resolve()),
    ]
    if run.get("use_pretrained_backbone", True):
        cmd.extend(["--usepretrain", "--pretrained_ckpt", str(Path(run["pretrained_checkpoint"]).resolve())])
    return cmd


def eval_command(run: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        "train/downstream/eval/evaluate_track_finding.py",
        "--analysis-config", str(Path(run["analysis_yaml"]).resolve()),
    ]


def preflight_dataset(data_root: Path, max_events: int = 10000, min_multitrack_fraction: float = 0.8) -> dict[str, Any]:
    """Validate aligned event sidecars for both supervised train and test splits."""
    split_rows: dict[str, dict[str, Any]] = {}
    for split in ("pretrain", "test"):
        features = RaggedMmap(str(data_root / f"features_{split}"))
        seg = RaggedMmap(str(data_root / f"seg_target_{split}"))
        coatjava = RaggedMmap(str(data_root / f"coatjava_seg_pred_{split}"))
        n = min(len(features), len(seg), len(coatjava), int(max_events))
        rows = []
        for idx in range(n):
            labels = np.asarray(seg[idx])
            coatjava_labels = np.asarray(coatjava[idx])
            signal = labels[labels != -1]
            rows.append({
                "n_points": int(len(labels)),
                "n_signal_tracks": int(len(np.unique(signal))) if signal.size else 0,
                "background_fraction": float(np.mean(labels == -1)) if labels.size else 0.0,
                "length_match": int(features[idx].shape[0]) == int(labels.shape[0]),
                "coatjava_length_match": int(features[idx].shape[0]) == int(coatjava_labels.shape[0]),
            })
        fraction = float(np.mean([row["n_signal_tracks"] > 1 for row in rows])) if rows else 0.0
        split_rows[split] = {
            "feature_events": len(features), "seg_target_events": len(seg),
            "coatjava_seg_pred_events": len(coatjava), "sampled_events": n,
            "multitrack_fraction": fraction,
            "mean_points": float(np.mean([row["n_points"] for row in rows])) if rows else None,
            "mean_signal_tracks": float(np.mean([row["n_signal_tracks"] for row in rows])) if rows else None,
            "mean_background_fraction": float(np.mean([row["background_fraction"] for row in rows])) if rows else None,
            "all_lengths_match": all(row["length_match"] for row in rows),
            "all_coatjava_lengths_match": all(row["coatjava_length_match"] for row in rows),
            "passed": bool(rows) and len(features) == len(seg) == len(coatjava)
                and all(row["length_match"] and row["coatjava_length_match"] for row in rows)
                and fraction >= min_multitrack_fraction,
        }
    train = split_rows["pretrain"]
    summary = {
        "data_root": str(data_root),
        "feature_events": train["feature_events"], "seg_target_events": train["seg_target_events"],
        "coatjava_seg_pred_events": train["coatjava_seg_pred_events"], "sampled_events": train["sampled_events"],
        "multitrack_fraction": train["multitrack_fraction"], "mean_points": train["mean_points"],
        "mean_signal_tracks": train["mean_signal_tracks"], "mean_background_fraction": train["mean_background_fraction"],
        "all_lengths_match": train["all_lengths_match"], "all_coatjava_lengths_match": train["all_coatjava_lengths_match"],
        "test_feature_events": split_rows["test"]["feature_events"],
        "test_sampled_events": split_rows["test"]["sampled_events"],
        "test_multitrack_fraction": split_rows["test"]["multitrack_fraction"],
        "test_passed": split_rows["test"]["passed"],
    }
    summary["passed"] = train["passed"] and split_rows["test"]["passed"]
    if not summary["passed"]:
        raise ValueError(f"Track-finding preflight failed: {summary}")
    return summary


def collate_summary(
    manifest: dict[str, Any],
    evaluation_suffix: str | None = None,
    summary_name: str = "summary",
) -> None:
    base_dir = Path(manifest["campaign_dir"]).resolve()
    summary_dir = base_dir / str(summary_name)
    summary_dir.mkdir(parents=True, exist_ok=True)
    table_rows = []
    with (summary_dir / "campaign_headline_metrics.jsonl").open("w") as headline_stream:
        for run in manifest.get("runs", []):
            evaluation_dir = Path(run["evaluation_dir"])
            if evaluation_suffix:
                evaluation_dir = evaluation_dir.with_name(
                    f"{evaluation_dir.name}_{evaluation_suffix}"
                )
            headline = evaluation_dir / "campaign_headline_metrics.jsonl"
            summary = evaluation_dir / "summary.json"
            if headline.exists():
                with headline.open() as stream:
                    shutil.copyfileobj(stream, headline_stream)
            table_row = {
                "run_id": run["run_id"],
                "backbone_run_id": run.get("backbone_run_id"),
                "use_pretrained_backbone": run.get("use_pretrained_backbone", True),
                "embed_dim": run["embed_dim"],
                "num_layers_backbone": run["num_layers_backbone"],
                "pretrain_events": run["pretrain_events"],
                "labeled_events": run.get("labeled_events", run.get("eventnumber")),
                "adapter_checkpoint": run["adapter_checkpoint"],
                "model_yaml": run["model_yaml"],
                "model_config": run["model_config"],
                "evaluation_dir": run["evaluation_dir"],
                "summary_found": summary.exists(),
            }
            if summary.exists():
                summary_data = read_json(summary)
                metrics = summary_data.get("metrics", {})
                coatjava_metrics = summary_data.get("baselines", {}).get("coatjava", {})
                deltas = summary_data.get("comparisons", {}).get("adapter_minus_coatjava", {})
                native_metrics = summary_data.get("native_metrics", {})
                native_coatjava_metrics = summary_data.get("native_baselines", {}).get("coatjava", {})
                native_deltas = summary_data.get("native_comparisons", {}).get("adapter_minus_coatjava", {})
                table_row["metric_view"] = summary_data.get("metric_view", "canonical")
                table_row["noise_attribution_mode"] = summary_data.get("noise_attribution_mode")
                table_row["assignment_threshold"] = summary_data.get("assignment_threshold")
                table_row["track_target_mode"] = summary_data.get("track_target_mode")
                for key in (
                    "ari_signal", "ari_with_background", "track_efficiency_global",
                    "track_purity_global", "fake_rate", "miss_rate", "split_rate",
                    "merge_rate", "background_rejection", "background_contamination",
                    "signal_loss_to_background", "matched_iou_mean",
                ):
                    table_row[key] = metrics.get(key)
                    table_row[f"coatjava_{key}"] = coatjava_metrics.get(key)
                    table_row[f"adapter_minus_coatjava_{key}"] = deltas.get(key)
                    table_row[f"native_{key}"] = native_metrics.get(key)
                    table_row[f"native_coatjava_{key}"] = native_coatjava_metrics.get(key)
                    table_row[f"native_adapter_minus_coatjava_{key}"] = native_deltas.get(key)
            table_rows.append(table_row)
    fields = sorted({key for row in table_rows for key in row})
    with (summary_dir / "run_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table_rows)


def last_validation_metrics(training_log: Path) -> dict[str, str]:
    """Return the newest validation row from the trainer's tab-separated log."""
    if not training_log.is_file():
        return {}
    try:
        with training_log.open(newline="") as stream:
            rows = csv.DictReader(stream, delimiter="\t")
            latest = None
            for row in rows:
                if row.get("Step"):
                    latest = row
    except (OSError, csv.Error):
        return {}
    if latest:
        selected_ari_name, selected_ari = next(
            (
                (key, value) for key, value in latest.items()
                if key.startswith("ARI_") and key.endswith("_option2")
            ),
            ("ARI_2", latest.get("ARI_2", latest.get("ARI"))),
        )
        if selected_ari is not None:
            latest["selected_ari"] = selected_ari
            latest["selected_ari_metric"] = selected_ari_name
    return latest or {}


def format_metric(value: str | None) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except ValueError:
        return value


def print_dry_run(manifest: dict[str, Any], runs: list[dict[str, Any]]) -> None:
    print(f"Campaign: {manifest['campaign_name']}")
    print(f"Campaign directory: {manifest['campaign_dir']}")
    print(f"Selected runs: {len(runs)}")
    for run in runs:
        validate_run_inputs(run)
        print(f"\n[{run['run_id']}]")
        print(f"model_yaml: {run['model_yaml']}")
        print(f"analysis_yaml: {run['analysis_yaml']}")
        print(f"train_log: {run['train_stdout']}")
        print(f"metrics_log: {run['training_log']}")
        print(f"eval_log: {run['eval_stdout']}")
        print(format_command(train_command(run, manifest)))
        print(format_command(eval_command(run)))


def print_status(manifest: dict, runs: list[dict], status_path: Path) -> None:
    status_data = load_status(status_path)
    rows = []
    counts = Counter()
    for run in runs:
        status = run_current_status(status_data, run)
        adapter_checkpoint = Path(run["adapter_checkpoint"])
        summary = Path(run["evaluation_dir"]) / "summary.json"
        train_log = Path(run["train_stdout"])
        metrics_log = Path(run["training_log"])
        eval_log = Path(run["eval_stdout"])
        progress = last_validation_metrics(metrics_log)
        counts[status] += 1
        rows.append({
            "run_id": run["run_id"],
            "status": status,
            "labeled_events": run.get("labeled_events", run.get("eventnumber")),
            "step": progress.get("Step", "-"),
            "val_loss": format_metric(progress.get("Val_Loss")),
            "ari": format_metric(progress.get("selected_ari", progress.get("ARI"))),
            "selected_ari_metric": progress.get("selected_ari_metric"),
            "adapter": "yes" if adapter_checkpoint.is_file() else "no",
            "eval": "yes" if summary.is_file() else "no",
            "log": str(eval_log if status == "running_eval" else train_log),
            "metrics_log": str(metrics_log),
        })

    print(f"Campaign: {manifest['campaign_name']}")
    print(f"Campaign directory: {manifest['campaign_dir']}")
    print(f"Status file: {status_path}")
    print(f"Runs: {len(rows)}")
    if counts:
        print("Counts: " + ", ".join(f"{key}={counts[key]}" for key in sorted(counts)))
    selector_names = sorted({
        row.get("selected_ari_metric")
        for row in rows if row.get("selected_ari_metric")
    })
    if selector_names:
        print("Validation selector: " + ", ".join(selector_names))
    print()
    header = ("status", "labeled", "step", "val_loss", "ari", "adapter", "eval", "run_id")
    print(
        f"{header[0]:<15} {header[1]:>8} {header[2]:>8} {header[3]:>9} "
        f"{header[4]:>7} {header[5]:>7} {header[6]:>5} {header[7]}"
    )
    for row in rows:
        print(
            f"{row['status']:<15} {str(row['labeled_events']):>8} "
            f"{row['step']:>8} {row['val_loss']:>9} {row['ari']:>7} "
            f"{row['adapter']:>7} {row['eval']:>5} {row['run_id']}"
        )
    active = [row for row in rows if row["status"] in {"running_train", "running_eval"}]
    if active:
        print("\nActive logs:")
        for row in active:
            print(f"{row['run_id']}: {row['log']}")
            if row["status"] == "running_train":
                print(f"  metrics: {row['metrics_log']}")
        print("Progress values are the latest completed validation point.")


def train_if_needed(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    status_path: Path,
    status_data: dict[str, Any],
    run: dict[str, Any],
) -> None:
    current_status = run_current_status(status_data, run)
    checkpoint = Path(run["adapter_checkpoint"])
    status_record = status_data.get("runs", {}).get(run["run_id"], {})
    if not args.force_train and current_status in TRAIN_DONE_STATUSES and checkpoint.is_file():
        print(f"[{run['run_id']}] training already complete; skipping")
        return
    if (
        not args.force_train
        and current_status == "failed"
        and status_record.get("stage") == "eval"
        and checkpoint.is_file()
    ):
        print(f"[{run['run_id']}] prior evaluation failed; preserving existing training checkpoint")
        return

    command = train_command(run, manifest)
    log_path = Path(run["train_stdout"])
    print(f"[{run['run_id']}] training")
    print(f"  log: {log_path}")
    print(f"  command: {format_command(command)}")
    update_status(status_path, run["run_id"], "running_train", log=str(log_path.resolve()))
    code = run_logged_command(command, log_path, command_env(args.cuda_device, manifest))
    if code != 0:
        update_status(
            status_path,
            run["run_id"],
            "failed",
            stage="train",
            returncode=code,
            log=str(log_path.resolve()),
        )
        raise RuntimeError(
            f"Training failed for {run['run_id']} with exit code {code}. See {log_path}"
        )
    if not checkpoint.is_file():
        update_status(
            status_path,
            run["run_id"],
            "failed",
            stage="train",
            reason="missing_adapter_checkpoint",
            log=str(log_path.resolve()),
        )
        raise FileNotFoundError(f"Training finished but adapter checkpoint was not created: {checkpoint}")
    update_status(
        status_path,
        run["run_id"],
        "train_done",
        adapter_checkpoint=str(checkpoint.resolve()),
        train_log=str(log_path.resolve()),
    )
    print(f"[{run['run_id']}] training complete")
    print(f"  checkpoint: {checkpoint}")


def eval_if_needed(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    status_path: Path,
    run: dict[str, Any],
) -> None:
    if args.skip_eval:
        return
    current_status = run_current_status(load_status(status_path), run)
    summary = Path(run["evaluation_dir"]) / "summary.json"
    if not args.force_eval and current_status in DONE_STATUSES and summary.is_file():
        print(f"[{run['run_id']}] evaluation already complete; skipping")
        return

    command = eval_command(run)
    log_path = Path(run["eval_stdout"])
    print(f"[{run['run_id']}] evaluating")
    print(f"  log: {log_path}")
    print(f"  command: {format_command(command)}")
    update_status(status_path, run["run_id"], "running_eval", log=str(log_path.resolve()))
    code = run_logged_command(command, log_path, command_env(args.cuda_device, manifest))
    if code != 0:
        update_status(
            status_path,
            run["run_id"],
            "failed",
            stage="eval",
            returncode=code,
            log=str(log_path.resolve()),
        )
        raise RuntimeError(
            f"Evaluation failed for {run['run_id']} with exit code {code}. See {log_path}"
        )
    if not summary.is_file():
        update_status(
            status_path,
            run["run_id"],
            "failed",
            stage="eval",
            reason="missing_summary_json",
            log=str(log_path.resolve()),
        )
        raise FileNotFoundError(f"Evaluation finished but summary.json was not created: {summary}")
    update_status(
        status_path,
        run["run_id"],
        "eval_done",
        evaluation_dir=str(Path(run["evaluation_dir"]).resolve()),
        eval_log=str(log_path.resolve()),
    )
    print(f"[{run['run_id']}] evaluation complete")
    print(f"  summary: {summary}")


def main() -> None:
    args = parse_args()
    manifest = normalize_manifest_paths(read_yaml(Path(args.manifest).resolve()))
    runs = selected_runs(manifest, args.only, args.limit)
    status_path = Path(manifest["campaign_dir"]).resolve() / "status.yaml"

    if args.collate_only:
        collate_summary(
            manifest,
            evaluation_suffix=args.collate_evaluation_suffix,
            summary_name=args.summary_name,
        )
        print(f"Wrote summary files under {Path(manifest['campaign_dir']) / args.summary_name}")
        return
    if args.status:
        print_status(manifest, runs, status_path)
        return

    if args.dry_run:
        print_dry_run(manifest, runs)
        return

    first_params = load_base_model_config(Path(manifest["base_model_yaml"]), "clas12_track_finding_adapteronly")
    first_params.update(manifest.get("training_overrides", {}))
    if runs:
        first_params.update(runs[0].get("training_overrides", {}))
    preflight = preflight_dataset(
        Path(first_params["data_root"]).resolve(),
        max_events=int(first_params.get("preflight_max_events", 10000)),
        min_multitrack_fraction=float(first_params.get("min_multitrack_fraction", 0.8)),
    )
    summary_dir = Path(manifest["campaign_dir"]).resolve() / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "data_preflight.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(preflight))
        writer.writeheader()
        writer.writerow(preflight)
    if args.preflight_only:
        print("Dataset preflight passed:")
        for key in sorted(preflight):
            print(f"  {key}: {preflight[key]}")
        return

    status_data = load_status(status_path)
    for run in runs:
        validate_run_inputs(run)
        ensure_run_dirs(run)
        render_model_yaml(manifest, run)
        render_analysis_yaml(manifest, run)
        train_if_needed(args, manifest, status_path, status_data, run)
        status_data = load_status(status_path)
        eval_if_needed(args, manifest, status_path, run)
        status_data = load_status(status_path)

    if not args.skip_eval:
        collate_summary(manifest)
        print(f"Wrote summary files under {Path(manifest['campaign_dir']) / 'summary'}")


if __name__ == "__main__":
    main()
