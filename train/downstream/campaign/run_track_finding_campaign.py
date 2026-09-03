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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--only", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--collate-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
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


def render_model_yaml(manifest: dict[str, Any], run: dict[str, Any]) -> None:
    params = load_base_model_config(Path(run.get("base_model_yaml", manifest["base_model_yaml"])), run.get("base_model_config", "clas12_track_finding_adapteronly"))
    params.update({
        "artifact_root": str(Path(manifest["artifact_root"]).resolve()),
        "downstream_dir": str(Path(run["run_dir"]).resolve()),
        "checkpoint_dir": str(Path(run["checkpoint_dir"]).resolve()),
        "base_dim": int(run["base_dim"]),
        "embed_dim": int(run["embed_dim"]),
        "num_layers_backbone": int(run["num_layers_backbone"]),
        "model_version": run["model_config"],
    })
    params.update(manifest.get("training_overrides", {}))
    params.update(run.get("training_overrides", {}))
    write_yaml(Path(run["model_yaml"]), {run["model_config"]: params})


def render_analysis_yaml(manifest: dict[str, Any], run: dict[str, Any]) -> None:
    data = read_yaml(Path(manifest["base_analysis_yaml"]))
    analysis = dict(data["analysis"])
    analysis.update({
        "artifact_root": str(Path(manifest["artifact_root"]).resolve()),
        "run_name": run["run_id"],
        "analysis_tag": f"{manifest['campaign_name']}_{run['run_id']}",
        "model_yaml": str(Path(run["model_yaml"]).resolve()),
        "model_config": run["model_config"],
        "checkpoint": str(Path(run["adapter_checkpoint"]).resolve()),
        "training_log": str(Path(run["training_log"]).resolve()),
        "output_dir": str(Path(run["evaluation_dir"]).resolve()),
        "run_num": run["run_id"],
        "max_samples": int(run["max_samples"]),
        "use_pretrained_backbone": bool(run.get("use_pretrained_backbone", True)),
        "pretrained_checkpoint": (
            str(Path(run["pretrained_checkpoint"]).resolve())
            if run.get("pretrained_checkpoint") else None
        ),
    })
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
    features = RaggedMmap(str(data_root / "features_pretrain"))
    seg = RaggedMmap(str(data_root / "seg_target_pretrain"))
    n = min(len(features), len(seg), int(max_events))
    rows = []
    for idx in range(n):
        labels = np.asarray(seg[idx])
        signal = labels[labels != -1]
        rows.append({
            "n_points": int(len(labels)),
            "n_signal_tracks": int(len(np.unique(signal))) if signal.size else 0,
            "background_fraction": float(np.mean(labels == -1)) if labels.size else 0.0,
            "length_match": int(features[idx].shape[0]) == int(labels.shape[0]),
        })
    multitrack_fraction = float(np.mean([row["n_signal_tracks"] > 1 for row in rows])) if rows else 0.0
    summary = {
        "data_root": str(data_root),
        "sampled_events": n,
        "multitrack_fraction": multitrack_fraction,
        "mean_points": float(np.mean([row["n_points"] for row in rows])) if rows else None,
        "mean_signal_tracks": float(np.mean([row["n_signal_tracks"] for row in rows])) if rows else None,
        "mean_background_fraction": float(np.mean([row["background_fraction"] for row in rows])) if rows else None,
        "all_lengths_match": all(row["length_match"] for row in rows),
        "passed": bool(rows) and all(row["length_match"] for row in rows) and multitrack_fraction >= min_multitrack_fraction,
    }
    if not summary["passed"]:
        raise ValueError(f"Track-finding preflight failed: {summary}")
    return summary


def collate_summary(manifest: dict[str, Any]) -> None:
    base_dir = Path(manifest["campaign_dir"]).resolve()
    summary_dir = base_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    table_rows = []
    with (summary_dir / "campaign_headline_metrics.jsonl").open("w") as headline_stream:
        for run in manifest.get("runs", []):
            evaluation_dir = Path(run["evaluation_dir"])
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
                "evaluation_dir": run["evaluation_dir"],
                "summary_found": summary.exists(),
            }
            if summary.exists():
                metrics = read_json(summary).get("metrics", {})
                for key in ("ari_signal", "track_efficiency_global", "track_purity_global", "fake_rate", "background_rejection"):
                    table_row[key] = metrics.get(key)
            table_rows.append(table_row)
    fields = sorted({key for row in table_rows for key in row})
    with (summary_dir / "run_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table_rows)


def print_status(manifest: dict, runs: list[dict], status_path: Path) -> None:
    status_data = load_status(status_path)
    counts = Counter(run_current_status(status_data, run) for run in runs)
    print(f"Campaign: {manifest['campaign_name']}")
    print(f"Campaign directory: {manifest['campaign_dir']}")
    print("Counts: " + ", ".join(f"{key}={counts[key]}" for key in sorted(counts)))


def main() -> None:
    args = parse_args()
    manifest = normalize_manifest_paths(read_yaml(Path(args.manifest).resolve()))
    runs = selected_runs(manifest, args.only, args.limit)
    status_path = Path(manifest["campaign_dir"]).resolve() / "status.yaml"

    if args.collate_only:
        collate_summary(manifest)
        return
    if args.status:
        print_status(manifest, runs, status_path)
        return

    first_params = load_base_model_config(Path(manifest["base_model_yaml"]), "clas12_track_finding_adapteronly")
    preflight = preflight_dataset(Path(first_params["data_root"]).resolve())
    summary_dir = Path(manifest["campaign_dir"]).resolve() / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "data_preflight.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(preflight))
        writer.writeheader()
        writer.writerow(preflight)
    if args.preflight_only:
        print(preflight)
        return

    if args.dry_run:
        for run in runs:
            render_model_yaml(manifest, run)
            render_analysis_yaml(manifest, run)
            print(f"\n[{run['run_id']}]")
            print(format_command(train_command(run, manifest)))
            print(format_command(eval_command(run)))
        return

    status_data = load_status(status_path)
    for run in runs:
        for key in ("config_dir", "train_dir", "checkpoint_dir", "evaluation_dir"):
            Path(run[key]).mkdir(parents=True, exist_ok=True)
        render_model_yaml(manifest, run)
        render_analysis_yaml(manifest, run)
        checkpoint = Path(run["adapter_checkpoint"])
        if not args.force_train and run_current_status(status_data, run) in TRAIN_DONE_STATUSES and checkpoint.exists():
            print(f"[{run['run_id']}] training already complete; skipping")
        else:
            update_status(status_path, run["run_id"], "running_train")
            code = run_logged_command(train_command(run, manifest), Path(run["train_stdout"]), command_env(args.cuda_device, manifest))
            if code != 0:
                update_status(status_path, run["run_id"], "failed", stage="train", returncode=code)
                raise RuntimeError(f"Training failed for {run['run_id']}; see {run['train_stdout']}")
            if not checkpoint.exists():
                update_status(status_path, run["run_id"], "failed", stage="train", reason="missing_adapter_checkpoint")
                raise FileNotFoundError(f"Training finished without checkpoint: {checkpoint}")
            update_status(status_path, run["run_id"], "train_done")
        status_data = load_status(status_path)
        if args.skip_eval:
            continue
        summary = Path(run["evaluation_dir"]) / "summary.json"
        if not args.force_eval and run_current_status(status_data, run) in DONE_STATUSES and summary.exists():
            print(f"[{run['run_id']}] evaluation already complete; skipping")
            continue
        update_status(status_path, run["run_id"], "running_eval")
        code = run_logged_command(eval_command(run), Path(run["eval_stdout"]), command_env(args.cuda_device, manifest))
        if code != 0:
            update_status(status_path, run["run_id"], "failed", stage="eval", returncode=code)
            raise RuntimeError(f"Evaluation failed for {run['run_id']}; see {run['eval_stdout']}")
        update_status(status_path, run["run_id"], "eval_done")
        status_data = load_status(status_path)

    if not args.skip_eval:
        collate_summary(manifest)


if __name__ == "__main__":
    main()
