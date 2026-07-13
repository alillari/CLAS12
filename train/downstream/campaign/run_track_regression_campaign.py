#!/usr/bin/env python3
"""Run a local CLAS12 track-regression adapter campaign."""

from __future__ import annotations

import argparse
from pathlib import Path

from campaign_util import (
    collate_summary,
    command_env,
    eval_command,
    format_command,
    load_status,
    normalize_manifest_paths,
    read_yaml,
    render_analysis_yaml,
    render_model_yaml,
    run_current_status,
    run_logged_command,
    train_command,
    update_status,
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
    return parser.parse_args()


def selected_runs(manifest: dict, only: list[str] | None, limit: int | None) -> list[dict]:
    runs = list(manifest.get("runs", []))
    if only:
        wanted = set(only)
        runs = [run for run in runs if run["run_id"] in wanted]
        missing = wanted - {run["run_id"] for run in runs}
        if missing:
            raise KeyError(f"Run id(s) not found in manifest: {', '.join(sorted(missing))}")
    if limit is not None:
        runs = runs[:limit]
    return runs


def validate_run_inputs(run: dict) -> None:
    checkpoint = Path(run["pretrained_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint does not exist: {checkpoint}")


def ensure_run_dirs(run: dict) -> None:
    for key in ("config_dir", "train_dir", "checkpoint_dir", "evaluation_dir"):
        Path(run[key]).mkdir(parents=True, exist_ok=True)


def print_dry_run(manifest: dict, runs: list[dict]) -> None:
    print(f"Campaign: {manifest['campaign_name']}")
    print(f"Campaign directory: {manifest['campaign_dir']}")
    print(f"Selected runs: {len(runs)}")
    for run in runs:
        validate_run_inputs(run)
        print(f"\n[{run['run_id']}]")
        print(f"model_yaml: {run['model_yaml']}")
        print(f"analysis_yaml: {run['analysis_yaml']}")
        print(f"train_log: {run['train_stdout']}")
        print(format_command(train_command(manifest, run)))
        print(format_command(eval_command(run)))


def train_if_needed(args: argparse.Namespace, manifest: dict, status_path: Path, status_data: dict, run: dict) -> None:
    current_status = run_current_status(status_data, run)
    checkpoint = Path(run["adapter_checkpoint"])
    if (
        not args.force_train
        and current_status in TRAIN_DONE_STATUSES
        and checkpoint.is_file()
    ):
        print(f"[{run['run_id']}] training already complete; skipping")
        return

    print(f"[{run['run_id']}] training")
    update_status(status_path, run["run_id"], "running_train")
    env = command_env(args.cuda_device, manifest)
    code = run_logged_command(train_command(manifest, run), Path(run["train_stdout"]), env)
    if code != 0:
        update_status(
            status_path,
            run["run_id"],
            "failed",
            stage="train",
            returncode=code,
            log=str(Path(run["train_stdout"]).resolve()),
        )
        raise RuntimeError(
            f"Training failed for {run['run_id']} with exit code {code}. "
            f"See {run['train_stdout']}"
        )
    if not checkpoint.is_file():
        update_status(status_path, run["run_id"], "failed", stage="train", reason="missing_adapter_checkpoint")
        raise FileNotFoundError(f"Training finished but adapter checkpoint was not created: {checkpoint}")
    update_status(
        status_path,
        run["run_id"],
        "train_done",
        adapter_checkpoint=str(checkpoint.resolve()),
        train_log=str(Path(run["train_stdout"]).resolve()),
    )


def eval_if_needed(args: argparse.Namespace, manifest: dict, status_path: Path, status_data: dict, run: dict) -> None:
    if args.skip_eval:
        return
    current_status = run_current_status(load_status(status_path), run)
    summary = Path(run["evaluation_dir"]) / "summary.json"
    if (
        not args.force_eval
        and current_status in DONE_STATUSES
        and summary.is_file()
    ):
        print(f"[{run['run_id']}] evaluation already complete; skipping")
        return

    print(f"[{run['run_id']}] evaluating")
    update_status(status_path, run["run_id"], "running_eval")
    env = command_env(args.cuda_device, manifest)
    code = run_logged_command(eval_command(run), Path(run["eval_stdout"]), env)
    if code != 0:
        update_status(
            status_path,
            run["run_id"],
            "failed",
            stage="eval",
            returncode=code,
            log=str(Path(run["eval_stdout"]).resolve()),
        )
        raise RuntimeError(
            f"Evaluation failed for {run['run_id']} with exit code {code}. "
            f"See {run['eval_stdout']}"
        )
    if not summary.is_file():
        update_status(status_path, run["run_id"], "failed", stage="eval", reason="missing_summary_json")
        raise FileNotFoundError(f"Evaluation finished but summary.json was not created: {summary}")
    update_status(
        status_path,
        run["run_id"],
        "eval_done",
        evaluation_dir=str(Path(run["evaluation_dir"]).resolve()),
        eval_log=str(Path(run["eval_stdout"]).resolve()),
    )


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = normalize_manifest_paths(read_yaml(manifest_path))
    status_path = Path(manifest["campaign_dir"]).resolve() / "status.yaml"
    runs = selected_runs(manifest, args.only, args.limit)

    if args.collate_only:
        collate_summary(manifest)
        print(f"Wrote summary files under {Path(manifest['campaign_dir']) / 'summary'}")
        return

    if args.dry_run:
        print_dry_run(manifest, runs)
        return

    status_data = load_status(status_path)
    for run in runs:
        validate_run_inputs(run)
        ensure_run_dirs(run)
        render_model_yaml(manifest, run)
        render_analysis_yaml(manifest, run)
        train_if_needed(args, manifest, status_path, status_data, run)
        status_data = load_status(status_path)
        eval_if_needed(args, manifest, status_path, status_data, run)
        status_data = load_status(status_path)

    if not args.skip_eval:
        collate_summary(manifest)
        print(f"Wrote summary files under {Path(manifest['campaign_dir']) / 'summary'}")


if __name__ == "__main__":
    main()
