#!/usr/bin/env python3
"""Shared helpers for local track-regression campaigns."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_ROOT = Path("/home/alessio/ML-work/pretrained-FMs/campaign_1")
DEFAULT_ARTIFACT_ROOT = Path("/home/alessio/ML-work/result_deep_storage")
DEFAULT_CAMPAIGN_NAME = "campaign_1_track_regression"
DEFAULT_BASE_MODEL_YAML = Path("scripts/configs/mamba_clas12_track_regression_pretrained.yaml")
DEFAULT_BASE_ANALYSIS_YAML = Path("train/downstream/eval/track_regression_analysis_adapteronly.yaml")
DEFAULT_EVENTNUMBER = 50000
DEFAULT_TRAIN_BATCH_SIZE = 32
DEFAULT_MAX_SAMPLES = 10000
DEFAULT_MODEL_FAMILY = "mamba1"

RUN_NAME_RE = re.compile(
    r"(?:^|_)w(?P<width>\d+)(?=_|$).*?"
    r"(?:^|_)d(?P<depth>\d+)(?=_|$).*?"
    r"(?:^|_)n(?P<events>\d+)(?=_|$)"
)
CHECKPOINT_SUFFIXES = {
    ".ckpt",
    ".pth",
    ".pt",
    ".tar",
    ".bin",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def yaml_parser(typ: str = "safe") -> YAML:
    yaml = YAML(typ=typ)
    yaml.default_flow_style = False
    yaml.width = 4096
    return yaml


def read_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        data = yaml_parser("safe").load(stream)
    return data or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    yaml = yaml_parser()
    with tmp.open("w") as stream:
        yaml.dump(data, stream)
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(data, stream, indent=2)
    tmp.replace(path)


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def campaign_dir(artifact_root: str | Path, campaign_name: str) -> Path:
    return Path(artifact_root).resolve() / "campaigns" / campaign_name


def parse_run_name(run_id: str) -> dict[str, int]:
    match = RUN_NAME_RE.search(run_id)
    if not match:
        raise ValueError(
            f"Cannot parse run metadata from {run_id!r}. Expected tokens like "
            "'w1536', 'd12', and 'n5483352'."
        )
    return {
        "base_dim": int(match.group("width")),
        "embed_dim": int(match.group("width")),
        "num_layers_backbone": int(match.group("depth")),
        "pretrain_events": int(match.group("events")),
    }


def load_sidecar_metadata(source_dir: Path) -> dict[str, Any]:
    yaml_path = source_dir / "metadata.yaml"
    json_path = source_dir / "metadata.json"
    if yaml_path.exists():
        return read_yaml(yaml_path)
    if json_path.exists():
        return read_json(json_path)
    return {}


def discover_checkpoint(source_dir: Path) -> Path:
    candidates = [
        path for path in source_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
    ]
    checkpoint_candidates = [
        path for path in candidates
        if path.suffix.lower() in CHECKPOINT_SUFFIXES
    ]
    if len(checkpoint_candidates) == 1:
        return checkpoint_candidates[0].resolve()
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(f"No checkpoint file found in {source_dir}")
    names = ", ".join(sorted(path.name for path in candidates))
    raise ValueError(
        f"Expected exactly one checkpoint-like file in {source_dir}; found: {names}"
    )


def run_paths(base_dir: Path, run_id: str) -> dict[str, Path]:
    run_dir = base_dir / "runs" / run_id
    return {
        "run_dir": run_dir,
        "config_dir": run_dir / "config",
        "train_dir": run_dir / "train",
        "checkpoint_dir": run_dir / "checkpoints",
        "evaluation_dir": run_dir / "evaluation",
        "model_yaml": run_dir / "config" / "model.yaml",
        "analysis_yaml": run_dir / "config" / "analysis.yaml",
        "artifact_summary": run_dir / "train" / "artifacts.json",
        "adapter_checkpoint": run_dir / "checkpoints" / f"{run_id}_adapter_checkpoint.pth",
        "training_log": run_dir / "checkpoints" / f"{run_id}.log",
        "train_stdout": base_dir / "logs" / f"{run_id}.train.stdout.log",
        "eval_stdout": base_dir / "logs" / f"{run_id}.eval.stdout.log",
    }


def normalize_manifest_paths(manifest: dict[str, Any]) -> dict[str, Any]:
    base_dir = Path(manifest["campaign_dir"]).resolve()
    for run in manifest.get("runs", []):
        paths = run_paths(base_dir, run["run_id"])
        for key, value in paths.items():
            run.setdefault(key, str(value))
    return manifest


def build_run_row(
    source_dir: Path,
    checkpoint: Path,
    base_dir: Path,
    eventnumber: int,
    train_batch_size: int,
    max_samples: int,
) -> dict[str, Any]:
    run_id = source_dir.name
    metadata = parse_run_name(run_id)
    sidecar = load_sidecar_metadata(source_dir)
    sidecar_checkpoint = sidecar.get("checkpoint", checkpoint)
    sidecar_checkpoint = Path(sidecar_checkpoint)
    if not sidecar_checkpoint.is_absolute():
        sidecar_checkpoint = source_dir / sidecar_checkpoint
    row = {
        "run_id": sidecar.get("run_id", run_id),
        "source_dir": str(source_dir.resolve()),
        "pretrained_checkpoint": str(sidecar_checkpoint.resolve()),
        "model_family": sidecar.get("model_family", DEFAULT_MODEL_FAMILY),
        "base_dim": int(sidecar.get("base_dim", metadata["base_dim"])),
        "embed_dim": int(sidecar.get("embed_dim", metadata["embed_dim"])),
        "num_layers_backbone": int(sidecar.get("num_layers_backbone", metadata["num_layers_backbone"])),
        "pretrain_events": int(sidecar.get("pretrain_events", metadata["pretrain_events"])),
        "eventnumber": int(sidecar.get("eventnumber", eventnumber)),
        "train_batch_size": int(sidecar.get("train_batch_size", train_batch_size)),
        "max_samples": int(sidecar.get("max_samples", max_samples)),
        "status": "pending",
    }
    row["model_config"] = sidecar.get(
        "model_config",
        f"clas12_track_regression_pretrained_{row['run_id']}",
    )
    row.update({key: str(value) for key, value in run_paths(base_dir, row["run_id"]).items()})
    return row


def load_base_model_config(path: Path, config_name: str = "clas12_track_regression_pretrained") -> dict[str, Any]:
    data = read_yaml(path)
    if config_name in data:
        return dict(data[config_name])
    if "default" in data:
        return dict(data["default"])
    raise KeyError(f"{path} must contain {config_name!r} or 'default'")


def render_model_yaml(manifest: dict[str, Any], run: dict[str, Any]) -> None:
    base_model_yaml = resolve_repo_path(manifest["base_model_yaml"])
    params = load_base_model_config(base_model_yaml)
    params.update({
        "artifact_root": str(Path(manifest["artifact_root"]).resolve()),
        "downstream_dir": str(Path(run["run_dir"]).resolve()),
        "checkpoint_dir": str(Path(run["checkpoint_dir"]).resolve()),
        "base_dim": int(run["base_dim"]),
        "embed_dim": int(run["embed_dim"]),
        "num_layers_backbone": int(run["num_layers_backbone"]),
        "mambaversion": run.get("model_family", DEFAULT_MODEL_FAMILY),
        "model_version": run["model_config"],
    })
    write_yaml(Path(run["model_yaml"]), {run["model_config"]: params})


def render_analysis_yaml(manifest: dict[str, Any], run: dict[str, Any]) -> None:
    base_analysis_yaml = resolve_repo_path(manifest["base_analysis_yaml"])
    data = read_yaml(base_analysis_yaml)
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
        "use_pretrained_backbone": True,
        "pretrained_checkpoint": str(Path(run["pretrained_checkpoint"]).resolve()),
    })
    write_yaml(Path(run["analysis_yaml"]), {"analysis": analysis})


def train_command(manifest: dict[str, Any], run: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        "train/downstream/train_track_regression.py",
        "--yaml_config", str(Path(run["model_yaml"]).resolve()),
        "--config", run["model_config"],
        "--run_num", run["run_id"],
        "--root_dir", str(Path(run["train_dir"]).resolve()),
        "--global_log_dir", str(Path(manifest["campaign_dir"]).resolve() / "logs" / "global"),
        "--eventnumber", str(int(run["eventnumber"])),
        "--train_batch_size", str(int(run["train_batch_size"])),
        "--usepretrain",
        "--pretrained_ckpt", str(Path(run["pretrained_checkpoint"]).resolve()),
        "--checkpoint_dir", str(Path(run["checkpoint_dir"]).resolve()),
        "--log_file_name", f"{run['run_id']}.log",
        "--checkpoint_file_name", f"{run['run_id']}_adapter_checkpoint.pth",
        "--artifact_summary", str(Path(run["artifact_summary"]).resolve()),
    ]


def eval_command(run: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        "train/downstream/eval/evaluate_track_regression.py",
        "--analysis-config", str(Path(run["analysis_yaml"]).resolve()),
    ]


def format_command(command: list[str]) -> str:
    return " ".join(shlex_quote(part) for part in command)


def shlex_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_@%+=:,./-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def command_env(cuda_device: str | None, manifest: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MPLCONFIGDIR", str(Path(manifest["campaign_dir"]).resolve() / ".matplotlib"))
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    return env


def run_logged_command(
    command: list[str],
    log_path: Path,
    env: dict[str, str],
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as stream:
        stream.write(f"$ {format_command(command)}\n\n")
        stream.flush()
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode


def load_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"runs": {}}
    data = read_yaml(path)
    data.setdefault("runs", {})
    return data


def update_status(path: Path, run_id: str, status: str, **extra: Any) -> None:
    data = load_status(path)
    data.setdefault("runs", {})
    record = dict(data["runs"].get(run_id, {}))
    record.update(extra)
    record["status"] = status
    record["updated_at"] = utc_now()
    data["runs"][run_id] = record
    write_yaml(path, data)


def run_current_status(status_data: dict[str, Any], run: dict[str, Any]) -> str:
    return status_data.get("runs", {}).get(run["run_id"], {}).get(
        "status",
        run.get("status", "pending"),
    )


def collate_summary(manifest: dict[str, Any]) -> None:
    base_dir = Path(manifest["campaign_dir"]).resolve()
    summary_dir = base_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    headline_out = summary_dir / "campaign_headline_metrics.jsonl"
    table_out = summary_dir / "run_table.csv"

    table_rows = []
    with headline_out.open("w") as headline_stream:
        for run in manifest.get("runs", []):
            evaluation_dir = Path(run["evaluation_dir"])
            headline = evaluation_dir / "campaign_headline_metrics.jsonl"
            summary = evaluation_dir / "summary.json"
            if headline.exists():
                with headline.open() as stream:
                    shutil.copyfileobj(stream, headline_stream)
            table_row = {
                "run_id": run["run_id"],
                "embed_dim": run["embed_dim"],
                "num_layers_backbone": run["num_layers_backbone"],
                "pretrain_events": run["pretrain_events"],
                "adapter_checkpoint": run["adapter_checkpoint"],
                "evaluation_dir": run["evaluation_dir"],
                "summary_found": summary.exists(),
            }
            if summary.exists():
                summary_data = read_json(summary)
                adapter = summary_data.get("methods", {}).get("adapter", {})
                momentum = adapter.get("momentum", {})
                table_row.update({
                    "adapter_relative_resolution_68": momentum.get("relative_resolution_68"),
                    "adapter_relative_bias": momentum.get("relative_bias"),
                    "adapter_tail_fraction_10pct": momentum.get("relative_tail_fraction_10pct"),
                    "adapter_to_cvt_resolution_ratio": summary_data.get("adapter_to_cvt_resolution_ratio"),
                })
            table_rows.append(table_row)

    fieldnames = sorted({key for row in table_rows for key in row})
    with table_out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)
