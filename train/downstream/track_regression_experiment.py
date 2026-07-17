#!/usr/bin/env python3
"""One-configuration track-regression training entrypoint."""

from __future__ import annotations

import gc
import json
import os
import random
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DOWNSTREAM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DOWNSTREAM_DIR))

from fm4npp.utils import YParams

try:
    from .track_regression_trainer import DownstreamTrainer
except ImportError:
    from track_regression_trainer import DownstreamTrainer


MODEL2CKPT = {
    "d9_m5_k5_p20": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m5_k5.ckpt",
    "d9_m1_k5_p20": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m1_k5.ckpt",
    "d9_m3_k5_p20": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m3_k5.ckpt",
    "d9_m4_k5_p20": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m4_k5.ckpt",
    "d9_m1_k30_p20": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m1_k30.ckpt",
    "d9_m3_k30_p20": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m3_k30.ckpt",
    "d9_m4_k30_p20": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m4_k30.ckpt",
    "d9_m5_k30_p20": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m5_k30.ckpt",
    "d9_m64_k5_p20": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m64_k5_debugged.ckpt",
    "d9_m64_k30_p20": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m64_k30.ckpt",
    "d9_m96_k5_p20": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m96_k5.ckpt",
    "d9_m96_k30_p20": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m96_k30.ckpt",
    "d9_m128_k5_p20": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m128_k5.ckpt",
    "d9_m128_k30_p20": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m128_k30.ckpt",
    "d9_m192_k5_p20": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m192_k5.ckpt",
    "d9_m192_k30_p20": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m192_k30.ckpt",
    "ablate_reference": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_reference.ckpt",
    "ablate_pe_PROJ": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_pe_PROJ.ckpt",
    "ablate_pe_FF": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_pe_FF.ckpt",
    "ablate_pe_CPE": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_pe_CPE.ckpt",
    "ablate_order_RPE": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_order_RPE.ckpt",
    "ablate_order_REP": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_order_REP.ckpt",
    "ablate_order_PER": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_order_PER.ckpt",
    "ablate_embedconcat": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_embedconcat.ckpt",
    "ablate_lossreweight": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_lossreweight.ckpt",
    "ablate_space_filling_hilbert": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_space_filling_hilbert.ckpt",
    "ablate_space_filling_z": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_space_filling_z.ckpt",
    "ablate_novoxelize": "/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_novoxelize.ckpt",
}


@dataclass
class TrackRegressionExperimentConfig:
    yaml_config: str
    config: str
    run_num: str = "0"
    root_dir: str = "./downstream_log/"
    global_log_dir: str = "globallogs"
    eventnumber: int = 50000
    usepretrain: bool = False
    train_batch_size: int = 32
    pretrained_ckpt: str | None = None
    checkpoint_dir: str | None = None
    log_file_name: str | None = None
    checkpoint_file_name: str | None = None
    artifact_summary: str | None = None
    resolved_config_path: str | None = None
    seed: int | None = None


def set_global_seed(seed: int | None, deterministic: bool = False) -> None:
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    return str(value)


def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as stream:
        json.dump(json_safe(payload), stream, indent=2, allow_nan=False)


def resolve_params(config: TrackRegressionExperimentConfig) -> YParams:
    params = YParams(os.path.abspath(config.yaml_config), config.config)

    if not hasattr(params, "task"):
        raise ValueError(
            "YAML config must define task: one of ['mom', 'momentum', "
            "'3vertex', '3vtx', 'Zvtx', 'Zvertex']"
        )

    params.continue_from_best = True
    params.batch_size = int(config.train_batch_size)
    params.limit_data = True
    params.limit_size = int(config.eventnumber)
    params.valid_batch_size = params.batch_size
    params.num_embedder_layers = 0

    if config.checkpoint_dir is not None:
        params.checkpoint_dir = os.path.abspath(config.checkpoint_dir)
    if config.seed is not None:
        params.seed = int(config.seed)

    if config.usepretrain:
        params.pretrained_ckpt = config.pretrained_ckpt or MODEL2CKPT.get(config.config)
        if params.pretrained_ckpt is None:
            raise ValueError(
                "--usepretrain was set, but no checkpoint was provided and "
                f"config={config.config!r} is not in MODEL2CKPT."
            )
        params.pretrained_ckpt = os.path.abspath(params.pretrained_ckpt)
        if not os.path.isfile(params.pretrained_ckpt):
            raise FileNotFoundError(
                f"Pretrained checkpoint does not exist: {params.pretrained_ckpt}"
            )
    else:
        params.pretrained_ckpt = None

    params.log_file_name = config.log_file_name or (
        f"{config.config}_nerf_{params.task}_d{params.limit_size}_{config.run_num}.log"
    )
    if not params.log_file_name.endswith(".log"):
        params.log_file_name = f"{params.log_file_name}.log"
    params.checkpoint_file_name = config.checkpoint_file_name or (
        params.log_file_name.rsplit(".", 1)[0] + "_checkpoint.pth"
    )
    return params


def train_experiment(
    config: TrackRegressionExperimentConfig,
    optuna_trial: Any | None = None,
    metrics_callback: Any | None = None,
) -> dict[str, Any]:
    """Train one resolved downstream track-regression configuration."""

    params = resolve_params(config)
    set_global_seed(
        getattr(params, "seed", None),
        deterministic=bool(getattr(params, "deterministic", False)),
    )
    if config.resolved_config_path:
        _write_json(config.resolved_config_path, params.params)

    trainer = DownstreamTrainer(params, config)
    try:
        trainer.launch()
        trainer.train(
            pretrain=config.usepretrain,
            train_from_checkpoint=False,
            checkpoint_path=None,
            optuna_trial=optuna_trial,
            metrics_callback=metrics_callback,
        )

        summary_path = config.artifact_summary or os.path.join(
            params.checkpoint_dir,
            params.log_file_name.rsplit(".", 1)[0] + "_artifacts.json",
        )
        artifact_summary = {
            "config": config.config,
            "run_num": config.run_num,
            "yaml_config": os.path.abspath(config.yaml_config),
            "usepretrain": bool(config.usepretrain),
            "pretrained_ckpt": params.pretrained_ckpt,
            "experiment_dir": getattr(params, "experiment_dir", None),
            "checkpoint_dir": os.path.abspath(params.checkpoint_dir),
            "log_file": getattr(params, "training_log_path", None),
            "checkpoint": getattr(params, "trained_checkpoint_path", None),
            "best_loss": json_safe(getattr(trainer, "best_loss", None)),
            "best_step": json_safe(getattr(trainer, "best_step", None)),
            "best_epoch": json_safe(getattr(trainer, "best_epoch", None)),
            "final_step": json_safe(getattr(trainer, "global_step", None)),
            "eventnumber": int(config.eventnumber),
            "train_batch_size": int(config.train_batch_size),
            "seed": json_safe(getattr(params, "seed", None)),
            "embed_dim": json_safe(getattr(params, "embed_dim", None)),
            "base_dim": json_safe(getattr(params, "base_dim", None)),
            "num_layers_backbone": json_safe(getattr(params, "num_layers_backbone", None)),
            "mambaversion": json_safe(getattr(params, "mambaversion", None)),
            "hostname": socket.gethostname(),
            "git_commit": git_commit(),
            "data_root": json_safe(getattr(params, "data_root", None)),
            "data_root_test": json_safe(getattr(params, "data_root_test", None)),
        }
        _write_json(summary_path, artifact_summary)
        print(f"Wrote training artifact summary to {summary_path}")

        return {
            "best_val_loss": json_safe(getattr(trainer, "best_loss", None)),
            "best_step": json_safe(getattr(trainer, "best_step", None)),
            "best_epoch": json_safe(getattr(trainer, "best_epoch", None)),
            "checkpoint_path": getattr(params, "trained_checkpoint_path", None),
            "log_file": getattr(params, "training_log_path", None),
            "artifact_summary": os.path.abspath(summary_path),
            "final_metrics": {
                "best_loss": json_safe(getattr(trainer, "best_loss", None)),
                "final_step": json_safe(getattr(trainer, "global_step", None)),
            },
            "seed": json_safe(getattr(params, "seed", None)),
        }
    finally:
        trainer.cleanup()
        torch.cuda.empty_cache()
        gc.collect()
