#!/usr/bin/env python3
"""
Downstream track-property regression training script.
"""
import os
import sys
import argparse
import gc
import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from fm4npp.utils import YParams
from track_regression_trainer import DownstreamTrainer


def json_safe(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Downstream track regression training script")
    parser.add_argument("--yaml_config", default='', type=str, help="Path to YAML config file")
    parser.add_argument("--config", default='', type=str, help="Model config name")
    parser.add_argument("--run_num", default='0', type=str, help="Sub run number")
    parser.add_argument("--root_dir", default='./downstream_log/', type=str, help="Root dir to store results")
    parser.add_argument("--global_log_dir", default='globallogs', type=str, help="Global dir to store logging only")
    parser.add_argument("--eventnumber", default=50000, type=int, help="downstream training event number")
    parser.add_argument("--usepretrain", action="store_true", help="use pretrain model")
    parser.add_argument("--train_batch_size", default=32, type=int, help="train batch size")
    parser.add_argument("--pretrained_ckpt", default=None, type=str, help="Optional path to pretrained checkpoint if --usepretrain is set.")
    parser.add_argument("--checkpoint_dir", default=None, type=str, help="Optional explicit directory for adapter checkpoints and training log.")
    parser.add_argument("--log_file_name", default=None, type=str, help="Optional deterministic training log filename.")
    parser.add_argument("--checkpoint_file_name", default=None, type=str, help="Optional deterministic adapter checkpoint filename.")
    parser.add_argument("--artifact_summary", default=None, type=str, help="Optional JSON path for training artifact metadata.")
    args = parser.parse_args()

    # Mapping from model name to log file and checkpoint paths
    model2log = {
        'd9_m1_k5_p20': '/home/shuhang/FMNP/PRETRAIN_MAMBA/globallogs/config_d9_m1_k5_p20_run_noAMP0_data_version:pp_12M|limit_size:10000000|model_version:mtest1.csv',
        'd9_m3_k5_p20': '/home/shuhang/FMNP/PRETRAIN_MAMBA/globallogs/config_d9_m4_k5_p20_run_noAMP0_data_version:pp_12M|limit_size:10000000|model_version:mtest1.csv',
        'd9_m4_k5_p20': '/home/shuhang/FMNP/PRETRAIN_MAMBA/globallogs/config_d9_m3_k5_p20_run_noAMP0_data_version:pp_12M|limit_size:10000000|model_version:mtest1.csv',
        'd9_m5_k5_p20': '/home/shuhang/FMNP/PRETRAIN_MAMBA/globallogs/config_d9_m5_k5_p20_run_noAMP1_data_version:pp_12M|limit_size:10000000|model_version:mtest1.csv',
        'd9_nerf_m1_k5_p20': '/home/shuhang/FMNP/PRETRAIN_MAMBA/globallogs/config_d9_m1_k5_p20_run_noAMP0_data_version:pp_12M|limit_size:10000000|model_version:mtest1.csv',
        'd9_nerf_m3_k5_p20': '/home/shuhang/FMNP/PRETRAIN_MAMBA/globallogs/config_d9_m4_k5_p20_run_noAMP0_data_version:pp_12M|limit_size:10000000|model_version:mtest1.csv',
        'd9_nerf_m4_k5_p20': '/home/shuhang/FMNP/PRETRAIN_MAMBA/globallogs/config_d9_m3_k5_p20_run_noAMP0_data_version:pp_12M|limit_size:10000000|model_version:mtest1.csv',
        'd9_nerf_m5_k5_p20': '/home/shuhang/FMNP/PRETRAIN_MAMBA/globallogs/config_d9_m5_k5_p20_run_noAMP1_data_version:pp_12M|limit_size:10000000|model_version:mtest1.csv',
    }

    model2ckpt = {
        'd9_m5_k5_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m5_k5.ckpt',
        'd9_m1_k5_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m1_k5.ckpt',
        'd9_m3_k5_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m3_k5.ckpt',
        'd9_m4_k5_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m4_k5.ckpt',
        'd9_m4_k5_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m4_k5.ckpt',
        'd9_m5_k5_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m5_k5.ckpt',
        'd9_m1_k30_p20':'/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m1_k30.ckpt',
        'd9_m3_k30_p20':'/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m3_k30.ckpt',
        'd9_m4_k30_p20':'/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m4_k30.ckpt',
        'd9_m5_k30_p20':'/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/pp_nerf_m5_k30.ckpt',
        'd9_m64_k5_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m64_k5_debugged.ckpt',
        'd9_m64_k30_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m64_k30.ckpt',
        'd9_m96_k5_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m96_k5.ckpt',
        'd9_m96_k30_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m96_k30.ckpt',
        'd9_m128_k5_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m128_k5.ckpt',   
        'd9_m128_k30_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m128_k30.ckpt',
        'd9_m192_k5_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m192_k5.ckpt',
        'd9_m192_k30_p20': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/m192_k30.ckpt',
        'ablate_reference': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_reference.ckpt',
        'ablate_pe_PROJ': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_pe_PROJ.ckpt',
        'ablate_pe_FF': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_pe_FF.ckpt',
        'ablate_pe_CPE': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_pe_CPE.ckpt',
        'ablate_order_RPE': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_order_RPE.ckpt',
        'ablate_order_REP': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_order_REP.ckpt',
        'ablate_order_PER': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_order_PER.ckpt',
        'ablate_embedconcat': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_embedconcat.ckpt',
        'ablate_lossreweight': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_lossreweight.ckpt',
        'ablate_space_filling_hilbert': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_space_filling_hilbert.ckpt',
        'ablate_space_filling_z': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_space_filling_z.ckpt',
        'ablate_novoxelize': '/mldata/sli/sphenix_fm/pretrained_checkpoints/pretrained_models/ablate_novoxelize.ckpt',
    }

    # Example overrides for running in a notebook; uncomment to hardcode
    # args.yaml_config = "/home/shuhangli/FMNP/FM4NPP/scripts/configs/mamba.yaml"
    # args.config = "d9_m96_k5_p20"
    # args.run_num = "2"

    # Initialize parameters
    params = YParams(os.path.abspath(args.yaml_config), args.config)

    if not hasattr(params, "task"):
        raise ValueError("YAML config must define task: one of ['mom', 'momentum', '3vertex', '3vtx', 'Zvtx', 'Zvertex']")

    params.continue_from_best = True
    params.batch_size = int(args.train_batch_size)
    params.limit_data = True
    params.limit_size = int(args.eventnumber)
    params.valid_batch_size = params.batch_size
    params.num_embedder_layers = 0

    if args.checkpoint_dir is not None:
        params.checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    if args.usepretrain:
        params.pretrained_ckpt = args.pretrained_ckpt or model2ckpt.get(args.config)

        if params.pretrained_ckpt is None:
            raise ValueError(
                f"--usepretrain was set, but no checkpoint was provided and "
                f"args.config={args.config!r} is not in model2ckpt."
            )
        params.pretrained_ckpt = os.path.abspath(params.pretrained_ckpt)
        if not os.path.isfile(params.pretrained_ckpt):
            raise FileNotFoundError(
                f"Pretrained checkpoint does not exist: {params.pretrained_ckpt}"
            )
    else:
        params.pretrained_ckpt = None

    params.log_file_name = args.log_file_name or (
        f"{args.config}_nerf_{params.task}_d{params.limit_size}_{args.run_num}.log"
    )
    if not params.log_file_name.endswith(".log"):
        params.log_file_name = f"{params.log_file_name}.log"
    params.checkpoint_file_name = args.checkpoint_file_name or (
        params.log_file_name.rsplit(".", 1)[0] + "_checkpoint.pth"
    )

    # Launch and train
    trainer = DownstreamTrainer(params, args)
    trainer.launch()
    checkpoint_path = None
    trainer.train(pretrain=args.usepretrain, train_from_checkpoint=False, checkpoint_path=checkpoint_path)

    artifact_summary = {
        "config": args.config,
        "run_num": args.run_num,
        "yaml_config": os.path.abspath(args.yaml_config),
        "usepretrain": bool(args.usepretrain),
        "pretrained_ckpt": params.pretrained_ckpt,
        "experiment_dir": getattr(params, "experiment_dir", None),
        "checkpoint_dir": os.path.abspath(params.checkpoint_dir),
        "log_file": getattr(params, "training_log_path", None),
        "checkpoint": getattr(params, "trained_checkpoint_path", None),
        "best_loss": json_safe(getattr(trainer, "best_loss", None)),
        "eventnumber": int(args.eventnumber),
        "train_batch_size": int(args.train_batch_size),
        "embed_dim": json_safe(getattr(params, "embed_dim", None)),
        "base_dim": json_safe(getattr(params, "base_dim", None)),
        "num_layers_backbone": json_safe(getattr(params, "num_layers_backbone", None)),
        "mambaversion": json_safe(getattr(params, "mambaversion", None)),
    }
    summary_path = args.artifact_summary or os.path.join(
        params.checkpoint_dir,
        params.log_file_name.rsplit(".", 1)[0] + "_artifacts.json",
    )
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
    with open(summary_path, "w") as stream:
        json.dump(artifact_summary, stream, indent=2, allow_nan=False)
    print(f"Wrote training artifact summary to {summary_path}")

    # Cleanup
    trainer.cleanup()
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
