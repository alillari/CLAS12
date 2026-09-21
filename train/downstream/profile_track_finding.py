#!/usr/bin/env python3
"""
Profile ONE short epoch of track-finding training to find the real per-step
bottleneck, using PyTorch's built-in profiler (correctly handles async CUDA
timing, unlike hand-rolled time.time() calls). Wraps the REAL trainer.train()
call -- not a reimplementation -- so results reflect actual behavior exactly.

Uses the small clas12_track_finding_m1_profile config block (limit_size=2000,
max_epochs=1) so this finishes in well under a minute instead of waiting
through full epochs.

Run (from train/downstream, same working directory as train_track_finding.py):
    python profile_track_finding.py \
        --yaml_config /workspace/PP_collision/scripts/configs/mamba_clas12_track_finding_pretrained_sweep.yaml \
        --config clas12_track_finding_m1_profile \
        --pretrained_ckpt /workspace/PP_collision/checkpoints/Unbiased_Cluster_Full_Sweep_2026-08-13/scale_w64_d12_n39553933/sweep/training_checkpoints/ckpt_best.tar
"""
import os
import sys
import argparse
import gc

import torch
from torch.profiler import profile, ProfilerActivity

sys.path.append('../..')

from fm4npp.utils import YParams
from track_finding_trainer import DownstreamTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--run_num", default="profile", type=str)
    parser.add_argument("--root_dir", default="/workspace/PP_collision/downstream_log/", type=str)
    parser.add_argument("--global_log_dir", default="globallogs", type=str)
    parser.add_argument("--eventnumber", default=2000, type=int)
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--mambaversion", default=None, type=str)
    parser.add_argument("--pretrained_ckpt", default=None, type=str)
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params.continue_from_best = False  # never resume for a profiling run
    params.batch_size = int(args.train_batch_size)
    params.limit_data = True
    params.valid_batch_size = 1
    if args.pretrained_ckpt:
        params.pretrained_ckpt = args.pretrained_ckpt
    if args.mambaversion is not None:
        params.mambaversion = args.mambaversion
    params.log_file_name = f"{args.config}_profile.log"
    params.num_embedder_layers = 0

    trainer = DownstreamTrainer(params, args)
    trainer.launch()

    print("\n" + "=" * 70)
    print("PROFILING one short epoch (this will take under a minute)...")
    print("=" * 70 + "\n")

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=False, with_stack=False) as prof:
        trainer.train(pretrain=True, train_from_checkpoint=False, checkpoint_path=None)

    print("\n" + "=" * 70)
    print("TOP OPERATIONS BY SELF CUDA TIME (GPU-side cost)")
    print("=" * 70)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

    print("\n" + "=" * 70)
    print("TOP OPERATIONS BY SELF CPU TIME (CPU-side cost, incl. data loading)")
    print("=" * 70)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

    trainer.cleanup()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()