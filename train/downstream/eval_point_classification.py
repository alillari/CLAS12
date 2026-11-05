#!/usr/bin/env python3
"""
Evaluation script for point classification downstream task.
"""
import os
import sys
import argparse
import gc
from pathlib import Path

import torch

# make sure the FM4NPP modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from fm4npp.utils import YParams
from point_classification_trainer import DownstreamTrainer

def main():
    parser = argparse.ArgumentParser(
        description="Evaluation script for point classification downstream task"
    )
    parser.add_argument(
        "--yaml_config",
        type=str,
        required=True,
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Model config name (e.g. d9_m64_k30_p20)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained downstream checkpoint",
    )
    parser.add_argument(
        "--run_num",
        type=str,
        default="0",
        help="Run number / seed identifier",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="evaluation_output/point_classification",
        help="Root directory to store evaluation outputs",
    )
    parser.add_argument(
        "--eventnumber",
        type=int,
        default=70000,
        help="Number of events (samples) to evaluate",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    parser.add_argument("--global_log_dir", default='globallogs', type=str, help="Global dir to store logging only")
    parser.add_argument("--pretrained_ckpt", help="Optional backbone checkpoint; omitted for AdapterOnly")
    parser.add_argument("--task", choices=("pid", "nid"))
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params["task"] = args.task or getattr(params, "task", "pid")
    params["num_output_classes"] = 2 if params.task == "nid" else 5
    params["limit_data"] = True
    params["limit_size"] = 1  # launch constructs an unused training view
    params["limit_test_data"] = True
    params["limit_test_size"] = args.eventnumber
    params["drop_last_test"] = False
    params["batch_size"] = args.eval_batch_size
    params["valid_batch_size"] = args.eval_batch_size
    params["pretrained_ckpt"] = args.pretrained_ckpt
    params["continue_from_best"] = False
    params["log_file_name"] = f"{args.config}_eval_{params.task}_{args.run_num}.log"
    params["num_embedder_layers"] = 0
    if getattr(params, "reader_type", None) == "hdf5":
        params["validation_split"] = "test"
    checkpoint_path = os.path.abspath(args.checkpoint)

    # Ensure output directory exists
    log_dir = args.root_dir
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, params.log_file_name)

    # Launch and run inference
    trainer = DownstreamTrainer(params, args)
    trainer.launch()
    trainer.inference(
        checkpoint_path=checkpoint_path,
        pretrain=bool(args.pretrained_ckpt),
        logfile=logfile
    )
    trainer.cleanup()

    # Free GPU memory
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
