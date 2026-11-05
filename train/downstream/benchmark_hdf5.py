#!/usr/bin/env python3
"""Bounded AdapterOnly integration benchmark using the production trainers.

Example: python -m train.downstream.benchmark_hdf5 --task noise \
  --data-root /data/cvt --output-dir /results/noise
Validation selects the checkpoint; the test split is evaluated only afterwards.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fm4npp.utils import YParams
from fm4npp.datasets.dataset import get_val_loader
from train.downstream.track_finding_metrics import MatchConfig, event_track_metrics, summarize_event_metrics
from train.downstream.loss import assign_points_to_masks


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def write_json(path, data):
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")


def cohort(dataset):
    rows = []
    for key in dataset.idxlist:
        index, segment = key if isinstance(key, tuple) else (key, None)
        rows.append(dict(dataset.hdf5_source.identity(int(index)),
                         source_event_index=int(index), segment_label=segment))
    return rows


def event_keys(rows):
    return {(r["source_id"], r["original_ordinal"]) for r in rows}


def confusion_metrics(matrix):
    tn, fp, fn, tp = matrix.ravel()
    div = lambda x, y: float(x / y) if y else None
    return dict(confusion_true_rows_pred_columns=matrix, n_clusters=int(matrix.sum()),
                noise_precision=div(tp, tp + fp), noise_recall=div(tp, tp + fn),
                signal_retention=div(tn, tn + fp), accuracy=div(tp + tn, matrix.sum()))


def momentum_metrics(truth, predicted):
    valid = np.isfinite(truth).all(1) & np.isfinite(predicted).all(1)
    truth, predicted = truth[valid], predicted[valid]
    if not len(truth):
        return {"n_tracks": 0}
    p = np.linalg.norm(truth, axis=1)
    pp = np.linalg.norm(predicted, axis=1)
    theta = lambda x: np.arctan2(np.linalg.norm(x[:, :2], axis=1), x[:, 2])
    phi = lambda x: np.arctan2(x[:, 1], x[:, 0])
    dp = (pp - p) / p
    dphi = np.arctan2(np.sin(phi(predicted) - phi(truth)), np.cos(phi(predicted) - phi(truth)))
    return dict(n_tracks=len(p), cartesian_mae_gev=float(np.abs(predicted-truth).mean()),
                p_mae_gev=float(np.abs(pp-p).mean()), p_relative_bias=float(dp.mean()),
                p_relative_c68=float(np.diff(np.quantile(dp, [.16, .84]))[0] / 2),
                p_relative_tail_gt_10pct=float((np.abs(dp) > .1).mean()),
                theta_mae_rad=float(np.abs(theta(predicted)-theta(truth)).mean()),
                phi_mae_rad=float(np.abs(dphi).mean()))


@torch.no_grad()
def evaluate(trainer, loader, task, output):
    trainer.down_model.eval()
    matrix = {name: np.zeros((2, 2), dtype=np.int64) for name in ("adapter", "coatjava_unassigned")}
    event_rows = {name: [] for name in ("adapter", "coatjava")}
    records = []
    for batch in loader:
        points = batch["points"].to(trainer.device)
        mask = points[..., 0] != -100
        pred = trainer.down_model(points, feature=None, padding_mask=mask)
        if task == "finding":
            assignment = assign_points_to_masks(
                {"pred_probs": pred["class_probs"], "pred_masks": pred["mask_probs"].permute(0, 2, 1)},
                option=trainer.params.assignment_option, threshold=trainer.params.assignment_threshold)
        elif task == "momentum":
            targets = trainer.build_regression_targets(batch["reg_target"].to(trainer.device), mask,
                                                       batch["target_segment_mask"].to(trainer.device))
            truth = targets["native_target"].cpu().numpy()
            prediction = trainer.down_model.target_normalizer.denormalize(pred["pred_regression"]).cpu().numpy()
            valid_target = targets["target_valid"].all(-1).cpu().numpy()
        for i in range(len(points)):
            identity = {key: int(batch[key][i]) for key in ("source_id", "original_ordinal", "source_event_index")}
            valid = mask[i].cpu().numpy()
            if task == "noise":
                y = batch["noise_target"][i].numpy()[valid]
                predictions = {"adapter": pred["pred_logits"][i].argmax(-1).cpu().numpy()[valid],
                               "coatjava_unassigned": (batch["coatjava_seg_pred"][i].numpy()[valid] < 0).astype(int)}
                row = dict(identity)
                for name, labels in predictions.items():
                    counts = np.bincount(2*y + labels, minlength=4).reshape(2, 2)
                    matrix[name] += counts
                    row[name] = counts
                records.append(row)
            elif task == "finding":
                y = batch["target"][i].numpy()
                for name in event_rows:
                    labels = (assignment["assignments"][i].cpu().numpy().copy() if name == "adapter"
                              else batch["coatjava_seg_pred"][i].numpy())
                    signal = ((assignment["classes"][i].cpu().numpy() != 0) if name == "adapter" else labels >= 0)
                    labels[~signal] = -1
                    row = event_track_metrics(y, labels, valid_mask=valid, pred_signal_mask=signal & valid,
                                              config=MatchConfig())
                    row.update(identity)
                    event_rows[name].append(row)
            else:
                truth_label = int(batch["truth_segment_label"][i])
                reference = loader.dataset.hdf5_source.benchmark(identity["source_event_index"], truth_label)
                p, theta, phi = reference.pop("values")
                vector = p * np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
                records.append(dict(identity, truth_segment_label=truth_label,
                                    truth_valid=bool(valid_target[i]), truth=truth[i], adapter=prediction[i],
                                    coatjava=vector, association=reference))
    if task == "noise":
        summary = {name: confusion_metrics(counts) for name, counts in matrix.items()}
    elif task == "finding":
        summary = {name: summarize_event_metrics(rows) for name, rows in event_rows.items()}
        records = event_rows
    else:
        truth = np.asarray([r["truth"] for r in records])
        truth[~np.asarray([r["truth_valid"] for r in records])] = np.nan
        adapter = np.asarray([r["adapter"] for r in records])
        coatjava = np.asarray([r["coatjava"] for r in records])
        paired = np.isfinite(truth).all(1) & np.isfinite(coatjava).all(1) & np.isfinite(adapter).all(1)
        summary = dict(adapter_all=momentum_metrics(truth, adapter),
                       adapter_paired=momentum_metrics(truth[paired], adapter[paired]),
                       coatjava_paired=momentum_metrics(truth[paired], coatjava[paired]),
                       associations=dict(Counter(r["association"]["status"] for r in records)),
                       invalid_truth=int((~np.isfinite(truth).all(1)).sum()),
                       input_scope="adapter uses truth cluster groups; COATJAVA uses its reconstructed groups")
    write_json(output / "test_predictions.json", records)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("noise", "finding", "momentum"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["CLAS12_DATA_ROOT"] = str(args.data_root.resolve())
    os.environ["CLAS12_ARTIFACT_ROOT"] = str(output)
    os.environ["CLAS12_MOMENTUM_STATS"] = str(output / "momentum_stats.json")
    config = f"hdf5_{args.task}_5k"
    params = YParams(str(ROOT / "scripts/configs/clas12_adapter_recipes.yaml"), config)
    overrides = dict(limit_size=args.train_size, limit_test_size=args.validation_size,
                     max_validation_events=args.validation_size, max_optimizer_steps=args.steps,
                     total_steps=args.steps, first_cycle_steps=args.steps, scheduler_first_cycle_steps=args.steps,
                     warmup_steps=min(100, args.steps//10), val_interval_steps=min(500, args.steps),
                     num_data_workers=args.workers, pretrained_ckpt=None,
                     log_file_name="training.log", checkpoint_file_name="training_checkpoint.pth",
                     checkpoint_dir=str(output / "checkpoints"), require_reg_target=args.task == "momentum")
    for key, value in overrides.items():
        params[key] = value
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    if args.task == "momentum":
        from train.downstream.scripts.compute_regression_target_stats import compute_stats
        stats = compute_stats(args.data_root, "train", params.low_thr, params.high_thr,
                              args.train_size, 1000, task="mom", reader_type="hdf5",
                              segment_min_clusters=params.segment_min_clusters)
        write_json(output / "momentum_stats.json", stats)
        from train.downstream.track_regression_trainer import DownstreamTrainer
    elif args.task == "finding":
        from train.downstream.track_finding_trainer import DownstreamTrainer
    else:
        from train.downstream.point_classification_trainer import DownstreamTrainer
    trainer_args = SimpleNamespace(root_dir=str(output / "runtime"), global_log_dir="logs", config=config, run_num="0")
    trainer = DownstreamTrainer(params, trainer_args)
    trainer.launch()
    cohorts = {"train": cohort(trainer.train_data_loader.dataset), "validation": cohort(trainer.val_data_loader.dataset)}
    assert not (event_keys(cohorts["train"]) & event_keys(cohorts["validation"])), "Train/validation event overlap"
    assert len(cohorts["train"]) == args.train_size and len(cohorts["validation"]) == args.validation_size
    write_json(output / "resolved_config.json", params.params)
    with (output / "resolved_config.yaml").open("w") as stream:
        YAML().dump({config: clean(params.params)}, stream)
    trainer.train(pretrain=False)
    assert trainer.global_step == args.steps, "Incomplete update budget"
    checkpoint = Path(params.checkpoint_dir) / "training_checkpoint.pth"
    trainer.load_checkpoint(str(checkpoint), inference=True)
    params["limit_test_size"] = args.test_size
    loader = get_val_loader(params, False)
    cohorts["test"] = cohort(loader.dataset)
    assert len(cohorts["test"]) == args.test_size
    assert not (event_keys(cohorts["test"]) & (event_keys(cohorts["train"]) | event_keys(cohorts["validation"]))), "Test event overlap"
    write_json(output / "cohorts.json", cohorts)
    summary = evaluate(trainer, loader, args.task, output)
    summary.update(task=args.task, optimizer_steps=trainer.global_step, seed=params.seed,
                   train_samples=len(cohorts["train"]), validation_samples=len(cohorts["validation"]),
                   test_samples=len(cohorts["test"]), data_contract=params.data_contract,
                   checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   sample_unit="truth_segment" if args.task == "momentum" else "event",
                   validation_selection="signal_ARI_then_loss" if args.task == "finding" else "loss", test_split="test",
                   adapter_parameters=sum(p.numel() for p in trainer.down_model.parameters()))
    write_json(output / "summary.json", summary)
    print(json.dumps(clean(summary), indent=2), flush=True)
    trainer.cleanup()


if __name__ == "__main__":
    main()
