"""Audited cluster-data launch using the shared Mamba pretraining Trainer.

Reuses coordinate normalization, serialization, model, loss, optimizer and the
training loop. Targets restore the historical calibrated outward-band rule to
avoid exposing target coordinates in the causal input prefix. No primary
checkout, image or dataset is modified remotely.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from ruamel.yaml import YAML

from fm4npp.datasets.dataset_pretrain import TPCBatchDataset, MyCollator
from fm4npp.models.mamba_config import build_backbone, architecture_metadata, optimizer_parameters, validate_checkpoint_architecture
from fm4npp.utils import YParams
from dev_scripts.chinchilla_restart import trainer as shared
from dev_scripts.chinchilla_restart.policy import parameter_groups, group_audit
from dev_scripts.chinchilla_restart.targets import outward_band_targets

CONFIG = None
RECIPE = Path(__file__).with_name("recipe.yaml")
OUT = Path("/results")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


class CohortDataset(TPCBatchDataset):
    """Reuse parent tokenization; select cohorts and construct causal band targets."""

    def __init__(self, params, split, count, seed):
        assert params.target_rule == "calibrated_outward_bands"
        self.selection_params = params
        self.selection_count = count
        self.selection_seed = seed
        self.selection_split = split
        super().__init__(params.data_root, split=split, train=split == "pretrain",
                         num_pred_points=params.klen, group_size=params.group_size,
                         normalize=True, chunk_training=False, voxelize=False,
                         space_filling_order=True, space_filling_curve="z",
                         clas12_band_config=params.clas12_band_config, bin_dir=params.stat_dir)

    def __getitem__(self, index):
        points, labels, _ = super().__getitem__(index)
        targets, _ = outward_band_targets(points, self.num_pred_points, self.clas12_band_config)
        return points, labels, targets

    def filter_data(self, low_thr=1, high_thr=100):
        features = self.memmap_feature
        assert len(features) == len(self.memmap_seg_target) == len(self.memmap_reg_target), "Misaligned event sidecars"
        # Features have the fixed three-column Cartesian cluster-center contract.
        for index in sorted({0, len(features) // 2, len(features) - 1}):
            shape = features[index].shape
            assert len(shape) == 2 and shape[1] == 3, (index, shape)
        sizes = np.asarray(features.ends) - np.asarray(features.starts)
        assert np.all(sizes % 3 == 0), "Feature storage violates three-column contract"
        lengths = sizes // 3
        low = self.selection_params.minimum_clusters
        high = self.selection_params.maximum_clusters
        eligible = np.flatnonzero((lengths >= low) & (lengths <= high))
        assert len(eligible) >= self.selection_count, "Requested cohort exceeds eligible population"
        indices = eligible[:self.selection_count].copy()
        self.idxlist = indices
        self.seqlens = lengths[indices]
        self.shortest, self.longest = int(self.seqlens.min()), int(self.seqlens.max())
        self.tooshort, self.toolong = [], []
        feature_file = Path(features.memmap.filename)
        path = OUT / (self.selection_split + "_event_indices.npy")
        np.save(path, indices)
        self.audit = {"split": self.selection_split, "raw_events": len(features),
                      "eligible_events": len(eligible), "selected_events": len(indices),
                      "removed_below_minimum": int((lengths < low).sum()),
                      "removed_above_maximum": int((lengths > high).sum()),
                      "cluster_bounds": [low, high], "selected_cluster_quantiles": np.quantile(self.seqlens, [0, .5, .95, .99, 1]).tolist(),
                      "seed": self.selection_seed, "selection": "first_eligible", "indices_sha256": sha(path),
                      "feature_file": str(feature_file), "feature_bytes": feature_file.stat().st_size,
                      "feature_mtime_ns": feature_file.stat().st_mtime_ns}
        print("COHORT " + json.dumps(self.audit), flush=True)


def make_loaders(params, distributed):
    assert not distributed, "This launch is explicitly single-GPU"
    train = CohortDataset(params, "pretrain", params.train_events, params.seed)
    validation = CohortDataset(params, "test", params.validation_events, params.validation_seed)
    write_json(OUT / "cohorts.json", {"training": train.audit, "validation": validation.audit,
               "validation_boundary": "Fixed subset of the historical test split, used for checkpoint selection; not an independent final holdout."})
    common = dict(num_workers=params.num_data_workers, pin_memory=True, collate_fn=MyCollator())
    if params.num_data_workers:
        common.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train, batch_size=params.local_batch_size, shuffle=True,
                              drop_last=True, generator=torch.Generator().manual_seed(params.seed), **common)
    validation_loader = DataLoader(validation, batch_size=params.local_valid_batch_size,
                                   shuffle=False, drop_last=True, **common)
    return train_loader, None, validation_loader, None


class AuditedTrainer(shared.Trainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.audited_steps = 0
        self.optimizer.register_step_pre_hook(self.before_step)

    def restore_checkpoint(self):
        if not Path(self.checkpoint_path).exists():
            return
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        validate_checkpoint_architecture(checkpoint, self.model, "backbone")
        assert checkpoint["recipe_sha256"] == sha(RECIPE)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.iters = checkpoint["iters"]
        self.startEpoch = checkpoint["epoch"] + 1
        self.best_loss = checkpoint["best_loss"]

    def before_step(self, optimizer, args, kwargs):
        base_lr = next(g["lr"] for g in optimizer.param_groups if not g["matrix"])
        if self.iters > 1:
            for group in optimizer.param_groups:
                expected = base_lr / self.mup_matrix_ratio if group["matrix"] else base_lr
                assert abs(group["lr"] - expected) <= 1e-12, "Width LR scaling was lost"
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        assert grads, "No trainable gradients"
        if not torch.stack([torch.isfinite(g).all() for g in grads]).all().item():
            raise FloatingPointError(f"Nonfinite gradients at step {self.iters}; update refused")
        self.audited_steps += 1
        if self.iters <= 3 or self.iters % 100 == 0:
            torch.cuda.synchronize()
            elapsed = time.monotonic() - self.audit_started
            write_json(OUT / "progress.json", {"step": self.iters, "total_steps": self.params.total_steps,
                       "elapsed_seconds": elapsed, "seconds_per_step": elapsed / max(1, self.audited_steps),
                       "lr": optimizer.param_groups[0]["lr"], "group_lrs": [g["lr"] for g in optimizer.param_groups], "finite_gradients": True,
                       "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()})

    def val_one_epoch(self, tr_time):
        super().val_one_epoch(tr_time)
        loss = float(self.logs["val_loss"])
        if not np.isfinite(loss):
            raise FloatingPointError(f"Nonfinite validation loss at {self.iters}")
        write_json(OUT / "latest_validation.json", {"step": self.iters, "validation_mse": loss,
                   "best_validation_mse": float(self.best_loss), "validation_events": self.params.validation_events, "validation_batches": min(self.params.max_val_batches, len(self.valid_data_loader))})

    def save_checkpoint(self, checkpoint_path, is_best=False):
        # Keep the shared checkpoint contract, writing through temporary files.
        temporary = checkpoint_path + ".writing"
        torch.save({"iters": self.iters, "epoch": self.epoch, "model_state": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "backbone_architecture": architecture_metadata(self.model),
                    "best_loss": float(self.best_loss), "recipe_sha256": sha(RECIPE)}, temporary)
        os.replace(temporary, checkpoint_path)
        if is_best:
            destination = checkpoint_path.replace(".tar", "_best.tar")
            shutil.copyfile(checkpoint_path, destination + ".writing")
            os.replace(destination + ".writing", destination)


def target_audit(dataset):
    k = dataset.num_pred_points
    counts = np.zeros(k, dtype=np.int64)
    tokens = 0
    visible_targets = repeated_slots = adjacent_valid_pairs = 0
    for index in range(min(256, len(dataset))):
        points, labels, targets = dataset[index]
        assert points.ndim == 2 and points.shape[1] == 3 and torch.isfinite(points).all()
        assert len(points) == len(labels) == len(targets)
        valid = targets.reshape(-1, k, 3) != -100
        assert torch.equal(valid[..., 0], valid[..., 1]) and torch.equal(valid[..., 1], valid[..., 2])
        assert valid.any(), "An eligible event has no valid next-point target"
        assert torch.isfinite(targets[targets != -100]).all()
        expected, indices = outward_band_targets(points, k, dataset.clas12_band_config)
        torch.testing.assert_close(targets, expected, rtol=0, atol=0)
        prefix = torch.arange(len(points))[None, :] <= torch.arange(len(points))[:, None]
        matches = (targets.reshape(-1, k, 3)[:, :, None, :] - points[None, None, :, :]).abs().amax(-1) < 1e-7
        visible_targets += int(((matches & prefix[:, None, :]).any(-1) & valid[..., 0]).sum())
        pair_valid = (indices[:, 1:] >= 0) & (indices[:, :-1] >= 0)
        repeated_slots += int(((indices[:, 1:] == indices[:, :-1]) & pair_valid).sum())
        adjacent_valid_pairs += int(pair_valid.sum())
        counts += valid[..., 0].sum(0).numpy()
        tokens += len(points)
    assert np.all(counts > 0)
    assert visible_targets == 0, "Target coordinate is already visible in the input prefix"
    return {"events_checked": min(256, len(dataset)), "cluster_tokens": tokens,
            "valid_targets_per_slot": counts.tolist(),
            "target_slots_already_in_input_prefix": visible_targets,
            "repeated_adjacent_slot_targets": repeated_slots,
            "adjacent_valid_slot_pairs": adjacent_valid_pairs,
            "objective": "Historical calibrated12 nearest next-band coordinate target, k=1. Targets are constructed after serialization and must be strictly later in the input sequence."}


def reload_check(trainer):
    checkpoint = torch.load(trainer.checkpoint_path, map_location="cpu", weights_only=False)
    clone = build_backbone(trainer.params).cuda().eval()
    validate_checkpoint_architecture(checkpoint, clone, "backbone")
    clone.load_state_dict(checkpoint["model_state"], strict=True)
    points = next(iter(trainer.valid_data_loader))[0][:2].cuda()
    trainer.model.eval()
    with torch.no_grad():
        torch.testing.assert_close(trainer.model(points), clone(points), rtol=1e-4, atol=1e-4)
    optimizer = torch.optim.AdamW(parameter_groups(clone, trainer.params.weight_decay),
                                  lr=trainer.params.min_lr, betas=(.9, .95))
    scheduler = shared.CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=trainer.params.total_steps,
                  max_lr=trainer.params.max_lr, min_lr=trainer.params.min_lr,
                  warmup_steps=trainer.params.warmup_steps, cycle_mult=1., gamma=1.)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    assert scheduler.state_dict() == trainer.scheduler.state_dict()
    assert [g["lr"] for g in optimizer.param_groups] == [g["lr"] for g in trainer.optimizer.param_groups]
    assert checkpoint["iters"] == trainer.iters
    trainer.model.train()
    del clone, optimizer, scheduler, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return "strict model reload, prediction equivalence, optimizer and scheduler restore passed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["preflight", "train"], required=True)
    parser.add_argument("--backbone", choices=["mamba2", "mamba3"], required=True)
    args = parser.parse_args()
    global CONFIG
    CONFIG = args.backbone + "_chinchilla_w512_d12_k1"
    OUT.mkdir(exist_ok=True)
    if args.phase == "preflight" and Path("/warmup/triton").exists():
        shutil.copytree("/warmup/triton", OUT / "triton", dirs_exist_ok=True)
    if args.phase == "train":
        passed = json.loads(Path("/pilot/preflight_pass.json").read_text())
        assert passed["recipe_sha256"] == sha(RECIPE), "Pilot recipe differs from production"
        assert passed["runner_sha256"] == sha(Path(__file__)), "Pilot code differs from production"
        assert passed["targets_sha256"] == sha(Path(__file__).with_name("targets.py")), "Pilot target code differs from production"
        assert passed["target_audit"]["target_slots_already_in_input_prefix"] == 0
        if Path("/pilot/triton").exists():
            shutil.copytree("/pilot/triton", OUT / "triton", dirs_exist_ok=True)
    params = YParams(str(RECIPE), CONFIG)
    params.stat_dir = str(OUT / "runtime_stats")
    Path(params.stat_dir).mkdir(exist_ok=True)
    for name, values in [("loss_bin_pp.pkl", np.array([0., 1e9])), ("loss_weight_pp.pkl", np.array([1.]))]:
        with (Path(params.stat_dir) / name).open("wb") as stream:
            pickle.dump(values, stream)
    if args.phase == "preflight":
        params.train_events = 4096
        params.validation_events = 512
        params.total_steps = 20
        params.warmup_steps = 2
        params.n_eval_steps = 10
        params.num_data_workers = 4
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "CUDA is required"
    shared.get_data_loader = make_loaders
    resolved = {k: v for k, v in vars(params).items() if not k.startswith("_") and k != "params"}
    with (OUT / "resolved_recipe.yaml").open("w") as stream:
        YAML().dump({CONFIG: resolved}, stream)
    start = time.monotonic()
    trainer = AuditedTrainer(params, SimpleNamespace(root_dir=str(OUT / "training"), config=CONFIG, run_num="seed42"))
    trainer.audit_started = time.monotonic()
    blocks = architecture_metadata(trainer.model)["blocks"]
    assert len(blocks) == 12 and all(c["version"] == args.backbone for c in blocks.values())
    if args.backbone == "mamba2":
        assert all(m.use_mem_eff_path for m in trainer.model.modules() if hasattr(m, "_mamba_config")), "Approximate fallback refused"
    assert trainer.model.output_layer.out_features == 3
    assert trainer.iters == 0, "Replacement starts fresh, never resumes k=5 weights"
    audit = {"phase": args.phase, "seed": params.seed, "recipe_sha256": sha(RECIPE),
             "runner_sha256": sha(Path(__file__)), "parameter_count": sum(p.numel() for p in trainer.model.parameters()),
             "targets_sha256": sha(Path(__file__).with_name("targets.py")),
             "trainable_parameter_count": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
             "initial_step": trainer.iters, "total_steps": params.total_steps,
             "train_events": params.train_events, "validation_events": params.validation_events,
             "architecture": architecture_metadata(trainer.model), "torch": torch.__version__,
             "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
             "packages": {name: importlib.metadata.version(name) for name in ["mamba-ssm", "triton", "causal-conv1d"]},
             "target_audit": target_audit(trainer.train_data_loader.dataset),
             "stats_note": "Unit-weight compatibility files only; loss_reweight=false and normalization uses existing fixed eta/phi/r bounds.",
             "recipe_origin": "origin/mike 38c82ec06265a530a4efccc853bba085d0742875 scripts/configs/mamba_chinchilla_sweep.yaml scale_w512_d12_chinchilla",
             "data_boundary": "Original mmap_cluster_full is unavailable by user confirmation; available v6_02 is shared identically across the new runs.",
             "effective_batch_size": params.local_batch_size,
             "effective_validation_events": min(params.max_val_batches * params.local_valid_batch_size, len(trainer.valid_data_loader.dataset)),
             "optimizer": {"name": "AdamW", "betas": [.9, .95], "weight_decay": params.weight_decay, "native_ssm_no_decay_flags_honored": True, "width_ratio": trainer.mup_matrix_ratio, "groups": group_audit(trainer.model, trainer.optimizer)},
             "validation_boundary": "Historical test subset selects checkpoints; not an independent final holdout.",
             "resume_boundary": "Model/optimizer/scheduler roundtrip is checked; these replacements start fresh. Intentional later resumption would not preserve data order bit-exactly."}
    write_json(OUT / "started.json", audit)
    trainer.launch()
    assert trainer.iters == params.total_steps
    audit.update(final_step=trainer.iters, best_validation_mse=float(trainer.best_loss),
                 finite_gradient_updates=trainer.audited_steps, elapsed_seconds=time.monotonic() - start,
                 checkpoint_reload=reload_check(trainer), peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(OUT / ("preflight_pass.json" if args.phase == "preflight" else "completed.json"), audit)
    print("RUN_COMPLETE " + json.dumps({k: audit[k] for k in ["phase", "final_step", "best_validation_mse", "elapsed_seconds", "checkpoint_reload"]}), flush=True)


if __name__ == "__main__":
    main()
