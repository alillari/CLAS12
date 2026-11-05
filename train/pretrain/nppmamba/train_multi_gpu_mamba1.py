"""
Mamba pretraining (legacy module name retained for existing launchers).
Mamba1 remains the default; Mamba2, Mamba3 SISO, and causal transformers are explicit selections.

The next-point objective and AdamW recipe are shared. Mamba1/2 retain FP32
execution; Mamba3 blocks manage BF16 locally. See train/MAMBA_SUPPORT.md.
"""

import os
import shutil
import time
import numpy as np
import argparse
from itertools import islice
from pathlib import Path

import torch
import torch.nn.init as init
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from fm4npp.artifact_paths import artifact_path
from fm4npp.hilbert import LEGACY_HILBERT_GRID
from fm4npp.models.mamba_config import (
    backbone_version, build_backbone, architecture_metadata,
    optimizer_parameters, validate_checkpoint_architecture,
)
from train.pretrain.nppmamba.optimizer_state import parameter_names, restore_optimizer
from fm4npp.datasets.dataset_pretrain import get_data_loader
from fm4npp.utils import YParams, pickle_load

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

VALIDATION_METRIC = "mse_per_valid_coordinate_v1"

# [W&B] Optional experiment tracking. Enabled per-config via `use_wandb: true`.
# Runs in OFFLINE mode (no account needed): logs are written to ./wandb/ locally
# and can be synced/inspected later with `wandb sync` or viewed with `wandb offline`.
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_bin_weights_torch(bin_list, weight_list, target):
    """Apply bin-based weighting for loss."""
    device = target.device
    weight_list = weight_list.to(device)
    bin_list = bin_list.to(device)

    target = target.unsqueeze(-1)
    mask_larger_than = (target >= bin_list[:-1].unsqueeze(0)).float()
    mask_smaller_than = (target < bin_list[1:].unsqueeze(0)).float()
    mask_in_bin = mask_larger_than * mask_smaller_than

    weights = (mask_in_bin * weight_list.unsqueeze(0)).sum(-1)
    return weights


class Trainer():
    def __init__(self, params, args):
        self.params = params
        self.hilbert_angular_grid = getattr(params, 'hilbert_angular_grid', LEGACY_HILBERT_GRID)
        # The shared adapter/backbone contract uses continuous coordinate targets.
        # Mike's retired band-classification head is not part of this integration.
        if getattr(params, 'band_classification', False):
            raise ValueError('band_classification is not supported by the shared backbone')
        self.args = args
        self.root_dir = args.root_dir
        self.config = args.config
        self.run_num = args.run_num

        # Initialize distributed training
        self.world_rank = 0
        self.local_rank = 0
        self.world_size = 1

        if dist.is_initialized():
            self.world_rank = dist.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = dist.get_world_size()

        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.local_rank)

        # Setup directories
        exp_dir = os.path.join(*[self.root_dir, self.config, self.run_num])
        self.exp_dir = exp_dir
        self.checkpoint_path = os.path.join(exp_dir, 'training_checkpoints/ckpt.tar')

        if self.world_rank == 0:
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

        # Setup logging
        self.iters = 0
        self.startEpoch = 0
        self.epoch = 0
        self.best_loss = np.inf
        self.logs = {}

        # Setup log files
        self.log_to_screen = True
        self.logfile = os.path.join(exp_dir, 'train.log')
        self.globalfile = os.path.join(exp_dir,
            'config_{}_run_{}.csv'.format(self.config, self.run_num))
        self.finisher = os.path.join(exp_dir, 'finished.txt')

        if self.world_rank == 0:
            if not os.path.exists(self.globalfile):
                with open(self.globalfile, 'w') as f:
                    f.write("split,step,loss,lr\n")

        if self.params.loss_reweight:
            self.loss_bin = torch.as_tensor(pickle_load(Path(params.stat_dir) / 'loss_bin_pp.pkl'),
                                            dtype=torch.float32, device=self.device)
            self.loss_weight = torch.as_tensor(pickle_load(Path(params.stat_dir) / 'loss_weight_pp.pkl'),
                                               dtype=torch.float32, device=self.device)
            if (self.loss_weight.ndim != 1 or self.loss_bin.shape != (len(self.loss_weight) + 1,)
                    or not torch.all(self.loss_bin[1:] > self.loss_bin[:-1])):
                raise ValueError("Loss reweighting requires one weight per increasing bin interval")

        # Get data loaders
        self.train_data_loader, self.train_sampler, self.valid_data_loader, _ = \
            get_data_loader(self.params, dist.is_initialized())

        # Keep the historical Mamba1 default for existing launchers/configs.
        self.klen = self.params.klen
        # sin/cos sets the output width (klen*4 vs klen*3). Strip-direction weighting is
        # independent: it applies only when the dataset returns each target's aux.
        self.use_sincos = getattr(self.params, 'use_sincos_target', False)
        self.target_aux_dim = (getattr(self.params, 'aux_dim', 4)
                               if getattr(self.params, 'return_knn_aux_target', False) else 0)
        if self.target_aux_dim and not self.use_sincos:
            raise ValueError('return_knn_aux_target requires use_sincos_target')
        version = backbone_version(self.params, default='mamba1')
        if version not in {'mamba1', 'mamba2', 'mamba3', 'transformer'}:
            raise ValueError('This pretraining entrypoint supports Mamba1, Mamba2, Mamba3 SISO, and causal transformers')
        self.model = build_backbone(self.params, default='mamba1')

        # Standard initialization
        if version == 'mamba1':
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if "norm.weight" in name:
                        init.ones_(param)
                    elif "bias" in name:
                        init.zeros_(param)

        if self.world_rank == 0:
            print(f"✅ {version} Model Initialized")
            print(f"   Architecture: {architecture_metadata(self.model)}")

        self.model = self.model.to(self.device)

        if self.world_rank == 0:
            print(f'Nparams: {count_parameters(self.model):,}')

        # Distributed wrapper
        if dist.is_initialized():
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank] if self.device.type == 'cuda' else None,
                find_unused_parameters=True
            )

        # [muP] Simplified width-transfer (mu-parameterization, practical form).
        # Split trainable params into two groups by tensor rank:
        #   - VECTOR params (biases, norm gains, Mamba A_log/D): base LR, NOT
        #     width-scaled (no fan-in that grows with width).
        #   - MATRIX params (Mamba block weights + output_layer): LR divided by
        #     the width ratio (embed_dim / mup_base_width). This is the piece
        #     that gives near-constant optimal LR across widths, so a recipe
        #     tuned at the base width transfers to other widths.
        # If use_mup is off, both groups get the same LR -> identical to before.
        self.mup_base_width = getattr(self.params, 'mup_base_width', 256)
        self.use_mup = getattr(self.params, 'use_mup', False)
        cur_width = self.params.embed_dim
        self.mup_matrix_ratio = (cur_width / self.mup_base_width) if self.use_mup else 1.0

        vector_params, matrix_params = [], []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            (vector_params if p.ndim <= 1 else matrix_params).append(p)

        wd = getattr(self.params, 'weight_decay', 0.1)
        # Keep rank-based width transfer after Mamba3 splits no-decay groups.
        vector_groups = optimizer_parameters(
            self.model, [{'params': vector_params, 'weight_decay': wd}])
        matrix_groups = optimizer_parameters(
            self.model, [{'params': matrix_params, 'weight_decay': wd}])
        self._mup_matrix_group_indices = list(
            range(len(vector_groups), len(vector_groups) + len(matrix_groups)))
        self.optimizer = torch.optim.AdamW(
            vector_groups + matrix_groups,
            lr=self.params.min_lr,
            betas=(0.9, 0.95),
        )
        # Retain the historical diagnostic attribute for existing callers.
        self._mup_matrix_group_idx = self._mup_matrix_group_indices[0]

        if self.world_rank == 0:
            if self.use_mup:
                print(f"✅ AdamW with muP width-transfer ENABLED")
                print(f"   base width: {self.mup_base_width}, current width: {cur_width}")
                print(f"   matrix-group LR divided by ratio: {self.mup_matrix_ratio:.4f}")
                print(f"   ({len(vector_params)} vector params @ base LR, "
                      f"{len(matrix_params)} matrix params @ base LR / {self.mup_matrix_ratio:.3f})")
            else:
                print(f"✅ AdamW, muP disabled (all params share one LR)")
                print(f"   Learning rate: {self.params.min_lr}")

        # Mamba1/2 keep their existing precision; Mamba3 blocks execute in BF16
        # locally and return FP32 to the existing prediction/loss path.

        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=self.params.total_steps,
            cycle_mult=1.0,
            max_lr=self.params.max_lr,
            min_lr=self.params.min_lr,
            warmup_steps=self.params.warmup_steps,
            gamma=1.0
        )

        # [W&B] Optional offline experiment tracking (no account needed).
        # Enable per-config with `use_wandb: true`. Logs hyperparameters,
        # train loss/lr, and val loss. Written locally to <root_dir>/wandb/.
        self.use_wandb = getattr(params, 'use_wandb', False) and _WANDB_AVAILABLE and self.world_rank == 0
        if self.use_wandb:
            os.environ['WANDB_MODE'] = 'offline'
            wandb.init(
                project='clas12-fm4npp',
                name=f"{getattr(params, 'save_version', 'run')}_{args.run_num}",
                config={k: v for k, v in vars(params).items()
                        if isinstance(v, (int, float, str, bool))},
                dir=os.path.join(args.root_dir, 'wandb'),
            )
            print("[W&B] offline logging enabled")
        elif getattr(params, 'use_wandb', False) and not _WANDB_AVAILABLE:
            print("[W&B] use_wandb requested but wandb not installed -- skipping.")

        # Load checkpoint if exists (previously dead code -- never ran)
        self.restore_checkpoint()

    def log_infile(self, log):
        """Write log to file"""
        if self.world_rank == 0:
            with open(self.logfile, "a") as f:
                f.write("{}\n".format(log))

    def log_globalfile(self, split, step, loss, lr):
        """Write to global CSV log"""
        if self.world_rank == 0:
            with open(self.globalfile, "a") as f:
                f.write("{},{},{},{}\n".format(split, step, loss, lr))

    def _scale_mup_matrix_lr(self):
        if self.use_mup and self.mup_matrix_ratio != 1.0:
            for index in self._mup_matrix_group_indices:
                group = self.optimizer.param_groups[index]
                group['lr'] /= self.mup_matrix_ratio

    def restore_checkpoint(self):
        """Restore from checkpoint if exists"""
        if os.path.isfile(self.checkpoint_path):
            if self.world_rank == 0:
                print(f"Loading checkpoint from {self.checkpoint_path}")

            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            saved_grid = checkpoint.get('hilbert_angular_grid', LEGACY_HILBERT_GRID)
            if saved_grid != self.hilbert_angular_grid:
                raise ValueError(f"Checkpoint Hilbert grid {saved_grid!r} differs from requested {self.hilbert_angular_grid!r}")
            validate_checkpoint_architecture(checkpoint, self.model, 'backbone')

            try:
                self.model.load_state_dict(checkpoint['model_state'])
            except:
                if dist.is_initialized():
                    self.model.module.load_state_dict(checkpoint['model_state'])
                else:
                    self.model.load_state_dict(checkpoint['model_state'])

            restore_optimizer(self.model, self.optimizer, checkpoint)
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if checkpoint.get('validation_metric') == VALIDATION_METRIC:
                self.best_loss = checkpoint.get('best_loss', self.best_loss)
            elif self.world_rank == 0:
                self.log_infile(f"Resuming at step {checkpoint['iters']} with a new best-loss baseline for {VALIDATION_METRIC}")
            self.iters = checkpoint['iters']
            self.startEpoch = checkpoint['epoch'] + 1

            if self.world_rank == 0:
                print(f"Resuming from epoch {self.startEpoch}, iteration {self.iters}")

    def save_checkpoint(self, checkpoint_path, is_best=False):
        """Save checkpoint"""
        if self.world_rank != 0:
            return

        try:
            model_state = self.model.module.state_dict() if dist.is_initialized() else self.model.state_dict()
        except:
            model_state = self.model.state_dict()

        payload = {
            'iters': self.iters,
            'epoch': self.epoch,
            'model_state': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'optimizer_param_names': parameter_names(self.model, self.optimizer),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'validation_metric': VALIDATION_METRIC,
            'hilbert_angular_grid': self.hilbert_angular_grid,
            'backbone_architecture': architecture_metadata(self.model),
        }
        torch.save(payload, checkpoint_path)

        if is_best:
            best_path = checkpoint_path.replace('.tar', '_best.tar')
            torch.save(payload, best_path)
            if self.world_rank == 0:
                print(f"Saved best checkpoint to {best_path}")
        else:
            if self.world_rank == 0:
                print(f"Saved checkpoint to {checkpoint_path}")

    def _batch_targets(self, grouped, knearest):
        """Mask is per coordinate for the plain target, per neighbor slot for sin/cos."""
        inputs = grouped.to(self.device)
        offset = (2 if self.params.nexttoken else 1) if self.params.rep_aaai else 0
        targets = knearest.to(self.device)[:, offset:]
        if self.use_sincos:
            b, n = targets.shape[:2]
            targets = targets.reshape(b, n, self.klen, 3 + self.target_aux_dim)
            return inputs, targets, targets[..., 0] != -100, offset
        return inputs, targets, targets != -100, offset

    def _squared_errors(self, inputs, targets, mask):
        """Per-event sum of squared errors over the supervised entries of `mask`."""
        predictions = self.model(inputs)[:, :targets.size(1)]
        if not self.use_sincos:
            return (predictions - targets).masked_fill(~mask, 0.).square().sum(dim=(1, 2))
        return self._sincos_errors(predictions, targets, mask)

    def _sincos_errors(self, predictions, targets, mask):
        """
        Predict (eta, sin phi, cos phi, r) per neighbor instead of (eta, phi, r), so
        the phi = +-pi seam is not penalized (Beyer et al., "Biternion Nets", GCPR 2015).

        targets: (b, N, k, 3 + aux) = [eta, phi, r, s_r, s_phi, s_z, ...] per neighbor,
        phi normalized to [0, 1] by the dataset's (phi + pi) / (2 pi).
        With the target neighbor's strip direction (aux >= 3), each coordinate's error
        is weighted by its squared component (s_r^2 + s_phi^2 + s_z^2 = 1); without it
        the three errors are averaged. Returns the per-event sum over valid slots.
        """
        b, n = targets.shape[:2]
        pred = predictions.reshape(b, n, self.klen, 4)
        eta_t, phi_t, r_t = targets[..., 0], targets[..., 1], targets[..., 2]
        # Back to the physical angle, so sin/cos are actually periodic at the seam.
        phi_phys = phi_t * (2 * torch.pi) - torch.pi
        err_eta = (pred[..., 0] - eta_t) ** 2
        # (dsin)^2 + (dcos)^2 ~ dphi^2 in radians; / (2 pi)^2 puts it on the same
        # normalized [0, 1] scale as eta and r.
        err_phi = ((pred[..., 1] - torch.sin(phi_phys)) ** 2
                   + (pred[..., 2] - torch.cos(phi_phys)) ** 2) / (2 * torch.pi) ** 2
        err_r = (pred[..., 3] - r_t) ** 2
        if self.target_aux_dim >= 3:
            s_r, s_phi, s_z = targets[..., 3], targets[..., 4], targets[..., 5]
            combined = s_r ** 2 * err_r + s_phi ** 2 * err_phi + s_z ** 2 * err_eta
        else:
            combined = (err_r + err_phi + err_eta) / 3.0
        return combined.masked_fill(~mask, 0.).sum(dim=(1, 2))

    def train_one_epoch(self):
        self.model.train()
        start_iter = self.iters
        tr_time = 0.
        tr_start = time.monotonic()
        for grouped, _, knearest in self.train_data_loader:
            inputs, targets, mask, offset = self._batch_targets(grouped, knearest)
            counts = mask.sum(dim=(1, 2))
            supervised = counts > 0
            # Unweighted MSE averages coordinates; reweighting averages supervised events.
            denominator = supervised.sum() if self.params.loss_reweight else counts.sum()
            if dist.is_initialized():
                dist.all_reduce(denominator)
            if denominator.item() == 0:
                continue

            self.optimizer.zero_grad()
            errors = self._squared_errors(inputs, targets, mask)
            if self.params.loss_reweight:
                lengths = (inputs[:, offset:, 0] != -100).sum(dim=1)
                weights = apply_bin_weights_torch(self.loss_bin, self.loss_weight, lengths)
                errors = errors / counts.clamp_min(1) * weights
            # DDP averages gradients across ranks after backward.
            loss = errors.sum() * self.world_size / denominator
            if self.params.ablate_loss_scale:
                loss = loss * self.params.ablate_loss_scale_rate

            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.params.grad_clip_value)
            self.optimizer.step()
            self.scheduler.step()
            self._scale_mup_matrix_lr()
            self.iters += 1

            reported = loss.detach().clone()
            if dist.is_initialized():
                dist.all_reduce(reported)
                reported /= self.world_size
            if self.world_rank == 0:
                value = reported.item()
                lr = self.scheduler.get_lr()[0]
                if self.iters % 100 == 0:
                    print(f'Iter {self.iters}: loss={value:.4f}')
                self.log_globalfile('train', self.iters, value, lr)
                if self.use_wandb:
                    wandb.log({'train/loss': value, 'train/lr': lr}, step=self.iters)

            finished = self.iters >= self.params.total_steps
            if finished or self.iters % self.params.n_eval_steps == 0:
                tr_time += time.monotonic() - tr_start
                self.val_one_epoch(tr_time)
                tr_start = time.monotonic()
            if finished:
                break

        if self.iters == start_iter:
            raise ValueError("Training epoch produced no valid outward-neighbor targets")
        return tr_time + (time.monotonic() - tr_start)

    def val_one_epoch(self, tr_time):
        """Evaluate MSE over all supervised coordinates in the selected batches."""
        self.model.eval()
        val_start = time.time()
        totals = torch.zeros(2, dtype=torch.float64, device=self.device)
        batch_limit = getattr(self.params, 'max_val_batches', None)
        with torch.no_grad():
            for grouped, _, knearest in islice(self.valid_data_loader, batch_limit):
                inputs, targets, mask, _ = self._batch_targets(grouped, knearest)
                errors = self._squared_errors(inputs, targets, mask)
                totals[0] += errors.sum(dtype=torch.float64)
                totals[1] += mask.sum()

        if dist.is_initialized():
            dist.all_reduce(totals)
        if totals[1].item() == 0:
            raise ValueError("Validation produced no valid outward-neighbor targets")
        val_loss = (totals[0] / totals[1]).float()
        if self.params.ablate_loss_scale:
            val_loss *= self.params.ablate_loss_scale_rate
        self.logs['val_loss'] = val_loss.view(1)
        val_time = time.time() - val_start

        # Track best model
        is_best_loss = False
        if self.logs['val_loss'] <= self.best_loss:
            is_best_loss = True
            self.best_loss = self.logs['val_loss']

        # Save checkpoint
        if self.params.save_checkpoint:
            self.save_checkpoint(self.checkpoint_path, is_best=is_best_loss)
            # Optional kept snapshots (ckpt_step<N>.tar) for diagnostics over training.
            every = getattr(self.params, 'snapshot_every', 0)
            if every and self.iters % every == 0 and self.world_rank == 0:
                shutil.copyfile(self.checkpoint_path,
                                self.checkpoint_path.replace('.tar', f'_step{self.iters}.tar'))

        # Print and log results
        tolog = 'Time taken {:.2f} sec; with {:.2f} / {:.2f} in tr/val\n'.format(
            time.time() - self.starttime if hasattr(self, 'starttime') else val_time,
            tr_time, val_time)
        tolog += 'Step = {}, Val loss = {}'.format(self.iters, float(self.logs['val_loss']))

        if self.world_rank == 0 and self.log_to_screen:
            print(tolog)

        if self.world_rank == 0:
            self.log_infile(tolog)
            self.log_globalfile('val', self.iters, float(self.logs['val_loss']), self.scheduler.get_lr()[0])
            if self.use_wandb:
                wandb.log({'val/loss': float(self.logs['val_loss'])}, step=self.iters)

        self.model.train()
        return 0

    def launch(self):
        """Launch training"""
        if self.world_rank == 0:
            print("="*80)
            print("STARTING MAMBA PRETRAINING")
            print("="*80)
            print(f"Model: {backbone_version(self.params, default='mamba1')}")
            print(f"Embed dim: {self.params.embed_dim}")
            print(f"Num layers: {self.params.num_layers_backbone}")
            print(f"Learning rate: {self.params.min_lr}")
            print(f"Total steps: {self.params.total_steps}")
            print("="*80)

        for epoch in range(self.startEpoch, self.params.max_epochs):
            if self.iters >= self.params.total_steps:
                break

            self.epoch = epoch
            self.starttime = time.time()

            if dist.is_initialized() and self.train_sampler:
                self.train_sampler.set_epoch(epoch)

            tr_time = self.train_one_epoch()

            if self.world_rank == 0:
                print(f"Epoch {epoch}: train_time={tr_time:.2f}s")

        if self.world_rank == 0:
            print("\n" + "="*80)
            print("TRAINING COMPLETE")
            print("="*80)

        if self.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(prog="python -m train.pretrain", description="Pretrain a CLAS12 backbone")
    default_recipe = Path(__file__).resolve().parents[3] / 'scripts/configs/mamba_chinchilla_sweep.yaml'
    parser.add_argument("--yaml_config", default=str(default_recipe), type=str)
    parser.add_argument("--config", default='scale_w64_d12_bgmerged_hdf5', type=str)
    parser.add_argument("--data-root", help="Override the recipe's pretraining dataset directory")
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--root_dir", default=artifact_path('pretrain'), type=str)
    parser.add_argument("--global_log_dir", default='globallogs', type=str)

    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)

    if args.data_root:
        params['data_root'] = args.data_root
    trainer = Trainer(params, args)
    trainer.launch()


if __name__ == '__main__':
    main()
