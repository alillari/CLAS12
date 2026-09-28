import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from sklearn.metrics import adjusted_rand_score
import os, sys, time, shutil, random, math
import argparse
import torch

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import gc, torch, torch.distributed as dist

from torch.utils.data import Dataset
from torch.nn.parallel import DistributedDataParallel

from tqdm import tqdm 
from ruamel.yaml import YAML
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import OrderedDict
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts # pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'

sys.path.append('../..')

from fm4npp.utils import *
from fm4npp.datasets.dataset import *
from fm4npp.models.mambagpt import MambaGPT, Mamba1GPT
from train.downstream.lr_schedulers import CosineAnnealingWarmupThenHold
from fm4npp.models.longformer_gpt import LongformerGPT
from fm4npp.models.linformer_gpt import LinformerGPT
from fm4npp.models.embed import *
from fm4npp.models.rmsnorm import RMSNorm
from fm4npp.models.mamba2 import Mamba2

from model import *
from loss import *
from downstream_util import *
from regression_utils import (
    load_regression_loss_reference_stats,
    load_regression_target_stats,
    regression_output_dim,
    resolve_regression_loss,
    transform_regression_target_torch,
)
from validation_config import preserve_validation_rng
from physics_checkpoints import (
    MOMENTUM_TASKS, CheckpointSummary, resolve_config as resolve_physics_config,
    selection_mode, summarize_native, summarize, write_evaluation_summary, atomic_json, SCHEMA,
)

class DownstreamTrainer():
    
    def _find_available_gpu(self, max_memory_threshold=1000):
        '''Find the first GPU with memory usage below a threshold (in MB).'''
        for gpu_id in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2  # Convert to MB
            if allocated < max_memory_threshold:
                return gpu_id
        return None 


    """ trainer class """
    def __init__(self, params, args):
        
        ''' init vars for distributed training (ddp) and logging'''
        self.root_dir = args.root_dir
        self.global_log_dir = os.path.join(args.root_dir, args.global_log_dir)
        self.config = args.config 
        self.run_num = args.run_num
        self.world_size = 1
        
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])

        self.local_rank = 0
        self.world_rank = 0
        
        if self.world_size > 1: # multigpu, use DDP with standard NCCL backend for communication routines
            dist.init_process_group(backend='nccl',
                                    init_method='env://')
            self.world_rank = dist.get_rank()
            self.local_rank = int(os.environ["LOCAL_RANK"])

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            torch.backends.cudnn.benchmark = True

        self.log_to_screen = (self.world_rank==0)
        if torch.cuda.is_available():
            available_gpu = self._find_available_gpu()
            if available_gpu is not None:
                available_gpu = 0
                torch.cuda.set_device(available_gpu)
                print(f"Using GPU {available_gpu} with memory below threshold.")
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device('cpu')
        
        self.params = params
        input_representation = getattr(params, "input_representation", "center_only")
        if input_representation == "clas12_geometry_v1":
            geometry_stats_path = getattr(params, "geometry_pitch_stats", None)
            if not geometry_stats_path:
                raise ValueError(
                    "clas12_geometry_v1 requires geometry_pitch_stats generated from the v7 pretrain split"
                )
            with open(geometry_stats_path) as stream:
                geometry_stats = json.load(stream)
            if geometry_stats.get("schema") != "clas12_geometry_pitch_stats_v1":
                raise ValueError(f"Unexpected geometry pitch statistics schema: {geometry_stats.get('schema')!r}")
            if geometry_stats.get("split") != "pretrain" or int(geometry_stats.get("count", 0)) < 1:
                raise ValueError("Geometry pitch statistics must contain a non-empty pretrain split")
            self.params["geometry_pitch_mean_cm"] = float(geometry_stats["mean_cm"])
            self.params["geometry_pitch_std_cm"] = float(geometry_stats["std_cm"])
            self.params["geometry_pitch_stats"] = os.path.abspath(geometry_stats_path)
        elif input_representation == "clas12_pos_plus_aux_v1":
            if getattr(params, "embed_method", None) != "pos_plus_aux":
                raise ValueError(
                    "clas12_pos_plus_aux_v1 requires embed_method='pos_plus_aux'"
                )
            if int(getattr(params, "pos_dim", 3)) != 3:
                raise ValueError("clas12_pos_plus_aux_v1 requires pos_dim=3")
            aux_extra_feature = str(getattr(params, "aux_extra_feature", "length"))
            expected_aux_dim = 5 if aux_extra_feature == "both" else 4
            if aux_extra_feature not in {"length", "pitch", "both"}:
                raise ValueError(
                    "aux_extra_feature must be one of 'length', 'pitch', or 'both'"
                )
            if int(getattr(params, "aux_dim", expected_aux_dim)) != expected_aux_dim:
                raise ValueError(
                    f"aux_extra_feature={aux_extra_feature!r} produces {expected_aux_dim} "
                    f"auxiliary columns, but aux_dim={getattr(params, 'aux_dim', None)!r}"
                )
            if getattr(params, "mambaversion", "mamba2") not in {"mamba1", "mamba2"}:
                raise ValueError(
                    "clas12_pos_plus_aux_v1 is currently supported only by "
                    "Mamba1GPT or MambaGPT backbones"
                )
        if getattr(params, "adapter_sample_mode", "event_segment") != "event_segment":
            raise ValueError(
                "track_legacy regression is disabled. Use the v6 event product with "
                "adapter_sample_mode=event_segment."
            )
        # A zero-filled fallback is never meaningful for a regression target.
        self.params["require_reg_target"] = True
        default_stats_path = os.path.join(
            params.stat_dir,
            "regression_target_stats.json",
        )
        stats_path = getattr(params, "regression_target_stats", default_stats_path)
        self.regression_target_stats = load_regression_target_stats(stats_path, params.task)
        self.params["regression_target_stats"] = stats_path
        # The task mapping is the source of truth for the regression head width.
        # Set it here so training and inference construct compatible heads.
        self.params["num_output_classes"] = regression_output_dim(params.task)
        self.regression_loss = resolve_regression_loss(
            params.task, getattr(params, "regression_loss", None)
        )
        if self.regression_loss not in {
            "mse", "mae", "huber", "physical_resolution_l1",
            "physical_resolution_relative_huber",
        }:
            raise ValueError(
                f"Unsupported regression_loss {self.regression_loss!r}; choose "
                "one of ['mse', 'mae', 'huber', 'physical_resolution_l1', "
                "'physical_resolution_relative_huber']"
            )
        self.regression_loss_reference = None
        if self.regression_loss in {
            "physical_resolution_l1", "physical_resolution_relative_huber"
        }:
            reference_path = getattr(params, "regression_loss_reference_stats", None)
            if not reference_path:
                raise ValueError(
                    f"{self.regression_loss} requires regression_loss_reference_stats"
                )
            self.regression_loss_reference = load_regression_loss_reference_stats(
                reference_path,
                params.task,
                momentum_residual=(
                    "relative" if self.regression_loss == "physical_resolution_relative_huber"
                    else "absolute"
                ),
            )
            self.params["regression_loss_reference_stats"] = self.regression_loss_reference["path"]
        self.params["regression_loss"] = self.regression_loss
        self.checkpoint_selection = selection_mode(params.params)
        self.physics_config = (
            resolve_physics_config(self.regression_target_stats["task"], getattr(params, "physics_checkpoint", None))
            if self.regression_target_stats["task"] in MOMENTUM_TASKS else None
        )
        if self.checkpoint_selection == "physics":
            if self.physics_config is None:
                raise ValueError("Physics checkpoint selection requires a momentum/angle task")
            if self.world_size != 1:
                raise NotImplementedError("Physics checkpoint selection currently requires single-process validation")
            self.params["drop_last_test"] = False
        self.params["checkpoint_selection"] = self.checkpoint_selection
        if self.physics_config is not None:
            self.params["physics_checkpoint"] = self.physics_config
        print("running on rank {} with world size {}".format(self.world_rank, self.world_size))



        
    def init_exp_dir(self, exp_dir):
                   
        if self.world_rank==0:
            if not os.path.isdir(exp_dir):
                os.makedirs(exp_dir,exist_ok=True)
                os.makedirs(os.path.join(exp_dir, 'checkpoints/'))
                
        self.params['experiment_dir'] = os.path.abspath(exp_dir)
        self.params['checkpoint_path'] = os.path.join(exp_dir, 'checkpoints/ckpt.tar')

        if self.params.continue_from_best:
            self.params['checkpoint_path'] = os.path.join(exp_dir, 'checkpoints/ckpt_best.tar')

        self.params['resuming'] = True if os.path.isfile(self.params.checkpoint_path) else False
        idx = 0
        logfile = os.path.join(exp_dir, 'performance{}.log'.format(idx))
        
        if self.world_rank==0:    
            while os.path.exists(logfile):
                idx += 1
                logfile = os.path.join(exp_dir, 'performance{}.log'.format(idx))
                
        if dist.is_initialized():
            dist.barrier()
        
        self.logfile = logfile

        if self.world_rank==0:            
            with open(self.logfile, 'w') as f:
                f.write('Initialized at: {}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
    
            # Preparing global log directory
            if not os.path.isdir(self.global_log_dir):
                os.makedirs(self.global_log_dir)   
                
            global_details = self.parse_exp_details(
                self.params.params,
                partial=['data_version', 'limit_size', 'model_version'],
                globalfile=True,
            )
            global_filename = 'config_{}_run_{}_{}.csv'.format(
                self.config,
                self.run_num,
                global_details,
            )
            # Long campaign identifiers (for example versioned pretrained
            # backbones plus label budgets) can exceed the per-filename
            # NAME_MAX limit even though the containing path is valid. Keep
            # the historical descriptive form when it fits; otherwise retain
            # a readable prefix and add a digest of the complete identity.
            if len(global_filename.encode()) > 240:
                digest = hashlib.sha256(global_filename.encode()).hexdigest()[:12]
                suffix = f"_{digest}.csv"
                prefix = f"config_{self.config}_run_{self.run_num}"
                max_prefix_bytes = 240 - len(suffix.encode())
                prefix = prefix.encode()[:max_prefix_bytes].decode(errors="ignore")
                global_filename = f"{prefix}{suffix}"
            self.globalfile = os.path.join(self.global_log_dir, global_filename)
            print(self.globalfile)

        if dist.is_initialized():
            dist.barrier()
        
        if self.world_rank == 0 and not os.path.exists(self.globalfile):
            with open(self.globalfile, 'w') as f:
                pass
        if dist.is_initialized():
            dist.barrier()  
        
    def log_infile(self, log):
        with open(self.logfile, "a") as f:
            f.write("{}\n".format(log))

    def log_globalfile(self, split, step, loss, lr):
        with open(self.globalfile, "a") as f:
            f.write("{},{},{},{}\n".format(split, step, loss, lr))

    def finish_training(self):
        with open(self.finisher, 'w') as f:
            f.write(' ')
        raise FinishedTrainingError
    
    def parse_exp_details(self, D, partial=None, globalfile = False):
        """
        D: a dictionary listing parameters
        partial: a list of columns of interest
        """
        
        if globalfile:
            if partial is None:
                out = ','.join(['{}:{}'.format(a, b) for a,b in D.items()])
            else:
                out = ','.join(['{}:{}'.format(a, b) for a,b in D.items() if a in partial])
        else:
            out = 'Important Details:\n' + ''.join(['{}: {}\n'.format(a, b) for a,b in D.items()])
        return out

    def get_bin_index(self, seq_length):
        """Find the appropriate bin for a given sequence length."""
        for i in range(len(self.bins) - 1):
            if self.bins[i] <= seq_length < self.bins[i + 1]:
                return i
        return len(self.bins) - 2  # Assign to last bin if out of range

    def update_moving_average(self, bin_idx, loss_value):
        """Update exponential moving average of loss per bin."""
        self.loss_moving_avg[bin_idx] = (
            self.smoothing_factor * self.loss_moving_avg[bin_idx] +
            (1 - self.smoothing_factor) * loss_value
        )

    def compute_inverse_loss_weights(self):
        """Compute inverse loss weights for each bin."""
        weights = {i: 1 / (self.loss_moving_avg[i] + self.epsilon) for i in self.loss_moving_avg}
        total_weight = sum(weights.values())
        return {i: weights[i] / total_weight for i in weights}  # Normalize weights



    def cleanup(self):
        # 1) remove hooks
        for hook_list in ("fwd_hooks", "bwd_hooks"):
            for h in getattr(self, hook_list, []):
                h.remove()

        # 2) break references to big objects
        for obj in ("model", "down_model",
                    "optimizer", "down_optimizer",
                    "scheduler", "down_scheduler",
                    "train_data_loader", "val_data_loader"):
            if hasattr(self, obj):
                delattr(self, obj)

        # 3) empty CUDA cache and run GC
        torch.cuda.empty_cache()
        gc.collect()

        # 4) tear down DDP if we set it up
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        print("Cleanup complete. All resources released.")

    def launch(self):
        print(self.root_dir, self.config, self.run_num)
        exp_dir = os.path.join(*[self.root_dir, self.config, self.run_num])
        self.init_exp_dir(exp_dir)

        self.params['global_batch_size'] = self.params.batch_size
        self.params['local_batch_size'] = int(self.params.batch_size//self.world_size)
        self.params['global_valid_batch_size'] = self.params.valid_batch_size
        self.params['local_valid_batch_size'] = int(self.params.valid_batch_size//self.world_size)

        print('batch size: ', self.params['global_batch_size'])
        print('local batch size: ', self.params['local_batch_size'])

        self.log_infile(self.parse_exp_details(self.params.params))       

        # get the pretrained model
        self.klen = self.params.klen
        if self.params.mambaversion == 'mamba1':
            self.model = Mamba1GPT(embed_dim=self.params.embed_dim, num_layers=self.params.num_layers_backbone,
                                d_state=self.params.d_state, d_conv=4, expand=2, klen=self.klen, dropout=self.params.dropout,
                                embed_method=self.params.embed_method, pe_method=self.params.pe_method,
                                pos_dim=getattr(self.params, 'pos_dim', 3), aux_dim=getattr(self.params, 'aux_dim', 4))
        elif self.params.mambaversion == 'longformer':
            self.model = LongformerGPT(
                embed_dim=self.params.embed_dim,
                num_layers=self.params.num_layers_backbone,
                num_heads=self.params.num_heads_backbone,
                window_size=getattr(self.params, 'window_size', 256),
                mlp_ratio=getattr(self.params, 'mlp_ratio', 2.0),
                klen=self.klen,
                dropout=self.params.dropout,
                embed_method=self.params.embed_method,
                pe_method=self.params.pe_method
            )
        elif self.params.mambaversion == 'linformer':
            self.model = LinformerGPT(
                embed_dim=self.params.embed_dim,
                num_layers=self.params.num_layers_backbone,
                num_heads=self.params.num_heads_backbone,
                seq_len=getattr(self.params, 'seq_len', 512),
                proj_dim=getattr(self.params, 'proj_dim', 256),
                mlp_ratio=getattr(self.params, 'mlp_ratio', 2.0),
                klen=self.klen,
                dropout=self.params.dropout,
                embed_method=self.params.embed_method,
                pe_method=self.params.pe_method
            )
        else:
            self.model = MambaGPT(embed_dim=self.params.embed_dim, num_layers=self.params.num_layers_backbone,
                    d_state=self.params.d_state, d_conv=4, expand=2, klen=self.klen, dropout=self.params.dropout,
                    embed_method=self.params.embed_method, pe_method=self.params.pe_method,
                    pos_dim=getattr(self.params, 'pos_dim', 3), aux_dim=getattr(self.params, 'aux_dim', 4))
        

        def initialize_mamba2(model, d_state, embed_dim):
            """ Properly initializes Mamba v2 to ensure stable learning. """

            with torch.no_grad():
                for name, param in model.named_parameters():

                    if "lin_B" in name:
                        param.normal_(mean=0.0, std=(d_state / embed_dim)**0.5)

                    elif "lin_C" in name:
                        param.normal_(mean=0.0, std=(1.0 / (embed_dim*d_state))**0.5)

                    elif "norm.weight" in name:
                        init.ones_(param)

                    # Bias Terms
                    elif "bias" in name:
                        init.zeros_(param)

            if self.world_rank == 0:
                print(f"✅ Mamba v2 Model Initialized")

        # Set model dimension variables for optimizer (used by all models)
        Nu = self.params.embed_dim
        Nx = getattr(self.params, 'd_state', 16)  # Default to 16 for non-Mamba models

        # Only initialize Mamba models with Mamba-specific initialization
        if self.params.mambaversion in ['mamba1', 'mamba2']:
            initialize_mamba2(self.model, Nx, Nu)
        else:
            if self.world_rank == 0:
                print(f"✅ {self.params.mambaversion.capitalize()} Model Initialized")      
                
        self.model = self.model.to(self.device)
        print('Nparams: ', count_parameters(self.model))

        # distributed wrapper for data parallel
        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model,
                                                device_ids=[self.local_rank],
                                                output_device=[self.local_rank],
                                                find_unused_parameters=True)

            

        # set an optimizer and learning rate scheduler   
        params_a   = []
        params_b   = []
        params_c   = []
        params_else= []

        for name, p in self.model.named_parameters():
            if "A_log" in name:
                params_a.append(p)   # might do LR ~ Nu
            elif "lin_B" in name:
                params_b.append(p)   # might do LR ~ Nx / sqrt(Nu)
            elif "lin_C" in name:
                params_c.append(p)   # might do LR ~ sqrt(Nu) / Nx
            else:
                params_else.append(p)
                
        self.optimizer = torch.optim.AdamW([
            {"params": params_a,   "lr": self.params.min_lr * Nu},                   # e.g. for A
            {"params": params_b,   "lr": self.params.min_lr * Nx / (Nu**0.5)},       # e.g. for B
            {"params": params_c,   "lr": self.params.min_lr * (Nu**0.5) / Nx},       # e.g. for C
            {"params": params_else,"lr": self.params.min_lr},
        ], weight_decay=0.1, betas=(0.9, 0.95))

        self.scaler = torch.amp.GradScaler('cuda') 
        
        self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer,
                                          first_cycle_steps=self.params.total_steps,
                                          max_lr=self.params.max_lr,
                                          min_lr=self.params.min_lr,
                                          warmup_steps=self.params.warmup_steps)

        
        # get the dataloaders
        self.train_data_loader, self.train_sampler, self.val_data_loader, _ = get_data_loader(self.params, 
                                                                                              dist.is_initialized())

        # set loss functions
        self.loss_func = nn.MSELoss(reduction='none')
        self.centroid_loss_func = nn.MSELoss(reduction='none')
        self.loss_func_eval = nn.MSELoss(reduction='none')

        # checkpointing
        self.iters = 0
        self.startEpoch = 0
        self.resumed = False

        ##### Pretraining checkpoint
        if self.params.pretrained_ckpt is not None:
            print("Loading checkpoint %s" % self.params.pretrained_ckpt)
            self.restore_checkpoint(self.params.pretrained_ckpt, load_optimizer_state=False)
            self.resumed = True
        else:
            print("No pretrained checkpoint provided; using randomly initialized backbone.")
            self.resumed = False

        self.startEpoch = 0
        self.epoch = self.startEpoch
        self.logs = {}

        # 
        #  training
        #self.train()

    def inference(self, checkpoint_path, pretrain=True, logfile=None):
        """Initialize model and load weights for inference"""
        # 1. Initialize model architecture
        #self.down_model = MambaHead(input_dim=self.params.embed_dim, num_layers=1, num_output_dim = self.params.num_output_classes,
        #                          d_state=64, d_conv=4, expand=2, num_feature_layers=self.params.num_layers_backbone,
        #                          num_embedder_layers= self.params.num_embedder_layers, 
        #                          ).to(self.device)

        #if self.params.use_attention_head:
        #    self.down_model = AttentionHead(input_dim=self.params.embed_dim, num_layers=1, num_output_dim = self.params.num_output_classes,
        #                  num_heads = 4, num_feature_layers=self.params.num_layers_backbone,
        #                  num_embedder_layers= self.params.num_embedder_layers, 
        #                  ).to(self.device)
        
        #else:
        self.down_model = MambaTrackRegressionHead(input_dim=self.params.embed_dim, num_layers=1, num_output_dim=self.params.num_output_classes, d_state=64, d_conv=4, expand=2, num_feature_layers=self.params.num_layers_backbone, num_embedder_layers=self.params.num_embedder_layers, pooling=getattr(self.params, "pooling", "mean"), embed_method=self.params.embed_method, pe_method=self.params.pe_method, target_mean=self.regression_target_stats["mean"], target_std=self.regression_target_stats["std"], input_representation=getattr(self.params, "input_representation", "center_only"), geometry_pitch_mean_cm=getattr(self.params, "geometry_pitch_mean_cm", None), geometry_pitch_std_cm=getattr(self.params, "geometry_pitch_std_cm", None), pos_dim=getattr(self.params, "pos_dim", 3), aux_dim=getattr(self.params, "aux_dim", 4)).to(self.device)

    
        total_params = sum(p.numel() for p in self.down_model.parameters())
        print(f"Total parameters in down_model: {total_params}")
        self.down_optimizer = optim.AdamW(self.down_model.parameters(), 
                                         lr=self.params.max_lr, # Mamba: Linear-Time Sequence Modeling with Selective State Spaces
                                         weight_decay=0.0001) 
        
        torch.nn.utils.clip_grad_norm_(self.down_model.parameters(), max_norm=1.0)


        self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer,
                                          first_cycle_steps=self.params.total_steps,
                                          max_lr=self.params.max_lr,
                                          min_lr=self.params.min_lr,
                                          warmup_steps=self.params.warmup_steps)

        # Add safe global class
        from ruamel.yaml.scalarfloat import ScalarFloat
        torch.serialization.add_safe_globals([ScalarFloat])
    
        try:
            self.load_checkpoint(checkpoint_path, inference=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load regression checkpoint: {checkpoint_path}"
            ) from exc
        
        self.down_model.eval()
        self.model.eval()
        print(f"✅ Model loaded from {checkpoint_path}")

        output_list = []
        target_list = []
        target_valid_list = []
        truth_xyz_list = []
        loss_list = []

        with torch.no_grad():  # Disable gradient calculation
            for i, inputdict in enumerate(tqdm(self.val_data_loader)):
                #if i> 20000:
                #    break
                self.iters += 1
                grouped = inputdict['points'].to(self.device)  # B X N X C
                b, c = grouped.size(0), grouped.size(-1)
                grouped = grouped.reshape(b, -1, c).to(self.device) # B X N X C
                mask = grouped[..., 0] != -100 # B X N
                reg = inputdict['reg_target'].to(self.device)  # B X N X 8
                pid = inputdict['pid_target'].to(self.device)  # B X N tensor containing particle IDs
                #mid = inputdict['mid_target'].to(self.device)  # B X N tensor containing mother IDs

                trackinfo_noiselabel_dict = get_trackinfo_noiselabel(reg)
                noise_labels = trackinfo_noiselabel_dict["noise_labels"]
                pid_label_dict = get_pidlabel(pid)
                pid_class = pid_label_dict["pid_class"]  # B X N tensor with particle class information
                #weak_decay_label_dict = get_weakdecaylabel(mid)
                #weak_decay_class = weak_decay_label_dict["weak_decay_class"]  # B X N tensor with weak decay labels
                #if self.params.task == "pid":
                #    targets = {
                #        'labels': pid_class,  # B X N tensor with particle class information
                #    }
                #elif self.params.task == "nid":
                #    targets = {
                #        'labels': noise_labels,  # B X N tensor with noise id
                #    }
                target_segment_mask = inputdict.get('target_segment_mask')
                if target_segment_mask is not None:
                    target_segment_mask = target_segment_mask.to(self.device).bool()
                targets = self.build_regression_targets(reg, mask, target_segment_mask)

                self.down_optimizer.zero_grad()
                geometry_kwargs = self._geometry_context_kwargs(inputdict, pretrain)
                model_input = self._representation_input(grouped, inputdict)
                if pretrain:
                    with torch.no_grad():
                        _, pre_embed, _ = self.model(model_input, return_z = True)
                    #feature = torch.stack(pre_embed).mean(0)
                    feature = torch.stack(pre_embed)
                    #print('feature: ', feature.size())
                    pred_dict = self.down_model(model_input, feature, pretrain=pretrain, padding_mask=mask)

                else:
                    pred_dict = self.down_model(model_input, feature=None, padding_mask=mask, **geometry_kwargs)

                pred = pred_dict['pred_regression']
                outputs = {
                    "pred": pred,
                }

                losses = masked_regression_loss(
                    outputs=outputs,
                    targets=targets,
                    option=self.regression_loss,
                    angular_indices=self.regression_target_stats["angular_indices"],
                    target_std=self.regression_target_stats["std"],
                    target_mean=self.regression_target_stats["mean"],
                    phi_pairs=self.regression_target_stats.get("phi_pairs", ()),
                    physical_scales=self.regression_loss_reference,
                )
                truth_xyz_list.append(self.physics_truth_xyz(reg, mask, target_segment_mask).cpu().numpy())
                target_list.append(targets['target'].cpu())
                target_valid_list.append(targets['target_valid'].cpu())
                output_list.append(outputs['pred'].cpu())
               
                loss = losses['loss']
                loss_list.append(loss.cpu().numpy())

        all_predictions = torch.cat(output_list, dim=0)
        all_targets = torch.cat(target_list, dim=0)
        all_target_valid = torch.cat(target_valid_list, dim=0)
        avg_loss = np.mean(loss_list)
        mae = torch.mean(
            torch.abs(all_predictions[all_target_valid] - all_targets[all_target_valid])
        ).item()
        if self.physics_config is not None:
            normalizer = self.down_model.target_normalizer
            pred_native = normalizer.denormalize(all_predictions.to(self.device)).cpu().numpy()
            truth_native = normalizer.denormalize(all_targets.to(self.device)).cpu().numpy()
            truth_native[~all_target_valid.numpy()] = np.nan
            write_evaluation_summary(Path(logfile).parent / (Path(logfile).stem + "_physics"),
                pred_native, truth_native, self.regression_target_stats["task"], self.physics_config,
                truth_xyz=np.concatenate(truth_xyz_list), checkpoint=checkpoint_path,
                metadata=self.loaded_checkpoint_metadata)
        header = "Avg_Loss MAE\n"
        values = f"{avg_loss:.4f} {mae:.4f}\n"

        # 5) Write (or append) to the log file
        with open(logfile, "w") as f:
            f.write(header)
            f.write(values)

        


    def train(
        self,
        pretrain=True,
        train_from_checkpoint=False,
        checkpoint_path=None,
        optuna_trial=None,
        metrics_callback=None,
    ):
        ###%%%%%%%
        # Debugging
        self.fwd_hooks = register_fine_grained_forward_hooks(self.model)
        self.bwd_hooks = register_param_backward_nan_hooks(self.model)
        ###%%%%%%%%

        def initialize_mamba2(model, num_layers, num_residuals=1):
            """ Properly initializes Mamba v2 to ensure stable learning. """
            for name, param in model.named_parameters():
            
                # Stable State-Space Matrix (A_t)
                if "A" in name:  
                    init.uniform_(param, -0.1 / num_layers, 0.1 / num_layers)
        
                # State Decay D (Ensure nonzero values)
                elif "D" in name:
                    init.normal_(param, mean=0.1, std=0.02)
        
                # Convolution Weights
                elif "conv1d.weight" in name:
                    init.kaiming_uniform_(param, mode="fan_in", nonlinearity="linear")
        
                # Projection Layers (Mapping Activations)
                elif "out_proj.weight" in name or "in_proj.weight" in name:
                    init.xavier_uniform_(param, gain=1.0 / (num_layers ** 0.5))
        
                # Normalization Layers (LayerNorm, RMSNorm)
                elif "norm.weight" in name:
                    init.ones_(param)
        
                # Bias Terms
                elif "bias" in name:
                    init.zeros_(param)
        
            print(f"✅ Mamba v2 Model Initialized (Safe Scaling for {num_layers} Layers")
                
        #self.down_model = MambaHead(input_dim=self.params.embed_dim, num_layers=2, 
        #                          d_state=self.params.d_state, d_conv=4, expand=2, num_feature_layers=self.params.num_layers_backbone, num_output_dim = self.params.max_gt_classes).to(self.device)
        #self.down_model = MambaHead(input_dim=self.params.embed_dim, num_layers=1, num_output_dim = self.params.num_output_classes,
        #                          d_state=64, d_conv=4, expand=2, num_feature_layers=self.params.num_layers_backbone,
        #                          num_embedder_layers= self.params.num_embedder_layers, 
        #                          ).to(self.device)

        self.down_model = MambaTrackRegressionHead(
            input_dim=self.params.embed_dim,
            num_layers=1,
            num_output_dim=self.params.num_output_classes,
            d_state=64,
            d_conv=4,
            expand=2,
            num_feature_layers=self.params.num_layers_backbone,
            num_embedder_layers=self.params.num_embedder_layers,
            pooling=getattr(self.params, "pooling", "mean"),
            embed_method=self.params.embed_method,
            pe_method=self.params.pe_method,
            dropout=float(getattr(self.params, "dropout", 0.0)),
            target_mean=self.regression_target_stats["mean"],
            target_std=self.regression_target_stats["std"],
            input_representation=getattr(self.params, "input_representation", "center_only"),
            geometry_pitch_mean_cm=getattr(self.params, "geometry_pitch_mean_cm", None),
            geometry_pitch_std_cm=getattr(self.params, "geometry_pitch_std_cm", None),
            pos_dim=getattr(self.params, "pos_dim", 3),
            aux_dim=getattr(self.params, "aux_dim", 4),
        ).to(self.device)

        #print number of parameters in the model
        total_params = sum(p.numel() for p in self.down_model.parameters())
        print(f"Total parameters in down_model: {total_params}")
        
        initialize_mamba2(self.down_model, 3, num_residuals=1)

        self.down_optimizer = optim.AdamW(self.down_model.parameters(), 
                                         lr=self.params.max_lr, # Mamba: Linear-Time Sequence Modeling with Selective State Spaces
                                         weight_decay=float(getattr(self.params, "adapter_weight_decay", 0.0001))) 
        
        self.grad_clip_value = float(getattr(self.params, "grad_clip_value", 1.0))
        torch.nn.utils.clip_grad_norm_(self.down_model.parameters(), max_norm=self.grad_clip_value)

        scheduler_mode = str(getattr(self.params, "scheduler_mode", "cosine_restarts"))
        max_optimizer_steps = getattr(self.params, "max_optimizer_steps", None)
        scheduler_steps = int(getattr(
            self.params,
            "scheduler_first_cycle_steps",
            getattr(self.params, "first_cycle_steps", max_optimizer_steps or 200),
        ))
        anneal_steps = int(getattr(self.params, "scheduler_anneal_steps", 0))
        if scheduler_mode == "cosine_hold" and anneal_steps <= 0:
            raise ValueError(
                "scheduler_mode='cosine_hold' requires a positive "
                "scheduler_anneal_steps"
            )
        if (
            scheduler_mode == "cosine_hold"
            and max_optimizer_steps is not None
            and anneal_steps > int(max_optimizer_steps)
        ):
            raise ValueError(
                "scheduler_anneal_steps cannot exceed max_optimizer_steps for "
                "scheduler_mode='cosine_hold'"
            )
        warmup_reference_steps = anneal_steps if scheduler_mode == "cosine_hold" else scheduler_steps
        warmup_steps = getattr(self.params, "warmup_steps", 20)
        if hasattr(self.params, "warmup_fraction"):
            warmup_steps = max(
                1,
                int(float(self.params.warmup_fraction) * warmup_reference_steps),
            )

        if scheduler_mode == "cosine_restarts":
            self.down_scheduler = CosineAnnealingWarmupRestarts(
                self.down_optimizer,
                first_cycle_steps=scheduler_steps,
                max_lr=self.params.max_lr,
                min_lr=self.params.min_lr,
                warmup_steps=int(warmup_steps),
            )
        elif scheduler_mode == "cosine_hold":
            self.down_scheduler = CosineAnnealingWarmupThenHold(
                self.down_optimizer,
                anneal_steps=anneal_steps,
                max_lr=self.params.max_lr,
                min_lr=self.params.min_lr,
                warmup_steps=int(warmup_steps),
            )
        else:
            raise ValueError(
                "scheduler_mode must be 'cosine_restarts' or 'cosine_hold'; "
                f"got {scheduler_mode!r}"
            )


        # Add safe global class
        from ruamel.yaml.scalarfloat import ScalarFloat
        torch.serialization.add_safe_globals([ScalarFloat])
    
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.params.checkpoint_dir, exist_ok=True)

        log_file_path = os.path.abspath(
            os.path.join(self.params.checkpoint_dir, self.params.log_file_name)
        )
        self.params["training_log_path"] = log_file_path

        checkpoint_file_name = getattr(
            self.params,
            "checkpoint_file_name",
            self.params.log_file_name.split('.')[0] + '_checkpoint.pth',
        )
        self.params["trained_checkpoint_path"] = None
        
        if self.log_to_screen:
            print("Starting training loop...")
            with open(log_file_path, "w") as f:
                if max_optimizer_steps is None:
                    f.write("Epoch\tTrain_Loss\tVal_Loss\tTime\n")
                else:
                    f.write("Step\tEpoch\tTrain_Loss\tVal_Loss\tLR\tTime\n")

        self.best_loss = np.inf
        self.best_step = None
        self.best_epoch = None
        self.global_step = 0
        self.best_ARI = 0
        self.down_results = {'epoch': 0, 'train': [], 'val': [], 'precision':[], 'recall':[], 'accuracy': []}
        # early stopping
        self.patience, self.min_delta, self.warmup_steps = get_early_stopping_config(
            self.params,
            default_patience=5,
            default_min_delta=1e-4,
            default_warmup_steps=20,
        )
        self.stagnation_counter = 0
        self.best_loss_step = None
        self.best_loss_epoch = None
        self.last_validation_step = None
        self.early_stopping_min_steps = int(getattr(self.params, "early_stopping_min_steps", self.warmup_steps))
        if train_from_checkpoint:
            try:
                self.load_checkpoint(checkpoint_path, inference=False)
            except Exception as e:
                raise RuntimeError(f"Checkpoint loading failed: {checkpoint_path}") from e

            self.down_model.eval()
            print(f"✅ Model loaded from {checkpoint_path}")

        if self.checkpoint_selection == "physics":
            self._initialize_physics_history(checkpoint_file_name)
            self._preflight_validation_support()
        

        if getattr(self.params, "loss_reweight", False):
            self.loss_bin = pickle_load(f"{self.params.stat_dir}/loss_bin_pp.pkl")
            self.loss_weight = pickle_load(f"{self.params.stat_dir}/loss_weight_pp.pkl")
        else:
            self.loss_bin = None
            self.loss_weight = None

        if max_optimizer_steps is not None:
            self._train_by_optimizer_step(
                pretrain=pretrain,
                log_file_path=log_file_path,
                checkpoint_file_name=checkpoint_file_name,
                optuna_trial=optuna_trial,
                metrics_callback=metrics_callback,
            )
            self._finish_checkpoint_selection(pretrain, checkpoint_file_name)
            return
        
        for epoch in range(self.startEpoch, self.params.max_epochs):
            self.down_results['epoch'] = epoch
            self.down_results['train'] = []
            self.down_results['val'] = []
            self.down_results['precision'] = []
            self.down_results['recall'] = []
            self.down_results['accuracy'] = []
            self.epoch = epoch
            if dist.is_initialized():
                # shuffles data before every epoch
                self.train_sampler.set_epoch(epoch)
                
            self.resumed = False
                
            self.starttime = time.time()
            self.downstream_end_to_end_one_epoch(pretrain = pretrain)
            train_epoch_loss = np.mean(self.down_results['train'])
            val_epoch_loss = 0

            if epoch % 1 == 0:
                val_epoch_loss = self.validate_end_to_end_one_epoch(pretrain=pretrain)
            epoch_time = time.time() - self.starttime
            with open(log_file_path, "a") as f:
                f.write(
                    f"{epoch}\t{train_epoch_loss:.8f}\t{val_epoch_loss:.8f}\t{epoch_time:.2f}\n"
                )
            epoch_loss = val_epoch_loss
            print('Epoch: ', epoch, 'Loss: ', train_epoch_loss)
            self._record_validation_result(epoch_loss, checkpoint_file_name, epoch, self.global_step,
                                           allow_stagnation=epoch >= self.warmup_steps)
            if self.stagnation_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}: selected checkpoint unchanged for {self.patience} checks.")
                break
            self.down_scheduler.step()
            if metrics_callback is not None:
                metrics_callback({
                    "step": self.global_step,
                    "epoch": epoch,
                    "train/loss": float(train_epoch_loss),
                    "val/loss": float(val_epoch_loss),
                    "lr": float(self._current_lr()),
                    "best/val_loss": float(self.best_loss),
                    "best/step": self.best_step,
                })
        self._finish_checkpoint_selection(pretrain, checkpoint_file_name)


    def build_regression_targets(self, reg, hit_mask, target_segment_mask=None):
        """
        reg_target columns:
            0: px
            1: py
            2: pz
            3: vtx_x
            4: vtx_y
            5: vtx_z
            6: energy
        """
        per_hit_target = transform_regression_target_torch(reg, self.params.task)

        # Regression truth is stored once per hit, although the prediction is
        # event-level. Collapse valid, finite copies to one target per event.
        effective_mask = hit_mask
        if target_segment_mask is not None:
            effective_mask = effective_mask & target_segment_mask
        valid = effective_mask.unsqueeze(-1) & torch.isfinite(per_hit_target)
        counts = valid.sum(dim=1)
        target = torch.where(
            valid, per_hit_target, torch.zeros_like(per_hit_target)
        ).sum(dim=1)
        target = target / counts.clamp_min(1)
        if self.regression_target_stats["task"] == "phi":
            # Average repeated angle labels across the branch cut correctly.
            safe = torch.where(valid, per_hit_target, torch.zeros_like(per_hit_target))
            sine = torch.where(valid, torch.sin(safe), 0.0).sum(dim=1)
            cosine = torch.where(valid, torch.cos(safe), 0.0).sum(dim=1)
            target = torch.atan2(sine, cosine)
        down_model = self.down_model
        if isinstance(down_model, torch.nn.parallel.DistributedDataParallel):
            down_model = down_model.module
        target = down_model.target_normalizer.normalize(target)

        return {
            "target": target,
            "target_valid": counts > 0,
        }

    def _current_lr(self):
        return self.down_optimizer.param_groups[0]["lr"]

    def _geometry_context_kwargs(self, inputdict, pretrain):
        """Move v7 typed sidecars alongside their already-serialized point rows."""
        input_representation = getattr(self.params, "input_representation", "center_only")
        if input_representation in {"center_only", "clas12_pos_plus_aux_v1"}:
            return {}
        if pretrain:
            raise ValueError("clas12_geometry_v1 is intentionally adapter-only")
        try:
            token_context = inputdict['token_context'].to(self.device, dtype=torch.long)
            geometry_context = inputdict['geometry_context'].to(self.device, dtype=torch.float32)
        except KeyError as exc:
            raise KeyError(
                "Geometry representation requested but data loader did not return v7 context sidecars"
            ) from exc
        return {"token_context": token_context, "geometry_context": geometry_context}

    def _representation_input(self, grouped, inputdict):
        """Return the backbone/adapter token tensor for the selected input contract.

        ``clas12_pos_plus_aux_v1`` reproduces Mike's v7 preprocessing from our
        typed sidecars: normalized center ``[eta, phi, r]`` plus strip direction
        in the local radial/tangential/z frame and a normalized length or pitch.
        The points and sidecars have already undergone the same serialization in
        ``TPCBatchDataset``; this method only packages their aligned rows.
        """
        if getattr(self.params, "input_representation", "center_only") != "clas12_pos_plus_aux_v1":
            return grouped
        if grouped.size(-1) != 3:
            raise ValueError(
                "clas12_pos_plus_aux_v1 expects normalized center points with three columns"
            )
        try:
            geometry_context = inputdict["geometry_context"].to(
                self.device, dtype=grouped.dtype
            )
        except KeyError as exc:
            raise KeyError(
                "clas12_pos_plus_aux_v1 requires v7 geometry_context sidecars"
            ) from exc
        if geometry_context.shape[:2] != grouped.shape[:2] or geometry_context.size(-1) != 11:
            raise ValueError(
                "geometry_context must be row-aligned (B, N, 11) with serialized points; "
                f"got points={tuple(grouped.shape)}, geometry={tuple(geometry_context.shape)}"
            )

        # The loader's phi normalization is (phi + pi) / (2*pi).  Constructing
        # this frame from serialized normalized phi is algebraically identical to
        # Mike's raw-x/raw-y calculation, while preserving our typed sidecars.
        phi = grouped[..., 1] * (2.0 * math.pi) - math.pi
        r_hat = torch.stack((torch.cos(phi), torch.sin(phi), torch.zeros_like(phi)), dim=-1)
        phi_hat = torch.stack((-torch.sin(phi), torch.cos(phi), torch.zeros_like(phi)), dim=-1)
        strip_direction = geometry_context[..., 6:9]
        s_r = (strip_direction * r_hat).sum(dim=-1, keepdim=True)
        s_phi = (strip_direction * phi_hat).sum(dim=-1, keepdim=True)
        s_z = strip_direction[..., 2:3]

        aux_extra_feature = str(getattr(self.params, "aux_extra_feature", "length"))
        length = geometry_context[..., 9:10] / 45.0
        pitch = geometry_context[..., 10:11] / 0.10
        if aux_extra_feature == "length":
            scalar = length
        elif aux_extra_feature == "pitch":
            scalar = pitch
        elif aux_extra_feature == "both":
            scalar = torch.cat((length, pitch), dim=-1)
        else:
            raise RuntimeError(f"Unexpected aux_extra_feature={aux_extra_feature!r}")

        packed = torch.cat((grouped, s_r, s_phi, s_z, scalar), dim=-1)
        expected_width = int(getattr(self.params, "pos_dim", 3)) + int(
            getattr(self.params, "aux_dim", 4)
        )
        if packed.size(-1) != expected_width:
            raise ValueError(
                f"Packed CLAS12 token has width {packed.size(-1)}, but pos_dim + aux_dim "
                f"is {expected_width}; check aux_extra_feature and aux_dim"
            )

        # Preserve the established all--100 padding contract for both Mamba
        # backbones (which replace it with zero) and AdapterOnly masking.
        padding_mask = grouped[..., 0] != -100
        return torch.where(
            padding_mask.unsqueeze(-1), packed, torch.full_like(packed, -100.0)
        )

    def _log_geometry_branch_rms(self, pred_dict):
        norms = pred_dict.get("geometry_branch_rms")
        if norms is None or self.world_rank != 0:
            return
        if self.global_step not in {0, 1} and self.global_step % 1000 != 0:
            return
        formatted = ", ".join(f"{name}={value.item():.3f}" for name, value in norms.items())
        print(f"[geometry branch RMS] step={self.global_step}: {formatted}")

    def _train_one_batch(self, inputdict, pretrain=False):
        grouped = inputdict['points'].to(self.device)  # B X N X C
        b, c = grouped.size(0), grouped.size(-1)
        grouped = grouped.reshape(b, -1, c).to(self.device) # B X N X C
        mask = grouped[..., 0] != -100 # B X N
        reg = inputdict['reg_target'].to(self.device)  # B X N X 8

        target_segment_mask = inputdict.get('target_segment_mask')
        if target_segment_mask is not None:
            target_segment_mask = target_segment_mask.to(self.device).bool()
        targets = self.build_regression_targets(reg, mask, target_segment_mask)

        self.down_optimizer.zero_grad()
        geometry_kwargs = self._geometry_context_kwargs(inputdict, pretrain)
        model_input = self._representation_input(grouped, inputdict)
        if pretrain:
            with torch.no_grad():
                _, pre_embed, _ = self.model(model_input, return_z=True)
            feature = torch.stack(pre_embed)
            pred_dict = self.down_model(model_input, feature, pretrain=pretrain, padding_mask=mask)
        else:
            pred_dict = self.down_model(model_input, feature=None, padding_mask=mask, **geometry_kwargs)
        self._log_geometry_branch_rms(pred_dict)

        pred = pred_dict["pred_regression"]  # B x num_output_classes

        outputs = {
            "pred": pred,
        }

        losses = masked_regression_loss(
            outputs=outputs,
            targets=targets,
            option=self.regression_loss,
            angular_indices=self.regression_target_stats["angular_indices"],
            target_std=self.regression_target_stats["std"],
            target_mean=self.regression_target_stats["mean"],
            phi_pairs=self.regression_target_stats.get("phi_pairs", ()),
            physical_scales=self.regression_loss_reference,
        )

        loss = losses['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.down_model.parameters(),
            max_norm=self.grad_clip_value,
            norm_type=2.0,
        )

        self.down_optimizer.step()
        return loss.item()

    def _initialize_physics_history(self, filename):
        self.selected_checkpoint_path = Path(self.params.checkpoint_dir, filename).resolve()
        self.selected_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        directory = tempfile.mkdtemp(prefix=self.selected_checkpoint_path.stem + "_physics_",
                                     dir=self.selected_checkpoint_path.parent)
        self.physics_history = CheckpointSummary(self.physics_config, directory)
        # A fresh selection must never accidentally expose a previous run's alias.
        if self.selected_checkpoint_path.exists():
            shutil.move(str(self.selected_checkpoint_path), str(Path(directory, "previous_selected.pth")))
        self.params["physics_checkpoint_summary"] = str(Path(directory, "checkpoint_summary.json"))
        self.physics_report_pointer = self.selected_checkpoint_path.with_name(
            self.selected_checkpoint_path.stem + "_physics_report.json")
        self.physics_history.write(plots=False)
        self._write_physics_pointer("awaiting_validation")

    def _write_physics_pointer(self, status):
        atomic_json(self.physics_report_pointer, {
            "schema": SCHEMA, "task": self.regression_target_stats["task"],
            "checkpoint_summary": self.params.physics_checkpoint_summary,
            "validation_support": getattr(self.params, "validation_support", None),
            "selection_status": status,
        })

    def _preflight_validation_support(self):
        """Count the actual fixed validation truth bins before optimizer steps."""
        truths, xyz_rows = [], []
        cap = getattr(self.params, "max_val_batches", None)
        # DataLoader iterator construction consumes CPU RNG even without shuffle.
        with torch.no_grad(), preserve_validation_rng(self.val_data_loader):
            for i, batch in enumerate(tqdm(self.val_data_loader, desc="Validation bin occupancy")):
                if cap is not None and i >= cap:
                    break
                points = batch['points'].to(self.device)
                points = points.reshape(points.size(0), -1, points.size(-1))
                mask = points[..., 0] != -100
                reg = batch['reg_target'].to(self.device)
                segment = batch.get('target_segment_mask')
                if segment is not None:
                    segment = segment.to(self.device).bool()
                targets = self.build_regression_targets(reg, mask, segment)
                truth = self.down_model.target_normalizer.denormalize(targets['target']).cpu().numpy()
                truth[~targets['target_valid'].cpu().numpy()] = np.nan
                truths.append(truth)
                xyz_rows.append(self.physics_truth_xyz(reg, mask, segment).cpu().numpy())
        if truths:
            truth = np.concatenate(truths)
            support = summarize_native(truth, truth, self.regression_target_stats['task'],
                                       self.physics_config, np.concatenate(xyz_rows))
        else:
            support = summarize([], [], self.physics_config)
        self.validation_truth_support_hash = support['truth_support_hash']
        passed = support['n_valid_bins'] >= self.physics_config['min_valid_bins']
        path = self.physics_history.output_dir / 'validation_support.json'
        payload = {
            'schema': SCHEMA, 'purpose': 'validation_truth_occupancy',
            'task': self.regression_target_stats['task'], 'config': self.physics_config,
            'data_root_test': str(getattr(self.params, 'data_root_test', '')),
            'requested_sample_limit': getattr(self.params, 'limit_test_size', None)
                if getattr(self.params, 'limit_test_data', True) else None,
            'validation_batch_size': getattr(self.params, 'valid_batch_size', None),
            'max_val_batches': cap, 'passed': passed,
            **{key: support[key] for key in ('n_samples', 'n_valid_truth', 'n_invalid_truth',
                'n_outside_bins', 'n_valid_bins', 'valid_bin_indices', 'truth_support_hash')},
            'per_bin': [{key: row[key] for key in ('index','low','high','n_truth','valid','invalid_reason')}
                        for row in support['per_bin']],
        }
        atomic_json(path, payload)
        self.params['validation_support'] = str(path)
        self._write_physics_pointer('awaiting_validation' if passed else 'insufficient_validation_bins')
        print(f"Validation occupancy: {support['n_valid_bins']}/{len(support['per_bin'])} valid bins; "
              f"counts={[row['n_truth'] for row in support['per_bin']]}; report={path}")
        if not passed:
            raise ValueError(f"Insufficient validation bin occupancy before training: "
                f"{support['n_valid_bins']} valid bins, require {self.physics_config['min_valid_bins']}. "
                f"Inspect {path}; increase the validation sample or explicitly revise binning/occupancy settings.")

    def _publish_physics_selection(self):
        selected = self.physics_history.selected
        self.best_step = selected["step"] if selected else None
        self.best_epoch = selected["epoch"] if selected else None
        if selected:
            if getattr(self, "_published_physics_checkpoint", None) != selected["checkpoint"]:
                temporary = self.selected_checkpoint_path.with_suffix(".tmp")
                shutil.copyfile(selected["checkpoint"], temporary)
                os.replace(temporary, self.selected_checkpoint_path)
                self._published_physics_checkpoint = selected["checkpoint"]
            self.params["trained_checkpoint_path"] = str(self.selected_checkpoint_path)
        else:
            self.params["trained_checkpoint_path"] = None
        payload = self.physics_history.write()
        self.params["checkpoint_selection_status"] = payload["selection_status"]
        self._write_physics_pointer(payload["selection_status"])
        self.params["selected_physics_metrics"] = (
            {key: selected[key] for key in ("W_macro", "B_macro", "B_worst", "T_macro", "n_valid_bins")}
            if selected else None)
        print(f"Physics checkpoint selection: {payload['selection_status']}; selected step={self.best_step}")
        if not selected:
            print(f"No checkpoint passed. Best candidates: {payload['best_candidates']}")

    def _finish_checkpoint_selection(self, pretrain, filename):
        if self.checkpoint_selection != "physics":
            return
        if self.global_step > 0 and self.last_validation_step != self.global_step:
            loss = self.validate_end_to_end_one_epoch(pretrain=pretrain)
            self._record_validation_result(loss, filename, getattr(self, "epoch", getattr(self, "startEpoch", 1) - 1), self.global_step)
        elif not self.physics_history.records:
            self._publish_physics_selection()

    def _record_validation_result(
        self, val_loss, checkpoint_file_name, epoch, step, optuna_trial=None,
        allow_stagnation=True,
    ):
        self.last_validation_step = step
        improved_loss = val_loss < (self.best_loss - self.min_delta)
        if val_loss < self.best_loss and (self.checkpoint_selection == "physics" or improved_loss):
            self.best_loss = val_loss
            self.best_loss_step, self.best_loss_epoch = step, epoch
        objective = None
        if self.checkpoint_selection == "physics":
            previous = self.physics_history.selected
            previous_path = previous["checkpoint"] if previous else None
            path = self.physics_history.output_dir / f"step_{step:09d}_epoch_{epoch:05d}.pth"
            row = self.physics_history.add(self.last_validation_physics, step=step, epoch=epoch,
                                          validation_loss=val_loss, checkpoint=path)
            selected = self.physics_history.selected
            self.best_step = selected["step"] if selected else None
            self.best_epoch = selected["epoch"] if selected else None
            self._save_checkpoint(str(path), epoch, row["selected"], val_loss, publish=False)
            self._publish_physics_selection()
            improved = selected is not None and selected["checkpoint"] != previous_path
            if row["eligible"]:
                objective = row["W_macro"]
            # Do not stop for stagnation before any checkpoint is acceptable.
            allow_stagnation = allow_stagnation and selected is not None
        else:
            improved = improved_loss
            if improved:
                self.best_epoch, self.best_step = epoch, step
                self._save_checkpoint(checkpoint_file_name, epoch, True, val_loss)
            objective = float(val_loss)
        if improved:
            self.stagnation_counter = 0
        elif allow_stagnation and step >= self.early_stopping_min_steps:
            self.stagnation_counter += 1
        if optuna_trial is not None and objective is not None:
            optuna_trial.report(float(objective), int(step))
            if step >= self.early_stopping_min_steps and optuna_trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

    def _train_by_optimizer_step(
        self,
        pretrain,
        log_file_path,
        checkpoint_file_name,
        optuna_trial=None,
        metrics_callback=None,
    ):
        max_optimizer_steps = int(self.params.max_optimizer_steps)
        val_interval_steps = int(getattr(self.params, "val_interval_steps", max_optimizer_steps))
        if val_interval_steps < 1:
            raise ValueError("val_interval_steps must be >= 1")
        self.early_stopping_min_steps = int(getattr(
            self.params,
            "early_stopping_min_steps",
            getattr(self.params, "early_stopping_warmup_steps", 0),
        ))
        max_epochs = int(getattr(self.params, "max_epochs", 10**9))

        for epoch in range(self.startEpoch, max_epochs):
            self.down_results['epoch'] = epoch
            self.down_results['train'] = []
            self.down_results['val'] = []
            self.epoch = epoch
            if dist.is_initialized():
                self.train_sampler.set_epoch(epoch)

            self.model.eval()
            self.down_model.train()
            self.starttime = time.time()

            max_train_batches = getattr(self.params, "max_train_batches", None)
            for i, inputdict in enumerate(tqdm(self.train_data_loader)):
                if max_train_batches is not None and i >= int(max_train_batches):
                    break
                if self.global_step >= max_optimizer_steps:
                    break

                self.iters += 1
                loss_item = self._train_one_batch(inputdict, pretrain=pretrain)
                self.global_step += 1
                self.down_results['train'].append(loss_item)
                self.down_scheduler.step()

                should_validate = (
                    self.global_step % val_interval_steps == 0
                    or self.global_step >= max_optimizer_steps
                )
                if should_validate:
                    val_loss = self.validate_end_to_end_one_epoch(pretrain=pretrain)
                    train_loss = np.mean(self.down_results['train'])
                    elapsed = time.time() - self.starttime
                    with open(log_file_path, "a") as f:
                        f.write(
                            f"{self.global_step}\t{epoch}\t{train_loss:.8f}\t"
                            f"{val_loss:.8f}\t{self._current_lr():.8e}\t{elapsed:.2f}\n"
                        )
                    self._record_validation_result(
                        val_loss,
                        checkpoint_file_name,
                        epoch=epoch,
                        step=self.global_step,
                        optuna_trial=optuna_trial,
                    )
                    if metrics_callback is not None:
                        metrics_callback({
                            "step": self.global_step,
                            "epoch": epoch,
                            "train/loss": float(train_loss),
                            "val/loss": float(val_loss),
                            "lr": float(self._current_lr()),
                            "best/val_loss": float(self.best_loss),
                            "best/step": self.best_step,
                        })
                    if self.stagnation_counter >= self.patience:
                        print(
                            "Early stopping triggered at step "
                            f"{self.global_step} after {self.patience} validation "
                            "checks without improvement."
                        )
                        return
                    self.down_results['train'] = []
                    self.down_results['val'] = []
                    self.starttime = time.time()

            if self.global_step >= max_optimizer_steps:
                return

        print(
            f"Reached max_epochs={max_epochs} before max_optimizer_steps="
            f"{max_optimizer_steps}."
        )

    def downstream_end_to_end_one_epoch(self, pretrain = False):
        tr_time = 0
        self.model.eval()
        self.down_model.train()
        max_train_batches = getattr(self.params, "max_train_batches", 1001)
        # Buffers for logs
        tr_start = time.time()
        start_idx = 0
        for i, inputdict in enumerate(tqdm(self.train_data_loader)):
            if max_train_batches is not None and i >= int(max_train_batches):
                break
            self.iters += 1
            loss_item = self._train_one_batch(inputdict, pretrain=pretrain)
            self.global_step += 1
            self.down_results['train'].append(loss_item)

    @staticmethod
    def physics_truth_xyz(reg, mask, target_segment_mask=None):
        effective = mask if target_segment_mask is None else mask & target_segment_mask
        xyz = reg[..., :3]
        valid = effective.unsqueeze(-1) & torch.isfinite(xyz)
        counts = valid.sum(dim=1)
        mean = torch.where(valid, xyz, 0.).sum(dim=1) / counts.clamp_min(1)
        return mean.masked_fill(counts == 0, float("nan"))

    def validate_end_to_end_one_epoch(self, pretrain=False):
        backbone_training, head_training = self.model.training, self.down_model.training
        self.model.eval()
        self.down_model.eval()
        self.down_results['val'] = []
        predictions, truths, truth_xyz = [], [], []
        max_val_batches = getattr(self.params, "max_val_batches", 2001)
        try:
            with torch.no_grad():
                for i, inputdict in enumerate(tqdm(self.val_data_loader)):
                    if max_val_batches is not None and i >= int(max_val_batches):
                        break
                    grouped = inputdict['points'].to(self.device)
                    grouped = grouped.reshape(grouped.size(0), -1, grouped.size(-1))
                    mask = grouped[..., 0] != -100
                    reg = inputdict['reg_target'].to(self.device)
                    segment = inputdict.get('target_segment_mask')
                    if segment is not None:
                        segment = segment.to(self.device).bool()
                    targets = self.build_regression_targets(reg, mask, segment)
                    model_input = self._representation_input(grouped, inputdict)
                    if pretrain:
                        _, embeddings, _ = self.model(model_input, return_z=True)
                        pred = self.down_model(model_input, torch.stack(embeddings), pretrain=True,
                                               padding_mask=mask)["pred_regression"]
                    else:
                        pred = self.down_model(model_input, feature=None, padding_mask=mask,
                            **self._geometry_context_kwargs(inputdict, pretrain))["pred_regression"]
                    losses = masked_regression_loss(
                        outputs={"pred": pred}, targets=targets, option=self.regression_loss,
                        angular_indices=self.regression_target_stats["angular_indices"],
                        target_std=self.regression_target_stats["std"],
                        target_mean=self.regression_target_stats["mean"],
                        phi_pairs=self.regression_target_stats.get("phi_pairs", ()),
                        physical_scales=self.regression_loss_reference)
                    self.down_results['val'].append(losses['loss'].item())
                    if self.physics_config is not None:
                        head = self.down_model.module if isinstance(self.down_model, DistributedDataParallel) else self.down_model
                        predictions.append(head.target_normalizer.denormalize(pred).cpu().numpy())
                        truth = head.target_normalizer.denormalize(targets['target']).cpu().numpy()
                        truth[~targets['target_valid'].cpu().numpy()] = np.nan
                        truths.append(truth)
                        truth_xyz.append(self.physics_truth_xyz(reg, mask, segment).cpu().numpy())
        finally:
            self.model.train(backbone_training)
            self.down_model.train(head_training)
        if not self.down_results['val']:
            raise RuntimeError("Validation yielded no batches; cannot select a checkpoint")
        avg_loss = float(np.mean(self.down_results['val']))
        if self.physics_config is not None:
            self.last_validation_physics = summarize_native(
                np.concatenate(predictions), np.concatenate(truths), self.regression_target_stats['task'],
                self.physics_config, np.concatenate(truth_xyz))
            expected = getattr(self, 'validation_truth_support_hash', None)
            if expected is not None and self.last_validation_physics['truth_support_hash'] != expected:
                raise ValueError('Validation truth sample changed since occupancy preflight')
        if self.log_to_screen:
            print(f"\nValidation Loss: {avg_loss:.4f}")
        return avg_loss

    def _save_checkpoint(self, filename, epoch, is_best, loss, publish=True):
        checkpoint = {
            'epoch': epoch,
            'global_step': getattr(self, "global_step", None),
            'model_state_dict': self.down_model.state_dict(),
            'optimizer_state_dict': self.down_optimizer.state_dict(),
            'scheduler_state_dict': self.down_scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_step': getattr(self, "best_step", None),
            'best_epoch': getattr(self, "best_epoch", None),
            'current_loss': loss,
            'best_loss_step': getattr(self, 'best_loss_step', None),
            'best_loss_epoch': getattr(self, 'best_loss_epoch', None),
            'checkpoint_selection': self.checkpoint_selection,
            'physics_checkpoint_config': self.physics_config,
            'physics_validation_summary': getattr(self, 'last_validation_physics', None),
            'physics_checkpoint_summary_path': getattr(self.params, 'physics_checkpoint_summary', None),
            'regression_task': self.regression_target_stats["task"],
            'regression_target_columns': self.regression_target_stats["columns"],
            'regression_target_stats': self.regression_target_stats["path"],
            'input_representation': getattr(self.params, 'input_representation', 'center_only'),
            'params': vars(self.params)  # Save all hyperparameters
        }

        # Handle DistributedDataParallel wrapper
        if isinstance(self.down_model, torch.nn.parallel.DistributedDataParallel):
            checkpoint['model_state_dict'] = self.down_model.module.state_dict()

        checkpoint_path = os.path.abspath(os.path.join(self.params.checkpoint_dir, filename))
        torch.save(checkpoint, checkpoint_path)
        if publish:
            self.params["trained_checkpoint_path"] = checkpoint_path
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path, inference=False):
        """Load checkpoint with proper device mapping and DDP handling. 
           If inference=True, only loads the model weights."""
        
        # 1. Get proper device string
        if isinstance(self.device, int):
            device_str = f'cuda:{self.device}' if torch.cuda.is_available() else 'cpu'
        else:
            device_str = str(self.device)
    
        checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Adapter checkpoint does not exist: {checkpoint_path}")

        # 2. Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device_str, weights_only=False)
        if "model_state_dict" not in checkpoint:
            raise KeyError(
                f"Adapter checkpoint {checkpoint_path} is missing 'model_state_dict'. "
                "This loader expects a downstream adapter checkpoint, not a pretrained backbone."
            )
        checkpoint_task = checkpoint.get("regression_task")
        current_task = self.regression_target_stats["task"]
        if checkpoint_task is not None and checkpoint_task != current_task:
            raise ValueError(
                "Adapter checkpoint regression task does not match the current "
                f"configuration: checkpoint has {checkpoint_task!r}, config has "
                f"{current_task!r}. Use the matching model YAML/task and stats file."
            )
        checkpoint_representation = checkpoint.get('input_representation')
        current_representation = getattr(self.params, 'input_representation', 'center_only')
        if checkpoint_representation is not None and checkpoint_representation != current_representation:
            raise ValueError(
                "Adapter checkpoint input representation does not match the current configuration: "
                f"checkpoint has {checkpoint_representation!r}, config has {current_representation!r}."
            )
    
        self.loaded_checkpoint_metadata = {key: checkpoint.get(key) for key in (
            "epoch", "global_step", "current_loss", "physics_checkpoint_config",
            "physics_validation_summary", "physics_checkpoint_summary_path", "checkpoint_selection")}
        if not inference and checkpoint.get("physics_checkpoint_config") not in (None, self.physics_config):
            raise ValueError("Resume physics configuration differs from checkpoint")
        # Resume starts a new comparison history: past cohorts may no longer be available.
        # 3. Handle DDP keys
        state_dict = checkpoint['model_state_dict']
        if "weighted_avg_weights" in state_dict:
            print("Trained weighted_avg_weights:", state_dict["weighted_avg_weights"])
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
        # 4. Load model weights
        try:
            if isinstance(self.down_model, torch.nn.parallel.DistributedDataParallel):
                self.down_model.module.load_state_dict(new_state_dict, strict=False)
            else:
                self.down_model.load_state_dict(new_state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load adapter checkpoint with the current regression-head "
                f"configuration. checkpoint={checkpoint_path}, "
                f"embed_dim={getattr(self.params, 'embed_dim', None)}, "
                f"num_layers_backbone={getattr(self.params, 'num_layers_backbone', None)}, "
                f"mambaversion={getattr(self.params, 'mambaversion', None)}"
            ) from exc
    
        # 5. If not inference mode, load optimizer/scheduler states
        if not inference:
            self.down_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.down_scheduler is not None:
                self.down_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
            self.startEpoch = checkpoint.get('epoch', 0) + 1
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.best_loss_step = checkpoint.get('best_loss_step')
            self.best_loss_epoch = checkpoint.get('best_loss_epoch')
            self.best_step = checkpoint.get('best_step', None)
            self.best_epoch = checkpoint.get('best_epoch', None)
            self.global_step = checkpoint.get('global_step', 0) or 0
    
        # 6. Log info
        if self.log_to_screen:
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")



    


    def report_loss(self, loss_, dist):
        step_loss = torch.zeros((1), dtype=torch.float32, device=self.device)
        step_loss += loss_.detach()

        if dist.is_initialized():
            dist.all_reduce(step_loss)
            loss_log = float(step_loss.item()/dist.get_world_size())
        else:
            loss_log = step_loss.item()
        return loss_log

    def set_portion_condition(self, tmask, portion = 0.2):
        """tmask: a mask showing effective (i.e., non-padding area) region as 1"""
        total = tmask.sum(-1)
        condidx = torch.ceil(total * portion).long()        
        index_tensor = torch.arange(tmask.size(1)).expand(tmask.size(0), -1).to(tmask.device)  # Shape (B, N)
        newmask = (index_tensor < condidx.unsqueeze(1)).float()
        return newmask.bool()


    
            
    def restore_checkpoint(self, checkpoint_path, load_optimizer_state=True):
        """
        Load checkpoint from file.

        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer_state: If True, load optimizer/scheduler state (for resuming training).
                                 If False, only load model weights (for pretrained initialization).
        """
        checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Pretrained backbone checkpoint does not exist: {checkpoint_path}")

        if isinstance(self.device, int):
            device_str = f'cuda:{self.device}' if torch.cuda.is_available() else 'cpu'
        else:
            device_str = str(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=device_str, weights_only=False)
        if "model_state" not in checkpoint:
            raise KeyError(
                f"Pretrained checkpoint {checkpoint_path} is missing 'model_state'. "
                "This loader expects a backbone checkpoint produced by pretraining."
            )
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state'].items()}
        try:
            #self.model.load_state_dict(checkpoint['model_state'])
            self.model.load_state_dict(new_state_dict)
        except RuntimeError as exc:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                name = key[7:]
                new_state_dict[name] = val
            try:
                self.model.load_state_dict(new_state_dict)
            except RuntimeError as second_exc:
                raise RuntimeError(
                    "Failed to load pretrained backbone with the current model "
                    f"configuration. checkpoint={checkpoint_path}, "
                    f"embed_dim={getattr(self.params, 'embed_dim', None)}, "
                    f"base_dim={getattr(self.params, 'base_dim', None)}, "
                    f"num_layers_backbone={getattr(self.params, 'num_layers_backbone', None)}, "
                    f"klen={getattr(self.params, 'klen', None)}, "
                    f"mambaversion={getattr(self.params, 'mambaversion', None)}"
                ) from second_exc

        if load_optimizer_state:
            # Load optimizer and scheduler state for resuming training
            self.iters = checkpoint['iters']
            self.startEpoch = checkpoint['epoch']+1 if self.iters % len(self.train_data_loader) == 0 else checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            # Only loaded model weights for pretrained initialization
            self.iters = 0
            self.startEpoch = 0
            if self.world_rank == 0:
                print(f"✅ Loaded pretrained weights only (optimizer state not loaded)")
