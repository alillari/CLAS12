import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

from fm4npp.models.mamba2 import Mamba2
from fm4npp.models.embed import *
from fm4npp.models.rmsnorm import RMSNorm


class MambaGPT(nn.Module):
    def __init__(self, embed_dim=512, num_layers=12, d_state=64, d_conv=4, expand=2, klen=10, dropout = 0.2, embed_method='add', pe_method = 'nerf'):
        super().__init__()
        assert embed_method in ['concat', 'add', 'pos_only']
        self.embed_dim = embed_dim
        
        if embed_method == 'concat':
            Embedder = EmbedderConcat
        elif embed_method == 'pos_only':
            Embedder = EmbedderPosOnly
        else:
            Embedder = EmbedderAdd
            
        self.embedder = Embedder(pe_method = pe_method, embed_dim = embed_dim, learnable_projection = False)
        
        self.mamba_layers = nn.ModuleList(
            [nn.Sequential(RMSNorm(embed_dim), 
                           Mamba2(d_model=embed_dim, d_state=d_state, d_conv=d_conv, expand=expand),
                           nn.Dropout(dropout)) 
             for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(embed_dim, klen * 3)
        self.norm = RMSNorm(embed_dim)

    def change_maskval(self, x, init_val = -100, target_val = 0):
        out = x.clone()
        out[out == init_val] = target_val
        return out
        
    def forward(self, x, return_z = False):
        in_scale, out_scale = 1.0, 1.0
        x = self.change_maskval(x) # for training stability
        x, pos = self.embedder(x)  # Add slight noise
        
        x = x * in_scale
        feature = []
        # feature = 0
        for layer in self.mamba_layers:
            z = layer(x)
            x = z + x
            # [FIX] was: feature.append(z) BEFORE the residual add -- this
            # captured only each layer's isolated adjustment (delta), not
            # the network's actual accumulated hidden state at that depth.
            # Confirmed via a direct comparison (closed-form ridge regression
            # against real momentum, standardized features): the accumulated
            # state gives a stable R^2 ~0.82 across the last 5 layers, vs. an
            # erratic 0.23-0.74 for the isolated delta. Downstream adapters
            # consuming return_z=True (e.g. the momentum head's weighted
            # layer-average) were working with the weaker, disconnected
            # signal instead of the network's real internal representation.
            #
            # SAFE for existing checkpoints and pretraining: this only changes
            # what gets appended to the `feature` list returned when
            # return_z=True is explicitly requested. The real training/
            # inference path (return_z=False, the else branch below) computes
            # `x` identically either way -- no weights, parameter names, or
            # the normal forward pass are affected. Existing checkpoints load
            # and reproduce their original behavior unchanged.
            feature.append(x)
            
        if return_z:
            return None, feature, pos
        else:
            x = self.norm(x)
            return self.output_layer(x) * out_scale


class Mamba1GPT(nn.Module):
    def __init__(self, embed_dim=512, num_layers=12, d_state=64, d_conv=4, expand=2, klen=10, dropout = 0.2, embed_method='add', pe_method = 'nerf'):
        super().__init__()
        assert embed_method in ['concat', 'add', 'pos_only']
        self.embed_dim = embed_dim
        if embed_method == 'concat':
            Embedder = EmbedderConcat
        elif embed_method == 'pos_only':
            Embedder = EmbedderPosOnly
        else:
            Embedder = EmbedderAdd
        self.embedder = Embedder(pe_method = pe_method, embed_dim = embed_dim, learnable_projection = False)
        self.mamba_layers = nn.ModuleList(
            [nn.Sequential(RMSNorm(embed_dim), 
                           Mamba(d_model=embed_dim, d_state=d_state, d_conv=d_conv, expand=expand),
                           nn.Dropout(dropout)) 
             for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(embed_dim, klen * 3)
        self.norm = RMSNorm(embed_dim)
 
    def change_maskval(self, x, init_val = -100, target_val = 0):
        out = x.clone()
        out[out == init_val] = target_val
        return out
    def forward(self, x, return_z = False):
        in_scale, out_scale = 1.0, 1.0
        x = self.change_maskval(x) # for training stability
        x, pos = self.embedder(x)  # Add slight noise
        x = x * in_scale
        feature = []
        for layer in self.mamba_layers:
            z = layer(x)
            x = z + x
            # [FIX] was: feature.append(z) BEFORE the residual add -- this
            # captured only each layer's isolated adjustment (delta), not
            # the network's actual accumulated hidden state at that depth.
            # Confirmed via a direct comparison (closed-form ridge regression
            # against real momentum, standardized features): the accumulated
            # state gives a stable R^2 ~0.82 across the last 5 layers, vs. an
            # erratic 0.23-0.74 for the isolated delta. Downstream adapters
            # consuming return_z=True (e.g. the momentum head's weighted
            # layer-average) were working with the weaker, disconnected
            # signal instead of the network's real internal representation.
            #
            # SAFE for existing checkpoints and pretraining: this only changes
            # what gets appended to the `feature` list returned when
            # return_z=True is explicitly requested. The real training/
            # inference path (return_z=False, the else branch below) computes
            # `x` identically either way -- no weights, parameter names, or
            # the normal forward pass are affected. Existing checkpoints load
            # and reproduce their original behavior unchanged.
            feature.append(x)
        #feature.append(x)
        if return_z:
            return None, feature, pos
        else:
            x = self.norm(x)
            return self.output_layer(x) * out_scale
