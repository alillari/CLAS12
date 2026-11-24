import torch
import torch.nn as nn

from fm4npp.models.embed import *
from fm4npp.models.rmsnorm import RMSNorm
from fm4npp.models.mamba_config import build_adapter_layer, apply_adapter_layer
from train.downstream.feature_mix import make_layer_mixer, mix_layers, valid_tokens


class RegressionTargetNormalizer(nn.Module):
    def __init__(self, output_dim, mean=None, std=None, eps=1e-8):
        super().__init__()
        mean = torch.zeros(output_dim) if mean is None else torch.as_tensor(mean, dtype=torch.float32)
        std = torch.ones(output_dim) if std is None else torch.as_tensor(std, dtype=torch.float32)
        expected_shape = (output_dim,)
        if tuple(mean.shape) != expected_shape or tuple(std.shape) != expected_shape:
            raise ValueError(
                f"Regression normalization statistics must have shape {expected_shape}; "
                f"got mean={tuple(mean.shape)}, std={tuple(std.shape)}"
            )
        constant = std <= eps
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", torch.where(constant, torch.ones_like(std), std))
        self.register_buffer("constant", constant)

    def normalize(self, target):
        normalized = (target - self.mean) / self.std
        return torch.where(self.constant, torch.zeros_like(normalized), normalized)

    def denormalize(self, target):
        denormalized = target * self.std + self.mean
        return torch.where(self.constant, self.mean, denormalized)



class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, prenorm=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.prenorm = prenorm

    def forward(self, x, key_padding_mask=None):
        if self.prenorm:
            x_norm = self.norm(x)
            attn_output, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
            return x + attn_output
        else:
            attn_output, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
            return self.norm(x + attn_output)


class FFNBlock(nn.Module):
    def __init__(self, embed_dim, ffn_dim, prenorm=True):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.prenorm = prenorm

    def forward(self, x):
        if self.prenorm:
            x_norm = self.norm(x)
            return x + self.ffn(x_norm)
        else:
            return self.norm(x + self.ffn(x))


class MLPHead(nn.Module):
    def __init__(self, embed_dim, output_dim, dropout=0.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# very simple adapter head with few mamba layer for refinement and MLP head
class MambaHead(nn.Module):
    def __init__(self, input_dim, embed_dim=256, num_layers=3, d_state=64, d_conv=4, expand=2, 
                 num_embedder_layers=1, d_state_embedder=64, d_conv_embedder=4, expand_embedder=2,
                 num_feature_layers=15, num_output_dim=5, return_embedding=False, dropout=0.0, *, embed_method="add", pe_method="nerf", adapter_version="mamba2", adapter_block_kwargs=None, pos_dim=3, aux_dim=4,
                 feature_mix="raw", feature_stat_batches=200):
        super().__init__()
        self.adapter_version = adapter_version
        if adapter_version == "transformer" and num_layers < 1:
            raise ValueError("Transformer adapters require at least one refinement layer (num_adapter_layers >= 1)")
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.return_embedding = return_embedding

        # Input processing
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim)
        )


        # Mamba feature extractor
        self.mamba_layers = nn.ModuleList([
            build_adapter_layer(
                adapter_version, block_kwargs=adapter_block_kwargs,
                d_model=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(num_layers)
        ])
        # if not using the pretrained mamba2, we can use the embedder layers
        self.mamba_embedder_layers = nn.ModuleList([
            build_adapter_layer(
                adapter_version, block_kwargs=adapter_block_kwargs,
                d_model=input_dim,
                d_state=d_state_embedder,
                d_conv=d_conv_embedder,
                expand=expand_embedder
            ) for _ in range(num_embedder_layers)
        ])
        self.embedder_norm = RMSNorm(input_dim)
        self.norm = RMSNorm(embed_dim)

        # Noise prediction head go from point embedding
        self.out_mlp = MLPHead(embed_dim, num_output_dim, dropout=dropout)

        embedders = {"add": EmbedderAdd, "concat": EmbedderConcat, "pos_only": EmbedderPosOnly, "pos_plus_aux": EmbedderPosPlusAux}
        if embed_method not in embedders:
            raise ValueError(f"Unknown embed_method: {embed_method}")
        self.embedder = embedders[embed_method](pe_method=pe_method, embed_dim=input_dim,
                                                  **(dict(pos_dim=pos_dim, aux_dim=aux_dim) if embed_method == "pos_plus_aux" else {}))
        self.weighted_avg_weights = nn.Parameter(torch.ones(num_feature_layers))
        # None for the historical raw mix (state dict unchanged); see feature_mix.py
        self.feature_standardizer = make_layer_mixer(num_feature_layers, input_dim, feature_mix, feature_stat_batches)


    def forward(self, x, feature=None, padding_mask=None, pretrain=False):
        if self.adapter_version == "transformer":
            if padding_mask is None:
                padding_mask = x[..., 0] != -100
            x = x.masked_fill(~padding_mask[..., None], 0.)
        # Input processing
        if pretrain:
            x = mix_layers(feature, self.weighted_avg_weights, self.feature_standardizer,
                           valid_tokens(x, padding_mask))
        else:
            x = self.embedder(x)
            if isinstance(x, tuple):
                x = x[0]
            for layer in self.mamba_embedder_layers:
                x = apply_adapter_layer(layer, x, padding_mask)
            x = self.embedder_norm(x)
        
        if self.adapter_version == "transformer":
            x = x.masked_fill(~padding_mask[..., None], 0.)
        embedding_pre_projection = None
        embedding_post_projection = None
        if self.return_embedding:
            embedding_pre_projection = x

        x = self.input_proj(x)

        if self.return_embedding:
            embedding_post_projection = x
        
        # Process through Mamba layers
        for layer in self.mamba_layers:
            x = apply_adapter_layer(layer, x, padding_mask)
        x = self.norm(x)

        #noise prediction
        out_logits = self.out_mlp(x) # (B, N, num_output_dim)
        
        return {
            'pred_logits': out_logits,  # (B, N, num_output_dim)
            'embedding_pre_projection': embedding_pre_projection,
            'embedding_post_projection': embedding_post_projection
        }

# very simple regression adapter head with few mamba layer for refinement and MLP head
class MambaTrackRegressionHead(nn.Module):
    """
    Track-level regression head.

    Input:
        x:              (B, N, C)
        feature:        pretrained FM features, if pretrain=True
        padding_mask:   (B, N), True for real hits, False for padding

    Output:
        pred_regression: (B, num_output_dim)
    """

    def __init__(self, input_dim, embed_dim=256, num_layers=3, d_state=64, d_conv=4, expand=2, 
                 num_embedder_layers=1, d_state_embedder=64, d_conv_embedder=4, expand_embedder=2,
                 num_feature_layers=15, num_output_dim=3, return_embedding=False, dropout=0.0,
                 pooling="mean", embed_method="add", pe_method="nerf",
                 target_mean=None, target_std=None,
                 input_representation="center_only",
                 pos_dim=3, aux_dim=4, *, adapter_version="mamba2", adapter_block_kwargs=None,
                 feature_mix="raw", feature_stat_batches=200):
        super().__init__()
        self.adapter_version = adapter_version
        if adapter_version == "transformer" and num_layers < 1:
            raise ValueError("Transformer adapters require at least one refinement layer (num_adapter_layers >= 1)")
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.return_embedding = return_embedding
        self.input_representation = str(input_representation)
        if self.input_representation not in {
            "center_only", "clas12_pos_plus_aux_v1"
        }:
            raise ValueError(
                "input_representation must be 'center_only' "
                "or 'clas12_pos_plus_aux_v1', "
                f"got {self.input_representation!r}"
            )
        self.target_normalizer = RegressionTargetNormalizer(
            num_output_dim, mean=target_mean, std=target_std
        )

        if embed_method == "concat":
            Embedder = EmbedderConcat
        elif embed_method == "pos_only":
            Embedder = EmbedderPosOnly
        elif embed_method == "pos_plus_aux":
            Embedder = EmbedderPosPlusAux
        elif embed_method == "add":
            Embedder = EmbedderAdd
        else:
            raise ValueError(f"Unknown embed_method: {embed_method}")

        self.pooling = pooling

        if self.pooling == "attention":
            self.pool_score = nn.Linear(embed_dim, 1)
        elif self.pooling != "mean":
            raise ValueError(f"Unknown pooling mode: {self.pooling}")

        # Input processing
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim)
        )


        # Mamba feature extractor
        self.mamba_layers = nn.ModuleList([
            build_adapter_layer(
                adapter_version, block_kwargs=adapter_block_kwargs,
                d_model=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(num_layers)
        ])
        # if not using the pretrained mamba2, we can use the embedder layers
        self.mamba_embedder_layers = nn.ModuleList([
            build_adapter_layer(
                adapter_version, block_kwargs=adapter_block_kwargs,
                d_model=input_dim,
                d_state=d_state_embedder,
                d_conv=d_conv_embedder,
                expand=expand_embedder
            ) for _ in range(num_embedder_layers)
        ])
        self.embedder_norm = RMSNorm(input_dim)
        self.norm = RMSNorm(embed_dim)

        # Noise prediction head go from point embedding
        self.out_mlp = MLPHead(embed_dim, num_output_dim, dropout=dropout)

        if embed_method == "pos_plus_aux":
            self.embedder = Embedder(
                pe_method=pe_method,
                embed_dim=input_dim,
                pos_dim=pos_dim,
                aux_dim=aux_dim,
            )
        else:
            self.embedder = Embedder(pe_method=pe_method, embed_dim=input_dim)
        self.weighted_avg_weights = nn.Parameter(torch.ones(num_feature_layers))
        # None for the historical raw mix (state dict unchanged); see feature_mix.py
        self.feature_standardizer = make_layer_mixer(num_feature_layers, input_dim, feature_mix, feature_stat_batches)

    def pool(self, x, padding_mask=None):
        """
        x:            (B, N, D)
        padding_mask: (B, N), True for real hits, False for padding
        """
        if self.pooling == "mean":
            if padding_mask is None:
                return x.mean(dim=1)

            mask = padding_mask.unsqueeze(-1).to(dtype=x.dtype)
            denom = mask.sum(dim=1).clamp_min(1.0)
            return (x * mask).sum(dim=1) / denom

        if self.pooling == "attention":
            scores = self.pool_score(x).squeeze(-1)  # B x N

            if padding_mask is not None:
                scores = scores.masked_fill(~padding_mask, torch.finfo(scores.dtype).min)

            weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # B x N x 1
            return (x * weights).sum(dim=1)

    def forward(
        self,
        x,
        feature=None,
        padding_mask=None,
        pretrain=False,
    ):
        if self.adapter_version == "transformer":
            if padding_mask is None:
                padding_mask = x[..., 0] != -100
            x = x.masked_fill(~padding_mask[..., None], 0.)
        if pretrain:
            x = mix_layers(feature, self.weighted_avg_weights, self.feature_standardizer,
                           valid_tokens(x, padding_mask))
        else:
            x = self.embedder(x)

            if isinstance(x, tuple):
                x = x[0]

            for layer in self.mamba_embedder_layers:
                x = apply_adapter_layer(layer, x, padding_mask)

            x = self.embedder_norm(x)

        if self.adapter_version == "transformer":
            x = x.masked_fill(~padding_mask[..., None], 0.)
        embedding_pre_projection = None
        embedding_post_projection = None

        if self.return_embedding:
            embedding_pre_projection = x

        x = self.input_proj(x)

        if self.return_embedding:
            embedding_post_projection = x

        for layer in self.mamba_layers:
            x = apply_adapter_layer(layer, x, padding_mask)

        x = self.norm(x)

        pooled = self.pool(x, padding_mask)

        pred = self.out_mlp(pooled)                          # B x num_output_dim

        return {
            "pred_regression": pred,
            "pooled_embedding": pooled,
            "embedding_pre_projection": embedding_pre_projection,
            "embedding_post_projection": embedding_post_projection,
        } 


# Backward-compatible name for existing imports and code.
MambaRegressionHead = MambaTrackRegressionHead

# very simple adapter head with few SA for refinement and MLP head
class AttentionHead(nn.Module):
    def __init__(self, input_dim, embed_dim=256, num_layers=3, num_heads = 4, 
                 num_embedder_layers=1, d_state_embedder=64, d_conv_embedder=4, expand_embedder=2,
                 num_feature_layers=15, num_output_dim=5, return_embedding=False, embed_method='add', pe_method = 'nerf', dropout=0.0, ffn_dim=512, adapter_FFN = True, *, adapter_version="mamba2", adapter_block_kwargs=None, pos_dim=3, aux_dim=4,
                 feature_mix="raw", feature_stat_batches=200):
        super().__init__()
        self.adapter_version = adapter_version
        if adapter_version == "transformer" and num_layers < 1:
            raise ValueError("Transformer adapters require at least one refinement layer (num_adapter_layers >= 1)")
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.return_embedding = return_embedding
        self.FFN = adapter_FFN
        if embed_method == 'concat':
            Embedder = EmbedderConcat
        elif embed_method == 'pos_only':
            Embedder = EmbedderPosOnly
        elif embed_method == 'pos_plus_aux':
            Embedder = EmbedderPosPlusAux
        elif embed_method == 'add':
            Embedder = EmbedderAdd
        else:
            raise ValueError(f"Unknown embed_method: {embed_method}")

        # Input processing
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim)
        )


        # SA feature extractor
        self.SA_layers = nn.ModuleList([
            build_adapter_layer("transformer", d_model=embed_dim, block_kwargs=adapter_block_kwargs)
            if adapter_version == "transformer" else
            SelfAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        #ffn layers
        if self.FFN and adapter_version != "transformer":
            self.ffn_layers = nn.ModuleList([
                FFNBlock(
                    embed_dim=embed_dim,
                    ffn_dim=ffn_dim,
                )
                for _ in range(num_layers)
            ])
        else:
            self.ffn_layers = []
        # if not using the pretrained mamba2, we can use the embedder layers
        self.mamba_embedder_layers = nn.ModuleList([
            build_adapter_layer(
                adapter_version, block_kwargs=adapter_block_kwargs,
                d_model=input_dim,
                d_state=d_state_embedder,
                d_conv=d_conv_embedder,
                expand=expand_embedder
            ) for _ in range(num_embedder_layers)
        ])
        self.embedder_norm = RMSNorm(input_dim)
        self.norm = RMSNorm(embed_dim)

        # Noise prediction head go from point embedding
        self.out_mlp = MLPHead(embed_dim, num_output_dim, dropout=dropout)

        # Historical attention-head checkpoints always used NeRF. Preserve that
        # topology for existing embeddings; auxiliary backbones declare theirs.
        embedding_pe_method = pe_method if embed_method == "pos_plus_aux" else "nerf"
        self.embedder = Embedder(pe_method=embedding_pe_method, embed_dim=input_dim,
                                 **(dict(pos_dim=pos_dim, aux_dim=aux_dim) if embed_method == "pos_plus_aux" else {}))
        self.weighted_avg_weights = nn.Parameter(torch.ones(num_feature_layers))
        # None for the historical raw mix (state dict unchanged); see feature_mix.py
        self.feature_standardizer = make_layer_mixer(num_feature_layers, input_dim, feature_mix, feature_stat_batches)


    def forward(self, x, feature=None, padding_mask=None, pretrain=False):
        if self.adapter_version == "transformer":
            if padding_mask is None:
                padding_mask = x[..., 0] != -100
            x = x.masked_fill(~padding_mask[..., None], 0.)
        # Input processing
        if pretrain:
            x = mix_layers(feature, self.weighted_avg_weights, self.feature_standardizer,
                           valid_tokens(x, padding_mask))
        else:
            x = self.embedder(x)
            if isinstance(x, tuple):
                x = x[0]
            for layer in self.mamba_embedder_layers:
                x = apply_adapter_layer(layer, x, padding_mask)
            x = self.embedder_norm(x)
        
        if self.adapter_version == "transformer":
            x = x.masked_fill(~padding_mask[..., None], 0.)
        embedding_pre_projection = None
        embedding_post_projection = None
        if self.return_embedding:
            embedding_pre_projection = x

        x = self.input_proj(x) #(B,N,D)

        

        if self.return_embedding:
            embedding_post_projection = x
        
        if self.adapter_version == "transformer":
            for layer in self.SA_layers:
                x = apply_adapter_layer(layer, x, padding_mask)
            x = self.norm(x)
        else:
            x = x.transpose(0, 1) #(N,B,D)
            # Existing bidirectional attention + FFN path.
            key_padding_mask = None if padding_mask is None else ~padding_mask
            if self.FFN:
                for sa_layer, ffn_layer in zip(self.SA_layers, self.ffn_layers):
                    x = sa_layer(x, key_padding_mask=key_padding_mask)
                    x = ffn_layer(x)
            else:
                for sa_layer in self.SA_layers:
                    x = sa_layer(x, key_padding_mask=key_padding_mask)
            x = self.norm(x)
            x = x.transpose(0, 1) #(B,N,D)
        #noise prediction
        out_logits = self.out_mlp(x) # (B, N, num_output_dim)

        return {
            'pred': out_logits,  # (B, N, num_output_dim)
            'pred_logits': out_logits,  # Classification trainer contract.
            'embedding_pre_projection': embedding_pre_projection,
            'embedding_post_projection': embedding_post_projection
        }
