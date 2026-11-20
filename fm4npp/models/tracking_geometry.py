"""Embed independent tracking measurements before contextualization."""

from torch import cat, log1p, where
from torch import nn

from fm4npp.tracking_geometry import (
    ARC, CENTER, DETECTOR, ENDPOINT_OFFSETS, KIND, LAYER, LENGTH, NUM_DETECTOR_IDS,
    NUM_LAYER_IDS, PITCH, SWEEP, TOKEN_WIDTH, valid_tracking_hits,
)
from .embed import CoordinateEmbedder


class TrackingGeometryEmbedder(nn.Module):
    """Shared position and endpoint encoders with line and arc projections."""

    def __init__(self, pe_method, embed_dim):
        super().__init__()
        self.position = CoordinateEmbedder(pe_method, 3, embed_dim, learnable_projection=True)
        self.position_norm = nn.LayerNorm(embed_dim)
        self.endpoint = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 32))
        self.line = nn.Sequential(nn.Linear(34, 64), nn.SiLU(), nn.Linear(64, embed_dim))
        self.arc = nn.Sequential(nn.Linear(38, 64), nn.SiLU(), nn.Linear(64, embed_dim))
        self.geometry_norm = nn.LayerNorm(embed_dim)
        self.detector = nn.Embedding(NUM_DETECTOR_IDS, embed_dim, padding_idx=0)
        self.layer = nn.Embedding(NUM_LAYER_IDS, embed_dim, padding_idx=0)
        for table in (self.detector, self.layer):
            nn.init.normal_(table.weight[1:], std=0.02)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != TOKEN_WIDTH:
            raise ValueError(f"Expected tracking geometry tokens of shape (batch, hits, {TOKEN_WIDTH})")
        valid = valid_tracking_hits(tokens)
        tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0)
        position = self.position_norm(self.position(tokens[..., CENTER] / 100.0))

        offsets = tokens[..., ENDPOINT_OFFSETS].unflatten(-1, (2, 3)) / 10.0
        # Evaluate the same network at both endpoints, then pool only this pair.
        # The hit remains one token and its value is independent of endpoint order.
        endpoints = self.endpoint(offsets).mean(dim=-2)
        length = log1p(tokens[..., LENGTH:LENGTH + 1] / 10.0)
        pitch = log1p(tokens[..., PITCH:PITCH + 1] / 0.1)
        shape = cat((endpoints, length, pitch), dim=-1)
        # The arc anchor is the arc midpoint. Its displacement from the chord
        # midpoint, together with the sweep, tells the model which way it bows.
        bow = -offsets.mean(dim=-2)
        arc = cat((shape, bow, tokens[..., SWEEP:SWEEP + 1]), dim=-1)
        geometry = where(tokens[..., KIND:KIND + 1] == ARC, self.arc(arc), self.line(shape))
        context = self.detector(tokens[..., DETECTOR].long()) + self.layer(tokens[..., LAYER].long())
        embedded = position + self.geometry_norm(geometry) + context
        padding = ~valid.unsqueeze(-1)
        return embedded.masked_fill(padding, 0), position.masked_fill(padding, 0)
