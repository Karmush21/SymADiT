"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import math
from typing import Dict
import torch
from torch import nn
from torch_geometric.utils import to_dense_batch



class FractionalCoordEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_frequencies: int = 8,
        hidden_dim: int | None = None,
    ):
        super().__init__()

        hidden_dim = hidden_dim or d_model

        # Integer frequencies preserve exact period-1 behavior.
        frequencies = torch.arange(
            1,
            num_frequencies + 1,
            dtype=torch.float32,
        )
        self.register_buffer("frequencies", frequencies)

        input_dim = 3 * 2 * num_frequencies

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, frac_coords: torch.Tensor) -> torch.Tensor:
        # frac_coords: [..., 3]
        frac_coords = torch.remainder(frac_coords, 1.0)

        angles = (
            2.0
            * math.pi
            * frac_coords.unsqueeze(-1)
            * self.frequencies
        )
        # [..., 3, num_frequencies]

        features = torch.cat(
            [torch.sin(angles), torch.cos(angles)],
            dim=-1,
        )
        # [..., 3, 2 * num_frequencies]

        features = features.flatten(start_dim=-2)
        # [..., 3 * 2 * num_frequencies]

        return self.proj(features)


class SymTransformerEncoder(nn.Module):
    def __init__(self, max_num_elements=100, d_model=512, nhead=8,
                 dim_feedforward=2048, activation="gelu", dropout=0.0,
                 norm_first=True, bias=True, num_layers=6, num_wy_tokens=1731):
        super().__init__()

        self.d_model = d_model

        # Local/node features
        self.atom_type_embedder = nn.Embedding(max_num_elements, d_model)
        self.wyckoff_embedder  = nn.Embedding(num_wy_tokens, d_model)
        
        
        self.frac_coord_embedder = nn.Sequential(nn.Linear(3, d_model, bias=False), nn.SiLU(), nn.Linear(d_model, d_model))
        
        # NOTE Would be interesting trying with this (not used in the paper though)
        #self.frac_coord_embedder = FractionalCoordEmbedding(d_model=d_model, num_frequencies=8)


        self.sg_embedder   = nn.Embedding(231, self.d_model)
      
          
        self.cell_free_proj  = nn.Sequential( # project [a,b,c,α,β,γ] with mask applied
            nn.Linear(6, self.d_model, bias=False),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model, bias=True),
        )


        act = {"gelu": nn.GELU(approximate="tanh"), "relu": nn.ReLU()}[activation]
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                activation=act, dropout=dropout, batch_first=True,
                norm_first=norm_first, bias=bias,
            ),
            norm=nn.LayerNorm(d_model),
            num_layers=num_layers,
        )


    def forward(self, batch) -> Dict[str, torch.Tensor]:
      
        coords = torch.remainder(batch.full_rep, 1.0)
        free_rep = self.frac_coord_embedder(coords)
        

        at = self.atom_type_embedder(batch.atom_types) 
        wy = self.wyckoff_embedder(batch.wyckoff_ids)         


        x = at + wy + free_rep


        # --- GLOBALS nodes ---
        sg_feat = self.sg_embedder(batch.sg_number)

        
        cell_feat = self.cell_free_proj(batch.full_cell_rad)  # [B, d]

        
        g = sg_feat + cell_feat 
          
        
        g_nodes = g[batch.batch]         # broadcast to nodes -> [N,d]
        
        
        x = x + g_nodes     


        # --- pad to dense for the Transformer ---
        x, token_mask = to_dense_batch(x, batch.batch)         # [B,T,d], [B,T]
        x = self.transformer(x, src_key_padding_mask=(~token_mask))
        x = x[token_mask]                                      # back to [N,d]


        return {
            "x": x,
            "batch": batch.batch,               
            "num_orbits": batch.num_orbits,
        }
