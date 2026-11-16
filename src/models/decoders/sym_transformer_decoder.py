"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from typing import Dict

import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter

def graph_softmax(scores, batch):
    scores = scores - scatter(scores, batch, dim=0, reduce="max")[batch]
    exp = torch.exp(scores)
    Z = scatter(exp, batch, dim=0, reduce="sum")[batch].clamp_min(1e-9)
    return exp / Z

def periodic_unit(x): 
    return (x % 1.0 + 1.0) % 1.0



class SymTransformerDecoder(nn.Module):

    def __init__(
        self,
        max_num_elements=100,
        d_model: int = 1024,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_first: bool = True,
        bias: bool = True,
        num_layers: int = 6,
    ):
        super().__init__()

        self.max_num_elements = max_num_elements
        self.d_model = d_model
        self.num_layers = num_layers

        activation = {
            "gelu": nn.GELU(approximate="tanh"),
            "relu": nn.ReLU(),
        }[activation]
        
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
                bias=bias,
            ),
            norm=nn.LayerNorm(d_model),
            num_layers=num_layers,
        )

        self.atom_types_head = nn.Linear(d_model, max_num_elements, bias=True)
        self.wy_head  = nn.Linear(d_model, 1731)  # logits over Wyckoff tokens
        
        self.xyz_head = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 3))

        self.cell_head = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 6))

        self.pool_score = nn.Linear(d_model, 1)

    def forward(self, encoded_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        x = encoded_batch["x"]                  # [N, d]


        x, token_mask = to_dense_batch(x, encoded_batch["batch"])         # [B,T,d], [B,T]
        
        x = self.transformer.forward(x, src_key_padding_mask=(~token_mask))
        x = x[token_mask]                                # back to [N,d]


        scores = self.pool_score(x).squeeze(-1)  # [N]
        alpha = graph_softmax(scores, encoded_batch["batch"])
        x_global = scatter(x * alpha.unsqueeze(-1), encoded_batch["batch"], dim=0, reduce="sum")  # [B, d]

        
        # Node-level predictions
        atom_types_out = self.atom_types_head(x)
        wy_logits = self.wy_head(x)                            # [N, 1731]
        frac_pred = self.xyz_head(x)                           # [N, 3] -> free positional DOFs
        
       
        cell_pred = self.cell_head(x_global) # [B, 6]


        lengths_pred = cell_pred[:, :3]        # unconstrained 
        angles_pred  = cell_pred[:, 3:]        # unconstrained 



        return {
            "atom_types": atom_types_out,
            "wyckoff": wy_logits,   
            "pos_pred":  frac_pred,    
            "cell_pred": cell_pred,  
            "lengths_pred": lengths_pred,
            "angles_pred": angles_pred
        }
