from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class PerturbationConditioner(nn.Module):
    """
    Fuse perturbation identity/dosage D and semantic perturbation feature P.
    Both D and P can contribute.
    """

    def __init__(
        self,
        n_treatments: int,
        pert_dim: int,
        pert_embed_dim: int,
        d_hidden_dim: int = 128,
        p_hidden_dim: int = 128,
        fusion_hidden_dim: int = 128,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.n_treatments = n_treatments
        self.pert_dim = pert_dim
        self.pert_embed_dim = pert_embed_dim

        # encode D
        self.d_encoder = nn.Sequential(
            nn.Linear(n_treatments, d_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden_dim, pert_embed_dim),
        )

        # encode P
        self.p_encoder = nn.Sequential(
            nn.Linear(pert_dim, p_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(p_hidden_dim, pert_embed_dim),
        )

        # fuse [c_D, c_P]
        fusion_layers = [
            nn.Linear(2 * pert_embed_dim, fusion_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, pert_embed_dim),
        ]
        if use_layernorm:
            fusion_layers.append(nn.LayerNorm(pert_embed_dim))
        self.fusion = nn.Sequential(*fusion_layers)

        # optional learnable balance
        self.alpha = nn.Parameter(torch.tensor(1.0))  # D branch
        self.beta = nn.Parameter(torch.tensor(1.0))   # P branch

    def forward(
        self,
        D: Optional[torch.Tensor] = None,
        P: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if D is None and P is None:
            raise ValueError("At least one of D or P must be provided.")

        if D is not None:
            c_d = self.d_encoder(D.float())
        else:
            if P is None:
                raise ValueError("Both D and P are None.")
            c_d = torch.zeros(
                P.shape[0], self.pert_embed_dim,
                device=P.device, dtype=P.dtype
            )

        if P is not None:
            # 推荐先 normalize，避免 P 范数太大压过 D
            P = F.normalize(P.float(), p=2, dim=1)
            c_p = self.p_encoder(P)
        else:
            c_p = torch.zeros(
                D.shape[0], self.pert_embed_dim,
                device=D.device, dtype=D.dtype
            )

        # 两路都保留
        c_d = self.alpha * c_d
        c_p = self.beta * c_p

        c = self.fusion(torch.cat([c_d, c_p], dim=-1))
        return c