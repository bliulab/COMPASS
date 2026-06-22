from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from COMPASS.data.utils.perturbation_datamodule import (
    ObservationNormalizationStatistics,
)
from COMPASS.models.utils.gumbel_softmax_bernoulli import (
    GumbelSoftmaxBernoulliStraightThrough,
)
from COMPASS.models.utils.mlp import get_likelihood_mlp
from COMPASS.models.utils.normalization import get_normalization_module
from COMPASS.models.utils.perturbation_conditioner import PerturbationConditioner


class SAMSVAEMeanFieldNormalGuide(nn.Module):
    def __init__(
        self,
        n_latent: int,
        n_treatments: int,
        n_phenos: int,
        basal_encoder_n_layers: int,
        basal_encoder_n_hidden: int,
        basal_encoder_input_normalization: Optional[
            Literal["standardize", "log_standardize"]
        ],
        x_normalization_stats: Optional[ObservationNormalizationStatistics],
        embedding_loc_init_scale: float = 0,
        embedding_scale_init: float = 1,
        mask_init_logits: float = 0,
        gs_temperature: float = 1,
        mean_field_encoder: bool = False,
        pert_dim: int = 0,
        pert_embed_dim: int = 64,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_treatments = n_treatments
        self.n_phenos = n_phenos
        self.mean_field_encoder = mean_field_encoder

        self.param_dict = torch.nn.ParameterDict()
        self.param_dict["q_mask_logits"] = torch.nn.Parameter(
            mask_init_logits * torch.ones((n_treatments, n_latent))
        )
        self.param_dict["q_E_loc"] = torch.nn.Parameter(
            embedding_loc_init_scale * torch.randn((n_treatments, n_latent))
        )
        self.param_dict["q_E_log_scale"] = torch.nn.Parameter(
            np.log(embedding_scale_init) * torch.ones((n_treatments, n_latent))
        )

        if basal_encoder_input_normalization is None:
            self.normalization_module = None
        else:
            assert x_normalization_stats is not None, "Missing x_normalization_stats"
            self.normalization_module = get_normalization_module(
                key=basal_encoder_input_normalization,
                normalization_stats=x_normalization_stats,
            )

        self.pert_conditioner = PerturbationConditioner(
            n_treatments=n_treatments,
            pert_dim=pert_dim,
            pert_embed_dim=pert_embed_dim,
        )

        self.z_basal_encoder = get_likelihood_mlp(
            likelihood_key="normal",
            n_input=n_phenos + pert_embed_dim if mean_field_encoder else n_phenos + n_latent + pert_embed_dim,
            n_output=n_latent,
            n_layers=basal_encoder_n_layers,
            n_hidden=basal_encoder_n_hidden,
            use_batch_norm=False,
        )

        self.register_buffer("gs_temperature", gs_temperature * torch.ones((1,)))
        self.var_eps = 1e-4

    def get_var_keys(self):
        return ["z_basal", "E", "mask"]

    def forward(
        self,
        X: Optional[torch.Tensor] = None,
        D: Optional[torch.Tensor] = None,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        n_particles: int = 1,
    ) -> Tuple[Dict[str, torch.distributions.Distribution], Dict[str, torch.Tensor]]:
        if condition_values is None:
            condition_values = dict()

        guide_distributions: Dict[str, torch.distributions.Distribution] = {}
        guide_samples: Dict[str, torch.Tensor] = {}

        P = condition_values.get("P", None)

        guide_distributions["q_mask"] = GumbelSoftmaxBernoulliStraightThrough(
            temperature=self.gs_temperature,
            logits=self.param_dict["q_mask_logits"],
        )

        guide_distributions["q_E"] = Normal(
            self.param_dict["q_E_loc"],
            torch.exp(self.param_dict["q_E_log_scale"]) + self.var_eps,
        )

        for k in ["mask", "E"]:
            if k not in condition_values:
                guide_samples[k] = guide_distributions[f"q_{k}"].rsample((n_particles,))
            else:
                guide_samples[k] = condition_values[k]

        if X is not None and D is not None:
            encoder_input = X

            if self.normalization_module is not None:
                encoder_input = self.normalization_module(encoder_input)

            encoder_input = torch.unsqueeze(encoder_input, dim=0).expand(
                n_particles, -1, -1
            )

            c_pert = self.pert_conditioner(D=D, P=P)
            c_pert = c_pert.unsqueeze(0).expand(n_particles, -1, -1)

            if not self.mean_field_encoder:
                latent_offset = torch.matmul(
                    D, guide_samples["mask"] * guide_samples["E"]
                )
                encoder_input = torch.cat([encoder_input, latent_offset, c_pert], dim=-1)
            else:
                encoder_input = torch.cat([encoder_input, c_pert], dim=-1)

            guide_distributions["q_z_basal"] = self.z_basal_encoder(encoder_input)
            guide_samples["z_basal"] = guide_distributions["q_z_basal"].rsample()

        if "z_basal" in condition_values:
            guide_samples["z_basal"] = condition_values["z_basal"]

        return guide_distributions, guide_samples