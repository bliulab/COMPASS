from typing import Dict, List, Optional, Tuple

import torch
from torch.distributions import Bernoulli, Distribution, Normal

from COMPASS.models.utils.mlp import LIKELIHOOD_KEY_DTYPE, get_likelihood_mlp
from COMPASS.models.utils.perturbation_conditioner import PerturbationConditioner


class SAMSVAEModel(torch.nn.Module): 
    def __init__(
        self,
        n_latent: int,
        n_treatments: int,
        n_phenos: int,
        n_phenos_2: int,
        mask_prior_prob: float,
        embedding_prior_scale: float,
        likelihood_key: LIKELIHOOD_KEY_DTYPE,
        likelihood_key_2: LIKELIHOOD_KEY_DTYPE,
        decoder_n_layers: int,
        decoder_n_layers_2: int,
        decoder_n_hidden: int,
        decoder_n_hidden_2: int,
        pert_dim: int = 0,
        pert_embed_dim: int = 64,
        transition_hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_treatments = n_treatments
        self.n_phenos = n_phenos
        self.n_phenos_2 = n_phenos_2
        self.likelihood_key = likelihood_key
        self.likelihood_key_2 = likelihood_key_2

        self.register_buffer("p_E_loc", torch.zeros((n_treatments, n_latent)))
        self.register_buffer(
            "p_E_scale", embedding_prior_scale * torch.ones((n_treatments, n_latent))
        )
        self.register_buffer(
            "p_mask_probs", mask_prior_prob * torch.ones((n_treatments, n_latent))
        )

        self.pert_conditioner = PerturbationConditioner(
            n_treatments=n_treatments,
            pert_dim=pert_dim,
            pert_embed_dim=pert_embed_dim,
        )

        self.transition = torch.nn.Sequential(
            torch.nn.Linear(n_latent + pert_embed_dim, transition_hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(transition_hidden_dim, n_latent),
        )

        self.decoder = get_likelihood_mlp(
            likelihood_key=likelihood_key,
            n_input=n_latent + pert_embed_dim,
            n_output=n_phenos,
            n_layers=decoder_n_layers,
            n_hidden=decoder_n_hidden,
            use_batch_norm=False,
            activation_fn=torch.nn.LeakyReLU,
        )

        self.decoder_2 = get_likelihood_mlp(
            likelihood_key=likelihood_key_2,
            n_input=n_latent + pert_embed_dim,
            n_output=n_phenos_2,
            n_layers=decoder_n_layers_2,
            n_hidden=decoder_n_hidden_2,
            use_batch_norm=False,
            activation_fn=torch.nn.LeakyReLU,
        )

    def get_var_keys(self) -> List[str]:
        return ["z_basal", "E", "mask"]

    def forward(
        self,
        D: torch.Tensor,
        condition_values: Optional[Dict[str, torch.Tensor]] = None,
        n_particles: int = 1,
    ) -> Tuple[Dict[str, Distribution], Dict[str, torch.Tensor]]:
        n = D.shape[0]
        device = D.device

        if condition_values is None:
            condition_values = dict()

        P = condition_values.get("P", None)

        generative_dists = {}
        generative_dists["p_z_basal"] = Normal(
            torch.zeros((n, self.n_latent), device=device),
            torch.ones((n, self.n_latent), device=device),
        )
        generative_dists["p_E"] = Normal(self.p_E_loc, self.p_E_scale)
        generative_dists["p_mask"] = Bernoulli(logits=torch.logit(self.p_mask_probs))

        samples = {}
        for k in self.get_var_keys():
            if condition_values.get(k) is not None:
                value = condition_values[k]
                if len(value.shape) == 2:
                    value = value.unsqueeze(0).expand((n_particles, -1, -1))
                samples[k] = value
            else:
                samples[k] = generative_dists[f"p_{k}"].sample((n_particles,))

        # classical SAMS latent offset
        z_base = samples["z_basal"] + torch.matmul(D, samples["E"] * samples["mask"])
        
        # semantic perturbation transition for OOD extrapolation
        c_pert = self.pert_conditioner(D=D, P=P)  # [n, pert_embed_dim]
        c_pert_expanded = c_pert.unsqueeze(0).expand(n_particles, -1, -1)
        delta_sem = self.transition(torch.cat([z_base, c_pert_expanded], dim=-1))
        z = z_base + delta_sem

        decoder_input = torch.cat([z, c_pert_expanded], dim=-1)

        if self.likelihood_key != "library_nb":
            generative_dists["p_x"] = self.decoder(decoder_input)
        else:
            generative_dists["p_x"] = self.decoder(decoder_input, condition_values["library_size"])

        if self.likelihood_key_2 != "library_nb":
            generative_dists["p_x_2"] = self.decoder_2(decoder_input)
        else:
            generative_dists["p_x_2"] = self.decoder_2(decoder_input, condition_values["library_size_2"])

        samples["x"] = generative_dists["p_x"].sample()
        samples["x_2"] = generative_dists["p_x_2"].sample()
        samples["z"] = z
        samples["delta_sem"] = delta_sem

        return generative_dists, samples
