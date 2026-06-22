from typing import Literal

import torch
from torch import nn

from COMPASS.data.utils.perturbation_datamodule import (
    ObservationNormalizationStatistics,
)


def get_normalization_module(
    key: Literal["standardize", "log_standardize"],
    normalization_stats: ObservationNormalizationStatistics,
):
    if key == "standardize":
        module = StandardizationModule(normalization_stats)
    elif key == "log_standardize":
        module = LogStandardizationModule(normalization_stats)
    else:
        raise ValueError(f"Unknown key {key}")

    return module


class StandardizationModule(nn.Module):
    def __init__(self, normalization_stats: ObservationNormalizationStatistics):
        super().__init__()
        self.register_buffer("mean", normalization_stats.embed_mean)#x_mean
        self.register_buffer("scale", normalization_stats.embed_std)#x_std

    def forward(self, x):
        return (x - self.mean) / self.scale
 

class LogStandardizationModule(nn.Module):
    def __init__(self, normalization_stats: ObservationNormalizationStatistics):
        super().__init__()
        self.register_buffer("log_mean", normalization_stats.log_embed_mean)#log_x_mean
        self.register_buffer("log_scale", normalization_stats.log_embed_std)#log_x_std

    def forward(self, x):
        logx = torch.log(x + 1)
        return (logx - self.log_mean) / self.log_scale
