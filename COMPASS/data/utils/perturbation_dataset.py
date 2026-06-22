from typing import Iterable, Optional, TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset


class PerturbationDataSample(TypedDict):
    idx: int
    X: torch.Tensor
    D: torch.Tensor


class SCRNASeqPerturbationDataSample(PerturbationDataSample):
    library_size: int


class MultiOmicsPerturbationDataSample(PerturbationDataSample):
    X2: torch.Tensor
    embed: torch.Tensor
    P: torch.Tensor
    batch: torch.Tensor
    ood_score: torch.Tensor
    ood_level: torch.Tensor


class PerturbationDataset(Dataset):
    def __getitem__(self, idx: int) -> PerturbationDataSample:
        raise NotImplementedError

    def get_dosage_obs_per_dim(self) -> torch.Tensor:
        raise NotImplementedError

    def convert_idx_to_ids(self, idx: np.array) -> np.array:
        raise NotImplementedError


class TensorPerturbationDataset(PerturbationDataset):
    def __init__(
        self,
        X: torch.Tensor,
        D: torch.Tensor,
        embed: torch.Tensor = None,
        X2: torch.Tensor = None,
        P: torch.Tensor = None,
        ids: Optional[Iterable] = None,
        batch: torch.Tensor = None,
        phase: torch.Tensor = None,
        ood_score: torch.Tensor = None,
        ood_level: torch.Tensor = None,
    ):
        self.X = X
        self.X2 = X2
        self.D = D
        self.embed = embed
        self.P = P
        self.batch = batch
        self.phase = phase
        self.ood_score = ood_score
        self.ood_level = ood_level

        if ids is None:
            self.ids = np.arange(len(X))
        else:
            self.ids = np.array(ids)

        self.D_obs_per_dim = (self.D != 0).sum(0)
        self.library_size = self.X.sum(1)
        self.library_size_2 = self.X2.sum(1) if self.X2 is not None else None

    def __getitem__(self, idx: int) -> PerturbationDataSample:
        return dict(idx=idx, X=self.X[idx], D=self.D[idx])

    def __len__(self):
        return len(self.X)

    def get_dosage_obs_per_dim(self):
        return self.D_obs_per_dim

    def convert_idx_to_ids(self, idx: np.array) -> np.array:
        return self.ids[idx]


class SCRNASeqTensorPerturbationDataset(TensorPerturbationDataset):
    def __getitem__(self, idx: int) -> SCRNASeqPerturbationDataSample:
        return dict(
            idx=idx, X=self.X[idx], D=self.D[idx], library_size=self.library_size[idx]
        )


class MultiOmicsTensorPerturbationDataset(TensorPerturbationDataset):
    def __getitem__(self, idx: int) -> MultiOmicsPerturbationDataSample:
        out = dict(
            idx=idx,
            X=self.X[idx],
            D=self.D[idx],
            X2=self.X2[idx],
            embed=self.embed[idx],
            library_size=self.library_size[idx],
            library_size_2=self.library_size_2[idx],
            batch=self.batch[idx],
        )

        if self.P is not None:
            out["P"] = self.P[idx]
        if self.ood_score is not None:
            out["ood_score"] = self.ood_score[idx]
        if self.ood_level is not None:
            out["ood_level"] = self.ood_level[idx]
        if self.phase is not None:
            out["phase"] = self.phase[idx]

        return out