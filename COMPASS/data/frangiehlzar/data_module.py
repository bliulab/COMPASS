from typing import Dict, Literal, Optional, Sequence, Iterable

import anndata
import scanpy as sc
from scanpy.get import _get_obs_rep, _set_obs_rep
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Tuple, Union
from scipy.sparse import issparse
from COMPASS.analysis.average_treatment_effects import (
    estimate_data_average_treatment_effects,
)
from COMPASS.data.utils.batch_statistics import batch_log_mean, batch_log_std
from COMPASS.data.utils.perturbation_datamodule import (
    ObservationNormalizationStatistics,
    PerturbationDataModule,
) 
from COMPASS.data.utils.perturbation_dataset import SCRNASeqTensorPerturbationDataset, MultiOmicsTensorPerturbationDataset


class BaseFrangiehlzarDataModule(PerturbationDataModule):
    def __init__(
        self,
        split_kwargs: Dict,
        encode_combos_as_unique: bool = False,
        batch_size: int = 128,
        highly_variable_genes_only: bool = False,
        # adata_rna: anndata.AnnData = None,
        # adata_adt: anndata.AnnData = None,
        data_path_1: Optional[str] = None,
        data_path_2: Optional[str] = None,
        filter_gene_by_counts_rna: Union[int, bool] = False,
        filter_cell_by_counts_rna: Union[int, bool] = False,
        normalize_total_rna: Union[float, bool] = 1e4,
        use_key_rna: Optional[str] = None,
        result_normed_key_rna: Optional[str] = "X_normed",
        log1p_rna: bool = False,
        result_log1p_key_rna: str = "X_log1p",
        subset_hvg_rna: Union[int, bool] = False,
        hvg_use_key_rna: Optional[str] = None,
        hvg_flavor_rna: str = "seurat_v3",
        filter_gene_by_counts_adt: Union[int, bool] = False,
        filter_cell_by_counts_adt: Union[int, bool] = False,
        normalize_total_adt: Union[float, bool] = 1e4,
        use_key_adt: Optional[str] = None,
        result_normed_key_adt: Optional[str] = "X_normed",
        log1p_adt: bool = False,
        result_log1p_key_adt: str = "X_log1p",
        subset_hvg_adt: Union[int, bool] = False,
        hvg_use_key_adt: Optional[str] = None,
        hvg_flavor_adt: str = "seurat_v3",
    ):
        """
        Base data module for Frangiehlzar dataset tasks (OOD combos and data efficiency)
        Implements all functionality except data splitting, which is implemented
        by each subclass through `get_split_labels`


        Parameters
        ----------
        split_kwargs: dictionary of kwargs passed to `get_split_labels`
        encode_combos_as_unique: if True, encodes combinations as new perturbations
        rather than sum of individual dosages in the combo
        batch_size: batch size for data loaders
        highly_variable_genes_only: filter dataset to highly variable genes
        data_path: path of Norman anndata. If not provided, downloads to current
        directory
        """
        super().__init__()

        self.encode_combos_as_unique = encode_combos_as_unique
        self.batch_size = batch_size
        self.filter_gene_by_counts_rna = filter_gene_by_counts_rna
        self.filter_cell_by_counts_rna = filter_cell_by_counts_rna
        self.normalize_total_rna = normalize_total_rna
        self.use_key_rna = use_key_rna
        self.result_normed_key_rna = result_normed_key_rna
        self.log1p_rna = log1p_rna
        self.result_log1p_key_rna = result_log1p_key_rna
        self.subset_hvg_rna = subset_hvg_rna
        self.hvg_use_key_rna = hvg_use_key_rna
        self.hvg_flavor_rna = hvg_flavor_rna
        key_to_process_rna = self.use_key_rna

        self.filter_gene_by_counts_adt = filter_gene_by_counts_adt
        self.filter_cell_by_counts_adt = filter_cell_by_counts_adt
        self.normalize_total_adt = normalize_total_adt
        self.use_key_adt = use_key_adt
        self.result_normed_key_adt = result_normed_key_adt
        self.log1p_adt = log1p_adt
        self.result_log1p_key_adt = result_log1p_key_adt
        self.subset_hvg_adt = subset_hvg_adt
        self.hvg_use_key_adt = hvg_use_key_adt
        self.hvg_flavor_adt = hvg_flavor_adt
        key_to_process_adt = self.use_key_adt
        
        self.adata = sc.read_h5ad(data_path_1)#data_path_1)
        self.adata_adt = sc.read_h5ad(data_path_2) #data_path_2

        # check gene sets and ensure matching with measurements
        targets = list(set(self.adata.obs['perturbation'].astype(str).unique()))
        # preliminary checks, will use later
        if key_to_process_rna == "X":
            key_to_process_rna = None  # the following scanpy apis use arg None to use X
        is_logged_rna = self.check_logged(self.adata, obs_key=key_to_process_rna)

        if key_to_process_adt == "X":
            key_to_process_adt = None  # the following scanpy apis use arg None to use X
        is_logged_adt = self.check_logged(self.adata_adt, obs_key=key_to_process_adt)
        
        print("Filtering genes by counts for RNA-seq ...")
        raw_X = self.adata.X# if self.use_key is None else adata.layers[self.use_key]
        raw_X = raw_X.toarray() if issparse(raw_X) else raw_X
        gene_counts = raw_X.sum(axis=0)

        keep_gene_mask = (gene_counts >= self.filter_gene_by_counts_rna) | self.adata.var.index.isin(targets)
        n_before = self.adata.shape[1]
        self.adata._inplace_subset_var(keep_gene_mask)
        n_after = self.adata.shape[1]
        print(f"Filtered genes: {n_before - n_after} removed, {n_after} kept for RNA-seq.")
        
        print("Filtering genes by counts for ADT-seq ... ")
        sc.pp.filter_genes(
            self.adata_adt,
            min_counts=self.filter_gene_by_counts_adt
            if isinstance(self.filter_gene_by_counts_adt, int)
            else None,
        )
        # step 2
        if (
            isinstance(self.filter_cell_by_counts_rna, int)
            and self.filter_cell_by_counts_rna > 0
        ):
            print("Filtering cells by counts for scRNA-seq ...")
            sc.pp.filter_cells(
                self.adata,
                min_counts=self.filter_cell_by_counts_rna
                if isinstance(self.filter_cell_by_counts_rna, int)
                else None,
            )
            print("Filtering cells by counts for ADT-seq ...")
            sc.pp.filter_cells(
                self.adata_adt,
                min_counts=self.filter_cell_by_counts_adt
                if isinstance(self.filter_cell_by_counts_adt, int)
                else None,
            )
        # step 3:
        if self.normalize_total_rna:
            print("Normalizing total counts for scRNA-seq...")
            normed_ = sc.pp.normalize_total(
                self.adata,
                target_sum=self.normalize_total_rna
                if isinstance(self.normalize_total_rna, float)
                else None,
                layer=key_to_process_rna,
                inplace=False,
            )["X"]
            key_to_process_rna = self.result_normed_key_rna or key_to_process_rna
            _set_obs_rep(self.adata, normed_, layer=key_to_process_rna)
        if self.normalize_total_adt:
            print("Normalizing total counts for ADT-seq...")
            normed_ = sc.pp.normalize_total(
                self.adata_adt,
                target_sum=self.normalize_total_adt
                if isinstance(self.normalize_total_adt, float)
                else None,
                layer=key_to_process_adt,
                inplace=False,
            )["X"]
            key_to_process_adt = self.result_normed_key_adt or key_to_process_adt
            _set_obs_rep(self.adata_adt, normed_, layer=key_to_process_adt)

        # step 4: log1p
        if self.log1p_rna:
            print("Log1p transforming scRNA-seq ...")
            if is_logged_rna:
                print(
                    "The input data seems to be already log1p transformed. "
                    "Set `log1p=False` to avoid double log1p transform."
                )
            if self.result_log1p_key_rna:
                _set_obs_rep(
                    self.adata,
                    _get_obs_rep(self.adata, layer=key_to_process_rna),
                    layer=self.result_log1p_key_rna,
                )
                key_to_process_rna = self.result_log1p_key_rna
            sc.pp.log1p(self.adata, layer=key_to_process_rna)
        
        if self.log1p_adt:   
            print("Log1p transforming ADT-seq ...")
            if is_logged_adt:
                print(
                    "The input data seems to be already log1p transformed. "
                    "Set `log1p=False` to avoid double log1p transform."
                )
            if self.result_log1p_key_adt:
                _set_obs_rep(
                    self.adata_adt,
                    _get_obs_rep(self.adata_adt, layer=key_to_process_adt),
                    layer=self.result_log1p_key_adt,
                )
                key_to_process_adt = self.result_log1p_key_adt
            sc.pp.log1p(self.adata_adt, layer=key_to_process_adt)
        
        # step 5: subset hvg
        if highly_variable_genes_only:
            print("Subsetting highly variable genes ...")
            sc.pp.highly_variable_genes(
                self.adata,
                #layer=self.hvg_use_key_rna,
                n_top_genes=self.subset_hvg_rna
                if isinstance(self.subset_hvg_rna, int)
                else None,
                flavor=self.hvg_flavor_rna,
                subset=False,
            )
            # Step 2: 手动保留 HVG 和 sgRNA 基因
            keep_hvg = self.adata.var["highly_variable"]
            keep_sgRNA = self.adata.var.index.isin(targets)
            keep_mask = keep_hvg | keep_sgRNA
            print(f"HVG count: {keep_hvg.sum()}, sgRNA rescued: {keep_sgRNA.sum()}, final kept: {keep_mask.sum()}")
            self.adata._inplace_subset_var(keep_mask)

            sc.pp.highly_variable_genes(
                    self.adata_adt,
                    layer=self.hvg_use_key_adt,
                    n_top_genes=self.subset_hvg_adt
                    if isinstance(self.subset_hvg_adt, int)
                    else None,
                    flavor=self.hvg_flavor_adt,
                    subset=True,
                )

        # encode perturbation dosages
        treatment_labels = self.adata.obs["perturbation"].astype(str)
        self.adata.obs['treatment'] = treatment_labels

        # get scRNA-seq observation matrix
        X = torch.from_numpy(self.adata.X.toarray())
        X_adt = torch.from_numpy(self.adata_adt.X.toarray())
        cell_embed = torch.from_numpy(self.adata.obsm['scGPT'])
        
        #是否将组合扰动视为新的扰动
        if not self.encode_combos_as_unique:
            # combinations encoded as application of two individual guides
            D, df_var_info = build_guide_gene_matrix(self.adata)
            self.d_var_info = df_var_info
        else:
            # combinations encoded as new treatments
            D_df = pd.get_dummies(self.adata.obs["treatment"])
            D_df = D_df.drop(columns=["control"])
            self.d_var_info = D_df.T[[]]
            D = torch.from_numpy(D_df.to_numpy().astype(np.float32))

        self.dosage = D

        
        pert_freq = D.sum(0, keepdim=True)  # [1, n_treatments]
        pert_freq = pert_freq / (pert_freq.max() + 1e-8)
        P = torch.load('/home/dataset-local/chengyue/scGPT-main/scMultiomics-perturb/save/dev_Frangiehlzar2021_IFN_single_pancancer-DAR-4000-mask0.15-split-Mar19-12-45/P_gene.pt')
        

        self.adata_adt.obs = self.adata.obs

        self.adata.obs['batch_label'] = np.array(self.adata.obs['perturbation'].values)
        from sklearn.preprocessing import LabelEncoder
        labelencoder = LabelEncoder()
        batch_label = torch.from_numpy(labelencoder.fit_transform(self.adata.obs['batch_label']))

        train_mask = (self.adata.obs["split"] == "train").to_numpy()
        P_train = P[train_mask]
        train_proto = P_train.mean(0, keepdim=True)  # 一个很基础的 prototype
        ood_score = torch.norm(P - train_proto, dim=1)  # 距离越大，越像 far OOD
        q1 = torch.quantile(ood_score[train_mask], 0.50)
        q2 = torch.quantile(ood_score[train_mask], 0.90)
        ood_level = torch.zeros_like(ood_score, dtype=torch.long)
        ood_level[(ood_score > q1) & (ood_score <= q2)] = 1   # near / medium OOD
        ood_level[ood_score > q2] = 2                         # far OOD

        # generate datasets
        ids_tr = self.adata.obs[self.adata.obs["split"] == "train"].index
        X_tr = X[(self.adata.obs["split"] == "train").to_numpy()]
        X_adt_tr = X_adt[(self.adata_adt.obs["split"] == "train").to_numpy()]
        D_tr = D[(self.adata.obs["split"] == "train").to_numpy()]
        embed_tr = cell_embed[(self.adata.obs["split"] == "train").to_numpy()]
        batch_label_tr = batch_label[(self.adata.obs["split"] == "train").to_numpy()]

        ids_val = self.adata.obs[self.adata.obs["split"] == "val"].index
        X_val = X[(self.adata.obs["split"] == "val").to_numpy()]
        X_adt_val = X_adt[(self.adata_adt.obs["split"] == "val").to_numpy()]
        D_val = D[(self.adata.obs["split"] == "val").to_numpy()]
        embed_val = cell_embed[(self.adata.obs["split"] == "val").to_numpy()]
        batch_label_val = batch_label[(self.adata.obs["split"] == "val").to_numpy()]

        ids_test = self.adata.obs[self.adata.obs["split"] == "test"].index
        X_test = X[(self.adata.obs["split"] == "test").to_numpy()]
        X_adt_test = X_adt[(self.adata_adt.obs["split"] == "test").to_numpy()]
        D_test = D[(self.adata.obs["split"] == "test").to_numpy()]
        embed_test = cell_embed[(self.adata.obs["split"] == "test").to_numpy()]
        batch_label_test = batch_label[(self.adata.obs["split"] == "test").to_numpy()]

        # newly add
        P_tr = P[(self.adata.obs["split"] == "train").to_numpy()]
        P_val = P[(self.adata.obs["split"] == "val").to_numpy()]
        P_test = P[(self.adata.obs["split"] == "test").to_numpy()]

        ood_score_tr = ood_score[(self.adata.obs["split"] == "train").to_numpy()]
        ood_score_val = ood_score[(self.adata.obs["split"] == "val").to_numpy()]
        ood_score_test = ood_score[(self.adata.obs["split"] == "test").to_numpy()]

        ood_level_tr = ood_level[(self.adata.obs["split"] == "train").to_numpy()]
        ood_level_val = ood_level[(self.adata.obs["split"] == "val").to_numpy()]
        ood_level_test = ood_level[(self.adata.obs["split"] == "test").to_numpy()]

        self.train_dataset = MultiOmicsTensorPerturbationDataset(
            X=X_tr, X2=X_adt_tr, D=D_tr, ids=ids_tr, embed=embed_tr, batch = batch_label_tr,
            P=P_tr, ood_score=ood_score_tr, ood_level=ood_level_tr
        )
        self.val_dataset = MultiOmicsTensorPerturbationDataset(
            X=X_val, X2=X_adt_val, D=D_val, ids=ids_val, embed=embed_val, batch = batch_label_val,
            P=P_val, ood_score=ood_score_val, ood_level=ood_level_val
        )
        self.test_dataset = MultiOmicsTensorPerturbationDataset(
            X=X_test, X2=X_adt_test, D=D_test, ids=ids_test, embed=embed_test, batch = batch_label_test,
            P=P_test, ood_score=ood_score_test, ood_level=ood_level_test
        )
        self.dataset = MultiOmicsTensorPerturbationDataset(
            X=X, X2=X_adt, D=D, ids=self.adata.obs.index, embed=cell_embed, batch = batch_label,
            P=P, ood_score=ood_score, ood_level=ood_level
        )

        # compute normalization statistics
        x_tr_mean = X_tr.mean(0)
        x_tr_std = X_tr.std(0)
        X_adt_tr_mean = X_adt_tr.mean(0)
        X_adt_tr_std = X_adt_tr.std(0)
        embed_tr_mean = embed_tr.mean(0)
        embed_tr_std = embed_tr.std(0)
        log_x_tr_mean = batch_log_mean(X_tr)
        log_x_tr_std = batch_log_std(X_tr, log_x_tr_mean)
        log_x_adt_tr_mean = batch_log_mean(X_adt_tr)
        log_x_adt_tr_std = batch_log_std(X_adt_tr, log_x_adt_tr_mean)
        log_embed_tr_mean = batch_log_mean(embed_tr)
        log_embed_tr_std = batch_log_std(embed_tr, log_embed_tr_mean)


        self.x_train_statistics = ObservationNormalizationStatistics(
            x_mean=x_tr_mean,
            x_std=x_tr_std,
            x_adt_mean = X_adt_tr_mean,
            x_adt_std = X_adt_tr_std,
            log_x_mean=log_x_tr_mean,
            log_x_std=log_x_tr_std,
            log_x_adt_mean=log_x_adt_tr_mean,
            log_x_adt_std=log_x_adt_tr_std,
            embed_mean=embed_tr_mean,
            embed_std=embed_tr_std,
            log_embed_mean=log_embed_tr_mean,
            log_embed_std=log_embed_tr_std
        )

        # generate unique intervention info dataframe
        df = self.adata.obs.groupby("treatment")["split"].agg(set).reset_index()
        for split in ["train", "val", "test"]:
            df[split] = df["split"].apply(lambda x: split in x)
        df = df.set_index("treatment").drop(columns=["split"])
        self.unique_observed_intervention_df = df # 每种扰动是否出现在train/val/test中，用于判断OOD

        # generate mapping from intervention names to dosages
        self.adata.obs["i"] = np.arange(self.adata.shape[0])
        idx_map = (
            self.adata.obs.drop_duplicates("treatment")
            .set_index("treatment")["i"]
            .to_dict()
        )
        self.unique_intervention_dosage_map = {k: D[v] for k, v in idx_map.items()} # 每种扰动对应的dosage编码

    def check_logged(self, adata: anndata.AnnData, obs_key: Optional[str] = None) -> bool:
        """
        Check if the data is already log1p transformed.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        obs_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        data = _get_obs_rep(adata, layer=obs_key)
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False

        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False

        return True

    @staticmethod
    def _get_split_labels(
        obs: pd.DataFrame,
        split_seed: int,
        # used only by OOD data module
        frac_combinations_train: float = 0,
        frac_combinations_test: float = 0.2,
        # used only by data efficiency data module
        frac_combination_cells_train: float = 0,
    ) -> pd.Series:
        """
        Returns split labels for each cell (series with ID as index, aligned with obs)
        Implemented by subclasses
        """
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_train_perturbation_obs_counts(self) -> torch.Tensor:
        return self.train_dataset.get_dosage_obs_per_dim()

    def get_val_perturbation_obs_counts(self) -> torch.Tensor:
        return self.val_dataset.get_dosage_obs_per_dim()

    def get_test_perturbation_obs_counts(self) -> torch.Tensor:
        return self.test_dataset.get_dosage_obs_per_dim()

    def get_x_var_info(self) -> pd.DataFrame:
        return self.adata.var.copy(), self.adata_adt.var.copy()
    
    def get_embed_var_info(self) -> pd.DataFrame:
        return self.adata.obsm['scGPT'].copy()

    def get_d_var_info(self) -> pd.DataFrame:
        return self.d_var_info.copy()

    def get_obs_info(self) -> pd.DataFrame:
        return self.adata.obs.copy()

    def get_x_train_statistics(self) -> ObservationNormalizationStatistics:
        return self.x_train_statistics

    def get_unique_observed_intervention_info(self) -> pd.DataFrame:
        return self.unique_observed_intervention_df.copy()

    def get_unique_observed_intervention_dosages(
        self, pert_names: Sequence
    ) -> torch.Tensor:
        D = torch.zeros((len(pert_names), self.d_var_info.shape[0]))
        for i, pert_name in enumerate(pert_names):
            D[i] = self.unique_intervention_dosage_map[pert_name]
        return D

    def get_estimated_average_treatment_effects(
        self, method_rna: Literal["mean", "perturbseq"], 
        method_adt: Literal["mean", "adtseq"], split: Optional[str] = None
    ) -> Optional[anndata.AnnData]:
        adata = self.adata
        adata_adt = self.adata_adt
        if split is not None:
            adata = adata[adata.obs["split"] == split]
            adata_adt = adata_adt[adata_adt.obs["split"] == split]
        return estimate_data_average_treatment_effects(
            adata,
            label_col="treatment",
            control_label="control",
            method=method_rna,
        ),estimate_data_average_treatment_effects(
            adata_adt,
            label_col="treatment",
            control_label="control",
            method=method_adt,
        )

    def get_simulated_latent_effects(self) -> Optional[anndata.AnnData]:
        return None


class FrangiehlzarOODCombinationDataModule(BaseFrangiehlzarDataModule):
    def __init__(
        self,
        frac_combinations_train: float,
        frac_combinations_test: float = 0.2,
        split_seed: int = 0,
        encode_combos_as_unique: bool = False,
        batch_size: int = 128,
        highly_variable_genes_only: bool = False,
        data_path_1: Optional[str] = None,
        data_path_2: Optional[str] = None,
        filter_gene_by_counts_rna: Union[int, bool] = False,
        filter_cell_by_counts_rna: Union[int, bool] = False,
        normalize_total_rna: Union[float, bool] = 1e4,
        use_key_rna: Optional[str] = None,
        result_normed_key_rna: Optional[str] = "X_normed",
        log1p_rna: bool = False,
        result_log1p_key_rna: str = "X_log1p",
        subset_hvg_rna: Union[int, bool] = False,
        hvg_use_key_rna: Optional[str] = None,
        hvg_flavor_rna: str = "seurat_v3",
        filter_gene_by_counts_adt: Union[int, bool] = False,
        filter_cell_by_counts_adt: Union[int, bool] = False,
        normalize_total_adt: Union[float, bool] = 1e4,
        use_key_adt: Optional[str] = None,
        result_normed_key_adt: Optional[str] = "X_normed",
        log1p_adt: bool = False,
        result_log1p_key_adt: str = "X_log1p",
        subset_hvg_adt: Union[int, bool] = False,
        hvg_use_key_adt: Optional[str] = None,
        hvg_flavor_adt: str = "seurat_v3",
    ):
        """
        Data module to assess out of distribution generalization to new combinations
        in the Frangiehlzar dataset

        The combinations (more than 2 targeting CRISPR guides) are randomly shuffled
        using `split_seed`. A subset of cells with  `frac_combinations_train` of the combinations
        are included in the train / val splits. A subset of cells with the last
        `frac_combinations_test` of the combinations are included in the test set

        The train / val splits additionally include cells that received 0 or 1
        targeting CRISPR guides

        Therefore, if `frac_combinations_train` < 1 - `frac_combinations_test`, the test set will
        consist of cells that received held out combinations. Additionally, as
        `frac_combinations_train` is increased, additional combinations are added to the
        train / val splits (superset of splits with smaller values).


        Parameters
        ----------
        frac_combinations_train: fraction of combinations to include in train / val set
        frac_combinations_test: fraction of combinations to include in test set
        split_seed: seed used to shuffle / split combinations
        encode_combos_as_unique: if True, represents combinations as new 1-hot perturbation
        batch_size: batch size for data loader
        highly_variable_genes_only: filter dataset to highly variable genes
        data_path: path to Norman anndata
        """
        split_kwargs = dict(
            frac_combinations_train=frac_combinations_train,
            frac_combinations_test=frac_combinations_test,
            split_seed=split_seed
        )
        super().__init__(
            split_kwargs=split_kwargs,
            encode_combos_as_unique=encode_combos_as_unique,
            batch_size=batch_size,
            highly_variable_genes_only=highly_variable_genes_only,
            data_path_1=data_path_1,
            data_path_2=data_path_2,
            filter_gene_by_counts_rna=filter_gene_by_counts_rna,
            filter_cell_by_counts_rna=filter_cell_by_counts_rna,
            normalize_total_rna=normalize_total_rna,
            use_key_rna=use_key_rna,
            result_normed_key_rna=result_normed_key_rna,
            log1p_rna=log1p_rna,
            result_log1p_key_rna=result_log1p_key_rna,
            subset_hvg_rna=subset_hvg_rna,
            hvg_use_key_rna=hvg_use_key_rna,
            hvg_flavor_rna=hvg_flavor_rna,
            filter_gene_by_counts_adt=filter_gene_by_counts_adt,
            filter_cell_by_counts_adt=filter_cell_by_counts_adt,
            normalize_total_adt=normalize_total_adt,
            use_key_adt=use_key_adt,
            result_normed_key_adt=result_normed_key_adt,
            log1p_adt=log1p_adt,
            result_log1p_key_adt=result_log1p_key_adt,
            subset_hvg_adt=subset_hvg_adt,
            hvg_use_key_adt=hvg_use_key_adt,
            hvg_flavor_adt=hvg_flavor_adt,
        )

    @staticmethod
    def _get_split_labels(
        obs: pd.DataFrame,
        split_seed: int,
        # used only by OOD data module
        frac_combinations_train: float = 0, #控制训练集包含的组合比例
        frac_combinations_test: float = 0.2, #控制测试集包含的组合比例
        # used only by data efficiency data module
        frac_combination_cells_train: float = 0,
    ):
        # TODO: how to implement cleanly? Mypy error for changing signature
        # from superclass
        combo_guide_identities = np.sort(
            obs[obs["MOI"] >= 1]["treatment"].astype(str).unique()
        )

        # randomly shuffle combo guide identities using split seed
        rng = np.random.default_rng(split_seed)
        rng.shuffle(combo_guide_identities)

        # select first frac_combinations_train combos for train/val sets
        num_train_combos = int(frac_combinations_train * len(combo_guide_identities))
        train_combos = combo_guide_identities[:num_train_combos]

        # select last frac_combinations_test combos for test set
        num_test_combos = int(
            np.ceil(frac_combinations_test * len(combo_guide_identities))
        )
        test_combos = combo_guide_identities[-num_test_combos:]

        # split combo cells
        # splitting is done before filtering by train / test combos to ensure that samples
        # accumulate across different frac_combinations_train values
        obs_combo = obs[obs["MOI"] >= 1]
        obs_combo_tr_val, obs_combo_test = train_test_split(
            obs_combo,
            test_size=0.2,
            random_state=split_seed,
        )
        obs_combo_tr, obs_combo_val = train_test_split(
            obs_combo_tr_val,
            test_size=0.2,
            random_state=split_seed,
        )
        obs_combo_tr = obs_combo_tr[obs_combo_tr["treatment"].isin(train_combos)]
        obs_combo_val = obs_combo_val[
            obs_combo_val["treatment"].isin(train_combos)
        ]
        obs_combo_test = obs_combo_test[
            obs_combo_test["treatment"].isin(test_combos)
        ]

        # split non-combo cells
        obs_no_combo = obs[obs["MOI"] < 1]
        obs_no_combo_tr, obs_no_combo_val = train_test_split(
            obs_no_combo, test_size=0.2, random_state=split_seed
        )

        # generate full split obs dataframes
        obs_tr = pd.concat([obs_no_combo_tr, obs_combo_tr])
        obs_val = pd.concat([obs_no_combo_val, obs_combo_val])
        obs_test = obs_combo_test

        split_labels = pd.Series(index=obs.index.copy(), data=None)
        split_labels.loc[obs.index.isin(obs_tr.index)] = "train"
        split_labels.loc[obs.index.isin(obs_val.index)] = "val"
        split_labels.loc[obs.index.isin(obs_test.index)] = "test"

        missing_mask = split_labels.isna()
        if missing_mask.any():
            split_labels.loc[missing_mask] = "train"

        return split_labels


class FrangiehlzarDataEfficiencyDataModule(BaseFrangiehlzarDataModule):
    def __init__(
        self,
        frac_combination_cells_train: float,
        split_seed: int = 0,
        encode_combos_as_unique: bool = False,
        batch_size: int = 128,
        highly_variable_genes_only: bool = False,
        data_path_1: Optional[str] = None,
        data_path_2: Optional[str] = None,
        filter_gene_by_counts_rna: Union[int, bool] = False,
        filter_cell_by_counts_rna: Union[int, bool] = False,
        normalize_total_rna: Union[float, bool] = 1e4,
        use_key_rna: Optional[str] = None,
        result_normed_key_rna: Optional[str] = "X_normed",
        log1p_rna: bool = False,
        result_log1p_key_rna: str = "X_log1p",
        subset_hvg_rna: Union[int, bool] = False,
        hvg_use_key_rna: Optional[str] = None,
        hvg_flavor_rna: str = "seurat_v3",
        filter_gene_by_counts_adt: Union[int, bool] = False,
        filter_cell_by_counts_adt: Union[int, bool] = False,
        normalize_total_adt: Union[float, bool] = 1e4,
        use_key_adt: Optional[str] = None,
        result_normed_key_adt: Optional[str] = "X_normed",
        log1p_adt: bool = False,
        result_log1p_key_adt: str = "X_log1p",
        subset_hvg_adt: Union[int, bool] = False,
        hvg_use_key_adt: Optional[str] = None,
        hvg_flavor_adt: str = "seurat_v3",
    ):
        split_kwargs = dict(
            frac_combination_cells_train=frac_combination_cells_train,
            split_seed=split_seed,
        )
        super().__init__(
            split_kwargs=split_kwargs,
            encode_combos_as_unique=encode_combos_as_unique,
            batch_size=batch_size,
            highly_variable_genes_only=highly_variable_genes_only,
            data_path_1=data_path_1,
            data_path_2=data_path_2,
            filter_gene_by_counts_rna=filter_gene_by_counts_rna,
            filter_cell_by_counts_rna=filter_cell_by_counts_rna,
            normalize_total_rna=normalize_total_rna,
            use_key_rna=use_key_rna,
            result_normed_key_rna=result_normed_key_rna,
            log1p_rna=log1p_rna,
            result_log1p_key_rna=result_log1p_key_rna,
            subset_hvg_rna=subset_hvg_rna,
            hvg_use_key_rna=hvg_use_key_rna,
            hvg_flavor_rna=hvg_flavor_rna,
            filter_gene_by_counts_adt=filter_gene_by_counts_adt,
            filter_cell_by_counts_adt=filter_cell_by_counts_adt,
            normalize_total_adt=normalize_total_adt,
            use_key_adt=use_key_adt,
            result_normed_key_adt=result_normed_key_adt,
            log1p_adt=log1p_adt,
            result_log1p_key_adt=result_log1p_key_adt,
            subset_hvg_adt=subset_hvg_adt,
            hvg_use_key_adt=hvg_use_key_adt,
            hvg_flavor_adt=hvg_flavor_adt,
        )

    @staticmethod
    def _get_split_labels(
        obs: pd.DataFrame,
        split_seed: int,
        # used only by OOD data module
        frac_combinations_train: float = 0,
        frac_combinations_test: float = 0.2,
        # used only by data efficiency data module
        frac_combination_cells_train: float = 0,
    ):
        # TODO: how to implement cleanly? Mypy error for changing signature
        # from superclass

        # split combo cells
        # splitting is done before filtering by train / test combos to ensure that samples
        # accumulate across different frac_combinations_train values
        obs_combo = obs[obs["MOI"] >= 1]
        obs_combo_tr_val, obs_combo_test = train_test_split(
            obs_combo,
            test_size=0.2,
            random_state=split_seed,
        )
        obs_combo_tr, obs_combo_val = train_test_split(
            obs_combo_tr_val,
            test_size=0.2,
            random_state=split_seed,
        )
        # further subsample obs_combo_tr to assess data efficiency
        n_combo_tr_cells = int(frac_combination_cells_train * obs_combo_tr.shape[0])
        obs_combo_tr = obs_combo_tr.iloc[:n_combo_tr_cells]

        # split non-combo cells
        obs_no_combo = obs[obs["MOI"] < 1]
        obs_no_combo_tr, obs_no_combo_val = train_test_split(
            obs_no_combo, test_size=0.2, random_state=split_seed
        )

        # generate full split obs dataframes
        obs_tr = pd.concat([obs_no_combo_tr, obs_combo_tr])
        obs_val = pd.concat([obs_no_combo_val, obs_combo_val])
        obs_test = obs_combo_test

        split_labels = pd.Series(index=obs.index.copy(), data=None)
        split_labels.loc[obs.index.isin(obs_tr.index)] = "train"
        split_labels.loc[obs.index.isin(obs_val.index)] = "val"
        split_labels.loc[obs.index.isin(obs_test.index)] = "test"

        missing_mask = split_labels.isna()
        if missing_mask.any():
            split_labels.loc[missing_mask] = "train"

        return split_labels


def get_guide_one_hot_cols(obs: pd.DataFrame):
    guide_one_hot_cols = [
        col
        for col in obs.columns
        if "guide_" in col and col not in ("guide_identity", "guide_ids")
    ]
    return guide_one_hot_cols


def build_guide_gene_matrix(adata: anndata.AnnData) -> np.ndarray:
    """
    Build a multi-hot matrix for guide genes based on guide_id in adata.obs.
    
    Parameters:
    adata (AnnData): AnnData object containing guide_id in adata.obs
    
    Returns:
    np.ndarray: Multi-hot matrix where rows are cells and columns are unique guide genes
    """
    # Step 1: Extract guide_id from adata.obs
    guide_ids = adata.obs.get('perturbation', pd.Series(index=adata.obs.index, dtype=str))#guide_id
    
    # Step 2: Get unique guide genes, handling nan and splitting by semicolon
    all_genes = set()
    for guide_id in guide_ids:
        if guide_id == 'nan' or pd.isna(guide_id):
            continue
        # Split by semicolon and remove suffix after last underscore
        genes = [gene.rsplit('_', 1)[0] for gene in guide_id.split(';')]
        all_genes.update(genes)
    
    # Convert to sorted list for consistent column ordering
    unique_genes = sorted(list(all_genes))
    
    # Step 3: Initialize multi-hot matrix (n_cells x n_unique_genes)
    matrix = np.zeros((adata.n_obs, len(unique_genes)), dtype=np.float32)
    
    # Step 4: Populate the multi-hot matrix
    for i, guide_id in enumerate(guide_ids):
        if guide_id == 'nan' or pd.isna(guide_id):
            continue
        # Get genes for this cell, removing suffix after last underscore
        genes = [gene.rsplit('_', 1)[0] for gene in guide_id.split(';')]
        # Set 1 for each gene present in this cell
        for gene in genes:
            if gene in unique_genes:
                matrix[i, unique_genes.index(gene)] = 1
    D = torch.from_numpy(matrix)
    d_var_info = pd.DataFrame(index = unique_genes)
    
    return D, d_var_info