"""
Script to generate test set evaluation metrics from a training run

Usage:
python [experiment_path] [--wandb] [--perturbseq] [--batch_size {int}]
"""

import argparse
import os
from os.path import basename, join, splitext
from typing import Any, Dict, Literal, Optional, List

import numpy as np
import pandas as pd
import torch
import wandb
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import rbf_kernel

from COMPASS.data.utils.anndata import align_adatas
from COMPASS.models.utils.perturbation_lightning_module import (
    TrainConfigPerturbationLightningModule,
)


def compute_pairwise_corrs(df):
    corr = df.corr().rename_axis(index='lhs', columns='rhs')
    return (
        corr
        .where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
        .set_index(['lhs', 'rhs'])
        .squeeze()
        .rename()
    )

def mmd_distance(x, y, gamma):
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()

def cosine_distance_between_vectors(x, y, eps=1e-8):
    """
    Compute cosine distance between two vectors:
    cosine distance = 1 - cosine similarity
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    numerator = np.dot(x, y)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)

    if denominator < eps:
        return np.nan

    cosine_similarity = numerator / denominator
    cosine_distance = 1.0 - cosine_similarity

    return cosine_distance

def compute_scalar_mmd(target, transport, gammas=None):
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    def safe_mmd(*args):
        try:
            mmd = mmd_distance(*args)
        except ValueError:
            mmd = np.nan
        return mmd

    return np.mean(list(map(lambda x: safe_mmd(target, transport, x), gammas)))

def compute_mmd_loss(lhs, rhs, gammas):
    return np.mean([mmd_distance(lhs, rhs, g) for g in gammas])

# newly add
def _get_dataset_condition_values(data_module, device):
    """
    Collect full-dataset condition values for direct model(...) calls.
    """
    dataset = data_module.dataset
    cond = {}

    if hasattr(dataset, "library_size") and dataset.library_size is not None:
        cond["library_size"] = dataset.library_size.to(device)

    if hasattr(dataset, "library_size_2") and dataset.library_size_2 is not None:
        cond["library_size_2"] = dataset.library_size_2.to(device)

    if hasattr(dataset, "P") and dataset.P is not None:
        cond["P"] = dataset.P.to(device)

    if hasattr(dataset, "batch") and dataset.batch is not None:
        cond["batch"] = dataset.batch.to(device)

    return cond


def evaluate_checkpoint(
    checkpoint_path: str,
    average_treatment_effect_method_rna: Literal["mean", "perturbseq"],
    average_treatment_effect_method_adt: Literal["mean", "adtseq"],
    batch_size: int = 500,
    ate_n_particles: int = 2500
) -> Dict[str, Any]:
    """
    Compute test set metrics for a given checkpoint


    Parameters
    ----------
    checkpoint_path: path to checkpoint
    average_treatment_effect_method: method to compute average treatment effect. "perturbseq"
        normalizes for library size and applies log transform before assessing effect
    batch_size: batch size to use for IWELBO computation

    Returns
    -------
    dictionary with test set metrics
    """
    lightning_module = load_checkpoint(checkpoint_path)
    data_module = lightning_module.get_data_module()
    predictor = lightning_module.predictor

    d_var_info = data_module.get_d_var_info()
    
    model = predictor.model

    metrics = {}
    metrics_adt = {}

    # compute test set IWELBO
    test_loader = DataLoader(
        data_module.test_dataloader().dataset,
        batch_size=batch_size,
    )
    test_iwelbo_df = predictor.compute_predictive_iwelbo(
        loaders=test_loader, n_particles=100
    )
    test_iwelbo = test_iwelbo_df["IWELBO"].mean()
    metrics["test/IWELBO"] = test_iwelbo

    # assess correlation between estimated average treatment effects from model and data
    data_ate, data_ate_adt = data_module.get_estimated_average_treatment_effects(
        method_rna=average_treatment_effect_method_rna,
        method_adt=average_treatment_effect_method_adt
    )
    
    device = next(model.parameters()).device
    library_size = data_module.dataset.library_size.to(device)
    library_size_2 = data_module.dataset.library_size_2.to(device)
    # newly add
    full_condition_values = _get_dataset_condition_values(data_module, device)
    _, samples = model(D=data_module.dosage.to(device), 
                       condition_values=full_condition_values,#dict(library_size=library_size,
                                                              #library_size_2=library_size_2), 
                       n_particles=1)
    
    rna_recons = samples["x"].squeeze(0).detach().cpu().numpy()
    obs = data_module.get_obs_info()
    var_rna, var_adt = data_module.get_x_var_info()
    adt_recons = samples['x_2'].squeeze(0).detach().cpu().numpy()

    import anndata
    rna_recons = anndata.AnnData(X = rna_recons, obs = obs, var = var_rna)
    adt_recons = anndata.AnnData(X = adt_recons, obs = obs, var = var_adt)

    if data_ate is not None:
        model_ate, model_ate_adt = predictor.estimate_average_effects_data_module(
            data_module=data_module,
            control_label=data_ate.uns["control"],
            method_rna=average_treatment_effect_method_rna,
            method_adt=average_treatment_effect_method_adt,
            n_particles=ate_n_particles,
            condition_values=dict(library_size=data_module.dataset.library_size.mean().reshape(1),#,  10000 * torch.ones((1,))
                                  library_size_2=data_module.dataset.library_size_2.mean().reshape(1),
                                  P = data_module.dataset.P.mean(0, keepdim=True)), #  data_module.dataset.library_size_2.mean().reshape(1))
            batch_size=batch_size,
        )
        
        eval_rna, eval_adt = predictor.sample_observations_data_module(data_module = data_module,
                                                                       n_particles = ate_n_particles,
                                                                       condition_values=dict(library_size=data_module.dataset.library_size.mean().reshape(1),#
                                                                                             library_size_2=data_module.dataset.library_size_2.mean().reshape(1),
                                                                                             P = data_module.dataset.P.mean(0, keepdim=True))) # 10000 * torch.ones((1,))
        
        data_ate, model_ate = align_adatas(data_ate, model_ate)

        intervention_info = data_module.get_unique_observed_intervention_info()
        
        metrics["ATE_n_particles"] = ate_n_particles
        
        # compute average treatment effect metrics for all perturbations
        ate_metrics_all_splits = get_ate_metrics(data_ate, model_ate, deg=False)
        for k, v in ate_metrics_all_splits.items():
            metrics[f"{k}-all"] = v

        # compute average treatment effect metrics for perturbations available
        # in each split
        for split in ["train", "val", "test"]:
            split_perturbations = intervention_info[intervention_info[split]].index
            idx = data_ate.obs.index.isin(split_perturbations)
            ate_metrics_split = get_ate_metrics(data_ate[idx], model_ate[idx], deg=True)
            for k, v in ate_metrics_split.items():
                metrics[f"{k}-{split}"] = v
        
        # for adt
        data_ate_adt, model_ate_adt = align_adatas(data_ate_adt, model_ate_adt)

        intervention_info = data_module.get_unique_observed_intervention_info()

        metrics_adt["ATE_n_particles"] = ate_n_particles

        # compute average treatment effect metrics for all perturbations
        ate_metrics_all_splits_adt = get_ate_metrics(data_ate_adt, model_ate_adt, deg=True)
        for k, v in ate_metrics_all_splits_adt.items():
            metrics_adt[f"{k}-all"] = v

        # compute average treatment effect metrics for perturbations available
        # in each split
        for split in ["train", "val", "test"]:
            split_perturbations = intervention_info[intervention_info[split]].index
            idx = data_ate_adt.obs.index.isin(split_perturbations)
            ate_metrics_split_adt = get_ate_metrics(data_ate_adt[idx], model_ate_adt[idx], deg=True)
            for k, v in ate_metrics_split_adt.items():
                metrics_adt[f"{k}-{split}"] = v


    return metrics, metrics_adt


def get_ate_metrics(data_ate, model_ate, deg, n=20):
    metrics = {}
    gammas = np.logspace(1, -3, num=50)
    if deg:
        top_20_idx = np.argpartition(np.abs(data_ate.X.copy()), data_ate.X.shape[1] - n)[
            :, -n:
        ]
        # evaluate correlation / R2 across top 20 DE genes per perturbation
        count_X = np.take_along_axis(data_ate.X.copy(), top_20_idx, axis=-1)
        count_Y = np.take_along_axis(model_ate.X.copy(), top_20_idx, axis=-1)
        x = np.take_along_axis(data_ate.X.copy(), top_20_idx, axis=-1).flatten()
        y = np.take_along_axis(model_ate.X.copy(), top_20_idx, axis=-1).flatten()
        
        
        metrics["ATE_pearsonr_top20"] = pearsonr(x, y)[0]
        metrics["ATE_spearman-top20"] = spearmanr(x, y, nan_policy="omit")
        cosine_distance = cosine_distance_between_vectors(x, y, eps=1e-8)
        cosine_similarity = 1.0 - cosine_distance if not np.isnan(cosine_distance) else np.nan
        metrics["ATE_cosine-top20"] = cosine_similarity

    x = data_ate.X.flatten()
    y = model_ate.X.flatten()
    
    metrics["ATE_pearsonr"] = pearsonr(x, y)[0]
    metrics["ATE_spearman"] = spearmanr(x, y, nan_policy="omit")
    cosine_distance = cosine_distance_between_vectors(x, y, eps=1e-8)
    cosine_similarity = 1.0 - cosine_distance if not np.isnan(cosine_distance) else np.nan
    metrics["ATE_cosine"] = cosine_similarity

    return metrics


def evaluate_local_experiment(
    experiment_path: str,
    average_treatment_effect_method_rna: Literal["mean", "perturbseq"],
    average_treatment_effect_method_adt: Literal["mean", "adtseq"],
    batch_size: int = 128,
    ate_n_particles: int = 2500,
):
    """
    Compute and save evaluation metrics for checkpoint with best eval loss in
     local experiment to `{experiment_path}/test_metrics.csv`

    Parameters
    ----------
    experiment_path: path to experiment (typically in results/ directory)
    average_treatment_effect_method
    batch_size: batch size used during IWELBO computation
    """
    checkpoint_names = os.listdir(join(experiment_path, "checkpoints"))
    # TODO: add better logic if needed
    best_checkpoints = [x for x in checkpoint_names if x[:4] == "best"]
    assert len(best_checkpoints) == 1
    checkpoint_path = join(experiment_path, "checkpoints", best_checkpoints[0])
    checkpoint_name = splitext(basename(checkpoint_path))[0]

    metrics, metrics_adt = evaluate_checkpoint(
        checkpoint_path,
        average_treatment_effect_method_rna=average_treatment_effect_method_rna,
        average_treatment_effect_method_adt=average_treatment_effect_method_adt,
        batch_size=batch_size,
        ate_n_particles=ate_n_particles,
    )
    metrics["checkpoint"] = checkpoint_name

    metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items()}).T
    metrics_path = join(experiment_path, "test_metrics.csv")
    metrics_df.to_csv(metrics_path)

    metrics_adt_df = pd.DataFrame({k: [v] for k, v in metrics_adt.items()}).T
    metrics_path = join(experiment_path, "test_metrics_adt.csv")
    metrics_adt_df.to_csv(metrics_path)


def evaluate_local_checkpoint(
    checkpoint_path: str,
    average_treatment_effect_method_rna: Literal["mean", "perturbseq"],
    average_treatment_effect_method_adt: Literal["mean", "adtseq"],
    batch_size: int = 128,
    ate_n_particles: int = 2500
):
    """
    Compute and save evaluation metrics specified checkpoint_path,
    saves results to {checkpoint_path}_test_metrics.csv

    Parameters
    ----------
    experiment_path: path to experiment (typically in results/ directory)
    average_treatment_effect_method
    batch_size: batch size used during IWELBO computation
    """
    checkpoint_base = splitext(checkpoint_path)[0]
    checkpoint_name = splitext(basename(checkpoint_path))[0]

    metrics, metrics_adt = evaluate_checkpoint(
        checkpoint_path,
        average_treatment_effect_method_rna=average_treatment_effect_method_rna,
        average_treatment_effect_method_adt=average_treatment_effect_method_adt,
        batch_size=batch_size,
        ate_n_particles=ate_n_particles
    )
    metrics["checkpoint"] = checkpoint_name

    metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items()}).T
    metrics_path = checkpoint_base + "_test_metrics.csv"
    metrics_df.to_csv(metrics_path)

    metrics_adt_df = pd.DataFrame({k: [v] for k, v in metrics_adt.items()}).T
    metrics_path = checkpoint_base + "_test_metrics_adt.csv"
    metrics_adt_df.to_csv(metrics_path)


def evaluate_wandb_experiment(
    experiment_path: str,
    average_treatment_effect_method: Literal["mean", "perturbseq"],
    batch_size: int = 128,
    ate_n_particles: int = 2500,
):
    """
    Compute and save evaluation metrics for checkpoint with best eval loss
    Metrics are saved to wandb run summary
    """
    api = wandb.Api()
    run = api.run(experiment_path)

    # TODO: improve logic if needed
    run_file_paths = [x.name for x in run.files()]
    best_checkpoint_paths = [
        x
        for x in run_file_paths
        if os.path.split(x)[0] == "checkpoints" and "best" in x
    ]
    assert len(best_checkpoint_paths) == 1
    wandb_file = run.file(best_checkpoint_paths[0])

    # download checkpoint
    basedir = run.name + "/"
    os.makedirs(basedir, exist_ok=True)
    checkpoint_path = wandb_file.download(root=basedir, replace=True).name

    metrics = evaluate_checkpoint(
        checkpoint_path,
        average_treatment_effect_method=average_treatment_effect_method,
        batch_size=batch_size,
        ate_n_particles=ate_n_particles,
    )

    # save metrics to run summary
    for k in metrics:
        run.summary[k] = metrics[k]

    run.summary.update()


def load_checkpoint(checkpoint_path: str):
    lightning_module = TrainConfigPerturbationLightningModule.load_from_checkpoint(
        checkpoint_path, weights_only = False
    )
    return lightning_module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--experiment_path", type=str)
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--perturbseq_rna", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--ate_n_particles", type=int, default=2500)

    args = parser.parse_args()
    
    method_rna: Literal["mean", "perturbseq"] = "perturbseq" if args.perturbseq_rna else "mean"
    method_adt = "adtseq" 
    if args.wandb:
        evaluate_wandb_experiment(
            args.experiment_path,
            method_rna,
            batch_size=args.batch_size,
            ate_n_particles=args.ate_n_particles,
        )
    elif os.path.isdir(args.experiment_path):
        evaluate_local_experiment(
            args.experiment_path,
            method_rna,
            method_adt,
            batch_size=args.batch_size,
            ate_n_particles=args.ate_n_particles,
        )
    else:
        evaluate_local_checkpoint(
            args.experiment_path,
            method_rna,
            method_adt,
            batch_size=args.batch_size,
            ate_n_particles=args.ate_n_particles,
        )
