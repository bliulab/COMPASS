# COMPASS

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Genetic perturbation screens are powerful for dissecting immune regulation. However, RNA signals in complex microenvironments are often sparse, noisy, and highly sensitive to external stimuli. As a result, relying exclusively on transcriptomic readouts often confounds stable functional phenotypes with transient transcriptional noise, restricting the characterization of context-dependent perturbation dynamics. Here, we present Cross-Omic Modeling of Perturbation and State Shifts (COMPASS), a framework that redefines perturbation modeling by mapping context-dependent responses within a joint RNA and surface-protein (ADT) space. COMPASS leverages surface proteins as stable phenotypic anchors and integrates the semantic representation of perturbed genes with the basal microenvironmental state. This design enables the model to predict how the same genetic perturbation drives distinct, microenvironment-specific cross-omic state shifts. Across comprehensive and rigorous out-of-distribution benchmarks, including unseen perturbations, unseen microenvironments and their combined hold-out settings, COMPASS improves perturbation-effect recovery in both RNA and protein modalities. The resulting predictions also reveal interpretable regulatory programs, including attenuation of immune-responsive surface phenotypes after JAK/STAT-axis disruption and a shift toward E2F1-associated proliferation. Our results demonstrate that COMPASS enables high-resolution in-silico prioritization of genetic perturbations from multi-omics screens, facilitating the discovery of candidate therapeutic regulators that govern immune-state transitions in complex microenvironments.
![The framework plot of COMPASS](https://github.com/bliulab/COMPASS/blob/main/modelplot.png)

## Contents

- [Installation](#installation)
- [Data And Paths](#data-and-paths)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Outputs](#outputs)
- [License](#license)

## Installation

COMPASS is tested on Linux with Python 3.9. GPU training is recommended for full experiments.

Create the environment from the provided conda file:

```bash
conda env create -f environment.yml
conda activate COMPASS
pip install -e .
```

If the `COMPASS` environment already exists locally, activate it and install the repository in editable mode:

```bash
conda activate COMPASS
pip install -e .
```

The `COMPASS` environment includes the core runtime stack:

```text
python 3.9
torch 2.8
pytorch-lightning 2.6
scanpy 1.10
anndata 0.10
numpy 1.26
pandas 2.3
scikit-learn 1.6
pyro-ppl 1.9
```

Weights & Biases logging is optional. Set `use_wandb: False` in YAML configs for local CSV logging under `results/`.

## Data And Paths

Configuration files use environment variables instead of machine-specific absolute paths. Before training or evaluation, point these variables to your local data and perturbation-embedding directories:

```bash
export COMPASS_DATA_DIR=/path/to/dataset
export COMPASS_EMBEDDING_DIR=/path/to/perturbation-embeddings
```

Expected layout:

```text
$COMPASS_DATA_DIR/
    RNA.h5ad
    protein.h5ad

$COMPASS_EMBEDDING_DIR/
    P_gene.pt
```

The YAML parser expands both `${ENV_VAR}` and `~`, so paths can be configured either through environment variables or by editing the YAML values directly.

## Quick Start

Run the Frangiehlzar OOD training config:

```bash
conda activate COMPASS
export COMPASS_DATA_DIR=/path/to/dataset
export COMPASS_EMBEDDING_DIR=/path/to/perturbation-embeddings
python train.py --config COMPASS/data/frangiehlzar/sams_vae_ood.yaml
```

Run the PapalexiSatija OOD training config:

```bash
python train.py --config COMPASS/data/PapalexiSatija2021/sams_vae_ood.yaml
```

Training writes logs, checkpoints and metrics to `results/<experiment-name>/`.

## Training

The main entry point is `train.py`:

```bash
python train.py --config <config.yaml>
```

Important config fields:

```yaml
name: sams_vae_frangieh_IFN-iid
use_wandb: False
seed: 0
max_steps: 30000
gradient_clip_norm: 100

data_module: FrangiehlzarOODCombinationDataModule
data_module_kwargs.data_path_1: ${COMPASS_DATA_DIR}/Frangiehlzar/RNA_IFN_iid-new.h5ad
data_module_kwargs.data_path_2: ${COMPASS_DATA_DIR}/Frangiehlzar/protein_IFN_iid-new.h5ad
data_module_kwargs.perturbation_embedding_path: ${COMPASS_EMBEDDING_DIR}/P_gene.pt

model: SAMSVAEModel
guide: SAMSVAECorrelatedNormalGuide
loss_module: SAMSVAE_ELBOLossModule
predictor: SAMSVAEPredictor
```

For a quick smoke run, copy a config and add:

```yaml
fast_dev_run: True
val_iwelbo_n_particles: 2
eval_ate_n_particles: 10
```

`train.py` automatically evaluates the best checkpoint at the end of training using `evaluate_checkpoint(...)`.

## Evaluation

Evaluate a finished experiment directory:

```bash
python eval.py \
  --experiment_path results/<experiment-name> \
  --perturbseq_rna True \
  --batch_size 512 \
  --ate_n_particles 2500
```

`eval.py` loads the saved Lightning checkpoint, reconstructs the configured data module and reports RNA and ADT average-treatment-effect metrics.

## Outputs

Each local training run creates a unique directory under `results/`:

```text
results/<experiment-name>/
  checkpoints/
    best-epoch=...-step=....ckpt
  lightning_logs/
  summary.csv
  metric_control-cancer.csv
  metric_adt_control-cancer.csv
```

`summary.csv` stores the validation IWELBO and best checkpoint path. The metric CSV files store RNA and ADT evaluation summaries.

## License

This project is covered under the MIT License.
