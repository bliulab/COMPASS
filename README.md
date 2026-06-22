# COMPASS

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

`COMPASS` is a python package containing tools for single-cell multi-omics perturbation prediction.

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Usage](#usage)
- [License](#license)

# Overview
Genetic perturbation screens are powerful for dissecting immune regulation. However, RNA signals in complex microenvironments are often sparse, noisy, and highly sensitive to external stimuli. As a result, relying exclusively on transcriptomic readouts often confounds stable functional phenotypes with transient transcriptional noise, restricting the characterization of context-dependent perturbation dynamics. Here, we present Cross-Omic Modeling of Perturbation and State Shifts (COMPASS), a framework that redefines perturbation modeling by mapping context-dependent responses within a joint RNA and surface-protein (ADT) space. COMPASS leverages surface proteins as stable phenotypic anchors and integrates the semantic representation of perturbed genes with the basal microenvironmental state. This design enables the model to predict how the same genetic perturbation drives distinct, microenvironment-specific cross-omic state shifts. Across comprehensive and rigorous out-of-distribution benchmarks, including unseen perturbations, unseen microenvironments and their combined hold-out settings, COMPASS improves perturbation-effect recovery in both RNA and protein modalities. The resulting predictions also reveal interpretable regulatory programs, including attenuation of immune-responsive surface phenotypes after JAK/STAT-axis disruption and a shift toward E2F1-associated proliferation. Our results demonstrate that COMPASS enables high-resolution in-silico prioritization of genetic perturbations from multi-omics screens, facilitating the discovery of candidate therapeutic regulators that govern immune-state transitions in complex microenvironments.
![The framework plot of COMPASS](https://github.com/bliulab/COMPASS/blob/main/modelplot.png)

# System Requirements
## Hardware requirements
`COMPASS` package requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements
### OS requirements
This package is supported for *Linux*. The package has been tested on the following systems:
* Linux: Ubuntu 18.04

### Python Dependencies
`COMPASS` mainly depends on the Python scientific stack.
    numpy
    pytorch
    scanpy
    pandas
    scikit-learn
For specific setting, please see <a href="https://github.com/bliulab/COMPASS/blob/main/environment.yml">environment</a>.

# Installation Guide
## Install from Conda
    conda env create -f environment.yml

# Usage
`COMPASS` is a method for single-cell multi-omics perturbation prediction, which can be used to:
* Unseen perturbation target effect prediction. The example can be seen in the <a href="https://github.com/bliulab/COMPASS/blob/main/COMPASS/data/PapalexiSatija2021/sams_vae_ood.yaml">tutorial_ood</a>
* Cross-microenvironment perturbation effect prediction. The example can be seen in the <a href="https://github.com/bliulab/COMPASS/blob/main/COMPASS/data/frangiehlzar/sams_vae_ood.yaml">tutorial_oos</a>

# License
This project is covered under the **MIT License**.
