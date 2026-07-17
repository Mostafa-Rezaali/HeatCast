# HeatCast

<p align="center">
  <img src="docs/assets/model_output_heatcast.gif" width="100%" alt="Observed and HeatCast-hindcast 14-day mean temperature anomaly fields over CONUS for week 3-4 leads 15 through 28">
</p>

[![License: MIT](https://img.shields.io/github/license/Mostafa-Rezaali/HeatCast)](https://github.com/Mostafa-Rezaali/HeatCast/blob/main/LICENSE)
[![CI](https://github.com/Mostafa-Rezaali/HeatCast/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/Mostafa-Rezaali/HeatCast/actions/workflows/python-package.yml)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-distributional%20mesh%20GNN-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.9.1-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Slurm](https://img.shields.io/badge/HPC-Slurm-0D73BA)](https://slurm.schedmd.com/)
[![pytest](https://img.shields.io/badge/tests-pytest%20passing-0A9EDC?logo=pytest&logoColor=white)](https://github.com/Mostafa-Rezaali/HeatCast/actions/workflows/python-package.yml)
[![Flake8](https://img.shields.io/badge/lint-Flake8%20checked-4C9C39)](https://github.com/Mostafa-Rezaali/HeatCast/actions/workflows/python-package.yml)
[![Integrity](https://img.shields.io/badge/repository%20integrity-checked-brightgreen)](https://github.com/Mostafa-Rezaali/HeatCast/blob/main/src/repo_integrity.py)
<br>
[![Forecast window](https://img.shields.io/badge/W34-leads%20%2B15...%2B28-6A5ACD)](https://github.com/Mostafa-Rezaali/HeatCast/blob/main/slurm/submit_w34_tube_all.slurm)
[![Cross-validation](https://img.shields.io/badge/cross--validation-5%20year--disjoint%20folds-2E8B57)](https://github.com/Mostafa-Rezaali/HeatCast/blob/main/README.md#exceedance-definition)
[![Parameters](https://img.shields.io/badge/parameters-4%2C637%2C891-8A2BE2)](https://github.com/Mostafa-Rezaali/HeatCast/blob/main/README.md#w34-model-configuration)
[![NetCDF4](https://img.shields.io/badge/output-NetCDF4-4B8BBE)](https://unidata.github.io/netcdf4-python/)
[![MATLAB](https://img.shields.io/badge/export-MATLAB%20compatible-F7941D?logo=mathworks&logoColor=white)](https://github.com/Mostafa-Rezaali/HeatCast/tree/main/matlab)
[![Last commit](https://img.shields.io/github/last-commit/Mostafa-Rezaali/HeatCast?branch=main)](https://github.com/Mostafa-Rezaali/HeatCast/commits/main)
[![Repository size](https://img.shields.io/github/repo-size/Mostafa-Rezaali/HeatCast)](https://github.com/Mostafa-Rezaali/HeatCast)
[![Stars](https://img.shields.io/github/stars/Mostafa-Rezaali/HeatCast?style=flat)](https://github.com/Mostafa-Rezaali/HeatCast/stargazers)
[![Open issues](https://img.shields.io/github/issues/Mostafa-Rezaali/HeatCast)](https://github.com/Mostafa-Rezaali/HeatCast/issues)

HeatCast is a GraphCast-style mesh graph neural network for probabilistic
CONUS week-3--4 (W34) warm-season T2max prediction. The reference model is the
**HeatCast-W34 Distributional Mesh GNN**:

- direct, single-pass inference rather than CFM sampling;
- a 14-lead tube covering `t+15...t+28`;
- a distributional head producing per-pixel mean and sigma;
- Gaussian CRPS training;
- five-fold, year-disjoint cross-validation; and
- held-out calibration and evaluation for W34 q95 exceedance probabilities.

The model has 4,637,891 trainable parameters in the reference W34
configuration. Large datasets and trained checkpoints are not distributed in
this repository.

## Quick Start

Clone the repository and run the data-free verification suite:

```bash
git clone https://github.com/Mostafa-Rezaali/HeatCast.git
cd HeatCast
python src/repo_integrity.py
python -m pytest -q
```

On a configured Slurm system, train and evaluate all five W34 folds with:

```bash
sbatch slurm/submit_w34_tube_all.slurm
```

The training workflow resumes incomplete folds and submits
`slurm/submit_w34_eval_stitch.slurm` after training completes. External data
paths and the Python environment in the Slurm scripts must be configured for
the target system.

## Repository Layout

```text
src/       Python model, evaluation, ENS, and analysis code
slurm/     Current production submission workflows
tests/     Data-free regression and experiment-contract tests
docs/      Model-input and integrity-test documentation
matlab/    MATLAB visualization utilities
```

## Core Code

| File | Purpose |
|---|---|
| `src/cfm_mesh_train.py` | Data setup, cross-validation, training, validation, and checkpoint handling |
| `src/mesh_backbone.py` | Grid-to-mesh encoder, multimesh processor, decoder, tube attention, and distributional head |
| `src/mode_dispatch.py` | Deterministic-direct mean/sigma semantics and Gaussian CRPS |
| `src/icosahedral_mesh.py` | Icosahedral mesh and grid/mesh connectivity |
| `src/exceedance_eval.py` | Daily/windowed q95 evaluation, calibration, leakage guards, and incremental arrays |
| `src/stitch_exceedance_folds.py` | Cross-fitted five-fold out-of-sample stitching and year-block bootstrap |
| `src/ens_ingest.py` | Resume-safe parallel ECMWF S2S ingestion and CONUS regridding |
| `src/ens_score.py` | Fold-safe ENS bias correction, calibration, and scoring |
| `src/ens_compare.py` | Matched HeatCast/ENS comparison and cycle merging |
| `src/ens_heatcast_stack_opportunity.py` | Cross-fitted HeatCast+ENS stack and robustness analyses |
| `src/build_driver_tables.py` | MJO, ENSO, soil, and teleconnection driver tables |
| `src/forecasts_of_opportunity.py` | Driver/opportunity stratification and paired year bootstrap |
| `src/export_w34_stack_netcdf.py` | MATLAB NetCDF export with latitude x longitude x time fields |
| `src/repo_integrity.py` | Fast data-free experiment and submission contract audit |

`docs/MODEL_INPUTS.md` lists the local, vector, and global predictor channels.

## W34 Model Configuration

The reference training configuration is:

```text
prediction leads:        15,16,...,28
tube loss:               0.80 daily CRPS + 0.20 14-day-mean CRPS
distributional head:     enabled
sigma floor:             0.1 z
early-stop monitor:      tube_weekly7_tac
cross-validation:        five year-disjoint folds
```

Despite the historical monitor name `tube_weekly7_tac`, the W34 script passes
14 leads, so the monitored tube mean is the 14-day mean over `t+15...t+28`.

## Exceedance Definition

For initialization day `t`, continuous W34 truth and forecast are the means
over leads `15...28`. The center month determines the threshold month.

The exceedance threshold is computed independently for each fold from training
years only:

```text
q95_week[pixel, month] =
    95th percentile of observed 14-day means for train-year initializations
```

Observed exceedance is `truth_week > q95_week`. Validation years fit
calibration; test years report skill. Training, calibration, and test year
sets are asserted disjoint.

## ENS And Stacking

`slurm/submit_ens_widen_cycles.slurm` is the consolidated ENS ingestion and scoring
workflow. It uses resume-safe parallel ingestion, cycle-specific quantile
mapping, fold-safe calibration, and matched HeatCast test dates.

The HeatCast+ENS stack is fit cross-fitted: each scored fold is excluded from
its stacker fit. Comparisons use identical initialization dates, PRISM truth,
q95 thresholds, land cells, and calendar-year bootstrap blocks.

## MATLAB Export

Export the matched W34 fields and probabilities to NetCDF with:

```bash
sbatch slurm/submit_export_w34_stack_netcdf.slurm
```

The NetCDF contains continuous W34 truth/hindcast fields and probabilistic
HeatCast, ENS, and stacked exceedance products.

## Verification

Run before every submission:

```bash
python src/repo_integrity.py
python -m pytest -q
```

The integrity audit checks the W34 lead window, distributional semantics,
fold-safe evaluation, active Slurm contracts, ENS cycle handling, and absence
of tracked runtime artifacts. See `docs/INTEGRITY_TEST_KIT.md` for scope and
limitations.

## Data Requirements

Primary external inputs are configured in `src/cfm_mesh_train.py`. They include
the PRISM/local training dataset, extended coarse global conditions,
topography, and fold-specific normalization caches. ECMWF S2S GRIB data and
teleconnection index sources are required only for the corresponding
benchmark and driver analyses.

## License

HeatCast is released under the [MIT License](LICENSE).
