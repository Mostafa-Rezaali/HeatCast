# HeatCast

[![CI](https://github.com/Mostafa-Rezaali/HeatCast/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/Mostafa-Rezaali/HeatCast/actions/workflows/python-package.yml)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-distributional%20mesh%20GNN-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.9.1-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Slurm](https://img.shields.io/badge/HPC-Slurm-0D73BA)](https://slurm.schedmd.com/)
[![pytest](https://img.shields.io/badge/tests-pytest%20passing-0A9EDC?logo=pytest&logoColor=white)](https://github.com/Mostafa-Rezaali/HeatCast/actions/workflows/python-package.yml)
[![Flake8](https://img.shields.io/badge/lint-Flake8%20checked-4C9C39)](https://github.com/Mostafa-Rezaali/HeatCast/actions/workflows/python-package.yml)
[![Integrity](https://img.shields.io/badge/repository%20integrity-checked-brightgreen)](https://github.com/Mostafa-Rezaali/HeatCast/blob/main/repo_integrity.py)
<br>
[![Forecast window](https://img.shields.io/badge/W34-leads%20%2B15...%2B28-6A5ACD)](https://github.com/Mostafa-Rezaali/HeatCast/blob/main/submit_w34_tube_all.slurm)
[![Cross-validation](https://img.shields.io/badge/cross--validation-5%20year--disjoint%20folds-2E8B57)](https://github.com/Mostafa-Rezaali/HeatCast/blob/main/README.md#exceedance-definition)
[![Parameters](https://img.shields.io/badge/parameters-4%2C637%2C891-8A2BE2)](https://github.com/Mostafa-Rezaali/HeatCast/blob/main/README.md#w34-training-contract)
[![NetCDF4](https://img.shields.io/badge/output-NetCDF4-4B8BBE)](https://unidata.github.io/netcdf4-python/)
[![MATLAB](https://img.shields.io/badge/export-MATLAB%20compatible-F7941D?logo=mathworks&logoColor=white)](https://github.com/Mostafa-Rezaali/HeatCast/tree/main/matlab_plots)
[![Last commit](https://img.shields.io/github/last-commit/Mostafa-Rezaali/HeatCast?branch=main)](https://github.com/Mostafa-Rezaali/HeatCast/commits/main)
[![Repository size](https://img.shields.io/github/repo-size/Mostafa-Rezaali/HeatCast)](https://github.com/Mostafa-Rezaali/HeatCast)
[![Stars](https://img.shields.io/github/stars/Mostafa-Rezaali/HeatCast?style=flat)](https://github.com/Mostafa-Rezaali/HeatCast/stargazers)
[![Open issues](https://img.shields.io/github/issues/Mostafa-Rezaali/HeatCast)](https://github.com/Mostafa-Rezaali/HeatCast/issues)

HeatCast is a GraphCast-style mesh graph neural network for probabilistic
CONUS week-3--4 (W34) warm-season T2max prediction. The current production
model is the **HeatCast-W34 Distributional Mesh GNN**:

- direct, single-pass inference rather than CFM sampling;
- a 14-lead tube covering `t+15...t+28`;
- a distributional head producing per-pixel mean and sigma;
- Gaussian CRPS training;
- five-fold, year-disjoint cross-validation; and
- held-out calibration and evaluation for W34 q95 exceedance probabilities.

The model has 4,637,891 trainable parameters in the executed W34
configuration. Training data, checkpoints, caches, generated tables, figures,
movies, and manuscript working files are intentionally excluded from Git.

## Production Workflow

The repository keeps one current Slurm entry point for each production stage.

| Stage | Submission |
|---|---|
| Train all five W34 folds | `submit_w34_tube_all.slurm` |
| Evaluate, stitch, and save fold-safe arrays | `submit_w34_eval_stitch.slurm` |
| Ingest and score ECMWF S2S cycles | `submit_ens_widen_cycles.slurm` |
| Fit the paired HeatCast+ENS stack | `submit_ens_stack_opportunity.slurm` |
| Add teleconnection and opportunity analyses | `submit_teleconnection_stack_analysis.slurm` |
| Build paper evidence tables | `submit_paper_evidence_blocks.slurm` |
| Build all journal figures and tables | `submit_paper_figures_journal.slurm` |
| Export MATLAB-readable W34 NetCDF | `submit_export_w34_stack_netcdf.slurm` |

`submit_w34_tube_all.slurm` automatically submits
`submit_w34_eval_stitch.slurm` after all folds complete. The evaluation
defaults to the `best_monitor` checkpoint and accepts `CHECKPOINT_OVERRIDE`
when an explicit alternative is required.

## Core Code

| File | Purpose |
|---|---|
| `cfm_mesh_train.py` | Data setup, cross-validation, training, validation, and checkpoint handling |
| `mesh_backbone.py` | Grid-to-mesh encoder, multimesh processor, decoder, tube attention, and distributional head |
| `mode_dispatch.py` | Deterministic-direct mean/sigma semantics and Gaussian CRPS |
| `icosahedral_mesh.py` | Icosahedral mesh and grid/mesh connectivity |
| `exceedance_eval.py` | Daily/windowed q95 evaluation, calibration, leakage guards, and incremental arrays |
| `stitch_exceedance_folds.py` | Cross-fitted five-fold out-of-sample stitching and year-block bootstrap |
| `ens_ingest.py` | Resume-safe parallel ECMWF S2S ingestion and CONUS regridding |
| `ens_score.py` | Fold-safe ENS bias correction, calibration, and scoring |
| `ens_compare.py` | Matched HeatCast/ENS comparison and cycle merging |
| `ens_heatcast_stack_opportunity.py` | Cross-fitted HeatCast+ENS stack and robustness analyses |
| `build_driver_tables.py` | MJO, ENSO, soil, and teleconnection driver tables |
| `forecasts_of_opportunity.py` | Driver/opportunity stratification and paired year bootstrap |
| `build_paper_evidence_blocks.py` | Paper-facing mechanism, robustness, and operational tables |
| `build_paper_figures_tables.py` | Primary probabilistic figures and tables |
| `build_paper_figures_extended.py` | Spatial, reliability, case-study, and supporting figures |
| `export_w34_stack_netcdf.py` | MATLAB NetCDF export with latitude x longitude x time fields |
| `repo_integrity.py` | Fast data-free experiment and submission contract audit |

`Model_Inputs.txt` lists the local, vector, and global predictor channels.

## W34 Training Contract

The production training script fixes the executed experiment:

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

Run on HiPerGator:

```bash
cd /blue/nessie/mostafarezaali/Teleconnection
sbatch submit_w34_tube_all.slurm
```

The script requests eight B200 GPUs and 500 GB memory, resumes incomplete
folds, and skips folds with a completed best-monitor checkpoint.

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

`submit_ens_widen_cycles.slurm` is the consolidated ENS ingestion and scoring
workflow. It uses resume-safe parallel ingestion, cycle-specific quantile
mapping, fold-safe calibration, and matched HeatCast test dates.

The HeatCast+ENS stack is fit cross-fitted: each scored fold is excluded from
its stacker fit. Comparisons use identical initialization dates, PRISM truth,
q95 thresholds, land cells, and calendar-year bootstrap blocks.

## Paper Outputs

After the matched stack and teleconnection analyses complete:

```bash
sbatch submit_paper_evidence_blocks.slurm
sbatch submit_paper_figures_journal.slurm
```

The journal wrapper runs both figure builders and is the only retained figure
submission entry point. Generated outputs remain untracked.

For MATLAB export:

```bash
sbatch submit_export_w34_stack_netcdf.slurm
```

The NetCDF contains continuous W34 truth/hindcast fields and probabilistic
HeatCast, ENS, and stacked exceedance products.

## Verification

Run before every submission:

```bash
python repo_integrity.py
python -m pytest -q
```

The integrity audit checks the W34 lead window, distributional semantics,
fold-safe evaluation, active Slurm contracts, current submission allowlist,
ENS cycle handling, paper workflow, and absence of tracked runtime artifacts.
See `INTEGRITY_TEST_KIT.md` for scope and limitations.

## Environment And External Data

Production scripts target UF HiPerGator and use:

```text
/blue/nessie/mostafarezaali/Teleconnection
/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python
CUDA 12.9.1 for model training/evaluation
```

Primary external files are configured in `cfm_mesh_train.py`, including the
PRISM/local training dataset, extended coarse global conditions, topography,
and train-only normalization caches. ECMWF S2S GRIB data and teleconnection
index sources are also external.

## Repository Policy

Commit source code, tests, active Slurm workflows, and concise documentation.
Do not commit datasets, checkpoints, caches, generated results, manuscripts,
meeting/presentation packs, figures, movies, logs, or personal proposal files.
