# HeatCast Repository Integrity Test Kit

This kit checks that the code still implements the intended HeatCast experiment,
not merely that Python files compile.

## Protected Contracts

- Daily exceedance labels use month-specific MJJAS train-year q95 thresholds.
- Five CV test folds are disjoint and cover 1981-2023 exactly once.
- Model-training, calibration, and test roles remain leakage-separated.
- Distributional output means are persistence residuals and sigma is positive.
- Gaussian CRPS matches numerical quadrature.
- The grid refiner changes the mean only, never predicted variance.
- W34 training and evaluation both use leads 15-28.
- W34 uses the distributional CRPS head, bounded decoder memory, and window TAC monitor.
- W34 evaluation saves fold arrays and uses leakage-clean cross-fitted stitching.
- Critical Slurm jobs retain the established memory, GPU, Git-pull, and email settings.
- Runtime model/data artifacts are not accidentally committed.

## Commands

Fast contract audit, suitable before every commit or Slurm submission:

```bash
python src/repo_integrity.py
```

Machine-readable audit:

```bash
python src/repo_integrity.py --json
```

Complete data-free test kit:

```bash
python -m pytest -q
```

Undefined-name and syntax checks:

```bash
python -m py_compile src/repo_integrity.py src/cfm_mesh_train.py src/mesh_backbone.py src/mode_dispatch.py src/exceedance_eval.py
git ls-files -z '*.py' | xargs -0 python -m flake8 --count --select=E9,F63,F7,F82 --show-source --statistics
```

## What This Kit Does Not Prove

The data-free kit cannot prove that external NetCDF/GRIB files are complete,
that a checkpoint is scientifically skillful, or that a full B200 job fits in
memory. Those require runtime preflight and post-run acceptance checks.

Before a production W34 submission, run the fast audit and then inspect the
first training batch for GPU memory, finite loss, sigma above its floor, and the
expected 14-lead output shape. After training, require the existing leakage,
valid-cell-count, and year-block-bootstrap assertions from the evaluation stack.
