#!/bin/bash
#SBATCH --job-name=export_w34_nc
#SBATCH --output=export_w34_nc_%j.log
#SBATCH --error=export_w34_nc_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

WORK_DIR=/blue/nessie/mostafarezaali/Teleconnection
PY=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python
WINDOW_LEADS=${WINDOW_LEADS:-15,16,17,18,19,20,21,22,23,24,25,26,27,28}
OUT=${OUT:-matlab_exports/w34_heatcast_ens_stack.nc}

unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$WORK_DIR"
git pull --ff-only origin codex/tube_v1

"$PY" -m py_compile export_w34_stack_netcdf.py repo_integrity.py
"$PY" repo_integrity.py

"$PY" -u export_w34_stack_netcdf.py \
  --output "$OUT" \
  --window_leads "$WINDOW_LEADS" \
  --heatcast_runs cvfold0_w34_dist_v1,cvfold1_w34_dist_v1,cvfold2_w34_dist_v1,cvfold3_w34_dist_v1,cvfold4_w34_dist_v1 \
  --ens_runs 'cvfold{F}_ens_w34,cvfold{F}_ens_w34_rt2024' \
  --sort_by target \
  --fold_workers 5 \
  --max_stack_samples_per_fold 300000 \
  --require_continuous_fields \
  --progress_every 50

echo "W34 STACK NETCDF EXPORT COMPLETE: $OUT"
