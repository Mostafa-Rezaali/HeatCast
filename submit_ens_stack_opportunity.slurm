#!/bin/bash
#SBATCH --job-name=ens_stack_opp
#SBATCH --output=ens_stack_opp_%j.log
#SBATCH --error=ens_stack_opp_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

unset PYTHONHOME
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
FOLD_WORKERS=${FOLD_WORKERS:-5}
STRATUM_BOOTSTRAP_REPS=${STRATUM_BOOTSTRAP_REPS:-1000}
DRIVER_TABLE_DIR=${DRIVER_TABLE_DIR:-data_cache/slow_driver_tables_w34_alldata}
DRIVER_ARGS=()
if [ -n "${DRIVER_TABLE_DIR:-}" ]; then
  if [ ! -f "$DRIVER_TABLE_DIR/mjo_enso_by_init.npz" ]; then
    echo "Missing driver table: $DRIVER_TABLE_DIR/mjo_enso_by_init.npz"
    echo "Build it first with submit_teleconnection_stack_analysis.slurm or build_driver_tables.py."
    exit 1
  fi
  DRIVER_ARGS=(--driver_table_dir "$DRIVER_TABLE_DIR")
fi

PY=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python
WORK_DIR=/blue/nessie/mostafarezaali/Teleconnection

cd "$WORK_DIR"
git pull --ff-only origin main

"$PY" -m py_compile ens_heatcast_stack_opportunity.py ens_compare.py exceedance_eval.py stitch_exceedance_folds.py forecasts_of_opportunity.py
"$PY" repo_integrity.py

"$PY" -u ens_heatcast_stack_opportunity.py \
  --heatcast_runs cvfold0_w34_dist_v1,cvfold1_w34_dist_v1,cvfold2_w34_dist_v1,cvfold3_w34_dist_v1,cvfold4_w34_dist_v1 \
  --ens_runs cvfold{F}_ens_w34,cvfold{F}_ens_w34_rt2024 \
  --window_leads 15,16,17,18,19,20,21,22,23,24,25,26,27,28 \
  --heatcast_root exceedance_eval_incremental \
  --ens_root ens_exceedance_incremental \
  --output_dir ens_heatcast_stack_opportunity \
  --bootstrap_reps 5000 \
  --stratum_bootstrap_reps "$STRATUM_BOOTSTRAP_REPS" \
  --max_stack_samples_per_fold 500000 \
  --fold_workers "$FOLD_WORKERS" \
  --emit_per_year \
  "${DRIVER_ARGS[@]}"
