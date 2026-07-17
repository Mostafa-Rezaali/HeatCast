#!/bin/bash
#SBATCH --job-name=tele_stack
#SBATCH --output=tele_stack_%j.log
#SBATCH --error=tele_stack_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

WORK_DIR=/blue/nessie/mostafarezaali/Teleconnection
PY=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python

LEADS=${LEADS:-15,16,17,18,19,20,21,22,23,24,25,26,27,28}
RUNS=${RUNS:-cvfold0_w34_dist_v1,cvfold1_w34_dist_v1,cvfold2_w34_dist_v1,cvfold3_w34_dist_v1,cvfold4_w34_dist_v1}
INPUT_ROOT=${INPUT_ROOT:-exceedance_eval_incremental}
DRIVER_DIR=${DRIVER_DIR:-data_cache/slow_driver_tables_w34_teleconnections}
STACK_OUTPUT_DIR=${STACK_OUTPUT_DIR:-ens_heatcast_stack_opportunity_teleconnections}
RMM_PATH=${RMM_PATH:?Set RMM_PATH to the BOM rmm.74toRealtime.txt file}
NINO34_PATH=${NINO34_PATH:?Set NINO34_PATH to the NOAA PSL Nino3.4 text file}
TELECONNECTION_INDEX_PATHS=${TELECONNECTION_INDEX_PATHS:?Set TELECONNECTION_INDEX_PATHS to NAME=/path entries, for example pna=$PWD/data_cache/slow_driver_sources/pna.data,nao=$PWD/data_cache/slow_driver_sources/nao.data}
TELECONNECTION_THRESHOLD=${TELECONNECTION_THRESHOLD:-0.5}
BOOTSTRAP_AXES=${BOOTSTRAP_AXES:-all_drivers}
FOLD_WORKERS=${FOLD_WORKERS:-5}
STRATUM_BOOTSTRAP_REPS=${STRATUM_BOOTSTRAP_REPS:-2000}

cd "$WORK_DIR"
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

git pull --ff-only origin main
"$PY" -m py_compile src/repo_integrity.py src/recover_chunk_init_dates.py src/build_driver_tables.py src/forecasts_of_opportunity.py src/ens_heatcast_stack_opportunity.py src/ens_compare.py src/exceedance_eval.py src/stitch_exceedance_folds.py
"$PY" src/repo_integrity.py

"$PY" -u src/recover_chunk_init_dates.py \
  --input_root "$INPUT_ROOT" \
  --run_names "$RUNS" \
  --window_leads "$LEADS"

"$PY" -u src/build_driver_tables.py \
  --rmm_path "$RMM_PATH" \
  --nino34_path "$NINO34_PATH" \
  --teleconnection_index_paths "$TELECONNECTION_INDEX_PATHS" \
  --teleconnection_threshold "$TELECONNECTION_THRESHOLD" \
  --input_root "$INPUT_ROOT" \
  --run_names "$RUNS" \
  --window_leads "$LEADS" \
  --output_dir "$DRIVER_DIR"

"$PY" -u src/forecasts_of_opportunity.py \
  --input_root "$INPUT_ROOT" \
  --run_names "$RUNS" \
  --window_leads "$LEADS" \
  --driver_table_dir "$DRIVER_DIR" \
  --bootstrap_axes "$BOOTSTRAP_AXES" \
  --n_bootstrap "$STRATUM_BOOTSTRAP_REPS"

"$PY" -u src/ens_heatcast_stack_opportunity.py \
  --heatcast_runs cvfold0_w34_dist_v1,cvfold1_w34_dist_v1,cvfold2_w34_dist_v1,cvfold3_w34_dist_v1,cvfold4_w34_dist_v1 \
  --ens_runs cvfold{F}_ens_w34,cvfold{F}_ens_w34_rt2024 \
  --window_leads "$LEADS" \
  --heatcast_root exceedance_eval_incremental \
  --ens_root ens_exceedance_incremental \
  --output_dir "$STACK_OUTPUT_DIR" \
  --bootstrap_reps 5000 \
  --stratum_bootstrap_reps "$STRATUM_BOOTSTRAP_REPS" \
  --max_stack_samples_per_fold 500000 \
  --fold_workers "$FOLD_WORKERS" \
  --driver_table_dir "$DRIVER_DIR" \
  --emit_per_year

echo "TELECONNECTION STACK ANALYSIS COMPLETE"
