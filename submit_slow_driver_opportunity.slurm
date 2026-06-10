#!/bin/bash
#SBATCH --job-name=slow_driver_opp
#SBATCH --output=slow_driver_opp_%j.log
#SBATCH --error=slow_driver_opp_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=2-00:00:00
#SBATCH --partition=hpg-b200
#SBATCH --qos=nessie
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

WORK_DIR=/blue/nessie/mostafarezaali/Teleconnection
PY=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python
LEADS=${LEADS:-12,13,14,15,16,17,18}
RUNS=${RUNS:-cvfold0_dist_v2_normfix,cvfold1_dist_v2_normfix,cvfold2_dist_v2_normfix,cvfold3_dist_v2_normfix,cvfold4_dist_v2_normfix}
INPUT_ROOT=${INPUT_ROOT:-exceedance_eval_incremental}
DRIVER_DIR=${DRIVER_DIR:-data_cache/slow_driver_tables}
RMM_PATH=${RMM_PATH:?Set RMM_PATH to the manually downloaded BOM rmm.74toRealtime.txt file}
NINO34_PATH=${NINO34_PATH:?Set NINO34_PATH to the manually downloaded NOAA PSL Nino3.4 text file}

cd "$WORK_DIR"
module load cuda/12.9.1
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1

git pull --ff-only origin codex/tube_v1
"$PY" -m py_compile forecasts_of_opportunity.py recover_chunk_init_dates.py build_driver_tables.py

"$PY" -u recover_chunk_init_dates.py \
  --input_root "$INPUT_ROOT" \
  --run_names "$RUNS" \
  --window_leads "$LEADS"

"$PY" -u build_driver_tables.py \
  --rmm_path "$RMM_PATH" \
  --nino34_path "$NINO34_PATH" \
  --input_root "$INPUT_ROOT" \
  --run_names "$RUNS" \
  --window_leads "$LEADS" \
  --output_dir "$DRIVER_DIR"

"$PY" -u forecasts_of_opportunity.py \
  --input_root "$INPUT_ROOT" \
  --run_names "$RUNS" \
  --window_leads "$LEADS" \
  --driver_table_dir "$DRIVER_DIR" \
  --bootstrap_axes mjo_phase,enso_state,soil_moisture_tercile \
  --n_bootstrap 2000

echo "SLOW-DRIVER OPPORTUNITY STRATIFICATION COMPLETE"
