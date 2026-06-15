#!/bin/bash
#SBATCH --job-name=ens_widen_cycles
#SBATCH --output=ens_widen_cycles_%j.log
#SBATCH --error=ens_widen_cycles_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=3-00:00:00
#SBATCH --partition=hpg-b200
#SBATCH --qos=nessie
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

WORK_DIR=/blue/nessie/mostafarezaali/Teleconnection
PY=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python
LEADS=15,16,17,18,19,20,21,22,23,24,25,26,27,28
HEATCAST_RUNS=cvfold0_w34_dist_v1,cvfold1_w34_dist_v1,cvfold2_w34_dist_v1,cvfold3_w34_dist_v1,cvfold4_w34_dist_v1
ENS_RUN_TEMPLATES='cvfold{F}_ens_w34,cvfold{F}_ens_w34_rt2024'
FOLD_WORKERS=${FOLD_WORKERS:-2}
INGEST_WORKERS=${INGEST_WORKERS:-16}

cd "$WORK_DIR"
module load cuda/12.9.1
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

git pull --ff-only origin codex/tube_v1
"$PY" -m py_compile repo_integrity.py ens_common.py ens_ingest.py ens_score.py ens_compare.py
"$PY" repo_integrity.py

"$PY" -u ens_ingest.py \
  --raw_dir ens_reforecast/raw \
  --rt_tag rt2024 \
  --variable mx2t6 \
  --max_lead 28 \
  --expected_members 11 \
  --workers "$INGEST_WORKERS"

score_fold_cycle() {
    local fold=$1
    local tag=$2
    local run="cvfold${fold}_ens_w34"
    local extra=()
    if [ -n "$tag" ]; then
        run="${run}_${tag}"
        extra=(--rt_tag "$tag")
    fi
    "$PY" -u ens_score.py \
      --cv_fold "$fold" \
      --run_name "$run" \
      --window_leads "$LEADS" \
      --output_root ens_exceedance_incremental \
      "${extra[@]}" \
      >"ens_score_fold${fold}_${tag:-legacy}_${SLURM_JOB_ID}.log" \
      2>"ens_score_fold${fold}_${tag:-legacy}_${SLURM_JOB_ID}.err"
}

run_cycle() {
    local tag=$1
    local pids=()
    local folds=()
    for fold in 0 1 2 3 4; do
        score_fold_cycle "$fold" "$tag" &
        pids+=("$!")
        folds+=("$fold")
        if [ "${#pids[@]}" -ge "$FOLD_WORKERS" ]; then
            for index in "${!pids[@]}"; do
                wait "${pids[$index]}" || {
                    echo "Scoring failed for fold ${folds[$index]}, cycle ${tag:-legacy}"
                    exit 1
                }
            done
            pids=()
            folds=()
        fi
    done
    for index in "${!pids[@]}"; do
        wait "${pids[$index]}" || {
            echo "Scoring failed for fold ${folds[$index]}, cycle ${tag:-legacy}"
            exit 1
        }
    done
}

echo "Rescoring legacy rt2022 hdates without historical-weekday filtering."
run_cycle ""
echo "Scoring rt2024 with cycle-specific quantile mappings."
run_cycle rt2024

"$PY" -u ens_compare.py \
  --heatcast_runs "$HEATCAST_RUNS" \
  --ens_runs "$ENS_RUN_TEMPLATES" \
  --window_leads "$LEADS" \
  --heatcast_root exceedance_eval_incremental \
  --ens_root ens_exceedance_incremental \
  --output_dir ens_head_to_head_cycles \
  --bootstrap_reps 5000 \
  --emit_per_year
