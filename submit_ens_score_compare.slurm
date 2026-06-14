#!/bin/bash
#SBATCH --job-name=ens_score_compare
#SBATCH --output=ens_score_compare_%j.log
#SBATCH --error=ens_score_compare_%j.err
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
ENS_RUNS=cvfold0_ens_w34,cvfold1_ens_w34,cvfold2_ens_w34,cvfold3_ens_w34,cvfold4_ens_w34
FOLD_WORKERS=${FOLD_WORKERS:-2}

cd "$WORK_DIR"
module load cuda/12.9.1
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if ! [[ "$FOLD_WORKERS" =~ ^[1-5]$ ]]; then
    echo "FOLD_WORKERS must be an integer from 1 to 5, got: $FOLD_WORKERS"
    exit 2
fi

git pull --ff-only origin codex/tube_v1
"$PY" -m py_compile repo_integrity.py ens_common.py ens_score.py ens_compare.py
"$PY" repo_integrity.py

score_fold() {
    local fold=$1
    local fold_log="ens_score_fold${fold}_${SLURM_JOB_ID}.log"
    local fold_err="ens_score_fold${fold}_${SLURM_JOB_ID}.err"
    echo "Starting ENS score fold $fold -> $fold_log"
    "$PY" -u ens_score.py \
      --cv_fold "$fold" \
      --run_name "cvfold${fold}_ens_w34" \
      --window_leads "$LEADS" \
      --output_root ens_exceedance_incremental \
      >"$fold_log" 2>"$fold_err"
}

wait_for_fold_batch() {
    local failed=0
    local index
    for index in "${!PIDS[@]}"; do
        if wait "${PIDS[$index]}"; then
            echo "ENS score fold ${FOLDS[$index]} complete"
        else
            echo "ENS score fold ${FOLDS[$index]} failed; see ens_score_fold${FOLDS[$index]}_${SLURM_JOB_ID}.err"
            failed=1
        fi
    done
    PIDS=()
    FOLDS=()
    return "$failed"
}

echo "Scoring five ENS folds with up to $FOLD_WORKERS concurrent fold workers"
PIDS=()
FOLDS=()
for FOLD in 0 1 2 3 4; do
    score_fold "$FOLD" &
    PIDS+=("$!")
    FOLDS+=("$FOLD")
    if [ "${#PIDS[@]}" -ge "$FOLD_WORKERS" ]; then
        wait_for_fold_batch
    fi
done
if [ "${#PIDS[@]}" -gt 0 ]; then
    wait_for_fold_batch
fi

echo "All ENS folds complete; starting pooled comparison"
"$PY" -u ens_compare.py \
  --heatcast_runs "$HEATCAST_RUNS" \
  --ens_runs "$ENS_RUNS" \
  --window_leads "$LEADS" \
  --heatcast_root exceedance_eval_incremental \
  --ens_root ens_exceedance_incremental \
  --bootstrap_reps 5000
