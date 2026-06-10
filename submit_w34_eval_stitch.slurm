#!/bin/bash
#SBATCH --job-name=w34_eval_stitch
#SBATCH --output=w34_eval_stitch_%j.log
#SBATCH --error=w34_eval_stitch_%j.err
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
RUNS=cvfold0_w34_dist_v1,cvfold1_w34_dist_v1,cvfold2_w34_dist_v1,cvfold3_w34_dist_v1,cvfold4_w34_dist_v1
INPUT_ROOT=exceedance_eval_incremental

cd "$WORK_DIR"
module load cuda/12.9.1
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

git pull --ff-only origin codex/tube_v1
"$PY" -m py_compile exceedance_eval.py stitch_exceedance_folds.py forecasts_of_opportunity.py

for FOLD in 0 1 2 3 4; do
    RUN=cvfold${FOLD}_w34_dist_v1
    echo "Evaluating fold $FOLD: $RUN"
    "$PY" -u exceedance_eval.py \
      --run_name "$RUN" \
      --cv_fold "$FOLD" \
      --checkpoint best_tac \
      --prediction_leads "$LEADS" \
      --target_mode window \
      --window_leads "$LEADS" \
      --use_model_sigma \
      --calibration_split val \
      --eval_split test \
      --calibrator platt \
      --incremental_skill_diagnostic \
      --save_incremental_arrays \
      --year_bootstrap_reps 1000 \
      --output_dir "$INPUT_ROOT"
done

"$PY" -u stitch_exceedance_folds.py \
  --runs "$RUNS" \
  --window_leads "$LEADS" \
  --calibrator platt \
  --fit_mode cross_fitted \
  --input_root "$INPUT_ROOT" \
  --bootstrap_reps 5000

"$PY" -u forecasts_of_opportunity.py \
  --input_root "$INPUT_ROOT" \
  --run_names "$RUNS" \
  --window_leads "$LEADS" \
  --n_bootstrap 2000

echo "WEEK 3-4 EVALUATION, STITCH, AND OPPORTUNITY VERIFICATION COMPLETE"
