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

cd "$WORK_DIR"
module load cuda/12.9.1
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1

git pull --ff-only origin codex/tube_v1
"$PY" -m py_compile ens_common.py ens_score.py ens_compare.py

for FOLD in 0 1 2 3 4; do
    "$PY" -u ens_score.py \
      --cv_fold "$FOLD" \
      --run_name "cvfold${FOLD}_ens_w34" \
      --window_leads "$LEADS" \
      --weekdays 0,3 \
      --output_root ens_exceedance_incremental
done

"$PY" -u ens_compare.py \
  --heatcast_runs "$HEATCAST_RUNS" \
  --ens_runs "$ENS_RUNS" \
  --window_leads "$LEADS" \
  --heatcast_root exceedance_eval_incremental \
  --ens_root ens_exceedance_incremental \
  --bootstrap_reps 5000
