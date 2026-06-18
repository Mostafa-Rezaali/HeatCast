#!/bin/bash
#SBATCH --job-name=ens_stack_opp
#SBATCH --output=ens_stack_opp_%j.log
#SBATCH --error=ens_stack_opp_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --partition=hpg-b200
#SBATCH --qos=nessie
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

module load cuda/12.9.1
unset PYTHONHOME
unset PYTHONPATH
export PYTHONUNBUFFERED=1

PY=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python
WORK_DIR=/blue/nessie/mostafarezaali/Teleconnection

cd "$WORK_DIR"
git pull --ff-only origin codex/tube_v1

"$PY" -m py_compile ens_heatcast_stack_opportunity.py ens_compare.py exceedance_eval.py stitch_exceedance_folds.py
"$PY" repo_integrity.py

"$PY" -u ens_heatcast_stack_opportunity.py \
  --heatcast_runs cvfold0_w34_dist_v1,cvfold1_w34_dist_v1,cvfold2_w34_dist_v1,cvfold3_w34_dist_v1,cvfold4_w34_dist_v1 \
  --ens_runs cvfold{F}_ens_w34,cvfold{F}_ens_w34_rt2024 \
  --window_leads 15,16,17,18,19,20,21,22,23,24,25,26,27,28 \
  --heatcast_root exceedance_eval_incremental \
  --ens_root ens_exceedance_incremental \
  --output_dir ens_heatcast_stack_opportunity \
  --bootstrap_reps 5000 \
  --max_stack_samples_per_fold 500000 \
  --emit_per_year
