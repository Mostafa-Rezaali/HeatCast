#!/bin/bash
#SBATCH --job-name=w34_tube_all
#SBATCH --output=w34_tube_all_%j.log
#SBATCH --error=w34_tube_all_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=3-00:00:00
#SBATCH --partition=hpg-b200
#SBATCH --qos=nessie
#SBATCH --gres=gpu:8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

WORK_DIR=/blue/nessie/mostafarezaali/Teleconnection
PY=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python
TORCHRUN=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/torchrun
LEADS=15,16,17,18,19,20,21,22,23,24,25,26,27,28

cd "$WORK_DIR"
module load cuda/12.9.1
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

git pull --ff-only origin codex/tube_v1
"$PY" -m py_compile cfm_mesh_train.py mesh_backbone.py mode_dispatch.py exceedance_eval.py

for FOLD in 0 1 2 3 4; do
    RUN=cvfold${FOLD}_w34_dist_v1
    echo "Starting fold $FOLD: $RUN"
    rm -f trained_cfm_direct15_${RUN}.pth
    rm -f trained_cfm_direct15_${RUN}_best_tac.pth
    rm -f trained_cfm_direct15_${RUN}_best_monitor.pth
    rm -f training_metrics/fold_epoch_metrics_${RUN}.csv
    rm -rf checkpoints/${RUN} test_prediction_plots/${RUN} /dev/shm/cfm_cache

    "$TORCHRUN" --standalone --nproc_per_node=8 cfm_mesh_train.py \
      --mode train \
      --deterministic \
      --cv_fold "$FOLD" \
      --run_name "$RUN" \
      --multi_lead_tube \
      --prediction_leads "$LEADS" \
      --epochs 30 \
      --batch_size 1 \
      --checkpoint_freq 1 \
      --num_validation_samples 928 \
      --learning_rate 5e-5 \
      --dropout 0.15 \
      --warmup_epochs 10 \
      --disable_anomaly_corr_loss \
      --gradient_loss_weight 0.0 \
      --tube_loss_daily_weight 0.80 \
      --tube_loss_center_weight 0.00 \
      --tube_loss_weekly_weight 0.20 \
      --distributional_head \
      --crps_loss \
      --sigma_floor 0.1 \
      --early_stop_metric tube_weekly7_tac \
      --early_stop_patience 5 \
      --early_stop_min_epoch 12 \
      --early_stop_min_delta 0.0001

    test -f trained_cfm_direct15_${RUN}_best_tac.pth
    echo "Completed fold $FOLD"
done

echo "ALL FIVE FOLDS COMPLETE"
