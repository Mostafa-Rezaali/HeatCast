#!/bin/bash
#SBATCH --job-name=tube7_grad_pair
#SBATCH --output=tube7_grad_pair_%j.log
#SBATCH --error=tube7_grad_pair_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=3-00:00:00
#SBATCH --partition=hpg-b200
#SBATCH --qos=nessie
#SBATCH --gres=gpu:8
#SBATCH --exclude=c1003a-s15
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

module load cuda/12.9.1

unset PYTHONHOME
unset PYTHONPATH

TORCHRUN=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/torchrun
PYTHON_EXE=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python

WORK_DIR=/blue/nessie/mostafarezaali/Teleconnection
SCRIPT=${WORK_DIR}/cfm_mesh_train.py

# Matched fold-2 pair:
#   control   grad=0.00
#   treatment grad=0.10
# Everything else is held fixed so the difference is attributable to gradient loss.
FOLDS=${FOLDS:-"2"}
GRADIENT_WEIGHTS=${GRADIENT_WEIGHTS:-"0.00 0.10"}
CLEAN_START=${CLEAN_START:-1}

cd "$WORK_DIR" || exit 1

$PYTHON_EXE -m py_compile \
  cfm_mesh_train.py mesh_backbone.py mode_dispatch.py icosahedral_mesh.py \
  stitch_hindcast_tac.py tac_skill_maps.py summarize_hindcast_diagnostics.py \
  compute_baselines.py export_per_year_stats.py bootstrap_significance.py || exit 1
$PYTHON_EXE -c "import cfm_mesh_train; print('cfm_mesh_train import OK')" || exit 1

nvidia-smi

export TORCH_NCCL_DUMP_ON_TIMEOUT=0
export TORCH_NCCL_TRACE_BUFFER_SIZE=0
export NCCL_NET_MERGE_LEVEL=LOC
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset CUDA_VISIBLE_DEVICES

rm -rf /dev/shm/cfm_cache

for FOLD in $FOLDS; do
    for GRAD_WEIGHT in $GRADIENT_WEIGHTS; do
        case "$GRAD_WEIGHT" in
            0|0.0|0.00) GRAD_TAG=grad000 ;;
            0.05|.05) GRAD_TAG=grad005 ;;
            0.10|.10) GRAD_TAG=grad010 ;;
            *) GRAD_TAG=grad$(echo "$GRAD_WEIGHT" | tr -d '0.') ;;
        esac

        RUN_SUFFIX=tube7_daily_${GRAD_TAG}
        RUN_NAME=cvfold${FOLD}_${RUN_SUFFIX}

        echo "============================================================"
        echo "Training matched tube diagnostic fold $FOLD"
        echo "Run name: $RUN_NAME"
        echo "Prediction leads: 12,13,14,15,16,17,18"
        echo "Tube loss: 0.90 daily + 0.10 center + 0.00 weekly"
        echo "Early-stop metric: daily center-lead TAC"
        echo "Gradient loss weight: $GRAD_WEIGHT"
        echo "============================================================"

        if [ "$CLEAN_START" = "1" ]; then
            echo "Cleaning run-specific artifacts for $RUN_NAME"
            rm -f ${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}.pth
            rm -f ${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}_best_tac.pth
            rm -f ${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}_best_monitor.pth
            rm -rf ${WORK_DIR}/checkpoints/${RUN_NAME}
            rm -rf ${WORK_DIR}/test_prediction_plots/${RUN_NAME}
            rm -f ${WORK_DIR}/hindcast_stats/hindcast_tac_stats_${RUN_NAME}_test.npz
            rm -f ${WORK_DIR}/hindcast_stats/stitched_tac_${RUN_SUFFIX}.npz
            rm -f ${WORK_DIR}/training_metrics/fold_epoch_metrics_${RUN_NAME}.csv
            rm -f ${WORK_DIR}/hindcast_paper_data/hindcast_*_${RUN_NAME}_test.npz
            rm -rf ${WORK_DIR}/paper_figures/${RUN_SUFFIX}
        else
            echo "CLEAN_START=0: preserving run-specific artifacts for $RUN_NAME"
        fi

        rm -rf /dev/shm/cfm_cache

        $TORCHRUN --standalone --nproc_per_node=8 "$SCRIPT" \
          --mode train \
          --deterministic \
          --multi_lead_tube \
          --prediction_leads 12,13,14,15,16,17,18 \
          --cv_fold "$FOLD" \
          --run_name "$RUN_NAME" \
          --epochs 30 \
          --batch_size 1 \
          --checkpoint_freq 1 \
          --num_validation_samples 1088 \
          --learning_rate 5e-5 \
          --dropout 0.15 \
          --warmup_epochs 10 \
          --tube_loss_daily_weight 0.90 \
          --tube_loss_center_weight 0.10 \
          --tube_loss_weekly_weight 0.00 \
          --gradient_loss_weight "$GRAD_WEIGHT" \
          --disable_anomaly_corr_loss \
          --early_stop_metric tac \
          --early_stop_patience 5 \
          --early_stop_min_epoch 12 \
          --early_stop_min_delta 0.0001

        TRAIN_STATUS=$?
        if [ $TRAIN_STATUS -ne 0 ]; then
            echo "Training failed for $RUN_NAME with exit code $TRAIN_STATUS"
            exit $TRAIN_STATUS
        fi

        CKPT=${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}_best_monitor.pth
        if [ ! -f "$CKPT" ]; then
            echo "Missing best-monitor checkpoint: $CKPT"
            echo "Falling back to best daily-TAC checkpoint."
            CKPT=${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}_best_tac.pth
        fi
        if [ ! -f "$CKPT" ]; then
            echo "Missing best daily-TAC checkpoint; falling back to best-R2 checkpoint."
            CKPT=${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}.pth
        fi
        if [ ! -f "$CKPT" ]; then
            echo "Missing checkpoint after training: $CKPT"
            exit 1
        fi

        echo "============================================================"
        echo "Exporting $RUN_NAME"
        echo "Export checkpoint: $CKPT"
        echo "============================================================"

        rm -rf /dev/shm/cfm_cache

        $TORCHRUN --standalone --nproc_per_node=1 "$SCRIPT" \
          --mode export_hindcast \
          --deterministic \
          --multi_lead_tube \
          --prediction_leads 12,13,14,15,16,17,18 \
          --cv_fold "$FOLD" \
          --run_name "$RUN_NAME" \
          --checkpoint "$CKPT" \
          --hindcast_splits test

        EXPORT_STATUS=$?
        if [ $EXPORT_STATUS -ne 0 ]; then
            echo "Export failed for $RUN_NAME with exit code $EXPORT_STATUS"
            exit $EXPORT_STATUS
        fi

        STAT_FILE=hindcast_stats/hindcast_tac_stats_${RUN_NAME}_test.npz
        if [ -f "$STAT_FILE" ]; then
            echo "============================================================"
            echo "Daily TAC map diagnostics for $RUN_NAME"
            echo "============================================================"
            $PYTHON_EXE -u tac_skill_maps.py \
              "$STAT_FILE" \
              --metric daily \
              --output_dir paper_figures/${RUN_SUFFIX} || exit 1
        else
            echo "Missing exported stat file: $STAT_FILE"
            exit 1
        fi
    done
done

echo "============================================================"
echo "MATCHED TUBE7 GRADIENT PAIR COMPLETE"
echo "Compare:"
echo "  training_metrics/fold_epoch_metrics_cvfold2_tube7_daily_grad000.csv"
echo "  training_metrics/fold_epoch_metrics_cvfold2_tube7_daily_grad010.csv"
echo "Accept grad010 only if daily TAC and gradient_ratio improve, with val MSE <= 1.05x grad000."
echo "============================================================"
