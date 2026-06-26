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
CHECKPOINT_CHOICE_FILE=${CHECKPOINT_CHOICE_FILE:-w34_checkpoint_choice.txt}
CHECKPOINT_OVERRIDE=${CHECKPOINT_OVERRIDE:-}
DRIVER_DIR=${DRIVER_DIR:-data_cache/slow_driver_tables_w34}
FORCE_EVAL=${FORCE_EVAL:-0}

cd "$WORK_DIR"
module load cuda/12.9.1
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

git pull --ff-only origin codex/tube_v1
"$PY" -m py_compile repo_integrity.py exceedance_eval.py stitch_exceedance_folds.py forecasts_of_opportunity.py
"$PY" repo_integrity.py

if [ -n "$CHECKPOINT_OVERRIDE" ]; then
    CHECKPOINT=$CHECKPOINT_OVERRIDE
elif [ -f "$CHECKPOINT_CHOICE_FILE" ]; then
    CHECKPOINT=$(tr -d '[:space:]' < "$CHECKPOINT_CHOICE_FILE")
else
    echo "Missing $CHECKPOINT_CHOICE_FILE. Run submit_w34_checkpoint_arbitration.slurm first."
    exit 1
fi
case "$CHECKPOINT" in
    best_tac|best_monitor) ;;
    *) echo "Invalid W34 checkpoint choice: $CHECKPOINT"; exit 1 ;;
esac
echo "Locked W34 evaluation checkpoint selector: $CHECKPOINT"

fold_outputs_complete() {
    local run_name="$1"
    local fold="$2"
    local window_label="${LEADS//,/-}"
    local array_dir="$INPUT_ROOT/$run_name/test/window_$window_label/incremental_arrays"
    if [ "$FORCE_EVAL" = "1" ]; then
        return 1
    fi
    "$PY" - "$array_dir" "$run_name" "$fold" "$LEADS" <<'PY'
import sys
from pathlib import Path
import numpy as np

array_dir = Path(sys.argv[1])
run_name = sys.argv[2]
fold = int(sys.argv[3])
leads = np.array([int(v) for v in sys.argv[4].split(",")], dtype=np.int16)
manifest_path = array_dir / "manifest.npz"
chunk_dir = array_dir / "test_chunks"
if not manifest_path.exists() or not chunk_dir.exists():
    raise SystemExit(1)
with np.load(manifest_path, allow_pickle=False) as m:
    if str(m["run_name"]) != run_name:
        raise SystemExit(1)
    if int(m["source_fold"]) != fold:
        raise SystemExit(1)
    schema_version = int(m["schema_version"]) if "schema_version" in m.files else 1
    if schema_version < 2:
        raise SystemExit(1)
    if not np.array_equal(np.asarray(m["window_leads"], dtype=np.int16), leads):
        raise SystemExit(1)
    sample_count = int(m["sample_count"])
chunks = sorted(chunk_dir.glob("sample_*.npz"))
if len(chunks) != sample_count or sample_count <= 0:
    raise SystemExit(1)
for path in (chunks[0], chunks[-1]):
    with np.load(path, allow_pickle=False) as c:
        if "mu_z" not in c.files or "truth_z" not in c.files:
            raise SystemExit(1)
        if c["mu_z"].size == 0 or c["truth_z"].size == 0:
            raise SystemExit(1)
print(f"Complete schema-v2 incremental arrays found: {array_dir}")
PY
}

for FOLD in 0 1 2 3 4; do
    RUN=cvfold${FOLD}_w34_dist_v1
    if fold_outputs_complete "$RUN" "$FOLD"; then
        echo "Skipping fold $FOLD: existing schema-v2 incremental arrays are complete."
        continue
    fi
    echo "Evaluating fold $FOLD: $RUN"
    "$PY" -u exceedance_eval.py \
      --run_name "$RUN" \
      --cv_fold "$FOLD" \
      --checkpoint "$CHECKPOINT" \
      --prediction_leads "$LEADS" \
      --tube_decode_chunk_size 2 \
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
  --driver_table_dir "$DRIVER_DIR" \
  --n_bootstrap 2000

echo "WEEK 3-4 EVALUATION, STITCH, AND OPPORTUNITY VERIFICATION COMPLETE"
