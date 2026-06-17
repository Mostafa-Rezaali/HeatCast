#!/bin/bash
#SBATCH --job-name=w34_monitor_h2h
#SBATCH --output=w34_monitor_h2h_%j.log
#SBATCH --error=w34_monitor_h2h_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=3-00:00:00
#SBATCH --partition=hpg-b200
#SBATCH --qos=nessie
#SBATCH --gres=gpu:5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

WORK_DIR=/blue/nessie/mostafarezaali/Teleconnection
PY=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python
LEADS=15,16,17,18,19,20,21,22,23,24,25,26,27,28
RUNS=cvfold0_w34_dist_v1,cvfold1_w34_dist_v1,cvfold2_w34_dist_v1,cvfold3_w34_dist_v1,cvfold4_w34_dist_v1
ENS_RUN_TEMPLATES='cvfold{F}_ens_w34,cvfold{F}_ens_w34_rt2024'
INPUT_ROOT=${INPUT_ROOT:-exceedance_eval_w34_best_monitor}
BASELINE_DIR=${BASELINE_DIR:-ens_head_to_head_cycles}
OUTPUT_DIR=${OUTPUT_DIR:-ens_head_to_head_best_monitor}
EVAL_WORKERS=${EVAL_WORKERS:-5}
WINDOW_LABEL=window_${LEADS//,/-}

cd "$WORK_DIR"
module load cuda/12.9.1
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

git pull --ff-only origin codex/tube_v1
"$PY" -m py_compile repo_integrity.py exceedance_eval.py ens_compare.py stitch_exceedance_folds.py
"$PY" repo_integrity.py

evaluate_fold() {
    local fold=$1
    local gpu=$2
    local run="cvfold${fold}_w34_dist_v1"
    local out_dir="${INPUT_ROOT}/${run}/test/${WINDOW_LABEL}"
    if [ -s "${out_dir}/exceedance_results.csv" ] && [ -s "${out_dir}/incremental_arrays/manifest.npz" ]; then
        echo "Skipping complete best-monitor HeatCast eval: fold=${fold}, output=${out_dir}"
        return 0
    fi
    rm -rf "$out_dir"
    echo "Evaluating best_monitor fold=${fold} on CUDA_VISIBLE_DEVICES=${gpu}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u exceedance_eval.py \
      --run_name "$run" \
      --cv_fold "$fold" \
      --checkpoint best_monitor \
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
      --output_dir "$INPUT_ROOT" \
      >"w34_best_monitor_fold${fold}_${SLURM_JOB_ID}.log" \
      2>"w34_best_monitor_fold${fold}_${SLURM_JOB_ID}.err"
}

pids=()
folds=()
for fold in 0 1 2 3 4; do
    while [ "${#pids[@]}" -ge "$EVAL_WORKERS" ]; do
        wait "${pids[0]}" || {
            echo "Best-monitor evaluation failed for fold ${folds[0]}"
            exit 1
        }
        pids=("${pids[@]:1}")
        folds=("${folds[@]:1}")
    done
    evaluate_fold "$fold" "$fold" &
    pids+=("$!")
    folds+=("$fold")
done

for index in "${!pids[@]}"; do
    wait "${pids[$index]}" || {
        echo "Best-monitor evaluation failed for fold ${folds[$index]}"
        exit 1
    }
done

"$PY" -u ens_compare.py \
  --heatcast_runs "$RUNS" \
  --ens_runs "$ENS_RUN_TEMPLATES" \
  --window_leads "$LEADS" \
  --heatcast_root "$INPUT_ROOT" \
  --ens_root ens_exceedance_incremental \
  --output_dir "$OUTPUT_DIR" \
  --bootstrap_reps 5000 \
  --emit_per_year

"$PY" - <<'PY'
import csv
from pathlib import Path

window = "window_15-16-17-18-19-20-21-22-23-24-25-26-27-28"
paths = {
    "best_tac_baseline": Path("ens_head_to_head_cycles") / window / "ens_heatcast_head_to_head.csv",
    "best_monitor": Path("ens_head_to_head_best_monitor") / window / "ens_heatcast_head_to_head.csv",
}

def score(path):
    rows = list(csv.DictReader(path.open()))
    heat = next(row for row in rows if row.get("section") == "score" and row.get("model") == "heatcast_C")
    ens = next(row for row in rows if row.get("section") == "score" and row.get("model") == "ens_calibrated")
    boot = next(row for row in rows if row.get("section") == "bootstrap" and row.get("metric") == "delta_bss_heatcast_minus_ens")
    return {
        "heat_bss": float(heat["bss_vs_monthly_climo"]),
        "heat_auc": float(heat["roc_auc"]),
        "heat_ece": float(heat["ece"]),
        "ens_bss": float(ens["bss_vs_monthly_climo"]),
        "delta_bss": float(boot["point_estimate"]),
        "ci_low": float(boot["ci_low"]),
        "ci_high": float(boot["ci_high"]),
    }

print("W34 checkpoint arbitration")
print("==========================")
results = {}
for name, path in paths.items():
    if not path.exists():
        print(f"{name}: missing {path}")
        continue
    results[name] = score(path)
    r = results[name]
    print(
        f"{name}: HeatCast BSS={r['heat_bss']:+.4f}, AUC={r['heat_auc']:.3f}, "
        f"ECE={r['heat_ece']:.4f}, delta_BSS_vs_ENS={r['delta_bss']:+.4f} "
        f"CI=[{r['ci_low']:+.4f},{r['ci_high']:+.4f}]"
    )
if set(results) == set(paths):
    winner = max(results, key=lambda key: (results[key]["heat_bss"], results[key]["heat_auc"]))
    print(f"Checkpoint winner by HeatCast BSS then AUC: {winner}")
PY

echo "BEST-MONITOR W34 HEAD-TO-HEAD COMPLETE"
