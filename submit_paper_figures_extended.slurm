#!/bin/bash
#SBATCH --job-name=paper_fig_ext
#SBATCH --output=paper_fig_ext_%j.log
#SBATCH --error=paper_fig_ext_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

WORK_DIR=/blue/nessie/mostafarezaali/Teleconnection
PY=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python
WINDOW=${WINDOW:-window_15-16-17-18-19-20-21-22-23-24-25-26-27-28}
LEADS=${LEADS:-15,16,17,18,19,20,21,22,23,24,25,26,27,28}
HEATCAST_RUNS=${HEATCAST_RUNS:-cvfold0_w34_dist_v1,cvfold1_w34_dist_v1,cvfold2_w34_dist_v1,cvfold3_w34_dist_v1,cvfold4_w34_dist_v1}
ENS_RUNS=${ENS_RUNS:-cvfold{F}_ens_w34,cvfold{F}_ens_w34_rt2024}

unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$WORK_DIR"
git pull --ff-only origin codex/tube_v1

"$PY" -m py_compile build_paper_figures_extended.py build_paper_figures_tables.py repo_integrity.py
"$PY" repo_integrity.py

"$PY" -u build_paper_figures_extended.py \
  --stack_dir "ens_heatcast_stack_opportunity/${WINDOW}" \
  --evidence_dir "paper_evidence_blocks/${WINDOW}" \
  --opportunity_dir "exceedance_eval_incremental/opportunity_${WINDOW}" \
  --output_dir "paper_figures_extended/${WINDOW}" \
  --heatcast_runs "$HEATCAST_RUNS" \
  --ens_runs "$ENS_RUNS" \
  --window_leads "$LEADS"

echo "EXTENDED PAPER FIGURES/TABLES COMPLETE"
