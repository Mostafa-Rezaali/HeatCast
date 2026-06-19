#!/bin/bash
#SBATCH --job-name=paper_fig_tables
#SBATCH --output=paper_fig_tables_%j.log
#SBATCH --error=paper_fig_tables_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

WORK_DIR=/blue/nessie/mostafarezaali/Teleconnection
PY=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python
WINDOW=${WINDOW:-window_15-16-17-18-19-20-21-22-23-24-25-26-27-28}

unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$WORK_DIR"
git pull --ff-only origin codex/tube_v1

"$PY" -m py_compile build_paper_figures_tables.py build_paper_evidence_blocks.py repo_integrity.py
"$PY" repo_integrity.py

"$PY" -u build_paper_figures_tables.py \
  --stack_dir "ens_heatcast_stack_opportunity/${WINDOW}" \
  --evidence_dir "paper_evidence_blocks/${WINDOW}" \
  --opportunity_dir "exceedance_eval_incremental/opportunity_${WINDOW}" \
  --output_dir "paper_figures_tables/${WINDOW}"

echo "PAPER FIGURES/TABLES PACKAGE COMPLETE"
