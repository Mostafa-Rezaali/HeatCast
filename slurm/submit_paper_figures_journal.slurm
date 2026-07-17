#!/bin/bash
#SBATCH --job-name=paper_fig_journal
#SBATCH --output=paper_fig_journal_%j.log
#SBATCH --error=paper_fig_journal_%j.err
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
W34_LOG_GLOB=${W34_LOG_GLOB:-w34_tube_all_*.log,w34_tube_all_*.err,*w34_dist*.log,*w34*.log,*w34*.err}
if [ -z "${ENS_RUNS:-}" ]; then
  ENS_RUNS='cvfold{F}_ens_w34,cvfold{F}_ens_w34_rt2024'
fi

unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$WORK_DIR"
git pull --ff-only origin main

"$PY" -m py_compile src/figure_style.py src/build_paper_figures_tables.py src/build_paper_figures_extended.py src/repo_integrity.py
"$PY" src/repo_integrity.py

"$PY" -u src/build_paper_figures_tables.py \
  --stack_dir "ens_heatcast_stack_opportunity/${WINDOW}" \
  --evidence_dir "paper_evidence_blocks/${WINDOW}" \
  --output_dir "paper_figures_tables/${WINDOW}"

"$PY" -u src/build_paper_figures_extended.py \
  --stack_dir "ens_heatcast_stack_opportunity/${WINDOW}" \
  --teleconnection_stack_dir "ens_heatcast_stack_opportunity_teleconnections/${WINDOW}" \
  --evidence_dir "paper_evidence_blocks/${WINDOW}" \
  --opportunity_dir "exceedance_eval_incremental/opportunity_${WINDOW}" \
  --output_dir "paper_figures_extended/${WINDOW}" \
  --heatcast_runs "$HEATCAST_RUNS" \
  --ens_runs "$ENS_RUNS" \
  --window_leads "$LEADS" \
  --w34_log_glob "$W34_LOG_GLOB"

echo "JOURNAL PAPER FIGURES/TABLES COMPLETE"
