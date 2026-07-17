#!/bin/bash
#SBATCH --job-name=paper_blocks
#SBATCH --output=paper_blocks_%j.log
#SBATCH --error=paper_blocks_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

unset PYTHONHOME
unset PYTHONPATH
export PYTHONUNBUFFERED=1

PY=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python
WORK_DIR=/blue/nessie/mostafarezaali/Teleconnection

cd "$WORK_DIR"
git pull --ff-only origin main

"$PY" -m py_compile src/build_paper_evidence_blocks.py
"$PY" src/repo_integrity.py

"$PY" -u src/build_paper_evidence_blocks.py \
  --stack_dir ens_heatcast_stack_opportunity/window_15-16-17-18-19-20-21-22-23-24-25-26-27-28 \
  --opportunity_dir exceedance_eval_incremental/opportunity_window_15-16-17-18-19-20-21-22-23-24-25-26-27-28 \
  --output_dir paper_evidence_blocks/window_15-16-17-18-19-20-21-22-23-24-25-26-27-28
