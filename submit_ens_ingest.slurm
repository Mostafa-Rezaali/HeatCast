#!/bin/bash
#SBATCH --job-name=ens_ingest
#SBATCH --output=ens_ingest_%j.log
#SBATCH --error=ens_ingest_%j.err
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

cd "$WORK_DIR"
module load cuda/12.9.1
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

git pull --ff-only origin codex/tube_v1
"$PY" -m py_compile repo_integrity.py ens_common.py ens_ingest.py
"$PY" repo_integrity.py
"$PY" -u ens_ingest.py \
  --raw_dir ens_reforecast/raw \
  --output_dir ens_reforecast/regridded \
  --init_list ens_reforecast/raw/init_list.txt \
  --variable mx2t6 \
  --weekdays 0,3 \
  --max_lead 28 \
  --expected_members 11 \
  --workers 8
