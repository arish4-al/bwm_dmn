#!/bin/bash
#SBATCH --job-name=seq_detect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p pi_fiete
#SBATCH --time=2:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o seq_detect_%x_%j.out

# Taxidis/Dorian sequence-cell detection (circular-shuffle peak test, trial
# reliability) for all good units on the six stimulus-aligned types, for one
# insertion shard (STEP=compute), or the concatenation of all shards (STEP=stack).
# Reads manifold/insertion_cache + bwm_tables/trials.pqt under ONE_CACHE_DIR
# and bwm_tables/clusters.pqt. No ONE, no Alyx.
# Submit with seq_detect/submit_seq_detect.sh (not from the laptop).
# Env: STEP, SHARD_IDX, N_SHARDS, OUT

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/bwm_dmn}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR

STEP="${STEP:?Set STEP=compute|stack}"
OUT="${OUT:-$ONE_CACHE_DIR/dmn/res/seq_detect}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"
echo "Host: $(hostname) Date: $(date) STEP=$STEP OUT=$OUT"
git log -1 --oneline

if [[ "$STEP" == "compute" ]]; then
  python3 -u seq_detect/seq_detect.py compute \
    --shard "${SHARD_IDX:?Set SHARD_IDX}" --n-shards "${N_SHARDS:?Set N_SHARDS}" \
    --out "$OUT" --one-cache-dir "$ONE_CACHE_DIR"
elif [[ "$STEP" == "stack" ]]; then
  python3 -u seq_detect/seq_detect.py stack --out "$OUT"
  ls -lh "$OUT"/seq_detect.npz
else
  echo "unknown STEP=$STEP" >&2; exit 2
fi
echo "Done: $(date)"
