#!/bin/bash
#SBATCH --job-name=stint_thin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH -p pi_fiete
#SBATCH --time=4:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o stint_thin_%x_%j.out

# Spike-thinned stim/integ PETHs for one insertion shard (STEP=compute), or
# the p100 check + per-variant stacks after all shards (STEP=stack).
# Reads manifold/insertion_cache + bwm_tables/trials.pqt under ONE_CACHE_DIR
# and stint_thin/data/stint_cells.npz. No ONE, no Alyx.
# Submit with stint_thin/submit_stint_thin.sh (not from the laptop).
# Env: STEP, SHARD_IDX, N_SHARDS, SEEDS, OUT

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/bwm_dmn}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR

STEP="${STEP:?Set STEP=compute|stack}"
OUT="${OUT:-$ONE_CACHE_DIR/dmn/res/stint_thin}"
SEEDS="${SEEDS:-0 1 2 3 4}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"
echo "Host: $(hostname) Date: $(date) STEP=$STEP OUT=$OUT SEEDS=$SEEDS"
git log -1 --oneline

# shellcheck disable=SC2086
if [[ "$STEP" == "compute" ]]; then
  python3 -u stint_thin/stint_thin.py compute \
    --shard "${SHARD_IDX:?Set SHARD_IDX}" --n-shards "${N_SHARDS:?Set N_SHARDS}" \
    --seeds $SEEDS --out "$OUT" --one-cache-dir "$ONE_CACHE_DIR"
elif [[ "$STEP" == "stack" ]]; then
  python3 -u stint_thin/stint_thin.py stack --out "$OUT"
  ls -lh "$OUT"
else
  echo "unknown STEP=$STEP" >&2; exit 2
fi
echo "Done: $(date)"
