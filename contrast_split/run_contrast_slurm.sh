#!/bin/bash
#SBATCH --job-name=contrast_split
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -p pi_fiete
#SBATCH --time=1:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o contrast_split_%x_%j.out

# Odd/even PETHs of all good units on the six stimulus-aligned types, split
# by contrast, for one insertion shard (STEP=compute), or the 'all' check +
# per-group stacks after all shards (STEP=stack).
# Reads manifold/insertion_cache + bwm_tables/trials.pqt under ONE_CACHE_DIR
# and bwm_tables/clusters.pqt. No ONE, no Alyx.
# Submit with contrast_split/submit_contrast.sh (not from the laptop).
# Env: STEP, SHARD_IDX, N_SHARDS, OUT

set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/bwm_dmn}"
ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
export ONE_CACHE_DIR

STEP="${STEP:?Set STEP=compute|stack}"
OUT="${OUT:-$ONE_CACHE_DIR/dmn/res/contrast_split}"

module load miniforge
conda activate ~/conda_envs/ibl
cd "$REPO_DIR"
echo "Host: $(hostname) Date: $(date) STEP=$STEP OUT=$OUT"
git log -1 --oneline

if [[ "$STEP" == "compute" ]]; then
  python3 -u contrast_split/contrast_peths.py compute \
    --shard "${SHARD_IDX:?Set SHARD_IDX}" --n-shards "${N_SHARDS:?Set N_SHARDS}" \
    --out "$OUT" --one-cache-dir "$ONE_CACHE_DIR"
elif [[ "$STEP" == "stack" ]]; then
  python3 -u contrast_split/contrast_peths.py stack --out "$OUT"
  ls -lh "$OUT"/contrast_peths.npz
else
  echo "unknown STEP=$STEP" >&2; exit 2
fi
echo "Done: $(date)"
