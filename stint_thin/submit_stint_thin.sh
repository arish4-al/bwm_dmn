#!/bin/bash
# Spike-thinned stim/integ cells: N_SHARDS compute jobs over the 436
# insertions that hold stim/integ cells, then one stack job (afterok).
#
#   bash stint_thin/submit_stint_thin.sh
#
# Needs: stint_thin/data/stint_cells.npz (prep_cells.py, committed), and
# manifold/insertion_cache + bwm_tables/trials.pqt under ONE_CACHE_DIR.
# Locally a cached insertion took 4-8 s with 2 seeds; expect ~10-20 s with 5.
# Shards skip insertions already written, so a rerun only fills gaps.
#
# Override: N_SHARDS=16 MEM_SHARD=8G PARTITION=mit_preemptable SEEDS="0 1 2" \
#   bash stint_thin/submit_stint_thin.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/bwm_dmn}"
cd "$REPO_DIR"

ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
OUT="${OUT:-$ONE_CACHE_DIR/dmn/res/stint_thin}"
SEEDS="${SEEDS:-0 1 2 3 4}"
N_SHARDS="${N_SHARDS:-8}"
MEM_SHARD="${MEM_SHARD:-6G}"
MEM_STACK="${MEM_STACK:-8G}"
PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/stint_thin/sbatch_defaults.sh"

[[ -f stint_thin/data/stint_cells.npz ]] || { echo "missing stint_thin/data/stint_cells.npz" >&2; exit 1; }
[[ -f "$ONE_CACHE_DIR/bwm_tables/trials.pqt" ]] || {
  echo "missing $ONE_CACHE_DIR/bwm_tables/trials.pqt" >&2; exit 1; }
[[ -d "$ONE_CACHE_DIR/manifold/insertion_cache" ]] || {
  echo "missing $ONE_CACHE_DIR/manifold/insertion_cache" >&2; exit 1; }

COMMON="ALL,REPO_DIR=$REPO_DIR,ONE_CACHE_DIR=$ONE_CACHE_DIR,OUT=$OUT,SEEDS=$SEEDS"
echo "PARTITION=$PARTITION N_SHARDS=$N_SHARDS MEM_SHARD=$MEM_SHARD OUT=$OUT SEEDS=$SEEDS"

SHARD_JOBS=()
for ((k=0; k<N_SHARDS; k++)); do
  # shellcheck disable=SC2086
  JID=$(sbatch --parsable $SBATCH_EXTRA \
    --partition="$PARTITION" \
    --mem="$MEM_SHARD" --cpus-per-task=2 \
    --job-name="thin_s${k}" \
    --export="$COMMON,STEP=compute,SHARD_IDX=$k,N_SHARDS=$N_SHARDS" \
    stint_thin/run_stint_thin_slurm.sh)
  SHARD_JOBS+=("$JID")
  echo "  shard $k/$N_SHARDS -> $JID"
done
DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
# shellcheck disable=SC2086
SID=$(sbatch --parsable $SBATCH_EXTRA \
  --partition="$PARTITION" \
  --mem="$MEM_STACK" --cpus-per-task=2 \
  --dependency=afterok:"$DEP" \
  --job-name="thin_stack" \
  --export="$COMMON,STEP=stack" \
  stint_thin/run_stint_thin_slurm.sh)
echo "  stack -> $SID (after $DEP)"
