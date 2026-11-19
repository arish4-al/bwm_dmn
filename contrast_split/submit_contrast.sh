#!/bin/bash
# Contrast-split PETHs of all good units: N_SHARDS compute jobs over the
# 515 stack insertions, then one stack job (afterok).
#
#   bash contrast_split/submit_contrast.sh
#
# Needs bwm_tables/clusters.pqt (good units, label == 1),
# manifold/insertion_cache + bwm_tables/trials.pqt under ONE_CACHE_DIR.
# Shards skip insertions already written, so a rerun only fills gaps.
# Local timing: about 0.5 s per insertion (load + bin, all four groups at once)
# and 1.5 GB peak, so each of the 4 shards (128-129 insertions) should take a
# few minutes; the time limit is 1 h.
#
# Override: N_SHARDS=8 MEM_SHARD=8G PARTITION=mit_preemptable \
#   bash contrast_split/submit_contrast.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/bwm_dmn}"
cd "$REPO_DIR"

ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
OUT="${OUT:-$ONE_CACHE_DIR/dmn/res/contrast_split}"
N_SHARDS="${N_SHARDS:-4}"
MEM_SHARD="${MEM_SHARD:-4G}"
MEM_STACK="${MEM_STACK:-8G}"
PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/stint_thin/sbatch_defaults.sh"

[[ -f "$ONE_CACHE_DIR/bwm_tables/clusters.pqt" ]] || {
  echo "missing $ONE_CACHE_DIR/bwm_tables/clusters.pqt" >&2; exit 1; }
[[ -f "$ONE_CACHE_DIR/bwm_tables/trials.pqt" ]] || {
  echo "missing $ONE_CACHE_DIR/bwm_tables/trials.pqt" >&2; exit 1; }
[[ -d "$ONE_CACHE_DIR/manifold/insertion_cache" ]] || {
  echo "missing $ONE_CACHE_DIR/manifold/insertion_cache" >&2; exit 1; }

COMMON="ALL,REPO_DIR=$REPO_DIR,ONE_CACHE_DIR=$ONE_CACHE_DIR,OUT=$OUT"
echo "PARTITION=$PARTITION N_SHARDS=$N_SHARDS MEM_SHARD=$MEM_SHARD OUT=$OUT"

SHARD_JOBS=()
for ((k=0; k<N_SHARDS; k++)); do
  # shellcheck disable=SC2086
  JID=$(sbatch --parsable $SBATCH_EXTRA \
    --partition="$PARTITION" \
    --mem="$MEM_SHARD" --cpus-per-task=2 \
    --job-name="ctr_s${k}" \
    --export="$COMMON,STEP=compute,SHARD_IDX=$k,N_SHARDS=$N_SHARDS" \
    contrast_split/run_contrast_slurm.sh)
  SHARD_JOBS+=("$JID")
  echo "  shard $k/$N_SHARDS -> $JID"
done
DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
# shellcheck disable=SC2086
SID=$(sbatch --parsable $SBATCH_EXTRA \
  --partition="$PARTITION" \
  --mem="$MEM_STACK" --cpus-per-task=2 \
  --dependency=afterok:"$DEP" \
  --job-name="ctr_stack" \
  --export="$COMMON,STEP=stack" \
  contrast_split/run_contrast_slurm.sh)
echo "  stack -> $SID (after $DEP)"
