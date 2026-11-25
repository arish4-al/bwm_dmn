#!/bin/bash
# Sequence-cell detection of all good units: N_SHARDS compute jobs over the
# 515 stack insertions, then one stack job (afterok).
#
#   bash seq_detect/submit_seq_detect.sh
#
# Needs bwm_tables/clusters.pqt (good units, label == 1),
# manifold/insertion_cache + bwm_tables/trials.pqt under ONE_CACHE_DIR.
# Shards skip insertions already written, so a rerun only fills gaps.
# Local timing: about 0.06 s per unit (2000 shuffles x 2 halves x 6 types) and
# 1.4 GB peak for a 234-unit insertion. 16 shards of 2,850-4,290 units: about
# 5 min each; the time limit is 2 h.
#
# Pooled trials (one type, the union of the six; output seq_detect_pooled/):
#   POOLED=1 bash seq_detect/submit_seq_detect.sh
#
# One contrast group (low, high, low_nz; output seq_detect_pooled_{group}/):
#   POOLED=1 CONTRAST=low bash seq_detect/submit_seq_detect.sh
#   POOLED=1 CONTRAST=high bash seq_detect/submit_seq_detect.sh
#
# Override: N_SHARDS=32 MEM_SHARD=12G PARTITION=mit_preemptable \
#   bash seq_detect/submit_seq_detect.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/int-brain-lab/bwm_dmn}"
cd "$REPO_DIR"

ONE_CACHE_DIR="${ONE_CACHE_DIR:-/orcd/data/fiete/001/om2/arily/int-brain-lab/ONE/alyx}"
POOLED="${POOLED:-}"
CONTRAST="${CONTRAST:-}"
case "$CONTRAST" in ""|low|high|low_nz) ;; *) echo "CONTRAST=$CONTRAST: use low, high or low_nz" >&2; exit 2;; esac
if [[ -n "$POOLED" ]]; then
  OUT="${OUT:-$ONE_CACHE_DIR/dmn/res/seq_detect_pooled${CONTRAST:+_$CONTRAST}}"
else
  OUT="${OUT:-$ONE_CACHE_DIR/dmn/res/seq_detect${CONTRAST:+_$CONTRAST}}"
fi
N_SHARDS="${N_SHARDS:-16}"
MEM_SHARD="${MEM_SHARD:-8G}"
MEM_STACK="${MEM_STACK:-16G}"
PARTITION="${PARTITION:-pi_fiete}"
# shellcheck disable=SC1091
source "$REPO_DIR/stint_thin/sbatch_defaults.sh"

[[ -f "$ONE_CACHE_DIR/bwm_tables/clusters.pqt" ]] || {
  echo "missing $ONE_CACHE_DIR/bwm_tables/clusters.pqt" >&2; exit 1; }
[[ -f "$ONE_CACHE_DIR/bwm_tables/trials.pqt" ]] || {
  echo "missing $ONE_CACHE_DIR/bwm_tables/trials.pqt" >&2; exit 1; }
[[ -d "$ONE_CACHE_DIR/manifold/insertion_cache" ]] || {
  echo "missing $ONE_CACHE_DIR/manifold/insertion_cache" >&2; exit 1; }

COMMON="ALL,REPO_DIR=$REPO_DIR,ONE_CACHE_DIR=$ONE_CACHE_DIR,OUT=$OUT,POOLED=$POOLED,CONTRAST=$CONTRAST"
echo "PARTITION=$PARTITION N_SHARDS=$N_SHARDS MEM_SHARD=$MEM_SHARD OUT=$OUT POOLED=$POOLED CONTRAST=$CONTRAST"

SHARD_JOBS=()
for ((k=0; k<N_SHARDS; k++)); do
  # shellcheck disable=SC2086
  JID=$(sbatch --parsable $SBATCH_EXTRA \
    --partition="$PARTITION" \
    --mem="$MEM_SHARD" --cpus-per-task=2 \
    --job-name="sqd_s${k}" \
    --export="$COMMON,STEP=compute,SHARD_IDX=$k,N_SHARDS=$N_SHARDS" \
    seq_detect/run_seq_detect_slurm.sh)
  SHARD_JOBS+=("$JID")
  echo "  shard $k/$N_SHARDS -> $JID"
done
DEP=$(IFS=:; echo "${SHARD_JOBS[*]}")
# shellcheck disable=SC2086
SID=$(sbatch --parsable $SBATCH_EXTRA \
  --partition="$PARTITION" \
  --mem="$MEM_STACK" --cpus-per-task=2 \
  --dependency=afterok:"$DEP" \
  --job-name="sqd_stack" \
  --export="$COMMON,STEP=stack" \
  seq_detect/run_seq_detect_slurm.sh)
echo "  stack -> $SID (after $DEP)"
