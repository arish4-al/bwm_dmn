# Slurm submit defaults. Source after PARTITION is set (and after cd "$REPO_DIR").
#
#   # shellcheck disable=SC1091
#   source "$REPO_DIR/stint_thin/sbatch_defaults.sh"
#   sbatch --parsable $SBATCH_EXTRA --partition="$PARTITION" ...
#
# If PARTITION is mit_preemptable or mit_preem and SBATCH_EXTRA is unset,
# defaults SBATCH_EXTRA=--requeue. Explicit SBATCH_EXTRA (including "") wins.
# Requeued thinning shards skip insertions already written.
#
# Copied from prior-mech-analysis/scripts. New submit scripts must source this and pass $SBATCH_EXTRA to every sbatch.

if [[ -z "${SBATCH_EXTRA+x}" ]]; then
  case "${PARTITION:-}" in
    mit_preemptable|mit_preem) SBATCH_EXTRA="--requeue" ;;
    *) SBATCH_EXTRA="" ;;
  esac
fi
export SBATCH_EXTRA

# Keep the login node and every job off Alyx (one_offline.py): python run by
# the submit wrapper (e.g. --list-splits) opens ONE offline, and sbatch
# --export=ALL passes this on. Jobs are also blocked by SLURM_JOB_ID alone.
export BLOCK_ALYX="${BLOCK_ALYX:-1}"
