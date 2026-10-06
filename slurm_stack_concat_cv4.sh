#!/bin/bash
#SBATCH --job-name=stack_cv4_allvers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p mit_normal
#SBATCH --time=24:00:00
#SBATCH --mail-user=arily
#SBATCH --mail-type=FAIL
#SBATCH -o stack_cv4_allvers_%j.out

# Prevent oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module load miniforge
conda activate ~/conda_envs/ibl

python3 << EOF
import dmn_bwm

# All PETH_types_dict versions except "concat" (each has its own cache subfolder under dmn/<vers>/).
for vers in dmn_bwm.PETH_types_dict:
    if vers == "concat":
        continue
    print(f"\n{'='*60}\nstack_concat CV x4: vers={vers!r}\n{'='*60}")
    dmn_bwm.stack_concat(
        vers=vers,
        cv=True,
        cv_splits=4,
        cv_seed=0,
    )
EOF

