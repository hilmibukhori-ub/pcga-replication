#!/bin/bash
# Revision 1, study K (PREREG_REV1.md): RT-GCN(U) placebos perm019-perm098 at seed 0, val arm, with the FROZEN
# runners. One array task = one model = one line of rev1_k99_tasks.txt (first token = cell, then runner + args).
#   NASDAQ_U lines 1-80   : sbatch --array=1-80   --mem=6G  scripts/hyperion/rev1_k99.sbatch
#   NYSE_U   lines 81-160 : sbatch --array=81-160 --mem=10G scripts/hyperion/rev1_k99.sbatch
# Each task first verifies PREREG_REV1_FREEZE.md; the frozen runners then verify their own parent freeze records.
# Runners skip finished npz and resume checkpoints, so re-submitting a range fills gaps.
#SBATCH --job-name=rev1_k99
#SBATCH --account=bcam-exclusive
#SBATCH --partition=bcam-exclusive
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/hilmibukhori/pcgv/outputs/rev1_k99_logs/slurm_%A_%a.out
source /etc/profile >/dev/null 2>&1
module load PyTorch/2.1.2-foss-2022b
cd /scratch/hilmibukhori/pcgv
python -s scripts/rev1_verify_freeze.py || exit 1
read -r CELL SCRIPT ARGS <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/hyperion/rev1_k99_tasks.txt)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "task $SLURM_ARRAY_TASK_ID [$CELL] on $(hostname): $SCRIPT $ARGS"
python -s -u $SCRIPT $ARGS
