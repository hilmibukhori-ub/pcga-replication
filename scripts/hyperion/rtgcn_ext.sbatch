#!/bin/bash
# RT-GCN extension confirm runs (PREREG_RTGCN_EXT.md §11). One array task = one model = one line of
# rtgcn_ext_tasks.txt (first token = cell, rest = runner args). Submitted per cell with its own resources:
#   NYSE_T   lines 1-44   : sbatch --array=1-44   --cpus-per-task=4 --mem=40G
#   NASDAQ_T lines 45-88  : sbatch --array=45-88  --cpus-per-task=4 --mem=28G
#   NYSE_U   lines 89-157 : sbatch --array=89-157 --cpus-per-task=1 --mem=10G
# The runner verifies PREREG_RTGCN_EXT_FREEZE.md, fixes threads per cell, skips finished npz and resumes
# per-epoch checkpoints, so re-submitting the same range is safe (gap filling).
#SBATCH --job-name=rtgcn_ext
#SBATCH --account=bcam-exclusive
#SBATCH --partition=bcam-exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/hilmibukhori/pcgv/outputs/rtgcn_ext_v1/logs/slurm_%A_%a.out
source /etc/profile >/dev/null 2>&1
module load PyTorch/2.1.2-foss-2022b
cd /scratch/hilmibukhori/pcgv
read -r CELL ARGS <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/hyperion/rtgcn_ext_tasks.txt)"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "task $SLURM_ARRAY_TASK_ID [$CELL] on $(hostname): $ARGS"
python -s -u scripts/run_rtgcn_ext.py $ARGS
