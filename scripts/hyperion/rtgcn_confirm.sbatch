#!/bin/bash
# RT-GCN(U) NASDAQ confirm on hyperion (DEVIATIONS RT2). One array task = one model = one line of
# rtgcn_confirm_tasks.txt (exactly the model set of the frozen laptop queues). The runner is the frozen
# scripts/run_rtgcn_replica.py: it verifies the freeze, uses 1 thread, skips finished npz and resumes
# per-epoch checkpoints, so re-submitting the same array is safe (second pass = gap filling).
#SBATCH --job-name=rtgcn_confirm
#SBATCH --account=bcam-exclusive
#SBATCH --partition=bcam-exclusive
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --array=1-94
#SBATCH --output=/scratch/hilmibukhori/pcgv/outputs/rtgcn_v1/logs/slurm_%A_%a.out
source /etc/profile >/dev/null 2>&1
module load PyTorch/2.1.2-foss-2022b
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /scratch/hilmibukhori/pcgv
ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/hyperion/rtgcn_confirm_tasks.txt)
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): $ARGS"
python -s -u scripts/run_rtgcn_replica.py $ARGS
