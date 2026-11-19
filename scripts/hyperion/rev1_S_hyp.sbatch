#!/bin/bash
# Revision 1, study S on hyperion (DEVIATIONS R1-1: moved from the laptop because of power throttling).
# One array task = one model = one line of rev1_S_hyp_tasks.txt. ALL 200 study-S models (real s0 re-run + 99
# size-bin placebos per market) run here, so every graph in the H-S rank tests comes from one platform.
#   NASDAQ lines 1-100, NYSE lines 101-200:  sbatch --array=1-200 scripts/hyperion/rev1_S_hyp.sbatch
#SBATCH --job-name=rev1_S_hyp
#SBATCH --account=bcam-exclusive
#SBATCH --partition=bcam-exclusive
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/hilmibukhori/pcgv/outputs/rev1_rsr_sizebin_v1/logs/slurm_%A_%a.out
source /etc/profile >/dev/null 2>&1
module load PyTorch/2.1.2-foss-2022b
cd /scratch/hilmibukhori/pcgv
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ARGS="$(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/hyperion/rev1_S_hyp_tasks.txt)"
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): $ARGS"
python -s -u scripts/run_rev1_rsr_sizebin.py $ARGS
