#!/bin/bash
# Extension pre-freeze: tests (task 0) and dev-split timing/memory smoke per cell (tasks 1-3). No 2020 data.
#SBATCH --job-name=ext_smoke
#SBATCH --account=bcam-exclusive
#SBATCH --partition=bcam-exclusive
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/hilmibukhori/pcgv/logs/ext_smoke_%A_%a.out
source /etc/profile >/dev/null 2>&1
module load PyTorch/2.1.2-foss-2022b
cd /scratch/hilmibukhori/pcgv
echo "task $SLURM_ARRAY_TASK_ID on $(hostname) $(lscpu | grep 'Model name' | cut -c 34-)"
case $SLURM_ARRAY_TASK_ID in
  0) python -s -m pytest -q -p no:cacheprovider tests/test_rtgcn_ext.py tests/test_rtgcn_replica.py 2>&1 | tail -15 ;;
  1) C="--market NYSE --variant U" ;;
  2) C="--market NASDAQ --variant T" ;;
  3) C="--market NYSE --variant T" ;;
esac
if [ -n "$C" ]; then
  for th in 1 4; do
    /usr/bin/time -v python -s -u scripts/run_rtgcn_ext.py $C --mode dev --graph real --seeds 0 --epochs 2 --threads $th \
      --out outputs/rtgcn_ext_smoke/th$th --log-epochs 2>&1 | grep -E "done in|ep |Maximum resident|Elapsed|ERROR|Error"
  done
fi
