#!/bin/bash
#SBATCH --job-name=eels_bf
#SBATCH --partition=compute
#SBATCH --account=innovation
#SBATCH --time=04:00:00
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --output=slurm-%j.out

# --- environment ---
module load miniconda3
# (if 'module load miniconda3' is unavailable, source your own install:)
# source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate meep

# helps MPICH/Slurm cooperate on some stacks
export OMP_NUM_THREADS=1

echo "Job $SLURM_JOB_ID on $SLURM_NTASKS tasks, node(s): $SLURM_NODELIST"
echo "Started: $(date)"

# MEEP parallelizes via MPI: srun launches one rank per task.
srun python eels_brute_force.py --cavity --out eels_${SLURM_JOB_ID}.npz

echo "Finished: $(date)"