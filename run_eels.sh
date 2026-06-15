#!/bin/bash
#SBATCH --job-name=eels_bf
#SBATCH --partition=compute
#SBATCH --account=innovation
#SBATCH --time=08:00:00
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3900MB
#SBATCH --output=slurm-%j.out

module load miniconda3
# source $HOME/miniforge3/etc/profile.d/conda.sh   # uncomment if module not found
conda activate meep
export OMP_NUM_THREADS=1

echo "Job $SLURM_JOB_ID on $SLURM_NTASKS tasks: $(date)"

# FULL run: pulse-chain source (default), broadband over 0-1.2 eV, LONG ringdown
# so the broad alpha/beta/gamma slot/band modes decay enough for Pade to resolve
# their linewidths. (The Q=2.5e5 cavity mode never decays, but its weight is tiny
# and its width << 30 meV convolution, so it becomes a smooth bump, not ripple.)
# Fields saved so the transform can be re-run cheaply over any window / Pade order.
srun python -u eels_brute_force.py --cavity \
     --source-model pulse-chain \
     --src-Emin 0 --src-Emax 1.2 \
     --Textra 1800 \
     --save-fields fields_${SLURM_JOB_ID}.npz

echo "Finished: $(date)"