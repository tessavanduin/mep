#!/bin/bash
#SBATCH --job-name=eels_bf
#SBATCH --partition=compute
#SBATCH --account=innovation
#SBATCH --time=04:00:00
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3900MB
#SBATCH --output=slurm-%j.out

set -euo pipefail

#--------------------------- environment --------------------------------------
# Replace this block with however MEEP is provided on your machine.  Common
# options are a module, a conda env, or a container.  Pick ONE.
#
#   module load meep                       # site module
#   source activate meep                   # conda/mamba env named 'meep'
#
# If unset, we just use whatever `python` / `mpirun` are on PATH.
if command -v module >/dev/null 2>&1; then
    module load meep 2>/dev/null || true   # harmless if no such module
fi
# source activate meep 2>/dev/null || true

# Number of MPI ranks: use SLURM's value if present, else fall back.
NTASKS="${SLURM_NTASKS:-4}"
MPIRUN="${MPIRUN:-mpirun}"
PYTHON="${PYTHON:-python}"

echo "Host: $(hostname)   MPI ranks: ${NTASKS}   $(date)"
"${PYTHON}" -c "import meep; print('MEEP', meep.__version__)"

mkdir -p EELS_3D-out

#--------------------------- shared physical parameters -----------------------
# Keep these identical across the empty/crystal/cavity runs so the empty
# subtraction is exact.  Edit here once.
COMMON="-a 426 -d 220 -x 36 -r 0.245 -n 3.45 -v 100 --y0 0 --z0 0"
RINGDOWN="--ringdown-factor 2.0"          # extra run time for mode ring-down

run () {            # run() <label> <extra-args...>
    local label="$1"; shift
    echo "=============================================================="
    echo ">>> ${label}   $(date)"
    echo "=============================================================="
    "${MPIRUN}" -np "${NTASKS}" "${PYTHON}" EELS_3D.py ${COMMON} "$@"
}

#--------------------------- 0. geometry sanity check -------------------------
"${PYTHON}" verify_geometry.py

#--------------------------- 1. one-off charge calibration --------------------
# Vacuum only.  Prints the injected charge; if it is not ~1, divide the induced
# field by the reported value (see README_EELS_physics.md).
run "charge check (vacuum)"        --empty --charge-check

#--------------------------- 2. vacuum reference ------------------------------
run "empty / vacuum reference"     --empty ${RINGDOWN}

#--------------------------- 3. slotted crystal -------------------------------
run "slotted photonic crystal"     ${RINGDOWN}

#--------------------------- 4. width-modulated cavity ------------------------
run "high-Q cavity"                --cavity ${RINGDOWN}

#--------------------------- 5. post-processing -------------------------------
echo ">>> post-processing $(date)"
"${PYTHON}" postprocess_all.py --method dft        # add: --method pade  for sharper lines

echo "ALL DONE  $(date).  Spectra (CSV + PNG) are in ./eels-spectra/"