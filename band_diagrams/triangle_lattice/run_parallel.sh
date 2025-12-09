#!/bin/bash
# Code produced with help of chatGPT

# Save old values
OLD_OMP=$OMP_NUM_THREADS
OLD_MKL=$MKL_NUM_THREADS

# Limit each process to 1 core
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1



# Usage: ./run_parallel.sh N p
if [ $# -ne 2 ]; then
    echo "Usage: $0 N p"
    exit 1
fi

N="$1"
p="$2" # The least number of k values that can be calculated per core is 1, so p is at most N.


num_jobs=$((p))
outputs=()

for ((n=0; n<num_jobs; n++)); do
    python Band_Diagram_3D_Parallel.py -N "$N" -p "$p" -n "$n" > output_$n.txt 2>&1 &
    outputs+=($!)
done

# Wait for all jobs
for pid in "${outputs[@]}"; do
    wait "$pid"
done

# Concatenate all outputs
if [ "$N" -eq "$p" ]; then # 1 k point per core
    bash extract_freqs.sh "$N"
else
    cat output_*.txt | grep freqs: > fre.dat
    cat output_*.txt | grep freqs-im: > fim.dat
fi

# Clean up individual output files
rm output_*.txt

# Move the dat files to their own directory
outdir=dot_dat_files
mkdir -p $outdir
mv fre.dat $outdir/fre.dat
mv fim.dat $outdir/fim.dat


# Restore old settings
export OMP_NUM_THREADS=$OLD_OMP
export MKL_NUM_THREADS=$OLD_MKL
