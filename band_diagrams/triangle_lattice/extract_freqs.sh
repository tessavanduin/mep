#!/bin/bash
# Extract real and imaginary frequencies from output*_.txt files
# Code produced with help of chatGPT

# Usage: ./extract_freqs.sh N
if [ $# -ne 1 ]; then
    echo "Usage: $0 N"
    exit 1
fi

N="$1"
scale=$(echo "0.5/($N-1)" | bc -l)

real_out="fre.dat"
imag_out="fim.dat"

> "$real_out"
> "$imag_out"

line_num=1

for file in $(ls output_*.txt | sort -V); do
    # Extract k value from filename (number between 'output_' and '_.txt')
    k_value=$(echo "$file" | sed -E 's/.*output_([0-9]+)\.txt/\1/')
    if [ -z "$k_value" ]; then
        continue
    fi
    # Multiply k_value by scale and round to 17 significant digits
    k_scaled=$(printf "%.17g" "$(echo "$k_value * $scale" | bc -l)")
    # Pad with leading zero if needed
    k_scaled=$(echo "$k_scaled" | sed -E 's/^(\.[0-9]+)/0\1/')
    # If k_scaled is exactly 0, set to 0.0
    if [ "$k_scaled" = "0" ]; then
        k_scaled="0.0"
    fi

    # Extract real frequencies (2nd field after 'harminv0:,') and remove "frequency"
    real_freqs=$(grep "^harminv0:" "$file" | awk -F',' '{print $2}' | grep -v "frequency" | tr '\n' ',' | sed 's/,$//')
    # Extract imaginary frequencies (3rd field)
    imag_freqs=$(grep "^harminv0:" "$file" | awk -F',' '{print $3}' | grep -v "imag. freq." | tr '\n' ',' | sed 's/,$//')

    # Write to output files with line number
    echo "$line_num, $k_scaled, 0.0, 0.0,$real_freqs" >> "$real_out"
    echo "$line_num, $k_scaled, 0.0, 0.0,$imag_freqs" >> "$imag_out"
    line_num=$((line_num + 1))
done

