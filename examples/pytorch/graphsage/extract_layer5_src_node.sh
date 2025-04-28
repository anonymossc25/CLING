#!/bin/bash

input_file=$1
output_file=$2

# Clear the output file
> "$output_file"

# Arrays to store src_nodes for each layer index
declare -a layer1 layer2 layer3 layer4 layer5

# Counter to track number of blocks
declare -i block_index=0

# Read file line-by-line
while read -r line; do
    if [[ $line =~ ^([0-9]+)[a-z]{2}\ layer:\ Block\(num_src_nodes=([0-9]+), ]]; then
        layer=${BASH_REMATCH[1]}
        src_nodes=${BASH_REMATCH[2]}
        case $layer in
            1) layer1[$block_index]=$src_nodes ;;
            2) layer2[$block_index]=$src_nodes ;;
            3) layer3[$block_index]=$src_nodes ;;
            4) layer4[$block_index]=$src_nodes ;;
            5) layer5[$block_index]=$src_nodes
               ((block_index++))  # increase only after layer 5
               ;;
        esac
    fi
done < "$input_file"

# Print to output file
for ((i=0; i<block_index; i++)); do
    echo "${layer1[i]} ${layer2[i]} ${layer3[i]} ${layer4[i]} ${layer5[i]}" >> "$output_file"
done

echo "Formatted src_nodes written to $output_file"

