#!/bin/bash

# Run the command and save the output to a variable
output=$(python3 node_classification1.py --dataset ogbn-arxiv --mode puregpu --epoch 10  2>&1)

# Initialize variables to keep track of the total sampling time and the last line that contained "Epoch"
sampling_time=0
last_epoch_line=false

# Loop through the output lines
while read -r line; do
  # Check if the line contains the string "cuda,sapmling"
  if [[ $line == cuda,sapmling* ]]; then
    # Extract the time value and add it to the sampling time
    time_value=$(echo $line | awk '{print $4}')
    sampling_time=$((sampling_time + time_value))
  # Check if the line contains the string "Epoch"
  elif [[ $line == Epoch* ]]; then
    # If this is the first "Epoch" line, print it as-is
    if ! $last_epoch_line; then
      echo $line
      last_epoch_line=true
    # If this is a subsequent "Epoch" line, print it with the total sampling time
    else
      echo "Epoch ${line#*|} | Sampling Time: $sampling_time"
      last_epoch_line=false
    fi
  # If the line is not "Epoch" or "cuda,sapmling", print it as-is
  else
    echo $line
  fi  
done <<< "$output"

echo $sampling_time

# Print the total training time
echo "Total Training time: $(echo $output | grep -oP 'Total Training time \K[\d.]+')"
