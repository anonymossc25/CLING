#!/bin/bash

# Run the command and save the output to a variable
cd /data/Framework/gSampler/examples
dataset=$1
#fanout = $2
#batch_size = $3
epoch=$2
output=$(python3 ladies/ladies_e2e.py --dataset=$1 --batchsize=1024  --sample="20000,260000,1172000" --num-epoch=$2)
#filename="SPMM_time_surendra/$1_F${fanout}_B${batch_size}_puregpu_$2.txt"
filename="../logs/ladies/$1_F20_B1024_E$2.txt"
#filename="epoch1_time_surendra/$1_F${fanout}_B${batch_size}_puregpu_$2.txt"
echo "Dataset = $1" > $filename
#done <<< "$output"
# Check if epoch_data.txt exists
if [ -f "epoch_data.txt" ]; then
        cat "epoch_data.txt" >> $filename
        echo "Data copied successfully!"
else
        echo "Error: epoch_data.txt does not exist."
fi

 
