#!/bin/bash

# Run the command and save the output to a variable
#cd /data/Framework/gSampler/examples
cd $DGL_HOME/gSampler/example
dataset=$1
#fanout = $2
#batch_size = $3
epoch=$2
#batch_sizes=(1024 2048 4096 8192 16384 32768 65536 131072)
#batch_sizes=(1024 2048 4096 8192)
#fanouts=(10 15 20)
#fanouts=(15)


# output=$(python3 node_classification.py --dataset=$1 --batch_size=1024)
#python3 node_classification.py --dataset=ogbn-products --batch_size=1024
# Initialize variables to keep track of the total sampling time and the last line that contained "Epoch"
#methods = ("gsampler" "ladies") 
#for method in "${methods[@]}"; do
  # Loop through each batch size
  #for batch_size in "${batch_sizes[@]}"; do
    output=$(python3 graphsage/graphsage_e2e.py --dataset=$1 --batchsize=1024  --samples="20,20,20" --num-epoch=$2)
    #filename="SPMM_time_surendra/$1_F${fanout}_B${batch_size}_puregpu_$2.txt"
    filename="../logs/gsampler/$1_F20_B1024_E$2.txt"
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
#done
