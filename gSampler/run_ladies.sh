#!/bin/bash

# Run the command and save the output to a variable
#cd /data/Framework/gSampler/examples
cd $DGL_HOME/gSampler/examples
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
if [ "$dataset" == "ogbn-arxiv" ]; then
    output=$(python3 ladies/ladies_e2e.py --dataset=$1 --batchsize=1024  --sample="5700,19900,42000" --num-epoch=$2)
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
elif [ "$dataset" == "reddit" ]; then
    output=$(python3 ladies/ladies_e2e.py --dataset=$1 --batchsize=1024  --sample="18000,130000,206000" --num-epoch=$2)
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
elif [ "$dataset" == "ogbn-products" ]; then
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
else
   echo "please provide correct dataset"
fi  
