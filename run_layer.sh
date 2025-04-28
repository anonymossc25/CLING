#!/bin/bash

# Run the command and save the output to a variable
cd $DGL_HOME/examples/pytorch/graphsage
dataset=$1
#fanout = $2
#batch_size = $3
epoch=$2
#batch_sizes=(1024 2048 4096 8192 16384 32768 65536)
#batch_sizes=(2048)
#fanouts=(15 20)
#fanouts=(10 15 20)
methods=("graphsage" "cling")
for method in "${methods[@]}"; do
      output=$(python3 node_classification1.py --dataset=$1 --batch_size=1024 --mode=puregpu --fanout="20,20,20" --method=$method --epoch=$2)
      filename="$DGL_HOME/logs/layer/three-layer/${method}/$1_F20_B1024_E$2.txt"
      echo "Dataset = $1" > $filename 
          # Check if epoch_data.txt exists
          if [ -f "epoch_data.txt" ]; then
            cat "epoch_data.txt" >> $filename
            echo "Data copied successfully!"
          else
            echo "Error: epoch_data.txt does not exist."
          fi
      output=$(python3 node_classification_layer4.py --dataset=$1 --batch_size=1024 --mode=puregpu --fanout="20,20,20,20" --method=$method --epoch=$2)
      filename="$DGL_HOME/logs/layer/four-layer/${method}/$1_F20_B1024_E$2.txt"
      echo "Dataset = $1" > $filename 
          # Check if epoch_data.txt exists
          if [ -f "epoch_data.txt" ]; then
            cat "epoch_data.txt" >> $filename
            echo "Data copied successfully!"
          else
            echo "Error: epoch_data.txt does not exist."
          fi
      output=$(python3 node_classification_layer5.py --dataset=$1 --batch_size=1024 --mode=puregpu --fanout="20,20,20,20,20" --method=$method --epoch=$2)
      filename="$DGL_HOME/logs/layer/five-layer/${method}/$1_F20_B1024_E$2.txt"
      echo "Dataset = $1" > $filename 
          # Check if epoch_data.txt exists
          if [ -f "epoch_data.txt" ]; then
            cat "epoch_data.txt" >> $filename
            echo "Data copied successfully!"
          else
            echo "Error: epoch_data.txt does not exist."
          fi
done   
