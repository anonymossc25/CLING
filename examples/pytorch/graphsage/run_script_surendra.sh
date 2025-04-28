#!/bin/bash

# Run the command and save the output to a variable
dataset=$1
#fanout = $2
#batch_size = $3
epoch=$2
#batch_sizes=(1024 2048 4096 8192 16384 32768 65536 131072)
batch_sizes=(1024 2048 4096 8192)
#fanouts=(10 15 20)
fanouts=(15)


# output=$(python3 node_classification.py --dataset=$1 --batch_size=1024)
#python3 node_classification.py --dataset=ogbn-products --batch_size=1024
# Initialize variables to keep track of the total sampling time and the last line that contained "Epoch"
for fanout in "${fanouts[@]}"; do
  # Loop through each batch size
  for batch_size in "${batch_sizes[@]}"; do
    sampling_time=0.0
    training_time=0.0
    iteration_time=0.0
    add_cusparse_time=true
    array_time=0.0
    model_time=0.0
    spmm_time=0.0
    induced_time=0.0

    neighbor_time=0.0
    #counter=0.0
    #degree_kernel_time=0.0
    start_string="spmm"
    end_string="seconds"

    output=$(python3 node_classification1.py --dataset=$1 --batch_size=$batch_size --mode=puregpu --fan_out=$fanout,$fanout,$fanout --epoch=$2)
    #filename="SPMM_time_surendra/$1_F${fanout}_B${batch_size}_puregpu_$2.txt"
    filename="time_surendra/$1_F${fanout}_B${batch_size}_puregpu_$2.txt"
    #filename="epoch1_time_surendra/$1_F${fanout}_B${batch_size}_puregpu_$2.txt"
    echo "Dataset = $1" > $filename
    #python3 node_classification.py --dataset=ogbn-products --batch_size=1024
    #Loop through the output lines to calculate sampling time
    while read -r line; do
      if [[ $line == "Testing..."* ]]; then
        # Stop adding CusparseCsrmm2 timing when "Testing..." line occurs
        add_cusparse_time=false	    
      # Check if the line contains the string "cuda,sapmling"
      elif [[ $line == cuda* ]]; then
        # Extract the time value and add it to the sampling time
        #echo $line
        time_value=$(echo $line | awk '{print $4}')
        #echo $time_value
        sampling_time=$(echo "$sampling_time + $time_value" | bc -l)
      # Check if the line contains the string "neighbor"
      elif [[ $line == neighbor* ]]; then
	#counter=$(expr $counter+1)
        #counter=$(echo "$counter + 1.0" | bc -l)	
        # Extract the time value and add it to the sampling time
        #echo $line
        time_value=$(echo $line | awk '{print $4}')
        #echo $time_value
        neighbor_time=$(echo "$neighbor_time + $time_value" | bc -l)
      # Check if the line contains the string "Iteration"
      elif [[ $line == Iteration* ]]; then
        # Extract the time value and add it to the iteration time
        #echo $line
        time_value=$(echo $line | awk '{print $3}')
        #echo $time_value
        iteration_time=$(echo "$iteration_time + $time_value" | bc -l)
      # Check if the line contains the string "cuda,sapmling"
      elif [[ $line == model* ]]; then
        # Extract the time value and add it to the sampling time
        #echo $line
        time_value=$(echo $line | awk '{print $3}')
        #echo $time_value
        model_time=$(echo "$model_time + $time_value" | bc -l)
      # Check if the line contains the string "cuda,sapmling"
      elif [[ $line == array* ]]; then
        # Extract the time value and add it to the sampling time
        #echo $line
        time_value=$(echo $line | awk '{print $4}')
        #echo $time_value
        array_time=$(echo "$array_time + $time_value" | bc -l)
      # Check if the line contains the string "cuda,sapmling"
      elif [[ $line == induced* ]]; then
        # Extract the time value and add it to the sampling time
        #echo $line
        time_value=$(echo $line | awk '{print $4}')
        #echo $time_value
        induced_time=$(echo "$induced_time + $time_value" | bc -l)
      # if [[ $line == CusparseCsrmm2* ]]; then
      elif [[ $line == $start_string* && $line == *$end_string ]] && $add_cusparse_time; then
        # Extract the time value and add it to the sampling time
        #echo $line
        time_value1=$(echo $line | awk '{print $4}')
        #echo $time_value
        spmm_time=$(echo "$spmm_time + $time_value1" | bc -l)
      fi
    done <<< "$output"
    #echo $counter
    #counter=$(echo "$counter / 3.0" |bc -l)
    #if [[ $neighbor_time == .* ]]; then
	#    neighbor_time="0$neighbor_time"
    #fi
    #echo $neighbor_time    
    #neighbor_time_batch=$(echo "$neighbor_time / $counter" | bc -l)

    #Loop through the output lines to calculate degree kernel time
    #while read -r line; do
      # Check if the line contains the string "cuda,sapmling"
      #if [[ $line == degree* ]]; then
        # Extract the time value and add it to the sampling time
        #echo $line
        #time_value1=$(echo $line | awk '{print $4}')
        #echo $time_value
        #degree_kernel_time=$(echo "$degree_kernel_time + $time_value1" | bc -l)
      #fi
    #done <<< "$output"


    # Check if epoch_data.txt exists
    if [ -f "epoch_data.txt" ]; then
            cat "epoch_data.txt" >> $filename
            echo "Data copied successfully!"
    else
            echo "Error: epoch_data.txt does not exist."
    fi


    # Loop through the output lines
    #while read -r line; do
    # Check if the line contains the string "cuda,sapmling"
     # if [[ $line == Epoch* ]]; then
      #  t_time=$(echo $line | awk '{print $12}')
       # training_time=$(echo "$training_time + $t_time" | bc -l)
        #epoch_line=$line
        #echo $epoch_line >> $filename
     # fi
    #done <<< "$output"

   # while read -r line; do
    # Check if the line contains the string "cuda,sapmling"
    #  if [[ $line == Total* ]]; then
      # t_time=$(echo $line | awk '{print $12}')
      # training_time=$(echo "$training_time + $t_time" | bc -l)
      # echo $line
      tt_time=$line
   # fi
   # done <<< "$output"

    #while read -r line; do
    # Check if the line contains the string "cuda,sapmling"
     # if [[ $line == Test* ]]; then
      # t_time=$(echo $line | awk '{print $12}')
      # training_time=$(echo "$training_time + $t_time" | bc -l)
      #test=$line
   # fi
    #done <<< "$output"
    # Create filename
    # Print output to file
    #echo "$output" > "$filename"
    # Print total sampling time
    #print_total_sampling_time "$output" >> "$filename"
    #echo "Total sampling time : " $sampling_time  >> $filename
    #echo "For loop time with batch size= $batch_size and fanout= $fanout: $iteration_time"
    echo "Total sampling time with batch size= $batch_size and fanout= $fanout : $sampling_time"
    #echo "Total model time with batch size= $batch_size and fanout= $fanout:: $model_time"
    #echo "Total neighbor time with batch size= $batch_size and fanout= $fanout : $neighbor_time"
    #echo "Total array time with batch size= $batch_size and fanout= $fanout : $array_time" 
    #echo "Total induced subgraph time with batch size= $batch_size and fanout= $fanout : $induced_time"
    #echo "Total spmm time : " $spmm_time >> $filename
    #echo "Total neighbor_cc time : " $neighbor_time_batch  >> $filename 
    #echo "Total degree kernel time : " $degree_kernel_time  >> $filename
    #echo "Total sampling time : " $sampling_time/$epoch ", Total training time :" $training_time/$epoch >> $filename
    #echo $test >> $filename
    #echo $tt_time >> $filename
  done
done

