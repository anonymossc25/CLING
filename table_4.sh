#!/bin/bash

echo ""
echo "-------------------------------------------------------------------------------------------------------------------------------------"
printf "| %-12s | %-33s | %-33s | %-10s |\n" "Dataset" "GraphSAGE (Sampling(kernel) | Model Training(SPMM) | Total)" "CLING (Sampling(kernel) | Model Training(SPMM) | Total)" "Speedup"
echo "-------------------------------------------------------------------------------------------------------------------------------------"

# Declare datasets
datasets=("reddit" "ogbn-products" "ogbn-arxiv")

for dataset in "${datasets[@]}"; do
    # Files
    gs_file="logs/graphsage/${dataset}_F20_B1024_E100.txt"
    cling_file="logs/cling/${dataset}_F20_B1024_E100.txt"

    # Function to extract values from a file
    extract_values() {
        local file="$1"
        summary_line=$(grep "Sampling time:" "$file")
        spmm_kernel_line=$(grep "spmm time:" "$file")

        sampling_time=$(echo "$summary_line" | awk -F'[:,]' '{print $2}' | xargs)
        model_time=$(echo "$summary_line" | awk -F'[:,]' '{print $4}' | xargs)
        total_time=$(echo "$summary_line" | awk '{print $10}' | xargs)
        #total_time=$(echo "$summary_line" | awk -F'[:,]' '{print $6}' | xargs)

        kernel_time=$(echo "$spmm_kernel_line" | grep -oP 'sampling time \(kernel\):\s+\K[0-9.]+')
        spmm_time=$(echo "$spmm_kernel_line" | grep -oP 'spmm time:\s+\K[0-9.]+')

        echo "$sampling_time $kernel_time $model_time $spmm_time $total_time"
    }

    # Get values for GraphSAGE and CLING
    read gs_samp gs_kern gs_model gs_spmm gs_total <<< $(extract_values "$gs_file")
    read cl_samp cl_kern cl_model cl_spmm cl_total <<< $(extract_values "$cling_file")

    # Compute speedup
    speedup=$(echo "scale=2; $gs_total / $cl_total" | bc)

    # Print formatted row
    printf "| %-12s | %6.2f (%5.2f) | %6.2f (%5.2f) | %6.2f | %6.2f (%5.2f) | %6.2f (%5.2f) | %6.2f |  %5.2fx |\n" \
        "$dataset" \
        "$gs_samp" "$gs_kern" "$gs_model" "$gs_spmm" "$gs_total" \
        "$cl_samp" "$cl_kern" "$cl_model" "$cl_spmm" "$cl_total" \
        "$speedup"
done

echo "-------------------------------------------------------------------------------------------------------------------------------------"

