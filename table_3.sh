#!/bin/bash

GRAPH_DIR="logs/graphsage"
CLING_DIR="logs/cling"

DATASETS=("reddit" "ogbn-products" "ogbn-arxiv")
LAYERS=("Layer_1" "Layer_2" "Layer_3")

printf "%-15s" "Dataset"
for layer in "${LAYERS[@]}"; do
    printf "| %-10s | %-18s " "GraphSAGE" "CLING (% reduction)"
done
echo
echo "--------------------------------------------------------------------------------------------------------------"

for dataset in "${DATASETS[@]}"; do
    graph_file="${GRAPH_DIR}/${dataset}_F20_B1024_E100.txt"
    cling_file="${CLING_DIR}/${dataset}_F20_B1024_E100.txt"

    # Extract neighbor counts from line 2 of each file
    graph_line=$(sed -n '2p' "$graph_file")
    cling_line=$(sed -n '2p' "$cling_file")

    printf "%-15s" "$dataset"

    for layer in "${LAYERS[@]}"; do
        # Get the count for the current layer
        g_count=$(echo "$graph_line" | grep -oP "${layer} \K[0-9]+")
        c_count=$(echo "$cling_line" | grep -oP "${layer} \K[0-9]+")

        # Calculate reduction %
        reduction=$(( 100 * (g_count - c_count) / g_count ))

        # Print GraphSAGE and CLING counts
        printf "| %-10s | %-18s " "$g_count" "$c_count ($reduction)"
    done

    echo
done

echo
echo "Table: Reduction of number of neighbors in each layer on different datasets"

