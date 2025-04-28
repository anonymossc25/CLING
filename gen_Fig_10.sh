#!/bin/bash

# Dataset names (lowercase, as in file names)
datasets=("ogbn-arxiv" "reddit" "ogbn-products")
fanout=$1
#batch=1024
#epochs=100

# Output file
OUTPUT_FILE="gcn_F${fanout}.txt"
echo -e "#Dataset\tGraphSAGE\tCLING" > "$OUTPUT_FILE"


for dataset in "${datasets[@]}"; do
    gs_file="logs/gcn/graphsage/${dataset}_F${fanout}_B1024_E100.txt"
    cling_file="logs/gcn/cling/${dataset}_F${fanout}_B1024_E100.txt"

    if [[ -f "$gs_file" && -f "$cling_file" ]]; then
        gs_time=$(grep "Total time" "$gs_file" | awk '{print $10}' | xargs)
        cling_time=$(grep "Total time" "$cling_file" | awk '{print $10}' | xargs)

        # Capitalize dataset name for output (optional)
        #cap_dataset=$(echo "$dataset" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')
        printf "%s\t%.4f\t%.4f\n" "$dataset" "$gs_time" "$cling_time" >> "$OUTPUT_FILE"
    else
        echo "Missing file(s) for $dataset — skipping"
    fi
done

echo "Generated $OUTPUT_FILE"

gnuplot -e "datafile='gcn_F${fanout}.txt'; outfile='gcn_F${fanout}.eps'" gcn.p

evince "gcn_F${fanout}.eps"
