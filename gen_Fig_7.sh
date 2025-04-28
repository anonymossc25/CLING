#!/bin/bash

fanouts=(10 15 20)
dataset=$1
# Output file for plotting
OUTPUT_FILE="${dataset}_fanout.txt"
echo -e "# Fanout\tGraphSAGE\tOur approach" > "$OUTPUT_FILE"

# Fanouts to evaluate
#fanouts=(10 15 20)

for fanout in "${fanouts[@]}"; do
    #gs_file="logs/graphsage/ogbn-products_F${fanout}_B1024_E100.txt"
    #cling_file="logs/cling/ogbn-products_F${fanout}_B1024_E100.txt"
    gs_file="logs/graphsage/${dataset}_F${fanout}_B8192_E100.txt"
    cling_file="logs/cling/${dataset}_F${fanout}_B8192_E100.txt"

    #summary_line=$(grep "Sampling time:" "$")
    # Extract total time
    gs_time=$(grep "Sampling time" "$gs_file" | awk '{print $10}' | xargs)
    cling_time=$(grep "Sampling time" "$cling_file" | awk '{print $10}' | xargs)

    # Write to file
    printf "%d\t%.4f\t%.4f\n" "$fanout" "$gs_time" "$cling_time" >> "$OUTPUT_FILE"
done

echo "Generated $OUTPUT_FILE"

gnuplot -e "datafile='${dataset}_fanout.txt'; outfile='${dataset}_fanout.eps" fanout.p
evince "${dataset}_fanout.eps"
