#!/bin/bash

# Output data file
#OUTPUT_FILE="products_batch_F15.txt"
#echo -e "#Batch size\tGraphSAGE\tOur approach" > "$OUTPUT_FILE"

# Set dataset and fanout
dataset=$1
fanout=$2
epochs=100

OUTPUT_FILE="${dataset}_batch_F${fanout}.txt"
echo -e "#Batch size\tGraphSAGE\tOur approach" > "$OUTPUT_FILE"
# List of batch sizes
batches=(1024 2048 4096 8192 16384 32768 65536)

for batch in "${batches[@]}"; do
    gs_file="logs/graphsage/${dataset}_F${fanout}_B${batch}_E100.txt"
    cling_file="logs/cling/${dataset}_F${fanout}_B${batch}_E100.txt"

    if [[ -f "$gs_file" && -f "$cling_file" ]]; then
        gs_time=$(grep "Total time" "$gs_file" | awk '{print $10}' | xargs)
        cling_time=$(grep "Total time" "$cling_file" | awk '{print $10}' | xargs)

        printf "%d\t%.4f\t%.4f\n" "$batch" "$gs_time" "$cling_time" >> "$OUTPUT_FILE"
    else
        echo "Missing file for batch $batch — skipping"
    fi
done

echo "Generated $OUTPUT_FILE"

outfile="${dataset}_batch_F${fanout}.eps"

gnuplot -e "datafile='$OUTPUT_FILE'; outfile='$outfile'" batch_plot.p
evince $outfile

