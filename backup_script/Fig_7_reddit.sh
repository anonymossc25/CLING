#!/bin/bash

# Output file for plotting
OUTPUT_FILE="reddit_fanout.txt"
echo -e "# Fanout\tGraphSAGE\tOur approach" > "$OUTPUT_FILE"

# Fanouts to evaluate
fanouts=(10 15 20)

for fanout in "${fanouts[@]}"; do
    gs_file="logs/graphsage/reddit_F${fanout}_B1024_E100.txt"
    cling_file="logs/cling/reddit_F${fanout}_B1024_E100.txt"
    #summary_line=$(grep "Sampling time:" "$")
    # Extract total time
    gs_time=$(grep "Sampling time" "$gs_file" | awk '{print $10}' | xargs)
    cling_time=$(grep "Sampling time" "$cling_file" | awk '{print $10}' | xargs)

    # Write to file
    printf "%d\t%.4f\t%.4f\n" "$fanout" "$gs_time" "$cling_time" >> "$OUTPUT_FILE"
done

echo "Generated $OUTPUT_FILE"

gnuplot -e "datafile='reddit_fanout.txt'" fanout.p
evince fanout.eps

