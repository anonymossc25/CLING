#!/bin/bash

# Input and output files
INPUT_LOG=$1  # Change if your log filename is different
OUTPUT_CSV=$2

# Write CSV header
echo "num_nodes,num_edges" > "$OUTPUT_CSV"

# Extract lines containing 'Graph(num_nodes=' and parse with awk
grep "Graph(num_nodes=" "$INPUT_LOG" | \
awk -F'[=,)]' '{print $2","$4}' >> "$OUTPUT_CSV"

echo "Saved subgraph stats to $OUTPUT_CSV"

