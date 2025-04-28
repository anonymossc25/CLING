#!/bin/bash


# Base path where the layer folders exist
base_path="logs/layer"


dataset=$1
# Output file
OUTPUT_FILE="${dataset}_fig_9.txt"
echo -e "#Layers\tGraphSAGE\tCLING" > "$OUTPUT_FILE"

# Loop over layer directories (e.g. three-layer, four-layer, five-layer)
for layer_dir in "$base_path"/*-layer; do
    layer_name=$(basename "$layer_dir")

    #gs_file="$layer_dir/graphsage/ogbn-products_F20_B1024_E100.txt"
    #cling_file="$layer_dir/cling/ogbn-products_F20_B1024_E100.txt"
    gs_file="$layer_dir/graphsage/${dataset}_F20_B1024_E100.txt"
    cling_file="$layer_dir/cling/${dataset}_F20_B1024_E100.txt"


    if [[ -f "$gs_file" && -f "$cling_file" ]]; then
        gs_time=$(grep "Total time" "$gs_file" | awk '{print $10}' | xargs)
        cling_time=$(grep "Total time" "$cling_file" | awk '{print $10}' | xargs)

        printf "%s\t%.4f\t%.4f\n" "$layer_name" "$gs_time" "$cling_time" >> "$OUTPUT_FILE"
    else
        echo "Missing file(s) for $layer_name — skipping"
    fi
done

echo "Generated $OUTPUT_FILE"

gnuplot -e "datafile='$OUTPUT_FILE'; outfile='${dataset}_fig_9.eps'" gen_Fig_9.p
evince "${dataset}_fig_9.eps"
