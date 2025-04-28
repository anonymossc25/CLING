#!/bin/bash

start=$(date +%s)

#bash run_CLING.sh ogbn-arxiv 100
bash run_layer.sh reddit 100
bash run_layer.sh ogbn-products 100

end=$(date +%s)
duration=$((end - start))

echo "Total execution time: $duration seconds"

