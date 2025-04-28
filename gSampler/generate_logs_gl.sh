#!/bin/bash

start=$(date +%s)

bash run_gsampler.sh ogbn-arxiv 100
bash run_gsampler.sh reddit 100
bash run_gsampler.sh ogbn-products 100
bash run_ladies.sh ogbn-arxiv 100
bash run_ladies.sh reddit 100
bash run_ladies.sh ogbn-products 100


end=$(date +%s)
duration=$((end - start))

echo "Total execution time: $duration seconds"

