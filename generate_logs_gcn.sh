#!/bin/bash

start=$(date +%s)

bash run_gcn.sh ogbn-arxiv 100
bash run_gcn.sh reddit 100
bash run_gcn.sh ogbn-products 100

end=$(date +%s)
duration=$((end - start))

echo "Total execution time: $duration seconds"

