#!/bin/bash

function job(){
    echo "running FNF for health transfer with gamma=$1"
    mkdir -p result/health_transfer/fnf/gamma="$1"
    python3 -m src.fnf.main --resultdir result/ --device "cuda" --dataset "health_transfer" --with_test --encode --gamma $1 2>&1 | tee result/health_transfer/fnf/gamma="$1"/log.txt 
}
export -f job
 
# jobs
parallel -j5 --ungroup job {1} ::: 0.0 0.01 0.02 0.04 0.06 0.08 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0