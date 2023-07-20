#!/bin/bash

function job(){
    echo "running FAIR-PATH for health_notransfer with kappa=$1"
    mkdir -p result/health_notransfer/fair-path/kappa="$1"
    python3 -m src.fair-path.main --resultdir result/ --n_epoch 2000 --dataset "health_notransfer" --kappa $1 2>&1 | tee result/health_notransfer/fair-path/kappa="$1"/log.txt 
}
export -f job

# jobs
parallel -j1 --ungroup job {1} ::: 0.0 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0 5.0 10.0 50.0 100.0
