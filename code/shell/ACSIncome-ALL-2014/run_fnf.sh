#!/bin/bash

function job(){
    echo "running FNF for ACS-ALL with gamma=$1"
    mkdir -p result/ACSIncome-ALL-2014/fnf/gamma="$1"
    python3 -m src.fnf.main --resultdir result/ --device "cuda" --dataset "ACSIncome-ALL-2014" --with_test --encode --gamma $1 2>&1 | tee result/ACSIncome-ALL-2014/fnf/gamma="$1"/log.txt 
}
export -f job
 
# jobs
parallel -j5 --ungroup job {1} ::: 0.0 0.01 0.02 0.04 0.06 0.08 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0