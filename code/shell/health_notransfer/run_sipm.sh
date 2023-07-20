#!/bin/bash
function job(){
    echo "running SIPM for Health notransfer with lmda=$1 lmdaF=$2"
    mkdir -p result/health_notransfer/sipm/lmda=$1,lmdaR=0.0,lmdaF="$2"
    python3 -m src.sipm.main --resultdir result/ --dataset "health_notransfer" --head_net 1smooth --epochs 300 --finetune_epochs 100 --lmda $1 --lmdaR 0.0 --lmdaF $2 2>&1 | tee result/health_notransfer/sipm/lmda="$1",lmdaR=0.0,lmdaF="$2"/log.txt 
}
export -f job

# jobs
parallel -j5 --ungroup job {1} {2} ::: 0.0001 0.001 0.01 0.1 1.0 ::: 0.1 1.0 10.0 50.0
