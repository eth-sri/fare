#!/bin/bash
function job(){
    echo "running SIPM for Health Transfer with lmda=$1 lmbdaR=$2 lmdaF=$3"
    mkdir -p result/health_transfer/sipm/lmda=$1,lmdaR=$2,lmdaF="$3"
    python3 -m src.sipm.main --resultdir result/ --dataset "health_transfer" --head_net 1smooth --epochs 300 --finetune_epochs 100 --lmda $1 --lmdaR $2 --lmdaF $3 2>&1 | tee result/health_transfer/sipm/lmda="$1",lmdaR="$2",lmdaF="$3"/log.txt
}
export -f job

# jobs
parallel -j5 --ungroup job {1} {2} {3} ::: 0.0001 0.001 0.01 0.1 1.0 10.0 ::: 0.0 1.0 ::: 10.0 50.0 100.0
parallel -j5 --ungroup job {1} {2} {3} ::: 0.01 ::: 0.0 1.0 ::: 0.1 1.0
parallel -j5 --ungroup job {1} {2} {3} ::: 1.0 ::: 0.0 1.0 ::: 0.0001 0.001 0.01 0.1 1.0