#!/bin/bash

function job(){
    echo "running SIPM for ACS-CA with lmda=$1 lmdaF=$2"
    mkdir -p result/ACSIncome-CA-2014/sipm/lmda=$1,lmdaR=0.0,lmdaF="$2"
    python3 -m src.sipm.main --resultdir result/ --dataset "ACSIncome-CA-2014" --head_net 1smooth --lmda $1 --lmdaR 0.0 --lmdaF $2 2>&1 | tee result/ACSIncome-CA-2014/sipm/lmda="$1",lmdaR=0.0,lmdaF="$2"/log.txt 
}
export -f job

# jobs
parallel -j5 --ungroup job {1} {2} ::: 0.0001 0.001 0.01 0.1 1.0 ::: 0.0001 0.001 0.01 0.1 1.0 10.0 50.0 100.0
