#!/bin/bash
function job(){
    echo "running for health_notransfer with coeff $1"
    python3 -m src.laftr.src.run_laftr src/laftr/conf/transfer/laftr_then_naive.json \
        -o exp_name="health_notransfer/laftr/laftr_g_"$1".run_1",train.n_epochs=500,model.fair_coeff="$1",model.zdim=$2  \
        --data health_notransfer --dirs local 2>&1 | tee logs/health_notransfer/laftr_g_"$1".log
}

export -f job

# prep data: run this from pytorch env before running the rest
#   python3 -m src.laftr.src.EXPORT_DATA health_notransfer datakey
# make sure the same key is in laftr_then_naive.json

# jobs
parallel -j3 job {1} {2} ::: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.5 2.0 2.5 3.0 3.5 4.0 10.0 50.0 ::: 8 32