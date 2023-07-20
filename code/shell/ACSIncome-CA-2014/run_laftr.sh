#!/bin/bash
function job(){
    echo "running for ACSIncome-CA-2014 with coeff $1"
    python3 -m src.laftr.src.run_laftr src/laftr/conf/transfer/laftr_then_naive.json \
        -o exp_name="ACSIncome-CA-2014/laftr/laftr_g_"$1".run_1",train.n_epochs=500,model.fair_coeff="$1"  \
        --data ACSIncome-CA-2014 --dirs local 2>&1 | tee logs/ACSIncome-CA-2014/laftr_g_"$1".log
}

export -f job

# prep data: run this from pytorch env before running the rest
#   python3 -m src.laftr.src.EXPORT_DATA ACSIncome-CA-2014 datakey
# make sure the same key is in laftr_then_naive.json

# jobs
parallel -j3 job {1} ::: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.5 2.0 2.5 3.0 3.5 4.0 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 10.0 18 20 50 100 500 1000 10000