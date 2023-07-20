
#!/bin/bash

function job(){
    mkdir -p result/ACSIncome-CA-2014/kmeans/k="$1"
    python3 -m src.kmeans.main --dataset ACSIncome-CA-2014 --k $1 2>&1
}
export -f job
 
# jobs
parallel -j1 --ungroup job {1} ::: 200 500 1000 10000