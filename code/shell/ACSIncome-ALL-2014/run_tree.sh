function job(){
    echo "running TREE for ACSIncome-ALL-2014 with max-k=$1, min-ni=$2, alpha=$3, split=$4"
    mkdir -p result/ACSIncome-ALL-2014/tree/k="$1",ni="$2",a="$3",s="$4"
    python3 -m src.tree.main --resultdir result/ --dataset "ACSIncome-ALL-2014" --max-k $1 --min-ni $2 --alpha $3 --val-split $4 2>&1 | tee result/ACSIncome-ALL-2014/tree/k="$1",ni="$2",a="$3",s="$4"/log.txt 
}
export -f job

# runs
parallel -j10 --ungroup job {1} {2} {3} {4} ::: 2 4 6 8 10 20 50 100 200 ::: 1000 ::: 0.3 0.5 0.7 0.9 0.99 0.999 ::: 0.2