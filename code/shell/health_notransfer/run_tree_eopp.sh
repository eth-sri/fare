function job(){
    echo "running TREE for health_notransfer with max-k=$1, min-ni=$2, alpha=$3, split=$4"
    mkdir -p result/health_notransfer/tree-eopp/k="$1",ni="$2",a="$3",s="$4"
    python3 -m src.tree.main --eval-metric eopp --gini-metric eopp --resultdir result/ --dataset "health_notransfer" --max-k $1 --min-ni $2 --alpha $3 --val-split $4 2>&1 | tee result/health_notransfer/tree-eopp/k="$1",ni="$2",a="$3",s="$4"/log.txt 
}
export -f job

# run
parallel -j1 --ungroup job {1} {2} {3} {4} ::: 2 5 20 40 60 80 100 120 ::: 60 ::: 0.3 0.5 0.7 0.8 0.85 ::: 0.2 0.5