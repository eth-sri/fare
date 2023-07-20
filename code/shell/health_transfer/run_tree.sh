function job(){
    echo "running TREE for health_transfer_all with max-k=$1, min-ni=$2, alpha=$3, split=$4"
    mkdir -p result/health_transfer_all/tree/k="$1",ni="$2",a="$3",s="$4"
    python3 -m src.tree.main --resultdir result/ --dataset "health_transfer_all" --max-k $1 --min-ni $2 --alpha $3 --val-split $4 2>&1 | tee result/health_transfer_all/tree/k="$1",ni="$2",a="$3",s="$4"/log.txt 
}
export -f job

# run
parallel -j10 --ungroup job {1} {2} {3} {4} ::: 2 4 6 8 10 20 50 100 ::: 500 ::: 0.1 0.2 0.3 0.4 0.5 0.6 0.9 0.99 0.999 ::: 0.1 0.2 0.5
# TODO fix other methods to also use the "_all" version