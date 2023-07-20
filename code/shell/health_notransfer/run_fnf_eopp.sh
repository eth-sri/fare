function job(){
    echo "running FNF for health notransfer with gamma=$1"
    mkdir -p result/health_notransfer/fnf-eopp/gamma="$1"
    python3 -m src.fnf.run_health_gammas --metric eq_opp --resultdir result/ --device "cuda" --dataset "health_notransfer" --with_test --gamma $1 2>&1 | tee result/health_notransfer/fnf-eopp/gamma="$1"/log.txt 
}
export -f job
 
parallel -j1 --ungroup job {1} {#} ::: 0.0 0.05 0.1 0.5 0.95