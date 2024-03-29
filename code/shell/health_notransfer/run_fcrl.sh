#!/bin/bash

function job() {
  lambda=$(printf "%1.2f" "$1")
  beta=$(awk '{print $1*$2}' <<<"${lambda} ${2}")
  devicenum=$(expr $3 % 2)
  device=cuda:$devicenum

  string=$(echo output_"$DATANAME"_fcrl_l="$lambda"_b="$beta")
  log_file=$(python3 src/shell/get_log_filename.py -f "$LOGS_FOLDER" -s "$string")
  result_folder=$(echo "$FOLDER"/l="$lambda"_b="$beta")

  echo "[health] Running FCRL for beta=$beta lambda=$lambda"
  echo -e "\t on device $device"
  echo -e "\t for data $DATANAME and"
  echo -e "\t storing logs in $log_file, result in $result_folder"

  python3 -m src.scripts.main -c config/config_fcrl.py \
    --exp_name "$FOLDER"/l="$lambda"_b="$beta" \
    --result_folder "$result_folder" --device "$device" --data.name "$DATANAME" \
    --model.arch_file src/arch/health/health_fcrl.py \
    --model.lambda_ "$lambda" --model.beta "$beta" \
    --train.max_epoch 200
}

export DATANAME="health_notransfer"
export LOGS_FOLDER=result/logs/
export FOLDER=result/$DATANAME/fcrl
mkdir -p $LOGS_FOLDER
mkdir -p $FOLDER
export DEVICE="cuda"

export -f job

# job
parallel -j6 --ungroup job {1} {2} :::  1e-2 2e-2 3e-2 4e-2 5e-2 6e-2 7e-2 8e-2 9e-2 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1 1e0 11e-1 12e-1 13e-1 14e-1 15e-1 16e-1 17e-1 18e-1 19e-1 2e0  ::: 0.5
