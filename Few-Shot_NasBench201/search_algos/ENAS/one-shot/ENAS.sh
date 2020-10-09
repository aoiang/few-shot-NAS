#!/bin/bash
# Efficient Neural Architecture Search via Parameter Sharing, ICML 2018

echo script name: $0
echo $# arguments
if [ "$#" -ne 5 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 5 parameters for dataset, tracking_status, seed, ith run, and path of benchmark_file"
  exit 1
fi

dataset=$1
BN=$2
seed=$3
run=$4
TORCH_HOME=$5

if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

channel=16
num_cells=5
max_nodes=4
space=nas-bench-201

if [ "$dataset" == "cifar10" ] || [ "$dataset" == "cifar100" ]; then
  data_path="$TORCH_HOME/cifar.python"
else
  data_path="$TORCH_HOME/cifar.python/ImageNet16"
fi
#benchmark_file=${TORCH_HOME}/NAS-Bench-201-v1_0-e61699.pth
benchmark_file=${TORCH_HOME}/NAS-Bench-201-v1_1-096897.pth

save_dir=./output/one-shot-NAS/ENAS-${dataset}-BN${BN}/runs-${run}

OMP_NUM_THREADS=4 python ./exps/algos/ENAS.py \
	--save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
	--dataset ${dataset} --data_path ${data_path} \
	--search_space_name ${space} \
	--arch_nas_dataset ${benchmark_file} \
	--track_running_stats ${BN} \
	--config_path ./configs/nas-benchmark/algos/ENAS.config \
	--controller_entropy_weight 0.0001 \
	--controller_bl_dec 0.99 \
	--controller_train_steps 50 \
	--controller_num_aggregate 20 \
	--controller_num_samples 100 \
	--workers 4 --print_freq 200 --rand_seed ${seed}
