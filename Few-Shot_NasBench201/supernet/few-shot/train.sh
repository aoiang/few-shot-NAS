#!/bin/bash
echo script name: $0
echo $# arguments
if [ "$#" -ne 5 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 5 parameters for dataset, BN-tracking-status, path of benchmark_file, one-shot supernet model checkpoint, and operation index"
  exit 1
fi

dataset=$1
BN=$2
TORCH_HOME=$3
ONE_SHOT_SUPERNET=$4
OP=$5

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


benchmark_file=${TORCH_HOME}/NAS-Bench-201-v1_1-096897.pth

save_dir=./supernet_checkpoint/few-shot

OMP_NUM_THREADS=4 python ./exps/supernet/few-shot-supernet_train.py \
	--save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
	--dataset ${dataset} --data_path ${data_path} \
	--search_space_name ${space} \
	--arch_nas_dataset ${benchmark_file} \
	--weight_tranferring_path  ./supernet_checkpoint/one-shot \
	--weight_tranferring_model ${ONE_SHOT_SUPERNET} \
	--config_path configs/nas-benchmark/algos/FEW-SHOT-SUPERNET.config \
	--track_running_stats ${BN} \
	--select_num 100 \
	--workers 4 --print_freq 200 --rand_seed 0 --edge_op ${OP}





