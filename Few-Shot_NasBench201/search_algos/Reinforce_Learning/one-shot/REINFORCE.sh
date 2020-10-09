#!/bin/bash
# bash ./scripts-search/algos/REINFORCE.sh -1
echo script name: $0
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for dataset, and seed"
  exit 1
fi


dataset=$1
seed=$2
channel=16
num_cells=5
max_nodes=4
space=nas-bench-201


benchmark_file=./network_info/nasbench201
supernet_file=./network_info/one-shot-supernet
save_dir=./output/one-shot-NAS/REINFORCE-${dataset}


OMP_NUM_THREADS=4 python ./exps/algos/reinforce.py \
	--save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
	--dataset ${dataset} \
	--search_space_name ${space} \
	--arch_nas_dataset ${benchmark_file} \
	--arch_supernet_dataset ${supernet_file} \
	--time_budget 12000 \
	--learning_rate 0.005 --EMA_momentum 0.9 \
	--RL_steps 15000 \
	--workers 4 --print_freq 200 --rand_seed ${seed}
