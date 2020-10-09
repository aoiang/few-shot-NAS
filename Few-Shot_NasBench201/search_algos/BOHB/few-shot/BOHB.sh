#!/bin/bash
# bash ./scripts-search/algos/BOHB.sh cifar10 -1 nasbench201
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameters for dataset and seed, and path of benchmark_file"
  exit 1
fi


dataset=$1
seed=$2
nasbench_201=$3
channel=16
num_cells=5
max_nodes=4
space=nas-bench-201

benchmark_file=./network_info/nasbench201
supernet_file=./network_info/few-shot-supernet

save_dir=./output/few-shot-NAS/BOHB-${dataset}

OMP_NUM_THREADS=4 python ./exps/algos/BOHB.py \
	--save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
	--dataset ${dataset} \
	--search_space_name ${space} \
	--arch_nas_dataset ${benchmark_file} \
	--arch_supernet_dataset ${supernet_file} \
	--arch_nasbench201 ${nasbench_201} \
	--time_budget 12000  \
	--n_iters 50 --num_samples 4 --random_fraction 0.0 --bandwidth_factor 3 \
	--workers 4 --print_freq 200 --rand_seed ${seed}

