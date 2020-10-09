#!/bin/bash
# Regularized Evolution for Image Classifier Architecture Search, AAAI 2019
# bash ./scripts-search/algos/R-EA.sh cifar10 20 -1

echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 5 parameters for dataset, sample_size, seed"
  exit 1
fi



dataset=$1
sample_size=$2
seed=$3
channel=16
num_cells=5
max_nodes=4
space=nas-bench-201


benchmark_file=./network_info/nasbench201
supernet_file=./network_info/one-shot-supernet

save_dir=./output/one-shot-NAS/R-EA-${dataset}-SS${sample_size}




OMP_NUM_THREADS=4 python ./exps/algos/R_EA.py \
	--save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
	--dataset ${dataset} \
	--search_space_name ${space} \
	--arch_nas_dataset ${benchmark_file} \
	--arch_supernet_dataset ${supernet_file} \
	--time_budget 12000 \
	--ea_cycles 15625 --ea_population 10 --ea_sample_size ${sample_size} --ea_fast_by_api 1 \
	--workers 4 --print_freq 200 --rand_seed ${seed}



