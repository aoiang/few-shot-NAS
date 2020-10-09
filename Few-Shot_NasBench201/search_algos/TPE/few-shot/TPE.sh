#!/bin/bash
# Regularized Evolution for Image Classifier Architecture Search, AAAI 2019
# bash ./scripts-search/algos/R-EA.sh cifar10 -1

echo script name: $0
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 5 parameters for dataset, and seed"
  exit 1
fi



dataset=$1
seed=$2
channel=16
num_cells=5
max_nodes=4
space=nas-bench-201


benchmark_file=./network_info/nasbench201
supernet_file=./network_info/few-shot-supernet

save_dir=./output/few-shot-NAS/TPE-${dataset}

for (( c=1; c<=50; c++ ))
do
    OMP_NUM_THREADS=4 python ./exps/algos/TPE.py \
        --save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
        --dataset ${dataset} \
        --search_space_name ${space} \
        --arch_nas_dataset ${benchmark_file} \
        --arch_supernet_dataset ${supernet_file} \
        --rand_seed ${seed}
done



