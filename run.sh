source  /private/home/linnanwang/virtual_envs/nas/bin/activate



#!/usr/bin/env bash



screen -d -m -S nasbench201_gdas_0 -L -Logfile ./logs/nasbench201_gdas_0.log srun -p dev --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/GDAS_0.sh cifar10 1 -1
screen -d -m -S nasbench201_gdas_1 -L -Logfile ./logs/nasbench201_gdas_1.log srun -p dev --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/GDAS_1.sh cifar10 1 -1
screen -d -m -S nasbench201_gdas_2 -L -Logfile ./logs/nasbench201_gdas_2.log srun -p dev --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/GDAS_2.sh cifar10 1 -1
screen -d -m -S nasbench201_gdas_3 -L -Logfile ./logs/nasbench201_gdas_3.log srun -p dev --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/GDAS_3.sh cifar10 1 -1
screen -d -m -S nasbench201_gdas_4 -L -Logfile ./logs/nasbench201_gdas_4.log srun -p dev --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/GDAS_4.sh cifar10 1 -1



screen -d -m -S nasbench201_darts_0 -L -Logfile ./logs/nasbench201_darts_0.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/DARTS-V2_0.sh cifar10 1 -1
screen -d -m -S nasbench201_darts_1 -L -Logfile ./logs/nasbench201_darts_1.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/DARTS-V2_1.sh cifar10 1 -1
screen -d -m -S nasbench201_darts_2 -L -Logfile ./logs/nasbench201_darts_2.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/DARTS-V2_2.sh cifar10 1 -1
screen -d -m -S nasbench201_darts_3 -L -Logfile ./logs/nasbench201_darts_3.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/DARTS-V2_3.sh cifar10 1 -1
screen -d -m -S nasbench201_darts_4 -L -Logfile ./logs/nasbench201_darts_4.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/DARTS-V2_4.sh cifar10 1 -1


screen -d -m -S nasbench201_ENAS_0 -L -Logfile ./logs/nasbench201_ENAS_0.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/ENAS_0.sh cifar10 1 -1
screen -d -m -S nasbench201_ENAS_1 -L -Logfile ./logs/nasbench201_ENAS_1.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/ENAS_1.sh cifar10 1 -1
screen -d -m -S nasbench201_ENAS_2 -L -Logfile ./logs/nasbench201_ENAS_2.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/ENAS_2.sh cifar10 1 -1
screen -d -m -S nasbench201_ENAS_3 -L -Logfile ./logs/nasbench201_ENAS_3.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/ENAS_3.sh cifar10 1 -1
screen -d -m -S nasbench201_ENAS_4 -L -Logfile ./logs/nasbench201_ENAS_4.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/ENAS_4.sh cifar10 1 -1



screen -d -m -S nasbench201_PCDARTS_0 -L -Logfile ./logs/nasbench201_PCDARTS_0.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/PCDARTS_0.sh cifar10 1 -1
screen -d -m -S nasbench201_PCDARTS_1 -L -Logfile ./logs/nasbench201_PCDARTS_1.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/PCDARTS_1.sh cifar10 1 -1
screen -d -m -S nasbench201_PCDARTS_2 -L -Logfile ./logs/nasbench201_PCDARTS_2.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/PCDARTS_2.sh cifar10 1 -1
screen -d -m -S nasbench201_PCDARTS_3 -L -Logfile ./logs/nasbench201_PCDARTS_3.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/PCDARTS_3.sh cifar10 1 -1
screen -d -m -S nasbench201_PCDARTS_4 -L -Logfile ./logs/nasbench201_PCDARTS_4.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/PCDARTS_4.sh cifar10 1 -1



screen -d -m -S nasbench201_SETN_0 -L -Logfile ./logs/nasbench201_SETN_0.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/SETN_0.sh cifar10 1 -1
screen -d -m -S nasbench201_SETN_1 -L -Logfile ./logs/nasbench201_SETN_1.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/SETN_1.sh cifar10 1 -1
screen -d -m -S nasbench201_SETN_2 -L -Logfile ./logs/nasbench201_SETN_2.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/SETN_2.sh cifar10 1 -1
screen -d -m -S nasbench201_SETN_3 -L -Logfile ./logs/nasbench201_SETN_3.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/SETN_3.sh cifar10 1 -1
screen -d -m -S nasbench201_SETN_4 -L -Logfile ./logs/nasbench201_SETN_4.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/SETN_4.sh cifar10 1 -1



screen -d -m -S nasbench201_RANK -L -Logfile ./logs/nasbench201_RANK.log srun -p dev --comment="nips" --gres=gpu:1 --time=24:00:00 --cpus-per-task=4 bash ./scripts-search/algos/RANK.sh cifar10 0 -1


screen -d -m -S nasbench201_RANK_0 -L -Logfile ./logs/nasbench201_RANK_0.log srun -p dev --comment="nips" --gres=gpu:1 --time=08:00:00 --cpus-per-task=4 bash ./scripts-search/algos/RANK_0.sh cifar10 0 -1
screen -d -m -S nasbench201_RANK_1 -L -Logfile ./logs/nasbench201_RANK_1.log srun -p dev --comment="nips" --gres=gpu:1 --time=08:00:00 --cpus-per-task=4 bash ./scripts-search/algos/RANK_1.sh cifar10 0 -1
screen -d -m -S nasbench201_RANK_2 -L -Logfile ./logs/nasbench201_RANK_2.log srun -p dev --comment="nips" --gres=gpu:1 --time=08:00:00 --cpus-per-task=4 bash ./scripts-search/algos/RANK_2.sh cifar10 0 -1
screen -d -m -S nasbench201_RANK_3 -L -Logfile ./logs/nasbench201_RANK_3.log srun -p dev --comment="nips" --gres=gpu:1 --time=08:00:00 --cpus-per-task=4 bash ./scripts-search/algos/RANK_3.sh cifar10 0 -1
screen -d -m -S nasbench201_RANK_4 -L -Logfile ./logs/nasbench201_RANK_4.log srun -p dev --comment="nips" --gres=gpu:1 --time=08:00:00 --cpus-per-task=4 bash ./scripts-search/algos/RANK_4.sh cifar10 0 -1



screen -d -m -S 1super_eval_rank_0  srun -p dev --comment="nips" --gres=gpu:1 --time=24:00:00 --cpus-per-task=4 bash ./scripts-search/algos/eval_rank_0.sh cifar10 0 -1
screen -d -m -S 1super_eval_rank_1  srun -p dev --comment="nips" --gres=gpu:1 --time=24:00:00 --cpus-per-task=4 bash ./scripts-search/algos/eval_rank_1.sh cifar10 0 -1
screen -d -m -S 1super_eval_rank_2  srun -p dev --comment="nips" --gres=gpu:1 --time=24:00:00 --cpus-per-task=4 bash ./scripts-search/algos/eval_rank_2.sh cifar10 0 -1
screen -d -m -S 1super_eval_rank_3  srun -p dev --comment="nips" --gres=gpu:1 --time=24:00:00 --cpus-per-task=4 bash ./scripts-search/algos/eval_rank_3.sh cifar10 0 -1
screen -d -m -S 1super_eval_rank_4  srun -p dev --comment="nips" --gres=gpu:1 --time=24:00:00 --cpus-per-task=4 bash ./scripts-search/algos/eval_rank_4.sh cifar10 0 -1



screen -d -m -S nasbench201_eval_rank -L -Logfile ./logs/nasbench201_eval_rank.log srun -p dev --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/eval_rank.sh cifar10 1 -1





screen -d -m -S nasbench201_RANK -L -Logfile ./logs/nasbench201_RANK.log srun -p dev --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/RANK.sh cifar10 1 -1
screen -d -m -S nasbench201_RANK_imagenet -L -Logfile ./logs/nasbench201_RANK_imagenet.log srun -p dev --comment="nips" --gres=gpu:1 --time=72:00:00 --cpus-per-task=4 bash ./scripts-search/algos/RANK.sh cifar100 1 -1


screen -d -m -S nasbench201_transfer_0 -L -Logfile ./logs/nasbench201_transfer_0.log srun -p dev --comment="nips" --gres=gpu:1 --time=08:00:00 --cpus-per-task=4 bash ./scripts-search/algos/transfer_RANK_0.sh cifar10 0 -1
screen -d -m -S nasbench201_transfer_1 -L -Logfile ./logs/nasbench201_transfer_1.log srun -p dev --comment="nips" --gres=gpu:1 --time=08:00:00 --cpus-per-task=4 bash ./scripts-search/algos/transfer_RANK_1.sh cifar10 0 -1
screen -d -m -S nasbench201_transfer_2 -L -Logfile ./logs/nasbench201_transfer_2.log srun -p dev --comment="nips" --gres=gpu:1 --time=08:00:00 --cpus-per-task=4 bash ./scripts-search/algos/transfer_RANK_2.sh cifar10 0 -1
screen -d -m -S nasbench201_transfer_3 -L -Logfile ./logs/nasbench201_transfer_3.log srun -p dev --comment="nips" --gres=gpu:1 --time=08:00:00 --cpus-per-task=4 bash ./scripts-search/algos/transfer_RANK_3.sh cifar10 0 -1
screen -d -m -S nasbench201_transfer_4 -L -Logfile ./logs/nasbench201_transfer_4.log srun -p dev --comment="nips" --gres=gpu:1 --time=08:00:00 --cpus-per-task=4 bash ./scripts-search/algos/transfer_RANK_4.sh cifar10 0 -1



screen -d -m -S nasbench201_eval_rank_0 -L -Logfile ./logs/nasbench201_eval_rank_0.log srun -p dev --comment="nips" --gres=gpu:1 --time=24:00:00 --cpus-per-task=4 bash ./scripts-search/algos/eval_rank_0.sh cifar10 0 -1
screen -d -m -S nasbench201_eval_rank_1 -L -Logfile ./logs/nasbench201_eval_rank_1.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=08:00:00 --cpus-per-task=4 bash ./scripts-search/algos/eval_rank_1.sh cifar10 0 -1
screen -d -m -S nasbench201_eval_rank_2 -L -Logfile ./logs/nasbench201_eval_rank_2.log srun -p dev --comment="nips" --gres=gpu:1 --time=24:00:00 --cpus-per-task=4 bash ./scripts-search/algos/eval_rank_2.sh cifar10 0 -1
screen -d -m -S nasbench201_eval_rank_3 -L -Logfile ./logs/nasbench201_eval_rank_3.log srun -p dev --comment="nips" --gres=gpu:1 --time=24:00:00 --cpus-per-task=4 bash ./scripts-search/algos/eval_rank_3.sh cifar10 0 -1
screen -d -m -S nasbench201_eval_rank_4 -L -Logfile ./logs/nasbench201_eval_rank_4.log srun -p dev --comment="nips" --gres=gpu:1 --time=24:00:00 --cpus-per-task=4 bash ./scripts-search/algos/eval_rank_4.sh cifar10 0 -1
screen -d -m -S nasbench201_eval_rank_5 -L -Logfile ./logs/nasbench201_eval_rank_5.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=08:00:00 --cpus-per-task=4 bash ./scripts-search/algos/eval_rank_5.sh cifar10 0 -1


screen -d -m -S nasbench201_eval_transfer_0 -L -Logfile ./logs/nasbench201_eval_transfer_0.log srun -p dev --comment="nips" --gres=gpu:1 --time=24:00:00 --cpus-per-task=4 bash ./scripts-search/algos/eval_transfer_0.sh cifar10 0 -1
screen -d -m -S nasbench201_eval_transfer_1 -L -Logfile ./logs/nasbench201_eval_transfer_1.log srun -p dev --comment="nips" --gres=gpu:1 --time=24:00:00 --cpus-per-task=4 bash ./scripts-search/algos/eval_transfer_1.sh cifar10 0 -1
screen -d -m -S nasbench201_eval_transfer_2 -L -Logfile ./logs/nasbench201_eval_transfer_2.log srun -p dev --comment="nips" --gres=gpu:1 --time=24:00:00 --cpus-per-task=4 bash ./scripts-search/algos/eval_transfer_2.sh cifar10 0 -1
screen -d -m -S nasbench201_eval_transfer_3 -L -Logfile ./logs/nasbench201_eval_transfer_3.log srun -p dev --comment="nips" --gres=gpu:1 --time=24:00:00 --cpus-per-task=4 bash ./scripts-search/algos/eval_transfer_3.sh cifar10 0 -1
screen -d -m -S nasbench201_eval_transfer_4 -L -Logfile ./logs/nasbench201_eval_transfer_4.log srun -p dev --comment="nips" --gres=gpu:1 --time=24:00:00 --cpus-per-task=4 bash ./scripts-search/algos/eval_transfer_4.sh cifar10 0 -1
screen -d -m -S nasbench201_eval_transfer_5 -L -Logfile ./logs/nasbench201_eval_transfer_5.log srun -p dev --comment="nips" --gres=gpu:1 --time=24:00:00 --cpus-per-task=4 bash ./scripts-search/algos/eval_transfer_5.sh cifar10 0 -1






