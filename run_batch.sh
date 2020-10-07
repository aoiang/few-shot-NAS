source  /private/home/linnanwang/virtual_envs/nas/bin/activate

#!/usr/bin/env bash

for (( c=4; c<5; c++ ))
do
    echo "----$c"
    screen -d -m -S nasbench201_gdas_0_runs_$c -L -Logfile ./logs/nasbench201_gdas_0_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=1 bash ./scripts-search/algos/GDAS_0.sh cifar10 0 -1 $c
    screen -d -m -S nasbench201_gdas_1_runs_$c -L -Logfile ./logs/nasbench201_gdas_1_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=1 bash ./scripts-search/algos/GDAS_1.sh cifar10 0 -1 $c
    screen -d -m -S nasbench201_gdas_2_runs_$c -L -Logfile ./logs/nasbench201_gdas_2_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=1 bash ./scripts-search/algos/GDAS_2.sh cifar10 0 -1 $c
    screen -d -m -S nasbench201_gdas_3_runs_$c -L -Logfile ./logs/nasbench201_gdas_3_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=1 bash ./scripts-search/algos/GDAS_3.sh cifar10 0 -1 $c
    screen -d -m -S nasbench201_gdas_4_runs_$c -L -Logfile ./logs/nasbench201_gdas_4_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=1 bash ./scripts-search/algos/GDAS_4.sh cifar10 0 -1 $c


    screen -d -m -S nasbench201_darts_0_runs_$c -L -Logfile ./logs/nasbench201_darts_0_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=1 bash ./scripts-search/algos/DARTS-V2_0.sh cifar10 0 -1 $c
    screen -d -m -S nasbench201_darts_1_runs_$c -L -Logfile ./logs/nasbench201_darts_1_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=1 bash ./scripts-search/algos/DARTS-V2_1.sh cifar10 0 -1 $c
    screen -d -m -S nasbench201_darts_2_runs_$c -L -Logfile ./logs/nasbench201_darts_2_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=1 bash ./scripts-search/algos/DARTS-V2_2.sh cifar10 0 -1 $c
    screen -d -m -S nasbench201_darts_3_runs_$c -L -Logfile ./logs/nasbench201_darts_3_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=1 bash ./scripts-search/algos/DARTS-V2_3.sh cifar10 0 -1 $c
    screen -d -m -S nasbench201_darts_4_runs_$c -L -Logfile ./logs/nasbench201_darts_4_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=1 bash ./scripts-search/algos/DARTS-V2_4.sh cifar10 0 -1 $c


    screen -d -m -S nasbench201_ENAS_0_runs_$c -L -Logfile ./logs/nasbench201_ENAS_0_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=1 bash ./scripts-search/algos/ENAS_0.sh cifar10 0 -1 $c
    screen -d -m -S nasbench201_ENAS_1_runs_$c -L -Logfile ./logs/nasbench201_ENAS_1_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=1 bash ./scripts-search/algos/ENAS_1.sh cifar10 0 -1 $c





#    screen -d -m -S nasbench201_ENAS_2_runs_$c -L -Logfile ./logs/nasbench201_ENAS_2_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=4 bash ./scripts-search/algos/ENAS_2.sh cifar10 0 -1 $c
#    screen -d -m -S nasbench201_ENAS_3_runs_$c -L -Logfile ./logs/nasbench201_ENAS_3_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=4 bash ./scripts-search/algos/ENAS_3.sh cifar10 0 -1 $c
#    screen -d -m -S nasbench201_ENAS_4_runs_$c -L -Logfile ./logs/nasbench201_ENAS_4_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=4 bash ./scripts-search/algos/ENAS_4.sh cifar10 0 -1 $c
#
#
#
#    screen -d -m -S nasbench201_PCDARTS_0_runs_$c -L -Logfile ./logs/nasbench201_PCDARTS_0_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=4 bash ./scripts-search/algos/PCDARTS_0.sh cifar10 0 -1 $c
#    screen -d -m -S nasbench201_PCDARTS_1_runs_$c -L -Logfile ./logs/nasbench201_PCDARTS_1_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=4 bash ./scripts-search/algos/PCDARTS_1.sh cifar10 0 -1 $c
#    screen -d -m -S nasbench201_PCDARTS_2_runs_$c -L -Logfile ./logs/nasbench201_PCDARTS_2_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=4 bash ./scripts-search/algos/PCDARTS_2.sh cifar10 0 -1 $c
#    screen -d -m -S nasbench201_PCDARTS_3_runs_$c -L -Logfile ./logs/nasbench201_PCDARTS_3_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=4 bash ./scripts-search/algos/PCDARTS_3.sh cifar10 0 -1 $c
#    screen -d -m -S nasbench201_PCDARTS_4_runs_$c -L -Logfile ./logs/nasbench201_PCDARTS_4_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=4 bash ./scripts-search/algos/PCDARTS_4.sh cifar10 0 -1 $c
#
#
#
#    screen -d -m -S nasbench201_SETN_0_runs_$c -L -Logfile ./logs/nasbench201_SETN_0_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=4 bash ./scripts-search/algos/SETN_0.sh cifar10 0 -1 $c
#    screen -d -m -S nasbench201_SETN_1_runs_$c -L -Logfile ./logs/nasbench201_SETN_1_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=4 bash ./scripts-search/algos/SETN_1.sh cifar10 0 -1 $c
#    screen -d -m -S nasbench201_SETN_2_runs_$c -L -Logfile ./logs/nasbench201_SETN_2_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=4 bash ./scripts-search/algos/SETN_2.sh cifar10 0 -1 $c
#    screen -d -m -S nasbench201_SETN_3_runs_$c -L -Logfile ./logs/nasbench201_SETN_3_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=4 bash ./scripts-search/algos/SETN_3.sh cifar10 0 -1 $c
#    screen -d -m -S nasbench201_SETN_4_runs_$c -L -Logfile ./logs/nasbench201_SETN_4_runs_$c.log srun -p learnfair --comment="nips" --gres=gpu:1 --time=15:00:00 --cpus-per-task=4 bash ./scripts-search/algos/SETN_4.sh cifar10 0 -1 $c


done









