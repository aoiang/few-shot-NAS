# [DARTS: Differentiable Architecture Search][1]

Please go to the root path of FEW-SHOT_NASBENCH201 to run these scripts.

## One-shot Version
  ```
  bash ./search_algos/DARTS/one-shot/DARTS-V2.sh cifar10 0 -1 0 NASBENCH201_PATH
  ```
  The output will place on ./output/one-shot-NAS

## Few-shot Version

  These scripts files can be run simultaneously on multiple GPUs. 
  ```
  bash ./search_algos/DARTS/few-shot/DARTS-V2_0.sh cifar10 0 -1 0 NASBENCH201_PATH

  bash ./search_algos/DARTS/few-shot/DARTS-V2_1.sh cifar10 0 -1 0 NASBENCH201_PATH

  bash ./search_algos/DARTS/few-shot/DARTS-V2_2.sh cifar10 0 -1 0 NASBENCH201_PATH

  bash ./search_algos/DARTS/few-shot/DARTS-V2_3.sh cifar10 0 -1 0 NASBENCH201_PATH

  bash ./search_algos/DARTS/few-shot/DARTS-V2_4.sh cifar10 0 -1 0 NASBENCH201_PATH
  ```
  Alternatively, you can simply run at once
  ```
  bash ./search_algos/DARTS/few-shot/DARTS-V2_BATCH.sh cifar10 0 -1 0 NASBENCH201_PATH
  ```
The output will place on ./output/few-shot-NAS.

For the details of the scripts, 'cifar10' means searched dataset; First '0' means the BN type; '-1' is the seed value(negative value means random seed); Second '0' means 0th runs; 'NASBENCH201_PATH' is the path that contains NasBench201 dataset. 

[1]: https://arxiv.org/abs/1806.09055







      
    
      
      

                 
                 
         
               
    






