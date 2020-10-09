# [BOHB: Robust and Efficient Hyperparameter Optimization at Scale][1]

Please go to the root path of FEW-SHOT_NASBENCH201 to run these scripts.

## One-shot Version
  ```
  bash ./search_algos/BOHB/one-shot/BOHB.sh cifar10 -1 NASBENCH201_PATH
  ```
  The output will place on ./output/one-shot-NAS

## Few-shot Version
  ```
  bash ./search_algos/BOHB/few-shot/BOHB.sh cifar10 -1 NASBENCH201_PATH
  ```
The output will place on ./output/few-shot-NAS.

## Vanilla NAS Version
  ```
  bash ./search_algos/BOHB/vanilla-NAS/BOHB.sh cifar10 -1 NASBENCH201_PATH
  ```
  The output will place on ./output/vanilla-NAS


For the details of the scripts, 'cifar10' means searched dataset; '-1' is the seed value(negative value means random seed); 'NASBENCH201_PATH' is the path that contains NasBench201 dataset. 


The search algorithm will run 50 times.

[1]: https://arxiv.org/abs/1807.01774






      
    
      
      

                 
                 
         
               
    






