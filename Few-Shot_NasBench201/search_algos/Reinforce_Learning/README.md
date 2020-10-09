# [Learning Transferable Architectures for Scalable Image Recognition][1]

Please go to the root path of FEW-SHOT_NASBENCH201 to run these scripts.

## One-shot Version
  ```
  bash ./search_algos/Reinforce_Learning/one-shot/REINFORCE.sh cifar10 -1
  ```
  The output will place on ./output/one-shot-NAS

## Few-shot Version
  ```
  bash ./search_algos/Reinforce_Learning/few-shot/REINFORCE.sh cifar10 -1
  ```
The output will place on ./output/few-shot-NAS.

## Vanilla NAS Version
  ```
  bash ./search_algos/Reinforce_Learning/vanilla-NAS/REINFORCE.sh cifar10 -1
  ```
  The output will place on ./output/vanilla-NAS


For the details of the scripts, 'cifar10' means searched dataset; '-1' is the seed value(negative value means random seed);

The search algorithm will run 50 times.

[1]: https://arxiv.org/abs/1707.07012






      
    
      
      

                 
                 
         
               
    






