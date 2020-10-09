# [Algorithms for Hyper-Parameter Optimization(TPE)][1]

Please go to the root path of FEW-SHOT_NASBENCH201 to run these scripts.

## One-shot Version
  ```
  bash ./search_algos/TPE/one-shot/TPE.sh cifar10 -1
  ```
  The output will place on ./output/one-shot-NAS

## Few-shot Version
  ```
  bash ./search_algos/TPE/few-shot/TPE.sh cifar10 -1
  ```
The output will place on ./output/few-shot-NAS.

## Vanilla NAS Version
  ```
  bash ./search_algos/TPE/vanilla-NAS/TPE.sh cifar10 -1
  ```
  The output will place on ./output/vanilla-NAS


For the details of the scripts, 'cifar10' means searched dataset; '-1' is the seed value(negative value means random seed);

The search algorithm will run 50 times.

[1]: https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf






      
    
      
      

                 
                 
         
               
    






