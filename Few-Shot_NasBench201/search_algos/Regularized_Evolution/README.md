# [Regularized Evolution for Image Classifier Architecture Search][1]

Please go to the root path of FEW-SHOT_NASBENCH201 to run these scripts.

## One-shot Version
  ```
  bash ./search_algos/Regularized_Evolution/one-shot/R-EA.sh cifar10 20 -1
  ```
  The output will place on ./output/one-shot-NAS

## Few-shot Version
  ```
  bash ./search_algos/Regularized_Evolution/few-shot/R-EA.sh cifar10 20 -1
  ```
The output will place on ./output/few-shot-NAS.

## Vanilla NAS Version
  ```
  bash ./search_algos/Regularized_Evolution/vanilla-NAS/R-EA.sh cifar10 20 -1
  ```
  The output will place on ./output/vanilla-NAS


For the details of the scripts, 'cifar10' means searched dataset; '20'  means sample size of evolutionary algorithm; '-1' is the seed value(negative value means random seed);

The search algorithm will run 50 times.

[1]: https://arxiv.org/abs/1802.01548






      
    
      
      

                 
                 
         
               
    






