# Supernet(s) Training and Evaluating

Please go to the root path of FEW-SHOT_NASBENCH201 to run these scripts. 

Since we use weight-transferring technique to accelerate the supernet(s) training process, please make sure that you should first train the one-shot supernet and the few-shot supernet should be trained after training process of one-shot supernet is done. The one-shot supernet model is used to transfer the weights to few-shot supernet models.  

## Train One-shot Supernet
  ```
  bash ./supernet/one-shot/train.sh cifar10 0 NASBENCH201_PATH
  ```
  The checkpoint will be saved on ./supernet_checkpoint/one-shot

## Evaluate One-shot Supernet
  ```
  bash ./supernet/one-shot/eval.sh cifar10 0 NASBENCH201_PATH OUTPUT_PATH
  ```
  The output will place on OUTPUT_PATH, and you can use this output file to replace ./supernet_info/one-shot-supernet

## Train Few-shot Supernets
  ```
  bash ./supernet/few-shot/train.sh cifar10 0 NASBENCH201_PATH ONE-SHOT_SUPERNET_MODEL_NAME OPERATION_TO_SPLIT(0-4)
  ```
The checkpoint will be saved on ./supernet_checkpoint/few-shot

## Evaluate Few-shot Supernets
  ```
  bash ./supernet/few-shot/eval.sh cifar10 0 NASBENCH201_PATH OUTPUT_PATH
  ```
  The output will place on OUTPUT_PATH, and you can use this output file to replace ./supernet_info/few-shot-supernet



For the details of the scripts, 'cifar10' means searched dataset; '0' means the BN type; 'NASBENCH201_PATH' is the path that contains NasBench201 dataset. 'OUTPUT_PATH' is customized by yourself to save the accuracy information of all architectures. 'ONE-SHOT_SUPERNET_MODEL_NAME' is the name of one-shot supernet model, typically the format is like seed-0-last-info.pth; 'OPERATION_TO_SPLIT' is a hyper-parameters to control which edge to split in the supernet. Be sure that, to train the few-shot supernets, we should run the scripts 5 times with the OPERATION_TO_SPLIT from 0 to 4. 








      
    
      
      

                 
                 
         
               
    






