# Few-shot NAS on NASBench-201

## Few-shot NAS improves different NAS algorithms on NASBench-201

<p align="center">
<img src='https://github.com/aoiang/paper-images/blob/master/few-shot-nas/grad_algos.png?raw=true' width="1000">
<img src='https://github.com/aoiang/paper-images/blob/master/few-shot-nas/search_algos.png?raw=true' width="1000">
</p>


## How to Use Few-shot NAS to Reproduce above Results

### Download the dataset

The full NASBench-201 dataset can be found at [here](https://drive.google.com/file/d/16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_/view)(4.7G). 

### Gradient-based NAS Algorithms

- DARTS
    - One-shot Version
        ```
        bash ./search_algos/DARTS/one-shot/DARTS-V2.sh cifar10 0 -1 0 NASBENCH201_PATH
        ```
      The output will place on ./output/one-shot-NAS

    - few-shot Version

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








      
    
      
      

                 
                 
         
               
    






