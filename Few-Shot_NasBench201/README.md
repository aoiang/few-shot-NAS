# Few-shot NAS on NASBench-201

## Few-shot NAS Greatly Improves Different NAS Algorithms on NASBench-201

<p>
<img src='https://github.com/aoiang/paper-images/blob/master/few-shot-nas/grad_algos.png?raw=true' width="860">
<img src='https://github.com/aoiang/paper-images/blob/master/few-shot-nas/search_algos.png?raw=true' width="860">
</p>


## How to Use Few-shot NAS to Reproduce above Results

### Environment Requirements
```
python >= 3.6, numpy >= 1.9.1, torch >= 1.5.0, hpbandster, json
```


### Download the dataset

The full NASBench-201 dataset can be found at [here](https://drive.google.com/file/d/16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_/view)(4.7G). 

### Gradient-based NAS Algorithms

- <a href="./search_algos/DARTS">**DARTS**</a>: Differentiable Architecture Search
- <a href="./search_algos/PCDARTS">**PCDARTS**</a>: Partial Channel Connections for Memory-Efficient Architecture Search
- <a href="./search_algos/ENAS">**ENAS**</a>: Efficient Neural Architecture Search via Parameter Sharing
- <a href="./search_algos/SETN">**SETN**</a>: One-Shot Neural Architecture Search via Self-Evaluated Template Network


### Vanilla NAS Algorithms
- <a href="./search_algos/Regularized_Evolution">**REA**</a>: Regularized Evolution for Image Classifier Architecture Search
- <a href="./search_algos/Reinforce_Learning">**RL**</a>: Learning Transferable Architectures for Scalable Image Recognition
- <a href="./search_algos/BOHB">**BOHB**</a>: Robust and Efficient Hyperparameter Optimization at Scale
- <a href="./search_algos/TPE">**TPE**</a>: Algorithms for Hyper-Parameter Optimization

For one(few)-shot vanilla NAS algorithms, the search is guided by the individual architecture performance, which is estimated by the supernet. Therefore, we provide the estimated accuracies of all 15625 architectures in NasBench201, which are approximated by both one-shot supernet and few-shot supernets. These files are located on ./supernet_info. Folder './supernet_info' also contains a file named 'nasbench201', which contains the real accuracies of architectures in NasBench201.

If you would like to train and evaluate the supernet(s) by yourself, please follow the instruction <a href="./supernet">**here**</a>.





