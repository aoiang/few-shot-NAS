# Few-shot Neural Architecture Search

<p align="center">
<img src='https://github.com/aoiang/paper-images/blob/master/few-shot-nas/terser.png?raw=true' width="600">
</p>


```
@misc{zhao2020fewshot,
      title={Few-shot Neural Architecture Search}, 
      author={Yiyang Zhao and Linnan Wang and Yuandong Tian and Rodrigo Fonseca and Tian Guo},
      year={2020},
      eprint={2006.06863},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Introduction

One-shot Neural Architecture Search uses a single supernet to approximate the performance each architecture. However, this performance estimation is super inaccurate because of co-adaption among operations in supernet. Few-shot NAS uses multiple supernets with less edges(operations) and each of them covers different regions of the search space to alleviate the undesired co-adaption. Compared to one-shot NAS, few-shot NAS greatly improve the performance of architecture evaluation with a small increase of overhead. Please click [here][1] to see our paper.


## Few-shot NAS on NasBench201
Please refer <a href="./Few-Shot_NasBench201">**here**</a> to see how to use few-shot NAS improve the search performance on NasBench201.

## Few-shot NAS on Cifar10
Please refer <a href="./Few-Shot-NAS_cifar10">**here**</a> to test our state-of-the-art models searched by few-shot NAS.











[1]: https://arxiv.org/abs/2006.06863






