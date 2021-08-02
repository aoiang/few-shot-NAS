# Few-shot Neural Architecture Search
[Yiyang Zhao](https://zhaoyiyang.me), [Linnan Wang](https://linnanwang.github.io/), [Yuandong Tian](https://yuandong-tian.com/), [Rodrigo Fonseca](http://cs.brown.edu/~rfonseca/), [Tian Guo](https://tianguo.info)


<p align="center">
<img src='https://github.com/aoiang/paper-images/blob/master/few-shot-nas/terser.png?raw=true' width="600">
</p>

## Introduction

One-shot Neural Architecture Search uses a single supernet to approximate the performance each architecture. However, this performance estimation is super inaccurate because of co-adaption among operations in supernet. Few-shot NAS uses multiple supernets with less edges(operations) and each of them covers different regions of the search space to alleviate the undesired co-adaption. Compared to one-shot NAS, few-shot NAS greatly improve the performance of architecture evaluation with a small increase of overhead. Please click [here][1] to see our paper.


## Paper 
[Few-shot Neural Architecture Search](http://proceedings.mlr.press/v139/zhao21d/zhao21d.pdf)

If you use the few-shot NAS data or code, please cite: 

```bibtex
@InProceedings{pmlr-v139-zhao21d,
  title = 	 {Few-Shot Neural Architecture Search},
  author =       {Zhao, Yiyang and Wang, Linnan and Tian, Yuandong and Fonseca, Rodrigo and Guo, Tian},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {12707--12718},
  year = 	 {2021},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/zhao21d/zhao21d.pdf},
  url = 	 {http://proceedings.mlr.press/v139/zhao21d.html},
}
```


## How to use 

### Few-shot NAS on NasBench201
Please refer <a href="./Few-Shot_NasBench201">**here**</a> to see how to use few-shot NAS improve the search performance on NasBench201.

### Few-shot NAS on Cifar10
Please refer <a href="./Few-Shot-NAS_cifar10">**here**</a> to test our state-of-the-art models searched by few-shot NAS.


## Media Coverage

### English version
[Facebook AI Research blog post](https://ai.facebook.com/blog/introducing-few-shot-neural-architecture-search/)

[Poster](https://drive.google.com/file/d/1veYZpNgUpoedyEqvQ0iH3tM-qXNLmxYB/view?usp=sharing)


### Chinese version
[机器之心专栏文字介绍](https://mp.weixin.qq.com/s/X0flnuKKOnQkamjCxD_KrQ)

[机器之心专栏直播回放](https://jmq.h5.xeknow.com/s/2nndbz)

[Bilibili](https://b23.tv/1uQxRu)






[1]: https://arxiv.org/abs/2006.06863






