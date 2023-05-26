# ARTrack

The official PyTorch implementation of our **CVPR 2023 Highlight** paper:

**Autoregressive Visual Trecking**

[Yifan Bai](https://github.com/AlexDotHam)

[[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2023/papers/Wei_Autoregressive_Visual_Tracking_CVPR_2023_paper.pdf)] 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/autoregressive-visual-tracking/visual-object-tracking-on-got-10k)](https://paperswithcode.com/sota/visual-object-tracking-on-got-10k?p=autoregressive-visual-tracking)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/autoregressive-visual-tracking/visual-object-tracking-on-tnl2k)](https://paperswithcode.com/sota/visual-object-tracking-on-tnl2k?p=autoregressive-visual-tracking)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/autoregressive-visual-tracking/visual-object-tracking-on-lasot)](https://paperswithcode.com/sota/visual-object-tracking-on-lasot?p=autoregressive-visual-tracking)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/autoregressive-visual-tracking/visual-object-tracking-on-trackingnet)](https://paperswithcode.com/sota/visual-object-tracking-on-trackingnet?p=autoregressive-visual-tracking)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/autoregressive-visual-tracking/visual-object-tracking-on-lasot-ext)](https://paperswithcode.com/sota/visual-object-tracking-on-lasot-ext?p=autoregressive-visual-tracking)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/autoregressive-visual-tracking/visual-object-tracking-on-uav123)](https://paperswithcode.com/sota/visual-object-tracking-on-uav123?p=autoregressive-visual-tracking)

## Highlight

![](figure/overview.jpg)

### :bookmark:Brief Introduction

We present **ARTrack**, an autoregressive framework for visual object tracking. ARTrack tackles tracking as a coordinate sequence interpretation task that estimates object trajectories progressively, where the current estimate is induced by previous states and in turn affects subsequences. This time-autoregressive approach models the sequential evolution of trajectories to keep tracing the object **across frames**, making it superior to existing template matching based trackers that only consider the **per-frame** localization accuracy. ARTrack is simple and direct, eliminating customized localization heads and post-processings. Despite its simplicity, ARTrack achieves state-of-the-art performance on prevailing benchmark datasets.
### :bookmark:Strong Performance

|             Variant             |       ARTrack-256       |       ARTrack-384       |      ARTrack-L-384      |
|:-------------------------------:|:-----------------------:|:-----------------------:|:-----------------------:|
|          Model Config           | ViT-B, 256^2 resolution | ViT-B, 384^2 resolution | ViT-L, 384^2 resolution |
| GOT-10k (AO / SR 0.5 / SR 0.75) |   73.5 / 82.2 / 70.9    |   75.5 / 84.3 / 74.3    |   78.5 / 87.4 / 77.8    |
|    LaSOT (AUC / Norm P / P)     |   70.4 / 79.5 / 76.6    |   72.6 / 81.7 / 79.1    |   73.1 / 82.2 / 80.3    |
| TrackingNet (AUC / Norm P / P)  |   84.2 / 88.7 / 83.5    |   85.1 / 89.1 / 84.8    |   85.6 / 89.6 / 84.8    |
|  LaSOT_ext (AUC / Norm P / P)   |   46.4 / 56.5 / 52.3    |   51.9 / 62.0 / 58.5    |   52.8 / 62.9 / 59.7    |
|          TNL-2K (AUC)           |          57.5           |          59.8           |          60.3           |
|           NfS30 (AUC)           |          64.3           |          66.8           |          67.9           |
|          UAV123 (AUC)           |          67.7           |          70.5           |          71.2           |

### :bookmark:Inference Speed

Our baseline model (backbone: ViT-B, resolution: 256x256) can run at **26 fps** (frames per second) on a single NVIDIA GeForce RTX 3090, our alter decoder version can run at **45 fps** on a single NVIDIA GeForce RTX 3090.

## Acknowledgement

:heart::heart::heart:Our idea is implemented base on the following projects. We really appreciate their excellent open-source works!

- [OSTrack](https://github.com/botaoye/OSTrack) [[related paper](https://arxiv.org/abs/2203.11991)]
- [PyTracking](https://github.com/visionml/pytracking) [[related paper](https://arxiv.org/abs/2208.06888)]

## Citation

If any parts of our paper and code help your research, please consider citing us and giving a star to our repository.

```
@InProceedings{Wei_2023_CVPR,
    author    = {Wei, Xing and Bai, Yifan and Zheng, Yongchao and Shi, Dahu and Gong, Yihong},
    title     = {Autoregressive Visual Tracking},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {9697-9706}
}
```

## Contact

If you have any questions or concerns, feel free to open issues or directly contact me through the ways on my GitHub homepage **provide below paper's title**.