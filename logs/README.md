# 2. Experimental Results

A framework for Fine-Grained Sketch-Based Image Retrieval (FG-SBIR).

### 2.1 QMUL-Chair-V2

Since some baselines do not provide access to the source code or the key parts of the code, it can be difficult to reproduce their results, or the reproducible results may not be satisfactory. Therefore, we use the data submitted by the authors in their papers as the baselines.

Our training was performed on a single Nvidia GeForce GTX 1080Ti 11G GPU, while the baselines were trained on either a single Nvidia GeForce RTX 2080Ti 11G or a single Nvidia GeForce RTX 3090 24G GPU. For the Nvidia GeForce RTX 3090 24G GPU case, we trained on a single Nvidia GeForce RTX 4090 24G GPU and the results were not great either.

<div align=left>

| Methods (QMUL-Chair-V2) |  Acc.@1  |  Acc.@5  |  Acc.@10  |  logs  |
|:--------:|:--------:|:--------:|:--------:|:--------:|
| **SketchAA** (ICCV 2021) | 52.89 | 73.80 | 94.88 |  [logs](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_SketchAA_Abstract_Representation_for_Abstract_Sketches_ICCV_2021_paper.pdf)  |
| **Semi-Sup** (CVPR 2021) | 60.20 | 78.10 | 90.81 |  [logs](https://arxiv.org/abs/2103.13990)  |
| **StyleMeUp** (CVPR 2021) | 62.86 | 79.60 | 91.14 |  [logs](https://arxiv.org/abs/2103.15706)  |
| **NT-SBIR** (CVPR 2022) | 64.80 | 79.10 | - |  [logs](https://arxiv.org/abs/2203.14817)  |
| **XModalViT** (BMVC 2022) | 63.48 | - | 95.02 |  [logs](https://arxiv.org/abs/2210.10486)  |
| **SketchViT (Ours)** | **67.62** | **91.10** | **95.37** |  [logs](./logs/ours.log)  |

</div>

