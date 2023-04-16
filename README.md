# SketchMLP
Fine-Grained Sketch-Based Image Retrieval


## 1. Experimental results

<div class="center">

| Methods (QMUL-Chair-V2) |  Acc.@1  |  Acc.@5  |  Acc.@10  |  logs  |
|:--------:|:--------:|:--------:|:--------:|:--------:|
| **Triplet-SN** (CVPR 2016) | 33.75 | 65.94 | 79.26 |  [logs](./logs/baseline/triplet.txt)  |
| **Triplet-Att-SN** (CVPR 2017) | 37.15 | 67.80 | 82.97 |  [logs](./logs/baseline/triplet_att.txt)  |
| **OnTheFly** (CVPR 2020) | 39.01 | 75.85 | 87.00 |  [logs](./logs/baseline/onthefly.txt)  |
| **CMHM-SBIR** (BMVC 2020) | 51.70 | 80.50 | 88.85 |  [logs](./logs/baseline/CMHM.txt)  |
| **SketchAA** (ICCV 2021) | 52.89 | 73.80 | 94.88 |  [logs](./logs/)  |
| **Semi-Sup** (CVPR 2021) | 60.20 | 78.10 | 90.81 |  [logs](./logs/)  |
| **StyleMeUp** (CVPR 2021) | 62.86 | 79.60 | 91.14 |  [logs](./logs/)  |
| **NT-SBIR** (CVPR 2022) | 64.80 | 79.10 | - |  [logs](./logs/)  |
| **XModalViT** (BMVC 2022) | 63.48 | - | 95.02 |  [logs](./logs/)  |
| **SkechMLP (Ours)** | **67.62** | **91.10** | **95.37** |  [logs](./logs/ours.log)  |

</div>


## 2. Ablation Study

### 2.1 Encoder Type

<div class="center">

| Type |  Acc.@1  |  Acc.@5  |  Acc.@10  |  logs  |
|:--------:|:--------:|:--------:|:--------:|:--------:|
| feat[0] | 64.41 | 89.32 | 95.37 |  [logs](./logs/type/0.log)  |
| mean | 62.99 | 88.97 | 94.66 |  [logs](./logs/type/mean.log)  |
| max | 59.79 | 91.46 | 95.02 |  [logs](./logs/type/max.log)  |
| **MLP Block** | **67.62** | **91.10** | **95.37** |  [logs](./logs/ours.log)  |

</div>

### 2.2 Cross Block Nums

<div class="center">

| Nums |  Acc.@1  |  Acc.@5  |  Acc.@10  |  logs  |
|:--------:|:--------:|:--------:|:--------:|:--------:|
| 1 | 65.13 | 91.10 | 93.59 |  [logs](./logs/nums/1.log)  |
| 2 | 67.26 | 90.39 | 95.73 |  [logs](./logs/nums/2.log)  |
| **3** | **67.62** | 91.10 | 95.37 |  [logs](./logs/ours.log)  |
| 4 | 65.48 | **91.82** | 94.66 |  [logs](./logs/nums/4.log)  |
| 5 | 62.63 | 89.68 | **96.09** |  [logs](./logs/nums/5.log)  |

</div>

### 2.3 Loss Function

<div class="center">

| L<sub>cross</sub> | L<sub>self</sub> | L<sub>con</sub> |  Acc.@1  |  Acc.@5  |  Acc.@10  |  logs  |
|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| √ | - | - | 53.74 | 81.85 | 90.75 |  [logs](./logs/loss/both.log)  |
| √ | √ | - | 52.31 | 79.72 | 90.75 |  [logs](./logs/loss/lcon.log)  |
| √ | √* | - | 52.67 | 83.63 | 90.75 |  [logs](./logs/loss/last.log)  |
| √ | - | √ | 62.99 | 90.04 | 95.02 |  [logs](./logs/loss/self.log)  |
| √ | √* | √ | 66.90 | 91.10 | 95.37 |  [logs](./logs/loss/3loss.log)  |
| √ | √ | √ | **67.62** | **91.10** | **95.37** |  [logs](./logs/ours.log)  |

</div>

### 2.4 Embedding Dimension

<div class="center">

| Dims |  Acc.@1  |  Acc.@5  |  Acc.@10  |  logs  |
|:--------:|:--------:|:--------:|:--------:|:--------:|
| 128 | 64.41 | 86.48 | **95.73** |  [logs](./logs/dims/128.log)  |
| **256** | **67.62** | **91.10** | 95.37 |  [logs](./logs/ours.log)  |
| 512 | 61.21 | 90.39 | 95.02 |  [logs](./logs/dims/512.log)  |

</div>

### 2.5 Load Datasets

<div class="center">

| Methods |  Acc.@1  |  Acc.@5  |  Acc.@10  |  logs  |
|:--------:|:--------:|:--------:|:--------:|:--------:|
| LD@A | 52.67 | 83.99 | 92.17 |  [logs](./logs/load/a.log)  |
| **LD@B** | **67.62** | 91.10 | 95.37 |  [logs](./logs/ours.log)  |

</div>

