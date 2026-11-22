# Mitigating the Higher-Density Bias of Diffusion Models under Recursive Training on Synthetic Data

IE 643: Deep Learning - Theory and Practice (July-Dec 2026), course project
**Team Tensor Titans** - Kolanu Sarvajith (24B0924)

## The problem

A generative model never copies its training data perfectly: it reproduces common, easy kinds of images more
reliably than rare or harder ones. When the next model is trained on these generated images, it starts from an
already shifted mix and shifts it further. After a few rounds, a few kinds of images take over the output and rare
kinds become scarce or vanish. This project measures that higher-density bias and reduces it.

**Setup.** A small unconditional diffusion model (DDPM, 24.3M-parameter U-Net) is trained on a density-imbalanced
subset of CIFAR-10 (27,400 images: five common classes with about 4,900 images each, and bird 1,500, deer 800,
cat 400, dog 150, horse 50). It is then retrained twice on its own samples:

```
D0 (real) -> M0 -> D1 -> M1 -> D2 -> M2 -> D3
```

Every model is trained from scratch on 27,400 images (20k steps, images generated with 50-step DDIM).

## Method

1. **Fixed density clusters of the original data.** DINOv2 turns each image into a feature vector so that visually
   similar images get similar vectors. k-means (K = 30) on these features of D0 gives 30 clusters, each one kind of
   image (for example airplanes against a plain sky, or trucks). The clusters are computed once and kept fixed for
   every round. A ResNet-18 trained on D0 only assigns generated images to them.
2. **Cluster-based loss weighting.** Before training the next model on a generated set, each cluster gets the ratio
   `r_k = (share of cluster k in the original data) / (share of cluster k in the current training set)`,
   clipped to [0.1, 10]. Each image's diffusion loss is multiplied by the ratio of its cluster (rescaled to mean 1),
   so over-produced clusters count less and lost clusters count more. Only the loss changes; no class labels are used.
3. **Comparison** with unweighted training and with class-based weighting (the same ratio over the 10 classes),
   all starting from the same M0 and D1.

## Results (round 3)

| | Unweighted | Class-based weighting | Cluster-based weighting |
|---|---|---|---|
| Cluster KL to the original mix (lower is better) | 0.406 | 0.257 | **0.177** |
| Diversity over clusters (original data: 24.6) | 17.3 | 19.5 | **21.3** |
| Image quality, FD (lower is better) | 1072 | 994 | **942** |

Without weighting, an airplane-against-sky cluster grows from 4.3% to 19.2% of the images while the largest truck
cluster falls from 8.8% to 2.0%. Cluster-based weighting holds them at 12.2% and 4.6%. Details are in the reports.

## Repository contents

| File | What |
|---|---|
| `TensorTitans_novelty_kaggle.ipynb` | Complete code: data split, diffusion training, recursive loop, fixed clusters, cluster-based weighting, evaluation and figures. Every module is written out by the notebook itself (`%%writefile` cells). |
| `reports/` | Intensive assessment report and the 2-page novelty assessment report (PDF). |
| `README.md` | This file. |

## How to run

The notebook is written for **Kaggle** with **GPU T4 x2** and **Internet on**.

1. Upload the notebook to Kaggle (*File -> Import Notebook*).
2. In the right panel set *Accelerator: GPU T4 x2* and *Internet: On*.
3. Choose *Save Version -> Save & Run All (Commit)*.

The settings cell at the top selects the methods (`CHAINS`). The notebook is set up to reuse the models of an
earlier run, added as a Kaggle dataset containing its `run/` folder, and then regenerates those chains and trains the
cluster-weighted chain (about 10 h). Outputs are written to `run/results/` and packed into `novelty_results.zip`.

Main dependencies: PyTorch, torchvision, diffusers (>= 0.30), DINOv2 (via torch.hub, or timm as fallback), NumPy, matplotlib (all available on
Kaggle; the notebook installs diffusers).

## Reference

M. Oquab et al. DINOv2: Learning robust visual features without supervision. TMLR 2024.
https://arxiv.org/abs/2304.07193
