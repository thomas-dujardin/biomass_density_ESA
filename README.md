# <p align="center">Use Case 2b, Work Packages 4 and 5</p>
### <p align="center">Project: 101130544 — ThinkingEarth — HORIZON-EUSPA-2022-SPACE</p>

# <h1 align="center">Biomass Density Estimation in the Amazon basin using Synthetic Aperture Radar (SAR) and multispectral (MS) data</h1>

[![License: Code](https://img.shields.io/badge/License--Code-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License: model](https://img.shields.io/badge/License--Model-CC--BY--NC--SA--4.0-blue.svg)]([https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by-nc-sa/4.0/))

This GitHub directory contains the implementation of biomass density estimation for the Amazon basin of Use Case 2b. The grant agreement stipulates:

> <p align="center">For UC2b, we will estimate AGB and assess forest carbon stock at fine spatial resolution and large geographic coverage using DL. To achieve this, we will utilize GEDI LiDAR data fused with S1&S2, elevation, meteo and land cover data for the Amazon basin. We will also model the amount of sequestered carbon and its year-to-year dynamics, as well as the performance of forests in carbon sinks for any location, country, or specific carbon REDD+ project areas. All DL models will incorporate xAI, and we will thoroughly evaluate the generalization of methods.</p>

This directory only estimates biomass density in the Amazon basin, allowing the rest of the Use Case to be carried on. It uses a slightly modified version of the pretrained [Copernicus-FM v1](https://github.com/zhu-xlab/Copernicus-FM) multimodal foundation model, with additional ML components, to predict single-channel biomass density maps of size 264 × 264. The dataset currently contains 504 Amazon Basin tiles. Each input tile is represented as a 264 × 264 spatial patch, corresponding approximately to 2.64 km × 2.64 km at 10 m resolution. The target is an above-ground biomass density map derived from ESA biomass data, whose effective spatial support is coarser than the 10 m input grid. Its ability to process a wide variety of spectral data is especially useful for biomass density estimation in forests.

## Target data

The target is above-ground biomass density, not a semantic mask. It is treated as a continuous regression target. The target grid appears at a coarser effective spatial resolution than the Sentinel-1/Sentinel-2 input grid. Therefore, predictions are optionally pixelized/downsampled (in a differentiable way) before computing the loss.

## Model Overview

Copernicus-FM v1 is slightly modified. At its core, it is mainly supported by a ViT-Base/16 Vision Transformer, which, by default, outputs one [CLS] token. We tweak this output to obtain the rest of the output patch embeddings. Let \(n_{\text{embed}}\) denote the ViT embedding dimension, \(N\) the number of patch tokens, \(B\) the batch size, \(C\) the number of channels:
- Sentinel-1: SAR VV, VH;
-  Sentinel 2: RGB and MS B02, B03, B04, B08, B11 and B12 (SWIR, important for moisture/disturbance/biomass)),
and \(H, W\) respectively the height and width of the input image. Then, roughly, the pipeline is as follows:

<p align="left"> The original Copernicus-FM ViT uses a `[CLS]` token for global image representation. In this repository, `src/model_vit.py` is modified so that the model returns the spatial patch tokens instead. These tokens are reshaped into a 16 × 16 feature grid before being decoded into a biomass map. </p>

**<p align="center">
(B, C, H, W) -> Copernicus-FM v1 = (B, N, n_embed) -> UNet upper branch (Conv2Ds) = (B, 1, H, W) -> (optionally) UResNet34 refiner = (B, 1, H, W) -> pixelize = (B, 1, H, W) (final output map) </p>**

Which is summarized in the diagram below:

![biomass density estimation diagram](https://github.com/thomas-dujardin/biomass_density_ESA/blob/main/assets/biomass.png?raw=true)

For a simpler explanation:

IMAGE

## Installation

run (...)

conda create -n biomass python=3.11
conda activate biomass
pip install -r requirements.txt

## Minimal command, experiment examples, inference
- ### Runs the model with the random seed and hyperparameters we've used, on the same data split. The Copernicus-FM v1 encoder is frozen, and the refiner is turned off;
  python biomass_density.py --use_gpu --nr_epochs 20 --train_batch_size 4

- ### Frozen Copernicus-FM encoder, train decoder/refiner
  python biomass_density.py
  
- ### Train all components
  python biomass_density.py --train_everything

- ### Randomly initialize Copernicus-FM
  python biomass_density.py --random_init_copernicus

- ### Disable refiner
  python biomass_density.py --no-refiner_on

These arguments can be combined.

## Inference

Run:

python infer_biomass.py \
  --input_tensor data/cache/tile_000123/x.pt \
  --biomass_checkpoint checkpoints/biomass_best.pt \
  --out_dir data/predictions/tile_000123

## Experimental results:

## Results

| Setting | Encoder | Refiner | Loss | RMSE | MAE | Notes |
|---|---|---|---|---:|---:|---|
| baseline | frozen Copernicus-FM | off | L1 | TBD | TBD | coarse target |
| + refiner | frozen Copernicus-FM | UResNet34 | L1 + Laplacian | TBD | TBD | slower |
| random encoder | random ViT-B | off | L1 | TBD | TBD | ablation |

## Current limitations

- GEDI LiDAR is not currently used as an input modality. Could be implemented from scratch by correctly fusing the embeddings and the GEDI LiDAR data.
- The target biomass map is not a true 10 m pixelwise label; it has coarser effective spatial support.
- The current implementation requires access to private Google Earth Engine assets.
- The current model is trained on a small number of Amazon Basin tiles.
- Edge artifacts in GT tiles.
- Currently, single-GPU only.
