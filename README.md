# <p align="center">Use Case 2b, Work Packages 4 and 5</p>
### <p align="center">Project: 101130544 — ThinkingEarth — HORIZON-EUSPA-2022-SPACE</p>

# <p align="center">Biomass Density Estimation in the Amazon basin using SAR and MS</p>

[![License: Code](https://img.shields.io/badge/License--Code-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License: model](https://img.shields.io/badge/License--Model-CC--BY--NC--SA--4.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)

This GitHub directory contains the implementation of biomass density estimation for the Amazon basin of Use Case 2b. The grant agreement stipulates:

**<p align="center">For UC2b, we will estimate AGB and assess forest carbon stock at fine spatial resolution and large geographic coverage using DL. To achieve this, we will utilize GEDI LiDAR data fused with S1&S2, elevation, meteo and land cover data for the Amazon basin. We will also model the amount of sequestered carbon and its year-to-year dynamics, as well as the performance of forests in carbon sinks for any location, country, or specific carbon REED+ project areas. All DL models will incorporate xAI, and we will thoroughly evaluate the generalization of methods.</p>**

This directory only estimates biomass density in the Amazon basin, allowing the rest of the Use Case to be carried on. It uses a slightly modified version of the pretrained [Copernicus-FM v1](https://github.com/zhu-xlab/Copernicus-FM) multimodal foundation model, with additional ML components, to obtain 1-channel grayscales of 264 x 264 patches (real life size: 2,64 x 2,64 km), predicting the biomass density of 504 100 m x 100 m resolution Amazon basin patches (extracted from ESA data). Its ability to process a wide variety of spectral data is especially useful for biomass density estimation in forests.

## Model Overview

Copernicus-FM v1 is slightly modified. At the core, it is supported by a ViT-base-patch-16 Vision Transformers, which, by default, outputs one [CLS] token. We tweak this output to obtain the rest of the output patch embeddings. If *n_embed* is the dimension of the ViT-B used by CFM-v1, *N* is the number of embedded patches, *B* is the batch size, *C* is the number of channels (>3 in this sceneratio, as we are also interested in SAR and MS imagery), and *H, W* are respectively the height and width of the input image. Roughly, the pipeline is as follows:

**<p align="center">
(B, C, H, W) -> Copernicus-FM v1 = (B, N, n_embed) ->
UNet upper branch (Conv2Ds) = (B, 1, H, W) -> (optionally) UResNet34 refiner = (B, 1, H, W) ->
pixelize = (B, 1, H, W) (final output map)
</p>**

Which is summarized in the diagram below:
