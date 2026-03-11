# Self-Supervised Learning (SSL)

This module contains the self-supervised learning (SSL) algorithms (models) for learning latent representations of ECGs.

Self-supervised learning aims to leverage large-scale unlabeled ECG data to pre-train "Foundation Models" (Backbones) that can be fine-tuned for various downstream tasks (e.g., classification, segmentation, R-peak detection).

## Overview of SSL Paradigms in ECG

Self-supervised learning in ECG can generally be categorized into **Contrastive Learning** and **Masked Modeling**. Both CNN and Transformer architectures can serve as the backbone.

| Model Name | Backbone | Paradigm | Key Characteristics |
| :--- | :--- | :--- | :--- |
| **CLOCS** | CNN (ResNet-like) | Contrastive | Exploits multi-lead consistency (Patient/Lead/Temporal contrast). |
| **ST-MEM** | Transformer (ViT) | Masked Modeling (MAE) | Spatio-Temporal Masked Autoencoder. Treats ECG as 2D patches. |
| **SimCLR (ECG)** | CNN / Transformer | Contrastive | Uses data augmentations (noise, crop, flip) to build positive/negative pairs. |
| **CPC (ECG)** | CNN + RNN/Transformer | Predictive | Contrastive Predictive Coding. Predicts future representations from history. |
| **3M-ECG** | Transformer | Hybrid | Combines Masked Modeling, Metadata matching, and Multi-lead contrastive. |
| **TF-C** | CNN / Transformer | Cross-domain | Time-Frequency Consistency. Contrasts time-domain and frequency-domain. |
| **MAE-ECG** | ViT | Masked Modeling | Vanilla Masked Autoencoder applied to 1D ECG signals. |
| **CMSC** | CNN | Contrastive | Contrastive Multi-segmental Coding within the same ECG record. |
| **ECG-BERT** | Transformer | Masked Modeling | BERT-like masking on tokenized ECG segments. |

## Backbone Architectures

### CNN Backbones (e.g., ResNet, Multi-scopic CNN)
- **Pros**: Translation invariance, efficient local feature extraction, faster inference.
- **Suitability**: Excellent for Contrastive Learning (e.g., CLOCS, SimCLR) as they are sensitive to local waveform changes.

### Transformer Backbones (e.g., ViT, Swin)
- **Pros**: Long-range dependency modeling, global context awareness, highly scalable.
- **Suitability**: The preferred choice for Masked Modeling (e.g., ST-MEM, MAE), as Attention mechanisms handle masked tokens flexibly.

## Roadmap

- [ ] Implement Contrastive Learning Framework (Base class for SimCLR, MoCo).
- [ ] Implement Masked Autoencoder (MAE) logic for 1D signals.
- [ ] Integrate classic models: CLOCS, ST-MEM.
- [ ] Support fine-tuning API for downstream tasks in `torch_ecg.models`.
