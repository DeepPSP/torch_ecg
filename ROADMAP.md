# Torch-ECG Roadmap

This document outlines the architectural upgrades and future development plans for the `torch_ecg` library.

## Architectural Upgrades

### 1. Introduction of Registry Pattern
**Objective**: Eliminate lengthy `if-elif` branches in downstream models like `ECG_CRNN` to enhance code maintainability and extensibility.
- **Current State**: Adding a new backbone currently requires modifying multiple files, including `models/ecg_crnn.py`.
- **Strategy**: Implement a registry mechanism similar to the one used in the `fl-sim` library. Establish `BACKBONES`, `MODELS`, and `SSL` registries.
  - Use decorators like `@register_backbone("resnet")` to register modules.
  - Use a unified `BACKBONES.build(name, **kwargs)` method for module instantiation.
- **Benefits**: Decouples model definition from construction logic, making it easier for both maintainers and users to inject custom backbones.

### 2. Standardized Backbone API
**Objective**: Provide a unified feature extraction interface for Self-Supervised Learning (SSL) and multi-task learning.
- **Strategy**: Follow the convention used in the `timm` library by providing a `forward_features(x)` method for all CNN and Transformer backbones.
- **Features**:
  - Unified return of feature maps instead of classification logits.
  - Support for accessing intermediate activations for feature fusion or saliency analysis (e.g., Grad-CAM).

### 3. Leveraging Lazy Modules for Configuration Optimization
**Objective**: Reduce the burden of manually calculating and specifying `in_channels` in configuration files.
- **Strategy**: Introduce `nn.LazyLinear` or `nn.LazyConv1d` in complex SSL modules.
- **Benefits**: Simplifies `model_configs` by allowing the model to automatically infer input dimensions during the first forward pass, reducing boilerplate code.

### 4. Consolidation and Optimization of Preprocessors and Augmenters
**Objective**: Eliminate redundancy between NumPy and PyTorch implementations and optimize performance by keeping computations on the GPU.
- **Pure PyTorch Filtering**: Implement `BandPass` and `BaselineRemove` using pure PyTorch (e.g., using `torchaudio.functional` or custom FFT-based filters) to avoid expensive CPU-GPU data transfers.
- **Unification of Managers**: Refactor `PreprocManager` and `AugmenterManager` to share a common base or registry, as their logic for managing sequences of transforms is very similar.
- **Dimension Agnostic Transforms**: Ensure all preprocessors and augmenters can handle arbitrary batch and lead dimensions (using `...` in slicing and einops where possible), reducing the need for functions like `preprocess_multi_lead_signal`.
- **Numpy Version Maintenance**: Keep `_preprocessors` (NumPy version) only for offline data preparation or deployment environments without PyTorch, while making the PyTorch `preprocessors` the primary choice for training pipelines.

### 5. Pandas 3.0 Migration and Dtype Consistency
**Objective**: Ensure compatibility with Pandas 3.0+, particularly concerning Arrow-backed strings and stricter type checking.
- **Explicit Object Dtypes**: Explicitly set `dtype=object` for DataFrame columns intended to hold list-like objects (e.g., diagnoses, available signals) to prevent errors when Arrow-backed strings are used.
- **Initialization Refactoring**: Replace patterns of initializing columns with `None` or `""` and then populating with lists via `.at[]` with more robust initializations like `[[] for _ in range(len(df))]` or using `.apply()`.
- **Vectorized Operations**: Prefer `.apply()` or other vectorized pandas operations over `iterrows()` loops for better performance and type consistency.

---

## Self-Supervised Learning (SSL) Roadmap

- [ ] Implement a base contrastive learning framework (supporting paradigms like SimCLR, MoCo).
- [ ] Implement Masked Autoencoder (MAE) logic optimized for 1D physiological signals.
- [ ] Official implementation of classic models: CLOCS, ST-MEM, MAE-ECG.
- [ ] Provide a unified Fine-tuning API for seamless integration with downstream tasks in `torch_ecg.models`.

---

## Other Tasks

- [ ] Improve test coverage, especially for the newly introduced SSL and Transformer modules.
- [ ] Optimize automated documentation generation.
- [ ] Release more pre-trained weights for SOTA models from PhysioNet/CinC challenges.
