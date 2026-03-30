# Torch-ECG Roadmap

This document outlines the architectural upgrades and future development plans for the `torch_ecg` library.

## Architectural Upgrades

### 1. Introduction of Registry Pattern ✅ Done (PR #29)
**Objective**: Eliminate lengthy `if-elif` branches in downstream models like `ECG_CRNN` to enhance code maintainability and extensibility.
- **Status**: Done. `MODELS`, `BACKBONES`, `ATTN_LAYERS` registries are implemented in `models/registry.py`; `OPTIMIZERS`, `SCHEDULERS`, `LOSSES` in `components/registry.py`; `PREPROCESSORS` in `preprocessors/registry.py`. All CNN backbones and downstream models use `@BACKBONES.register()` / `@MODELS.register()` decorators. `Registry.build(name, **kwargs)` is the unified construction interface. Adding a new backbone no longer requires modifying `models/ecg_crnn.py` or any other existing file.
- **Strategy**: Implement a registry mechanism similar to the one used in the `fl-sim` library. Establish `BACKBONES`, `MODELS`, and `SSL` registries.
  - Use decorators like `@register_backbone("resnet")` to register modules.
  - Use a unified `BACKBONES.build(name, **kwargs)` method for module instantiation.
- **Benefits**: Decouples model definition from construction logic, making it easier for both maintainers and users to inject custom backbones.

### 2. Standardized Backbone API ✅ Done (PR #30)
**Objective**: Provide a unified feature extraction interface for Self-Supervised Learning (SSL) and multi-task learning.
- **Status**: Done. All CNN backbones (`ResNet`, `VGG16`, `DenseNet`, `MobileNetV1/V2/V3`, `MultiScopicCNN`, `RegNet`, `Xception`) and the `Transformer` now implement `forward_features(x)` (returns feature maps before the classifier head) and `compute_features_output_shape(seq_len, batch_size)` for shape inference without running a forward pass.
- **Strategy**: Follow the convention used in the `timm` library by providing a `forward_features(x)` method for all CNN and Transformer backbones.
- **Features**:
  - Unified return of feature maps instead of classification logits.
  - Support for accessing intermediate activations for feature fusion or saliency analysis (e.g., Grad-CAM).

### 3. Leveraging Lazy Modules for Configuration Optimization ⬜ Not started
**Objective**: Reduce the burden of manually calculating and specifying `in_channels` in configuration files.
- **Strategy**: Introduce `nn.LazyLinear` or `nn.LazyConv1d` in complex SSL modules.
- **Benefits**: Simplifies `model_configs` by allowing the model to automatically infer input dimensions during the first forward pass, reducing boilerplate code.

### 4. Consolidation and Optimization of Preprocessors and Augmenters 🔄 In Progress
**Objective**: Eliminate redundancy between NumPy and PyTorch implementations and optimize performance by keeping computations on the GPU.
- **Pure PyTorch Filtering** ✅: `BandPass` now uses a zero-phase FFT-based filter and `BaselineRemove` uses dual `avg_pool1d`, both implemented in `utils/utils_signal_t.py`. No more CPU-GPU data transfers.
- **Unification of Managers** ⬜: Refactor `PreprocManager` and `AugmenterManager` to share a common base or registry, as their logic for managing sequences of transforms is very similar.
- **Dimension Agnostic Transforms** ⬜ (augmenters pending): Preprocessors now handle arbitrary leading batch dimensions (`..., n_leads, siglen`). Augmenters still need to be updated.
- **Numpy Version Maintenance** ✅: `_preprocessors` (NumPy) is kept only for offline data preparation or deployment environments without PyTorch, while making the PyTorch `preprocessors` the primary choice for training pipelines.

### 5. Pandas 3.0 Migration and Dtype Consistency ⬜ Not started
**Objective**: Ensure compatibility with Pandas 3.0+, particularly concerning Arrow-backed strings and stricter type checking.
- **Explicit Object Dtypes**: Explicitly set `dtype=object` for DataFrame columns intended to hold list-like objects (e.g., diagnoses, available signals) to prevent errors when Arrow-backed strings are used.
- **Initialization Refactoring**: Replace patterns of initializing columns with `None` or `""` and then populating with lists via `.at[]` with more robust initializations like `[[] for _ in range(len(df))]` or using `.apply()`.
- **Vectorized Operations**: Prefer `.apply()` or other vectorized pandas operations over `iterrows()` loops for better performance and type consistency.

---

## Complete Incomplete Models ⬜

Several model files in `torch_ecg/models/` are currently stubs (`raise NotImplementedError` throughout). All stubs already inherit the correct mixins (`SizeMixin`, `CitationMixin`) and have backbone API signatures (`forward_features`, `compute_features_output_shape`) scaffolded. These should be completed before the SSL phase.

| File | Classes | Notes |
|---|---|---|
| `models/cnn/darknet.py` | `DarkNet` | Backbone for YOLO-style detection |
| `models/cnn/efficientnet.py` | `EfficientNet`, `EfficientNetV2` | Mobile-efficient compound scaling |
| `models/cnn/ho_resnet.py` | `MidPointResNet`, `RK4ResNet`, `RK8ResNet` | Higher-Order ODE ResNets |
| `models/grad_cam.py` | `GradCam` | Saliency analysis (logic partially drafted, not wired up) |
| `models/ecg_fcn.py` | `ECG_FCN` | Fully-convolutional segmentation (fully commented out) |

---

## Self-Supervised Learning (SSL) Roadmap ⬜

The `torch_ecg/ssl/` module is an empty shell. `ssl/README.md` contains a survey of target architectures (CLOCS, ST-MEM, MAE-ECG, SimCLR, TF-C, 3M-ECG, CMSC, ECG-BERT).

**Prerequisites**: Item 3 (Lazy Modules) and the model stubs above should be completed first.

- [ ] Define base classes: `BaseContrastiveLearner`, `BaseMaskedAutoencoder` (in `ssl/base.py`).
- [ ] Implement a base contrastive learning framework (supporting paradigms like SimCLR, MoCo).
- [ ] Implement Masked Autoencoder (MAE) logic optimized for 1D physiological signals.
- [ ] Official implementation of classic models: CLOCS, ST-MEM, MAE-ECG.
- [ ] Provide a unified Fine-tuning API for seamless integration with downstream tasks in `torch_ecg.models`.
- [ ] Add `SSL` registry (analogous to `MODELS`, `BACKBONES`).

---

## Other Tasks

- [ ] Improve test coverage, especially for the newly introduced SSL and Transformer modules.
- [ ] Optimize automated documentation generation.
- [ ] Release more pre-trained weights for SOTA models from PhysioNet/CinC challenges.
