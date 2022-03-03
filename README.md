# [torch_ecg](https://github.com/DeepPSP/torch_ecg/)

Deep learning ecg models implemented using PyTorch

## Main Modules

### [Augmenters](torch_ecg/augmenters)
Augmenters are classes (subclasses of `torch` `Module`) that perform data augmentation in a uniform way and are managed by the [`AugmenterManager`](torch_ecg/augmenters/augmenter_manager.py) (also a subclass of `torch` `Module`). Augmenters and the manager share a common signature of the `formward` method:
```python
forward(self, sig:Tensor, label:Optional[Tensor]=None, *extra_tensors:Sequence[Tensor], **kwargs:Any) -> Tuple[Tensor, ...]:
```

The following augmenters are implemented:
1. baseline wander (adding sinusoidal and gaussian noises)
2. cutmix
3. mixup
4. random flip
5. random masking
6. random renormalize
7. stretch-or-compress (scaling)
8. label smooth (not actually for data augmentation, but has simimlar behavior)

Usage example (this example uses all augmenters except cutmix, each with default config):
```python
import torch
from easydict import EasyDict as ED
from torch_ecg.augmenters import AugmenterManager

config = ED(
    random=False,
    fs=500,
    baseline_wander={},
    label_smooth={},
    mixup={},
    random_flip={},
    random_masking={},
    random_renormalize={},
    stretch_compress={},
)
am = AugmenterManager.from_config(config)
sig, label, mask = torch.rand(2,12,5000), torch.rand(2,26), torch.rand(2,5000,1)
sig, label, mask = am(sig, label, mask)
```

Augmenters can be stochastic along the batch dimension and (or) the channel dimension (ref. the `get_indices` method of the [`Augmenter`](torch_ecg/augmenters/base.py) base class).

### [Preprocessors](torch_ecg/preprocessors)
Also [preprecessors](torch_ecg/_preprocessors) acting on `numpy` `array`s. Similarly, preprocessors are monitored by a manager
```python
import torch
from easydict import EasyDict as ED
from torch_ecg._preprocessors import PreprocManager

config = ED(
    random=False,
    resample={"fs": 500},
    bandpass={},
    normalize={},
)
ppm = PreprocManager.from_config(config)
sig = torch.rand(12,80000).numpy()
sig, fs = ppm(sig, 200)
```

The following preprocessors are implemented
1. baseline removal (detrend)
2. normalize (z-score, min-max, naïve)
3. bandpass
4. resample

## [Databases](torch_ecg/databases)
This module include classes that manipulate the io of the ECG signals and labels in an ECG database, and maintains metadata (statistics, paths, plots, list of records, etc.) of it. This module is migrated and improved from [DeepPSP/database_reader](https://github.com/DeepPSP/database_reader)

After migration, all should be tested again, the progression:

| Database      | Source    | Tested             |
| ------------- | --------- | ------------------ |
| AFDB          | PhysioNet | :heavy_check_mark: |
| ApneaECG      | PhysioNet | :x:                |
| CinC2017      | PhysioNet | :x:                |
| CinC2018      | PhysioNet | :x:                |
| CinC2020      | PhysioNet | :heavy_check_mark: |
| CinC2021      | PhysioNet | :heavy_check_mark: |
| LTAFDB        | PhysioNet | :x:                |
| LUDB          | PhysioNet | :heavy_check_mark: |
| MITDB         | PhysioNet | :x:                |
| SHHS          | NSRR      | :x:                |
| CPSC2018      | CPSC      | :heavy_check_mark: |
| CPSC2019      | CPSC      | :heavy_check_mark: |
| CPSC2020      | CPSC      | :heavy_check_mark: |
| CPSC2021      | CPSC      | :heavy_check_mark: |

NOTE that these classes should not be confused with a `torch` `Dataset`, which is strongly related to the task (or the model). However, one can build `Dataset`s based on these classes, for example the [`Dataset`](benchmarks/train_hybrid_cpsc2021/dataset.py) for the The 4th China Physiological Signal Challenge 2021 (CPSC2021)

## [Implemented Neural Network Architectures](torch_ecg/models)
1. CRNN, both for classification and sequence tagging (segmentation)
2. U-Net
3. RR-LSTM

A typical signature of the instantiation (`__init__`) function of a model is as follows
```python
__init__(self, classes:Sequence[str], n_leads:int, config:Optional[CFG]=None, **kwargs:Any) -> NoReturn:
```
if a `config` is not specified, then the default config will be used (stored in the [`model_configs`](torch_ecg/model_configs) module.

### [CNN backbone](torch_ecg/models/cnn)
#### Implemented
1. VGG
2. ResNet (including vanilla ResNet, ResNet-B, ResNet-C, ResNet-D, ResNeXT, TResNet, Stanford ResNet, Nature Communications ResNet, etc.)
3. MultiScopicNet (CPSC2019 SOTA)
4. DenseNet (CPSC2020 SOTA)
5. Xception

In general, variants of ResNet are the most commonly used architectures, as can be inferred from CinC2020 and CinC2021.

#### Ongoing
1. MobileNet
2. DarkNet
3. EfficientNet

#### TODO
1. HarDNet
2. HO-ResNet
3. U-Net++
4. U-Squared Net
5. etc.

## Other useful tools
### [Loggers](torch_ecg/utils/loggers.py)
Loggers including
1. CSV logger
2. text logger
3. tensorboard logger
are implemented and manipulated uniformly by a manager.

### [R peaks detection algorithms](torch_ecg/utils/rpeaks.py)
This is a collection of traditional (non deep learning) algorithms for R peaks detection collected from [WFDB](https://github.com/MIT-LCP/wfdb-python) and [BioSPPy](https://github.com/PIA-Group/BioSPPy).

### [Trainer](torch_ecg/utils/trainer.py)
An abstract base class `BaseTrainer` is implemented, in which some common steps in building a training pipeline (workflow) are impemented. A few task specific methods are assigned as `abstractmethod`s, for example the method
```python
evaluate(self, data_loader:DataLoader) -> Dict[str, float]:
```
for evaluation on the validation set during training and perhaps further for model selection and early stopping.

## Usage
See case studies in the [benchmarks folder](/benchmarks/).

Taking [CPSC2021](benchmarks/train_hybrid_cpsc2021) for example, the steps are
1. Write a [`Dataset`](benchmarks/train_hybrid_cpsc2021/dataset.py) to fit the training data for the model(s) and the training workflow. In this example, 3 tasks are considered, 2 of which use a [`MaskedBCEWithLogitsLoss`](torch_ecg/models/loss.py) function, hence the `Dataset` produces an extra tensor for these 2 tasks
```python
def __getitem__(self, index:int) -> Tuple[np.ndarray, ...]:
    if self.lazy:
        if self.task in ["qrs_detection"]:
            return self.fdr[index][:2]
        else:
            return self.fdr[index]
    else:
        if self.task in ["qrs_detection"]:
            return self._all_data[index], self._all_labels[index]
        else:
            return self._all_data[index], self._all_labels[index], self._all_masks[index]
```
2. Inherit a [base model](torch_ecg/models/ecg_seq_lab_net.py) to create [task specific models](benchmarks/train_hybrid_cpsc2021/model.py), along with [tailored model configs](benchmarks/train_hybrid_cpsc2021/cfg.py)
3. Inherit the [`BaseTrainer`](torch_ecg/utils/trainer.py) to build the [training pipeline](benchmarks/train_hybrid_cpsc2021/trainer.py), with the `abstractmethod`s (`_setup_dataloaders`, `run_one_step`, `evaluate`, `batch_dim`, etc.) implemented.

## Thanks
Much is learned, especially the modular design, from the adversarial NLP library [`TextAttack`](https://github.com/QData/TextAttack) and from Hugging Face [`transformers`](https://github.com/huggingface/transformers).
