# [torch_ecg](https://github.com/DeepPSP/torch_ecg/)

Deep learning ecg models implemented using PyTorch.

The system design is depicted as follows

<!-- ![system_design](/images/system_design.jpg) -->
<p align="middle">
  <img src="/images/system_design.jpg" width="80%" />
</p>

<!-- toc -->

- [Installation](#installation)
- [Main Modules](#main-modules)
    - [Augmenters](#augmenters)
    - [Preprocessors](#preprocessors)
    - [Databases](#databases)
    - [Implemented Neural Network Architectures](#implemented-neural-network-architectures)
    - [CNN Backbones](#cnn-backbones)
- [Other Useful Tools](#other-useful-tools)
- [Usage Examples](#usage-examples)

<!-- tocstop -->

## Installation
`torch_ecg` requires Python 3.6+ and is available through pip:
```bash
pip install torch-ecg
```
One can download the development version hosted at [GitHub](https://github.com/DeepPSP/torch_ecg/) via
```bash
git clone https://github.com/DeepPSP/torch_ecg.git
cd torch_ecg
pip install .
```

## Main Modules

### [Augmenters](/torch_ecg/augmenters)
Augmenters are classes (subclasses of `torch` `Module`) that perform data augmentation in a uniform way and are managed by the [`AugmenterManager`](/torch_ecg/augmenters/augmenter_manager.py) (also a subclass of `torch` `Module`). Augmenters and the manager share a common signature of the `formward` method:
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

Augmenters can be stochastic along the batch dimension and (or) the channel dimension (ref. the `get_indices` method of the [`Augmenter`](/torch_ecg/augmenters/base.py) base class).

### [Preprocessors](/torch_ecg/preprocessors)
Also [preprecessors](/torch_ecg/_preprocessors) acting on `numpy` `array`s. Similarly, preprocessors are monitored by a manager
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

### [Databases](/torch_ecg/databases)
This module include classes that manipulate the io of the ECG signals and labels in an ECG database, and maintains metadata (statistics, paths, plots, list of records, etc.) of it. This module is migrated and improved from [DeepPSP/database_reader](https://github.com/DeepPSP/database_reader)

After migration, all should be tested again, the progression:

| Database      | Source                                                           | Tested             |
| ------------- | ---------------------------------------------------------------- | ------------------ |
| AFDB          | [PhysioNet](https://physionet.org/content/afdb/1.0.0/)           | :heavy_check_mark: |
| ApneaECG      | [PhysioNet](https://physionet.org/content/apnea-ecg/1.0.0/)      | :x:                |
| CinC2017      | [PhysioNet](https://physionet.org/content/challenge-2017/1.0.0/) | :x:                |
| CinC2018      | [PhysioNet](https://physionet.org/content/challenge-2018/1.0.0/) | :x:                |
| CinC2020      | [PhysioNet](https://physionet.org/content/challenge-2020/1.0.1/) | :heavy_check_mark: |
| CinC2021      | [PhysioNet](https://physionet.org/content/challenge-2021/1.0.2/) | :heavy_check_mark: |
| LTAFDB        | [PhysioNet](https://physionet.org/content/ltafdb/1.0.0/)         | :x:                |
| LUDB          | [PhysioNet](https://physionet.org/content/ludb/1.0.1/)           | :heavy_check_mark: |
| MITDB         | [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)          | :x:                |
| SHHS          | [NSRR](https://sleepdata.org/datasets/shhs)                      | :x:                |
| CPSC2018      | [CPSC](http://2018.icbeb.org/Challenge.html)                     | :heavy_check_mark: |
| CPSC2019      | [CPSC](http://2019.icbeb.org/Challenge.html)                     | :heavy_check_mark: |
| CPSC2020      | [CPSC](http://2020.icbeb.org/CSPC2020)                           | :heavy_check_mark: |
| CPSC2021      | [CPSC](http://2021.icbeb.org/CPSC2021)                           | :heavy_check_mark: |

NOTE that these classes should not be confused with a `torch` `Dataset`, which is strongly related to the task (or the model). However, one can build `Dataset`s based on these classes, for example the [`Dataset`](/benchmarks/train_hybrid_cpsc2021/dataset.py) for the The 4th China Physiological Signal Challenge 2021 (CPSC2021)

### [Implemented Neural Network Architectures](/torch_ecg/models)
1. CRNN, both for classification and sequence tagging (segmentation)
2. U-Net
3. RR-LSTM

A typical signature of the instantiation (`__init__`) function of a model is as follows
```python
__init__(self, classes:Sequence[str], n_leads:int, config:Optional[CFG]=None, **kwargs:Any) -> NoReturn
```
if a `config` is not specified, then the default config will be used (stored in the [`model_configs`](/torch_ecg/model_configs) module).

A quick example is as follows:
```python
import torch
from torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths
from torch_ecg.model_configs import ECG_CRNN_CONFIG
from torch_ecg.models.ecg_crnn import ECG_CRNN

config = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG, fs=400)
config.cnn.name="resnet_nature_comm_bottle_neck_gc"  # bottleneck with global context attention variant of Nature Communications ResNet

classes = ["NSR", "AF", "PVC", "SPB"]
n_leads = 12
model = ECG_CRNN(classes, n_leads, config)

model(torch.rand(2, 12, 4000))  # signal length 4000, batch size 2
```
One can check the size of a model, in terms of the number of parameters via
```python
model.module_size
```
or in terms of memory consumption via
```python
model.module_size_
```

### [CNN Backbones](/torch_ecg/models/cnn)
#### Implemented
1. VGG
2. ResNet (including vanilla ResNet, ResNet-B, ResNet-C, ResNet-D, ResNeXT, TResNet, [Stanford ResNet](https://github.com/awni/ecg), [Nature Communications ResNet](https://github.com/antonior92/automatic-ecg-diagnosis), etc.)
3. MultiScopicNet (CPSC2019 SOTA)
4. DenseNet (CPSC2020 SOTA)
5. Xception

In general, variants of ResNet are the most commonly used architectures, as can be inferred from [CinC2020](https://cinc.org/archives/2020/) and [CinC2021](https://cinc.org/archives/2021/).

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

More details and a list of references can be found in the [README file](/torch_ecg/models/cnn/README.md) of this module.

## Other Useful Tools
### [Loggers](/torch_ecg/utils/loggers.py)
Loggers including
1. CSV logger
2. text logger
3. tensorboard logger
are implemented and manipulated uniformly by a manager.

### [R peaks detection algorithms](/torch_ecg/utils/rpeaks.py)
This is a collection of traditional (non deep learning) algorithms for R peaks detection collected from [WFDB](https://github.com/MIT-LCP/wfdb-python) and [BioSPPy](https://github.com/PIA-Group/BioSPPy).

### [Trainer](/torch_ecg/utils/trainer.py)
An abstract base class `BaseTrainer` is implemented, in which some common steps in building a training pipeline (workflow) are impemented. A few task specific methods are assigned as `abstractmethod`s, for example the method
```python
evaluate(self, data_loader:DataLoader) -> Dict[str, float]
```
for evaluation on the validation set during training and perhaps further for model selection and early stopping.

## Usage Examples
See case studies in the [benchmarks folder](/benchmarks/).

a large part of the case studies are migrated from other DeepPSP repositories, some are implemented in the old fasion, being inconsistent with the new system architecture of `torch_ecg`, hence need updating and testing

| Benchmark  | Architecture              | Source                                                  | Finished           | Updated            | Tested             |
| ---------- | ------------------------- | ------------------------------------------------------- | ------------------ | ------------------ | ------------------ |
| CinC2020   | CRNN                      | [DeepPSP/cinc2020](https://github.com/DeepPSP/cinc2020) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| CinC2021   | CRNN                      | [DeepPSP/cinc2021](https://github.com/DeepPSP/cinc2021) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| CPSC2019   | SequenceTagging/U-Net     | NA                                                      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| CPSC2020   | CRNN/SequenceTagging      | [DeepPSP/cpsc2020](https://github.com/DeepPSP/cpsc2020) | :heavy_check_mark: | :x:                | :x:                |
| CPSC2021   | CRNN/SequenceTagging/LSTM | [DeepPSP/cpsc2021](https://github.com/DeepPSP/cpsc2021) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| LUDB       | U-Net                     | NA                                                      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |


Taking [CPSC2021](/benchmarks/train_hybrid_cpsc2021) for example, the steps are
1. Write a [`Dataset`](/benchmarks/train_hybrid_cpsc2021/dataset.py) to fit the training data for the model(s) and the training workflow. In this example, 3 tasks are considered, 2 of which use a [`MaskedBCEWithLogitsLoss`](/torch_ecg/models/loss.py) function, hence the `Dataset` produces an extra tensor for these 2 tasks
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
2. Inherit a [base model](/torch_ecg/models/ecg_seq_lab_net.py) to create [task specific models](/benchmarks/train_hybrid_cpsc2021/model.py), along with [tailored model configs](/benchmarks/train_hybrid_cpsc2021/cfg.py)
3. Inherit the [`BaseTrainer`](/torch_ecg/utils/trainer.py) to build the [training pipeline](/benchmarks/train_hybrid_cpsc2021/trainer.py), with the `abstractmethod`s (`_setup_dataloaders`, `run_one_step`, `evaluate`, `batch_dim`, etc.) implemented.

## Work in progress
See the [projects page](https://github.com/DeepPSP/torch_ecg/projects).

## Thanks
Much is learned, especially the modular design, from the adversarial NLP library [`TextAttack`](https://github.com/QData/TextAttack) and from Hugging Face [`transformers`](https://github.com/huggingface/transformers).
