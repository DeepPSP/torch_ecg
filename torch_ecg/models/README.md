# Models for ECG tasks

![pytest](https://github.com/DeepPSP/torch_ecg/actions/workflows/test-models.yml/badge.svg?branch=dev-models)

[GitHub Action](https://github.com/DeepPSP/torch_ecg/actions/workflows/test-models.yml).

1. CRNN, both for classification and sequence tagging (segmentation)
2. U-Net
3. RR-LSTM

A typical signature of the instantiation (`__init__`) function of a model is as follows

```python
__init__(self, classes:Sequence[str], n_leads:int, config:Optional[CFG]=None, **kwargs:Any) -> None
```

if a `config` is not specified, then the default config will be used (stored in the [`model_configs`](torch_ecg/model_configs) module).

## Quick Example

A quick example is as follows:

```python
import torch
from torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths
from torch_ecg.model_configs import ECG_CRNN_CONFIG
from torch_ecg.models.ecg_crnn import ECG_CRNN

config = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG, fs=400)
# change the default CNN backbone
# bottleneck with global context attention variant of Nature Communications ResNet
config.cnn.name="resnet_nature_comm_bottle_neck_gc"

classes = ["NSR", "AF", "PVC", "SPB"]
n_leads = 12
model = ECG_CRNN(classes, n_leads, config)

model(torch.randn((2, 12, 4000)))  # signal length 4000, batch size 2
```

Then a model for the classification of 4 classes, namely "NSR", "AF", "PVC", "SPB", on 12-lead ECGs is created. One can check the size of a model, in terms of the number of parameters via

```python
model.module_size
```

or in terms of memory consumption via

```python
model.module_size_
```

## Custom Model

One can adjust the configs to create a custom model. For example, the building blocks of the 4 stages of a `TResNet` backbone are `basic`, `basic`, `bottleneck`, `bottleneck`. If one wants to change the second block to be a `bottleneck` block with sequeeze and excitation (`SE`) attention, then

```python
from copy import deepcopy

from torch_ecg.models.ecg_crnn import ECG_CRNN
from torch_ecg.model_configs import (
    ECG_CRNN_CONFIG,
    tresnetF, resnet_bottle_neck_se,
)

my_resnet = deepcopy(tresnetP)
my_resnet.building_block[1] = "bottleneck"
my_resnet.block[1] = resnet_bottle_neck_se
```

The convolutions in a `TResNet` are anti-aliasing convolutions, if one wants further to change the convolutions to normal convolutions, then

```python
for b in my_resnet.block:
    b.conv_type = None
```

or change them to separable convolutions via

```python
for b in my_resnet.block:
    b.conv_type = "separable"
```

Finally, replace the default CNN backbone via

```python
my_model_config = deepcopy(ECG_CRNN_CONFIG)
my_model_config.cnn.name = "my_resnet"
my_model_config.cnn.my_resnet = my_resnet

model = ECG_CRNN(["NSR", "AF", "PVC", "SPB"], 12, my_model_config)
```

## [CNN Backbones](cnn)

Details and a list of references can be found in the [README file](torch_ecg/models/cnn/README.md) of this module.
