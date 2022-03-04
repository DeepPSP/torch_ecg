# Preprocessors

Also [preprocessors](/torch_ecg/_preprocessors) acting on `numpy` `array`s. 

`Preprocessor`s do the job of ECG signal preprocessing before fed into some neural network and are monitored by a manager. A short example is as follows
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
1. [baseline removal (detrend)](#baseline-removal)
2. [normalize (z-score, min-max, naïve)](#normalize)
3. [bandpass](#bandpass)
4. [resample](#resample)


## baseline removal
to write

## normalize
to write

## bandpass
to write

## resample
to write

## Issues

