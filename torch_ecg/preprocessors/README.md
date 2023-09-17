# Preprocessors

Also [preprocessors](/torch_ecg/_preprocessors) acting on `numpy` `array`s.

## Basic Usage

`Preprocessor`s do the job of ECG signal preprocessing before fed into some neural network and are monitored by a manager. A short example is as follows

```python
import torch
from torch_ecg.cfg import CFG
from torch_ecg.preprocessors import PreprocManager


config = CFG(
    random=False,
    bandpass={"fs":500},
    normalize={"method": "min-max"},
)
ppm = PreprocManager.from_config(config)
sig = torch.randn(2, 12, 8000)
sig = ppm(sig)
```

For `Preprocessor`s that operate on the `numpy` `array`s, see the following example

```python
import torch
from torch_ecg.cfg import CFG
from torch_ecg._preprocessors import PreprocManager


config = CFG(
    random=False,
    resample={"fs": 500},
    bandpass={"filter_type": "fir"},
    normalize={"method": "min-max"},
)
ppm = PreprocManager.from_config(config)
sig = torch.randn(12, 80000).numpy()
sig, fs = ppm(sig, 200)
```

## Custom Preprocessors

One can create custom preprocessors to be maintained by the manager. The following is a simple example

```python
import torch
from torch_ecg.cfg import CFG
from torch_ecg.preprocessors import PreprocManager

class DummyPreProcessor(torch.nn.Module):
    """
    a dummy preprocessor that does nothing, similar to `torch.nn.Identity`
    """
    __name__ = "DummyPreProcessor"
    def forward(self, sig:torch.Tensor) -> torch.Tensor:
        """
        """
        return sig

config = CFG(
    random=False,
    bandpass={"fs":500},
    normalize={"method": "min-max"},
)
ppm = PreprocManager.from_config(config)

dp = DummyPreProcessor()
ppm.add_(dp, pos=1)

sig = torch.randn(2, 12, 8000)
sig = ppm(sig)
```

Here is another example for `numpy` version custom preprocessors

```python
from numbers import Real
from typing import Tuple

import numpy as np
import torch
from torch_ecg.cfg import CFG

from torch_ecg._preprocessors import PreprocManager, PreProcessor


class DummyPreProcessor(PreProcessor):
    """
    a dummy preprocessor that does nothing
    """
    __name__ = "DummyPreProcessor"
    def apply(self, sig:np.ndarray, fs:Real) -> Tuple[np.ndarray, int]:
        """
        """
        return sig, fs


config = CFG(
    random=False,
    resample={"fs": 500},
    bandpass={"filter_type": "fir"},
    normalize={"method": "min-max"},
)
ppm = PreprocManager.from_config(config)

dp = DummyPreProcessor()
ppm.add_(dp, pos=1)

sig = torch.randn(12, 80000).numpy()
sig, fs = ppm(sig, 200)
```

The following preprocessors are implemented

1. [baseline removal (detrend)](#baseline-removal)
2. [normalize (z-score, min-max, naïve)](#normalize)
3. [bandpass](#bandpass)
4. [resample](#resample)

## baseline removal

Also known as detrending, via median filter, which removes baseline drifts.

## normalize

Normalization methods including z-score normalization, min-max normalization, etc. are implemented. Refer to [wikipedia](https://en.wikipedia.org/wiki/Feature_scaling#Methods) for more details.

## bandpass

This procedure is performed using finite impulse response (FIR) filters, Butterworth filters, etc., which removes noises of frequencies outside the given pass band.

## resample

Resampling to a fixed sampling rate is done using the `interpolate` function from `torch.nn.functional` or using `resample_poly` function from `scipy.signal`.

## Issues
