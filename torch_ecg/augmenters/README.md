# Augmenters

Augmenters are classes (subclasses of `torch` `Module`) that perform data augmentation in a uniform way and are managed by the [`AugmenterManager`](/torch_ecg/augmenters/augmenter_manager.py) (also a subclass of `torch` `Module`). Augmenters and the manager share a common signature of the `formward` method:
```python
forward(self, sig:Tensor, label:Optional[Tensor]=None, *extra_tensors:Sequence[Tensor], **kwargs:Any) -> Tuple[Tensor, ...]:
```

The following augmenters are implemented:
1. [baseline wander (adding sinusoidal and gaussian noises)](#baseline-wander)
2. [cutmix](#cutmix)
3. [mixup](#mixup)
4. [random flip](#random-flip)
5. [random masking](#random-masking)
6. [random renormalize](#random-renormalize)
7. [stretch-or-compress (scaling)](#stretch-or-compress)
8. [label smooth (not actually for data augmentation, but has simimlar behavior)](#label-smooth)

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

## baseline wander
to write

## cutmix
to write

## mixup
to write

## random flip
to write

## random masking
to write

## random renormalize
to write

## stretch-or-compress
to write

## label smooth
to write
