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
This technique is sometimes used by researchers and in challenges (e.g. the authors' rank 3 solutions to the problem of supraventricular premature beats (SPB) detection in CPSC2020), but its efficacy still needs proving and should not be used when preprocessing contains band-pass filtering and/or detrending. The following is an example from the record `A01` of `CPSC2020` (`sampfrom = 0, sampto = 4000`), with default configs of this augmenter.

![aug_bl](/images/aug_bl.jpg)

## cutmix
CutMix inserts a part of one sample to another for data distribution extrapolation. This technique has proven quite useful in the CPSC2021 rank 1 solution.

## mixup
Mixup uses convex linear combination of two samples for extrapolating the data distribution. The coefficient is randomly generated from some Beta distribution.

## random flip
Random Flip, using which the values of the ECGs are multiplied by -1 with a certain probability. This technique also has to be treated carefully whose usage might should be restricted to ambulatory ECGs, since for standard-leads ECGs, altering in relative or absolute values in different leads might completely change its interpretation, for example, the electrical axis.

## random masking
This technique randomly masks a small proportion of the signal with zeros, as similar to the Cutout technique in computer vision. This can also be done randomly at critical points, e.g. randomly masking R peaks helps reduce the probability of CNN models to misclassify sinus arrhythmia to atrial fibrillation.

## random renormalize
This method renormalizes the ECG signal to some random mean and standard deviation. One can find literature using this method for ECG augmentation. 

## stretch-or-compress
This augmentation method scales the ECG in the time axis, and has proven its validity in some literature

## label smooth
Strictly speaking, label smoothing is not a data augmentation method, but a technique to prevent the model from overconfidence thus improving its capability of generalization. This technique generates soft labels from hard labels.
