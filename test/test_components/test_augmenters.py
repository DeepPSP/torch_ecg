"""
"""

import numpy as np
import torch
import pytest

from torch_ecg.cfg import CFG
from torch_ecg.augmenters import (
    Augmenter,
    BaselineWanderAugmenter,
    LabelSmooth,
    Mixup,
    RandomFlip,
    RandomMasking,
    RandomRenormalize,
    StretchCompress,
    StretchCompressOffline,
    AugmenterManager,
)


def test_base_augmenter():
    with pytest.raises(
        TypeError,
        match=f"Can't instantiate abstract class {Augmenter.__name__} with abstract method",
    ):
        aug = Augmenter()


def test_augmenter_manager():
    # all use default config
    config = CFG(
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

    sig = torch.randn(32, 12, 5000)
    label = torch.randint(0, 2, (32, 26), dtype=torch.float32)
    mask1 = torch.randint(0, 2, (32, 5000, 3), dtype=torch.float32)
    mask2 = torch.randint(0, 3, (32, 5000), dtype=torch.long)
    sig, label, mask1, mask2 = am(sig, label, mask1, mask2)


def test_baseline_wander_augmenter():
    blw = BaselineWanderAugmenter(300, prob=0.7)
    sig = torch.randn(32, 12, 5000)
    label = torch.randint(0, 2, (32, 26), dtype=torch.float32)
    sig, _ = blw(sig, label)


def test_cutmix_augmenter():
    pass


def test_label_smooth():
    ls = LabelSmooth()
    label = torch.randint(0, 2, (32, 26), dtype=torch.float32)
    _, label = ls(None, label)


def test_mixup():
    mixup = Mixup()
    sig = torch.randn(32, 12, 5000)
    label = torch.randint(0, 2, (32, 26), dtype=torch.float32)
    sig, label = mixup(sig, label)


def test_random_flip():
    rf = RandomFlip()
    sig = torch.randn(32, 12, 5000)
    sig, _ = rf(sig, None)


def test_random_masking():
    rm = RandomMasking(fs=500, prob=0.7)
    sig = torch.randn(32, 12, 5000)
    critical_points = [np.arange(250, 5000 - 250, step=400) for _ in range(32)]
    sig, _ = rm(sig, None, critical_points=critical_points)


def test_random_renormalize():
    rrn = RandomRenormalize()
    sig = torch.randn(32, 12, 5000)
    sig, _ = rrn(sig, None)


def test_stretch_compress():
    sc = StretchCompress()
    sig = torch.randn((32, 12, 5000))
    labels = torch.randint(0, 2, (32, 5000, 26))
    label = torch.randint(0, 2, (32, 26), dtype=torch.float32)
    mask = torch.randint(0, 2, (32, 5000, 3), dtype=torch.float32)
    sig, label, mask = sc(sig, label, mask)


def test_stretch_compress_offline():
    sco = StretchCompressOffline()
    seglen = 600
    sig = torch.randn((12, 60000)).numpy()
    labels = torch.ones((60000, 3)).numpy().astype(int)
    masks = torch.ones((60000, 1)).numpy().astype(int)
    segments = sco(600, sig, labels, masks, critical_points=[10000, 30000])
