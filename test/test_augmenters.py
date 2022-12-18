"""
"""

import re

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
from torch_ecg.augmenters.baseline_wander import _gen_baseline_wander


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

    am.rearrange(
        new_ordering=[
            "stretch_compress",
            "random_masking",
            "baseline_wander",
            "random_renormalize",
            "random_flip",
            "label_smooth",
            "mixup",
        ]
    )

    am.random = True
    sig, label, mask1, mask2 = am(sig, label, mask1, mask2)

    with pytest.warns(
        RuntimeWarning, match="The augmenters are applied in random order"
    ):
        am.random = True
        am.rearrange(
            new_ordering=[
                "mixup",
                "random_masking",
                "random_flip",
                "baseline_wander",
                "random_renormalize",
                "label_smooth",
                "stretch_compress",
            ]
        )
    am.random = False

    with pytest.raises(AssertionError, match="Duplicate augmenter names"):
        am.rearrange(
            new_ordering=[
                "stretch_compress",
                "random_masking",
                "baseline_wander",
                "random_renormalize",
                "random_flip",
                "label_smooth",
                "mixup",
                "random_masking",
            ]
        )

    with pytest.raises(AssertionError, match="Number of augmenters mismatch"):
        am.rearrange(
            new_ordering=[
                "stretch_compress",
                "random_masking",
                "baseline_wander",
                "random_renormalize",
                "random_flip",
                "label_smooth",
            ]
        )

    with pytest.raises(AssertionError, match="Unknown augmenter name: `.+`"):
        am.rearrange(
            new_ordering=[
                "stretch_compress",
                "random_masking",
                "baseline_wander",
                "random_normalize",  # typo
                "random_flip",
                "label_smooth",
                "mixup",
            ]
        )

    assert re.search("augmenters = \\[", repr(am))


def test_baseline_wander_augmenter():
    blw = BaselineWanderAugmenter(300, prob=0.7, inplace=False)
    sig = torch.randn(32, 12, 5000)
    label = torch.randint(0, 2, (32, 26), dtype=torch.float32)
    sig, _ = blw(sig, label)

    assert str(blw) == repr(blw)

    noise = _gen_baseline_wander(
        siglen=sig.shape[-1],
        fs=blw.fs,
        bw_fs=blw.bw_fs,
        amplitude=blw.ampl_ratio[1],
        amplitude_gaussian=blw.gaussian[1],
    )
    noise = _gen_baseline_wander(
        siglen=sig.shape[-1],
        fs=blw.fs,
        bw_fs=1.5,
        amplitude=0.05,
        amplitude_gaussian=blw.gaussian[1],
    )


def test_cutmix_augmenter():
    pass


def test_label_smooth():
    ls = LabelSmooth(inplace=False)
    label = torch.randint(0, 2, (32, 26), dtype=torch.float32)
    _, label = ls(None, label)
    ls = LabelSmooth(smoothing=0.0)
    label = torch.randint(0, 2, (32, 26), dtype=torch.float32)
    _, label = ls(None, label)

    assert str(ls) == repr(ls)


def test_mixup():
    mixup = Mixup()
    sig = torch.randn(32, 12, 5000)
    label = torch.randint(0, 2, (32, 26), dtype=torch.float32)
    sig, label = mixup(sig, label)
    mixup = Mixup(inplace=False)
    sig = torch.randn(32, 12, 5000)
    label = torch.randint(0, 2, (32, 26), dtype=torch.float32)
    sig, label = mixup(sig, label)

    assert str(mixup) == repr(mixup)


def test_random_flip():
    rf = RandomFlip()
    sig = torch.randn(32, 12, 5000)
    sig, _ = rf(sig, None)
    rf = RandomFlip(inplace=False, per_channel=False)
    sig = torch.randn(32, 12, 5000)
    sig, _ = rf(sig, None)
    rf = RandomFlip(prob=0.0, per_channel=False)
    sig = torch.randn(32, 12, 5000)
    sig, _ = rf(sig, None)

    assert str(rf) == repr(rf)


def test_random_masking():
    rm = RandomMasking(fs=500, prob=0.7)
    sig = torch.randn(32, 12, 5000)
    critical_points = [np.arange(250, 5000 - 250, step=400) for _ in range(32)]
    sig, _ = rm(sig, None, critical_points=critical_points)
    rm = RandomMasking(fs=500, prob=0.3, inplace=False)
    sig = torch.randn(32, 12, 5000)
    sig, _ = rm(sig, None, critical_points=critical_points)
    rm = RandomMasking(fs=500, prob=0.0)
    sig = torch.randn(32, 12, 5000)
    sig, _ = rm(sig, None, critical_points=critical_points)

    assert str(rm) == repr(rm)


def test_random_renormalize():
    rrn = RandomRenormalize(per_channel=True, prob=0.7)
    sig = torch.randn(32, 12, 5000)
    sig, _ = rrn(sig, None)

    # TODO: fix errors in the following tests
    # rrn = RandomRenormalize(mean=np.zeros((12, 1)), std=np.ones((12, 1)), per_channel=True)
    # sig = torch.randn(32, 12, 5000)
    # sig, _ = rrn(sig, None)

    rrn = RandomRenormalize(mean=np.zeros((12,)), std=np.ones((12,)), inplace=False)
    sig = torch.randn(32, 12, 5000)
    sig, _ = rrn(sig, None)

    rrn = RandomRenormalize(prob=0.0)
    sig = torch.randn(32, 12, 5000)
    sig, _ = rrn(sig, None)

    assert str(rrn) == repr(rrn)


def test_stretch_compress():
    sc = StretchCompress(inplace=False)
    sig = torch.randn((32, 12, 5000))
    # labels = torch.randint(0, 2, (32, 5000, 26))
    label = torch.randint(0, 2, (32, 26), dtype=torch.float32)
    mask = torch.randint(0, 2, (32, 5000, 3), dtype=torch.float32)
    sig, label, mask = sc(sig, label, mask)
    assert sig.shape == (32, 12, 5000)
    assert label.shape == (32, 26)
    assert mask.shape == (32, 5000, 3)

    sig = torch.randn((32, 12, 5000))
    # labels = torch.randint(0, 2, (32, 5000, 26))
    label = torch.randint(0, 2, (32, 26), dtype=torch.float32)
    mask = torch.randint(0, 2, (32, 5000 // 8, 3), dtype=torch.float32)
    sig, label, mask = sc._generate(sig, label, mask)

    sc = StretchCompress(prob=0.0)
    sig = torch.randn((32, 12, 5000))
    # labels = torch.randint(0, 2, (32, 5000, 26))
    label = torch.randint(0, 2, (32, 26), dtype=torch.float32)
    mask = torch.randint(0, 2, (32, 5000 // 8, 3), dtype=torch.float32)
    sig, label, mask = sc(sig, label, mask)
    assert sig.shape == (32, 12, 5000)
    assert label.shape == (32, 26)
    assert mask.shape == (32, 5000 // 8, 3)

    sig = torch.randn((32, 12, 5000))
    # labels = torch.randint(0, 2, (32, 5000, 26))
    label = torch.randint(0, 2, (32, 26), dtype=torch.float32)
    mask = torch.randint(0, 2, (32, 5000 // 8, 3), dtype=torch.float32)
    sig, label, mask = sc._generate(sig, label, mask)

    assert str(sc) == repr(sc)


def test_stretch_compress_offline():
    sco = StretchCompressOffline()
    seglen = 600
    sig = torch.randn((12, 60000)).numpy()
    labels = torch.ones((60000, 3)).numpy().astype(int)
    masks = torch.ones((60000, 1)).numpy().astype(int)
    segments = sco(seglen, sig, labels, masks, critical_points=[10000, 30000])

    sig = torch.randn((12, 60)).numpy()
    labels = torch.ones((60, 3)).numpy().astype(int)
    masks = torch.ones((60, 1)).numpy().astype(int)
    segments = sco(seglen, sig, labels, masks)

    sig = torch.randn((12, 800)).numpy()
    labels = torch.ones((800, 3)).numpy().astype(int)
    masks = torch.ones((800, 1)).numpy().astype(int)
    segments = sco(seglen, sig, labels, masks)

    assert str(sco) == repr(sco)
