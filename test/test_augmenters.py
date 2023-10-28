"""
"""

import re

import numpy as np
import pytest
import torch

from torch_ecg.augmenters import (
    Augmenter,
    AugmenterManager,
    BaselineWanderAugmenter,
    CutMix,
    LabelSmooth,
    Mixup,
    RandomFlip,
    RandomMasking,
    RandomRenormalize,
    StretchCompress,
    StretchCompressOffline,
)
from torch_ecg.augmenters.baseline_wander import _gen_baseline_wander
from torch_ecg.cfg import CFG

SIG_LEN = 2000
BATCH_SIZE = 2
N_LEADS = 12


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

    sig = torch.randn(BATCH_SIZE, N_LEADS, SIG_LEN)
    label = torch.randint(0, 2, (BATCH_SIZE, 26), dtype=torch.float32)
    mask1 = torch.randint(0, 2, (BATCH_SIZE, SIG_LEN, 3), dtype=torch.float32)
    mask2 = torch.randint(0, 3, (BATCH_SIZE, SIG_LEN), dtype=torch.long)
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

    with pytest.warns(RuntimeWarning, match="The augmenters are applied in random order"):
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
    sig = torch.randn(BATCH_SIZE, N_LEADS, SIG_LEN)
    label = torch.randint(0, 2, (BATCH_SIZE, 26), dtype=torch.float32)
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
    sig = torch.randn(BATCH_SIZE, N_LEADS, SIG_LEN)
    label = torch.randint(0, 2, (BATCH_SIZE, 26), dtype=torch.float32)
    mask = torch.randint(0, 2, (BATCH_SIZE, SIG_LEN, 3), dtype=torch.float32)
    label = torch.randint(0, 2, (BATCH_SIZE, 26), dtype=torch.float32)

    cm = CutMix(fs=500, prob=0.7, beta=0.6)
    sig, label, mask = cm(sig, label, mask)
    assert sig.shape == (BATCH_SIZE, N_LEADS, SIG_LEN)
    assert label.shape == (BATCH_SIZE, 26)
    assert mask.shape == (BATCH_SIZE, SIG_LEN, 3)

    sig = torch.randn(BATCH_SIZE, N_LEADS, SIG_LEN)
    label = torch.randint(0, 2, (BATCH_SIZE, 26), dtype=torch.float32)
    mask = torch.randint(0, 2, (BATCH_SIZE, SIG_LEN, 3), dtype=torch.float32)
    label = torch.randint(0, 2, (BATCH_SIZE, 26), dtype=torch.float32)

    cm = CutMix(fs=500, prob=0.4, inplace=False, num_mix=2)
    new_sig, new_label, new_mask = cm(sig, label, mask)
    assert new_sig.shape == (BATCH_SIZE, N_LEADS, SIG_LEN)
    assert new_label.shape == (BATCH_SIZE, 26)
    assert new_mask.shape == (BATCH_SIZE, SIG_LEN, 3)
    new_sig, new_mask = cm(sig, mask)
    assert new_sig.shape == (BATCH_SIZE, N_LEADS, SIG_LEN)
    assert new_mask.shape == (BATCH_SIZE, SIG_LEN, 3)

    assert str(cm) == repr(cm)

    with pytest.raises(AssertionError, match="`label` should NOT be categorical labels"):
        label = torch.randint(0, 26, (BATCH_SIZE,), dtype=torch.long)
        cm(sig, label)

    with pytest.raises(AssertionError, match="`num_mix` must be a positive integer, but got `.+`"):
        cm = CutMix(fs=500, num_mix=0)
    with pytest.raises(AssertionError, match="Probability must be between 0 and 1"):
        cm = CutMix(fs=500, prob=1.1)
    with pytest.raises(
        AssertionError,
        match="`alpha` and `beta` must be positive, but got `.+` and `.+`",
    ):
        cm = CutMix(fs=500, alpha=0, beta=0)


def test_label_smooth():
    ls = LabelSmooth(inplace=False)
    label = torch.randint(0, 2, (BATCH_SIZE, 26), dtype=torch.float32)
    _, label = ls(None, label)
    ls = LabelSmooth(smoothing=0.0)
    label = torch.randint(0, 2, (BATCH_SIZE, 26), dtype=torch.float32)
    _, label = ls(None, label)

    assert str(ls) == repr(ls)


def test_mixup():
    mixup = Mixup()
    sig = torch.randn(BATCH_SIZE, N_LEADS, SIG_LEN)
    label = torch.randint(0, 2, (BATCH_SIZE, 26), dtype=torch.float32)
    sig, label = mixup(sig, label)
    mixup = Mixup(inplace=False)
    sig = torch.randn(BATCH_SIZE, N_LEADS, SIG_LEN)
    label = torch.randint(0, 2, (BATCH_SIZE, 26), dtype=torch.float32)
    sig, label = mixup(sig, label)

    assert str(mixup) == repr(mixup)


def test_random_flip():
    rf = RandomFlip()
    sig = torch.randn(BATCH_SIZE, N_LEADS, SIG_LEN)
    sig, _ = rf(sig, None)
    rf = RandomFlip(inplace=False, per_channel=False)
    sig = torch.randn(BATCH_SIZE, N_LEADS, SIG_LEN)
    sig, _ = rf(sig, None)
    rf = RandomFlip(prob=0.0, per_channel=False)
    sig = torch.randn(BATCH_SIZE, N_LEADS, SIG_LEN)
    sig, _ = rf(sig, None)

    assert str(rf) == repr(rf)


def test_random_masking():
    rm = RandomMasking(fs=500, prob=0.7)
    sig = torch.randn(BATCH_SIZE, N_LEADS, SIG_LEN)
    critical_points = [np.arange(250, SIG_LEN - 250, step=400) for _ in range(BATCH_SIZE)]
    sig, _ = rm(sig, None, critical_points=critical_points)
    rm = RandomMasking(fs=500, prob=0.3, inplace=False)
    sig = torch.randn(BATCH_SIZE, N_LEADS, SIG_LEN)
    sig, _ = rm(sig, None, critical_points=critical_points)
    rm = RandomMasking(fs=500, prob=0.0)
    sig = torch.randn(BATCH_SIZE, N_LEADS, SIG_LEN)
    sig, _ = rm(sig, None, critical_points=critical_points)

    assert str(rm) == repr(rm)


def test_random_renormalize():
    rrn = RandomRenormalize(per_channel=True, prob=0.7)
    sig = torch.randn(BATCH_SIZE, N_LEADS, SIG_LEN)
    sig, _ = rrn(sig, None)

    # TODO: fix errors in the following tests
    # rrn = RandomRenormalize(mean=np.zeros((N_LEADS, 1)), std=np.ones((N_LEADS, 1)), per_channel=True)
    # sig = torch.randn(BATCH_SIZE, N_LEADS, SIG_LEN)
    # sig, _ = rrn(sig, None)

    rrn = RandomRenormalize(mean=np.zeros((N_LEADS,)), std=np.ones((N_LEADS,)), inplace=False)
    sig = torch.randn(BATCH_SIZE, N_LEADS, SIG_LEN)
    sig, _ = rrn(sig, None)

    rrn = RandomRenormalize(prob=0.0)
    sig = torch.randn(BATCH_SIZE, N_LEADS, SIG_LEN)
    sig, _ = rrn(sig, None)

    assert str(rrn) == repr(rrn)


def test_stretch_compress():
    sc = StretchCompress(inplace=False)
    sig = torch.randn((BATCH_SIZE, N_LEADS, SIG_LEN))
    # labels = torch.randint(0, 2, (BATCH_SIZE, SIG_LEN, 26))
    label = torch.randint(0, 2, (BATCH_SIZE, 26), dtype=torch.float32)
    mask = torch.randint(0, 2, (BATCH_SIZE, SIG_LEN, 3), dtype=torch.float32)
    sig, label, mask = sc(sig, label, mask)
    assert sig.shape == (BATCH_SIZE, N_LEADS, SIG_LEN)
    assert label.shape == (BATCH_SIZE, 26)
    assert mask.shape == (BATCH_SIZE, SIG_LEN, 3)
    label = torch.randint(0, 2, (BATCH_SIZE, SIG_LEN // 2, 26), dtype=torch.float32)
    assert label.shape == (BATCH_SIZE, SIG_LEN // 2, 26)

    sig = torch.randn((BATCH_SIZE, N_LEADS, SIG_LEN))
    # labels = torch.randint(0, 2, (BATCH_SIZE, SIG_LEN, 26))
    label = torch.randint(0, 2, (BATCH_SIZE, 26), dtype=torch.float32)
    mask = torch.randint(0, 2, (BATCH_SIZE, SIG_LEN // 8, 3), dtype=torch.float32)
    for _ in range(5):
        # generate 5 times
        sig, label, mask = sc._generate(sig, label, mask)
    # generate with only sig
    sig = sc._generate(sig)

    sc = StretchCompress(prob=0.0)
    sig = torch.randn((BATCH_SIZE, N_LEADS, SIG_LEN))
    # labels = torch.randint(0, 2, (BATCH_SIZE, SIG_LEN, 26))
    label = torch.randint(0, 2, (BATCH_SIZE, 26), dtype=torch.float32)
    mask = torch.randint(0, 2, (BATCH_SIZE, SIG_LEN // 8, 3), dtype=torch.float32)
    sig, label, mask = sc(sig, label, mask)
    assert sig.shape == (BATCH_SIZE, N_LEADS, SIG_LEN)
    assert label.shape == (BATCH_SIZE, 26)
    assert mask.shape == (BATCH_SIZE, SIG_LEN // 8, 3)

    sig = torch.randn((BATCH_SIZE, N_LEADS, SIG_LEN))
    # labels = torch.randint(0, 2, (BATCH_SIZE, SIG_LEN, 26))
    label = torch.randint(0, 2, (BATCH_SIZE, 26), dtype=torch.float32)
    mask = torch.randint(0, 2, (BATCH_SIZE, SIG_LEN // 8, 3), dtype=torch.float32)
    for _ in range(5):
        # generate 5 times
        sig, label, mask = sc._generate(sig, label, mask)
    # generate with only sig
    sig = sc._generate(sig)

    assert str(sc) == repr(sc)


def test_stretch_compress_offline():
    sco = StretchCompressOffline()
    seglen = 600
    sig = torch.randn((N_LEADS, 60000)).numpy()
    labels = torch.ones((60000, 3)).numpy().astype(int)
    masks = torch.ones((60000, 1)).numpy().astype(int)
    segments = sco(seglen, sig, labels, masks, critical_points=[10000, 30000])

    sig = torch.randn((N_LEADS, 60)).numpy()
    labels = torch.ones((60, 3)).numpy().astype(int)
    masks = torch.ones((60, 1)).numpy().astype(int)
    segments = sco(seglen, sig, labels, masks)

    sig = torch.randn((N_LEADS, 800)).numpy()
    labels = torch.ones((800, 3)).numpy().astype(int)
    masks = torch.ones((800, 1)).numpy().astype(int)
    segments = sco(seglen, sig, labels, masks)

    assert str(sco) == repr(sco)
