"""Tests for RNG utilities and reproducibility of the augmenters."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from torch_ecg.cfg import DEFAULTS
from torch_ecg.utils import seed_everything, worker_init_fn


def test_seed_everything():
    seed_everything(2026)
    rng_vals = DEFAULTS.RNG.random(8)
    np_vals = np.random.random(8)
    random_vals = [random.random() for _ in range(8)]
    torch_vals = torch.rand(8)

    seed_everything(2026)
    assert np.array_equal(DEFAULTS.RNG.random(8), rng_vals)
    assert np.array_equal(np.random.random(8), np_vals)
    assert [random.random() for _ in range(8)] == random_vals
    assert torch.equal(torch.rand(8), torch_vals)

    seed_everything(2027)
    assert not np.array_equal(DEFAULTS.RNG.random(8), rng_vals)


class _RNGDataset(Dataset):
    """Each item draws one number from `DEFAULTS.RNG`."""

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return np.array([DEFAULTS.RNG.random()])


def _collect_loader_values(num_workers: int) -> np.ndarray:
    # the workers' seeds derive from the data loader's base seed, which is
    # drawn from the global torch RNG, hence reseed it before each run
    torch.manual_seed(123)
    dl = DataLoader(
        _RNGDataset(),
        batch_size=2,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )
    return np.concatenate([batch.numpy().ravel() for batch in dl])


def test_worker_init_fn_distinct_streams():
    """Regression: without reseeding, all forked workers inherit a copy of
    the parent's `DEFAULTS.RNG` and produce identical random streams."""
    values = _collect_loader_values(num_workers=2)
    assert len(values) == 8
    # all draws must be distinct; with the bug, workers 0 and 1 produced
    # exactly the same four numbers each
    assert len(set(values.tolist())) == 8


def test_worker_init_fn_reproducible():
    vals_1 = _collect_loader_values(num_workers=2)
    vals_2 = _collect_loader_values(num_workers=2)
    assert np.array_equal(np.sort(vals_1), np.sort(vals_2))


def test_augmenter_manager_deterministic():
    """Regression: augmenters used to draw from the Python `random` module,
    which `DEFAULTS.set_seed` does control, but only incidentally; now all
    augmenters draw from `DEFAULTS.RNG`, so `seed_everything` fully
    determines the augmentation."""
    from torch_ecg.augmenters import AugmenterManager

    config = {
        "random": True,
        "fs": 500,
        "mixup": {},
        "random_masking": {},
        "random_renormalize": {},
        "label_smooth": {},
    }

    def _run() -> tuple:
        seed_everything(42)
        am = AugmenterManager.from_config(config)
        sig = torch.rand(4, 12, 5000)
        label = torch.rand(4, 26)
        sig, label, *_ = am(sig, label)
        return sig, label

    sig_1, label_1 = _run()
    sig_2, label_2 = _run()
    assert torch.equal(sig_1, sig_2)
    assert torch.equal(label_1, label_2)
