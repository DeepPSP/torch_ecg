.. _ug_preprocess:

Preprocessing and augmentation
==============================

Preprocessors
-------------

Preprocessing (baseline removal, band-pass filtering) is expressed as
composable configs.  The high-level helpers apply a standard chain to
whole segments:

.. code-block:: python

    from torch_ecg._preprocessors import PreprocManager, preprocess_multi_lead_signal

    filtered = preprocess_multi_lead_signal(
        raw_sig,
        fs=500,
        bl_win=[0.2, 0.6],      # median-filter baseline windows (s)
        band_fs=[0.5, 45.0],    # band-pass (Hz)
        filter_type="fir",
    )

Each step can also be used standalone via its
:class:`~torch_ecg._preprocessors.base.PreProcessor` wrapper (function +
hyper-parameters), or composed explicitly through
:class:`~torch_ecg._preprocessors.PreprocManager`, which is how the
datasets apply them.

Augmenters
----------

Augmenters are ``nn.Module``\s operating on batched
``(batch, lead, time)`` signals **together with their labels** (some
augmenters, e.g. lead flips, permute labels too):

.. code-block:: python

    import torch
    from torch_ecg.augmenters import AugmenterManager, BaselineWanderAugmenter, RandomFlip, RandomRenormalize

    augs = [
        BaselineWanderAugmenter(fs=500, bw=0.5, fs_bw=100),
        RandomFlip(),
        RandomRenormalize(),
    ]
    am = AugmenterManager(*augs, random=False)
    aug_sig, aug_label = am(sig, label)     # (B, L, T), (B, C) in; same shapes out

With ``random=True`` the manager picks a random augmenter per batch
(useful for aggressive augmentation budgets).  The available augmenters
are listed in :mod:`torch_ecg.augmenters` (baseline wander, flips,
stretch-compress, cutout, noise, ...).

Determinism
-----------

Reproducible runs need exactly one seeding point at the entry of the
program plus per-worker reseeding for ``DataLoader`` workers:

.. code-block:: python

    from torch_ecg.utils import seed_everything, worker_init_fn
    from torch.utils.data import DataLoader

    seed_everything(42)        # python / numpy / torch / DEFAULTS.RNG, all at once
    dl = DataLoader(ds, batch_size=32, worker_init_fn=worker_init_fn, ...)

``worker_init_fn`` seeds each worker from the base seed and the worker
id, so augmentation inside workers is reproducible across runs.  This is
the pattern every benchmark uses; with a fixed seed, a run is fully
determined.
