"""
==========================================
Online data augmentation
==========================================

This example visualizes the effect of three waveform augmenters -- baseline
wander, random masking, and stretch-compress -- on one record from the CPSC
2018 sample data.

Augmenters operate on batches of shape ``(batch, n_leads, n_samples)`` together
with their labels.  They are assembled into an
:class:`~torch_ecg.augmenters.AugmenterManager` from a configuration mapping,
mirroring the ``augmenter`` section of a training configuration.  Here each
augmenter is applied in isolation to make its individual effect visible;
during training they are composed in one manager and applied to every batch.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter

import torch_ecg
from torch_ecg.augmenters import AugmenterManager
from torch_ecg.cfg import CFG
from torch_ecg.databases import CPSC2018
from torch_ecg.utils.misc import plot_single_lead
from torch_ecg.utils.rng import seed_everything

# select the headless backend for deterministic rendering on CI; switching
# after the pyplot import is fine as long as no figure exists yet
matplotlib.use("Agg")

# the sample data ships with the repository checkout; the editable install
# used to build these docs places the package inside the checkout, so the
# package root is the repository root (sphinx-gallery executes examples
# without `__file__`, hence this indirection)
REPO_ROOT = Path(torch_ecg.__file__).resolve().parents[1]

seed_everything(42)

reader = CPSC2018(str(REPO_ROOT / "sample-data" / "cpsc2018"), verbose=0)
rec = reader.all_records[0]
sig = reader.load_data(rec)
fs = reader.fs

# one-element batches; the label is a dummy multi-label vector, unchanged by
# all three waveform augmenters
sig_batch = torch.from_numpy(sig).unsqueeze(0).to(torch.float32)
labels = torch.tensor([[1.0, 0.0, 0.0, 1.0]])

augmented = {}
# display name -> (config key, augmenter parameters); probabilities are set
# to 1 so that each augmenter is guaranteed to fire in this illustration;
# masking widths are in seconds; the baseline-wander amplitude candidates are
# restricted to one clearly visible level ("high" in the augmenter's scale)
for name, (key, kwargs) in {
    "baseline wander": ("baseline_wander", {"prob": 1.0, "ampl_ratio": np.array([[0.1, 0.15, 0.25, 0.3]])}),
    "random masking": ("random_masking", {"prob": [1.0, 1.0], "mask_width": [0.1, 0.2]}),
    "stretch-compress": ("stretch_compress", {"ratio": 6, "prob": 1.0}),
}.items():
    config = CFG(random=False, fs=fs, **{key: kwargs})
    am = AugmenterManager.from_config(config)
    sig_aug, _ = am(sig_batch.clone(), labels.clone())
    augmented[name] = sig_aug[0].numpy()

# render lead I over the first 5 seconds: original in the top-left panel,
# one augmenter per remaining panel; `plot_single_lead` draws the ECG-paper
# grid for input in microvolts
window = 5 * fs
t = np.arange(window) / fs

# label only whole seconds, otherwise the 0.2 s grid ticks crowd the axis
second_formatter = FuncFormatter(lambda x, _: f"{x:.0f}" if abs(x - round(x)) < 1e-9 else "")

fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
plot_single_lead(t, sig[0, :window] * 1000.0, ax=axes[0, 0], ticks_granularity=2)
axes[0, 0].set_ylabel("original")
for ax, (name, sig_a) in zip(axes.flat[1:], augmented.items()):
    plot_single_lead(t, sig_a[0, :window] * 1000.0, ax=ax, ticks_granularity=2)
    ax.set_ylabel(name)
for ax in axes.flat:
    ax.xaxis.set_major_formatter(second_formatter)
    ax.set_xlabel("")
for ax in axes[1]:
    ax.set_xlabel("time (s)")
axes[0, 0].set_title(f"Record {rec}, lead {reader.all_leads[0]}, first 5 s")
fig.tight_layout()
