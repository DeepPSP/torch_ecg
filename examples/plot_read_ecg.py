"""
======================
Reading an ECG record
======================

This example loads one record from the CPSC 2018 sample data shipped with the
repository and renders its 12 standard leads.

The database reader classes in :mod:`torch_ecg.databases` provide a uniform
interface over the heterogeneous on-disk formats of the supported databases:
a reader is constructed with the database directory, iterates over record
names, and returns each record as an array of shape ``(n_leads, n_samples)``.
"""

import contextlib
import io
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

import torch_ecg
from torch_ecg.databases import CPSC2018
from torch_ecg.utils.misc import plot_single_lead

# select the headless backend for deterministic rendering on CI; switching
# after the pyplot import is fine as long as no figure exists yet
matplotlib.use("Agg")

# the sample data ships with the repository checkout; the editable install
# used to build these docs places the package inside the checkout, so the
# package root is the repository root (sphinx-gallery executes examples
# without `__file__`, hence this indirection)
REPO_ROOT = Path(torch_ecg.__file__).resolve().parents[1]

# the sample data ships without the annotation file of the database, and
# the reader says so when it is created; the labels are not used here,
# hence the message is discarded
with contextlib.redirect_stderr(io.StringIO()):
    reader = CPSC2018(str(REPO_ROOT / "sample-data" / "cpsc2018"), verbose=0)
print(f"{len(reader)} records available, e.g. {reader.all_records[:3]}")

rec = reader.all_records[0]
sig = reader.load_data(rec)
print(f"record {rec}: shape = {sig.shape}, sampling rate = {reader.fs} Hz")

# render the first 5 seconds of all 12 leads in the standard grid layout;
# `plot_single_lead` draws the ECG-paper grid for input in microvolts
window = 5 * reader.fs
t = np.arange(window) / reader.fs
sig_uv = sig * 1000.0

# label only whole seconds, otherwise the 0.2 s grid ticks crowd the axis
second_formatter = FuncFormatter(lambda x, _: f"{x:.0f}" if abs(x - round(x)) < 1e-9 else "")

fig, axes = plt.subplots(6, 2, figsize=(12, 9), sharex=True)
for lead_idx, ax in enumerate(axes.flat):
    plot_single_lead(t, sig_uv[lead_idx, :window], ax=ax, ticks_granularity=2)
    ax.set_ylabel(reader.all_leads[lead_idx], rotation=0, ha="right", fontsize=9)
    ax.xaxis.set_major_formatter(second_formatter)
for ax in axes[:-1].flat:
    ax.tick_params(labelbottom=False)
    ax.set_xlabel("")
fig.suptitle(f"Record {rec}, CPSC 2018 sample data, first 5 s")
fig.tight_layout()
