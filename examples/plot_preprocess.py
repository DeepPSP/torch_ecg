"""
========================================================
Preprocessing: bandpass filtering and baseline removal
========================================================

This example applies a typical preprocessing chain -- FIR bandpass filtering
followed by baseline-wander removal -- to a raw record from the CPSC 2018
sample data.

Preprocessors are assembled into a :class:`~torch_ecg.preprocessors.PreprocManager`
from a configuration mapping, mirroring the ``preproc`` section of a training
configuration.  With ``random = False`` the chain is applied in the declared
order on every call.
"""

import contextlib
import io
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

import torch_ecg
from torch_ecg.cfg import CFG
from torch_ecg.databases import CPSC2018
from torch_ecg.preprocessors import PreprocManager
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
rec = reader.all_records[0]
sig = reader.load_data(rec)
fs = reader.fs

config = CFG(
    random=False,
    bandpass={"fs": fs, "lowcut": 0.5, "highcut": 45.0, "filter_type": "fir"},
    baseline_remove={"fs": fs, "window1": 0.3, "window2": 0.7},
)
ppm = PreprocManager.from_config(config)
print(ppm)

sig_filtered = ppm(sig.copy())

# compare raw and processed lead I over the first 5 seconds; the last panel
# shows what the chain removed (baseline wander and out-of-band noise);
# `plot_single_lead` draws the ECG-paper grid for input in microvolts
window = 5 * fs
t = np.arange(window) / fs
lead = sig[0, :window] * 1000.0
lead_f = sig_filtered[0, :window] * 1000.0

# label only whole seconds, otherwise the 0.2 s grid ticks crowd the axis
second_formatter = FuncFormatter(lambda x, _: f"{x:.0f}" if abs(x - round(x)) < 1e-9 else "")

fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
plot_single_lead(t, lead, ax=axes[0], ticks_granularity=2)
axes[0].set_ylabel("raw")
plot_single_lead(t, lead_f, ax=axes[1], ticks_granularity=2)
axes[1].set_ylabel("processed")
plot_single_lead(t, lead - lead_f, ax=axes[2], ticks_granularity=2)
axes[2].set_ylabel("difference")
for ax in axes:
    ax.xaxis.set_major_formatter(second_formatter)
    ax.set_xlabel("")
axes[-1].set_xlabel("time (s)")
axes[0].set_title(f"Record {rec}, lead {reader.all_leads[0]}, first 5 s")
fig.tight_layout()
