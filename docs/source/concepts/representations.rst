.. _concepts/representations:

Data representation conventions
===============================

torch_ecg standardizes a small set of representations for signals,
waveform annotations, intervals and labels. Sticking to them lets
readers, datasets, augmenters and models compose freely; this page
states the conventions once.

Signals: lead-first by default
------------------------------

A multi-lead signal is a numpy array (or tensor) of shape
``(n_leads, n_samples)`` — **lead-first** — matching
``(channel, time)`` in PyTorch. The 12-lead order is fixed by
:class:`~torch_ecg.utils.ecg_arrhythmia_knowledge.Standard12Leads`:

.. code-block:: python

   from torch_ecg.utils.ecg_arrhythmia_knowledge import Standard12Leads

   list(Standard12Leads)
   # ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', ..., 'V6']

Use :func:`~torch_ecg.utils.utils_data.ensure_lead_fmt` when a source
uses the lead-last layout, and
:func:`~torch_ecg.utils.utils_data.ensure_siglen` to pad or crop to a
fixed length:

.. code-block:: python

   from torch_ecg.utils.utils_data import ensure_lead_fmt

   sig  # (12, 500), lead-first
   ensure_lead_fmt(sig, 12, fmt="lead_last").shape  # (500, 12)

Waveform masks
--------------

Wave delineation (P/QRS/T boundaries) is carried as an **integer mask
of shape ``(n_leads, n_samples)``**, with a per-task ``class_map``
assigning wave names to integers, e.g. ``{"pwave": 1, "qrs": 2}``.
Two converters move between masks and explicit waveforms/intervals:

- :func:`~torch_ecg.utils.utils_data.mask_to_intervals` works on a
  **single lead** (a 1-D mask) and returns half-open intervals
  ``[a, b)`` by default (``right_inclusive=True`` switches to ``[a, b]``):

  .. code-block:: python

     mask = np.zeros(100, dtype=int)
     mask[10:20] = 1; mask[80:95] = 1; mask[50:60] = 2
     mask_to_intervals(mask, vals=1)        # [[10, 20], [80, 95]]

- :func:`~torch_ecg.utils.utils_data.masks_to_waveforms` converts the
  full ``(n_leads, n_samples)`` mask into per-lead lists of
  :class:`~torch_ecg.utils.utils_data.ECGWaveForm` dataclasses, with
  ``onset``/``offset``/``peak`` indices and a ``duration`` in
  milliseconds:

  .. code-block:: python

     wfs = masks_to_waveforms(mask_2d, class_map, fs=100, leads=["I", "II"])
     wfs["I"][0].name, wfs["I"][0].onset, wfs["I"][0].duration
     # ('pwave', 10, 200.0)

Intervals and generalized intervals
-----------------------------------

An :data:`~torch_ecg.utils.utils_interval.Interval` is a pair
``[a, b]``; a :data:`~torch_ecg.utils.utils_interval.GeneralizedInterval`
is a list of such pairs (a possibly non-contiguous span, e.g. the
union of all QRS complexes). The
:mod:`~torch_ecg.utils.utils_interval` module provides the algebra
over them — :func:`~torch_ecg.utils.utils_interval.intervals_union`,
:func:`~torch_ecg.utils.utils_interval.generalized_intervals_union`,
:func:`~torch_ecg.utils.utils_interval.intervals_intersection`,
:func:`~torch_ecg.utils.utils_interval.generalized_interval_complement`,
length helpers, and cover-finding utilities — which the dataset
samplers use to cut training segments around annotated structures.

Labels
------

Classification labels are **multi-hot vectors** over the task's
``classes`` list. :func:`~torch_ecg.utils.utils_data.one_hot_encode`
builds them from class indices, for both the single-label and the
multi-label case:

.. code-block:: python

   from torch_ecg.utils.utils_data import one_hot_encode

   one_hot_encode([[1], [0, 2]], num_classes=3)
   # array([[0., 1., 0.],
   #        [1., 0., 1.]], dtype=float32)

Models output raw logits over ``classes``; the sigmoid outputs are
binarized at *inference* time (challenge-specific thresholds or the
default 0.5), never inside the model. This keeps the training loss and
the official evaluation metrics free of hard-coded decision rules.
