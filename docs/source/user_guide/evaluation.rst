.. _ug_evaluation:

Evaluation and metrics
======================

Classification metrics
----------------------

The generic metrics are built around the confusion matrix, with
multi-label support:

.. code-block:: python

    import numpy as np
    from torch_ecg.utils.utils_metrics import (
        confusion_matrix,
        metrics_from_confusion_matrix,
        top_n_accuracy,
    )

    labels = np.array([[1, 0, 1], [0, 1, 0]])
    probs = np.array([[0.9, 0.1, 0.4], [0.2, 0.8, 0.3]])   # predicted probabilities
    cm = confusion_matrix(labels, probs > 0.5)
    metrics = metrics_from_confusion_matrix(labels, probs)  # sens/spec/prec/f1/auc per class + macro

When class sets differ between training and evaluation,
:func:`~torch_ecg.utils.utils_nn.extend_predictions` remaps prediction
arrays to a superset of classes.

Official challenge metrics
--------------------------

The exact scoring functions of the challenges are provided with the
corresponding database modules; use them instead of reimplementations:

.. code-block:: python

    from torch_ecg.databases.physionet_databases.cinc2020 import compute_metrics as cinc2020_score
    from torch_ecg.databases.physionet_databases.cinc2021 import compute_metrics as cinc2021_score
    from torch_ecg.databases.cpsc_databases.cpsc2020 import compute_metrics as cpsc2020_score

Each takes per-record truth/prediction entries and returns the official
score plus its components (the benchmarks show full usage).

Waveform delineation
--------------------

For P/QRS/T delineation (mask output of the U-Net models),
:func:`~torch_ecg.utils.utils_metrics.compute_wave_delineation_metrics`
turns truth/prediction masks into sensitivities, F1 and onset/offset
deviation statistics; LUDB additionally provides
``compute_metrics`` / ``compute_metrics_waveform`` used by its example:

.. code-block:: python

    from torch_ecg.utils.utils_metrics import compute_wave_delineation_metrics

    metrics = compute_wave_delineation_metrics(truth_masks, pred_masks, class_map, fs=500)

R-peak detection
----------------

:func:`~torch_ecg.utils.utils_metrics.QRS_score` computes the standard
score (Se/+P/F1) for QRS detectors from reference and detected R-peak
locations, given as **lists of per-record arrays**:

.. code-block:: python

    from torch_ecg.utils.utils_metrics import QRS_score

    truth = [np.array([100, 300, 500]), np.array([80, 290, 510, 700])]
    pred = [np.array([101, 299, 505]), np.array([82, 288, 511, 705, 900])]
    QRS_score(truth, pred, fs=500)      # 0.85

The CPSC2019 module additionally provides the official challenge
metric used by its example.
