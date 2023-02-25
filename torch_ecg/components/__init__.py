"""
torch_ecg.components
====================

This module contains the components for training and evaluating models.

.. contents:: torch_ecg.components
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: torch_ecg.components

Input classes
-------------
.. autosummary::
    :toctree: generated/

    InputConfig
    WaveformInput
    FFTInput
    SpectrogramInput

Output classes
--------------
.. autosummary::
    :toctree: generated/
    :recursive:

    ClassificationOutput
    MultiLabelClassificationOutput
    SequenceTaggingOutput
    SequenceLabellingOutput
    WaveDelineationOutput
    RPeaksDetectionOutput

Loggers
-------
.. autosummary::
    :toctree: generated/
    :recursive:

    LoggerManager

Metrics
-------
.. autosummary::
    :toctree: generated/
    :recursive:

    ClassificationMetrics
    RPeaksDetectionMetrics
    WaveDelineationMetrics

Trainer
-------
.. autosummary::
    :toctree: generated/
    :recursive:

    BaseTrainer

"""

from .inputs import (
    InputConfig,
    WaveformInput,
    FFTInput,
    SpectrogramInput,
)
from .loggers import LoggerManager
from .metrics import (
    ClassificationMetrics,
    RPeaksDetectionMetrics,
    WaveDelineationMetrics,
)
from .outputs import (
    ClassificationOutput,
    MultiLabelClassificationOutput,
    SequenceTaggingOutput,
    SequenceLabellingOutput,
    WaveDelineationOutput,
    RPeaksDetectionOutput,
)
from .trainer import BaseTrainer


__all__ = [
    "InputConfig",
    "WaveformInput",
    "FFTInput",
    "SpectrogramInput",
    "LoggerManager",
    "ClassificationMetrics",
    "RPeaksDetectionMetrics",
    "WaveDelineationMetrics",
    "ClassificationOutput",
    "MultiLabelClassificationOutput",
    "SequenceTaggingOutput",
    "SequenceLabellingOutput",
    "WaveDelineationOutput",
    "RPeaksDetectionOutput",
    "BaseTrainer",
]
