"""
Components
==========

This module contains the components for training and evaluating models.

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

    LoggerManager

Metrics
-------
.. autosummary::
    :toctree: generated/

    ClassificationMetrics
    RPeaksDetectionMetrics
    WaveDelineationMetrics

Trainer
-------
.. autosummary::
    :toctree: generated/

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
