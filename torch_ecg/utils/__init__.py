"""
torch_ecg.utils
===============

This module contains a collection of utility functions and classes that are used
throughout the package.

.. contents::
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: torch_ecg.utils

Neural network auxiliary functions and classes
----------------------------------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    extend_predictions
    compute_output_shape
    compute_conv_output_shape
    compute_deconv_output_shape
    compute_maxpool_output_shape
    compute_avgpool_output_shape
    compute_sequential_output_shape
    compute_module_size
    default_collate_fn
    compute_receptive_field
    adjust_cnn_filter_lengths
    SizeMixin
    CkptMixin

Signal processing functions
---------------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    smooth
    resample_irregular_timeseries
    detect_peaks
    remove_spikes_naive
    butter_bandpass_filter
    get_ampl
    normalize
    normalize_t
    resample_t

Data operations
---------------
.. autosummary::
    :toctree: generated/
    :recursive:

    get_mask
    class_weight_to_sample_weight
    ensure_lead_fmt
    ensure_siglen
    masks_to_waveforms
    mask_to_intervals
    uniform
    stratified_train_test_split
    cls_to_bin
    generate_weight_mask

Interval operations
-------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    overlaps
    validate_interval
    in_interval
    in_generalized_interval
    intervals_union
    generalized_intervals_union
    intervals_intersection
    generalized_intervals_intersection
    generalized_interval_complement
    get_optimal_covering
    interval_len
    generalized_interval_len
    find_extrema
    is_intersect
    max_disjoint_covering

Metrics computations
--------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    top_n_accuracy
    confusion_matrix
    ovr_confusion_matrix
    metrics_from_confusion_matrix
    compute_wave_delineation_metrics
    QRS_score

Decorators and Mixins
---------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    add_docstring
    remove_parameters_returns_from_docstring
    default_class_repr
    ReprMixin
    CitationMixin
    get_kwargs
    get_required_args
    add_kwargs

Path operations
---------------
.. autosummary::
    :toctree: generated/
    :recursive:

    get_record_list_recursive3

String operations
-----------------
.. autosummary::
    :toctree: generated/
    :recursive:

    dict_to_str
    str2bool
    nildent
    get_date_str

Visualization functions
------------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    ecg_plot

Miscellaneous
-------------
.. autosummary::
    :toctree: generated/
    :recursive:

    init_logger
    list_sum
    dicts_equal
    MovingAverage
    Timer
    timeout

"""
from . import ecg_arrhythmia_knowledge as EAK
from ._ecg_plot import ecg_plot
from .download import http_get
from .misc import (
    CitationMixin,
    MovingAverage,
    ReprMixin,
    Timer,
    add_docstring,
    add_kwargs,
    default_class_repr,
    dict_to_str,
    dicts_equal,
    get_date_str,
    get_kwargs,
    get_record_list_recursive3,
    get_required_args,
    init_logger,
    list_sum,
    make_serializable,
    nildent,
    remove_parameters_returns_from_docstring,
    str2bool,
    timeout,
)
from .utils_data import (
    ECGWaveForm,
    ECGWaveFormNames,
    class_weight_to_sample_weight,
    cls_to_bin,
    ensure_lead_fmt,
    ensure_siglen,
    generate_weight_mask,
    get_mask,
    mask_to_intervals,
    masks_to_waveforms,
    stratified_train_test_split,
    uniform,
)
from .utils_interval import (
    find_extrema,
    generalized_interval_complement,
    generalized_interval_len,
    generalized_intervals_intersection,
    generalized_intervals_union,
    get_optimal_covering,
    in_generalized_interval,
    in_interval,
    interval_len,
    intervals_intersection,
    intervals_union,
    is_intersect,
    max_disjoint_covering,
    overlaps,
    validate_interval,
)
from .utils_metrics import (
    QRS_score,
    compute_wave_delineation_metrics,
    confusion_matrix,
    metrics_from_confusion_matrix,
    ovr_confusion_matrix,
    top_n_accuracy,
)
from .utils_nn import (
    CkptMixin,
    SizeMixin,
    adjust_cnn_filter_lengths,
    compute_avgpool_output_shape,
    compute_conv_output_shape,
    compute_deconv_output_shape,
    compute_maxpool_output_shape,
    compute_module_size,
    compute_output_shape,
    compute_receptive_field,
    compute_sequential_output_shape,
    default_collate_fn,
    extend_predictions,
)
from .utils_signal import (
    butter_bandpass_filter,
    detect_peaks,
    get_ampl,
    normalize,
    remove_spikes_naive,
    resample_irregular_timeseries,
    smooth,
)
from .utils_signal_t import normalize as normalize_t
from .utils_signal_t import resample as resample_t

__all__ = [
    "EAK",
    "http_get",
    "get_record_list_recursive3",
    "dict_to_str",
    "str2bool",
    "init_logger",
    "get_date_str",
    "list_sum",
    "dicts_equal",
    "default_class_repr",
    "ReprMixin",
    "CitationMixin",
    "MovingAverage",
    "nildent",
    "add_docstring",
    "remove_parameters_returns_from_docstring",
    "timeout",
    "Timer",
    "get_kwargs",
    "get_required_args",
    "add_kwargs",
    "make_serializable",
    "get_mask",
    "class_weight_to_sample_weight",
    "ensure_lead_fmt",
    "ensure_siglen",
    "ECGWaveForm",
    "ECGWaveFormNames",
    "masks_to_waveforms",
    "mask_to_intervals",
    "uniform",
    "stratified_train_test_split",
    "cls_to_bin",
    "generate_weight_mask",
    "overlaps",
    "validate_interval",
    "in_interval",
    "in_generalized_interval",
    "intervals_union",
    "generalized_intervals_union",
    "intervals_intersection",
    "generalized_intervals_intersection",
    "generalized_interval_complement",
    "get_optimal_covering",
    "interval_len",
    "generalized_interval_len",
    "find_extrema",
    "is_intersect",
    "max_disjoint_covering",
    "top_n_accuracy",
    "confusion_matrix",
    "ovr_confusion_matrix",
    "QRS_score",
    "metrics_from_confusion_matrix",
    "compute_wave_delineation_metrics",
    "extend_predictions",
    "compute_output_shape",
    "compute_conv_output_shape",
    "compute_deconv_output_shape",
    "compute_maxpool_output_shape",
    "compute_avgpool_output_shape",
    "compute_sequential_output_shape",
    "compute_module_size",
    "default_collate_fn",
    "compute_receptive_field",
    "adjust_cnn_filter_lengths",
    "SizeMixin",
    "CkptMixin",
    "smooth",
    "resample_irregular_timeseries",
    "detect_peaks",
    "remove_spikes_naive",
    "butter_bandpass_filter",
    "get_ampl",
    "normalize",
    "normalize_t",
    "resample_t",
    "ecg_plot",
]
