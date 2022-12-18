"""
"""

from abc import ABC, abstractmethod
from typing import Any, Union, Sequence, Set, Dict

import numpy as np
import pandas as pd

from ..cfg import CFG
from ..utils.misc import add_docstring
from ..utils.utils_data import ECGWaveFormNames
from .metrics import (
    ClassificationMetrics,
    RPeaksDetectionMetrics,
    WaveDelineationMetrics,
)


__all__ = [
    "BaseOutput",
    "ClassificationOutput",
    "MultiLabelClassificationOutput",
    "SequenceTaggingOutput",
    "SequenceLabellingOutput",
    "WaveDelineationOutput",
    "RPeaksDetectionOutput",
]


_KNOWN_ISSUES = """
    Known Issues
    ------------
    - fields of type `dict` are not well supported due to the limitations of the base class `CFG`, for example:{}
    """
_ClassificationOutput_ISSUE_EXAMPLE = """
    >>> output = ClassificationOutput(classes=["AF", "N", "SPB"], pred=np.ones((1,3)), prob=np.ones((1,3)), d={"d":1})
    >>> output
    {'classes': ['AF', 'N', 'SPB'],
        'prob': array([[1., 1., 1.]]),
        'pred': array([[1., 1., 1.]]),
        'd': {'d': 1}}
    >>> output.d  # has to access via `output["d"]`
    AttributeError: 'ClassificationOutput' object has no attribute 'd'
    """
_MultiLabelClassificationOutput_ISSUE_EXAMPLE = """
    >>> output = MultiLabelClassificationOutput(classes=["AF", "N", "SPB"], thr=0.5, pred=np.ones((1,3)), prob=np.ones((1,3)), d={"d":1})
    >>> output
    {'classes': ['AF', 'N', 'SPB'],
        'prob': array([[1., 1., 1.]]),
        'pred': array([[1., 1., 1.]]),
        'thr': 0.5,
        'd': {'d': 1}}
    >>> output.d  # has to access via `output["d"]`
    AttributeError: 'MultiLabelClassificationOutput' object has no attribute 'd'
    """
_SequenceTaggingOutput_ISSUE_EXAMPLE = """
    >>> output = SequenceTaggingOutput(classes=["AF", "N", "SPB"], thr=0.5, pred=np.ones((1,3,3)), prob=np.ones((1,3,3)), d={"d":1})
    >>> output
    {'classes': ['AF', 'N', 'SPB'],
        'prob': array([[[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]]]),
        'pred': array([[[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]]]),
        'thr': 0.5,
        'd': {'d': 1}}
    >>> output.d  # has to access via `output["d"]`
    AttributeError: 'SequenceTaggingOutput' object has no attribute 'd'
    """
_WaveDelineationOutput_ISSUE_EXAMPLE = """
    >>> output = WaveDelineationOutput(classes=["N", "P", "Q",], thr=0.5, mask=np.ones((1,3,3)), prob=np.ones((1,3,3)), d={"d":1})
    >>> output
    {'classes': ['AF', 'N', 'SPB'],
        'prob': array([[[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]]]),
        'mask': array([[[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]]]),
        'd': {'d': 1}}
    >>> output.d  # has to access via `output["d"]`
    AttributeError: 'WaveDelineationOutput' object has no attribute 'd'
    """
_RPeaksDetectionOutput_ISSUE_EXAMPLE = """
    >>> output = RPeaksDetectionOutput(rpeak_indices=[[2]], thr=0.5, prob=np.ones((1,3,3)), d={"d":1})
    >>> output
    {'rpeak_indices': [[2]],
        'prob': array([[[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]]]),
        'thr': 0.5,
        'd': {'d': 1}}
    >>> output.d  # has to access via `output["d"]`
    AttributeError: 'RPeaksDetectionOutput' object has no attribute 'd'
    """


class BaseOutput(CFG, ABC):
    """
    Base class for all outputs
    """

    __name__ = "BaseOutput"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """ """
        super().__init__(*args, **kwargs)
        pop_fields = [
            k
            for k in self
            if k in ["required_fields", "append", "compute_metrics"]
            or k.startswith("_abc")
        ]
        for f in pop_fields:
            self.pop(f, None)
        assert all(field in self.keys() for field in self.required_fields()), (
            f"{self.__name__} requires {self.required_fields()}, "
            f"but `{', '.join(self.required_fields() - set(self.keys()))}` are missing"
        )
        assert all(
            self[field] is not None for field in self.required_fields()
        ), f"Fields `{', '.join([field for field in self.required_fields() if self[field] is None])}` are not set"

    @abstractmethod
    def required_fields(self) -> Set[str]:
        """ """
        raise NotImplementedError("Subclass must implement method `required_fields`")

    def append(self, values: Union["BaseOutput", Sequence["BaseOutput"]]) -> None:
        """
        append other `Output`s to self

        Parameters
        ----------
        values: `Output` or sequence of `Output`,
            the values to be appended

        """
        if not isinstance(values, Sequence):
            values = [values]
        for v in values:
            assert (
                v.__class__ == self.__class__
            ), "`values` must be of the same type as `self`"
            assert set(v.keys()) == set(
                self.keys()
            ), "`values` must have the same fields as `self`"
            for k, v_ in v.items():
                if k in ["classes"]:
                    assert (
                        v_ == self[k]
                    ), f"the field of ordered sequence `{k}` must be the identical"
                    continue
                if isinstance(v_, np.ndarray):
                    self[k] = np.concatenate((self[k], v_))
                elif isinstance(v_, pd.DataFrame):
                    self[k] = pd.concat([self[k], v_], axis=0, ignore_index=True)
                elif isinstance(v_, Sequence):  # list, tuple, etc.
                    self[k] += v_
                else:
                    raise ValueError(
                        f"field `{k}` of type `{type(v_)}` is not supported"
                    )


@add_docstring(_KNOWN_ISSUES.format(_ClassificationOutput_ISSUE_EXAMPLE), "append")
class ClassificationOutput(BaseOutput):
    """
    Class that maintains the output of a (typically single-label) classification task.

    Required parameters for `__init__`
    ----------------------------------
    classes : sequence of str,
        class names
    prob : np.ndarray,
        probabilities of each class,
        of shape (batch_size, num_classes)
    pred : np.ndarray,
        predicted class indices, or binary predictions,
        of shape (batch_size,) or (batch_size, num_classes)
    """

    __name__ = "ClassificationOutput"

    def required_fields(self) -> Set[str]:
        """ """
        return set(
            [
                "classes",
                "prob",
                "pred",
            ]
        )

    def compute_metrics(self) -> ClassificationMetrics:
        """
        compute metrics from the output

        Returns
        -------
        metrics : `ClassificationMetrics`
            metrics computed from the output

        """
        assert hasattr(self, "labels") or hasattr(
            self, "label"
        ), "`labels` or `label` must be stored in the output for computing metrics"
        clf_met = ClassificationMetrics(multi_label=False, macro=True)
        return clf_met.compute(
            self.get("labels", self.get("label")), self.pred, len(self.classes)
        )


@add_docstring(
    _KNOWN_ISSUES.format(_MultiLabelClassificationOutput_ISSUE_EXAMPLE), "append"
)
class MultiLabelClassificationOutput(BaseOutput):
    """
    Class that maintains the output of a multi-label classification task.

    Required parameters for `__init__`
    ----------------------------------
    classes : sequence of str,
        class names
    thr : float,
        threshold for making binary predictions
    prob : np.ndarray,
        probabilities of each class,
        of shape (batch_size, num_classes)
    pred : np.ndarray,
        binary predictions,
        of shape (batch_size, num_classes)
    """

    __name__ = "MultiLabelClassificationOutput"

    def required_fields(self) -> Set[str]:
        """ """
        return set(
            [
                "classes",
                "thr",
                "prob",
                "pred",
            ]
        )

    def compute_metrics(self, macro: bool = True) -> ClassificationMetrics:
        """
        compute metrics from the output

        Parameters
        ----------
        macro: bool,
            whether to use macro-averaged metrics

        Returns
        -------
        metrics : `ClassificationMetrics`
            metrics computed from the output

        """
        assert hasattr(self, "labels") or hasattr(
            self, "label"
        ), "`labels` or `label` must be stored in the output for computing metrics"
        clf_met = ClassificationMetrics(multi_label=True, macro=macro)
        return clf_met.compute(
            self.get("labels", self.get("label")), self.pred, len(self.classes)
        )


@add_docstring(_KNOWN_ISSUES.format(_SequenceTaggingOutput_ISSUE_EXAMPLE), "append")
class SequenceTaggingOutput(BaseOutput):
    """
    Class that maintains the output of a sequence tagging task.

    Required parameters for `__init__`
    ----------------------------------
    classes : sequence of str,
        class names
    prob : np.ndarray,
        probabilities of each class at each time step (each sample point),
        of shape (batch_size, signal_length, num_classes)
    pred : np.ndarray,
        predicted class indices at each time step (each sample point),
        or binary predictions at each time step (each sample point),
        of shape (batch_size, signal_length), or (batch_size, signal_length, num_classes)
    """

    __name__ = "SequenceTaggingOutput"

    def required_fields(self) -> Set[str]:
        """ """
        return set(
            [
                "classes",
                "prob",
                "pred",
            ]
        )

    def compute_metrics(self, macro: bool = True) -> ClassificationMetrics:
        """
        compute metrics from the output

        Parameters
        ----------
        macro: bool,
            whether to use macro-averaged metrics

        Returns
        -------
        metrics : `ClassificationMetrics`
            metrics computed from the output

        """
        assert hasattr(self, "labels") or hasattr(
            self, "label"
        ), "`labels` or `label` must be stored in the output for computing metrics"
        clf_met = ClassificationMetrics(multi_label=False, macro=macro)
        labels = self.get("labels", self.get("label"))
        return clf_met.compute(
            labels.reshape((-1, labels.shape[-1])),
            self.pred.reshape((-1, self.pred.shape[-1])),
            len(self.classes),
        )


# alias
SequenceLabellingOutput = SequenceTaggingOutput
SequenceLabellingOutput.__name__ = "SequenceLabellingOutput"


@add_docstring(_KNOWN_ISSUES.format(_WaveDelineationOutput_ISSUE_EXAMPLE), "append")
class WaveDelineationOutput(SequenceTaggingOutput):
    """
    Class that maintains the output of a wave delineation task.

    Required parameters for `__init__`
    ----------------------------------
    classes : sequence of str,
        class names
    prob : np.ndarray,
        probabilities of each class at each time step (each sample point),
        of shape (batch_size, signal_length, num_classes)
    mask : np.ndarray,
        predicted class indices at each time step (each sample point),
        or binary predictions at each time step (each sample point),
        of shape (batch_size, num_channels, signal_length)
    """

    __name__ = "WaveDelineationOutput"

    def required_fields(self) -> Set[str]:
        """ """
        return set(
            [
                "classes",
                "prob",
                "mask",
            ]
        )

    def compute_metrics(
        self,
        fs: int,
        class_map: Dict[str, int],
        macro: bool = True,
        tol: float = 0.15,
    ) -> ClassificationMetrics:
        f"""
        compute metrics from the output

        Parameters
        ----------
        fs: real number,
            sampling frequency of the signal corresponding to the masks,
            used to compute the duration of each waveform,
            hence the error and standard deviations of errors
        class_map: dict,
            class map, mapping names to waves to numbers from 0 to n_classes-1,
            the keys should contain {", ".join([f'"{item}"' for item in ECGWaveFormNames])}
        macro: bool,
            whether to use macro-averaged metrics
        tol: float, default 0.15,
            tolerance for the duration of the waveform,
            with units in seconds

        Returns
        -------
        metrics : `WaveDelineationMetrics`
            metrics computed from the output

        """
        assert hasattr(self, "labels") or hasattr(
            self, "label"
        ), "`labels` or `label` must be stored in the output for computing metrics"
        wd_met = WaveDelineationMetrics(macro=macro, tol=tol)
        labels = self.get("labels", self.get("label"))
        return wd_met.compute(labels, self.mask, class_map=class_map, fs=fs)


@add_docstring(_KNOWN_ISSUES.format(_RPeaksDetectionOutput_ISSUE_EXAMPLE), "append")
class RPeaksDetectionOutput(BaseOutput):
    """
    Class that maintains the output of an R peaks detection task.

    Required parameters for `__init__`
    ----------------------------------
    rpeak_indices : sequence of sequence of int,
        r-peak indices for each batch sample
    prob : np.ndarray,
        probabilities at each time step (each sample point),
        of shape (batch_size, signal_length)
    """

    __name__ = "RPeaksDetectionOutput"

    def required_fields(self) -> Set[str]:
        """ """
        return set(
            [
                "rpeak_indices",
                "prob",
            ]
        )

    def compute_metrics(self, fs: int, thr: float = 0.075) -> ClassificationMetrics:
        """
        compute metrics from the output

        Parameters
        ----------
        fs: int,
            sampling frequency of the signal corresponding to the masks
        thr: float, default 0.075,
            threshold for a prediction to be truth positive,
            with units in seconds

        Returns
        -------
        metrics : `RPeaksDetectionMetrics`
            metrics computed from the output

        """
        assert hasattr(self, "labels") or hasattr(
            self, "label"
        ), "`labels` or `label` must be stored in the output for computing metrics"
        rpd_met = RPeaksDetectionMetrics(thr=thr)
        return rpd_met.compute(
            self.get("labels", self.get("label")), self.rpeak_indices, fs=fs
        )
