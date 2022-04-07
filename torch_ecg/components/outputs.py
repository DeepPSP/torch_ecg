"""
"""

from abc import ABC, abstractmethod
from typing import Any, NoReturn, Sequence, Set, Union

import numpy as np
import pandas as pd

from ..cfg import CFG
from ..utils.misc import add_docstring
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
    "SequenceLabelingOutput",
    "WaveDelineationOutput",
    "RPeaksDetectionOutput",
]


_KNOWN_ISSUES = """
        Known Issues
        ------------
        - fields of type `dict` are not well supported, for example:{}
        """
_ClassificationOutput_ISSUE_EXAMPLE = """
        >>> output = ClassificationOutput(classes=["AF", "N", "SPB"], pred=np.ones((1,3)), prob=np.ones((1,3)), d={"d":1})
        >>> output
        {'classes': ['AF', 'N', 'SPB'],
         'prob': array([[1., 1., 1.]]),
         'pred': array([[1., 1., 1.]]),
         'd': {'d': 1, 'classes': None, 'prob': None, 'pred': None}}
        >>> output.d
        AttributeError: 'ClassificationOutput' object has no attribute 'd'
        """
_MultiLabelClassificationOutput_ISSUE_EXAMPLE = """
        >>> output = MultiLabelClassificationOutput(classes=["AF", "N", "SPB"], thr=0.5, pred=np.ones((1,3)), prob=np.ones((1,3)), d={"d":1})
        >>> output
        {'classes': ['AF', 'N', 'SPB'],
         'prob': array([[1., 1., 1.]]),
         'pred': array([[1., 1., 1.]]),
         'thr': 0.5,
         'd': {'d': 1, 'classes': None, 'thr': None, 'prob': None, 'pred': None}}
        >>> output.d
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
         'd': {'d': 1, 'classes': None, 'prob': None, 'pred': None}}
        >>> output.d
        AttributeError: 'SequenceTaggingOutput' object has no attribute 'd'
        """
_WaveDelineationOutput_ISSUE_EXAMPLE = """
        >>> output = WaveDelineationOutput(classes=["N", "P", "Q",], thr=0.5, pred=np.ones((1,3,3)), prob=np.ones((1,3,3)), d={"d":1})
        >>> output
        {'classes': ['AF', 'N', 'SPB'],
         'prob': array([[[1., 1., 1.],
                 [1., 1., 1.],
                 [1., 1., 1.]]]),
         'mask': array([[[1., 1., 1.],
                 [1., 1., 1.],
                 [1., 1., 1.]]]),
         'd': {'d': 1, 'classes': None, 'prob': None, 'pred': None, 'mask': None}}
        >>> output.d
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
         'd': {'d': 1, 'rpeak_indices': None, 'prob': None}}
        >>> output.d
        AttributeError: 'RPeaksDetectionOutput' object has no attribute 'd'
        """


class BaseOutput(CFG, ABC):
    """

    Base class for all outputs

    """

    __name__ = "BaseOutput"

    def __init__(self, *args: Any, **kwargs: Any) -> NoReturn:
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
        raise NotImplementedError

    def append(self, values: Union["BaseOutput", Sequence["BaseOutput"]]) -> NoReturn:
        """

        append other `Output`s to self

        Parameters
        ----------
        values: `Output` or sequence of `Output`,
            the values to be appended

        """
        if isinstance(values, BaseOutput):
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


class ClassificationOutput(BaseOutput):
    """

    Class that maintains the output of a (typically single-label) classification task.

    """

    __name__ = "ClassificationOutput"

    @add_docstring(_KNOWN_ISSUES.format(_ClassificationOutput_ISSUE_EXAMPLE), "append")
    def __init__(
        self,
        *args: Any,
        classes: Sequence[str] = None,
        prob: np.ndarray = None,
        pred: np.ndarray = None,
        **kwargs: Any,
    ) -> NoReturn:
        """

        classification output

        Parameters
        ----------
        classes : sequence of str,
            class names
        prob : np.ndarray,
            probabilities of each class,
            of shape (batch_size, num_classes)
        pred : np.ndarray,
            predicted class indices, or binary predictions,
            of shape (batch_size,) or (batch_size, num_classes)"""
        super().__init__(*args, classes=classes, prob=prob, pred=pred, **kwargs)

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
        return clf_met(
            self.get("labels", self.get("label")), self.pred, len(self.classes)
        )


class MultiLabelClassificationOutput(BaseOutput):
    """

    Class that maintains the output of a multi-label classification task.

    """

    __name__ = "MultiLabelClassificationOutput"

    @add_docstring(
        _KNOWN_ISSUES.format(_MultiLabelClassificationOutput_ISSUE_EXAMPLE), "append"
    )
    def __init__(
        self,
        *args: Any,
        classes: Sequence[str] = None,
        thr: float = None,
        prob: np.ndarray = None,
        pred: np.ndarray = None,
        **kwargs: Any,
    ) -> NoReturn:
        """

        multi-label classification output

        Parameters
        ----------
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
        super().__init__(
            *args, classes=classes, thr=thr, prob=prob, pred=pred, **kwargs
        )

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
        return clf_met(
            self.get("labels", self.get("label")), self.pred, len(self.classes)
        )


class SequenceTaggingOutput(BaseOutput):
    """

    Class that maintains the output of a sequence tagging task.

    """

    __name__ = "SequenceTaggingOutput"

    @add_docstring(_KNOWN_ISSUES.format(_SequenceTaggingOutput_ISSUE_EXAMPLE), "append")
    def __init__(
        self,
        *args: Any,
        classes: Sequence[str] = None,
        prob: np.ndarray = None,
        pred: np.ndarray = None,
        **kwargs: Any,
    ) -> NoReturn:
        """

        sequence tagging output

        Parameters
        ----------
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
        super().__init__(*args, classes=classes, prob=prob, pred=pred, **kwargs)

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
        return clf_met(
            labels.reshape((-1, labels.shape[-1])),
            self.pred.reshape((-1, self.pred.shape[-1])),
            len(self.classes),
        )


# alias
SequenceLabelingOutput = SequenceTaggingOutput
SequenceLabelingOutput.__name__ = "SequenceLabelingOutput"


class WaveDelineationOutput(SequenceTaggingOutput):
    """

    Class that maintains the output of a wave delineation task.

    """

    __name__ = "WaveDelineationOutput"

    @add_docstring(_KNOWN_ISSUES.format(_WaveDelineationOutput_ISSUE_EXAMPLE), "append")
    def __init__(
        self,
        *args: Any,
        classes: Sequence[str] = None,
        prob: np.ndarray = None,
        mask: np.ndarray = None,
        **kwargs: Any,
    ) -> NoReturn:
        """

        wave delineation output

        Parameters
        ----------
        classes : sequence of str,
            class names
        prob : np.ndarray,
            probabilities of each class at each time step (each sample point),
            of shape (batch_size, signal_length, num_classes)
        mask : np.ndarray,
            predicted class indices at each time step (each sample point),
            or binary predictions at each time step (each sample point),
            of shape (batch_size, signal_length), or (batch_size, signal_length, num_classes)

        """
        super().__init__(
            *args, classes=classes, prob=prob, pred=mask, mask=mask, **kwargs
        )
        self.pop("pred")

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
        self, macro: bool = True, tol: float = 0.15
    ) -> ClassificationMetrics:
        """

        compute metrics from the output

        Parameters
        ----------
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
        return wd_met(
            labels.reshape((-1, labels.shape[-1])),
            self.mask.reshape((-1, self.mask.shape[-1])),
            len(self.classes),
        )


class RPeaksDetectionOutput(BaseOutput):
    """

    Class that maintains the output of an R peaks detection task.

    """

    __name__ = "RPeaksDetectionOutput"

    @add_docstring(_KNOWN_ISSUES.format(_RPeaksDetectionOutput_ISSUE_EXAMPLE), "append")
    def __init__(
        self,
        *args: Any,
        rpeak_indices: Sequence[Sequence[int]] = None,
        prob: np.ndarray = None,
        pred: np.ndarray = None,
        **kwargs: Any,
    ) -> NoReturn:
        """

        r-peaks detection output

        Parameters
        ----------
        rpeak_indices : sequence of sequence of int,
            r-peak indices for each batch sample
        prob : np.ndarray,
            probabilities at each time step (each sample point),
            of shape (batch_size, signal_length)

        """
        super().__init__(*args, rpeak_indices=rpeak_indices, prob=prob, **kwargs)

    def required_fields(self) -> Set[str]:
        """ """
        return set(
            [
                "rpeak_indices",
                "prob",
            ]
        )

    def compute_metrics(self, thr: float = 0.15) -> ClassificationMetrics:
        """

        compute metrics from the output

        Parameters
        ----------
        tol: float, default 0.15,
            tolerance for the duration of the waveform,
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
        return rpd_met(self.get("labels", self.get("label")), self.rpeak_indices)
