"""
"""

from abc import ABC, abstractmethod
from typing import NoReturn, Any, Union, Optional, List, Callable

import numpy as np
from torch import Tensor

from ..utils.misc import ReprMixin, nildent
from ..utils.utils_metrics import (
    _metrics_from_confusion_matrix,
    ovr_confusion_matrix,
    confusion_matrix,
    top_n_accuracy,
)


__all__ = [
    "Metrics",
    "ClassificationMetrics",
]


class Metrics(ReprMixin, ABC):
    """ """

    __name__ = "Metrics"

    def __call__(self, *args: Any, **kwargs: Any) -> "Metrics":
        return self.compute()

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> "Metrics":
        """ """
        raise NotImplementedError


class ClassificationMetrics(Metrics):
    """ """

    __name__ = "ClassificationMetrics"

    def __init__(
        self,
        multi_label: bool = True,
        macro: bool = True,
        extra_metrics: Optional[Callable] = None,
    ) -> NoReturn:
        """

        Parameters
        ----------
        multi_label: bool,
            whether is multi-label classification
        macro: bool,
            whether to use macro-averaged metrics
        extra_metrics: Callable,
            extra metrics to compute,
            has to be a function with signature:
            `def extra_metrics(
                labels: np.ndarray,
                outputs: np.ndarray,
                num_classes: Optional[int]=None,
                weights: Optional[np.ndarray]=None
            ) -> dict`
        """
        self.multi_label = multi_label
        self.set_macro(macro)
        self._extra_metrics = extra_metrics
        self._em = {}
        self._metrics = {
            k: np.nan
            for k in [
                "sens",  # sensitivity, recall, hit rate, or true positive rate
                "spec",  # specificity, selectivity or true negative rate
                "prec",  # precision or positive predictive value
                "npv",  # negative predictive value
                "jac",  # jaccard index, threat score, or critical success index
                "acc",  # accuracy
                "phi",  # phi coefficient, or matthews correlation coefficient
                "fnr",  # false negative rate, miss rate
                "fpr",  # false positive rate, fall-out
                "fdr",  # false discovery rate
                "for",  # false omission rate
                "plr",  # positive likelihood ratio
                "nlr",  # negative likelihood ratio
                "pt",  # prevalence threshold
                "ba",  # balanced accuracy
                "f1",  # f1-measure
                "fm",  # fowlkes-mallows index
                "bm",  # bookmaker informedness
                "mk",  # markedness
                "dor",  # diagnostic odds ratio
                "auroc",  # area under the receiver-operater characteristic curve (ROC AUC)
                "auprc",  # area under the precision-recall curve
            ]
        }
        self._metrics.update({f"macro_{k}": np.nan for k in self._metrics})
        self._cm = None
        self._cm_ovr = None

    def set_macro(self, macro: bool) -> "ClassificationMetrics":
        """

        Parameters
        ----------
        macro: bool,
            whether to use macro-averaged metrics
        """
        self.__prefix = ""
        self.macro = macro
        if macro:
            self.__prefix = "macro_"

    def compute(
        self,
        labels: Union[np.ndarray, Tensor],
        outputs: Union[np.ndarray, Tensor],
        num_classes: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
    ) -> "ClassificationMetrics":
        self._cm = confusion_matrix(labels, outputs, num_classes)
        self._cm_ovr = ovr_confusion_matrix(labels, outputs, num_classes)
        self._metrics = _metrics_from_confusion_matrix(
            labels, outputs, num_classes, weights
        )
        if self._extra_metrics is not None:
            self._em = self._extra_metrics(labels, outputs, num_classes, weights)
            self._metrics.update(self._em)

        return self

    def __call__(self, *args: Any, **kwargs: Any) -> "ClassificationMetrics":
        return self.compute()

    compute.__doc__ = _metrics_from_confusion_matrix.__doc__.replace(
        "metrics: dict,", f"{__name__},"
    ).replace(
        "metrics = _metrics_from_confusion_matrix(labels, outputs)",
        """metrics = ClassificationMetrics()
    >>> metrics = metrics.compute(labels, outputs)
    >>> metrics.fl_measure
    0.5062821146226457
    >>> metrics.set_macro(False)
    >>> metrics.fl_measure
    array([0.46938776, 0.4742268 , 0.4375    , 0.52941176, 0.58      ,
       0.57692308, 0.55769231, 0.48351648, 0.55855856, 0.3956044 ])""",
    )
    __call__.__doc__ = compute.__doc__

    @property
    def sensitivity(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}sens"]

    @property
    def recall(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}sens"]

    @property
    def hit_rate(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}sens"]

    @property
    def true_positive_rate(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}sens"]

    @property
    def specificity(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}spec"]

    @property
    def selectivity(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}spec"]

    @property
    def true_negative_rate(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}spec"]

    @property
    def precision(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}prec"]

    @property
    def positive_predictive_value(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}prec"]

    @property
    def negative_predictive_value(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}npv"]

    @property
    def jaccard_index(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}jac"]

    @property
    def threat_score(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}jac"]

    @property
    def critical_success_index(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}jac"]

    @property
    def accuracy(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}acc"]

    @property
    def phi_coefficient(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}phi"]

    @property
    def matthews_correlation_coefficient(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}phi"]

    @property
    def false_negative_rate(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}fnr"]

    @property
    def miss_rate(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}fnr"]

    @property
    def false_positive_rate(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}fpr"]

    @property
    def fall_out(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}fpr"]

    @property
    def false_discovery_rate(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}fdr"]

    @property
    def false_omission_rate(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}for"]

    @property
    def positive_likelihood_ratio(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}plr"]

    @property
    def negative_likelihood_ratio(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}nlr"]

    @property
    def prevalence_threshold(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}pt"]

    @property
    def balanced_accuracy(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}ba"]

    @property
    def f1_measure(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}f1"]

    @property
    def fowlkes_mallows_index(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}fm"]

    @property
    def bookmaker_informedness(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}bm"]

    @property
    def markedness(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}mk"]

    @property
    def diagnostic_odds_ratio(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}dor"]

    @property
    def area_under_the_receiver_operater_characteristic_curve(
        self,
    ) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}auroc"]

    @property
    def auroc(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}auroc"]

    @property
    def area_under_the_precision_recall_curve(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}auprc"]

    @property
    def auprc(self) -> Union[float, np.ndarray]:
        return self._metrics[f"{self.__prefix}auprc"]

    @property
    def extra_metrics(self) -> dict:
        return self._em

    def extra_repr_keys(self) -> List[str]:
        return [
            "multi_label",
            "macro",
        ]
