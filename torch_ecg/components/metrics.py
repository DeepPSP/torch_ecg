"""
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from torch import Tensor

from ..utils.misc import ReprMixin, add_docstring
from ..utils.utils_data import ECGWaveFormNames
from ..utils.utils_metrics import (
    QRS_score,
    compute_wave_delineation_metrics,
    confusion_matrix,
    metrics_from_confusion_matrix,
    one_hot_pair,
    ovr_confusion_matrix,
)

__all__ = [
    "Metrics",
    "ClassificationMetrics",
    "RPeaksDetectionMetrics",
    "WaveDelineationMetrics",
]


class Metrics(ReprMixin, ABC):
    """Base class for metrics."""

    __name__ = "Metrics"

    def __call__(self, *args: Any, **kwargs: Any) -> "Metrics":
        return self.compute(*args, **kwargs)

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> "Metrics":
        raise NotImplementedError


class ClassificationMetrics(Metrics):
    """Metrics for the task of classification.

    Parameters
    ----------
    multi_label : bool, default True
        Whether is multi-label classification task.
    macro : bool, default True
        Whether to use macro-averaged metrics.
    extra_metrics : callable, optional
        Extra metrics to compute,
        has to be a function with signature:

        .. code-block:: python

            def extra_metrics(
                labels : np.ndarray
                outputs : np.ndarray
                num_classes : Optional[int]=None
                weights : Optional[np.ndarray]=None
            ) -> dict

    """

    __name__ = "ClassificationMetrics"

    def __init__(
        self,
        multi_label: bool = True,
        macro: bool = True,
        extra_metrics: Optional[Callable] = None,
    ) -> None:
        self.multi_label = multi_label
        self.set_macro(macro)
        self._extra_metrics = extra_metrics
        self._em = {}
        self._metrics = {
            k: np.nan
            for k in [
                "sens",  # sensitivity, recall, hit rate, or true positive rate
                "spec",  # specificity, selectivity, or true negative rate
                "prec",  # precision, or positive predictive value
                "npv",  # negative predictive value
                "jac",  # jaccard index, threat score, or critical success index
                "acc",  # accuracy
                "phi",  # phi coefficient, or matthews correlation coefficient
                "fnr",  # false negative rate, or miss rate
                "fpr",  # false positive rate, or fall-out
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

    def set_macro(self, macro: bool) -> None:
        """Set whether to use macro-averaged metrics.

        Parameters
        ----------
        macro : bool
            Whether to use macro-averaged metrics.

        """
        self.__prefix = ""
        self.macro = macro
        if macro:
            self.__prefix = "macro_"

    @add_docstring(
        metrics_from_confusion_matrix.__doc__.replace("metrics : dict", f"self : {__name__},")
        .replace(
            "Metrics computed from the one-vs-rest confusion matrix.",
            "The metrics object itself with the computed metrics.",
        )
        .replace(
            "metrics = metrics_from_confusion_matrix(labels, outputs)",
            """metrics = ClassificationMetrics()
    >>> metrics = metrics.compute(labels, outputs)
    >>> metrics.fl_measure
    0.5062821146226457
    >>> metrics.set_macro(False)
    >>> metrics.fl_measure
    array([0.46938776, 0.4742268 , 0.4375    , 0.52941176, 0.58      ,
       0.57692308, 0.55769231, 0.48351648, 0.55855856, 0.3956044 ])""",
        )
    )
    def compute(
        self,
        labels: Union[np.ndarray, Tensor],
        outputs: Union[np.ndarray, Tensor],
        num_classes: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
        thr: float = 0.5,
    ) -> "ClassificationMetrics":
        labels, outputs = one_hot_pair(labels, outputs, num_classes)
        num_samples, num_classes = np.shape(labels)
        # probability outputs to binary outputs
        bin_outputs = np.zeros_like(outputs, dtype=int)
        bin_outputs[outputs >= thr] = 1
        bin_outputs[outputs < thr] = 0
        self._cm = confusion_matrix(labels, bin_outputs, num_classes)
        self._cm_ovr = ovr_confusion_matrix(labels, bin_outputs, num_classes)
        self._metrics = metrics_from_confusion_matrix(labels, outputs, num_classes, weights)
        if self._extra_metrics is not None:
            self._em = self._extra_metrics(labels, outputs, num_classes, weights)
            self._metrics.update(self._em)

        return self

    @add_docstring(compute.__doc__)
    def __call__(
        self,
        labels: Union[np.ndarray, Tensor],
        outputs: Union[np.ndarray, Tensor],
        num_classes: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
        thr: float = 0.5,
    ) -> "ClassificationMetrics":
        return self.compute(labels, outputs, num_classes, weights)

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
    def classification_report(self) -> dict:
        if self.__prefix == "macro_":
            return {k.replace("macro_", ""): v for k, v in self._metrics.items() if k.startswith("macro_")}
        else:
            return {k: v for k, v in self._metrics.items() if not k.startswith("macro_")}

    @property
    def extra_metrics(self) -> dict:
        return self._em

    def extra_repr_keys(self) -> List[str]:
        return [
            "multi_label",
            "macro",
        ]


class RPeaksDetectionMetrics(Metrics):
    """Metrics for the task of R peaks detection,
    as proposed in CPSC2019.

    Parameters
    ----------
    thr : float, default 0.075
        Threshold for a prediction to be truth positive,
        with units in seconds,
    extra_metrics : callable, optional
        Extra metrics to compute,
        has to be a function with signature

        .. code-block:: python

            def extra_metrics(
                labels : Sequence[Union[Sequence[int], np.ndarray]],
                outputs : Sequence[Union[Sequence[int], np.ndarray]],
                fs : int
            ) -> dict

    """

    __name__ = "RPeaksDetectionMetrics"

    def __init__(
        self,
        thr: float = 0.075,
        extra_metrics: Optional[Callable] = None,
    ) -> None:
        self.thr = thr
        self._extra_metrics = extra_metrics
        self._em = {}
        self._metrics = {"qrs_score": np.nan}

    @add_docstring(
        QRS_score.__doc__.replace("rpeaks_truths", "labels")
        .replace("rpeaks_preds", "outputs")
        .replace(
            "thr : float, default 0.075",
            "thr : float, optional, defaults to `self.thr`",
        )
        .replace("rec_acc : float", f"self : {__name__}")
        .replace(
            "Accuracy of predictions.",
            "The metrics object itself with the computed metrics.",
        )
        .rstrip(" \n")
        + """

        Examples
        --------
        >>> labels = [np.array([500, 1000])]
        >>> outputs = [np.array([500, 700, 1000])]  # a false positive at 700
        >>> metrics = RPeaksDetectionMetrics()
        >>> metrics = metrics.compute(labels, outputs, fs=500)
        >>> metrics.qrs_score
        0.7

        """
    )
    def compute(
        self,
        labels: Sequence[Union[Sequence[int], np.ndarray]],
        outputs: Sequence[Union[Sequence[int], np.ndarray]],
        fs: int,
        thr: Optional[float] = None,
    ) -> "RPeaksDetectionMetrics":
        self._metrics["qrs_score"] = QRS_score(labels, outputs, fs, thr or self.thr)
        if self._extra_metrics is not None:
            self._em = self._extra_metrics(labels, outputs, fs)
            self._metrics.update(self._em)

        return self

    @add_docstring(compute.__doc__)
    def __call__(
        self,
        labels: Sequence[Union[Sequence[int], np.ndarray]],
        outputs: Sequence[Union[Sequence[int], np.ndarray]],
        fs: int,
        thr: Optional[float] = None,
    ) -> "RPeaksDetectionMetrics":
        return self.compute(labels, outputs, fs, thr)

    @property
    def qrs_score(self) -> float:
        return self._metrics["qrs_score"]

    @property
    def extra_metrics(self) -> dict:
        return self._em

    def extra_repr_keys(self) -> List[str]:
        return ["thr"]


class WaveDelineationMetrics(Metrics):
    """Metrics for the task of ECG wave delineation.

    Parameters
    ----------
    macro : bool, default True
        Whether to use macro-averaged metrics or not.
    tol : float, default 0.15
        Tolerance for the duration of the waveform,
        with units in seconds.
    extra_metrics : callable, optional
        Extra metrics to compute,
        has to be a function with signature

        .. code-block:: python

            def extra_metrics(
                labels: Sequence[Union[Sequence[int], np.ndarray]],
                outputs: Sequence[Union[Sequence[int], np.ndarray]],
                fs: int
            ) -> dict

    """

    __name__ = "WaveDelineationMetrics"

    def __init__(
        self,
        macro: bool = True,
        tol: float = 0.15,
        extra_metrics: Optional[Callable] = None,
    ) -> None:
        self.set_macro(macro)
        self.tol = tol
        self._extra_metrics = extra_metrics
        self._em = {}
        self._metrics = {
            k: None
            for k in [
                "sensitivity",
                "precision",
                "f1_score",
                "mean_error",
                "standard_deviation",
                "jaccard",
            ]
        }
        self._metrics.update({f"macro_{k}": np.nan for k in self._metrics})

    def set_macro(self, macro: bool) -> None:
        """Set whether to use macro-averaged metrics or not.

        Parameters
        ----------
        macro : bool
            Shether to use macro-averaged metrics.

        """
        self.__prefix = ""
        self.macro = macro
        if macro:
            self.__prefix = "macro_"

    @add_docstring(
        f"""
        Compute metrics for the task of ECG wave delineation
        (sensitivity, precision, f1_score, mean error and standard deviation of the mean errors)
        for multiple evaluations.

        Parameters
        ----------
        labels : numpy.ndarray or torch.Tensor
            Ground truth masks,
            of shape ``(n_samples, n_channels, n_timesteps)``.
        outputs : numpy.ndarray or torch.Tensor
            Predictions corresponding to `labels`,
            of the same shape.
        class_map : dict
            Class map, mapping names to waves to numbers from 0 to n_classes-1,
            the keys should contain {", ".join([f'"{item}"' for item in ECGWaveFormNames])}.
        fs : numbers.Real
            Sampling frequency of the signal corresponding to the masks,
            used to compute the duration of each waveform,
            and thus the error and standard deviations of errors.
        mask_format : str, default "channel_first"
            Format of the mask, one of the following:
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first").
        tol : float, optional
            Tolerance for the duration of the waveform,
            with units in seconds.
            Defaults to `self.tol`.

        Returns
        -------
        self : WaveDelineationMetrics
            The metrics object itself with the computed metrics.

        """
    )
    def compute(
        self,
        labels: Union[np.ndarray, Tensor],
        outputs: Union[np.ndarray, Tensor],
        class_map: Dict[str, int],
        fs: int,
        mask_format: str = "channel_first",
        tol: Optional[float] = None,
    ) -> "WaveDelineationMetrics":
        # wave delineation specific metrics
        truth_masks = labels.numpy() if isinstance(labels, Tensor) else labels
        pred_masks = outputs.numpy() if isinstance(outputs, Tensor) else outputs
        raw_metrics = compute_wave_delineation_metrics(truth_masks, pred_masks, class_map, fs, mask_format, tol or self.tol)
        self._metrics = {
            metric: {
                f"{wf}_{pos}": raw_metrics[f"{wf}_{pos}"][metric]
                for wf in class_map
                for pos in [
                    "onset",
                    "offset",
                ]
            }
            for metric in [
                "sensitivity",
                "precision",
                "f1_score",
                "mean_error",
                "standard_deviation",
            ]
        }
        self._metrics.update(
            {
                f"macro_{metric}": np.nanmean(list(self._metrics[metric].values()))
                for metric in [
                    "sensitivity",
                    "precision",
                    "f1_score",
                    "mean_error",
                    "standard_deviation",
                ]
            }
        )
        # sample-wise metrics
        warnings.simplefilter("ignore")
        clf_mtx = ClassificationMetrics()
        swm = clf_mtx.compute(
            truth_masks.reshape((-1)).copy(),
            pred_masks.reshape((-1)).copy(),
            max(class_map.values()) + 1,
        )
        warnings.simplefilter("default")
        self._metrics.update(
            {
                "jaccard": {k: swm._metrics["jac"][v] for k, v in class_map.items()},
                "macro_jaccard": swm._metrics["macro_jac"],
            }
        )

        return self

    @add_docstring(compute.__doc__)
    def __call__(
        self,
        labels: Union[np.ndarray, Tensor],
        outputs: Union[np.ndarray, Tensor],
        class_map: Dict[str, int],
        fs: int,
        mask_format: str = "channel_first",
        tol: Optional[float] = None,
    ) -> "WaveDelineationMetrics":
        return self.compute(labels, outputs, class_map, fs, mask_format, tol)

    @property
    def sensitivity(self) -> Union[float, Dict[str, float]]:
        return self._metrics[f"{self.__prefix}sensitivity"]

    @property
    def precision(self) -> Union[float, Dict[str, float]]:
        return self._metrics[f"{self.__prefix}precision"]

    @property
    def f1_score(self) -> Union[float, Dict[str, float]]:
        return self._metrics[f"{self.__prefix}f1_score"]

    @property
    def mean_error(self) -> Union[float, Dict[str, float]]:
        return self._metrics[f"{self.__prefix}mean_error"]

    @property
    def standard_deviation(self) -> Union[float, Dict[str, float]]:
        return self._metrics[f"{self.__prefix}standard_deviation"]

    @property
    def jaccard_index(self) -> Union[float, Dict[str, float]]:
        return self._metrics[f"{self.__prefix}jaccard"]

    @property
    def extra_metrics(self) -> dict:
        return self._em

    def extra_repr_keys(self) -> List[str]:
        return ["tol"]
