import numpy as np
from itertools import product
from sklearn.metrics import confusion_matrix, fbeta_score, f1_score
from numbers import Real
from typing import Optional


__all__ = ["f_beta", "g_beta", "ConfusionMatrixDisplay",]


def f_beta(y_true:np.ndarray, y_pred:np.ndarray, labels:list, beta:Real=2, class_weights:Optional[List[Real]]=None) -> float:
    """
    """
    cw = np.ones(len(labels)) if class_weights is None else np.array(class_weights)
    score_for_each_class = fbeta_score(
        y_true=y_true,
        y_pred=y_pred,
        beta=beta,
        labels=labels,
        average=None,
    )
    score = score_for_each_class.dot(cw) / np.sum(cw)
    return score


def g_beta(y_true:np.ndarray, y_pred:np.ndarray, labels:list, beta:Real=2, class_weights:Optional[List[Real]]=None) -> float:
    """
    """
    cw = np.ones(len(labels)) if class_weights is None else np.array(class_weights)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    score_for_each_class = []
    for i in range(len(labels)):
        tp = cm[i,i]  # true positive
        fp = np.sum(cm[:,i]) - tp  # false positive
        fn = np.sum(cm[i,:]) - tp  # false negative
        s = tp / (tp+fp+beta*fn)
        score_for_each_class.append(s)
    score_for_each_class = np.array(score_for_each_class)
    score = score_for_each_class.dot(cw) / np.sum(cw)
    return score


class ConfusionMatrixDisplay:
    """Confusion Matrix visualization.
    
    from sklearn.metrics._plot, in support for older versions of sklearn (<=0.20.x)
    url: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/_plot/confusion_matrix.py

    Parameters
    ----------
    confusion_matrix : ndarray of shape (n_classes, n_classes)
        Confusion matrix.
    display_labels : ndarray of shape (n_classes,)
        Display labels for plot.
    Attributes
    ----------
    im_ : matplotlib AxesImage
        Image representing the confusion matrix.
    text_ : ndarray of shape (n_classes, n_classes), dtype=matplotlib Text, \
            or None
        Array of matplotlib axes. `None` if `include_values` is false.
    ax_ : matplotlib Axes
        Axes with confusion matrix.
    figure_ : matplotlib Figure
        Figure containing the confusion matrix.
    """
    def __init__(self, confusion_matrix, display_labels):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

    def plot(self, include_values=True, cmap='viridis',
             xticks_rotation='horizontal', values_format=None, ax=None):
        """Plot visualization.
        Parameters
        ----------
        include_values : bool, default=True
            Includes values in confusion matrix.
        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.
        xticks_rotation : {'vertical', 'horizontal'} or float, \
                         default='horizontal'
            Rotation of xtick labels.
        values_format : str, default=None
            Format specification for values in confusion matrix. If `None`,
            the format specification is '.2g'.
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        Returns
        -------
        display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
        """
        if 'plt' not in dir():
            import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        self.text_ = None

        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)
            if values_format is None:
                values_format = '.2g'

            # print text with appropriate color depending on background
            thresh = (cm.max() + cm.min()) / 2.0
            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min
                self.text_[i, j] = ax.text(j, i,
                                           format(cm[i, j], values_format),
                                           ha="center", va="center",
                                           color=color)

        fig.colorbar(self.im_, ax=ax)
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=self.display_labels,
               yticklabels=self.display_labels,
               ylabel="True label",
               xlabel="Predicted label")

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self
        