"""
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, Tuple, NoReturn, Sequence, Any, Set

import numpy as np

from ..cfg import CFG


__all__ = [
    "BaseOutput",
    "ClassificationOutput",
    "MultiLabelClassificationOutput",
    "SequenceTaggingOutput",
    "SequenceLabelingOutput",
    "WaveDelineationOutput",
    "RPeaksDetectionOutput",
]


class BaseOutput(CFG, ABC):
    """
    """
    __name__ = "BaseOutput"

    def __init__(self, *args:Any, **kwargs:Any) -> NoReturn:
        """
        """
        super().__init__(*args, **kwargs)
        pop_fields = [k for k in self if k == "required_fields" or k.startswith("_abc")]
        for f in pop_fields:
            self.pop(f)
        assert all(field in self.keys() for field in self.required_fields()), \
            f"{self.__name__} requires {self.required_fields()}, " \
            f"but `{', '.join(self.required_fields() - set(self.keys()))}` are missing"
        assert all(self[field] is not None for field in self.required_fields()), \
            f"Fields `{', '.join([field for field in self.required_fields() if self[field] is None])}` are not set"

    @abstractmethod
    def required_fields(self) -> Set[str]:
        """
        """
        raise NotImplementedError
    

class ClassificationOutput(BaseOutput):
    """
    """
    __name__ = "ClassificationOutput"

    def __init__(self,
                 *args:Any,
                 classes:Sequence[str]=None,
                 prob:np.ndarray=None,
                 pred:np.ndarray=None,
                 **kwargs:Any) -> NoReturn:
        """
        """
        super().__init__(*args, classes=classes, prob=prob, pred=pred, **kwargs)

    def required_fields(self) -> Set[str]:
        """
        """
        return set(["classes", "prob", "pred",])


class MultiLabelClassificationOutput(BaseOutput):
    """
    """
    __name__ = "MultiLabelClassificationOutput"

    def __init__(self,
                 *args:Any,
                 classes:Sequence[str]=None,
                 thr:float=None,
                 prob:np.ndarray=None,
                 pred:np.ndarray=None,
                 **kwargs:Any) -> NoReturn:
        """
        """
        super().__init__(*args, classes=classes, thr=thr, prob=prob, pred=pred, **kwargs)

    def required_fields(self) -> Set[str]:
        """
        """
        return set(["classes", "thr", "prob", "pred",])


class SequenceTaggingOutput(BaseOutput):
    """
    """
    __name__ = "SequenceTaggingOutput"

    def __init__(self,
                 *args:Any,
                 classes:Sequence[str]=None,
                 prob:np.ndarray=None,
                 pred:np.ndarray=None,
                 **kwargs:Any) -> NoReturn:
        """
        """
        super().__init__(*args, classes=classes, prob=prob, pred=pred, **kwargs)

    def required_fields(self) -> Set[str]:
        """
        """
        return set(["classes", "prob", "pred",])

# alias
SequenceLabelingOutput = SequenceTaggingOutput
SequenceLabelingOutput.__name__ = "SequenceLabelingOutput"


class WaveDelineationOutput(SequenceTaggingOutput):
    """
    """
    __name__ = "WaveDelineationOutput"

    def __init__(self,
                 *args:Any,
                 classes:Sequence[str]=None,
                 prob:np.ndarray=None,
                 mask:np.ndarray=None,
                 **kwargs:Any) -> NoReturn:
        """
        """
        super().__init__(*args, classes=classes, prob=prob, pred=mask, mask=mask, **kwargs)
        self.pop("pred")

    def required_fields(self) -> Set[str]:
        """
        """
        return set(["classes", "prob", "mask",])


class RPeaksDetectionOutput(BaseOutput):
    """
    """
    __name__ = "RPeaksDetectionOutput"

    def __init__(self,
                 *args:Any,
                 rpeak_indices:Sequence[Sequence[int]]=None,
                 prob:np.ndarray=None,
                 pred:np.ndarray=None,
                 **kwargs:Any) -> NoReturn:
        """
        """
        super().__init__(*args, rpeak_indices=rpeak_indices, prob=prob, **kwargs)

    def required_fields(self) -> Set[str]:
        """
        """
        return set(["rpeak_indices", "prob",])
