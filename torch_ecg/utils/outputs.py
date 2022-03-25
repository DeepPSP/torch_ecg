"""
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, List, Tuple, NoReturn, Sequence, Any

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
            f"but `{', '.join([field for field in self.keys() if field not in self.required_fields()])}` is missing"
        assert all(self[field] is not None for field in self.required_fields()), \
            f"Fields `{', '.join([field for field in self.required_fields() if self[field] is None])}` are not set"

    @abstractmethod
    def required_fields(self) -> List[str]:
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

    def required_fields(self) -> List[str]:
        """
        """
        return ["classes", "prob", "pred",]


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

    def required_fields(self) -> List[str]:
        """
        """
        return ["classes", "thr", "prob", "pred",]


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

    def required_fields(self) -> List[str]:
        """
        """
        return ["classes", "prob", "pred",]

# alias
SequenceLabelingOutput = SequenceTaggingOutput
SequenceLabelingOutput.__name__ = "SequenceLabelingOutput"


class WaveDelineationOutput(SequenceTaggingOutput):
    """
    """
    __name__ = "WaveDelineationOutput"


class RPeaksDectectionOutput(BaseOutput):
    """
    """
    __name__ = "RPeaksDectectionOutput"

    def __init__(self,
                 *args:Any,
                 rpeak_indices:Sequence[Sequence[int]]=None,
                 prob:np.ndarray=None,
                 pred:np.ndarray=None,
                 **kwargs:Any) -> NoReturn:
        """
        """
        super().__init__(*args, rpeak_indices=rpeak_indices, prob=prob, pred=pred, **kwargs)

    def required_fields(self) -> List[str]:
        """
        """
        return ["rpeak_indices", "prob", "pred",]
