"""
"""

from transformers import Wav2Vec2ForPreTraining as HFWav2Vec2ForPreTraining
from torch_ecg.utils import SizeMixin, add_docstring


__all__ = [
    "Wav2Vec2ForPreTraining",
]


@add_docstring(HFWav2Vec2ForPreTraining.__doc__)
class Wav2Vec2ForPreTraining(HFWav2Vec2ForPreTraining, SizeMixin):
    """ """

    __name__ = "Wav2Vec2ForPreTraining"
