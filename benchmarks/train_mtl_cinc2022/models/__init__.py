"""
Models, including:
    - 1D models accepcting raw audio signal input
    - 2D models accepcting spectrogram input
"""

from .crnn import CRNN_CINC2022
from .wav2vec2 import Wav2Vec2_CINC2022, HFWav2Vec2_CINC2022
from .seg import SEQ_LAB_NET_CINC2022, UNET_CINC2022
from .model_ml import OutComeClassifier_CINC2022


__all__ = [
    "CRNN_CINC2022",
    "Wav2Vec2_CINC2022",
    "HFWav2Vec2_CINC2022",
    "SEQ_LAB_NET_CINC2022",
    "UNET_CINC2022",
    "OutComeClassifier_CINC2022",
]
