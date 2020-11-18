"""

This folder acts as examples for configuring corresponding models.
These configurations are those when the corresponding models were initially developed,
but can be generalized for a broader range of scenarios

Problems:
---------
1. CNN:
    1.1. small scale:
        filter length (kernel size, dilation), downsampling (stride),
        these mainly depend on frequency bands of regions of interest,
        like QRS complex, P wave, T wave, or even intervals like qt interval,
        and also finer structures like notches on QRS complexes
    1.2. large scale:
        network depth, and block structures (e.g. ResNet v.s. VGG);
        upsampling?
2. RNN:
    2.1. choice between LSTM and attention
    2.2. use the last state for the last classifying layer or use the whole sequence

Frequency bands (from literature):
----------------------------------
QRS complex: 8 - 25 Hz
P wave: 5 - 20 Hz
T wave: 2.5 - 7 Hz
notch: 30 - 50 Hz (?)
NOTE that different literatures have different conlusions,
the above takes into considerations of many literatures

Frequency bands (from ludb ref. [4]):
-------------------------------------
from the annotations of ludb, the [0.05, 0.95] percentile of the durations of waves are
QRS complex: 70 - 144 ms
P wave: 60 - 134 ms
T wave: 116 - 240 ms
which roughly corr. to the following frequency bands:
QRS complex: 7 - 15 Hz
P wave: 7 - 17 Hz
T wave: 4 - 9 Hz
NOTE that there are records in ludb that there are no onsets (offsets) of certain waves.
in this case, the duration is from the peaks to the offsets (onsets).

according to ref [7], typical kernel sizes are 8 and 9

References:
-----------
[1] Lin, Chia-Hung. "Frequency-domain features for ECG beat discrimination using grey relational analysis-based classifier." Computers & Mathematics with Applications 55.4 (2008): 680-690.
[2] Elgendi, Mohamed, Mirjam Jonkman, and Friso De Boer. "Frequency Bands Effects on QRS Detection." BIOSIGNALS 2003 (2010): 2002.
[3] Tereshchenko, Larisa G., and Mark E. Josephson. "Frequency content and characteristics of ventricular conduction." Journal of electrocardiology 48.6 (2015): 933-937.
[4] https://physionet.org/content/ludb/1.0.0/
[5] Kalyakulina, Alena, et al. "Lobachevsky University Electrocardiography Database" (version 1.0.0). PhysioNet (2020), https://doi.org/10.13026/qweb-sr17.
[6] Kalyakulina, A.I., Yusipov, I.I., Moskalenko, V.A., Nikolskiy, A.V., Kozlov, A.A., Kosonogov, K.A., Zolotykh, N.Yu., Ivanchenko, M.V.: LU electrocardio-graphy database: a new open-access validation tool for delineation algorithms
[7] Moskalenko, Viktor, Nikolai Zolotykh, and Grigory Osipov. "Deep Learning for ECG Segmentation." International Conference on Neuroinformatics. Springer, Cham, 2019.
"""

from .attn import *
from .cnn import *
from .rnn import *
from .ecg_crnn import *
from .ecg_unet import *
from .ecg_seq_lab_net import *
from .ecg_subtract_unet import *
from .ecg_yolo import *
from .rr_lstm import *
# from .ati_cnn import *
# from .cpsc import *


__all__ = [s for s in dir() if not s.startswith('_')]
