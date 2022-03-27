import numpy as np


def CPSC2019_challenge(ECG):
    """
    This function is your detection function, the input parameters is self-definedself.

    INPUT:
    ECG: single ecg data for 10 senonds
    .
    .
    .

    OUTPUT:
    hr: heart rate calculated based on the ecg data
    qrs: R peak location detected beased on the ecg data and your algorithm

    """
    hr = 10
    qrs = np.arange(1, 5000, 500)

    return hr, qrs
