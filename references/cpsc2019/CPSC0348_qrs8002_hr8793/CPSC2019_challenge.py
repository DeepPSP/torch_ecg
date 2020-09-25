import os
import numpy as np

from qrs_detector import qrs_detector

#os.environ['CUDA_VISIBLE_DEVICES'] = "1"

resume_model = 'model_best.pth'
detector = qrs_detector(model_path=resume_model, model_type='fcn')

def CPSC2019_challenge(ECG):
    '''
    This function is your detection function, the input parameters is self-definedself.

    INPUT:
    ECG: single ecg data for 10 senonds
    .
    .
    .

    OUTPUT:
    hr: heart rate calculated based on the ecg data
    qrs: R peak location detected beased on the ecg data and your algorithm

    '''

    fs = 500
    # resume_model = './runs/seq2seq_finetune_enhance_batch_size_4_lr_0.001/checkpoints/model_best.pth'


    qrs, _ = detector.run(ECG, fs=fs)
    if len(qrs) > 0:
        r_hr = np.array([loc for loc in qrs if (loc > 5.5 * fs and loc < 10.* fs - 0.5 * fs)])
        hr = round(60 * fs / np.mean(np.diff(r_hr)))
        if np.isnan(hr):
            hr = 0.
    else:
        qrs = np.zeros(1, int)
        hr = 0.

    return hr, qrs
