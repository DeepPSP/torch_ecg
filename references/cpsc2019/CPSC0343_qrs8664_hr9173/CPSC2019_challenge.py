import numpy as np
import matplotlib.pyplot as plt
from sig_tool import diff, med_smooth

def CPSC2019_challenge(ECG, models):
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
    ECG = ECG[4:4996]

    ECG = np.transpose(np.array([ECG]))
    ECG = np.reshape(ECG, newshape=(1,4992,1))
    seg = models[0].predict(ECG)
    seg = np.argmax(seg, axis=2)
    seg = seg[0]
    ss = [seg[0] for _ in range(4)]
    [ss.append(s) for s in seg]
    [ss.append(seg[-1]) for _ in range(4)]
    seg = np.array(ss)

    # median smoothing to reducec notch errors
    seg = med_smooth(seg, 10)

    qrs_interval = []
    qrs = []

    qrs_start = -1
    qrs_stop = -1
    for idx in range(len(seg)):
        if seg[idx] == 1:
            if qrs_start == -1:
                # new qrs segmentation
                qrs_start = idx
            else:
                continue
        else:
            if qrs_start >= 0:
                qrs_stop = idx
                qrs_interval.append((qrs_start, qrs_stop))

                qrs_start = -1
                qrs_stop = -1
            else:
                continue
    idx = 0
    while idx < len(qrs_interval):
        # searching for
        interval = qrs_interval[idx]
        central = -1
        central = (interval[1]+interval[0])//2
        idx += 1

        # if interval[1]-interval[0] < 20:
        #     # qrs glitch
        #     # searching for next qrs
        #     idx_next = idx + 1
        #     if idx_next == len(qrs_interval):
        #         central = (interval[1]+interval[0])//2
        #         idx += 1
        #     else:
        #         while idx_next < len(qrs_interval):
        #             interval_next = qrs_interval[idx_next]
        #             if interval_next[1]-interval_next[0] < 20:
        #                 if interval_next[1]-interval[0] < 160:
        #                     idx_next += 1
        #                     if idx_next == len(qrs_interval):
        #                         central = (interval_next[1]+interval[0])//2
        #                         idx = idx_next
        #                         break
        #                 else:
        #                     central = (qrs_interval[idx_next-1][1] + interval[0])//2
        #                     idx = idx_next
        #                     break
        #             else:
        #                 central = (interval_next[1] + interval[0])//2
        #                 idx = idx_next + 1
        #                 break
        # else:
        #     central = (interval[1]+interval[0])//2
        #     idx += 1

        '''calibrate central with -20 ms if fir applied'''
        qrs.append(central)

    rr = []
    r_prev = -1
    for r in qrs:
        if r < 5.5*500 or r > 5000 - 0.5*500:
            continue

        if r_prev < 0:
            r_prev = r
        else:
            rr.append(r-r_prev)
            r_prev = r

    if len(rr) == 0:
        hr = 60
    else:
        hr = 60 / (np.mean(rr) / 500)

    # plt.plot(ECG[0])
    # plt.plot(seg)
    # rwaves = np.zeros(len(seg))
    # for r in qrs:
    #     rwaves[r] = 1
    # plt.plot(rwaves)
    # plt.show()

    return np.round(hr), qrs
