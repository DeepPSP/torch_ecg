import numpy as np
from scipy.signal import convolve, butter, filtfilt


def postprocess(beat_locs2, margin, wl=100):
    '''
    后处理：滑动窗口积分 + NMS
    :param beat_locs2: FCN网络的输出,数值范围为(0,1)
    :param margin: （滑动窗口大小-1）/ 2
    :param wl: 非极大值抑制的窗口大小，根据0.2s内不出现重复心拍的生理学依据，对于采样率500最好小于100
    :return: 最后得到的心拍位置
    '''
    thres = (margin+1)*0.5 # 窗口中有一般以上的sample大于0.5被认为是心拍候选点
    accum = convolve(beat_locs2, [1]*(margin*2+1), mode='same') # 滑动窗口积分

    beat_locs = []
    ################### 非极大值抑制 ###################
    for idx in range(wl):
        if accum[idx] > thres and accum[idx] == max(accum[ : idx+wl+1]): # 左边缘
            beat_locs.append(idx)
    for idx in range(wl, len(accum)-wl):
        if accum[idx] > thres and accum[idx] == max(accum[idx-wl : idx+wl+1]): # 非边缘
            beat_locs.append(idx)
    for idx in range(len(accum)-wl, len(accum)):
        if accum[idx] > thres and accum[idx] == max(accum[idx-wl : ]): # 右边缘
            beat_locs.append(idx)
    return np.array(beat_locs)


def preprocess(signal, fs, lowcut=0.5, highcut=48., order=3):
    '''
    预处理：通过带通滤波器消除基线漂移和高频噪声
    '''
    nyquist_freq = 0.5 * fs
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    if low>0 and high>0:
        b, a = butter(order, [low, high], btype="bandpass")
    elif high>0:
        b, a = butter(order, high, btype="lowpass")
    else:
        b, a = butter(order, low, btype="highpass")
    filtedSignal = filtfilt(b, a, signal)
    # filtedSignal /= max(abs(filtedSignal))
    return np.array(filtedSignal)


def add_noise(signal, d, SNR):
    '''
    根据信噪比大小，对ECG信号进行加噪
    :param signal: ECG信号记录
    :param d: 噪声记录
    :param SNR: 信噪比
    :return: 加噪后的信号
    '''
    P_signal = np.sum(abs(signal) ** 2)
    P_d = np.sum(abs(d) ** 2)
    P_noise = P_signal / 10 ** (SNR / 10)
    noise_signal = signal + np.sqrt(P_noise / P_d) * d
    return noise_signal


if __name__ == '__main__':
    l = [0.3, 0.3, 0.3, 0.3, 0.0, 0.3, 0.3, 0.3, 0.5, 0.50, 0.3, 0.2]*2
    print(preprocess(np.array(l), fs=500))