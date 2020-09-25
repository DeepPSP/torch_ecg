# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 21:35:48 2017

@author: hsd
"""

import numpy as np
from scipy import stats
from scipy.signal import periodogram
import math
import ReadData
#import pywt


##################################################
### tools
##################################################
def zigzag(ts):
    '''
    number of zigzag
    '''
    num_zigzag = 1
    for i in range(len(ts)-2):
        num_1 = ts[i]
        num_2 = ts[i+1]
        num_3 = ts[i+2]
        if (num_2 - num_1) * (num_3 - num_2) < 0:
            num_zigzag += 1
    return num_zigzag

def autocorr(ts, t):
    return np.corrcoef(np.array([ts[0:len(ts)-t], ts[t:len(ts)]]))[0,1]

##################################################
### get features
##################################################
def LongBasicStat(ts):
    '''
    ### 1
    TODO: 
    
    1. why too much features will decrease F1
    2. how about add them and feature filter before xgb
    
    '''

    global feature_list

    Range = max(ts) - min(ts)
    Var = np.var(ts)
    Skew = stats.skew(ts)
    Kurtosis = stats.kurtosis(ts)
    Median = np.median(ts)
#    p_001 = np.percentile(ts, 0.01)
#    p_002 = np.percentile(ts, 0.02)
#    p_005 = np.percentile(ts, 0.05)
#    p_01 = np.percentile(ts, 0.1)
#    p_02 = np.percentile(ts, 0.2)
#    p_05 = np.percentile(ts, 0.5)
    p_1 = np.percentile(ts, 1)
#    p_2 = np.percentile(ts, 2)
    p_5 = np.percentile(ts, 5)
    p_10 = np.percentile(ts, 10)
    p_25 = np.percentile(ts, 25)
    p_75 = np.percentile(ts, 75)
    p_90 = np.percentile(ts, 90)
    p_95 = np.percentile(ts, 95)
#    p_98 = np.percentile(ts, 98)
    p_99 = np.percentile(ts, 99)
#    p_995 = np.percentile(ts, 99.5)
#    p_998 = np.percentile(ts, 99.8)
#    p_999 = np.percentile(ts, 99.9)
#    p_9995 = np.percentile(ts, 99.95)
#    p_9998 = np.percentile(ts, 99.98)
#    p_9999 = np.percentile(ts, 99.99)

    range_99_1 = p_99 - p_1
    range_95_5 = p_95 - p_5
    range_90_10 = p_90 - p_10
    range_75_25 = p_75 - p_25
    
#    return [Range, Var, Skew, Kurtosis, Median]

#    return [Range, Var, Skew, Kurtosis, Median, 
#            p_1, p_5, p_95, p_99]
    
    feature_list.extend(['LongBasicStat_Range', 
                         'LongBasicStat_Var', 
                        'LongBasicStat_Skew', 
                        'LongBasicStat_Kurtosis', 
                        'LongBasicStat_Median', 
                        'LongBasicStat_p_1', 
                        'LongBasicStat_p_5', 
                        'LongBasicStat_p_95', 
                        'LongBasicStat_p_99', 
                        'LongBasicStat_p_10', 
                        'LongBasicStat_p_25', 
                        'LongBasicStat_p_75', 
                        'LongBasicStat_p_90', 
                        'LongBasicStat_range_99_1', 
                        'LongBasicStat_range_95_5', 
                        'LongBasicStat_range_90_10', 
                        'LongBasicStat_range_75_25'])
    return [Range, Var, Skew, Kurtosis, Median, 
            p_1, p_5, p_95, p_99, 
            p_10, p_25, p_75, p_90, 
            range_99_1, range_95_5, range_90_10, range_75_25]
    
#    return [Range, Var, Skew, Kurtosis, Median, 
#            p_001, p_002, p_005, p_01, p_02, p_05, p_1, p_2, p_5, 
#            p_10, p_25, p_75, p_90, 
#            p_95, p_98, p_99, p_995, p_998, p_999, p_9995, p_9998, p_9999]

def LongZeroCrossing(ts, thres):
    '''
    ### 2
    '''
    global feature_list

    cnt = 0
    for i in range(len(ts)-1):
        if (ts[i] - thres) * (ts[i+1] - thres) < 0:
            cnt += 1
        if ts[i] == thres and (ts[i-1] - thres) * (ts[i+1] - thres) < 0:
            cnt += 1
    feature_list.extend(['LongZeroCrossing_cnt'])
    return [cnt]
    
def LongFFTBandPower(ts):
    '''
    ### 3
    return list of power of each freq band
    
    TODO: different band cut method
    '''
    global feature_list

    fs = 300
    nfft = len(ts)
    partition = [0, 1.5, 4, 8, 20, 100, fs/2]
    f, psd = periodogram(ts, fs)
    partition = [int(x * nfft / fs) for x in partition]
    p = [sum(psd[partition[x] : partition[x + 1]]) for x in range(len(partition)-1)]
    
    feature_list.extend(['LongFFTBandPower_'+str(i) for i in range(len(p))])

    return p

def LongFFTPower(ts):
    '''
    ### 4
    return power
    
    no effect
    '''
    global feature_list

    psd = periodogram(ts, fs=300.0, nfft=4500)
    power = np.sum(psd[1])
    feature_list.extend(['LongFFTPower_power'])
    return [power]

def LongFFTBandPowerShannonEntropy(ts):
    '''
    ### 5
    return entropy of power of each freq band
    refer to scipy.signal.periodogram
    
    TODO: different band cut method
    '''
    global feature_list

    fs = 300
    nfft = len(ts)
    partition = [0, 1.5, 4, 8, 20, 100, fs/2]
    f, psd = periodogram(ts, fs)
    partition = [int(x * nfft / fs) for x in partition]
    p = [sum(psd[partition[x] : partition[x + 1]]) for x in range(len(partition)-1)]
    prob = [x / sum(p) for x in p]
    entropy = sum([- x * math.log(x) for x in prob])
    feature_list.extend(['LongFFTBandPowerShannonEntropy_entropy'])
    return [entropy]

def LongSNR(ts):
    '''
    ### 6
    TODO
    '''
    global feature_list

    psd = periodogram(ts, fs=300.0)

    signal_power = 0
    noise_power = 0
    for i in range(len(psd[0])):
        if psd[0][i] < 5:
            signal_power += psd[1][i]
        else:
            noise_power += psd[1][i]
    
    feature_list.extend(['LongSNR_snr'])
      
    return [signal_power / noise_power]

def long_autocorr(ts):
    '''
    ### 7
    '''
    feat = []
    num_lag = 12
    
    global feature_list
    feature_list.extend(['long_autocorr_'+str(i) for i in range(num_lag)])
    
    for i in range(num_lag):
        feat.append(autocorr(ts, i))

    return feat

def long_zigzag(ts):
    '''
    ### 8
    '''
    feature_list.extend(['long_zigzag'])
    num_zigzag = zigzag(ts)
    return [num_zigzag]

def LongThresCrossing(ts):
    '''
    ### 9
    '''
    thres = np.mean(ts)
    global feature_list

    cnt = 0
    pair_flag = 1
    pre_loc = 0
    width = []
    for i in range(len(ts)-1):
        if (ts[i] - thres) * (ts[i+1] - thres) < 0:
            cnt += 1
            if pair_flag == 1:
                width.append(i-pre_loc)
                pair_flag = 0
            else:
                pair_flag = 1
                pre_loc = i
        if ts[i] == thres and (ts[i-1] - thres) * (ts[i+1] - thres) < 0:
            cnt += 1
    
    feature_list.extend(['LongThresCrossing_cnt', 'LongThresCrossing_width'])
    if len(width) > 1:
        return [cnt, np.mean(width)]
    else:
        return [cnt, 0.0]

def WaveletStat(ts):
    '''
    Statistic features for DWT
    '''

    global feature_list

    DWTfeat = []
    feature_list.extend(['WaveletStat_'+str(i) for i in range(48)])
    if len(ts) >= 1664:
        db7 = pywt.Wavelet('db7')      
        cAcD = pywt.wavedec(ts, db7, level = 7)
        for i in range(8):
            DWTfeat = DWTfeat + [max(cAcD[i]), min(cAcD[i]), np.mean(cAcD[i]),
                                    np.median(cAcD[i]), np.std(cAcD[i])]
            energy = 0
            for j in range(len(cAcD[i])):
                energy = energy + cAcD[i][j] ** 2
            DWTfeat.append(energy/len(ts))
        return DWTfeat
    else:
        return [0.0]*48
    
##################################################
###  get all features
##################################################

def get_long_feature(table):
    '''
    rows of table is 8000+
    '''
    
    global feature_list
    feature_list = []

    
    print('extract long begin')
    features = []
    step = 0
    for ts in table:
        row = []

        ### yes
        ### 1
        row.extend(LongBasicStat(ts))
        ### 2
        row.extend(LongZeroCrossing(ts,0))
        ### 3
        row.extend(LongFFTBandPower(ts))
        ### 4
        row.extend(LongFFTPower(ts))
        ### 5
        row.extend(LongFFTBandPowerShannonEntropy(ts))
        ### 6
        row.extend(LongSNR(ts))
        ### 7
        row.extend(long_autocorr(ts))
        ### 8
        row.extend(long_zigzag(ts))
        ### 9
        row.extend(LongThresCrossing(ts))

        ### no
#        row.extend(WaveletStat(ts))
#        row.extend(RandomNum(ts))
#        row.extend(LongARIMACoeff(ts))

        features.append(row)
        
        step += 1
        if step % 1000 == 0:
            print('extracting ...', step)
            # break
        
    print('extract long DONE')
    
    return feature_list, features


if __name__ == '__main__':
    long_pid, long_data, long_label = ReadData.ReadData( '../../data1/long.csv' )
    tmp_features = get_long_feature(long_data[:1])
    print(len(tmp_features[1][0]))
    