#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 08:50:07 2017

@author: shenda
"""

import numpy as np
from scipy import stats
from scipy import signal
from scipy import fftpack
#import matplotlib.pyplot as plt
# from statsmodels.tsa import arima_model
import ReadData
from sampen2 import sampen2
from copy import deepcopy
import MyEval
from scipy.signal import periodogram
import random
from sklearn.model_selection import StratifiedKFold
import math
from BasicCLF import MyXGB
from BasicCLF import MyLR
from BasicCLF import MyRF
import warnings
from minNCCE import minimumNCCE
import dill
from fastdtw import fastdtw
from sklearn.cluster import SpectralClustering
from collections import Counter
#import pywt

##################################################
### tools
##################################################

def Flatten(l):
    return [item for sublist in l for item in sublist]

def CombineFeatures(table1, table2):
    '''
    table1 and table2 should have the same length
    '''
    table = []
    n_row = len(table1)
    for i in range(n_row):
        table.append(table1[i]+table2[i])
        
    return table

def RandomNum(ts):
    '''
    baseline feature
    '''
    return [random.random()]

def ThreePointsMedianPreprocess(ts):
    '''
    8-beat sliding window RR interval irregularity detector [21]
    '''
    new_ts = []
    for i in range(len(ts)-2):
        new_ts.append(np.median([ts[i], ts[i+1], ts[i+2]]))
    return new_ts

def ExpAvgPreprocess(ts):
    '''
    8-beat sliding window RR interval irregularity detector [21]
    '''
    alpha = 0.02
    new_ts = [0] * len(ts)
    new_ts[0] = ts[0]
    for i in range(1, len(ts)):
        new_ts[i] = new_ts[i-1] + alpha * (ts[i] - new_ts[i-1])
    
    return new_ts

def Heaviside(x):
    if x < 0:
        return 0.0
    elif x == 0:
        return 0.5
    else:
        return 1.0

##################################################
### center wave
##################################################
def center_wave_raw(ts):
    length = 200
    feature_list.extend(['center_wave_raw_'+str(i) for i in range(length)])
    resampled_center_wave_raw = resample_unequal(ts, length)
    return resampled_center_wave_raw

def center_wave_basic_stat(ts):
    Max = max(ts)
    Min = min(ts)
    Range = max(ts) - min(ts)
    Var = np.var(ts)
    Skew = stats.skew(ts)
    Kurtosis = stats.kurtosis(ts)
    Median = np.median(ts)

    p_1 = np.percentile(ts, 1)
    p_5 = np.percentile(ts, 5)
    p_10 = np.percentile(ts, 10)
    p_25 = np.percentile(ts, 25)
    p_75 = np.percentile(ts, 75)
    p_90 = np.percentile(ts, 90)
    p_95 = np.percentile(ts, 95)
    p_99 = np.percentile(ts, 99)

    range_99_1 = p_99 - p_1
    range_95_5 = p_95 - p_5
    range_90_10 = p_90 - p_10
    range_75_25 = p_75 - p_25
    

    feature_list.extend(['center_wave_basic_stat_Max', 
                         'center_wave_basic_stat_Min', 
                        'center_wave_basic_stat_Range', 
                         'center_wave_basic_stat_Var', 
                        'center_wave_basic_stat_Skew', 
                        'center_wave_basic_stat_Kurtosis', 
                        'center_wave_basic_stat_Median', 
                        'center_wave_basic_stat_p_1', 
                        'center_wave_basic_stat_p_5', 
                        'center_wave_basic_stat_p_95', 
                        'center_wave_basic_stat_p_99', 
                        'center_wave_basic_stat_p_10', 
                        'center_wave_basic_stat_p_25', 
                        'center_wave_basic_stat_p_75', 
                        'center_wave_basic_stat_p_90', 
                        'center_wave_basic_stat_range_99_1', 
                        'center_wave_basic_stat_range_95_5', 
                        'center_wave_basic_stat_range_90_10', 
                        'center_wave_basic_stat_range_75_25'])
    
    return [Max, Min, Range, Var, Skew, Kurtosis, Median, 
            p_1, p_5, p_95, p_99, 
            p_10, p_25, p_75, p_90, 
            range_99_1, range_95_5, range_90_10, range_75_25]

def center_wave_zero_crossing(ts):
    cnt = 0
    for i in range(len(ts)-1):
        if ts[i] * ts[i+1] < 0:
            cnt += 1
        if ts[i] == 0 and ts[i-1] * ts[i+1] < 0:
            cnt += 1
    feature_list.extend(['center_wave_zero_crossing_cnt'])
    return [cnt]


##################################################
### short
##################################################

def ShortBasicStat(ts):
    Range = max(ts) - min(ts)
    Var = np.var(ts)
    Skew = stats.skew(ts)
    Kurtosis = stats.kurtosis(ts)
    Median = np.median(ts)
    feature_list.extend(['ShortBasicStat_Range', 
                         'ShortBasicStat_Var', 
                         'ShortBasicStat_Skew', 
                         'ShortBasicStat_Kurtosis', 
                         'ShortBasicStat_Median'])
    return [Range, Var, Skew, Kurtosis, Median]

def ShortZeroCrossing(ts):
    cnt = 0
    for i in range(len(ts)-1):
        if ts[i] * ts[i+1] < 0:
            cnt += 1
        if ts[i] == 0 and ts[i-1] * ts[i+1] < 0:
            cnt += 1
    feature_list.extend(['ShortZeroCrossing_cnt'])
    return [cnt]



##################################################
### long
##################################################

def WaveletStat(ts):
    '''
    Statistic features for DWT
    '''
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
    
def LongBasicStat(ts):
    '''
    TODO: 
    
    1. why too much features will decrease F1
    2. how about add them and feature filter before xgb
    
    '''
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
    cnt = 0
    for i in range(len(ts)-1):
        if (ts[i] - thres) * (ts[i+1] - thres) < 0:
            cnt += 1
        if ts[i] == thres and (ts[i-1] - thres) * (ts[i+1] - thres) < 0:
            cnt += 1
    feature_list.extend(['LongZeroCrossing_cnt'])
    return [cnt]
    
def LongThresCrossing(ts, thres):
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
    
def LongFFTBandPower(ts):
    '''
    return list of power of each freq band
    
    TODO: different band cut method
    '''
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
    return power
    
    no effect
    '''
    psd = periodogram(ts, fs=300.0, nfft=4500)
    power = np.sum(psd[1])
    feature_list.extend(['LongFFTPower_power'])
    return [power]

def LongFFTBandPowerShannonEntropy(ts):
    '''
    return entropy of power of each freq band
    refer to scipy.signal.periodogram
    
    TODO: different band cut method
    '''
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
    TODO
    '''
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


##################################################
### QRS
##################################################

def bin_stat_interval(ts):
    '''
    stat of bin Counter RR interval ts
    
    count, ratio
    '''
    pass


def bin_stat(ts):
    '''
    stat of bin Counter RR ts
    
    count, ratio
    '''
    feature_list.extend(['bin_stat_'+str(i) for i in range(52)])

    if len(ts) > 0:
        interval_1 = [1, 4, 8, 16, 32, 64, 128, 240]
        bins_1 = sorted([240 + i for i in interval_1] + [240 - i for i in interval_1], reverse=True)
        
        cnt_1 = [0.0] * len(bins_1)
        for i in ts:
            for j in range(len(bins_1)):
                if i > bins_1[j]:
                    cnt_1[j] += 1
                    break
        ratio_1 = [i/len(ts) for i in cnt_1]    
        
        interval_2 = [8, 32, 64, 128, 240]
        bins_2 = sorted([240 + i for i in interval_2] + [240 - i for i in interval_2], reverse=True)
        
        cnt_2 = [0.0] * len(bins_2)
        for i in ts:
            for j in range(len(bins_2)):
                if i > bins_2[j]:
                    cnt_2[j] += 1
                    break
        ratio_2 = [i/len(ts) for i in cnt_2]
        
        return cnt_1 + ratio_1 + cnt_2 + ratio_2
    else:
        
        return [0.0] * 52
    


def drddc(ts):
    '''
    TODO:
    '''
    pass

    
def SampleEn(ts):
    '''    
    sample entropy on QRS interval
    '''
    ts = [float(i) for i in ts]
    mm = 3
    out = []
    feature_list.extend(['SampleEn_'+str(i) for i in range(mm + 1)])

    if len(ts) >= (mm+1)*2:
        res = sampen2(ts, mm=mm, normalize=True)
        for ii in res:
            if ii[1] is None:
                out.append(100)
            else:
                out.append(ii[1])
        return out
    else:
        return [0] * (mm + 1)
    
    
def CDF(ts):
    '''
    analysis of cumulative distribution functions [17],
    '''
    n_bins = 60
    hist, _ = np.histogram(ts, range=(100, 400), bins=n_bins)
    cdf = np.cumsum(hist)/len(ts)
    cdf_density = np.sum(cdf) / n_bins
    feature_list.extend(['CDF_cdf_density'])
    return [cdf_density]
    
def CoeffOfVariation(ts):
    '''
    analysis of cumulative distribution functions [17],
    '''
    if len(ts) >= 3:
        tmp_ts = ts[1:-1]
        if np.mean(tmp_ts) == 0:
            coeff_ts = 0.0
        else:
            coeff_ts = np.std(tmp_ts) / np.mean(tmp_ts)
    else:
        coeff_ts = 0.0
    
    if len(ts) >= 4:
        tmp_ts = ts[1:-1]
        tmp_ts = np.diff(tmp_ts)
        if np.mean(tmp_ts) == 0:
            coeff_dts = 0.0
        else:
            coeff_dts = np.std(tmp_ts) / np.mean(tmp_ts)
    else:
        coeff_dts = 0.0
    
    feature_list.extend(['CoeffOfVariation_coeff_ts', 'CoeffOfVariation_coeff_dts'])
    return [coeff_ts, coeff_dts]

def MAD(ts):
    '''
    thresholding on the median absolute deviation (MAD) of RR intervals [18] 
    '''
    ts_median = np.median(ts)
    mad = np.median([np.abs(ii - ts_median) for ii in ts])
    feature_list.extend(['MAD_mad'])
    return [mad]


def QRSBasicStat(ts):
    
    feature_list.extend(['QRSBasicStat_Mean', 
                        'QRSBasicStat_HR', 
                        'QRSBasicStat_Count', 
                        'QRSBasicStat_Range', 
                        'QRSBasicStat_Var', 
                        'QRSBasicStat_Skew', 
                        'QRSBasicStat_Kurtosis', 
                        'QRSBasicStat_Median', 
                        'QRSBasicStat_Min', 
                        'QRSBasicStat_p_5', 
                        'QRSBasicStat_p_25', 
                        'QRSBasicStat_p_75', 
                        'QRSBasicStat_p_95', 
                        'QRSBasicStat_range_95_5', 
                        'QRSBasicStat_range_75_25'])
    
    if len(ts) >= 3:
        
        ts = ts[1:-1]
        
        Mean = np.mean(ts)
        if Mean == 0:
            HR = 0
        else:
            HR = 1 / Mean
        Count = len(ts)
        Range = max(ts) - min(ts)
        Var = np.var(ts)
        Skew = stats.skew(ts)
        Kurtosis = stats.kurtosis(ts)
        Median = np.median(ts)
        Min = min(ts)
        p_5 = np.percentile(ts, 5)
        p_25 = np.percentile(ts, 25)
        p_75 = np.percentile(ts, 75)
        p_95 = np.percentile(ts, 95)
        range_95_5 = p_95 - p_5
        range_75_25 = p_75 - p_25

        return [Mean, HR, Count, Range, Var, Skew, Kurtosis, Median, Min, 
                p_5, p_25, p_75, p_95, 
                range_95_5, range_75_25]
    
    else:
        return [0.0] * 15
    
def QRSBasicStatPointMedian(ts):
    
    feature_list.extend(['QRSBasicStatPointMedian_Mean', 
                         'QRSBasicStatPointMedian_HR', 
                         'QRSBasicStatPointMedian_Count', 
                         'QRSBasicStatPointMedian_Range', 
                         'QRSBasicStatPointMedian_Var', 
                         'QRSBasicStatPointMedian_Skew',
                         'QRSBasicStatPointMedian_Kurtosis',
                         'QRSBasicStatPointMedian_Median',
                         'QRSBasicStatPointMedian_Min',
                         'QRSBasicStatPointMedian_p_25',
                         'QRSBasicStatPointMedian_p_75'])
    
    ts = ThreePointsMedianPreprocess(ts)
    
    Mean = np.mean(ts)
    if Mean == 0:
        HR = 0
    else:
        HR = 1 / Mean

    Count = len(ts)
    if Count != 0:
        Range = max(ts) - min(ts)
        Var = np.var(ts)
        Skew = stats.skew(ts)
        Kurtosis = stats.kurtosis(ts)
        Median = np.median(ts)
        Min = min(ts)
        p_25 = np.percentile(ts, 25)
        p_75 = np.percentile(ts, 75)
    else:
        Range = 0.0
        Var = 0.0
        Skew = 0.0
        Kurtosis = 0.0
        Median = 0.0
        Min = 0.0
        p_25 = 0.0
        p_75 = 0.0
    
    return [Mean, HR, Count, Range, Var, Skew, Kurtosis, Median, Min, p_25, p_75]

      
def QRSBasicStatDeltaRR(ts):
    
    feature_list.extend(['QRSBasicStatDeltaRR_Mean', 
                        'QRSBasicStatDeltaRR_HR', 
                        'QRSBasicStatDeltaRR_Count', 
                        'QRSBasicStatDeltaRR_Range', 
                        'QRSBasicStatDeltaRR_Var', 
                        'QRSBasicStatDeltaRR_Skew', 
                        'QRSBasicStatDeltaRR_Kurtosis', 
                        'QRSBasicStatDeltaRR_Median', 
                        'QRSBasicStatDeltaRR_Min', 
                        'QRSBasicStatDeltaRR_p_25', 
                        'QRSBasicStatDeltaRR_p_75'])
    
    if len(ts) >= 4:
        ts = ts[1:-1]
        ts = np.diff(ts)
        
        Mean = np.mean(ts)
        if Mean == 0:
            HR = 0
        else:
            HR = 1 / Mean
        Count = len(ts)
        Range = max(ts) - min(ts)
        Var = np.var(ts)
        Skew = stats.skew(ts)
        Kurtosis = stats.kurtosis(ts)
        Median = np.median(ts)
        Min = min(ts)
        p_25 = np.percentile(ts, 25)
        p_75 = np.percentile(ts, 75)
        return [Mean, HR, Count, Range, Var, Skew, Kurtosis, Median, Min, p_25, p_75]
    
    else:
        return [0.0] * 11
    

def QRSYuxi(ts):
    '''
    pars: 
        tol = 0.05
            define if two QRS intervals are matched
    '''
    tol = 0.05
    feature_list.extend(['QRSYuxi'])

    
    if len(ts) >= 3:
        ts = ts[1:-1]
    
        avg_RR = np.median(ts)
        matched = [False] * len(ts)
        
        for i in range(len(ts)):
            seg_1 = ts[i]
            if abs(seg_1 - avg_RR) / avg_RR <= tol:
                matched[i] = True
            elif abs(seg_1 - 2 * avg_RR) / (2 * avg_RR) <= tol:
                matched[i] = True
                
        for i in range(len(ts)):
            if matched[i] is False:
                if i == 0:
                    seg_2_forward = ts[i]
                else:
                    seg_2_forward = ts[i-1] + ts[i]
                if i == len(ts)-1:
                    seg_2_backward = ts[i]
                else:
                    seg_2_backward = ts[i] + ts[i+1]
                    
                if abs(seg_2_forward - 2 * avg_RR) / (2 * avg_RR) <= tol:
                    matched[i] = True
                elif abs(seg_2_forward - 3 * avg_RR) / (3 * avg_RR) <= tol:
                    matched[i] = True
                elif abs(seg_2_backward - 2 * avg_RR) / (2 * avg_RR) <= tol:
                    matched[i] = True
                elif abs(seg_2_backward - 3 * avg_RR) / (3 * avg_RR) <= tol:
                    matched[i] = True
    
        return [sum(matched) / len(matched)]

    else:
        return [0.0] * 1
    

def Variability(ts):
    '''
    Variability(Time Domain) & Poincare plot
    https://zh.wikipedia.org/wiki/%E5%BF%83%E7%8E%87%E8%AE%8A%E7%95%B0%E5%88%86%E6%9E%90
    compute SDNN, NN50 count, pNN50
    [14] Atrial fibrillation detection by heart rate variability in Poincare plot
    Stepping: the mean stepping increment of the inter-beat intervals
    Dispersion: how spread the points in Poincaré plot are distributed around the diagonal line
    '''
    feature_list.extend(['Variability_SDNN', 
                        'Variability_NN50', 
                        'Variability_pNN50', 
                        'Variability_Stepping', 
                        'Variability_Dispersion'])

    if len(ts) >= 3:
        ts = ts[1:-1]
        SDNN = np.std(ts)
        freq = 300
        timelen = freq * 0.05
        if len(ts) < 3:
            NN50 = 0
            pNN50 = 0
            Stepping = 0
            Dispersion = 0
        else:
            NN = [abs(ts[x + 1] - ts[x]) for x in range(len(ts) - 1)]
            NN50 = sum([x > timelen for x in NN])
            pNN50 = float(NN50) / len(ts)
            Stepping = (sum([(NN[x] ** 2 + NN[x + 1] ** 2) ** 0.5 for x in range(len(NN) - 1)]) / (len(NN) - 1)) / (sum(ts) / len(ts))
            Dispersion = (sum([x ** 2 for x in NN]) / (2 * len(NN)) - sum(NN) ** 2 / (2 * (len(NN)) ** 2)) ** 0.5 / ((-ts[0] - 2 * ts[-1] + 2 * sum(ts)) / (2 * len(NN)))
        return [SDNN, NN50, pNN50, Stepping, Dispersion]
    
    else:
        return [0.0] * 5

##################################################
###  get features NO!!!!!
##################################################
def GetShortFeature(table):
    '''
    rows of table is 330000+
    
    no use now
    '''
    features = []
    step = 0
    for ts in table:
        row = []

        row.extend(ShortBasicStat(ts))
#        row.extend(ShortZeroCrossing(ts))
        
        features.append(row)
        
        step += 1
        if step % 100000 == 0:
            print('extracting ...')
        
    print('extract DONE')
    
    return features


##################################################
###  center waves
###  get features YES!!!!!
###  very slow, do not run everytime
##################################################
def dist(ts1, ts2):
    dist_num = np.linalg.norm(np.array(ts1) - np.array(ts2))
    return dist_num

def resample_unequal(ts, length):
    resampled = [0.0] * length
    resampled_idx = list(np.linspace(0.0, len(ts)-1, length))
    for i in range(length):
        idx_i = resampled_idx[i]
        low_idx = int(np.floor(idx_i))
        low_weight = abs(idx_i - np.ceil(idx_i))
        high_idx = int(np.ceil(idx_i))
        high_weight = abs(idx_i - np.floor(idx_i))
        resampled[i] = low_weight * ts[low_idx] + high_weight * ts[high_idx]
#        print(idx_i, resampled[i], low_weight, high_weight)
#        break
    return resampled
        
#if __name__ == '__main__':


def GetShortCenterWave(table, pid_list, long_pid_list):
#    table = short_data
#    pid_list = short_pid
#    long_pid_list = QRS_pid
    '''
    find majority mean short wave, or center wave, resample to fixed length
    
    pars:
        resampled_length
        n_clusters
        radius
    
    '''
    print('extract GetShortMajorityMeanWave begin')
    fout = open('../data/center_wave_euclid_direct.csv', 'w')
    resampled_length = 1000
    n_clusters = 3
    radius = 1

    features = []
    pid_short_dic = {}
    
    ### build pid_short_dic: (pid -> short_waves)
    for i in range(len(pid_list)):
        if pid_list[i] in pid_short_dic.keys():
            pid_short_dic[pid_list[i]].append(table[i])
        else:
            pid_short_dic[pid_list[i]] = [table[i]]
                
    step = 0
    for pid in long_pid_list:
        ### select pid who has more than 2 short waves
        if pid in pid_short_dic.keys() and len(pid_short_dic[pid]) > 5:
            ### sub_table contains all short_waves of pid
            sub_table = pid_short_dic[pid]
            sub_table_resampled = [resample_unequal(i, resampled_length) for i in sub_table]
            
            ### construct distance matrix of short waves
            n_short = len(sub_table)
            dist_mat = np.zeros([n_short, n_short])
            for i in range(n_short):
                dist_mat[i, i] = 0.0
                for j in range(i+1, n_short):
#                    tmp_dist = fastdtw(sub_table[i], sub_table[j], radius=radius)[0]
                    tmp_dist = dist(sub_table_resampled[i], sub_table_resampled[j])
                    dist_mat[i, j] = tmp_dist
                    dist_mat[j, i] = tmp_dist
            
            dist_mat_dist = np.sum(dist_mat, axis=1)
            resampled_center_wave = sub_table_resampled[np.argsort(dist_mat_dist)[0]]
            
            
            ### clustering vis distance matrix
#            sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
#            sc.fit(dist_mat)
#            ### find the most common labels
#            majority_labels = Counter(sc.labels_).most_common(1)[0][0]
#            ###### TODO: how to find the center of a dist matrix or a graph
#            ### selected_short_idx is int
#            selected_short_idx = np.array(list(range(n_short)))[np.array(sc.labels_) == majority_labels]
#            ### sub array of dists sum
#            majority_dist_mat_dist = np.sum(dist_mat, axis=1)[selected_short_idx]
#            ### min dists sum of sub array
#            majority_mean_idx = selected_short_idx[np.argsort(majority_dist_mat_dist)[0]]
##            center_wave = sub_table[majority_mean_idx]
#            resampled_center_wave = sub_table_resampled[majority_mean_idx]
#            
            ###### TODO: resample
#            resampled_idx = [int(i) for i in list(np.linspace(0.0, len(center_wave)-1, resampled_length))]
#            resampled_center_wave = [center_wave[i] for i in resampled_idx]
            
        else:
            resampled_center_wave = [0.0] * resampled_length
        
        features.append(resampled_center_wave)
        fout.write(pid + ',' + ','.join([str(i) for i in resampled_center_wave]) + '\n')
        
        step += 1
        print(step)
        ### for test
#        plt.plot(resampled_center_wave)
#        break
    
    
        if step % 100 == 0:
            print('extracting ...', step)
#            break
#    feature_list.extend(['GetShortCenterWave_'+str(i) for i in range(resampled_length)])
    print('extract GetShortCenterWave DONE')
    fout.close()

#    return features


##################################################
###  get features YES!!!!!
##################################################

def GetCenterWaveFeature(table):
    '''
    rows of table is 8000+
    '''
    print('extract center wave feature begin')
    features = []
    step = 0
    for ts in table:
        row = []
        
        ### yes
        row.extend(center_wave_raw(ts))
        row.extend(center_wave_zero_crossing(ts))
        row.extend(center_wave_basic_stat(ts))
        
        
        ### no
        

        features.append(row)
        
        step += 1
        if step % 1000 == 0:
            print('extracting ...', step)
        
    print('extract center wave feature DONE')
    
    return features


def GetShortStatWaveFeature(table, pid_list, long_pid_list):
    '''
    short stat feature, actually long feature
    
    Electrocardiogram Feature Extraction and Pattern Recognition Using a Novel Windowing Algorithm
    
    row of out feature is 8000+
    
    TODO: more on how to detect PT waves
    '''
    print('extract GetShortStatWaveFeature begin')

    features = []
    pid_short_dic = {}
    
                
    ### no-preprocess, performs better
    for i in range(len(pid_list)):
        if pid_list[i] in pid_short_dic.keys():
            pid_short_dic[pid_list[i]].append(table[i])
        else:
            pid_short_dic[pid_list[i]] = [table[i]]
                
    step = 0
    for pid in long_pid_list:
        if pid in pid_short_dic.keys() and len(pid_short_dic[pid])-2 > 0:
            
            ### init
            PR_interval_list = []
            QRS_duration_list = []
            QT_interval_list = []
            QT_corrected_list = []
            vent_rate_list = []
            
            RQ_amp_list = []
            RS_amp_list = []
            ST_amp_list = []
            PQ_amp_list = []
            QS_amp_list = []
            RP_amp_list = []
            RT_amp_list = []
            ST_interval_list = []
            RS_interval_list = []
            T_peak_list = []
            P_peak_list = []
            Q_peak_list = []
            R_peak_list = []
            S_peak_list = []
            RS_slope_list = []
            ST_slope_list = []
            NF_list = []
            Fwidth_list = []
            
            ### select short data of one patient
            sub_table = pid_short_dic[pid]
            
            for i in range(len(sub_table)-2):
                prev_ts = sub_table[i]
                ts = sub_table[i+1]
                
                ### select each short data
                T_start = round(0.15 * len(ts))
                T_end = round(0.55 * len(ts))
                P_start = round(0.65 * len(ts))
                P_end = round(0.95 * len(ts))
                
                T_wave = ts[T_start:T_end]
                P_wave = ts[P_start:P_end]
                
                T_peak = max(T_wave)
                P_peak = max(P_wave)
                Q_peak = min(prev_ts[-6:])
                R_peak = ts[0]
                S_peak = min(ts[:6])
                
                T_loc = np.argmax(T_wave)
                P_loc = np.argmax(P_wave)
                Q_loc = -np.argmin(prev_ts[-6:])
                R_loc = 0
                S_loc = np.argmin(ts[:6])
                                
                
                ### features (5)
                PR_interval = P_loc - 0
                QRS_duration = S_loc - Q_loc
                QT_interval = T_loc - Q_loc
                QT_corrected = QT_interval / len(ts)
                if QRS_duration == 0:
                    vent_rate = 0
                else:
                    vent_rate = 1 / QRS_duration
                
                ### number of f waves (2)
                TQ_interval = ts[T_loc:Q_loc]
                thres = np.mean(TQ_interval) + (T_peak - np.mean(TQ_interval))/50
                NF, Fwidth = LongThresCrossing(TQ_interval, thres)
                    
                ### more features (16)
                RQ_amp = R_peak - Q_peak
                RS_amp = R_peak - S_peak
                ST_amp = T_peak - S_peak
                PQ_amp = P_peak - Q_peak
                QS_amp = Q_peak - S_peak
                RP_amp = R_peak - P_peak
                RT_amp = R_peak - T_peak
                
                ST_interval = T_loc - S_loc
                RS_interval = S_loc - R_loc
                
                T_peak = T_peak
                P_peak = P_peak
                Q_peak = Q_peak
                R_peak = R_peak
                S_peak = S_peak
                
                if RS_interval == 0:
                    RS_slope = 0
                else:
                    RS_slope = RS_amp / RS_interval                
                if ST_interval == 0:
                    ST_slope = 0
                else:
                    ST_slope = ST_amp / ST_interval
                
                
                ### 
                PR_interval_list.append(PR_interval)
                QRS_duration_list.append(QRS_duration)
                QT_interval_list.append(QT_interval)
                QT_corrected_list.append(QT_corrected)
                vent_rate_list.append(vent_rate)
                
                NF_list.append(NF)
                Fwidth_list.append(Fwidth)

                RQ_amp_list.append(RQ_amp)
                RS_amp_list.append(RS_amp)
                ST_amp_list.append(ST_amp)
                PQ_amp_list.append(PQ_amp)
                QS_amp_list.append(QS_amp)
                RP_amp_list.append(RP_amp)
                RT_amp_list.append(RT_amp)
                ST_interval_list.append(ST_interval)
                RS_interval_list.append(RS_interval)
                T_peak_list.append(T_peak)
                P_peak_list.append(P_peak)
                Q_peak_list.append(Q_peak)
                R_peak_list.append(R_peak)
                S_peak_list.append(S_peak)
                RS_slope_list.append(RS_slope)
                ST_slope_list.append(ST_slope)


            features.append([np.mean(PR_interval_list), 
                             np.mean(QRS_duration_list), 
                             np.mean(QT_interval_list), 
                             np.mean(QT_corrected_list), 
                             np.mean(vent_rate_list),
                             np.mean(RQ_amp_list),
                             np.mean(RS_amp_list),
                             np.mean(ST_amp_list),
                             np.mean(PQ_amp_list),
                             np.mean(QS_amp_list),
                             np.mean(RP_amp_list),
                             np.mean(RT_amp_list),
                             np.mean(ST_interval_list),
                             np.mean(RS_interval_list),
                             np.mean(T_peak_list),
                             np.mean(P_peak_list),
                             np.mean(Q_peak_list),
                             np.mean(R_peak_list),
                             np.mean(S_peak_list),
                             np.mean(RS_slope_list),
                             np.mean(ST_slope_list),
                             np.mean(NF_list),
                             np.mean(Fwidth_list), 
                             
                             np.max(PR_interval_list), 
                             np.max(QRS_duration_list), 
                             np.max(QT_interval_list), 
                             np.max(QT_corrected_list), 
                             np.max(vent_rate_list),
                             np.max(RQ_amp_list),
                             np.max(RS_amp_list),
                             np.max(ST_amp_list),
                             np.max(PQ_amp_list),
                             np.max(QS_amp_list),
                             np.max(RP_amp_list),
                             np.max(RT_amp_list),
                             np.max(ST_interval_list),
                             np.max(RS_interval_list),
                             np.max(T_peak_list),
                             np.max(P_peak_list),
                             np.max(Q_peak_list),
                             np.max(R_peak_list),
                             np.max(S_peak_list),
                             np.max(RS_slope_list),
                             np.max(ST_slope_list),
                             np.max(NF_list),
                             np.max(Fwidth_list), 
                             
                             np.min(PR_interval_list), 
                             np.min(QRS_duration_list), 
                             np.min(QT_interval_list), 
                             np.min(QT_corrected_list), 
                             np.min(vent_rate_list),
                             np.min(RQ_amp_list),
                             np.min(RS_amp_list),
                             np.min(ST_amp_list),
                             np.min(PQ_amp_list),
                             np.min(QS_amp_list),
                             np.min(RP_amp_list),
                             np.min(RT_amp_list),
                             np.min(ST_interval_list),
                             np.min(RS_interval_list),
                             np.min(T_peak_list),
                             np.min(P_peak_list),
                             np.min(Q_peak_list),
                             np.min(R_peak_list),
                             np.min(S_peak_list),
                             np.min(RS_slope_list),
                             np.min(ST_slope_list),
                             np.min(NF_list),
                             np.min(Fwidth_list), 
                             
                             np.percentile(PR_interval_list, 25), 
                             np.percentile(QRS_duration_list, 25), 
                             np.percentile(QT_interval_list, 25), 
                             np.percentile(QT_corrected_list, 25), 
                             np.percentile(vent_rate_list, 25),
                             np.percentile(RQ_amp_list, 25),
                             np.percentile(RS_amp_list, 25),
                             np.percentile(ST_amp_list, 25),
                             np.percentile(PQ_amp_list, 25),
                             np.percentile(QS_amp_list, 25),
                             np.percentile(RP_amp_list, 25),
                             np.percentile(RT_amp_list, 25),
                             np.percentile(ST_interval_list, 25),
                             np.percentile(RS_interval_list, 25),
                             np.percentile(T_peak_list, 25),
                             np.percentile(P_peak_list, 25),
                             np.percentile(Q_peak_list, 25),
                             np.percentile(R_peak_list, 25),
                             np.percentile(S_peak_list, 25),
                             np.percentile(RS_slope_list, 25),
                             np.percentile(ST_slope_list, 25),
                             np.percentile(NF_list, 25),
                             np.percentile(Fwidth_list, 25), 
                             
                             np.percentile(PR_interval_list, 75), 
                             np.percentile(QRS_duration_list, 75), 
                             np.percentile(QT_interval_list, 75), 
                             np.percentile(QT_corrected_list, 75), 
                             np.percentile(vent_rate_list, 75),
                             np.percentile(RQ_amp_list, 75),
                             np.percentile(RS_amp_list, 75),
                             np.percentile(ST_amp_list, 75),
                             np.percentile(PQ_amp_list, 75),
                             np.percentile(QS_amp_list, 75),
                             np.percentile(RP_amp_list, 75),
                             np.percentile(RT_amp_list, 75),
                             np.percentile(ST_interval_list, 75),
                             np.percentile(RS_interval_list, 75),
                             np.percentile(T_peak_list, 75),
                             np.percentile(P_peak_list, 75),
                             np.percentile(Q_peak_list, 75),
                             np.percentile(R_peak_list, 75),
                             np.percentile(S_peak_list, 75),
                             np.percentile(RS_slope_list, 75),
                             np.percentile(ST_slope_list, 75),
                             np.percentile(NF_list, 75),
                             np.percentile(Fwidth_list, 75)
                             ])
    
        else:
            features.append([0.0] * ((5+16+2) * 5))

        step += 1
        if step % 1000 == 0:
            print('extracting ...', step)
    feature_list.extend(['GetShortStatWaveFeature_'+str(i) for i in range((5+16+2) * 5)])
    print('extract GetShortStatWaveFeature DONE')

    return features


def GetLongFeature(table):
    '''
    rows of table is 8000+
    '''
    print('extract long begin')
    features = []
    step = 0
    for ts in table:
        row = []

        ### yes
#        row.extend(WaveletStat(ts))
        row.extend(LongBasicStat(ts))
        row.extend(LongZeroCrossing(ts,0))
        row.extend(LongFFTBandPower(ts))
        row.extend(LongFFTPower(ts))
        row.extend(LongFFTBandPowerShannonEntropy(ts))
        row.extend(LongSNR(ts))
     
        
        
        ### no
#        row.extend(RandomNum(ts))
#        row.extend(LongARIMACoeff(ts))

        features.append(row)
        
        step += 1
        if step % 1000 == 0:
            print('extracting ...', step)
        
    print('extract long DONE')
    
    return features
    
def GetQRSFeature(table):
    '''
    rows of table is 8000+
    '''
    print('extract QRS begin')
    features = []
    step = 0
    for ts in table:
        row = []
 
        ### yes
        row.extend(SampleEn(ts))
        row.extend(CDF(ts))
        row.extend(CoeffOfVariation(ts))
        row.extend(MAD(ts))
        row.extend(QRSBasicStat(ts))
        row.extend(QRSBasicStatPointMedian(ts))
        row.extend(QRSBasicStatDeltaRR(ts))
        row.extend(QRSYuxi(ts))
        row.extend(Variability(ts))
        row.extend(minimumNCCE(ts))
        row.extend(bin_stat(ts))
    

        ### no

        
        features.append(row)
        
        step += 1
        if step % 1000 == 0:
            print('extracting ...', step)
        
    print('extract QRS DONE')
    
    return features

#########################
### write feature to file
#########################

def GetAllFeature(short_table, long_table, QRS_table, long_pid_list, short_pid_list, center_waves):

    short_stat_wave_feature = GetShortStatWaveFeature(short_table, short_pid_list, long_pid_list)
    long_feature = GetLongFeature(long_table)
    qrs_feature = GetQRSFeature(QRS_table)
    
#    out_feature = CombineFeatures(short_center_wave,
#                                  CombineFeatures(short_stat_wave_feature, 
#                                                  CombineFeatures(long_feature, 
#                                                                  qrs_feature)))
    out_feature = CombineFeatures(short_stat_wave_feature, 
                                  CombineFeatures(long_feature, 
                                                  qrs_feature))


    return out_feature

def GetAllFeature_test(short_table, long_table, QRS_table, long_pid_list, short_pid_list):

    short_center_wave = GetShortCenterWave(short_table, short_pid_list, long_pid_list)
    
    short_stat_wave_feature = GetShortStatWaveFeature(short_table, short_pid_list, long_pid_list)
    long_feature = GetLongFeature(long_table)
    qrs_feature = GetQRSFeature(QRS_table)
    center_wave_feature = GetCenterWaveFeature(short_center_wave)
    
    out_feature = CombineFeatures(center_wave_feature,
                                  CombineFeatures(short_stat_wave_feature, 
                                                  CombineFeatures(long_feature, 
                                                                  qrs_feature)))

    return out_feature

def ReadAndExtractAll(fname='../data/features_all_v2.2.pkl'):
    '''
    read all data, extract features, write to dill
    '''
    
    short_pid, short_data, short_label = ReadData.ReadData( '../../data1/short.csv' )
    long_pid, long_data, long_label = ReadData.ReadData( '../../data1/long.csv' )
    QRS_pid, QRS_data, QRS_label = ReadData.ReadData( '../../data1/QRSinfo.csv' )
    center_waves = ReadData.read_mean_wave('../../data1/center_wave_euclid_direct.csv')
    
    all_pid = QRS_pid
    all_feature = GetAllFeature(short_data, long_data, QRS_data, long_pid, short_pid, center_waves)
    all_label = QRS_label
    
    print('ReadAndExtractAll done')
    print('all_feature shape: ', np.array(all_feature).shape)
    
#    with open(fname, 'wb') as output:
#        dill.dump(all_pid, output)
#        dill.dump(all_feature, output)
#        dill.dump(all_label, output)

    return

#########################
### main
#########################

#feature_list = []

#if __name__ == '__main__':
    
#    ######### read data
#    short_pid, short_data, short_label = ReadData.ReadData( '../../data1/short.csv' )
#    long_pid, long_data, long_label = ReadData.ReadData( '../../data1/long.csv' )
#    QRS_pid, QRS_data, QRS_label = ReadData.ReadData( '../../data1/QRSinfo.csv' )
#
#
##    ######### extract feature
##    long_feature = GetLongFeature(long_data)
###    QRS_feature = GetQRSFeature(QRS_data[:1])
#    short_stat_feature = GetShortStatWaveFeature(short_data, short_pid, QRS_pid)
###    all_feature_test = GetAllFeature(short_data[:10], long_data[:1], QRS_data[:1], QRS_pid[:1], short_pid[:10])
#    short_center_wave = GetShortCenterWave(short_data, short_pid, QRS_pid)
##
#    all_feature = np.array(short_stat_feature)
#    all_label = np.array(QRS_label)
##    
##    
#    ######### test feature
#    F1_list = []
#    kf = StratifiedKFold(n_splits=10, shuffle=True)
#    for train_index, test_index in kf.split(all_feature, all_label):
#        train_data = all_feature[train_index]
#        train_label = all_label[train_index]
#        test_data = all_feature[test_index]
#        test_label = all_label[test_index]
#        clf = MyRF()
#        clf.fit(train_data, train_label)
#        
#        pred = clf.predict(test_data)
#        F1_list.append(MyEval.F1Score3(pred, test_label))
#    
#    print('\n\nAvg F1: ', np.mean(F1_list))
#
#
#    ReadAndExtractAll()
#    print(len(feature_list))
    
    
    
#    with open('../data/features_all_v2.2.pkl', 'rb') as my_input:
#        all_pid = dill.load(my_input)
#        all_feature = dill.load(my_input)
#        all_label = dill.load(my_input)
#    print(np.array(all_feature).shape)
#    print(np.array(all_label).shape)



#    print(len(bin_stat(QRS_data[0])))
