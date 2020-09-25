# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 21:26:17 2017

@author: hsd
"""

import numpy as np
from scipy import stats
import ReadData
#from matplotlib import pyplot as plt

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

def dist(ts1, ts2):
    '''
    Input: two vectors
    Output: distance, numeric
    '''
    dist_num = np.linalg.norm(np.array(ts1) - np.array(ts2))
    return dist_num

def resample_unequal(ts, length):
    '''
    TODO: 
        1. average of several points
    '''
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
    
    if len(width) > 1:
        return [cnt, np.mean(width)]
    else:
        return [cnt, 0.0]

def autocorr(ts, t):
    return np.corrcoef(np.array([ts[0:len(ts)-t], ts[t:len(ts)]]))[0,1]


##################################################
### get features
##################################################
def centerwave_simp(ts):
    '''
    ### 1
    resample centerwave to length 200, as features directly
    '''
    global feature_list
    length = 200
    feature_list.extend(['center_wave_simp_'+str(length) + '_' + str(i) for i in range(length)])
    resampled_center_wave_raw = resample_unequal(ts, length)
    return resampled_center_wave_raw

def centerwave_zero_crossing(ts):
    '''
    ### 2
    '''
    global feature_list
    feature_list.extend(['center_wave_zero_crossing_cnt'])

    cnt = 0
    for i in range(len(ts)-1):
        if ts[i] * ts[i+1] < 0:
            cnt += 1
        if ts[i] == 0 and ts[i-1] * ts[i+1] < 0:
            cnt += 1
    return [cnt]

def centerwave_basic_stat(ts):
    '''
    ### 3
    stat feat
    '''
    global feature_list
    feature_list.extend(['center_wave_basic_stat_length', 
                         'center_wave_basic_stat_area', 
                        'center_wave_basic_stat_Max', 
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
    
    length = len(ts)
    area = np.sum(np.abs(ts))
    
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

    
    return [length, area, 
            Max, Min, Range, Var, Skew, Kurtosis, Median, 
            p_1, p_5, p_95, p_99, 
            p_10, p_25, p_75, p_90, 
            range_99_1, range_95_5, range_90_10, range_75_25]

def centerwave_wave_feature(ts):
    '''
    ### 4
    Electrocardiogram Feature Extraction and Pattern Recognition Using a Novel Windowing Algorithm
        
    TODO: more on how to detect PT waves
    '''

    global feature_list
    feature_list.extend(['centerwave_wave_feature_'+str(i) for i in range(5+16+2)])
    feat = []


    ### key points
    T_start = round(0.15 * len(ts))
    T_end = round(0.55 * len(ts))
    P_start = round(0.65 * len(ts))
    P_end = round(0.95 * len(ts))
    
    T_wave = ts[T_start:T_end]
    P_wave = ts[P_start:P_end]
    
    T_peak = max(T_wave)
    P_peak = max(P_wave)
    Q_peak = min(ts[-6:])
    R_peak = ts[0]
    S_peak = min(ts[:6])
    
    T_loc = np.argmax(T_wave)
    P_loc = np.argmax(P_wave)
    Q_loc = -np.argmin(ts[-6:]) - len(ts)
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
    feat.append(PR_interval)
    feat.append(QRS_duration)
    feat.append(QT_interval)
    feat.append(QT_corrected)
    feat.append(vent_rate)
    
    feat.append(NF)
    feat.append(Fwidth)

    feat.append(RQ_amp)
    feat.append(RS_amp)
    feat.append(ST_amp)
    feat.append(PQ_amp)
    feat.append(QS_amp)
    feat.append(RP_amp)
    feat.append(RT_amp)
    feat.append(ST_interval)
    feat.append(RS_interval)
    feat.append(T_peak)
    feat.append(P_peak)
    feat.append(Q_peak)
    feat.append(R_peak)
    feat.append(S_peak)
    feat.append(RS_slope)
    feat.append(ST_slope)
    
    return feat

def centerwave_autocorr(ts):
    '''
    ### 5
    auto-coefficient, with lag
    '''
    feat = []
    num_lag = 12
    
    global feature_list
    feature_list.extend(['centerwave_autocorr_'+str(i) for i in range(num_lag)])
    
    for i in range(num_lag):
        feat.append(autocorr(ts, i))

    return feat

def centerwave_zigzag(ts):
    '''
    ### 6
    number of zigzag in centerwave
    '''
    feature_list.extend(['centerwave_zigzag'])
    num_zigzag = zigzag(ts)
    return [num_zigzag]
        
##################################################
###  get centerwave
###  very slow, do not run everytime
##################################################        
#if __name__ == '__main__':

def get_short_centerwave(table, pid_list, long_pid_list):
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
    print('extract get_short_centerwave begin')
#    fout = open('../../data1/centerwave_resampled.csv', 'w')
#    fout = open('../../data1/centerwave_raw.csv', 'w')
    resampled_length = 1000
#    n_clusters = 3
#    radius = 1

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
            
            raw_center_wave = sub_table[np.argsort(dist_mat_dist)[0]]
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
            ###### resample
#            resampled_idx = [int(i) for i in list(np.linspace(0.0, len(center_wave)-1, resampled_length))]
#            resampled_center_wave = [center_wave[i] for i in resampled_idx]
            
        else:
            raw_center_wave = [0.0] * 240
            resampled_center_wave = [0.0] * resampled_length
        
        ### schema 1: use raw centerwave
        features.append(raw_center_wave)
#        fout.write(pid + ',' + ','.join([str(i) for i in raw_center_wave]) + '\n')
        
        ### schema 2: use resampled centerwave
#        features.append(resampled_center_wave)
#        fout.write(pid + ',' + ','.join([str(i) for i in resampled_center_wave]) + '\n')
        
        step += 1
        # print(step)
        ### for debug
#        plt.plot(resampled_center_wave)
#        break
    
    
        if step % 100 == 0:
            print('extracting ...', step)
#            break
#    feature_list.extend(['GetShortCenterWave_'+str(i) for i in range(resampled_length)])
    print('extract GetShortCenterWave DONE')
#    fout.close()

    return features



##################################################
###  get all features
##################################################
def get_centerwave_feature(table):
    '''
    rows of table is 8000+
    '''
    
    global feature_list
    feature_list = []
    
    print('get_centerwave_feature begin')
    features = []
    step = 0
    for ts in table:
        row = []
        
        ### yes
        ### 1
        row.extend(centerwave_simp(ts))
        ### 2
        row.extend(centerwave_zero_crossing(ts))
        ### 3
        row.extend(centerwave_basic_stat(ts))
        ### 4
        row.extend(centerwave_wave_feature(ts))
        ### 5
        row.extend(centerwave_autocorr(ts))
        ### 6
        row.extend(centerwave_zigzag(ts))
        
        ### no
        

        features.append(row)
        
        step += 1
        if step % 1000 == 0:
            print('extracting ...', step)
            # break
        
    print('get_centerwave_feature done')
    
    return feature_list, features



if __name__ == '__main__':
    center_waves = ReadData.read_centerwave('../../data1/centerwave_raw.csv')
    tmp_features = get_centerwave_feature(center_waves[:1])
    print(len(tmp_features[1][0]))
    
    
