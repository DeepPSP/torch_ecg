# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 21:32:52 2017

@author: hsd
"""

import numpy as np
from scipy import stats
import ReadData

##################################################
### tools
##################################################
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

##################################################
### get features
##################################################

def short_basic_stat(ts):
    global feature_list
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

def short_zero_crossing(ts):

    global feature_list
    feature_list.extend(['short_zero_crossing_cnt'])

    cnt = 0
    for i in range(len(ts)-1):
        if ts[i] * ts[i+1] < 0:
            cnt += 1
        if ts[i] == 0 and ts[i-1] * ts[i+1] < 0:
            cnt += 1
    return [cnt]



##################################################
###  get all features
##################################################
def get_short_stat_wave_feature(table, pid_list, long_pid_list):
    '''
    short stat feature, actually long feature
    
    Electrocardiogram Feature Extraction and Pattern Recognition Using a Novel Windowing Algorithm
    
    row of out feature is 8000+
    
    TODO: more on how to detect PT waves
    '''

    global feature_list
    feature_list = []

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
            QRS_peak_list = []
            QRS_area_list = []
            
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
                                
                ### features, recent add (2)
                QRS_peak = max(ts)
                QRS_area = np.sum(np.abs(prev_ts[Q_loc: 0])) + np.sum(np.abs(ts[0: S_loc]))
                
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
                
                
                ### add to list
                QRS_peak_list.append(QRS_peak)
                QRS_area_list.append(QRS_area)
                
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


            features.append([np.mean(QRS_peak_list), 
                             np.mean(QRS_area_list), 
                             np.mean(PR_interval_list), 
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
                             
                             np.max(QRS_peak_list), 
                             np.max(QRS_area_list), 
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
                             
                             np.min(QRS_peak_list), 
                             np.min(QRS_area_list), 
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
                             
                             np.std(QRS_peak_list), 
                             np.std(QRS_area_list), 
                             np.std(PR_interval_list), 
                             np.std(QRS_duration_list), 
                             np.std(QT_interval_list), 
                             np.std(QT_corrected_list), 
                             np.std(vent_rate_list),
                             np.std(RQ_amp_list),
                             np.std(RS_amp_list),
                             np.std(ST_amp_list),
                             np.std(PQ_amp_list),
                             np.std(QS_amp_list),
                             np.std(RP_amp_list),
                             np.std(RT_amp_list),
                             np.std(ST_interval_list),
                             np.std(RS_interval_list),
                             np.std(T_peak_list),
                             np.std(P_peak_list),
                             np.std(Q_peak_list),
                             np.std(R_peak_list),
                             np.std(S_peak_list),
                             np.std(RS_slope_list),
                             np.std(ST_slope_list),
                             np.std(NF_list),
                             np.std(Fwidth_list), 
                             
                             np.percentile(QRS_peak_list, 25), 
                             np.percentile(QRS_area_list, 25), 
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
                             
                             np.percentile(QRS_peak_list, 75), 
                             np.percentile(QRS_area_list, 75), 
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
            features.append([0.0] * ((2+5+16+2) * 6))

        step += 1
        if step % 1000 == 0:
            print('extracting ...', step)
            # break
            
            
    feature_list.extend(['GetShortStatWaveFeature_'+str(i) for i in range((2+5+16+2) * 6)])
    print('extract GetShortStatWaveFeature DONE')

    return feature_list, features


def get_short_feature(table):
    '''
    rows of table is 330000+
    
    no use now
    '''

    global feature_list
    feature_list = []


    features = []
    step = 0
    for ts in table:
        row = []

        row.extend(short_basic_stat(ts))
#        row.extend(short_zero_crossing(ts))
        
        features.append(row)
        
        step += 1
        if step % 100000 == 0:
            print('extracting ...')
#            break
        
    print('extract DONE')
    
    return feature_list, features



if __name__ == '__main__':
    short_pid, short_data, short_label = ReadData.ReadData( '../../data1/short.csv' )
    QRS_pid, QRS_data, QRS_label = ReadData.ReadData( '../../data1/QRSinfo.csv' )
    tmp_features = get_short_stat_wave_feature(short_data[:10], short_pid[:10], QRS_pid[0])
    print(len(tmp_features[1][0]))
    
    
    