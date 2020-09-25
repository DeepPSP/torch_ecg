#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 08:50:07 2017

@author: shenda
"""

import dill
import numpy as np
import ReadData
import random
from collections import OrderedDict
from features_centerwave import get_centerwave_feature
from features_long import get_long_feature
from features_qrs import get_qrs_feature
from features_short import get_short_stat_wave_feature

from features_centerwave import get_short_centerwave
#from features_short import get_short_feature

##################################################
### tools
##################################################
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

##################################################
### get all features
##################################################

def GetAllFeature(short_table, long_table, QRS_table, long_pid_list, short_pid_list, center_waves):
    '''
    get all features, with feature name, need precomputed center_waves
    
    input:
        data: short_table, long_table, QRS_table, center_waves
        pid: long_pid_list, short_pid_list
    output:
        out_feature: 8528 rows
    
    1. centerwave_feature
    2. long_feature
    3. qrs_feature
    4. short_stat_wave_feature
    '''

    feature_list = []

    centerwave_names, centerwave_feature = get_centerwave_feature(center_waves)
    long_names, long_feature = get_long_feature(long_table)
    qrs_names, qrs_feature = get_qrs_feature(QRS_table)
    shortstat_name, short_stat_wave_feature = get_short_stat_wave_feature(short_table, short_pid_list, long_pid_list)
    
    feature_list.extend(centerwave_names[:len(centerwave_feature[0])])
    feature_list.extend(long_names[:len(long_feature[0])])
    feature_list.extend(qrs_names[:len(qrs_feature[0])])
    feature_list.extend(shortstat_name[:len(short_stat_wave_feature[0])])
    
    print('centerwave_feature shape: ', len(centerwave_feature[0]))
    print('long_feature shape: ', len(long_feature[0]))
    print('qrs_feature shape: ', len(qrs_feature[0]))
    print('short_stat_wave_feature shape: ', len(short_stat_wave_feature[0]))
    
    out_feature = CombineFeatures(centerwave_feature,
                                  CombineFeatures(long_feature, 
                                                  CombineFeatures(qrs_feature, 
                                                                  short_stat_wave_feature)))
    print('out_feature shape: ', len(out_feature[0]))

    return feature_list, out_feature

def GetAllFeature_test(short_table, long_table, QRS_table, long_pid_list, short_pid_list):
    '''
    get all features for test, without feature name, do not need precomputed center_waves
    
    input:
        data: short_table, long_table, QRS_table
        pid: long_pid_list, short_pid_list
    output:
        out_feature: 8528 rows
        
    1. centerwave_feature
    2. long_feature
    3. qrs_feature
    4. short_stat_wave_feature
    '''
    
    center_waves = get_short_centerwave(short_table, short_pid_list, long_pid_list)
    
    _, centerwave_feature = get_centerwave_feature(center_waves)
    _, long_feature = get_long_feature(long_table)
    _, qrs_feature = get_qrs_feature(QRS_table)
    _, short_stat_wave_feature = get_short_stat_wave_feature(short_table, short_pid_list, long_pid_list)

    out_feature = CombineFeatures(centerwave_feature,
                                  CombineFeatures(long_feature, 
                                                  CombineFeatures(qrs_feature, 
                                                                  short_stat_wave_feature)))
    
    ### TODO: potential bug, if last column all 0, may cause bug in xgboost
    # for feat in out_feature:
    #     if feat[-1] == 0.0:
    #         feat[-1] = 0.00000001
    return out_feature


##################################################
### write pkl file
##################################################
def ReadAndExtractAll(fname='../data/features_all_v2.5.pkl'):
    '''
    read all data, extract features, write to dill
    '''
    
    short_pid, short_data, short_label = ReadData.ReadData( '../../data1/short.csv' )
    long_pid, long_data, long_label = ReadData.ReadData( '../../data1/long.csv' )
    QRS_pid, QRS_data, QRS_label = ReadData.ReadData( '../../data1/QRSinfo.csv' )
    center_waves = ReadData.read_mean_wave('../../data1/centerwave_raw.csv')
    
    all_pid = QRS_pid
    feature_list, all_feature = GetAllFeature(short_data, long_data, QRS_data, long_pid, short_pid, center_waves)
    all_label = QRS_label
    
    print('ReadAndExtractAll done')
    print('all_feature shape: ', np.array(all_feature).shape)
    print('feature_list shape: ', len(feature_list))
    np.nan_to_num(all_feature)
    
    with open(fname+'_feature_list.csv', 'w') as fout:
        for i in feature_list:
            fout.write(i + '\n')
    
    with open(fname, 'wb') as output:
        dill.dump(all_pid, output)
        dill.dump(all_feature, output)
        dill.dump(all_label, output)
    print('write done')
    return

#########################
### main
#########################

if __name__ == '__main__':    

#    ######### read data
#    short_pid, short_data, short_label = ReadData.ReadData( '../../data1/short.csv' )
#    long_pid, long_data, long_label = ReadData.ReadData( '../../data1/long.csv' )
#    QRS_pid, QRS_data, QRS_label = ReadData.ReadData( '../../data1/QRSinfo.csv' )
#    center_waves = ReadData.read_mean_wave('../../data1/centerwave_raw.csv')
#    print('read all data done')
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
    ReadAndExtractAll()
    
    
    
#    with open('../data/features_all_v2.2.pkl', 'rb') as my_input:
#        all_pid = dill.load(my_input)
#        all_feature = dill.load(my_input)
#        all_label = dill.load(my_input)
#    print(np.array(all_feature).shape)
#    print(np.array(all_label).shape)

