#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 23:00:34 2017

@author: shenda
"""

import numpy as np
import MyEval
import ReadData
import dill
from features_mimic import get_mimic_proba
from features_all import GetAllFeature_test

def pred_one_sample(short_data, long_data, QRS_data, long_pid, short_pid):
    '''
    predict one sample
    
    input:
        short_data: [[7,7], [8,8,8]]
        long_data: [[7,7,8,8,8]]
        QRS_data: [[2,3]]
        long_pid: ['A00001']
        short_pid: ['A00001', 'A00001']
        
    output:
        label of ['A', 'N', 'O', '~']
    '''
    ### load clf
    labels = ['N', 'A', 'O', '~']
    
    with open('model/v2.5_xgb5_all_v2.pkl', 'rb') as fin:
        clf_ENCASE = dill.load(fin)

    ### extract features
    feature_ENCASE = GetAllFeature_test(short_data, long_data, QRS_data, long_pid, short_pid)
    if feature_ENCASE[0][-1] == 0.0:
            feature_ENCASE[0][-1] = 0.00000001
    
    ### pred 
    ### alert: encase is naop, lr is anop
    pred_proba_ENCASE = clf_ENCASE.predict_prob(feature_ENCASE)[0]
    pred_proba_LR = get_mimic_proba(long_data[0])
    pred_final = 1/2 * pred_proba_ENCASE + 1/2 * pred_proba_LR
    print('{0}\n{1}\n{2}'.format(pred_proba_ENCASE, pred_proba_LR, pred_final))
    
    pred_label = labels[np.argsort(pred_final)[-1]]
    
    return pred_label

if __name__ == '__main__':
    short_pid, short_data, short_label = ReadData.ReadData( '../../data1/short.csv' )
    long_pid, long_data, long_label = ReadData.ReadData( '../../data1/long.csv' )
    QRS_pid, QRS_data, QRS_label = ReadData.ReadData( '../../data1/QRSinfo.csv' )
    print('='*60)
    print('pred begin')

    res = pred_one_sample(short_data[0:40], long_data[0:1], QRS_data[0:1], long_pid[0:1], short_pid[0:40])
    print('pred done, the label of {0} is {1}'.format(long_pid[0], res))    
    
    # fout= open('answers.txt','a')
    # fout.write(res)
    # fout.write('\n')
    # fout.close
    
    
    
