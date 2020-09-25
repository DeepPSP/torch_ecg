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
from features_mimic import get_mimic_proba_all
from features_all import GetAllFeature_test

def group_label(out_pid, long_pid, preds):
    pred_dic = {k: [] for k in long_pid}
    final_preds = []
    for i in range(len(out_pid)):
        pred_dic[out_pid[i]].append(preds[i])
    for i in long_pid:
        if len(pred_dic[i]) > 1:
            final_preds.append(np.mean(np.array(pred_dic[i]), axis=0))
        else:
            final_preds.append(pred_dic[i][0])
    return np.array(final_preds)

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
    
    with open('../model/v2.5_xgb5_all_v2.pkl', 'rb') as fin:
        clf_ENCASE = dill.load(fin)

    ### extract features
    feature_ENCASE = GetAllFeature_test(short_data, long_data, QRS_data, long_pid, short_pid)
    
    ### pred 
    ### alert: encase is naop, lr is anop
    pred_proba_ENCASE = clf_ENCASE.predict_prob(feature_ENCASE)[0]
    pred_proba_mimic = get_mimic_proba(long_data[0])
    pred_final = 1/2 * pred_proba_ENCASE + 1/2 * pred_proba_mimic
    print('{0}\n{1}\n{2}'.format(pred_proba_ENCASE, pred_proba_mimic, pred_final))
    
    pred_label = labels[np.argsort(pred_final)[-1]]
    
    return pred_label

if __name__ == '__main__':
    short_pid, short_data, short_label = ReadData.ReadData( '../../data_val/short.csv' )
    long_pid, long_data, long_label = ReadData.ReadData( '../../data_val/long.csv' )
    QRS_pid, QRS_data, QRS_label = ReadData.ReadData( '../../data_val/QRSinfo.csv' )
    print('='*60)
    print('pred begin')
    
    # short_data = short_data[:100]
    # long_data = long_data[:3]
    # QRS_data = QRS_data[:3]
    # long_pid = long_pid[:3]
    # short_pid = short_pid[:100]
    
    with open('../model/v2.5_xgb5_all.pkl', 'rb') as fin:
        clf_ENCASE = dill.load(fin)
    feature_ENCASE = GetAllFeature_test(short_data, long_data, QRS_data, long_pid, short_pid)
    pred_proba_ENCASE = clf_ENCASE.predict_prob(feature_ENCASE)
    
    pred_proba_mimic_all, out_pid = get_mimic_proba_all(long_data, long_pid)
    pred_proba_mimic = group_label(out_pid, long_pid, pred_proba_mimic_all)
    
    pred_final = 1/2 * pred_proba_ENCASE + 1/2 * pred_proba_mimic
    labels = ['N', 'A', 'O', '~']
    pred_label = []
    for i in pred_final:
        pred_label.append(labels[np.argsort(i)[-1]])
    
    fout = open('../answers.txt','w')
    for i in range(len(long_pid)):
        fout.write('{0},{1}\n'.format(long_pid[i], pred_label[i]))
    fout.close()
    
    MyEval.F1Score3(pred_label, QRS_label)  