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
#from features_mimic import get_mimic_LR_proba
from features_all import GetAllFeature_test

from features_resNet import get_resNet_proba

def pred_encase(short_data, long_data, QRS_data, long_pid, short_pid):
    '''
    input: 
        short_data: [[7,7], [8,8,8]]
        long_data: [[7,7,8,8,8]]
        QRS_data: [[2,3]]
        long_pid: ['A00001']
        short_pid: ['A00001', 'A00001']    
        
    output: 
        proba of ['N', 'A', 'O', '~']: np.array([0.0, 0.0, 0.0, 0.0])
    '''
    model_name = '../model/v2.5_xgb5_all.pkl'
    with open(model_name, 'rb') as fin:
        clf_ENCASE = dill.load(fin)
    feature_ENCASE = GetAllFeature_test(short_data, long_data, QRS_data, long_pid, short_pid)
    pred_proba_ENCASE = clf_ENCASE.predict_prob(feature_ENCASE)[0]
    return pred_proba_ENCASE
    
def pred_resnet(long_data, long_pid):
    '''
    input: 
        long_data: [[7,7,8,8,8]]
        long_pid: ['A00001']
        
    output: 
        proba of ['N', 'A', 'O', '~']: np.array([0.0, 0.0, 0.0, 0.0])
    '''
    model_path = "../model/resNet/resnet_6000_500_10_5_v1"
    pred_proba_resnet = get_resNet_proba(long_data, long_pid, model_path)[0]
    return pred_proba_resnet
    
def pred_mimic(y_pre):
    '''
    input: 
        y_pre: [[0.0, 1.0, 0.0, 0.0]]
        
    output: 
        proba of ['N', 'A', 'O', '~']: np.array([0.0, 0.0, 0.0, 0.0])
    '''
    pred_proba_mimic = get_voted_proba(y_pre)
    return pred_proba_mimic
    
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
        label of ['N', 'A', 'O', '~']
    '''
    ### load clf
    labels = ['N', 'A', 'O', '~']
    
    ### pred each
    pred_proba_ENCASE = pred_encase(short_data, long_data, QRS_data, long_pid, short_pid)
    print(len(pred_proba_ENCASE))
    pred_proba_resnet = pred_resnet(long_data, long_pid)
    print(len(pred_proba_resnet))

    ### pred combine
    pred_final = 1/2 * pred_proba_ENCASE + 1/2 * pred_proba_resnet
    
    ### print and output label
    print('{0}\n{1}\n{2}'.format(pred_proba_ENCASE, pred_proba_resnet, pred_final))
    pred_label = labels[np.argsort(pred_final)[-1]]
    
    return pred_label

def get_voted_proba(pre):
    y_pre = [0. for j in range(4)]
    y_sec_pre = [0. for j in range(4)]
    y_third_pre = [0. for j in range(4)]
    y_pre = np.array(y_pre, dtype=np.float32)
    y_sec_pre = np.array(y_sec_pre, dtype=np.float32)
    y_third_pre = np.array(y_third_pre, dtype=np.float32)
    max_p = 0
    max_sec_p = 0
    max_third_p = 0
    sec_p = 0
    sec_sec_p = 0
    sec_third_p = 0
    
    for j in range(len(pre)):
        i_pred = np.array(pre[j], dtype=np.float32)
        
        cur_max_p = i_pred[np.argmax(i_pred)]
        cur_sec_p = 0
        for k in range(len(i_pred)):
            if i_pred[k] == cur_max_p:
                continue
            if i_pred[k] > cur_sec_p:
                cur_sec_p = i_pred[k]
        
        if (cur_max_p - cur_sec_p) > (max_p - sec_p):
            y_third_pre = y_sec_pre
            y_sec_pre = y_pre
            y_pre = i_pred
            max_p = cur_max_p
            sec_p = cur_sec_p
        elif len(pre) >= 2 and (cur_max_p - cur_sec_p) > (max_sec_p - sec_sec_p):
            y_third_pre = y_sec_pre
            y_sec_pre = i_pred
        elif len(pre) >= 3 and (cur_max_p - cur_sec_p) > (max_third_p - sec_third_p):
            y_third_pre = i_pred
            
    
    labels = [0. for j in range(4)]
    pred_1 = np.argmax(y_pre)
    labels[pred_1] +=1
    pred_2 = pred_3 = 0
    if len(pre) >= 2:
        pred_2 = np.argmax(y_sec_pre)
        labels[pred_2] +=1
    if len(pre) >= 3:
        pred_3 = np.argmax(y_third_pre)
        labels[pred_3] +=1

    '''if pred_1 == 2:# and (abs(y_pre[k][np.argmax(labels)] - y_pre[k][2])/y_pre[k][np.argmax(labels)] <= 0.2):
        return y_pre
    elif pred_2 == 2:# and (abs(y_pre[k][np.argmax(labels)] - y_sec_pre[k][2])/y_pre[k][np.argmax(labels)] <= 0.2):
        y_pre = y_sec_pre
    elif pred_3 == 2:# and (abs(y_pre[k][np.argmax(labels)] - y_third_pre[k][2])/y_pre[k][np.argmax(labels)] <= 0.2):
        y_pre = y_third_pre'''
    if pred_1 != np.argmax(labels):
        if pred_2 == np.argmax(labels):
            y_pre = y_sec_pre
    
    return y_pre



if __name__ == '__main__':
    short_pid, short_data, short_label = ReadData.ReadData( '../../data1/short.csv' )
    long_pid, long_data, long_label = ReadData.ReadData( '../../data1/long.csv' )
    QRS_pid, QRS_data, QRS_label = ReadData.ReadData( '../../data1/QRSinfo.csv' )
    print('='*60)
    print('pred begin')

    out_label = ReadData.Label2OneHot(long_label)
    out_label = np.array(out_label, dtype=np.float32)
    
    #res_pre = pred_resnet(long_data, long_label, long_pid)
    #print(len(res_pre))
    #MyEval.F1Score3_num(res_pre, out_label)
    num_data = len(long_data)
    pre = [[0. for j in range(4)] for i in  range(num_data)]
    
    #for i in range(num_data):
    res = pred_one_sample(short_data[0:40], long_data[0:1], QRS_data[0:1], long_pid[0:1], short_pid[0:40])
    print(res)
    labels = {'N':0, 'A':1, 'O':2, '~':3}
    pre = [0., 0., 0., 0.]
    pre[labels[res]] = 1
    MyEval.F1Score3_num(pre, out_label[0:1])
    print('pred done, the label of {0} is {1}'.format(long_pid[0], res)) 
    
    '''pre = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
    y = get_voted_proba(pre)
    print(y)'''
    

    
    
    