#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 11:07:19 2017

@author: shenda
"""
from collections import Counter
import numpy as np
import pandas as pd
import MyEval
import ReadData
from LevelTop import LevelTop
import dill
from sklearn.model_selection import KFold
from BasicCLF import RFSimp
from BasicCLF import LRSimp
from BasicCLF import MyXGB
from CascadeCLF import CascadeCLF
from OptF import OptF
import sklearn
import xgboost

def TestOptF():
#if __name__ == "__main__":
    '''
    result: 
                
    LR: 
    [[ 915.   97.]
     [ 267.  223.]]
    0.834092980857 0.550617283951
    0.692355132404
    [[ 970.   94.]
     [ 204.  233.]]
    0.866845397676 0.609947643979
    0.738396520828
    [[ 932.  102.]
     [ 259.  208.]]
    0.837752808989 0.535392535393
    0.686572672191
    [[ 884.   62.]
     [ 307.  248.]]
    0.827328029949 0.573410404624
    0.700369217286
    [[ 919.   75.]
     [ 292.  215.]]
    0.833560090703 0.539523212045
    0.686541651374
    Avg F1:  0.700847038817
    
    
    OptF: 
    [[ 721.  291.]
     [ 130.  360.]]
    0.774020397209 0.631025416301
    0.702522906755
    [[ 794.  270.]
     [ 116.  321.]]
    0.804457953394 0.624513618677
    0.714485786036
    [[ 709.  325.]
     [ 130.  337.]]
    0.757074212493 0.596988485385
    0.677031348939
    [[ 719.  227.]
     [ 160.  395.]]
    0.787945205479 0.671197960918
    0.729571583199
    [[ 719.  275.]
     [ 158.  349.]]
    0.768572955639 0.617152961981
    0.69286295881
    Avg F1:  0.703294916748
    
    conclusion: 
        F1 on O is promoted, but F1 on N is decreased
        after avg, no obvious diff
    '''
    with open('../../data2/features_all.pkl', 'rb') as my_input:
        all_pid = dill.load(my_input)
        all_feature = dill.load(my_input)
        all_label = dill.load(my_input)

    ### preprocess
    all_feature = np.array(all_feature)
    selected = [i for i, x in enumerate(all_label) if x == 'N' or x == 'O']
    all_label = np.array(all_label)
    all_feature = all_feature[selected]
    all_label = all_label[selected]
    all_label_num = np.array(ReadData.LabelTo2(all_label, 'O'))
    
    ## k-fold cross validation
#    F1_list = []
#    kf = KFold(n_splits=5)
#    for train_index, test_index in kf.split(all_label):
#        train_data = all_feature[train_index]
#        train_label = all_label[train_index]
#        train_label_num = all_label_num[train_index]
#        test_data = all_feature[test_index]
#        test_label = all_label[test_index]
#        test_label_num = all_label_num[test_index]
#        
#        clf = LRSimp()
#        clf.fit(train_data, train_label)
#        pred = []
#        n_row, n_col = test_data.shape
#        for i in range(n_row):
#            pred.extend(clf.predict(list(test_data[i])))
#    #        break
#        F1_list.append(MyEval.F1Score2(pred, test_label))
#    
#    print('\n\nAvg F1: ', np.mean(F1_list))

    F1_list = []
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(all_label):
        train_data = all_feature[train_index]
        train_label = all_label[train_index]
        train_label_num = all_label_num[train_index]
        test_data = all_feature[test_index]
        test_label = all_label[test_index]
        test_label_num = all_label_num[test_index]
        
        test_data = sklearn.preprocessing.scale(test_data, axis=0)
        
        clf = OptF()
        clf.fit(train_data, train_label_num)
        pred = []
        n_row, n_col = test_data.shape
        for i in range(n_row):
            pred_prob = clf.predict_prob(list(test_data[i]))[0]
            if pred_prob > 0.5:
                pred.append('O')
            else:
                pred.append('N')
    #        break
        F1_list.append(MyEval.F1Score2(pred, test_label))
    
    print('\n\nAvg F1: ', np.mean(F1_list))
