#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 00:30:06 2017

@author: shenda

par selection for xgboost, my own
"""

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import dill
from scipy import stats
import ReadData
import MyEval
from BasicCLF import MyXGB


#if __name__ == "__main__":
def XGBcv(all_pid, all_feature, all_label, 
          subsample, max_depth, colsample_bytree, min_child_weight):
    '''
    TODO: 
        try kf = StratifiedKFold(n_splits=5, shuffle=True)
    '''
    
    wrong_stat = []
    
    ## k-fold cross validation
    all_pid = np.array(all_pid)
    all_feature = np.array(all_feature)
    all_label = np.array(all_label)
    F1_list = []
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(all_feature, all_label):
        train_data = all_feature[train_index]
        train_label = all_label[train_index]
        test_data = all_feature[test_index]
        test_label = all_label[test_index]
        
        clf = MyXGB(subsample=subsample, 
                    max_depth=max_depth, 
                    colsample_bytree=colsample_bytree, 
                    min_child_weight=min_child_weight)
        clf.fit(train_data, train_label)
        
        pred = clf.predict(test_data)
        F1_list.append(MyEval.F1Score3(pred, test_label, False))
            
    print('\n\nAvg F1: ', np.mean(F1_list))

    return np.mean(F1_list)


if __name__ == "__main__":
    
    res = []
#    fout = open('../../reseult/xgbcv.txt', 'w')
    
    with open('../data/features_all_v2.5.pkl', 'rb') as my_input:
        all_pid = dill.load(my_input)
        all_feature = dill.load(my_input)
        all_label = dill.load(my_input)
        
    fout = open('../../stat/xgbcv1.txt', 'a')
    fout.write('{0},{1},{2},{3},{4}\n'.format('subsample', 'max_depth', 'colsample_bytree', 'min_child_weight', 'f1'))
    fout.close()
    
    # subsample_list = [0.8, 0.85, 0.9]
    # max_depth_list = [7, 8, 9, 10, 11]
    # colsample_bytree_list = [0.8, 0.85, 0.9]    
    subsample_list = [0.9]
    max_depth_list = [11]
    colsample_bytree_list = [0.9]
    min_child_weight_list = [2, 3, 4]

    for subsample in subsample_list:
        for max_depth in max_depth_list:
            for colsample_bytree in colsample_bytree_list:
                for min_child_weight in min_child_weight_list:
                    for i in range(5):
                        f1 = XGBcv(all_pid, all_feature, all_label, subsample, max_depth, colsample_bytree, min_child_weight)
                        fout = open('../../stat/xgbcv1.txt', 'a')
                        fout.write('{0},{1},{2},{3},{4}\n'.format(subsample, max_depth, colsample_bytree, min_child_weight, f1))
                        fout.close()





