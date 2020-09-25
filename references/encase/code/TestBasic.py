#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 11:08:54 2017

@author: shenda
"""

from collections import Counter
import numpy as np
import pandas as pd
import MyEval
import ReadData
import dill
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from BasicCLF import MyAdaBoost
from BasicCLF import MyRF
from BasicCLF import MyExtraTrees
from BasicCLF import MyXGB
from BasicCLF import MyGBDT
from BasicCLF import MyLR
from OptF import OptF
import sklearn
import xgboost

#def TestBasic():
if __name__ == "__main__":

    with open('../data/features_all_v2.2.pkl', 'rb') as my_input:
        all_pid = dill.load(my_input)
        feat_feature = dill.load(my_input)
        all_label = dill.load(my_input)
        
    mean_wave = ReadData.read_mean_wave_simp()
    
    ## k-fold cross validation
#    all_feature = np.array(np._c[mean_wave, feat_feature])
    all_feature = np.array(mean_wave)
    all_label = np.array(all_label)
    F1_list = []
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    i_fold = 1
    for train_index, test_index in kf.split(all_feature, all_label):
        train_data = all_feature[train_index]
        train_label = all_label[train_index]
        test_data = all_feature[test_index]
        test_label = all_label[test_index]
        
        clf = MyXGB()
        clf.fit(train_data, train_label)
#        clf.save_importance()
        
        pred_train = clf.predict(train_data)
        MyEval.F1Score3(pred_train, train_label)
        pred = clf.predict(test_data)
        MyEval.F1Score3(pred, test_label)
         
        F1_list.append(MyEval.F1Score3(pred, test_label))
    
    print('\n\nAvg F1: ', np.mean(F1_list))














