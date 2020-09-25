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
from sklearn.model_selection import StratifiedKFold
from BasicCLF import MyXGB
from BasicCLF import MyLR
import sklearn
import xgboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn import ensemble
from Encase import Encase
from features_all import ReadAndExtractAll



def TestExp(all_pid, all_feature, all_label, method, i_iter):

    kf = StratifiedKFold(n_splits=5, shuffle=True)
    i_fold = 1
    print('all feature shape: {0}'.format(all_feature.shape))
    for train_index, test_index in kf.split(all_feature, all_label):
        train_data = all_feature[train_index]
        train_label = all_label[train_index]
        test_data = all_feature[test_index]
        test_label = all_label[test_index]
        test_pid = all_pid[test_index]
        
        ### ENCASE
        if method == 'ENCASE_E':
            selected_cols = list(range(258, 558))
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_1 = MyXGB()
            clf_2 = MyXGB()
            clf_3 = MyXGB() 
            clf_4 = MyXGB()
            clf_5 = MyXGB()
            clf_final = Encase([clf_1, clf_2, clf_3, clf_4, clf_5])
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)
        
        if method == 'ENCASE_EC':
            selected_cols = list(range(0, 558))
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_1 = MyXGB()
            clf_2 = MyXGB()
            clf_3 = MyXGB() 
            clf_4 = MyXGB()
            clf_5 = MyXGB()
            clf_final = Encase([clf_1, clf_2, clf_3, clf_4, clf_5])
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)    

        if method == 'ENCASE_ECD':
            clf_1 = MyXGB()
            clf_2 = MyXGB()
            clf_3 = MyXGB() 
            clf_4 = MyXGB()
            clf_5 = MyXGB()
            clf_final = Encase([clf_1, clf_2, clf_3, clf_4, clf_5])
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)
        
        elif method == 'XGBoost_E':
            selected_cols = list(range(258, 558))
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_final = MyXGB(n_estimators=100, num_round=50)
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)
        
        elif method == 'XGBoost_EC':
            selected_cols = list(range(0, 558))
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_final = MyXGB(n_estimators=100, num_round=50)
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)
        
        elif method == 'XGBoost_ECD':
            clf_final = MyXGB(n_estimators=100, num_round=50)
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)
        
        elif method == 'LR_E':
            selected_cols = list(range(258, 558))
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_final = MyLR()
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)

        elif method == 'LR_EC':
            selected_cols = list(range(0, 558))
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_final = MyLR()
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)

        elif method == 'LR_ECD':
            clf_final = MyLR()
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)

            
        elif method == 'SampleEn':
            selected_cols = [300, 301, 302, 303]
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_final = MyLR()
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)        
        
        elif method == 'CDF':
            selected_cols = [304, 305, 306]
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_final = MyLR()
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)        

        elif method == 'MAD':
            selected_cols = [307]
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_final = MyLR()
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)        

        elif method == 'Variability':
            selected_cols = [346, 347, 348, 349, 350]
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_final = MyLR()
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)        

        

        res = MyEval.F14Exp(pred, test_label)
        print(res)
        
        with open('../../stat/res_exp_for_paper.csv', 'a') as fout:
            fout.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'
                       .format(method, i_iter, i_fold, res[0], res[1], res[2], res[3], res[4]))
    
        i_fold += 1
    
if __name__ == "__main__":
    
    with open('../data/features_all_v2.5.pkl', 'rb') as my_input:
        all_pid = dill.load(my_input)
        all_feature = np.array(dill.load(my_input))
        all_label = np.array(dill.load(my_input))
        print('features_all shape: ', all_feature.shape)
        
    with open('../data/feat_deep_centerwave_v0.1.pkl', 'rb') as my_input:
        feat_deep_centerwave = np.array(dill.load(my_input))
        print('feat_deep_centerwave shape: ', feat_deep_centerwave.shape)
    
    with open('../data/feat_resnet.pkl', 'rb') as my_input:
        feat_resnet = np.array(dill.load(my_input))
        print('feat_resnet shape: ', feat_resnet.shape)
        
    
    all_feature = np.c_[all_feature, feat_deep_centerwave, feat_resnet]
    all_label = np.array(all_label)
    all_pid = np.array(all_pid)
    
    with open('../../stat/res_exp_for_paper.csv', 'a') as fout:
        fout.write('Method,n_iter,n_fold,F1_N,F1_A,F1_O,F1_P,F1\n')
    
    # method_list = ['SampleEn', 'CDF', 'MAD', 'Variability', 
    #                'LR_E', 'LR_EC', 'LR_ECD', 
    #                'XGBoost_E', 'XGBoost_EC', 'XGBoost_ECD', 
    #                'ENCASE_E', 'ENCASE_EC', 'ENCASE_ECD']    
    method_list = ['ENCASE_EC', 'ENCASE_ECD']
                   
    for method in method_list:
        for i in range(5):
            TestExp(all_pid, all_feature, all_label, method, i)
    
    