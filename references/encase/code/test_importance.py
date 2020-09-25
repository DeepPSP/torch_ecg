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
        
    
    # k-fold cross validation
    all_feature = np.c_[all_feature, feat_deep_centerwave, feat_resnet]
    all_label = np.array(all_label)
    
    train_data = all_feature
    train_label = all_label
    
    clf = MyXGB()
    clf.fit(train_data, train_label)
    print('train done')
    
    imp_scores = clf.get_importance()
    feat_num = all_feature.shape[1]
    imp_scores_key_num = set([int(k[1:]) for k in imp_scores.keys()])
    print(feat_num)
    print(len(imp_scores))
    
    pred_train = clf.predict(train_data)
    MyEval.F1Score3(pred_train, train_label)
    
    with open('../../stat/feat_imp_v2.5_v0.1_v0.1.csv', 'w') as fout:
        for i in range(1,feat_num+1):
            if i in imp_scores_key_num:
                fout.write('{0},{1}\n'.format(i, imp_scores['f'+str(i)]))
            else:
                fout.write('{0},{1}\n'.format(i, 0))
                              










