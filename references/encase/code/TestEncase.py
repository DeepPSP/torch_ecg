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
from BasicCLF import MyGBDT
from BasicCLF import MyXGB
from OptF import OptF
import sklearn
import xgboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn import ensemble
from Encase import Encase
from features_all import ReadAndExtractAll

def test_load():
    # with open('../data/features_all_v1.6.pkl', 'rb') as my_input:
    #     all_pid = dill.load(my_input)
    #     all_feature = dill.load(my_input)
    #     all_label = dill.load(my_input) 
        
    with open('../model/v2.5_xgb5_all_v2.pkl', 'rb') as fin:
        clf_final_final = dill.load(fin)
        
    pred = clf_final_final.predict(all_feature)
    print(MyEval.F1Score3(pred, all_label))
    
def gen_model():
    '''
    for online submit
    train on entire data
    '''
    with open('../data/features_all_v2.5.pkl', 'rb') as my_input:
        all_pid = dill.load(my_input)
        all_feature = dill.load(my_input)
        all_label = dill.load(my_input)

    all_pid = np.array(all_pid)
    all_feature = np.array(all_feature)
    all_label = np.array(all_label)
    print('all feature shape: {0}'.format(all_feature.shape))
    
    clf_1 = MyXGB()
    clf_2 = MyXGB()
    clf_3 = MyXGB() 
    clf_4 = MyXGB()
    clf_5 = MyXGB()
    clf_final = Encase([clf_1, clf_2, clf_3, clf_4, clf_5])
    
    print('start training')
    clf_final.fit(all_feature, all_label)
    print('done training')

    pred = clf_final.predict(all_feature)
    print(MyEval.F1Score3(pred, all_label))
    
    # with open('../../tmp_model/v2.5_xgb4_all.pkl', 'wb') as fout:
    with open('../model/v2.5_xgb5_all_v2.pkl', 'wb') as fout:
        dill.dump(clf_final, fout)
    print('save model done')



#def gen_model_1():   
def TestEncase(all_pid, all_feature, all_label):
#if __name__ == "__main__":

#    with open('../data/features_all_v2.2.pkl', 'rb') as my_input:
#        all_pid = np.array(dill.load(my_input))
#        feat_feature = np.array(dill.load(my_input))
#        all_label = np.array(dill.load(my_input))
#    
##    mean_wave = np.array(ReadData.read_mean_wave())
#    mean_wave = np.array(ReadData.read_mean_wave_simp())
#    all_feature = np.array(np.c_[mean_wave, feat_feature])
#    all_feature = np.array(mean_wave)   
    
    wrong_stat = []
    
    clf_final_list = []

    ## k-fold cross validation

    all_feature = np.array(all_feature)
    all_label = np.array(all_label)
    all_pid = np.array(all_pid)
    F1_list = []
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    i_fold = 1
    print('all feature shape: {0}'.format(all_feature.shape))
    for train_index, test_index in kf.split(all_feature, all_label):
        train_data = all_feature[train_index]
        train_label = all_label[train_index]
        test_data = all_feature[test_index]
        test_label = all_label[test_index]
        test_pid = all_pid[test_index]
        
        clf_1 = MyXGB()
        clf_2 = MyXGB()
        clf_3 = MyXGB() 
        clf_4 = MyXGB()
        clf_5 = MyXGB()

        
        clf_final = Encase([clf_1, clf_2, clf_3, clf_4, clf_5])
        clf_final.fit(train_data, train_label)
        
        pred = clf_final.predict(test_data)
#        pred_train = clf_final.predict(train_data)
#        MyEval.F1Score3(pred_train, train_label)
        F1_list.append(MyEval.F1Score3(pred, test_label, False))
        # wrong_stat.extend(MyEval.WrongStat(i_fold, pred, test_label, test_pid))
        i_fold += 1
        
        clf_final_list.append(clf_final)
        
    avg_f1 = np.mean(F1_list)
    print('\n\nAvg F1: ', avg_f1)
    # wrong_stat = pd.DataFrame(wrong_stat, columns=['i_fold', 'pid', 'gt', 'pred'])
    # wrong_stat.to_csv('../../stat/wrong_stat_f1'+str(np.mean(F1_list))+'.csv')
    
    
    clf_final_final = Encase(clf_final_list)
    pred = clf_final_final.predict(all_feature)
    print(MyEval.F1Score3(pred, all_label))
    
    with open('../../tmp_model/v2.5_v0.1/v2.5_v0.1_'+str(avg_f1)+'.pkl', 'wb') as fout:
        dill.dump(clf_final_final, fout)

    
if __name__ == "__main__":
    
    gen_model()
    
    # ReadAndExtractAll()
        
#     with open('../data/features_all_v2.5.pkl', 'rb') as my_input:
#         all_pid = np.array(dill.load(my_input))
#         all_feature = np.array(dill.load(my_input))
#         all_label = np.array(dill.load(my_input))
#     print(all_feature.shape)    
    

#     for i in range(100):
#         TestEncase(all_pid, all_feature, all_label)
    
    
    
    # with open('../model/v2.5_xgb5_all_v2.pkl', 'rb') as fin:
    #     clf_final = dill.load(fin)
        
        