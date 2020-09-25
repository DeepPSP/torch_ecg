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
from sklearn.model_selection import KFold
from BasicCLF import MyKNN


#def gen_model_1():   
def TestKNN(all_pid, all_feature, all_label, fout):
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
    # kf = KFold(n_splits=5, shuffle=True)
    i_fold = 1
    print('all feature shape: {0}'.format(len(all_feature)))
    for train_index, test_index in kf.split(all_feature, all_label):
    # for train_index, test_index in kf.split(all_feature):
        train_data = all_feature[train_index]
        train_label = all_label[train_index]
        test_data = all_feature[test_index]
        test_label = all_label[test_index]
        test_pid = all_pid[test_index]
        
        clf_final = MyKNN(n_neighbors=1)
        clf_final.fit(train_data, train_label)
        
        pred = clf_final.predict(test_data)
#        pred_train = clf_final.predict(train_data)
#        MyEval.F1Score3(pred_train, train_label)
        F1_test, re_table = MyEval.F1Score3(pred, test_label, True)
        for line in re_table:
            for i in line:
                fout.write(str(i) + '\t')
            fout.write('\n')
        fout.write(str(F1_test)+'\n')
        F1_list.append(F1_test)
        wrong_stat.extend(MyEval.WrongStat(i_fold, pred, test_label, test_pid))
        i_fold += 1
        
        clf_final_list.append(clf_final)
    
    avg_f1 = np.mean(F1_list)
    print('\n\nAvg F1: ', avg_f1)
    fout.write(str(avg_f1)+'=============================\n')
    wrong_stat = pd.DataFrame(wrong_stat, columns=['i_fold', 'pid', 'gt', 'pred'])
    # wrong_stat.to_csv('../../stat/wrong_stat_f1'+str(np.mean(F1_list))+'.csv')

    
if __name__ == "__main__":
    
    all_feature = ReadData.read_centerwave('../../data1/centerwave_raw.csv')
    all_pid, _, all_label = ReadData.ReadData( '../../data1/QRSinfo.csv' )
    # print(sorted([len(i) for i in all_feature])[:100])
    all_feature = [np.array(i) for i in all_feature]
    
    # all_pid = all_pid[:5]
    # all_label = all_label[:5]
    # all_feature = all_feature[:5]
    
    print('read data done')
    fout = open('../../logs/knn', 'w')
    for i in range(100):
        TestKNN(all_pid, all_feature, all_label, fout)
    fout.close()
    
    