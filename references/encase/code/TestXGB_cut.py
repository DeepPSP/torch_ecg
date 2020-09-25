#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 23:00:34 2017

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
from BasicCLF import MyXGB
from BasicCLF import MyLR
from OptF import OptF
import sklearn
import xgboost
from ReadData import shrink_set_to_seq

def gen_model():
    with open('../data/features_all_v1.6.pkl', 'rb') as my_input:
        all_pid = dill.load(my_input)
        all_feature = dill.load(my_input)
        all_label = dill.load(my_input)

    all_pid = np.array(all_pid)
    all_feature = np.array(all_feature)
    all_label = np.array(all_label)
    
    
        
    clf = MyXGB()
    clf.fit(all_feature, all_label)
        
    pred = clf.predict(all_feature)
    print(MyEval.F1Score3(pred, all_label))
    
    with open('../model/v1.6_xgb.pkl', 'wb') as fout:
        dill.dump(clf, fout)


#if __name__ == "__main__":
def TestXGB(fout, original_pid, original_label, all_pid, all_feature, all_label):
    

    
#    wrong_stat = []
    
    ## k-fold cross validation
    original_pid = np.array(original_pid)
    original_label = np.array(original_label)
    all_feature = np.array(all_feature)
    all_label = np.array(all_label)
    
    F1_list_set = []
    F1_list_seq = []
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    i_fold = 1
    for original_train_index, original_test_index in kf.split(original_label, original_label):

        original_train_pid = set(original_pid[original_train_index])
        original_test_pid = set(original_pid[original_test_index])

        train_index = []
        test_index = []
        for ii in range(len(all_pid)):
            ii_pid = all_pid[ii].split('_')[0]
            if ii_pid in original_train_pid:
                train_index.append(ii)
            elif ii_pid in original_test_pid:
                test_index.append(ii)
            else:
                print('wrong')
        train_index = np.array(train_index)
        test_index = np.array(test_index)
        
        train_data = all_feature[train_index]
        train_label = all_label[train_index]
        train_pid = np.array(all_pid)[train_index]
        test_data = all_feature[test_index]
        test_label = all_label[test_index]
        test_pid = np.array(all_pid)[test_index]

        clf = MyXGB()
        clf.fit(train_data, train_label)
        
        pred = clf.predict(test_data)
        pred_train = clf.predict(train_data)

        _, pred_train_seq = shrink_set_to_seq(train_pid, pred_train)
        _, train_label_seq = shrink_set_to_seq(train_pid, train_label)
        print('pred_train')
        MyEval.F1Score3(pred_train, train_label)
        print('pred_train_seq')
        MyEval.F1Score3(pred_train_seq, train_label_seq)

        
        _, pred_seq = shrink_set_to_seq(test_pid, pred)
        _, test_label_seq = shrink_set_to_seq(test_pid, test_label)
        print('\n pred')
        F1_list_set.append(MyEval.F1Score3(pred, test_label))
        print('pred_seq')
        f1_pred = MyEval.F1Score3(pred_seq, test_label_seq)
        F1_list_seq.append(f1_pred)
        print('=====================================')
#        wrong_stat.extend(MyEval.WrongStat(i_fold, pred, test_label, test_pid))
        fout.write('{0}, {1} \n'.format(i_fold, f1_pred))
        i_fold += 1

#        with open('../tmp_model/v1.9_xgb_z_'+str(f1_pred)+'.pkl', 'wb') as fout:
#            dill.dump(f1_pred, fout)      
#        break
    avg_f1 = np.mean(F1_list_seq)
    print('\n\nAvg F1: ', avg_f1)
#    wrong_stat = pd.DataFrame(wrong_stat, columns=['i_fold', 'pid', 'gt', 'pred'])
#    wrong_stat.to_csv('../../result/wrong_stat.csv')
    fout.write('avg, {0} \n'.format(f1_pred))

#    print(clf.bst.get_fscore())
#    xgboost.plot_importance(clf.bst)
#    xgboost.plot_tree(clf.bst)
#    xgboost.to_graphviz(clf.bst)
#    with open('../tmp_model/v1.9_xgb_x_'+str(avg_f1)+'.pkl', 'wb') as fout:
#        dill.dump(avg_f1, fout)

   

if __name__ == "__main__":
#    with open('../data/features_all_v1.6.pkl', 'rb') as my_input:
#        original_pid = dill.load(my_input)
#        original_feature = dill.load(my_input)
#        original_label = dill.load(my_input)
#    del original_feature    
#    with open('../data/features_all_v1.9.pkl', 'rb') as my_input:
#        all_pid = dill.load(my_input)
#        all_feature = dill.load(my_input)
#        all_label = dill.load(my_input)
#    all_feature = np.nan_to_num(all_feature)
#    print('read all data done')
    
    fout = open('../tmp_model/stat.csv', 'w')

    for i in range(100):
        TestXGB(fout, original_pid, original_label, all_pid, all_feature, all_label)
#    gen_model()

    fout.close()

