#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 14:45:00 2017

@author: shenda

class order: ['A', 'N', 'O', '~']
"""
import numpy as np
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from OptF import OptF
from copy import deepcopy
import xgboost as xgb
import ReadData
import random
from collections import Counter



class MyXGB(object):
    """
    bottom basic classifier, a warpper for xgboost
    the proba order is ['N', 'A', 'O', '~']
    """
    def __init__(self, n_estimators=5000, 
                 max_depth=10, 
                 subsample=0.85, 
                 colsample_bytree=0.85, 
                 min_child_weight=4, 
                 num_round = 500):
        
        self.param = {'learning_rate':0.1, 'eta':0.1, 'silent':1, 
        'objective':'multi:softprob', 'num_class': 4}
        self.bst = None
        self.num_round = num_round
        self.pred = None
        
        my_seed = random.randint(0, 1000)
        
        self.param['n_estimators'] = n_estimators
        self.param['max_depth'] = max_depth
        self.param['subsample'] = subsample
        self.param['colsample_bytree'] = colsample_bytree
        self.param['min_child_weight'] = min_child_weight
#        self.param['random_state'] = my_seed
        self.param['seed'] = my_seed
        self.param['n_jobs'] = -1
        
        print(self.param.items())
        print(self.num_round)

    def fit(self, train_data, train_label):
        train_label = ReadData.Label2Index(train_label)
        dtrain = xgb.DMatrix(train_data, label=train_label)
        self.bst = xgb.train(self.param, dtrain, num_boost_round=self.num_round)

    def predict_prob(self, test_data):
        dtest = xgb.DMatrix(test_data)
        self.pred = self.bst.predict(dtest)
        return self.pred
    
    def predict(self, test_data):
        pred_prob = self.predict_prob(test_data)
        pred_num = np.argmax(pred_prob, axis=1)
        pred = ReadData.Index2Label(pred_num)
        return pred
    
    def get_importance(self):
        return self.bst.get_score(importance_type='gain')
    
    def plot_importance(self):
        xgb.plot_importance(self.bst)

        
class MyLR(object):
    """
    Top level classifier, a warpper for Logistic Regression
    """
    def __init__(self):
        self.clf = LogisticRegression()
        
    def fit(self, 
            train_qrs_data, train_qrs_label):
        train_data =np.array(train_qrs_data)
        train_label = train_qrs_label
        self.clf.fit(train_data, train_label)
    
    def predict(self, test_qrs_data):
        test_data = np.array(test_qrs_data)
        if test_data.ndim == 1:
            test_data = np.expand_dims(np.array(test_data), axis=0)
        return list(self.clf.predict(test_data))
    
    def predict_prob(self, test_qrs_data):
        test_qrs_data = np.array(test_qrs_data)
        if test_qrs_data.ndim == 1:
            test_qrs_data = np.expand_dims(np.array(test_qrs_data), axis=0)
            test_data = np.array(test_qrs_data)
            return list(list(self.clf.predict_proba(test_data))[0])
        else:
            test_data = np.array(test_qrs_data)
            return self.clf.predict_proba(test_data)

class MyKNN(object):
    """
    bottom basic
    support unequal length vector
    """
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.train_data = None
        self.train_label = None
        self.labels = ['N', 'A', 'O', '~']
        # self.thresh = [0.5, 0.3, 0.2, ]
        
    def fit(self, train_data, train_label):
        self.train_data = np.array(train_data)
        self.train_label = np.array(train_label)
    
    def dist(self, vec1, vec2):
        res = 0.0
        if len(vec1) <= len(vec2):
            vec1 = np.r_[vec1, np.zeros(len(vec2)-len(vec1))]
        else:
            vec2 = np.r_[vec2, np.zeros(len(vec1)-len(vec2))]
        dist_num = np.linalg.norm(vec1 - vec2)
        return dist_num
    
    def predict_prob(self, test_data):
        test_data = np.array(test_data)
        pred = []
        for i in test_data:
            tmp_dist_list = []
            tmp_pred = []
            for j in self.train_data:
                tmp_dist_list.append(self.dist(i, j))
            pred_n_neighbors = self.train_label[np.argsort(tmp_dist_list)[:self.n_neighbors]]
            pred_counter = Counter(pred_n_neighbors)
            # print(pred_counter)
            for ii in self.labels:
                tmp_pred.append(pred_counter[ii])
            pred.append(tmp_pred)
        return pred
    
    def predict(self, test_data):
        pred = self.predict_prob(test_data)
        pred_label = []
        for i in pred:
            pred_label.append(self.labels[np.argsort(i)[-1]])
        return pred_label

class MyGBDT(object):
    """
    bottom basic  a warpper for GradientBoostingClassifier
    """
    def __init__(self):
        self.clf = ensemble.GradientBoostingClassifier()
        
    def fit(self, 
            train_data, train_label):
        train_data =np.array(train_data)
        self.clf.fit(train_data, train_label)
    
    def predict(self, test_data):
        test_data = np.array(test_data)
        if test_data.ndim == 1:
            test_data = np.expand_dims(np.array(test_data), axis=0)
        return list(self.clf.predict(test_data))
    
    def predict_prob(self, test_data):
        test_data = np.array(test_data)
        if test_data.ndim == 1:
            test_data = np.expand_dims(np.array(test_data), axis=0)
            test_data = np.array(test_data)
            return list(list(self.clf.predict_proba(test_data))[0])
        else:
            test_data = np.array(test_data)
            return self.clf.predict_proba(test_data)
        
class MyExtraTrees(object):
    """
    bottom basic  a warpper for ExtraTreesClassifier
    """
    def __init__(self):
        self.clf = ensemble.ExtraTreesClassifier(n_estimators=100)
        
    def fit(self, 
            train_data, train_label):
        train_data =np.array(train_data)
        self.clf.fit(train_data, train_label)
    
    def predict(self, test_data):
        test_data = np.array(test_data)
        if test_data.ndim == 1:
            test_data = np.expand_dims(np.array(test_data), axis=0)
        return list(self.clf.predict(test_data))
    
    def predict_prob(self, test_data):
        test_data = np.array(test_data)
        if test_data.ndim == 1:
            test_data = np.expand_dims(np.array(test_data), axis=0)
            test_data = np.array(test_data)
            return list(list(self.clf.predict_proba(test_data))[0])
        else:
            test_data = np.array(test_data)
            return self.clf.predict_proba(test_data)

class MyAdaBoost(object):
    """
    bottom basic  a warpper for AdaBoostClassifier
    """
    def __init__(self):
        self.clf = ensemble.AdaBoostClassifier(n_estimators=100, learning_rate=0.1)
        
    def fit(self, 
            train_data, train_label):
        train_data =np.array(train_data)
        self.clf.fit(train_data, train_label)
    
    def predict(self, test_data):
        test_data = np.array(test_data)
        if test_data.ndim == 1:
            test_data = np.expand_dims(np.array(test_data), axis=0)
        return list(self.clf.predict(test_data))
    
    def predict_prob(self, test_data):
        test_data = np.array(test_data)
        if test_data.ndim == 1:
            test_data = np.expand_dims(np.array(test_data), axis=0)
            test_data = np.array(test_data)
            return list(list(self.clf.predict_proba(test_data))[0])
        else:
            test_data = np.array(test_data)
            return self.clf.predict_proba(test_data)
        
class MyRF(object):
    """
    bottom basic  a warpper for Random Forest
    """
    def __init__(self):
        self.clf = ensemble.RandomForestClassifier(
                n_estimators=1000, n_jobs=-1)
        
    def fit(self, 
            train_data, train_label):
        train_data =np.array(train_data)
        self.clf.fit(train_data, train_label)
    
    def predict(self, test_data):
        test_data = np.array(test_data)
        if test_data.ndim == 1:
            test_data = np.expand_dims(np.array(test_data), axis=0)
        return list(self.clf.predict(test_data))
    
    def predict_prob(self, test_data):
        test_data = np.array(test_data)
        if test_data.ndim == 1:
            test_data = np.expand_dims(np.array(test_data), axis=0)
            test_data = np.array(test_data)
            return list(list(self.clf.predict_proba(test_data))[0])
        else:
            test_data = np.array(test_data)
            return self.clf.predict_proba(test_data)
        
        

class MyOptF(object):
    """
    bottom basic classifier, a warpper for Opt F-score
    """
    def __init__(self, alpha=0.5, epochs=10):
        self.clf = OptF(alpha, epochs)
        
    def fit(self, train_data, train_label):
        self.clf.fit(train_data, deepcopy(train_label))
    
    def predict(self, test_data):
        return list(self.clf.predict(test_data))
    
    def predict_prob(self, test_data):
        return self.clf.predict_prob(test_data)


class RF(object):
    """
    Top level classifier, a warpper for Random Forest
    
    use long_feature and qrs_feature seperatedly, thus no use any more
    
    deprecated
    """
    def __init__(self):
        self.clf = ensemble.RandomForestClassifier()
        
    def fit(self, 
            train_long_data, train_long_label, 
            train_qrs_data, train_qrs_label):
        train_data = np.c_[np.array(train_long_data), np.array(train_qrs_data)]
        train_label = train_long_label
        self.clf.fit(train_data, train_label)
    
    def predict(self, test_long_data, test_qrs_data):
        test_data = np.c_[np.array(test_long_data), np.array(test_qrs_data)]
        if test_data.ndim == 1:
            test_data = np.expand_dims(np.array(test_data), axis=0)
        return list(self.clf.predict(test_data))
    
    def predict_prob(self, test_long_data, test_qrs_data):
        test_long_data = np.array(test_long_data)
        test_qrs_data = np.array(test_qrs_data)
        if test_long_data.ndim == 1 or test_qrs_data.ndim == 1:
            test_long_data = np.expand_dims(np.array(test_long_data), axis=0)
            test_qrs_data = np.expand_dims(np.array(test_qrs_data), axis=0)
            test_data = np.c_[np.array(test_long_data), np.array(test_qrs_data)]
        else:
            test_data = np.c_[np.array(test_long_data), np.array(test_qrs_data)]
        return list(list(self.clf.predict_proba(test_data))[0])
    
class RFSimp(object):
    """
    Top level classifier, a warpper for Random Forest
    
    use long/qrs feature
    
    deprecated

    """
    def __init__(self):
        self.clf = ensemble.RandomForestClassifier()
        
    def fit(self, 
            train_qrs_data, train_qrs_label):
        train_data =np.array(train_qrs_data)
        train_label = train_qrs_label
        self.clf.fit(train_data, train_label)
    
    def predict(self, test_qrs_data):
        test_data = np.array(test_qrs_data)
        if test_data.ndim == 1:
            test_data = np.expand_dims(np.array(test_data), axis=0)
        return list(self.clf.predict(test_data))
    
    def predict_prob(self, test_qrs_data):
        test_qrs_data = np.array(test_qrs_data)
        if test_qrs_data.ndim == 1:
            test_qrs_data = np.expand_dims(np.array(test_qrs_data), axis=0)
            test_data = np.array(test_qrs_data)
            return list(list(self.clf.predict_proba(test_data))[0])
        else:
            test_data = np.array(test_qrs_data)
            return self.clf.predict_proba(test_data)
   
class LR(object):
    """
    Top level classifier, a warpper for Logistic Regression
    
    deprecated
    """
    def __init__(self):
        self.clf = LogisticRegression()
        
    def fit(self, 
            train_long_data, train_long_label, 
            train_qrs_data, train_qrs_label):
        train_data = np.c_[np.array(train_long_data), np.array(train_qrs_data)]
        train_label = train_long_label
        self.clf.fit(train_data, train_label)
    
    def predict(self, test_long_data, test_qrs_data):
        test_data = np.c_[np.array(test_long_data), np.array(test_qrs_data)]
        if test_data.ndim == 1:
            test_data = np.expand_dims(np.array(test_data), axis=0)
        return list(self.clf.predict(test_data))
    
    def predict_prob(self, test_long_data, test_qrs_data):
        test_long_data = np.array(test_long_data)
        test_qrs_data = np.array(test_qrs_data)
        if test_long_data.ndim == 1 or test_qrs_data.ndim == 1:
            test_long_data = np.expand_dims(np.array(test_long_data), axis=0)
            test_qrs_data = np.expand_dims(np.array(test_qrs_data), axis=0)
            test_data = np.c_[np.array(test_long_data), np.array(test_qrs_data)]
        else:
            test_data = np.c_[np.array(test_long_data), np.array(test_qrs_data)]
        return list(list(self.clf.predict_proba(test_data))[0])     
