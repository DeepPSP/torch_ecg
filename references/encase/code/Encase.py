#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 18:54:12 2017

@author: shenda


class order: ['A', 'N', 'O', '~']

"""

from CDL import CDL
import dill
import numpy as np

class Encase(object):
    def __init__(self, clf_list):
        self.clf_list = clf_list
        self.n_clf = len(self.clf_list)
        self.prob_list = [[] for i in range(self.n_clf)]
        self.final_prob = None
        self.pred_list = []
        self.labels = ['N', 'A', 'O', '~']
        self.weight = [1/self.n_clf for i in range(self.n_clf)]
    
    def fit(self, train_data, train_label):
        for clf in self.clf_list:
            clf.fit(train_data, train_label)

    def predict_prob(self, test_data):
        for i in range(self.n_clf):
            self.prob_list[i] = self.weight[i] * self.clf_list[i].predict_prob(test_data)
        
        self.final_prob = np.sum(np.array(self.prob_list), axis=0)
        
        return self.final_prob
    
    def predict(self, test_data):
        self.final_prob = self.predict_prob(test_data)
        self.pred_list = []
        
        n_row, _ = self.final_prob.shape
        for i in range(n_row):
            tmp_pred = self.final_prob[i, :]
            self.pred_list.append(self.labels[list(tmp_pred).index(max(tmp_pred))])
        return self.pred_list
        

if __name__ == "__main__":
    pass