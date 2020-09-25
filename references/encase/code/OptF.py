# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:06:35 2017

@author: v-shon
"""

import numpy as np
import sklearn
from collections import Counter


class OptF(object):
    def __init__(self, alpha=0.5, epochs=100):
        self.alpha = alpha
        self.epochs = epochs
        self.theta = None
    
    def gradTheta(self, theta, train_data, train_label):
        """
        Jansche, Martin. EMNLP 2005
        "Maximum expected F-measure training of logistic regression models." 
        
        must be normalized first
        """
        n_row, n_col = train_data.shape
        m = 0.0
        A = 0.0
        dm = np.zeros([n_col, 1])
        dA = np.zeros([n_col, 1])
        p = np.zeros([n_row, 1])
        
        p = 1.0 / (1.0 + np.exp(-np.dot(train_data, theta)))
        
        m = sum(p)
        A = sum(p * train_label)
        
        dm = np.dot(np.transpose(train_data), p * (1 - p))
        dA = np.dot(np.transpose(train_data), p * (1 - p) * train_label)
        
        n_pos = sum(train_label)
        h = 1 / (self.alpha * n_pos + (1 - self.alpha) * m)
        F = h * A
        t = F * (1 - self.alpha)
    
        dF = h * (dA - t * dm)
        
        return F, dF
    
    def fit(self, train_data, train_label):
        train_feature = sklearn.preprocessing.scale(train_data, axis=0)
        n_row, n_col = train_feature.shape
        train_feature = np.c_[np.ones([n_row, 1]), train_feature]
        train_label = np.expand_dims(np.array(train_label), axis=1)
        
        self.theta = np.random.rand(n_col+1, 1)
        
        for epoch in range(self.epochs):
            F, dF = self.gradTheta(self.theta, train_feature, train_label)

            self.theta = self.theta + dF


    def predict_prob(self, test_data):
        test_data = np.array(test_data)
        if test_data.ndim == 1:
            test_data = np.expand_dims(test_data, axis=0)
        test_feature = test_data
        n_row, n_col = test_feature.shape
        test_feature = np.c_[np.ones([n_row, 1]), test_feature]
#        print(test_feature)

        z = np.dot(test_feature, self.theta)
        gz = 1 / (1 + np.exp(-z))
        
        return gz

    def predict(self, test_data):
        gz = self.predict_prob(test_data)
        out = []
        for prob in gz:
            if prob > 0.5:
                out.append(1)
            else:
                out.append(0)
        return out










