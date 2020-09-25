#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 21:34:19 2017

@author: shenda
"""

from collections import Counter
import numpy as np
import pandas as pd
import MyEval
import ReadData
from LevelTop import LevelTop
import dill
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from BasicCLF import MyAdaBoost
from BasicCLF import MyRF
from BasicCLF import MyExtraTrees
from BasicCLF import MyGBDT
from BasicCLF import MyXGB
from BasicCLF import MyLR
from CascadeCLF import CascadeCLF
from OptF import OptF
import sklearn
import xgboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn import ensemble
from Encase import Encase

if __name__ == "__main__":

    with open('../../data2/features_all_v1.3.pkl', 'rb') as my_input:
        all_pid = dill.load(my_input)
        all_feature = dill.load(my_input)
        all_label = dill.load(my_input)
    
    all_feature = np.array(all_feature)
    all_label = np.array(all_label)
    
    clf_1 = MyLR()
    
    clf_final = Encase([clf_1])
    clf_final.fit(all_feature, all_label)
    
    with open('../model/model0423.pkl', 'wb') as my_out:
        dill.dump(clf_final, my_out)




