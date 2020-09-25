#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 23:00:34 2017

@author: shenda
"""

from collections import Counter
import numpy as np
import FeatureExtract
import MyEval
import ReadData
import dill
import features_all
import challenge_encase_mimic

##############
#### load classifier
###############
#with open('model/v2.5_xgb5_all.pkl', 'rb') as my_in:
#    clf_final = dill.load(my_in)

##############
#### read and extract
###############
short_pid, short_data, short_label = ReadData.ReadData( 'data1/short.csv' )
long_pid, long_data, long_label = ReadData.ReadData( 'data1/long.csv' )
QRS_pid, QRS_data, QRS_label = ReadData.ReadData( 'data1/QRSinfo.csv' )


#############
### feature
#############
#all_feature = features_all.GetAllFeature_test(short_data, long_data, QRS_data, long_pid, short_pid)
#out_feats = features_mimic.get_mimic_feature(long_data[0])


############
## classifier
############
pred = []
pred = challenge_encase_mimic.pred_one_sample(short_data, long_data, QRS_data, long_pid, short_pid)

fout= open('answers.txt','a')
fout.write(pred[0])
fout.write('\n')
fout.close
