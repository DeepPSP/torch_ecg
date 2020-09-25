#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:50:23 2017

@author: shenda
"""

import ReadData
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
import dill
import os.path


#if __name__ == "__main__":
def read_seq():
    long_pid, long_data, long_label = ReadData.ReadData( '../../data1/long.csv' )

    seq_pid = []
    seq_data = []
    seq_label = []
    
    seq_len = 1000

    for i in range(len(long_pid)):
        ts = long_data[i]
        for j in range(len(ts) // seq_len):
            seq_data.append(ts[j*seq_len : (j+1)*seq_len])
            seq_pid.append(long_pid[i])
            seq_label.append(long_label[i])
    
    long_label = seq_label
    seq_data = np.array(seq_data, dtype=np.float32)
    seq_data = normalize(seq_data, axis=0)
    
    seq_label = ReadData.Label2OneHot(seq_label)
    seq_label = np.array(seq_label, dtype=np.float32)
    
    all_feature = seq_data
    all_label = seq_label
    
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(all_feature, long_label):
        train_data = all_feature[train_index]
        train_label = all_label[train_index]
        test_data = all_feature[test_index]
        test_label = all_label[test_index]
        break
    
    train_data = np.expand_dims(np.array(train_data, dtype=np.float32), axis=2)
    test_data = np.expand_dims(np.array(test_data, dtype=np.float32), axis=2)
    
    return train_data, train_label, test_data, test_label