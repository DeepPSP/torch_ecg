#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 21:07:18 2017

@author: shenda
"""

import ReadData
from collections import Counter
import pickle
import hickle
import numpy as np
from sklearn.model_selection import StratifiedKFold


def slide_and_cut(tmp_data, tmp_label, tmp_pid):
    '''
    slide to get more samples from long data
    
    Counter({'N': 5050, 'O': 2456, 'A': 738, '~': 284})
    '''
       
    out_pid = []
    out_data = []
    out_label = []
    
    window_size = 6000
    
    cnter = {'N': 0, 'O': 0, 'A': 0, '~': 0}
    for i in range(len(tmp_data)):
        cnter[tmp_label[i]] += len(tmp_data[i])
    
    stride_N = 500
    stride_O = int(stride_N // (cnter['N'] / cnter['O']))
    stride_A = int(stride_N // (cnter['N'] / cnter['A']))
    stride_P = int(0.85 * stride_N // (cnter['N'] / cnter['~']))
    
    stride = {'N': stride_N, 'O': stride_O, 'A': stride_A, '~': stride_P}
    print(stride)

    for i in range(len(tmp_data)):
        if i % 1000 == 0:
            print(i)
        tmp_stride = stride[tmp_label[i]]
        tmp_ts = tmp_data[i]
        for j in range(0, len(tmp_ts)-window_size, tmp_stride):
            out_pid.append(tmp_pid[i])
            out_data.append(tmp_ts[j:j+window_size])
            out_label.append(tmp_label[i])

    print(Counter(out_label))
    
    idx = np.array(list(range(len(out_label))))
    out_label = ReadData.Label2OneHot(out_label)
    out_data = np.expand_dims(np.array(out_data, dtype=np.float32), axis=2)
    out_label = np.array(out_label, dtype=np.float32)
    out_pid = np.array(out_pid, dtype=np.string_)

    idx_shuffle = np.random.permutation(idx)
    out_data = out_data[idx_shuffle]
    out_label = out_label[idx_shuffle]
    out_pid = out_pid[idx_shuffle]
    
    return out_data, out_label, out_pid

def expand_three_part():
    long_pid, long_data, long_label = ReadData.ReadData( '../../data1/long.csv' )
    
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, other_index in kf.split(np.array(long_data), np.array(long_label)):
        train_data = np.array(long_data)[train_index]
        train_label = np.array(long_label)[train_index]
        train_pid = np.array(long_pid)[train_index]
        other_data = np.array(long_data)[other_index]
        other_label = np.array(long_label)[other_index]
        other_pid = np.array(long_pid)[other_index]

        kf_1 = StratifiedKFold(n_splits=2, shuffle=True)
        for val_index, test_index in kf_1.split(np.array(other_data), np.array(other_label)):
            val_data = np.array(other_data)[val_index]
            val_label = np.array(other_label)[val_index]
            val_pid = np.array(other_pid)[val_index]
            test_data = np.array(other_data)[test_index]
            test_label = np.array(other_label)[test_index]
            test_pid = np.array(other_pid)[test_index]

            break
        break
    
    train_data_out, train_label_out, train_data_pid_out = slide_and_cut(
            list(train_data), list(train_label), list(train_pid))
    val_data_out, val_label_out, val_data_pid_out = slide_and_cut(
            list(val_data), list(val_label), list(val_pid))
    test_data_out, test_label_out, test_data_pid_out = slide_and_cut(
            list(test_data), list(test_label), list(test_pid))
    
    print(len(set(list(train_pid)) & set(list(val_pid)) & set(list(test_pid))) == 0)
    
    # with open('../../data1/expanded_three_part_window_6000_stride_500_6.pkl', 'wb') as fout:
    #     pickle.dump(train_data_out, fout)
    #     pickle.dump(train_label_out, fout)
    #     pickle.dump(val_data_out, fout)
    #     pickle.dump(val_label_out, fout)
    #     pickle.dump(test_data_out, fout)
    #     pickle.dump(test_label_out, fout)
    #     pickle.dump(test_data_pid_out, fout)

    ### use np.save to save larger than 4 GB data
    fout = open('../../data1/expanded_three_part_window_6000_stride_299.bin', 'wb')
    np.save(fout, train_data_out)
    np.save(fout, train_label_out)
    np.save(fout, val_data_out)
    np.save(fout, val_label_out)
    np.save(fout, test_data_out)
    np.save(fout, test_label_out)
    np.save(fout, test_data_pid_out)
    fout.close()
    print('save done')

def expand_all():
    long_pid, long_data, long_label = ReadData.ReadData( '../../data1/long.csv' )
    data_out, label_out, pid_out = slide_and_cut(long_data, long_label, long_pid)

    ### use np.save to save larger than 4 GB data
    fout = open('../../data1/expanded_all_window_6000_stride_500.bin', 'wb')
    np.save(fout, data_out)
    np.save(fout, label_out)
    fout.close()
    print('save done')



if __name__ == "__main__":
    expand_all()
    



