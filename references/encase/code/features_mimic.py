# -*- coding: utf-8 -*-

'''
split long seq into small sub_seq, 
feed sub_seq to lstm
'''

from __future__ import division, print_function, absolute_import

import tflearn
import tflearn.data_utils as du

import numpy as np
import ReadData
import tensorflow as tf
import MyEval
import dill
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.core import dropout
from collections import Counter
import util_vote

def slide_and_cut_all(tmp_data_list, tmp_pid):
    out_data = []
    out_pid = []
    
    window_size = 6000    
    stride = 100
    
    for i in range(len(tmp_data_list)):
        if len(tmp_data_list[i]) <= window_size+stride:
            tmp_data_list[i] = tmp_data_list[i] + [0.0]*(window_size+stride-len(tmp_data_list[i]))
    
    for i in range(len(tmp_data_list)):
        tmp_data = tmp_data_list[i]
        for j in range(0, len(tmp_data)-window_size, stride):
            out_data.append(tmp_data[j:j+window_size])
            out_pid.append(tmp_pid[i])
    
    out_data = np.expand_dims(np.array(out_data, dtype=np.float32), axis=2)
    
    return out_data, out_pid

# def group_label(pids, preds):
#     unique_pids = sorted(list(set(pids)))
#     pred_dic = {k: [] for k in unique_pids}
#     final_preds = []
#     for i in range(len(pids)):
#         pred_dic[pids[i]].append(preds[i])
#     for k, v in pred_dic.items():
#         final_preds.append(Counter(v).most_common(1)[0][0])
#     return final_preds

def slide_and_cut(tmp_data):
    out_data = []
    
    window_size = 6000    
    stride = 100
    
    if len(tmp_data) <= window_size+stride:
        tmp_data = tmp_data + [0.0]*(window_size+stride-len(tmp_data))

    for j in range(0, len(tmp_data)-window_size, stride):
        out_data.append(tmp_data[j:j+window_size])
    
    out_data = np.expand_dims(np.array(out_data, dtype=np.float32), axis=2)
    
    return out_data

# def get_deep_mimic_feats_v1(test_data):
#     n_dim = 6000
#     n_split = 300

#     tf.reset_default_graph()
#     sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# ################################################################################################
#     # Building Residual Network
#     net = tflearn.input_data(shape=[None, n_dim, 1])
#     print("input", net.get_shape())
#     ############ reshape for sub_seq 
#     net = tf.reshape(net, [-1, n_split, 1])
#     print("reshaped input", net.get_shape())
#     net = tflearn.conv_1d(net, 64, 16, 2)
#     print("cov1", net.get_shape())
#     net = tflearn.batch_normalization(net)
#     print("bn1", net.get_shape())
#     net = tflearn.activation(net, 'relu')
#     print("relu1", net.get_shape())

#     # Residual blocks
#     net = tflearn.residual_bottleneck(net, 2, 16, 64, downsample_strides = 4, downsample=True, is_first_block = True)
#     print("resn2", net.get_shape())
#     net = tflearn.residual_bottleneck(net, 2, 16, 128, downsample_strides = 4, downsample=True)
#     print("resn4", net.get_shape())
#     # net = tflearn.residual_bottleneck(net, 2, 16, 256, downsample_strides = 4, downsample=True)
#     # print("resn6", net.get_shape())
#     # net = tflearn.residual_bottleneck(net, 2, 16, 512, downsample_strides = 4, downsample=True)
#     # print("resn8", net.get_shape())


#     net = tflearn.batch_normalization(net)
#     net = tflearn.activation(net, 'relu')
#     print("before reshape", net.get_shape())
#     # LSTM
#     ############ reshape for sub_seq 
#     before_reshaped_shape = net.get_shape().as_list()
#     net = tf.reshape(net, [-1, n_dim//n_split, before_reshaped_shape[1]*before_reshaped_shape[2]])
#     print("before LSTM", net.get_shape())
#     net = bidirectional_rnn(net, BasicLSTMCell(64), BasicLSTMCell(64))
#     print("after LSTM", net.get_shape())
#     net = dropout(net, 0.5)

#     # Regression
#     net = tflearn.fully_connected(net, 32, activation='sigmoid')
#     net = tflearn.dropout(net, 0.5)
#     net = tflearn.fully_connected(net, 4)
#     print("dense", net.get_shape())
#     net = tflearn.regression(net, optimizer='adam', loss='mean_square')
# ################################################################################################

#     # Training
#     model = tflearn.DNN(net)
#     model.load('../model/mimic/mimic_model_online_v1')

#     feats = []
#     num_of_test = len(test_data)
#     cur_data = []
    
#     for i in range(num_of_test):
#         cur_data.append(test_data[i])
#         if (i % 100 == 0 or i == (num_of_test - 1)) and i !=0:
#             tmp_testX = np.array(cur_data, dtype=np.float32)
#             feats.extend(model.predict(tmp_testX.reshape([-1, n_dim, 1])))
#             cur_data = []
    
#     return feats

def get_deep_mimic_feats(test_data):
    n_dim = 6000
    n_split = 300

    tf.reset_default_graph()
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

################################################################################################
    # Building Residual Network
    net = tflearn.input_data(shape=[None, n_dim, 1])
    print("input", net.get_shape())
    ############ reshape for sub_seq 
    net = tf.reshape(net, [-1, n_split, 1])
    print("reshaped input", net.get_shape())
    net = tflearn.conv_1d(net, 64, 16, 2, regularizer='L2', weight_decay=0.0001, bias=True,
                          weights_init='variance_scaling', bias_init='zeros')
    print("cov1", net.get_shape())
    net = tflearn.batch_normalization(net)
    print("bn1", net.get_shape())
    net = tflearn.activation(net, 'relu')
    print("relu1", net.get_shape())

    # Residual blocks
    net = tflearn.residual_bottleneck(net, 2, 16, 64, downsample_strides = 2, downsample=True, is_first_block = True)
    print("resn2", net.get_shape())
    net = tflearn.residual_bottleneck(net, 2, 16, 128, downsample_strides = 3, downsample=True)
    print("resn4", net.get_shape())
    # net = tflearn.residual_bottleneck(net, 2, 16, 256, downsample_strides = 4, downsample=True)
    # print("resn6", net.get_shape())
    # net = tflearn.residual_bottleneck(net, 2, 16, 512, downsample_strides = 4, downsample=True)
    # print("resn8", net.get_shape())


    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    print("before reshape", net.get_shape())
    # LSTM
    ############ reshape for sub_seq 
    before_reshaped_shape = net.get_shape().as_list()
    net = tf.reshape(net, [-1, n_dim//n_split, before_reshaped_shape[1]*before_reshaped_shape[2]])
    print("before LSTM", net.get_shape())
    net = bidirectional_rnn(net, BasicLSTMCell(72), BasicLSTMCell(72))
    print("after LSTM", net.get_shape())
    net = dropout(net, 0.5)

    # Regression
    net = tflearn.fully_connected(net, 32, activation='sigmoid')
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, 4)
    print("dense", net.get_shape())
    net = tflearn.regression(net, optimizer='adam', loss='mean_square')
################################################################################################

    # Training
    model = tflearn.DNN(net)
    # model.load('../model/mimic/mimic_model_online_v3')
    model.load('model/mimic/mimic_model_online_v1.1')

    feats = []
    num_of_test = len(test_data)
    cur_data = []
    
    for i in range(num_of_test):
        cur_data.append(test_data[i])
        if (num_of_test >1 and (i%2000 ==0 or i == (num_of_test -1)) and i !=0) or (num_of_test==1):
            tmp_testX = np.array(cur_data, dtype=np.float32)
            feats.extend(model.predict(tmp_testX.reshape([-1, n_dim, 1])))
            cur_data = []
    
    return feats

#### v2
# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     return np.exp(x) / np.sum(np.exp(x), axis=0)

# def get_mimic_proba_v2(tmp_data):
#     '''
#     vote proba
    
#     input: raw long data
#     output: voted proba
#     '''
#     out_data = np.array(slide_and_cut(tmp_data))
#     out_feats = np.array(get_deep_mimic_feats_v2(out_data))
#     print(out_data.shape, out_feats.shape)
    
#     out_proba = softmax(out_feats)
#     pred_proba_voted = get_voted_proba(out_proba)

#     return pred_proba_voted
    
#### v1
# def get_mimic_proba(tmp_data):
#     out_data = np.array(slide_and_cut(tmp_data))
#     out_feats = np.array(get_deep_mimic_feats(out_data))
#     print(out_data.shape, out_feats.shape)
#     with open('../model/mimic/mimic_online_LR_v1.1.pkl', 'rb') as fin:
#         clf = dill.load(fin)
#     out_proba = clf.predict_prob(out_feats)
#     pred_proba_voted = util_vote.get_voted_proba_each_1(out_proba)
    
#     ### from ano~ to nao~
#     final_proba = [0,0,0,0]
#     final_proba[0] = pred_proba_voted[1]
#     final_proba[1] = pred_proba_voted[0]
#     final_proba[2] = pred_proba_voted[2]
#     final_proba[3] = pred_proba_voted[3]

#     return np.array(final_proba)

def get_mimic_proba_all(tmp_data, tmp_pid):
    out_data, out_pid = slide_and_cut_all(tmp_data, tmp_pid)
    out_feats = np.array(get_deep_mimic_feats(np.array(out_data)))
    return np.array(out_feats), out_pid

def get_mimic_proba(tmp_data):
    out_data = np.array(slide_and_cut(tmp_data))
    out_feats = np.array(get_deep_mimic_feats(out_data))
    print(out_data.shape, out_feats.shape)
    out_proba = out_feats
    pred_proba_voted = util_vote.get_voted_proba_each_1(out_proba)

    return np.array(pred_proba_voted)
        
def get_label(proba):
    labels = ['N', 'A', 'O', '~']
    pred_label = labels[np.argsort(proba)[-1]]
    return pred_label
    
#################################################
####### for test
#################################################
if __name__ == '__main__':
    long_pid, long_data, long_label = ReadData.ReadData( '../../data_val/long.csv' )
    
    long_pid = long_pid[0]
    long_data = long_data[0]
    long_label = long_label[0]
    
    out_proba = get_mimic_proba(long_data)
    print(out_proba)
    
#     out_data = np.array(slide_and_cut(long_data))
#     out_proba_1 = np.array(get_deep_mimic_feats(out_data))
    
#     with open('../model/mimic/mimic_online_LR.pkl', 'rb') as fin:
#         clf = dill.load(fin)
#     out_proba_2 = clf.predict_prob(out_proba_1)
        
#     print(out_proba, out_proba_1, out_proba_2)
    
#     with open('../model/mimic/mimic_online_LR.pkl', 'rb') as fin:
#         clf = dill.load(fin)    
#     set_proba = clf.predict_prob(out_feats)
    
#     seq_proba = []
#     proba_dic = {}
#     for i in range(len(out_pid)):
#         if out_pid[i] in proba_dic:
#             proba_dic[out_pid[i]].append(set_proba[i])
#         else:
#             proba_dic[out_pid[i]] = [set_proba[i]]
#     for pid in long_pid:
#         seq_proba.append(get_voted_proba(proba_dic[pid]))
    
#     set_pred = []
#     for i in range(len(out_pid)):
#         set_pred.append(get_label(set_proba[i]))    
#     MyEval.F1Score3(set_pred, out_label)    
    
#     seq_pred = []
#     for i in range(len(out_pid)):
#         seq_pred.append(get_label(seq_proba[i]))    
#     MyEval.F1Score3(seq_pred, long_label)
