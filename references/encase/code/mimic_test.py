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
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from sklearn.model_selection import StratifiedKFold
import MyEval
import pickle
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.core import dropout
from BasicCLF import MyXGB
from BasicCLF import MyLR
from collections import Counter
import util_vote

tf.logging.set_verbosity(tf.logging.INFO)

def group_label(pids, preds, gt):
    unique_pids = sorted(list(set(pids)))
    pred_dic = {k: [] for k in unique_pids}
    gt_dic = {k: [] for k in unique_pids}
    final_preds = []
    final_gt = []
    for i in range(len(pids)):
        pred_dic[pids[i]].append(preds[i])
        gt_dic[pids[i]].append(gt[i])
    for k, v in pred_dic.items():
        final_preds.append(Counter(v).most_common(1)[0][0])
    for k, v in gt_dic.items():
        final_gt.append(Counter(v).most_common(1)[0][0])
    return final_preds, final_gt
    

def read_data():
    with open('../data/mimic_data_offline_v3.pkl', 'rb') as fin:
        data_dict = pickle.load(fin)
        test_data = data_dict['test_data']
        mimic_train_feats = data_dict['mimic_train_feats']
        train_label = data_dict['train_label']
        test_label = data_dict['test_label']
        test_pid = data_dict['test_pid']

    return test_data, mimic_train_feats, train_label, test_label, test_pid

def get_deep_mimic_feats(test_data):
    n_dim = 6000
    n_split = 300

    tf.reset_default_graph()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    # Building Residual Network
    net = tflearn.input_data(shape=[None, n_dim, 1])
    print("input", net.get_shape())
    ############ reshape for sub_seq 
    net = tf.reshape(net, [-1, n_split, 1])
    print("reshaped input", net.get_shape())
    net = tflearn.conv_1d(net, 64, 16, 2)
    #net = tflearn.conv_1d(net, 64, 16, 2, regularizer='L2', weight_decay=0.0001)
    print("cov1", net.get_shape())
    net = tflearn.batch_normalization(net)
    print("bn1", net.get_shape())
    net = tflearn.activation(net, 'relu')
    print("relu1", net.get_shape())

    # Residual blocks
    net = tflearn.residual_bottleneck(net, 2, 16, 64, downsample_strides = 2, downsample=True, is_first_block = True)
    print("resn2", net.get_shape())
    net = tflearn.residual_bottleneck(net, 2, 16, 128, downsample_strides = 4, downsample=True)
    print("resn4", net.get_shape())
    # net = tflearn.residual_bottleneck(net, 2, 16, 256, downsample_strides = 4, downsample=True)
    # print("resn6", net.get_shape())
    # net = tflearn.residual_bottleneck(net, 2, 16, 512, downsample_strides = 4, downsample=True)
    # print("resn8", net.get_shape())


    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    print("before reshape", net.get_shape())

    # net = tf.reshape(net, [-1, n_dim//n_split*net.get_shape()[-2], net.get_shape()[-1]])
    # LSTM
    ############ reshape for sub_seq 
    before_reshaped_shape = net.get_shape().as_list()
    net = tf.reshape(net, [-1, n_dim//n_split, before_reshaped_shape[1]*before_reshaped_shape[2]])
    print("before LSTM", net.get_shape())
    net = bidirectional_rnn(net, BasicLSTMCell(64), BasicLSTMCell(64))
    print("after LSTM", net.get_shape())
    net = dropout(net, 0.5)

    # Regression
    net = tflearn.fully_connected(net, 32, activation='sigmoid')
    net = tflearn.fully_connected(net, 4, activation='softmax')
    print("dense", net.get_shape())
    net = tflearn.regression(net, optimizer='adam', loss='mean_square')


    # Training
    model = tflearn.DNN(net)
    model.load('../model/mimic/mimic_model_offline_v4.1')

    pred = []
    num_of_test = len(test_data)
    cur_data = []
    for i in range(num_of_test):
        cur_data.append(test_data[i])
        if (i % 2000 == 0 or i == (num_of_test - 1)) and i !=0:
            tmp_testX = np.array(cur_data, dtype=np.float32)
            pred.extend(model.predict(tmp_testX.reshape([-1, n_dim, 1])))
            cur_data = []
    
    return pred
    
def get_label(proba):
    labels = ['A', 'N', 'O', '~']
    pred_label = labels[np.argsort(proba)[-1]]
    return pred_label

if __name__ == '__main__':
    test_data, mimic_train_feats, train_label, test_label, test_pid = read_data()
    mimic_test_feats = get_deep_mimic_feats(test_data)
    
    clf = MyLR()
    clf.fit(mimic_train_feats, np.array(ReadData.OneHot2Label(train_label)))
    print(clf.clf.coef_)
    pred_train = clf.predict(mimic_train_feats)
    MyEval.F1Score3(pred_train, np.array(ReadData.OneHot2Label(train_label)))

    pred_test = clf.predict(mimic_test_feats)
    pred_test_prob = clf.predict_prob(mimic_test_feats)
    MyEval.F1Score3(pred_test, np.array(ReadData.OneHot2Label(test_label)))

    gt_symbol = np.array(ReadData.OneHot2Label(test_label))
    
    # final_preds, final_gt = group_label(test_pid, pred_test, gt_symbol)
    
    final_preds_proba = util_vote.get_voted_proba(pred_test_prob, test_pid)
    final_preds = []
    for i in final_preds_proba:
        final_preds.append(get_label(i))
    final_gt = util_vote.group_gt(gt_symbol, test_pid)
    
    MyEval.F1Score3(final_preds, final_gt)





