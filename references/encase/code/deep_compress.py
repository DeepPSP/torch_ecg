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
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.core import dropout
import dill
import MyEval
import pickle


def pruning(nd_vec):
    '''
    pruning small values and return 
    '''
    thresh = 1e-1
    nd_vec[np.abs(nd_vec) < thresh] = 0
    return nd_vec

def read_data():
    long_pid, long_data, long_label = ReadData.ReadData( '../../data1/long.csv' )
    all_pid = np.array(long_pid)
    all_feature = np.array(long_data)
    all_label = np.array(long_label)
    print('read data done')
    data_out, label_out, pid_out = slide_and_cut(all_feature, all_label, all_pid)
    
    pid_map = {}
    for i in range(len(all_pid)):
        pid_map[all_pid[i]] = i
    
    return data_out, label_out, pid_out, pid_map

    
def slide_and_cut(tmp_data, tmp_label, tmp_pid):
       
    out_pid = []
    out_data = []
    out_label = []
    
    window_size = 6000
    
    cnter = {'N': 0, 'O': 0, 'A': 0, '~': 0}
    for i in range(len(tmp_data)):
        #print(tmp_label[i])
        if cnter[tmp_label[i]] is not None:
            cnter[tmp_label[i]] += len(tmp_data[i])
    
    stride_N = 500
    stride_O = int(stride_N // (cnter['N'] / cnter['O']))
    stride_A = int(stride_N // (cnter['N'] / cnter['A']))
    stride_P = int(0.85 * stride_N // (cnter['N'] / cnter['~']))
    
    stride = {'N': stride_N, 'O': stride_O, 'A': stride_A, '~': stride_P}

    for i in range(len(tmp_data)):
        tmp_stride = stride[tmp_label[i]]
        tmp_ts = tmp_data[i]
        for j in range(0, len(tmp_ts)-window_size, tmp_stride):
            out_pid.append(tmp_pid[i])
            out_data.append(tmp_ts[j:j+window_size])
            out_label.append(tmp_label[i])
    
    out_label = ReadData.Label2OneHot(out_label)
    out_data = np.expand_dims(np.array(out_data, dtype=np.float32), axis=2)
    out_label = np.array(out_label, dtype=np.float32)
    out_pid = np.array(out_pid, dtype=np.string_)
    
    return out_data, out_label, out_pid


def get_model():
    n_dim = 6000
    n_split = 300

    tf.reset_default_graph()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    ### split
    #X = X.reshape([-1, n_split, 1])
    #testX = testX.reshape([-1, n_split, 1])

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
    '''net = tflearn.residual_bottleneck(net, 2, 16, 64, downsample_strides = 2, downsample=True, is_first_block = True)
    print("resn2", net.get_shape())
    net = tflearn.residual_bottleneck(net, 2, 16, 128, downsample_strides = 2, downsample=True)
    print("resn4", net.get_shape())
    net = tflearn.residual_bottleneck(net, 2, 16, 256, downsample_strides = 2, downsample=True)
    print("resn6", net.get_shape())
    net = tflearn.residual_bottleneck(net, 2, 16, 512, downsample_strides = 2, downsample=True)
    print("resn8", net.get_shape())
    net = tflearn.residual_bottleneck(net, 2, 16, 1024, downsample_strides = 2, downsample=True)
    print("resn10", net.get_shape())'''

    net = tflearn.residual_bottleneck(net, 2, 16, 64, downsample_strides = 2, downsample=True, is_first_block = True)
    print("resn2", net.get_shape())
    net = tflearn.residual_bottleneck(net, 2, 16, 64, downsample_strides = 2, downsample=True)
    print("resn4", net.get_shape())
    net = tflearn.residual_bottleneck(net, 2, 16, 128, downsample_strides = 2, downsample=True)
    print("resn6", net.get_shape())
    net = tflearn.residual_bottleneck(net, 2, 16, 128, downsample_strides = 2, downsample=True)
    print("resn8", net.get_shape())
    net = tflearn.residual_bottleneck(net, 2, 16, 256, downsample_strides = 2, downsample=True)
    print("resn10", net.get_shape())
    net = tflearn.residual_bottleneck(net, 2, 16, 256, downsample_strides = 2, downsample=True)
    print("resn12", net.get_shape())
    net = tflearn.residual_bottleneck(net, 2, 16, 512, downsample_strides = 2, downsample=True)
    print("resn14", net.get_shape())
    net = tflearn.residual_bottleneck(net, 2, 16, 512, downsample_strides = 2, downsample=True)
    print("resn16", net.get_shape())
    '''net = tflearn.residual_bottleneck(net, 2, 16, 1024, downsample_strides = 2, downsample=True)
    print("resn18", net.get_shape())
    net = tflearn.residual_bottleneck(net, 2, 16, 1024, downsample_strides = 2, downsample=True)
    print("resn20", net.get_shape())'''

    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    #net = tflearn.global_avg_pool(net)
    # LSTM
    print("before LSTM, before reshape", net.get_shape())
    ############ reshape for sub_seq 
    net = tf.reshape(net, [-1, n_dim//n_split, 512])
    print("before LSTM", net.get_shape())
    net = bidirectional_rnn(net, BasicLSTMCell(256), BasicLSTMCell(256))
    print("after LSTM", net.get_shape())
    #net = tflearn.layers.recurrent.lstm(net, n_units=512)
    #print("after LSTM", net.get_shape())
    net = dropout(net, 0.5)

    # Regression
    feature_layer = tflearn.fully_connected(net, 32, activation='sigmoid')
    net = tflearn.dropout(feature_layer, 0.5)
    net = tflearn.fully_connected(net, 4, activation='softmax')
    print("dense", net.get_shape())
    net = tflearn.regression(net, optimizer='adam',#momentum',
                             loss='categorical_crossentropy')
                             #,learning_rate=0.1)
    ## save model
    ### load
    model = tflearn.DNN(net)
    run_id = 'resnet_6000_500_10_5_v1'
    model.load('../model/resNet/'+run_id)
    
    
    all_names = tflearn.variables.get_all_variables()
    print(all_names[0])
    ttt = model.get_weights(all_names[0])
    print(type(ttt))
    print(ttt)
    
    # tflearn.variables.get_value(all_names[0], xxx)
        
        
    return all_names


def read_data_from_pkl():
    with open('../../data1/expanded_three_part_window_6000_stride_500_5.pkl', 'rb') as fin:
        train_data = pickle.load(fin)
        train_label = pickle.load(fin)
        val_data = pickle.load(fin)
        val_label = pickle.load(fin)
        test_data = pickle.load(fin)
        test_label = pickle.load(fin)
        test_pid= pickle.load(fin)
    return train_data, train_label, val_data, val_label, test_data, test_label, test_pid

if __name__ == '__main__':
    '''all_data, all_label, all_pid, pid_map = read_data()
    out_feature = get_resnet_feature(all_data, all_label, all_pid, pid_map)
    print('out_feature shape: ', out_feature.shape)
    with open('../data/feat_resnet.pkl', 'wb') as fout:
        dill.dump(out_feature, fout)
    '''
    '''
    #-----------------------------------------------test--------------------------------------------
    '''

    # all_names = get_model()
    
    vec = np.random.normal(size=[2,3,4])
    print(vec)
    vec = pruning(vec)
    print(vec)


