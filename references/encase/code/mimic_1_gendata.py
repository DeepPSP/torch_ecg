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


def get_resnet_feature(test_data):
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
    net = tflearn.fully_connected(net, 32, activation='sigmoid')
    net = tflearn.dropout(net, 0.5)
    # net, feature_layer = tflearn.fully_connected(net, 4, activation='softmax', return_logit = True)
    feature_layer = tflearn.fully_connected(net, 4, activation='softmax')
    print('feature_layer: ', feature_layer.get_shape())
    print("dense", net.get_shape())
    net = tflearn.regression(net, optimizer='adam',#momentum',
                             loss='categorical_crossentropy')
                             #,learning_rate=0.1)
    print('final output: ', net.get_shape())
    ## save model
    ### load
    model = tflearn.DNN(net)
    run_id = 'resnet_6000_500_10_5_v1'
    model.load('../model/resNet/'+run_id)
    
    # print(tflearn.variables.get_all_variables())

    ### create new model, and get features
    m2 = tflearn.DNN(feature_layer, session=model.session)
    tmp_feature = []
    num_of_test = len(test_data)
    cur_data = []
    pre = []
    for i in range(num_of_test):
        cur_data.append(test_data[i])
        if (i % 2000 == 0 or i == (num_of_test - 1)) and i !=0:
            #tmp_test_data = test_data[i].reshape([-1, n_dim, 1])
            tmp_testX = np.array(cur_data, dtype=np.float32)
            tmp_feature.extend(m2.predict(tmp_testX.reshape([-1, n_dim, 1])))
            cur_data = []
            pre.extend(model.predict(tmp_testX))
            print(i, len(tmp_feature), len(tmp_feature[0]))

    tmp_feature = np.array(tmp_feature)

    return tmp_feature

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

def read_data_from_npsave():
    fin = open('../../data1/expanded_all_window_6000_stride_500.bin', 'rb')
    data_out = np.load(fin)
    label_out = np.load(fin)
    fin.close()
    print('read done')
    return data_out, label_out

def gen_data_offline():
    train_data, train_label, val_data, val_label, test_data, test_label, test_pid = read_data_from_pkl()
    pid_map = {}
    pid_set = set(test_pid)
    pids = list(pid_set)
    for i in range(len(pids)):
        cur_pid = str(pids[i],'utf-8')
        pid_map[cur_pid] = i
    
    mimic_train_feats = get_resnet_feature(train_data)
    mimic_val_feats = get_resnet_feature(val_data)

    print('train_data shape: ', train_data.shape)
    print('mimic_train_feats shape: ', mimic_train_feats.shape)
    
    data_dict = {}
    data_dict['train_data'] = train_data
    data_dict['val_data'] = val_data
    data_dict['test_data'] = test_data
    data_dict['mimic_train_feats'] = mimic_train_feats
    data_dict['mimic_val_feats'] = mimic_val_feats
    data_dict['train_label'] = train_label
    data_dict['val_label'] = val_label
    data_dict['test_label'] = test_label
    data_dict['test_pid'] = test_pid
    
    with open('../data/mimic_data_offline_v3.pkl', 'wb') as fout:
        dill.dump(data_dict, fout)
    print('gen_data_offline done')
        
def gen_data_online():
    all_data, all_label = read_data_from_npsave()
    
    # all_data = all_data[:10]
    # all_label = all_label[:10]
    
    mimic_all_feats = get_resnet_feature(all_data)
    # print('mimic_all_feats shape: ', mimic_all_feats.shape)
    # print(mimic_all_feats)
    # print(all_label)

    fout = open('../data/mimic_data_online_v3.1.bin', 'wb')
    np.save(fout, all_data)
    np.save(fout, mimic_all_feats)
    np.save(fout, all_label)
    fout.close()
    print('save done')

if __name__ == '__main__':
    
    # gen_data_offline()
    gen_data_online()
    # read_data()

    
    


