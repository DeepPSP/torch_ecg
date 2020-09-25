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


def read_data_offline():
    with open('../data/mimic_data_offline_v3.pkl', 'rb') as fin:
        data_dict = pickle.load(fin)
        train_data = data_dict['train_data']
        mimic_train_feats = data_dict['mimic_train_feats']
        val_data = data_dict['val_data']
        mimic_val_feats = data_dict['mimic_val_feats']
    # return train_data[:1000], mimic_train_feats[:1000], val_data[:1000], mimic_val_feats[:1000]
    return train_data, mimic_train_feats, val_data, mimic_val_feats

def read_data_online_pkl():
    with open('../data/mimic_data_online_v1.pkl', 'rb') as fin:
        data_dict = pickle.load(fin)
        all_data = data_dict['all_data']
        mimic_all_feats = data_dict['mimic_all_feats']
    return all_data, mimic_all_feats, all_data[:100], mimic_all_feats[:100]

def read_data_online():
    with open('../data/mimic_data_online_v3.bin', 'rb') as fin:
        all_data = np.load(fin)
        mimic_all_feats = np.load(fin)
    # return all_data[:1000], mimic_all_feats[:1000], all_data[:100], mimic_all_feats[:100]
    return all_data, mimic_all_feats, all_data[:100], mimic_all_feats[:100]

def build_student_model():
    tf.logging.set_verbosity(tf.logging.INFO)

    n_dim = 6000
    n_split = 300

    tf.reset_default_graph()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    ######################################## online offline
    # X, Y, valX, valY = read_data_online_pkl()
    X, Y, valX, valY = read_data_offline()
    X = X.reshape([-1, n_dim, 1])

################################################################################################
##### v1.1
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
    ######################################## online offline
    # run_id = 'mimic_model_online_v1.1'
    run_id = 'mimic_model_offline_v4.2'
    model = tflearn.DNN(net, checkpoint_path='../../models3/offline_v4.2',
                        max_checkpoints=10, tensorboard_verbose=0)
    model.fit(X, Y, n_epoch=10, validation_set=(valX, valY),
              show_metric=True, batch_size=256, run_id=run_id, snapshot_step=100,
              snapshot_epoch=False)


    ## save model
    model.save('../model/mimic/'+run_id)

    
if __name__ == '__main__':
    build_student_model()
    print('build_student_model done')
    
    
    