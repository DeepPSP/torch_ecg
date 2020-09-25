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


tf.logging.set_verbosity(tf.logging.INFO)

def read_data():
    X = ReadData.read_centerwave('../../data1/centerwave_resampled.csv')
    _, _, Y = ReadData.ReadData( '../../data1/QRSinfo.csv' )
    all_feature = np.array(X)
    print(all_feature.shape)
    all_label = np.array(Y)
    all_label_num = np.array(ReadData.Label2OneHot(Y))
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    i_fold = 1
    print('all feature shape: {0}'.format(all_feature.shape))
    for train_index, test_index in kf.split(all_feature, all_label):
        train_data = all_feature[train_index]
        train_label = all_label_num[train_index]
        test_data = all_feature[test_index]
        test_label = all_label_num[test_index]
    print('read data done')
    return all_feature, all_label_num, train_data, train_label, test_data, test_label



tf.reset_default_graph()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

################### read data
all_data, all_label, X, Y, valX, valY = read_data()
print(X.shape, valX.shape)
print(Y.shape, valY.shape)
_, n_dim = X.shape
all_data = all_data.reshape([-1, n_dim, 1])
X = X.reshape([-1, n_dim, 1])
valX = valX.reshape([-1, n_dim, 1])

################### model

### input
net = tflearn.input_data(shape=[None, n_dim, 1], name='input')
print("input", net.get_shape())

### conv
net = tflearn.conv_1d(net, 64, 16, 2, regularizer='L2', weight_decay=0.0005, bias=True,
                        weights_init='variance_scaling', bias_init='zeros')
print("cov1", net.get_shape())
net = tflearn.batch_normalization(net)
print("bn1", net.get_shape())
net = tflearn.activation(net, 'relu')
print("relu1", net.get_shape())


### lstm
net = bidirectional_rnn(net, BasicLSTMCell(64), BasicLSTMCell(64))
print("lstm", net.get_shape())
feature_layer = tflearn.fully_connected(net, 32, activation='sigmoid', name='dense_1')
net = feature_layer
print("feature_layer", net.get_shape())
net = tflearn.fully_connected(feature_layer, 4, activation='softmax', name='output')
print("dense", net.get_shape())
net = tflearn.regression(net, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.1)


### Training
run_id = 'deep_centerwave_v1'
model = tflearn.DNN(net, checkpoint_path='../../models/model_deep_centerwave',
                    max_checkpoints=10, tensorboard_verbose=0)

model.fit(X, Y, n_epoch=10, validation_set=(valX, valY),
          show_metric=True, batch_size=128, run_id=run_id, snapshot_step=1000,
          snapshot_epoch=False)

# save model
model.save('../model/model_deep_centerwave/' + run_id)
print('model save done')