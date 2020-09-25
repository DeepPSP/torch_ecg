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

tf.logging.set_verbosity(tf.logging.INFO)

def read_data():
    with open('../../data1/expanded_three_part_window_3000_stride_500.pkl', 'rb') as fin:
        train_data = pickle.load(fin)
        train_label = pickle.load(fin)
        val_data = pickle.load(fin)
        val_label = pickle.load(fin)
        test_data = pickle.load(fin)
        test_label = pickle.load(fin)
    return train_data, train_label, val_data, val_label, test_data, test_label

    ## TODO normalization

n_dim = 3000
n_split = 300

tf.reset_default_graph()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
X, Y, valX, valY, testX, testY = read_data()
X = X.reshape([-1, n_dim, 1])
testX = testX.reshape([-1, n_dim, 1])

### split
#X = X.reshape([-1, n_split, 1])
#testX = testX.reshape([-1, n_split, 1])

# Building Residual Network
net = tflearn.input_data(shape=[None, n_dim, 1])
print("input", net.get_shape())
############ reshape for sub_seq 
net = tf.reshape(net, [-1, n_dim//n_split, n_split, 1])
print("reshaped input", net.get_shape())
net = tflearn.layers.conv.conv_2d_cnnlstm(net, 64, 16, 2)
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

net = tflearn.layers.conv.residual_bottleneck_cnnlstm(net, 2, 16, 64, downsample_strides = 2, downsample=True, is_first_block = True)
print("resn2", net.get_shape())
net = tflearn.layers.conv.residual_bottleneck_cnnlstm(net, 2, 16, 64, downsample_strides = 2, downsample=True)
print("resn4", net.get_shape())
net = tflearn.layers.conv.residual_bottleneck_cnnlstm(net, 2, 16, 128, downsample_strides = 2, downsample=True)
print("resn6", net.get_shape())
net = tflearn.layers.conv.residual_bottleneck_cnnlstm(net, 2, 16, 128, downsample_strides = 2, downsample=True)
print("resn8", net.get_shape())
net = tflearn.layers.conv.residual_bottleneck_cnnlstm(net, 2, 16, 256, downsample_strides = 2, downsample=True)
print("resn10", net.get_shape())
net = tflearn.layers.conv.residual_bottleneck_cnnlstm(net, 2, 16, 256, downsample_strides = 2, downsample=True)
print("resn12", net.get_shape())
'''net = tflearn.layers.conv.residual_bottleneck_cnnlstm(net, 2, 16, 512, downsample_strides = 2, downsample=True)
print("resn14", net.get_shape())
net = tflearn.layers.conv.residual_bottleneck_cnnlstm(net, 2, 16, 512, downsample_strides = 2, downsample=True)
print("resn16", net.get_shape())
net = tflearn.layers.conv.residual_bottleneck_cnnlstm(net, 2, 16, 1024, downsample_strides = 2, downsample=True)
print("resn18", net.get_shape())
net = tflearn.layers.conv.residual_bottleneck_cnnlstm(net, 2, 16, 1024, downsample_strides = 2, downsample=True)
print("resn20", net.get_shape())'''

net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
#net = tflearn.global_avg_pool(net)
# LSTM
print("before LSTM, before reshape", net.get_shape())
############ reshape for sub_seq 
net = tf.reshape(net, [-1, n_dim//n_split, 256*3])
print("before LSTM", net.get_shape())
net = bidirectional_rnn(net, BasicLSTMCell(16), BasicLSTMCell(16))
print("after LSTM", net.get_shape())
#net = dropout(net, 0.5)
# Regression
#net = tflearn.fully_connected(net, 64, activation='sigmoid')
#net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 4, activation='softmax')
print("dense", net.get_shape())
net = tflearn.regression(net, optimizer='momentum',
                         loss='categorical_crossentropy'
                         , learning_rate=0.1)
# Training
model = tflearn.DNN(net, checkpoint_path='../../models2/resnet_16_lstm',
                    max_checkpoints=10, clip_gradients=0.,  tensorboard_verbose=0)
model.fit(X, Y, n_epoch=10, validation_set=(valX, valY),
          show_metric=True, batch_size=200, run_id='resnet_3', snapshot_step=100,
          snapshot_epoch=False)


#Predict
cur_testX = []
y_predicted=[]
for i in range(13638):
    if (i % 300 == 0 or i/300 == 45) and i != 0:
        tmp_testX = np.array(cur_testX, dtype=np.float32)
        tmp_testX = tmp_testX.reshape([-1, n_dim, 1])
        y_predicted.extend(model.predict(tmp_testX))
        cur_testX = []
    cur_testX.append(testX[i])
#y_predicted=[model.predict(testX[i].reshape([-1, n_dim, 1])) for i in list(range(13638))]
#Calculate F1Score
MyEval.F1Score3_num(y_predicted, testY[:len(y_predicted)])

## save model
model.save('../model/ttt.tfl')

