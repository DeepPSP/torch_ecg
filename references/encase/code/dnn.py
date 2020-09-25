#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 08:01:35 2017

@author: shenda

class order: ['A', 'N', 'O', '~']
"""



import numpy as np
from matplotlib import pyplot as plt
import ReadData
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib



def next_batch(my_data, my_label, batch_idx):
    start = (batch_idx * batch_size) % my_data.shape[0]
    end = ((batch_idx + 1) * batch_size + my_data.shape[0]) % my_data.shape[0]
    if end-start != batch_size:
        data_in, label_in = next_batch(my_data, my_label, batch_idx+1)
    else:
        data_in = my_data[start:end]
        label_in = my_label[start:end]

    return data_in, label_in

def truncate_long(ts, my_len):
    if len(ts) >= my_len:
        return ts[:my_len]
    else:
        ts += [0] * (my_len - len(ts))
        return ts
    
def sample_long(ts, interv):
    ts1 = []
    for i in range(len(ts) // interv):
        ts1.append(ts[i * interv])
    return ts1

def read_data():
    long_pid, long_data, long_label = ReadData.ReadData( '../../data1/long.csv' )
    
    mat1 = [truncate_long(ts, 9000) for ts in long_data]
    mat2 = [truncate_long(ts, 6000) for ts in long_data]
    mat3 = [truncate_long(ts, 3000) for ts in long_data]
    
    mat4 = [sample_long(ts, 10) for ts in mat1]
    mat5 = [sample_long(ts, 10) for ts in mat2]
    mat6 = [sample_long(ts, 10) for ts in mat3]
    
    label_onehot = ReadData.Label2OneHot(long_label)
    
#    plt.plot(mat1[0])
#    plt.plot(mat4[0])

    mat1 = np.expand_dims(np.array(mat1), axis=2)
    label_onehot = np.array(label_onehot)
    
    return mat1, label_onehot

batch_size = 100
n_input = 9000
n_classes = 4
epochs = 10
train_data, train_label = read_data()

#def my_dnn(features, labels, mode):

with tf.variable_scope('input'):    
    features = tf.placeholder(tf.float32, [batch_size, n_input, 1])
    labels = tf.placeholder(tf.float32, [batch_size, n_classes])

with tf.variable_scope('conv_1'):    
    conv_1 = tf.layers.conv1d(features, 
                            filters=8, kernel_size=16, strides=2, 
                            activation=tf.nn.relu,
                            padding='SAME', use_bias=True, reuse=False)
    print("conv_1", conv_1.get_shape())
    pool_1 = tf.layers.max_pooling1d(conv_1, pool_size=16, strides=8)
    print("pool_1", pool_1.get_shape())
    
with tf.variable_scope('conv_2'):    
    conv_2 = tf.layers.conv1d(pool_1, 
                            filters=32, kernel_size=8, strides=2, 
                            padding='SAME', use_bias=True, reuse=False)
    print("conv_2", conv_2.get_shape())
    pool_2 = tf.layers.max_pooling1d(conv_2, pool_size=8, strides=2)
    print("pool_2", pool_2.get_shape())
    
with tf.variable_scope('fc'):  
    
    lstm_cell = tf.contrib.rnn.LSTMCell(32, forget_bias=1.0, state_is_tuple=False)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, pool_2, dtype=tf.float32, time_major = False)
    print("outputs", outputs.get_shape())
    print("states", states.get_shape())
    
    dense = tf.layers.dense(inputs=states, units=16, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)
    print('dropout', dropout.get_shape())
                                
    logits = tf.layers.dense(inputs=dropout, units=n_classes)
                                
    cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels))
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

    pred = tf.argmax(logits, 1)
    pred_prob = tf.slice(tf.nn.softmax(logits), [0, 1], [-1, 1])
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Train
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_loss_cnt = 0.0
        epoch_loss_test = 0.0
    
        for batch_idx in range(int((train_data.shape[0]-1)/batch_size + 1)):
            batch_xs, batch_ys = next_batch(train_data, train_label, batch_idx)
            loss_v = sess.run(accuracy, feed_dict={features: batch_xs, labels: batch_ys})
            epoch_loss += loss_v
            epoch_loss_cnt += 1
    
        epoch_loss /= epoch_loss_cnt
        print ("Epoch #%-5d | Train acc: %-4.3f" %
              (epoch, epoch_loss))
    

