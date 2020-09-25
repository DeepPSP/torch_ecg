#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 08:01:35 2017

@author: shenda

class order: ['A', 'N', 'O', '~']
"""



import numpy as np
import ReadData
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from sklearn.model_selection import StratifiedKFold
import MyEval
from read_data import read_seq

tf.logging.set_verbosity(tf.logging.INFO)


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
    
#    mat1 = [truncate_long(ts, 9000) for ts in long_data]
#    mat2 = [truncate_long(ts, 6000) for ts in long_data]
    mat3 = [truncate_long(ts, 3000) for ts in long_data]
    
#    mat4 = [sample_long(ts, 10) for ts in mat1]
#    mat5 = [sample_long(ts, 10) for ts in mat2]
#    mat6 = [sample_long(ts, 10) for ts in mat3]
    
    label_onehot = ReadData.Label2OneHot(long_label)
    
#    plt.plot(mat1[0])
#    plt.plot(mat4[0])

    mat = mat3

    all_feature = np.array(mat, dtype=np.float32)
    all_label = np.array(label_onehot, dtype=np.float32)

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


def model_fn(features, labels, mode, params):
   # Logic to do the following:
   # 1. Configure the model via TensorFlow operations
   # 2. Define the loss function for training/evaluation
   # 3. Define the training operation/optimizer
   # 4. Generate predictions
   # 5. Return predictions/loss/train_op/eval_metric_ops in ModelFnOps object  


    conv_1 = tf.layers.conv1d(features, 
                            filters=256, kernel_size=16, strides=2, 
                            activation=None,
                            kernel_initializer=tf.truncated_normal_initializer(),
                            bias_initializer=tf.truncated_normal_initializer(),
                            padding='SAME', use_bias=True, reuse=False)
    print("conv_1", conv_1.get_shape())
    bn_1 = tf.contrib.layers.batch_norm(conv_1)
    print("bn_1", bn_1.get_shape())
    relu_1 = tf.nn.relu(bn_1)
    print("relu_1", relu_1.get_shape())
    pool_1 = tf.layers.max_pooling1d(relu_1, pool_size=16, strides=2)
    print("pool_1", pool_1.get_shape())
    
#    conv_2 = tf.layers.conv1d(pool_1, 
#                            filters=256, kernel_size=16, strides=2, 
#                            activation=None,
#                            kernel_initializer=tf.truncated_normal_initializer(),
#                            bias_initializer=tf.truncated_normal_initializer(),
#                            padding='SAME', use_bias=True, reuse=False)
#    print("conv_2", conv_2.get_shape())
#    bn_2 = tf.contrib.layers.batch_norm(conv_2)
#    print("bn_2", bn_2.get_shape())
#    relu_2 = tf.nn.relu(bn_2)
#    print("relu_2", relu_2.get_shape())
#    pool_2 = tf.layers.max_pooling1d(relu_2, pool_size=16, strides=2)
#    print("pool_2", pool_2.get_shape())
    
    pool_shape = pool_1.get_shape().as_list()
    pool2_flat = tf.reshape(pool_1, [-1, pool_shape[1] * pool_shape[2]])
    print("pool2_flat", pool2_flat.get_shape())
    dense = tf.layers.dense(inputs=pool2_flat, units=1000, activation=tf.nn.sigmoid)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)
    print("dropout", dropout.get_shape())
#
#    features_shape = features.get_shape().as_list()
#    print(features_shape)
#    features = tf.reshape(features, [-1, features_shape[2], features_shape[1]])
#    print("features", features.get_shape())

#    lstm_cell = tf.contrib.rnn.LSTMCell(100, forget_bias=1.0, state_is_tuple=False)
#    att_cell = tf.contrib.rnn.AttentionCellWrapper(lstm_cell, attn_length=16, input_size=16)
#    outputs, states = tf.nn.dynamic_rnn(att_cell, pool_2, dtype=tf.float32, time_major = False)
#    print("outputs", outputs.get_shape())
#
#    outputs_shape = outputs.get_shape().as_list()
#    outputs_flat = tf.reshape(outputs, [-1, outputs_shape[1] * outputs_shape[2]])
#    print("outputs_flat", outputs_flat.get_shape())

    logits = tf.layers.dense(inputs=dropout, units=params['n_classes'], activation=tf.nn.sigmoid)
    print(logits.get_shape())
                                
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels))

    train_op = tf.contrib.layers.optimize_loss(
          loss=loss,
          global_step=tf.contrib.framework.get_global_step(),
          learning_rate=params["learning_rate"],
          optimizer="Adam")
    
    predictions = tf.argmax(logits, 1)
    predictions_dict = {"preds": predictions}
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
       
    eval_metric_ops = {
            "accuracy":
                tf.reduce_mean(tf.cast(correct_prediction, tf.float64)), 
            "preds":
                tf.nn.softmax(logits)
      }
    

    return model_fn_lib.ModelFnOps(
          mode=mode,
          predictions=predictions_dict,
          loss=loss,
          train_op=train_op,
          eval_metric_ops=eval_metric_ops)


def main(unused_argv):
#    train_data, train_label, test_data, test_label = read_data()
    
    model_params = {"learning_rate": 0.01,
                    "n_classes": 4}
    
    def batched_input_fn(dataset_x, dataset_y, batch_size):
        def _input_fn():
            all_x = tf.constant(dataset_x, shape=dataset_x.shape, dtype=tf.float32)
            all_y = tf.constant(dataset_y, shape=dataset_y.shape, dtype=tf.float32)
            sliced_input = tf.train.slice_input_producer([all_x, all_y])
            return tf.train.batch(sliced_input, batch_size=batch_size)
        return _input_fn

    def get_train_inputs():
        x = tf.constant(train_data)
        y = tf.constant(train_label)
        return x, y
    def get_test_inputs():
        x = tf.constant(test_data[:10,])
        y = tf.constant(test_label[:10,])
        return x, y
    
    nn = tf.contrib.learn.Estimator(
            model_fn=model_fn, params=model_params,
            model_dir="../../tmp7")
    
    nn.fit(x=train_data, y=train_label, batch_size = BATCH_SIZE,
           steps=1000)

    print("Training Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

    ev_train = nn.evaluate(input_fn=batched_input_fn(train_data, train_label, BATCH_SIZE), steps=10)
    print("Loss: %s" % ev_train["loss"])
    print("accuracy: %s" % ev_train["accuracy"])
    print("preds: %s" % ev_train["preds"])
    MyEval.F1Score3_num(ev_train["preds"], train_label)
    
    ev_test = nn.evaluate(input_fn=batched_input_fn(test_data, test_label, BATCH_SIZE), steps=10)
    print("Loss: %s" % ev_test["loss"])
    print("accuracy: %s" % ev_test["accuracy"])
    print("preds: %s" % ev_test["preds"])
    MyEval.F1Score3_num(ev_test["preds"], test_label)


if __name__ == "__main__":
    BATCH_SIZE = 100
    train_data, train_label, test_data, test_label = read_seq()
#    train_label = train_label[:,[0,2]]
#    test_label = test_label[:,[0,2]]
    tf.app.run()






