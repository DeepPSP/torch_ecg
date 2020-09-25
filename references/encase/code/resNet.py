# -*- coding: utf-8 -*-

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
    long_pid, long_data, long_label = ReadData.ReadData( '../data1/long.csv' )
    
    
    mat1 = [truncate_long(ts, 9000) for ts in long_data]
#    mat2 = [truncate_long(ts, 6000) for ts in long_data]
#    mat3 = [truncate_long(ts, 3000) for ts in long_data]
    
#    mat4 = [sample_long(ts, 10) for ts in mat1]
#    mat5 = [sample_long(ts, 10) for ts in mat2]
#    mat6 = [sample_long(ts, 10) for ts in mat3]

    
    label_onehot = ReadData.Label2OneHot(long_label)
    
#    plt.plot(mat1[0])
#    plt.plot(mat4[0])

    all_feature = np.array(mat1, dtype=np.float32)
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


tf.reset_default_graph()
X, Y, testX, testY = read_data()
X = X.reshape([-1, 9000, 1])
testX = testX.reshape([-1, 9000, 1])

# Building Residual Network
net = tflearn.input_data(shape=[None, 9000, 1])
net = tflearn.conv_1d(net, 64, 16, 2, activation='relu', bias=False)

# Residual blocks
net = tflearn.residual_bottleneck(net, 2, 16, 64, downsample_strides = 2, downsample=True, is_first_block = True)
print("resn2", net.get_shape())
'''net = tflearn.residual_bottleneck(net, 2, 16, 128, downsample_strides = 2, downsample=True)
print("resn4", net.get_shape())
net = tflearn.residual_bottleneck(net, 2, 16, 256, downsample_strides = 2, downsample=True)
print("resn6", net.get_shape())
net = tflearn.residual_bottleneck(net, 2, 16, 512, downsample_strides = 2, downsample=True)
print("resn8", net.get_shape())'''
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
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
print("beforeDense", net.get_shape())
# Regression
net = tflearn.fully_connected(net, 2, activation='softmax')
print("dense", net.get_shape())
net = tflearn.regression(net, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.1)
# Training
model = tflearn.DNN(net, checkpoint_path='model_resnet',
                    max_checkpoints=10, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=10, validation_set=(testX, testY),
          show_metric=True, batch_size=300, run_id='resnet', snapshot_step=10,
          snapshot_epoch=False)

#Predict
y_predicted=[i for i in model.predict(testX)]
#Calculate F1Score
MyEval.F1Score3_num(y_predicted, testY)
