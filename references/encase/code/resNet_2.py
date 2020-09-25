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
    with open('../../data1/expanded_three_part_window_6000_stride_500_1.pkl', 'rb') as fin:
        train_data = pickle.load(fin)
        train_label = pickle.load(fin)
        val_data = pickle.load(fin)
        val_label = pickle.load(fin)
        test_data = pickle.load(fin)
        test_label = pickle.load(fin)
        test_pid= pickle.load(fin)
    return train_data, train_label, val_data, val_label, test_data, test_label, test_pid

    ## TODO normalization

n_dim = 6000
n_split = 300

tf.reset_default_graph()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
X, Y, valX, valY, testX, testY, test_pid = read_data()
X = X.reshape([-1, n_dim, 1])
testX = testX.reshape([-1, n_dim, 1])

### split
#X = X.reshape([-1, n_split, 1])
#testX = testX.reshape([-1, n_split, 1])

# Building Residual Network
net = tflearn.input_data(shape=[None, n_dim, 1])
print("input", net.get_shape())
############ reshape for sub_seq 
net = tf.reshape(net, [-1, n_split, 1])
print("reshaped input", net.get_shape())
net = tflearn.conv_1d(net, 64, 16, 2, bias=True)
#net = tflearn.conv_1d(net, 64, 16, 2, regularizer='L2', weight_decay=0.0005, bias=True,
                        #weights_init='variance_scaling', bias_init='zeros')
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

net = tflearn.residual_bottleneck(net, 2, 16, 64, downsample_strides = 2, downsample=True, is_first_block = True, weight_decay=0.00005)
print("resn2", net.get_shape())
net = tflearn.residual_bottleneck(net, 2, 16, 64, downsample_strides = 2, downsample=True, weight_decay=0.00005)
print("resn4", net.get_shape())
net = tflearn.residual_bottleneck(net, 2, 16, 128, downsample_strides = 2, downsample=True, weight_decay=0.00005)
print("resn6", net.get_shape())
net = tflearn.residual_bottleneck(net, 2, 16, 128, downsample_strides = 2, downsample=True, weight_decay=0.00005)
print("resn8", net.get_shape())
net = tflearn.residual_bottleneck(net, 2, 16, 256, downsample_strides = 2, downsample=True, weight_decay=0.00005)
print("resn10", net.get_shape())
net = tflearn.residual_bottleneck(net, 2, 16, 256, downsample_strides = 2, downsample=True, weight_decay=0.00005)
print("resn12", net.get_shape())
net = tflearn.residual_bottleneck(net, 2, 16, 512, downsample_strides = 2, downsample=True, weight_decay=0.00005)
print("resn14", net.get_shape())
net = tflearn.residual_bottleneck(net, 2, 16, 512, downsample_strides = 2, downsample=True, weight_decay=0.00005)
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
net = tflearn.fully_connected(net, 4, activation='softmax')
print("dense", net.get_shape())
net = tflearn.regression(net, optimizer='adam',#momentum',
                         loss='categorical_crossentropy')
                         #,learning_rate=0.1)
# Training
run_id = 'resnet_6000_500_10_1_v4'
model = tflearn.DNN(net, checkpoint_path='../../models2/resnet_bilstm',
                    max_checkpoints=10, tensorboard_verbose=0)

x_list=list(X)
y_list=list(Y)
valX_list=list(valX)
valY_list=list(valY)

x_list.extend(valX_list)
y_list.extend(valY_list)

X=np.array(x_list)
Y=np.array(y_list)
model.fit(X, Y, n_epoch=10, #validation_set=(valX, valY),
          show_metric=True, batch_size=200, run_id=run_id, snapshot_step=100,
          snapshot_epoch=False)

#Predict
cur_testX = []
y_predicted=[]
for i in range(len(testX)):
    cur_testX.append(testX[i])
    if (i % 300 == 0 or i == (len(testX) -1)) and i != 0:
        tmp_testX = np.array(cur_testX, dtype=np.float32)
        tmp_testX = tmp_testX.reshape([-1, n_dim, 1])
        y_predicted.extend(model.predict(tmp_testX))
        cur_testX = []


f1, re_table = MyEval.F1Score3_num(y_predicted, testY[:len(y_predicted)])

f = open("../../logs/"+run_id, 'a')  
print(re_table, file=f)
print(f1, file=f)
f.close()


pid_set = set(test_pid)
pids = list(pid_set)
pids.sort()
y_num = len(pids)
pre = [[0., 0., 0., 0.] for i in  range(y_num)]#np.zeros([y_num,4])
y_sec_pre = [[0. for j in range(4)] for i in  range(y_num)]
y_third_pre = [[0. for j in range(4)] for i in  range(y_num)]
y_groundtruth= [[0. for j in range(4)] for i in range(y_num)]
#print (len(y_groundtruth))
#print (len(testY))
    
for j in range(len(y_predicted)):
    i_pred = y_predicted[j]
    i_pred = np.array(y_predicted[j], dtype=np.float32)
    cur_pid = test_pid[j]
    list_id = pids.index(cur_pid)
    #print (list_id)
    temp_pre = pre[list_id]
    temp_sec_pre = np.array(y_sec_pre[list_id], dtype=np.float32)
    temp_third_pre = np.array(y_third_pre[list_id], dtype=np.float32)
    
    max_p = temp_pre[np.argmax(temp_pre)]
    max_sec_p = temp_sec_pre[np.argmax(temp_sec_pre)]
    max_third_p = temp_third_pre[np.argmax(temp_third_pre)]
    sec_p = 0
    sec_sec_p = 0
    sec_third_p = 0
    for k in range(len(temp_pre)):
        if temp_pre[k] == max_p:
            continue
        if temp_pre[k] > sec_p:
            sec_p = temp_pre[k]

        if temp_sec_pre[k] == max_sec_p:
            continue
        if temp_sec_pre[k] > sec_sec_p:
            sec_sec_p = temp_sec_pre[k]

        if temp_third_pre[k] == max_third_p:
            continue
        if temp_third_pre[k] > sec_third_p:
            sec_third_p = temp_third_pre[k]
            
    cur_max_p = i_pred[np.argmax(i_pred)]
    cur_sec_p = 0
    for k in range(len(i_pred)):
        if i_pred[k] == cur_max_p:
            continue
        if i_pred[k] > cur_sec_p:
            cur_sec_p = i_pred[k]

    if (cur_max_p - cur_sec_p) > (max_p - sec_p):
        y_third_pre[list_id] = y_sec_pre[list_id]
        y_sec_pre[list_id] = pre[list_id]
        pre[list_id] = i_pred
    elif (cur_max_p - cur_sec_p) > (max_sec_p - sec_sec_p):
        y_third_pre[list_id] = y_sec_pre[list_id]
        y_sec_pre[list_id] = i_pred
    elif (cur_max_p - cur_sec_p) > (max_third_p - sec_third_p):
        y_third_pre[list_id] = i_pred
    
    #print(j)
    y_groundtruth[list_id] = testY[j]
#y_predicted=[model.predict(testX[i].reshape([-1, n_dim, 1])) for i in list(range(13638))]
#Calculate F1Score

for k in range(len(pre)):
    labels = [0. for j in range(4)]
    pred_1 = np.argmax(pre[k])
    labels[pred_1] +=1
    pred_2 = np.argmax(y_sec_pre[k])
    labels[pred_2] +=1
    pred_3 = np.argmax(y_third_pre[k])
    labels[pred_3] +=1

    if pred_1 == 2:
        print("O was selected!")
        continue
    elif pred_2 == 2:
        pre[k] = y_sec_pre[k]
        print("O was selected!")
    elif pred_3 == 2:
        pre[k] = y_third_pre[k]
        print("O was selected!")
    elif pred_1 != np.argmax(labels):
        if pred_2 == np.argmax(labels):
            pre[k] = y_sec_pre[k]
                
print (len(testY))
print (len(pre))
print (len(y_groundtruth))
f1, re_table = MyEval.F1Score3_num(pre, y_groundtruth)

f = open("../../logs/"+run_id, 'a')  
print(re_table, file=f)
print(f1, file=f)
f.close()
        
## save model
model.save('../model/resNet/'+run_id)
