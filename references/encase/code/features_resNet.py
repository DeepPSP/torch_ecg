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
    
    '''if tmp_label is not None and len(tmp_label) !=0:
        print(tmp_label)
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
    '''
    

    for i in range(len(tmp_data)):
        tmp_stride = 100
        #if tmp_label is not None and len(tmp_label) !=0:
        #    tmp_stride = stride[tmp_label[i]]
        tmp_ts = tmp_data[i]
        tmp_ts = list(tmp_ts)
        while len(tmp_ts) <= window_size:
            tmp_ts.extend(np.array([0.]))
        for j in range(0, len(tmp_ts)-window_size, tmp_stride):
            out_pid.append(tmp_pid[i])
            out_data.append(tmp_ts[j:j+window_size])
            if tmp_label is not None and len(tmp_label) !=0:
                out_label.append(tmp_label[i])
    
    out_label = ReadData.Label2OneHot(out_label)
    out_data = np.expand_dims(np.array(out_data, dtype=np.float32), axis=2)
    if tmp_label is not None and len(tmp_label) !=0:
        out_label = np.array(out_label, dtype=np.float32)
    else:
        out_label = np.array([])
    out_pid = np.array(out_pid, dtype=np.string_)
    
    return out_data, out_label, out_pid


def get_resnet_feature(test_data, test_label, test_pid, pid_map):
    n_dim = 6000
    n_split = 300

    tf.reset_default_graph()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    ### split
    #X = X.reshape([-1, n_split, 1])
    #testX = testX.reshape([-1, n_split, 1])

    # Building Residual Network
    net = tflearn.input_data(shape=[None, n_dim, 1])
    ############ reshape for sub_seq 
    net = tf.reshape(net, [-1, n_split, 1])
    net = tflearn.conv_1d(net, 64, 16, 2)
    #net = tflearn.conv_1d(net, 64, 16, 2, regularizer='L2', weight_decay=0.0001)
    net = tflearn.batch_normalization(net)

    # Residual blocks
    net = tflearn.residual_bottleneck(net, 2, 16, 64, downsample_strides = 2, downsample=True, is_first_block = True)
    net = tflearn.residual_bottleneck(net, 2, 16, 64, downsample_strides = 2, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 16, 128, downsample_strides = 2, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 16, 128, downsample_strides = 2, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 16, 256, downsample_strides = 2, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 16, 256, downsample_strides = 2, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 16, 512, downsample_strides = 2, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 16, 512, downsample_strides = 2, downsample=True)

    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    #net = tflearn.global_avg_pool(net)
    # LSTM
    ############ reshape for sub_seq 
    net = tf.reshape(net, [-1, n_dim//n_split, 512])
    net = bidirectional_rnn(net, BasicLSTMCell(256), BasicLSTMCell(256))
    #net = tflearn.layers.recurrent.lstm(net, n_units=512)
    #print("after LSTM", net.get_shape())
    net = dropout(net, 0.5)

    # Regression
    feature_layer = tflearn.fully_connected(net, 32, activation='sigmoid')
    net = tflearn.dropout(feature_layer, 0.5)
    net = tflearn.fully_connected(net, 4, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',#momentum',
                             loss='categorical_crossentropy')
                             #,learning_rate=0.1)
    ## save model
    ### load
    model = tflearn.DNN(net)
    run_id = 'resnet_6000_500_10_5_v1'
    model.load('../model/resNet/'+run_id)

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

    tmp_feature = np.array(tmp_feature)


    test_pid =  np.array(test_pid, dtype=np.string_)
    
    y_num = len(pid_map)
    features = [[0. for j in  range(32)] for i in  range(y_num)]
    re_labels = [[0. for j in range(4)] for i in  range(y_num)]
    y_pre = [[0. for j in range(4)] for i in  range(y_num)]
    y_sec_pre = [[0. for j in range(4)] for i in  range(y_num)]
    y_third_pre = [[0. for j in range(4)] for i in  range(y_num)]
    y_fourth_pre = [[0. for j in range(4)] for i in  range(y_num)]
    y_fifth_pre = [[0. for j in range(4)] for i in  range(y_num)]
    y_sixth_pre = [[0. for j in range(4)] for i in  range(y_num)]
    y_seventh_pre = [[0. for j in range(4)] for i in  range(y_num)]
    y_groundtruth= [[0. for j in range(4)] for i in range(y_num)]
    #print(y_num)
    
    for j in range(len(tmp_feature)):
        feature_pred = np.array(tmp_feature[j], dtype=np.float32)
        #print(len(feature_pred))
        i_pred = np.array(pre[j], dtype=np.float32)
        cur_pid = str(test_pid[j],'utf-8')
        
        list_id = pid_map[cur_pid]
        #print (list_id)
        temp_feature = np.array(features[list_id], dtype=np.float32)
        temp_pre = np.array(y_pre[list_id], dtype=np.float32)
        temp_sec_pre = np.array(y_sec_pre[list_id], dtype=np.float32)
        temp_third_pre = np.array(y_third_pre[list_id], dtype=np.float32)
        #print(temp_pre)
        
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
            y_seventh_pre[list_id] = y_sixth_pre[list_id]
            y_sixth_pre[list_id] = y_fifth_pre[list_id]
            y_fifth_pre[list_id] = y_fourth_pre[list_id]
            y_fourth_pre[list_id] = y_third_pre[list_id]
            y_third_pre[list_id] = y_sec_pre[list_id]
            y_sec_pre[list_id] = y_pre[list_id]
            y_pre[list_id] = i_pred
        elif (cur_max_p - cur_sec_p) > (max_sec_p - sec_sec_p):
            y_seventh_pre[list_id] = y_sixth_pre[list_id]
            y_sixth_pre[list_id] = y_fifth_pre[list_id]
            y_fifth_pre[list_id] = y_fourth_pre[list_id]
            y_fourth_pre[list_id] = y_third_pre[list_id]
            y_third_pre[list_id] = y_sec_pre[list_id]
            y_sec_pre[list_id] = i_pred
        elif (cur_max_p - cur_sec_p) > (max_third_p - sec_third_p):
            y_seventh_pre[list_id] = y_sixth_pre[list_id]
            y_sixth_pre[list_id] = y_fifth_pre[list_id]
            y_fifth_pre[list_id] = y_fourth_pre[list_id]
            y_fourth_pre[list_id] = y_third_pre[list_id]
            y_third_pre[list_id] = i_pred
        elif (cur_max_p - cur_sec_p) > (max_fourth_p - sec_fourth_p):
            y_seventh_pre[list_id] = y_sixth_pre[list_id]
            y_sixth_pre[list_id] = y_fifth_pre[list_id]
            y_fifth_pre[list_id] = y_fourth_pre[list_id]
            y_fourth_pre[list_id] = i_pred
        elif (cur_max_p - cur_sec_p) > (max_fifth_p - sec_fifth_p):
            y_seventh_pre[list_id] = y_sixth_pre[list_id]
            y_sixth_pre[list_id] = y_fifth_pre[list_id]
            y_fifth_pre[list_id] = i_pred
        elif (cur_max_p - cur_sec_p) > (max_sixth_p - sec_sixth_p):
            y_seventh_pre[list_id] = y_sixth_pre[list_id]
            y_sixth_pre[list_id] = i_pred
        elif (cur_max_p - cur_sec_p) > (max_seventh_p - sec_seventh_p):
            y_seventh_pre[list_id] = i_pred
            
        max_f = 0
        for k in range(len(temp_feature)):
            if temp_feature[k] > max_f:
                max_f = temp_feature[k]
        if max_f > 0:
            feature_pred = (feature_pred+temp_feature)/2
            #for k in range(len(temp_feature)):
            #    feature_pred[k] = (feature_pred[k]+temp_feature[k])/2
            
        features[list_id] = feature_pred
        
        
        y_groundtruth[list_id] = test_label[j]
        
        gt_list = ["N", "A", "O", "~"]
        pred_1 = gt_list[np.argmax(i_pred)]

        if pred_1 == 'N':
            re_labels[list_id][0] += 1
        elif pred_1 == 'A':
            re_labels[list_id][1] += 1
        elif pred_1 == 'O':
            re_labels[list_id][2] += 1
        elif pred_1 == '~':
            re_labels[list_id][3] += 1
        else:
            print('wrong label')
    
    
    
    out_feature = []
    for i in range(len(features)):
        out_feature.append(features[i])
        
    out_feature = np.array(out_feature)
    
    
    for k in range(len(y_pre)):
        labels = [0. for j in range(4)]
        pred_1 = np.argmax(y_pre[k])
        labels[pred_1] +=1
        pred_2 = np.argmax(y_sec_pre[k])
        labels[pred_2] +=1
        pred_3 = np.argmax(y_third_pre[k])
        labels[pred_3] +=1
        
        if pred_1 == 2:
            print("O was selected!")
            continue
        elif pred_2 == 2:
            y_pre[k] = y_sec_pre[k]
            print("O was selected!")
        elif pred_3 == 2:
            y_pre[k] = y_third_pre[k]
            print("O was selected!")
        if pred_1 != np.argmax(labels):
            if pred_2 == np.argmax(labels):
                y_pre[k] = y_sec_pre[k]
                print("Second was selected!")
    MyEval.F1Score3_num(pre, test_label[:len(pre)])

    MyEval.F1Score3_num(y_pre, y_groundtruth)
    MyEval.F1Score3_num(re_labels, y_groundtruth)
    
    return out_feature

def get_resNet_proba(long_data, long_pid, model_path):
    all_pid = np.array(long_pid)
    all_feature = np.array(long_data)
    all_label = np.array([])
    test_data, test_label, test_pid = slide_and_cut(all_feature, all_label, all_pid)
    
    pid_map = {}
    for i in range(len(all_pid)):
        pid_map[all_pid[i]] = i
        
    n_dim = 6000
    n_split = 300

    tf.reset_default_graph()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    # Building Residual Network
    net = tflearn.input_data(shape=[None, n_dim, 1])
    ############ reshape for sub_seq 
    net = tf.reshape(net, [-1, n_split, 1])
    net = tflearn.conv_1d(net, 64, 16, 2)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')

    # Residual blocks
    net = tflearn.residual_bottleneck(net, 2, 16, 64, downsample_strides = 2, downsample=True, is_first_block = True)
    net = tflearn.residual_bottleneck(net, 2, 16, 64, downsample_strides = 2, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 16, 128, downsample_strides = 2, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 16, 128, downsample_strides = 2, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 16, 256, downsample_strides = 2, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 16, 256, downsample_strides = 2, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 16, 512, downsample_strides = 2, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 16, 512, downsample_strides = 2, downsample=True)

    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    #net = tflearn.global_avg_pool(net)
    # LSTM
    ############ reshape for sub_seq 
    net = tf.reshape(net, [-1, n_dim//n_split, 512])
    net = bidirectional_rnn(net, BasicLSTMCell(256), BasicLSTMCell(256))
    net = dropout(net, 0.5)

    # Regression
    feature_layer = tflearn.fully_connected(net, 32, activation='sigmoid')
    net = tflearn.dropout(feature_layer, 0.5)
    net = tflearn.fully_connected(net, 4, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',#momentum',
                             loss='categorical_crossentropy')
                             #,learning_rate=0.1)
    ## save model
    ### load
    model = tflearn.DNN(net)
    model.load(model_path)

    ### create new model, and get features
    num_of_test = len(test_data)
    cur_data = []
    pre = []
    for i in range(num_of_test):
        cur_data.append(test_data[i])
        if (num_of_test >1 and (i % 2000 == 0 or i == (num_of_test - 1)) and i !=0) or (num_of_test ==1):
            tmp_testX = np.array(cur_data, dtype=np.float32)
            cur_data = []
            pre.extend(model.predict(tmp_testX))



    test_pid =  np.array(test_pid, dtype=np.string_)
    
    y_num = len(pid_map)
    y_pre = [[0. for j in range(4)] for i in  range(y_num)]
    y_sec_pre = [[0. for j in range(4)] for i in  range(y_num)]
    y_third_pre = [[0. for j in range(4)] for i in  range(y_num)]
    #print(y_num)
    
    for j in range(len(pre)):
        i_pred = np.array(pre[j], dtype=np.float32)
        cur_pid = str(test_pid[j],'utf-8')
        
        list_id = pid_map[cur_pid]
        temp_pre = np.array(y_pre[list_id], dtype=np.float32)
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
            y_sec_pre[list_id] = y_pre[list_id]
            y_pre[list_id] = i_pred
        elif (cur_max_p - cur_sec_p) > (max_sec_p - sec_sec_p):
            y_third_pre[list_id] = y_sec_pre[list_id]
            y_sec_pre[list_id] = i_pred
        elif (cur_max_p - cur_sec_p) > (max_third_p - sec_third_p):
            y_third_pre[list_id] = i_pred
            
        
    
    
    for k in range(len(y_pre)):
        labels = [0. for j in range(4)]
        pred_1 = np.argmax(y_pre[k])
        labels[pred_1] +=1
        pred_2 = np.argmax(y_sec_pre[k])
        labels[pred_2] +=1
        pred_3 = np.argmax(y_third_pre[k])
        labels[pred_3] +=1
        
        if pred_1 == 2:# and (abs(y_pre[k][np.argmax(labels)] - y_pre[k][2])/y_pre[k][np.argmax(labels)] <= 0.2):
            continue
        elif pred_2 == 2:# and (abs(y_pre[k][np.argmax(labels)] - y_sec_pre[k][2])/y_pre[k][np.argmax(labels)] <= 0.2):
            y_pre[k] = y_sec_pre[k]
        elif pred_3 == 2:# and (abs(y_pre[k][np.argmax(labels)] - y_third_pre[k][2])/y_pre[k][np.argmax(labels)] <= 0.2):
            y_pre[k] = y_third_pre[k]
        elif pred_1 != np.argmax(labels):
            if pred_2 == np.argmax(labels):
                y_pre[k] = y_sec_pre[k]
    
    return y_pre




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
    train_data, train_label, val_data, val_label, test_data, test_label, test_pid = read_data_from_pkl()
    long_pid, long_data, long_label = ReadData.ReadData( '../../data1/long.csv' )
    new_test = []
    new_pid = []
    new_label = []
    for j in range(len(long_pid)):
        for i in range(len(test_pid)):
            if long_pid[j] == test_pid[i]:
                new_test.append(long_data[j])
                new_pid.append(long_pid[j])
                new_label.append(long_label[j])
    
    out_label = ReadData.Label2OneHot(new_label)
    out_label = np.array(out_label, dtype=np.float32)
    
    new_test = np.array(new_test)
    new_pid = np.array(new_pid)
    test_data, test_label, test_pid = slide_and_cut(new_test, np.array(test_label), new_pid)
    
    pid_map = {}
    pid_set = set(new_pid)
    pids = list(pid_set)
    for i in range(len(pids)):
        cur_pid = str(pids[i],'utf-8')
        pid_map[cur_pid] = i
    
    
    get_resnet_feature(test_data, test_label, test_pid, pid_map)


