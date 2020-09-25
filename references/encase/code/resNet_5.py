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

class Model1(object):
    '''
    convnet MNIST
    '''
    def __init__(self):
        n_dim = 6000
        n_split = 300
        
        inputs = tflearn.input_data(shape=[None, n_dim, 1], name="input")
        net = self.make_core_network(inputs)
        net = regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name='target')
        
        model = tflearn.DNN(net, tensorboard_verbose=0)
        self.model = model

    @staticmethod
    def make_core_network(net,regularizer='L2'):
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
        return net, feature_layer

    def train(self, X, Y, testX, testY, n_epoch=1, snapshot_step=100, batch_size=200, run_id="model12"):
        # Training
        n_dim = 6000
        n_split = 300
        X = X.reshape([-1, n_dim, 1])
        testX = testX.reshape([-1, n_dim, 1])
        self.model.fit({'input': X}, {'target': Y}, n_epoch=n_epoch,
                       validation_set=({'input': testX}, {'target': testY}),
                       snapshot_step=snapshot_step, batch_size=batch_size, 
                       show_metric=True, run_id=run_id)

class Model12(object):
    '''
    Combination of two networks
    '''
    def __init__(self):
        inputs = tflearn.input_data(shape=[None, n_dim, 1], name="input")

        with tf.variable_scope("scope1") as scope:
            net_conv1, feature_layer1 = Model1.make_core_network(inputs,regularizer='L1')	# shape (?, 10)
        with tf.variable_scope("scope2") as scope:
            net_conv2, feature_layer2 = Model1.make_core_network(inputs, regularizer='nuc')	# shape (?, 10)

        network = tf.concat([feature_layer1, feature_layer2], 1, name="concat")	# shape (?, 20)
        network = tflearn.fully_connected(network, 4, activation="softmax")
        network = tflearn.regression(network, optimizer='adam', 
                             loss='categorical_crossentropy', name='target')

        self.model = tflearn.DNN(network, checkpoint_path='../../models2/resnet_bilstm',
                    max_checkpoints=10, tensorboard_verbose=0)

    def load_from_two(self, m1fn, m2fn):
        self.model.load(m1fn, scope_for_restore="scope1", weights_only=True)
        self.model.load(m2fn, scope_for_restore="scope2", weights_only=True, create_new_session=False)

    def train(self, X, Y, testX, testY, n_epoch=1, snapshot_step=100, batch_size=200, run_id="model12"):
        # Training
        self.model.fit(X, Y, n_epoch=n_epoch, validation_set=(testX, testY),
                       snapshot_step=snapshot_step, batch_size=batch_size, 
                       show_metric=True, run_id=run_id)
    
    def predict(self,testX):
        # Prediction
        testX = np.array(testX, dtype=np.float32)
        print('Predicting...')
        return self.model.predict(testX)
        
def read_data():
    with open('../../data1/expanded_three_part_window_6000_stride_500_5.pkl', 'rb') as fin:
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
run_id = 'resnet_6000_500_10_5_tltest'
m12 = Model12()
m12.load_from_two('../model/resNet/resnet_6000_500_10_5_v1', '../model/resNet/resnet_6000_500_10_5_v1')
print ("-"*60 + " Training mashup")
m12.train(X, Y, valX, valY, 1, batch_size = 64, run_id = run_id)

#Predict
cur_testX = []
y_predicted=[]
for i in range(len(testX)):
    cur_testX.append(testX[i])
    if (i % 300 == 0 or i == (len(testX) -1)) and i != 0:
        tmp_testX = np.array(cur_testX, dtype=np.float32)
        tmp_testX = tmp_testX.reshape([-1, n_dim, 1])
        y_predicted.extend(m12.predict(tmp_testX))
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
    max_v = temp_pre[np.argmax(temp_pre)]
    if max_v > 0:
        i_pred = (i_pred + temp_pre)/2
        #for k in range(len(i_pred)):
        #    i_pred[k] = (i_pred[k]+temp_pre[k])/2
    pre[list_id] = i_pred
    #print(j)
    y_groundtruth[list_id] = testY[j]
#y_predicted=[model.predict(testX[i].reshape([-1, n_dim, 1])) for i in list(range(13638))]
#Calculate F1Score
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

