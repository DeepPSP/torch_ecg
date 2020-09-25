import glob
import random
import os
import argparse
import scipy.io as sio
from keras import backend as K
from sklearn.model_selection import train_test_split
import csv
import numpy
import numpy as np

import pandas as pd
import tensorflow as tf
import scipy
from tensorflow.python.client import device_lib
import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional, LeakyReLU
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Reshape, GRU, CuDNNGRU
from keras.layers import Convolution1D, MaxPool1D, GlobalAveragePooling1D,concatenate,AveragePooling1D
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.models import Model
from keras.utils import multi_gpu_model
from keras import initializers, regularizers, constraints
from keras.layers import Layer
import numpy as np
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import scipy.io as sio
from os import listdir
## example:
# X: input data, whose shape is (72000,12)
# Y: output data, whose shape is  = (9,)
# Y = weighted_predict_for_one_sample_only(X)

def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        print(f"AttentionWithContext input_shape = {input_shape}")
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        print(f"AttentionWithContext W.shape = {self.W.shape}")
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
            print(f"AttentionWithContext b.shape = {self.b.shape}")
            self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
            print(f"AttentionWithContext u.shape = {self.u.shape}")
        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        print(f"AttentionWithContext forward: x.shape = {x.shape}, W.shape = {self.W.shape}")
        uit = dot_product(x, self.W)
        print(f"AttentionWithContext forward: uit.shape = {uit.shape}")
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
        print(f"AttentionWithContext forward: ait.shape = {ait.shape}")
        a = K.exp(ait)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        print(f"AttentionWithContext forward: without `keepdims`, sum(a).shape = {K.sum(a, axis=1).shape}")
        print(f"AttentionWithContext forward: with `keepdims`, sum(a).shape = {K.sum(a, axis=1, keepdims=True).shape}")
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        print(f"AttentionWithContext forward: a.shape = {a.shape} before `expand_dims`")
        a = K.expand_dims(a)
        print(f"AttentionWithContext forward: a.shape = {a.shape} after `expand_dims`")
        weighted_input = x * a
        print(f"AttentionWithContext forward: weighted_input.shape = {weighted_input.shape}")
        out = K.sum(weighted_input, axis=1)
        print(f"AttentionWithContext forward: out.shape = {out.shape}")
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
batch_size = 64
num_classes = 9
epochs = 1000000000000000000000000000000000
magicVector = np.load('magicVector_test_val_strategy.npy')
leadsLabel = np.asarray(['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'])

for fold in range(10):
    for lead in range(13):
        main_input = Input(shape=(72000,12), dtype='float32', name='main_input')
        x = Convolution1D(12, 3, padding='same')(main_input)
        x = LeakyReLU(alpha=0.3)(x)
        x = Convolution1D(12, 3, padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Convolution1D(12, 24, strides = 2, padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.2)(x)
        x = Convolution1D(12, 3, padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Convolution1D(12, 3, padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Convolution1D(12, 24, strides = 2, padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.2)(x)
        x = Convolution1D(12, 3, padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Convolution1D(12, 3, padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Convolution1D(12, 24, strides = 2, padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.2)(x)
        x = Convolution1D(12, 3, padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Convolution1D(12, 3, padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Convolution1D(12, 24, strides = 2, padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.2)(x)
        x = Convolution1D(12, 3, padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Convolution1D(12, 3, padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Convolution1D(12, 48, strides = 2, padding='same')(x)
        x = LeakyReLU(alpha=0.3)(x)
        cnnout = Dropout(0.2)(x)
        x = Bidirectional(CuDNNGRU(12, input_shape=(2250,12),return_sequences=True,return_state=False))(cnnout)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.2)(x)
        x = AttentionWithContext()(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.2)(x)
        main_output = Dense(num_classes,activation='sigmoid')(x)

        vars()['model'+str(lead)+'_'+str(fold)] = Model(inputs=main_input, outputs=main_output)
        if lead == 12:
            vars()['model'+str(lead)+'_'+str(fold)].load_weights('CPSC2018_10_fold_model_'+str(fold))
        else:
            vars()['model'+str(lead)+'_'+str(fold)].load_weights('CPSC2018_10_fold_model_'+str(leadsLabel[lead])+'_'+str(fold))

def cpsc2018(record_base_path):
    def weighted_prediction_for_one_sample_only(target):
        fold_predict = np.zeros((10,9))
        for fold in range(10):
            lead_predict = np.zeros((13,9))
            for lead in range(13):
                if lead == 12:
                    lead_predict[lead,:] = globals()['model'+str(lead)+'_'+str(fold)].predict(target)[0,:].copy()
                else:
                    zeroIndices = np.asarray(list(set(range(12)) - set([lead])))
                    target_temp = target.copy()
                    target_temp[0,:,zeroIndices] = 0
                    lead_predict[lead,:] = globals()['model'+str(lead)+'_'+str(fold)].predict(target_temp)[0,:].copy()
            lead_predict = np.mean(lead_predict, axis = 0)
            fold_predict[fold,:] = lead_predict.copy()
        y_pred = np.mean(fold_predict, axis = 0)
        return y_pred * magicVector
    with open('answers.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        # column name
        writer.writerow(['Recording', 'Result'])
        count = 1
        for mat_item in os.listdir(record_base_path):
            if mat_item.endswith('.mat') and (not mat_item.startswith('._')):
                print(mat_item)
                print(count)
                X_list = []
                ecg = np.zeros((72000,12), dtype=np.float32)
                record_path=os.path.join(record_base_path, mat_item)
                ecg[-sio.loadmat(record_path)['ECG'][0][0][2].shape[1]:,:] = sio.loadmat(record_path)['ECG'][0][0][2][:, -72000: ].T

                #print(ecg.shape)
                X_list.append(ecg)
                X_list = np.asarray(X_list)
                X =X_list[:,:,:]

                Y = weighted_prediction_for_one_sample_only(X)
                result=np.argmax(Y)

                result=result+1
                if result > 9 or result < 1 or not(str(result).isdigit()):
                    result = 1
                record_name = mat_item.rstrip('.mat')
                answer = [record_name, result]
                print(answer)
                writer.writerow(answer)
                count += 1
        csvfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--recording_path',
                        help='path saving test record file')

    args = parser.parse_args()

    result = cpsc2018(record_base_path=args.recording_path)
