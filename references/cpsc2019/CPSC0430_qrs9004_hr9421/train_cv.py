import matplotlib
import matplotlib.pyplot as plt

#import wfdb 
import os
import numpy as np
import math
import sys
import scipy.stats as st
import scipy.io as sio
import scipy
from scipy import signal
import glob, os
from os.path import basename
import yaml
import json
import tensorflow as tf
from keras.layers import Dense,Activation,Dropout
from keras.layers import LSTM,Bidirectional,CuDNNLSTM #could try TimeDistributed(Dense(...))
from keras.models import Sequential, load_model
from keras import optimizers,regularizers
from keras.layers.normalization import BatchNormalization
import keras.backend.tensorflow_backend as KTF
from tqdm import tqdm
from keras.utils import np_utils
from scipy import ndimage
import pandas as pd

from net import build_model

from keras.layers import Input
from keras.models import Model

import warnings
warnings.filterwarnings("ignore")

np.random.seed(2019)

data_path_base='train\\data\\'
ref_path_base='train\\ref\\'

recordname=os.listdir(data_path_base)
recordname.sort()

fs=128
ms_110 = int(fs*0.08)


len_ecg=fs*10

data_path_tr = 'split\\train\\data\\'
x_tr_ind = os.listdir(data_path_tr)

data_path_test = 'split\\test\\data\\'
x_val_ind = os.listdir(data_path_test)

from scipy import ndimage, misc
def med_filt_1d(ecg):
    first_filtered = ndimage.median_filter(ecg, size=7)
    second_filtered =  ndimage.median_filter(first_filtered, 215)
    ecg_deno = ecg - second_filtered
    return ecg_deno

import pywt
def WTfilt_1d(ecg):
    data = []
    for i in range(len(ecg)-1):
        data.append(float(ecg[i]))

    w = pywt.Wavelet('db4')        # 选用Daubechies4小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    threshold = 0.1  # Threshold for filtering
    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(data, 'db4', level=maxlev)  # 将信号进行小波分解

    coeffs[6] = pywt.threshold(coeffs[6], threshold*max(coeffs[6]))  # 将噪声滤波

    for i in range(7, len(coeffs)):
        coeffs[i] = np.zeros(len(coeffs[i]))
    datarec = pywt.waverec(coeffs, 'db4')  # 将信号进行小波重构
    mintime = 0
    maxtime = mintime + len(data) + 1

    return datarec.reshape(fs*10,1)

from keras import backend as K
from keras.optimizers import Adam
from keras.legacy import interfaces

class RAdam(Adam):
    """RAdam optimizer, also named Recifited Adam optimizer.
    Arguments
    ---------
        lr: float >= 0. Learning rate, default 0.001.
        beta_1: float, (0, 1). Generally close to 1.
        beta_2: float, (0, 1). Generally close to 1.
        epsilon: float >= 0. Fuzz factor, a negligible value (
            e.g. 1e-8), defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
    References
    ----------
      - [On the Variance of the Adaptive Learing Rate and Beyond](
         https://arxiv.org/abs/1908.03265)
    """

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay:
            lr = lr * (1. / (1. + self.decay * K.cast(
                self.iterations, K.dtype(self.decay)
            )))

        t = K.cast(self.iterations, K.floatx()) + 1.
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        beta_1_t = K.pow(beta_1, t)
        beta_2_t = K.pow(beta_2, t)
        rho_inf = 2. / (1. - beta_2) - 1.
        rho_t = rho_inf - 2. * t * beta_2_t / (1. - beta_2_t)
        r_t = K.sqrt(
            K.relu(rho_t - 4.) * (rho_t - 2.) * rho_inf / (
                (rho_inf - 4.) * (rho_inf - 2.) * rho_t )
        )
        flag = K.cast(rho_t > 4., K.floatx())

        ms = [K.zeros(K.int_shape(p)) for p in params]
        vs = [K.zeros(K.int_shape(p)) for p in params]

        self.weights = [self.iterations] + ms + vs
        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = beta_1 * m + (1. - beta_1) * g
            v_t = beta_2 * v + (1. - beta_2) * K.square(g)

            m_hat_t = m_t / (1. - beta_1_t)
            v_hat_t = K.sqrt(v_t / (1. - beta_2_t))
            new_p = p - lr * (r_t / (v_hat_t + self.epsilon) + flag - 1.)* m_hat_t

            if getattr(p, "constraint", None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
        return self.updates

import keras
from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping,CSVLogger
import time
from sklearn.model_selection import StratifiedKFold
import keras.backend.tensorflow_backend as KTF


def train_net():
	os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 程序最多只能占用指定gpu50%的显存
	config.gpu_options.allow_growth = True      #程序按需申请内存
	sess = tf.Session(config = config)
	# 设置session
	KTF.set_session(sess)

	output_length = fs*10
	input_length = fs*10
	n_classes = 2

	input_length = 1280

	for fold in range(1,6,1):
	    print("*******************fold {}*****************".format(fold))
	    tr_data_path = "./validation/train/validation_{}_data.npy".format(fold)#validation
	    tr_label_path = "./validation/train/validation_{}_target.npy".format(fold)
	    
	    val_data_path = "./validation/val/validation_{}_data.npy".format(fold)
	    val_label_path = "./validation/val/validation_{}_target.npy".format(fold) #cv
	    
	    X_tr = np.load(tr_data_path)
	    y_tr = np.load(tr_label_path).reshape(-1,1280,1)
	    
	    X_val = np.load(val_data_path)
	    y_val = np.load(val_label_path).reshape(-1,1280,1)
	    
	    # callback_lists
	    checkpoint = ModelCheckpoint(filepath="./model/"+ \
	                                 'model_weights_mse_0.08_1280_{}_fold_{}_ys.h5' \
	                                 .format(time.strftime('%Y-%m-%d %X').split(" ")[0],fold),
	                                 monitor= 'val_loss', 
	                                 mode='min',
	                                 verbose=1,
	                                 save_best_only='True',
	                                 save_weights_only='True')

	    csv_logger = CSVLogger(
	                filename='./log/log{}_fold{}.csv'.format( \
	                time.strftime('%Y-%m-%d %X').split(" ")[0],fold),
	                separator=',',
	                append=True)

	    earlystop = EarlyStopping(
	                monitor='val_loss',
	                min_delta=0,
	                patience=5,
	                verbose=1,
	                mode="min",
	                #baseline=None,
	                #restore_best_weights=True,
	                )

	    reducelr = ReduceLROnPlateau(
	                monitor='val_loss', 
	                factor=0.5,
	                patience=3, 
	                verbose=1,
	                min_lr=1e-5)

	    callback_lists = [earlystop, checkpoint, reducelr]
	    
	    input_layer = Input((input_length, 1))
	    output_layer = build_model(input_layer=input_layer,block="resnext",start_neurons=16,DropoutRatio=0.5,filter_size=32,nClasses=2)
	    model = Model(input_layer, output_layer)
	    #print(model.summary())
	    model.compile(loss='mse',#'categorical_crossentropy',
	                  optimizer=RAdam(1e-4),#RAdam(lr_schedule(0)),#
	                  metrics=['accuracy','mse','mae'])#focal_loss #mape

	    history = model.fit(X_tr,
	              y_tr,
	              epochs=400, #350 train
	              batch_size=32, 
	              verbose=2,
	              validation_data=(X_val, y_val), 
	              callbacks = callback_lists
	             )   #class_weight=class_weight


if __name__ == '__main__':

	train_net()