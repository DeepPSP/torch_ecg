# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 15:31:23 2019

@author: qls
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 20:35:53 2019

@author: qls
"""
import pandas as pd
import os
import pandas
import keras.backend as K
import tensorflow as tf
import random
import numpy as np
import scipy.io
import keras
from keras.layers import SimpleRNN,Permute,Reshape,CuDNNLSTM,LSTM,Dropout,Input, Add, Dense,\
 Activation,ZeroPadding1D, BatchNormalization, Flatten, Conv1D, Conv2D,AveragePooling1D,MaxPooling1D,MaxPooling2D,GlobalMaxPooling1D\
 ,AveragePooling1D,UpSampling1D,concatenate,Permute,SeparableConv1D,LeakyReLU,Conv2DTranspose
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau
from scipy import signal
import pywt

def wavelet_data(data):#基础方法
    w='db5'
    a = data
    ca = []#近似分量
    cd = []#细节分量
    mode = pywt.Modes.smooth
    for i in range(7):
        (a, d) = pywt.dwt(a, w,mode)#进行7阶离散小波变换
        ca.append(a)
        cd.append(d)
    rec   = [] 
    rec_a = []
    rec_d = []
    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))#重构
    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))
    
    rec_a=np.array(rec_a)
    rec_d=np.array(rec_d)
    rec=np.concatenate((rec_a,rec_d))
    rec=rec.T
    return rec

def wavelet_all_data(data):#基础方法
    rec_all=[]
    for i in range(len(data)):
            rec=wavelet_data(data[i])
            rec_all.append(rec)
    rec_all=np.array(rec_all)
    return rec_all



# alldata=scipy.io.loadmat('/home/lirongwang/CWQ/XG/jiangnanSTYLE/alldata_meiquzao.mat')
# alllabel=scipy.io.loadmat('/home/lirongwang/CWQ/XG/jiangnanSTYLE/alllabel_new.mat')

# new_data3=scipy.io.loadmat('/home/lirongwang/CWQ/XG/jiangnanSTYLE/alldata_7_9.mat')
# new_label3=scipy.io.loadmat('/home/lirongwang/CWQ/XG/jiangnanSTYLE/alllabel_7_9.mat')


# AFdata=scipy.io.loadmat('/home/lirongwang/CWQ/XG/jiangnanSTYLE/AFdata.mat')#D:/心梗/2019生理参数竞赛/自采ECG信号/房颤数据/合并/
# AFlabel=scipy.io.loadmat('/home/lirongwang/CWQ/XG/jiangnanSTYLE/AFlabel.mat')

# tianchi_data=scipy.io.loadmat('/home/lirongwang/CWQ/XG/jiangnanSTYLE/tianchi_data.mat')#D:/心梗/2019生理参数竞赛/自采ECG信号/房颤数据/合并/
# tianchi_label=scipy.io.loadmat('/home/lirongwang/CWQ/XG/jiangnanSTYLE/tianchi_label.mat')



# alldata=alldata['alldata_meiquzao']
# alllabel=alllabel['alllabel_new']

# new_data3=new_data3['alldata_7_9'][2000:]
# new_label3=new_label3['alllabel_7_9'][2000:]

# AFdata=AFdata['AFdata']
# AFlabel=AFlabel['AFlabel']

# tianchi_data=tianchi_data['tianchi_data']
# tianchi_label=tianchi_label['tianchi_label']

# random.seed(1)
# length=len(alldata)
# #length=2000
# c = range(0,length)
# q = random.sample(c,int(length*0.8))

# c=set(c)
# q=set(q)
# e=c-q
# e=list(e)
# q=list(q)

# train_x=alldata[q]
# train_y=alllabel[q]

# test_x=alldata[e]
# test_y=alllabel[e]


# train_x=np.concatenate((train_x,new_data3,AFdata,tianchi_data))
# train_y=np.concatenate((train_y,new_label3,AFlabel,tianchi_label))

# train_x=wavelet_all_data(train_x)
# test_x=wavelet_all_data(test_x)

# train_x=train_x.reshape(train_x.shape[0],train_x.shape[1],14)
# test_x=test_x.reshape(test_x.shape[0],test_x.shape[1],14)

# train_y=train_y.reshape(train_y.shape[0],train_y.shape[1],1)
# test_y=test_y.reshape(test_y.shape[0],test_y.shape[1],1)

inputs = Input((5000,14),name='inputs')

conv1 = BatchNormalization()(inputs)
conv1 = Conv1D(16, 20, padding='same',kernel_initializer='he_normal')(conv1)
conv1 = BatchNormalization()(conv1)
conv1=Activation('relu')(conv1)
conv1 = Conv1D(16, 20, padding='same',kernel_initializer='he_normal')(conv1)
conv1 = BatchNormalization()(conv1)
conv1=Activation('relu')(conv1)
conv1 = Dropout(0.15)(conv1)
conv1 = Conv1D(16, 20, padding='same', kernel_initializer='he_normal')(conv1)
conv1 = BatchNormalization()(conv1)
conv1=Activation('relu')(conv1)
pool1 = MaxPooling1D(pool_size=10)(conv1)

conv2 = Conv1D(24, 10, padding='same', kernel_initializer='he_normal')(pool1)
conv2 = BatchNormalization()(conv2)
conv2=Activation('relu')(conv2)
conv2 = Conv1D(24, 10,padding='same', kernel_initializer='he_normal')(conv2)
conv2 = BatchNormalization()(conv2)
conv2=Activation('relu')(conv2)
conv2 = Dropout(0.15)(conv2)
conv2 = Conv1D(24, 10,padding='same', kernel_initializer='he_normal')(conv2)
conv2 = BatchNormalization()(conv2)
conv2=Activation('relu')(conv2)
pool2 = MaxPooling1D(pool_size=5)(conv2)

conv3 = Conv1D(48, 5, padding='same', kernel_initializer='he_normal')(pool2)
conv3 = BatchNormalization()(conv3)
conv3 =Activation('relu')(conv3)
conv3 = Conv1D(48, 5, padding='same', kernel_initializer='he_normal')(conv3)
conv3 = BatchNormalization()(conv3)
conv3=Activation('relu')(conv3)
conv3 = Dropout(0.15)(conv3)
conv3 = Conv1D(48, 5, padding='same', kernel_initializer='he_normal')(conv3)
conv3 = BatchNormalization()(conv3)
conv3=Activation('relu')(conv3)
pool3 = MaxPooling1D(pool_size=2)(conv3)


conv4_1 = Conv1D(96, 5, padding='same', kernel_initializer='he_normal')(pool3)
conv4_1 = BatchNormalization()(conv4_1)
conv4_1=Activation('relu')(conv4_1)
conv4_1 = Dropout(0.15)(conv4_1)
conv4_1 = Conv1D(96, 5, padding='same', kernel_initializer='he_normal')(conv4_1)
conv4_1 = BatchNormalization()(conv4_1)
conv4_1=Activation('relu')(conv4_1)
conv4_2 = Conv1D(96, 5, padding='same',dilation_rate=10, kernel_initializer='he_normal')(pool3)
conv4_2 = BatchNormalization()(conv4_2)
conv4_2=Activation('relu')(conv4_2)
conv4_2 = Dropout(0.15)(conv4_2)
conv4_2 = Conv1D(96, 5, padding='same',dilation_rate=10, kernel_initializer='he_normal')(conv4_2)
conv4_2 = BatchNormalization()(conv4_2)
conv4_2=Activation('relu')(conv4_2)
conv4_3 = keras.layers.Subtract()([conv4_1, conv4_2])


conv4=concatenate([conv4_1,conv4_2,conv4_3], axis=-1)


temp1=UpSampling1D(size=2)(conv4)
merge1 = concatenate([temp1, conv3], axis=-1)
conv5 = Conv1D(48, 5, padding='same', kernel_initializer='he_normal')(merge1)
conv5 = BatchNormalization()(conv5)
conv5=Activation('relu')(conv5)
conv5 = Dropout(0.15)(conv5)
conv5 = Conv1D(48, 5,padding='same', kernel_initializer='he_normal')(conv5) 
conv5 = BatchNormalization()(conv5)
conv5=Activation('relu')(conv5)
conv5 = Dropout(0.15)(conv5)
conv5 = Conv1D(48, 5,padding='same', kernel_initializer='he_normal')(conv5)
conv5 = BatchNormalization()(conv5)
conv5=Activation('relu')(conv5)


temp2=UpSampling1D(size=5)(conv5)
merge2 = concatenate([temp2, conv2], axis=-1)
conv6 = Conv1D(24, 10, padding='same', kernel_initializer = 'he_normal')(merge2)
conv6 = BatchNormalization()(conv6)
conv6=Activation('relu')(conv6)
conv6 = Dropout(0.15)(conv6)
conv6 = Conv1D(24, 10, padding='same', kernel_initializer = 'he_normal')(conv6)
conv6 = BatchNormalization()(conv6)
conv6=Activation('relu')(conv6)
conv6 = Dropout(0.15)(conv6)
conv6 = Conv1D(24, 10, padding='same', kernel_initializer = 'he_normal')(conv6)
conv6 = BatchNormalization()(conv6)
conv6=Activation('relu')(conv6)


temp3=UpSampling1D(size=10)(conv6)
merge3 = concatenate([temp3, conv1], axis=-1)
conv7 = Conv1D(16, 20,padding='same', kernel_initializer='he_normal')(merge3)
conv7 = BatchNormalization()(conv7)
conv7=Activation('relu')(conv7)
conv7 = Dropout(0.15)(conv7)
conv7 = Conv1D(16, 20, padding='same', kernel_initializer='he_normal')(conv7)
conv7 = BatchNormalization()(conv7)
conv7=Activation('relu')(conv7)
conv7 = Dropout(0.15)(conv7)
conv7 = Conv1D(16, 20, padding='same', kernel_initializer='he_normal')(conv7)
conv7 = BatchNormalization()(conv7)
conv7=Activation('relu')(conv7)

conv8 = Conv1D(1, 1,padding='same', kernel_initializer='he_normal')(conv7,)
conv8 = Reshape((5000, 1))(conv8)
conv9 = Activation('sigmoid',name='output')(conv8)

model = Model(inputs=inputs, outputs=conv9)

checkpoint= keras.callbacks.ModelCheckpoint('jiangnastyle_10_7.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

adam=keras.optimizers.Adam(lr=0.001,beta_1=0.9, beta_2=0.999,epsilon=1e-08,clipvalue=0.5)

model.compile(loss='binary_crossentropy', optimizer=adam,metrics=['accuracy'])

# model.fit({'inputs':train_x}, {'output':train_y}, validation_data=(test_x,test_y),epochs=10000,batch_size=256,callbacks=[checkpoint])
