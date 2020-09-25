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
import MyEval
import dill

def read_data():
    X = ReadData.read_centerwave('../../data1/centerwave_resampled.csv')
    _, _, Y = ReadData.ReadData( '../../data1/QRSinfo.csv' )
    all_feature = np.array(X)
    all_label_num = np.array(ReadData.Label2OneHot(Y))
    print('read data done')
    return all_feature, all_label_num

def get_deep_centerwave_feature(test_data):

    tf.reset_default_graph()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    _, n_dim = test_data.shape

    ############################### model
    net = tflearn.input_data(shape=[None, n_dim, 1])
    print("input", net.get_shape())
    net = tflearn.avg_pool_1d(net, kernel_size=5, strides=5)
    print("avg_pool_1d", net.get_shape())
    net = tflearn.conv_1d(net, 64, 16, 2)
    print("cov1", net.get_shape())
    net = tflearn.batch_normalization(net)
    print("bn1", net.get_shape())
    net = tflearn.activation(net, 'relu')
    print("relu1", net.get_shape())
    net = tflearn.residual_bottleneck(net, 2, 16, 64, downsample_strides = 2, downsample=True, is_first_block = True)
    print("resn2", net.get_shape())
    net = tflearn.residual_bottleneck(net, 2, 16, 128, downsample_strides = 2, downsample=True)
    print("resn4", net.get_shape())
    net = tflearn.residual_bottleneck(net, 2, 16, 256, downsample_strides = 2, downsample=True)
    print("resn6", net.get_shape())
    # net = tflearn.residual_bottleneck(net, 2, 16, 512, downsample_strides = 2, downsample=True)
    # print("resn8", net.get_shape())
    # net = tflearn.residual_bottleneck(net, 2, 16, 1024, downsample_strides = 2, downsample=True)
    # print("resn10", net.get_shape())

    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)

    feature_layer = tflearn.fully_connected(net, 32, activation='sigmoid')
    print("feature_layer", feature_layer.get_shape())
    net = feature_layer
    net = tflearn.fully_connected(net, 4, activation='softmax')
    print("dense", net.get_shape())
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy',
                             learning_rate=0.01)
    ###############################



    ### load
    model = tflearn.DNN(net)
    model.load('../model/model_deep_centerwave_0810_all/model_deep_centerwave_resnet')

    ### create new model, and get features
    m2 = tflearn.DNN(feature_layer, session=model.session)
    out_feature = []
    pred = []
    num_of_test = len(test_data)
    for i in range(num_of_test):
        tmp_test_data = test_data[i].reshape([-1, n_dim, 1])
        out_feature.append(m2.predict(tmp_test_data)[0])
        # pred.append(model.predict(tmp_test_data)[0])

    out_feature = np.array(out_feature)

    # ### eval
    # print(len(pred), pred[0], all_label[0])
    # MyEval.F1Score3_num(pred, all_label[:num_of_test])
    
    return out_feature

if __name__ == '__main__':
    all_data, all_label = read_data()
    out_feature = get_deep_centerwave_feature(all_data)
    print('out_feature shape: ', out_feature.shape)
    # with open('../data/feat_deep_centerwave_resnet.pkl', 'wb') as fout:
        # dill.dump(out_feature, fout)
