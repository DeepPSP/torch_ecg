#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 08:01:35 2017

@author: shenda

class order: ['A', 'N', 'O', '~']
"""

from matplotlib import pyplot as plt

    
if __name__ == "__main__":

    nn = tf.contrib.learn.Estimator(model_dir="../../tmp6")
    print(nn.get_variable_names())
    
    tmp = nn.get_variable_value('dense/bias')
    print(tmp)    
    tmp = nn.get_variable_value('dense/kernel')
    print(tmp)
#    
#    conv1d_kernel = nn.get_variable_value('conv1d/kernel')
#    conv1d_bias = nn.get_variable_value('conv1d/bias')
#    
#    conv1d_1_kernel = nn.get_variable_value('conv1d_1/kernel')
#    conv1d_1_bias = nn.get_variable_value('conv1d_1/bias')
#    
#    dense_kernel = nn.get_variable_value('dense/kernel')
#    dense_bias = nn.get_variable_value('dense/bias')
#    
#    dense_1_kernel = nn.get_variable_value('dense_1/kernel')
#    dense_1_bias = nn.get_variable_value('dense_1/bias')
#
#    for i in range(16):
#        plt.subplot(16,1,i+1)
#        plt.plot(conv1d_kernel[:,:,i])

