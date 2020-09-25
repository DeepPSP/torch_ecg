from plain_data_make import PlainPreprocessor, gen_pp_data_dir_name
from plain_model import hyper_params
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

preprocessor = PlainPreprocessor(hyper_params)
from sig_tool import  normalize

if __name__ == '__main__':
    # name = 'aha_2010_ch0'
    name = 'mitdb_100_ch1'
    # name = 'aha_1004_ch0'
    train_sig, pre_train_sig, pre_train_label = pickle.load(open(os.path.join(gen_pp_data_dir_name(), name+'.dat'), 'rb'))

    # plt.plot(pre_train_label)
    # plt.show()
    # for idx in range(len(pre_train_sig)):
    #     if np.isnan(pre_train_sig[idx]):
    #         print(idx)

    # plt.plot(train_sig)
    plt.plot(normalize(pre_train_sig))
    plt.plot(pre_train_label)
    # preprocessor.preprocess([train_sig[515500:557500]], [])

    name = 'icbeb_00260'
    train_sig, pre_train_sig, pre_train_label = pickle.load(open(os.path.join(gen_pp_data_dir_name(), name+'.dat'), 'rb'))

    plt.figure()

    plt.plot(normalize(pre_train_sig))
    plt.plot(pre_train_label)

    plt.show()
