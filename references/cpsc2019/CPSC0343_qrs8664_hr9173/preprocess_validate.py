
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from plain_data_make import PlainPreprocessor, load_kfold_names, preload
from plain_model import PlainModel, hyper_params, gen_ckpt_prefix, gen_pp_data_dir_name

preprocessor = PlainPreprocessor(hyper_params)

if __name__ == '__main__':
    data_dir = gen_pp_data_dir_name()
    train_paired, val_paired = pickle.load(open('shuffle_names.dat', 'rb'))


    for name, offset in val_paired:
        train_sig, train_label, pre_train_sig, pre_train_label = pickle.load(open(os.path.join(data_dir, name+'.dat'), 'rb'))

        sig = preprocessor.single_preprocess(train_sig)


        plt.figure(figsize=(16,16))
        plt.plot(pre_train_sig[:,1])
        plt.plot(sig)
        plt.legend(['pre_sig', 'single_sig'])
        plt.show()


