import numpy as np
from keras.models import load_model
import pickle
import keras.backend as K
import tensorflow as tf
from plain_data_make import load_kfold_names, PlainPreprocessor
from plain_model import hyper_params, gen_pp_data_dir_name
import os
from model_tool import to_multilabel

import matplotlib.pyplot as plt

preprocessor = PlainPreprocessor(hyper_params)

if __name__ == '__main__':
    ## Add your codes to  classify normal and illness.

    ##  Classify the samples of the test set and write the results into answers.txt,
    ##  and each row representing a prediction of one sample.
    ##  Here we use random numbers as prediction labels as an example and
    ##  you should replace it with your own results.

    # preprocess the test data

    '''load kfold data'''
    data_dir = gen_pp_data_dir_name()
    kfold_idx, train_names = load_kfold_names()
    f1_score = []

    model = load_model('models/rematch_ckpt_plain45165_0_092_0.0994_0.1069_0.9600.h5')
    model.summary()
    for _, val_idx in kfold_idx:
        names = [train_names[idx] for idx in val_idx]
        x = []
        y = []
        for name in names:
            train_sig, pre_train_sig, pre_train_label = pickle.load(
                open(os.path.join(data_dir, name + '.dat'), 'rb'))
            x.append(np.transpose(pre_train_sig))
            y.append(pre_train_label)

        x = np.array(x)

        predicted = model.predict(x)
        predicted = np.argmax(predicted, axis=2)
        expected = y

        # for idx in range(len(predicted)):
        #     plt.plot(x[idx])
        #     plt.plot(predicted[idx])
        #     plt.plot(expected[idx])
        #     plt.show()





