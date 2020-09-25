import os
import re
import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.utils import to_categorical

'''ckpt model dropout by keeping the best performance on both val_loss and val_acc'''
def keep_candidate(prefix, cycle):
    file_h5 = []
    for root, dirs, files in os.walk('./'):
        files = [f for f in filter(lambda f: re.match(prefix+'_[0-9]*_'+str(cycle)+'_.*\.h5$', f), files)]

        [file_h5.append(f) for f in files]

    loss_acc = []
    idx = 0
    for file in file_h5:
        strs = str.split(file, '_')
        loss = float(strs[4])
        acc = float(strs[5])

        loss_acc.append((loss,acc, idx))
        idx += 1

    sort_byloss =[x for x in sorted(loss_acc, key=lambda x: x[0])]
    sort_byacc =[x for x in sorted(loss_acc, key=lambda x: x[1])]

    candidate_idx = -1
    for idx in range(len(loss_acc)):
        if (sort_byacc[idx][2] == sort_byloss[2]):
            candidate_idx = idx
            break

    if candidate_idx < 0:
        '''remove all checkpoint models of this cycle'''
        [os.remove(f) for f in file_h5]
    else:
        for idx in range(len(file_h5)):
            if idx != candidate_idx:
                os.remove(file[idx])


def to_multilabel(label, num_classes):
    multi = to_categorical(label, num_classes)
    return np.sum(np.array(multi), axis=0)


def multilabel_score(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)

    precision = tp/(tp+fp+K.epsilon())
    recall = tp / (tp+fn+K.epsilon())
    f1 = (2*precision*recall) / (precision+recall+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    f1 = K.mean(f1)

    return f1

'''k-fold generator'''
def gen_kfold(set, k=5, shuffle=False):
    if len(set) < k:
        yield None
    else:
        if shuffle:
            np.random.shuffle(set)

        fold_size = len(set) // k

        for idx in range(k):
            val_idx = [i for i in range(idx*fold_size,idx*fold_size+fold_size)]
            train_idx = [i for i in filter(lambda x:x<idx*fold_size or x>=idx*fold_size+fold_size, range(len(set)))]

            val_set = [set[i] for i in val_idx]
            train_set = [set[i] for i in train_idx]
            yield train_set, val_set
