import os
import numpy as np
import h5py as h5

from array_tool import paired_shuffle, kfold
from model import Preprocessor
from wfdb_tool import load_mitdb, load_aha
from plain_model import gen_pp_data_dir_name, hyper_params, gen_fold_name, gen_preload_name
import pickle

from icbeb_tool import load_icbeb2019
from sig_tool import med_filter, fir, normalize
import re
import matplotlib.pyplot as plt
from model import _filtering


class PlainPreprocessor(Preprocessor):

    def make_stag(self, sigs):
        stag_len = hyper_params['stag_len']
        stag_step = hyper_params['stag_step']
        stag_sigs = []
        for sig in sigs:
            stag_sigs.append([sig[idx*stag_step:idx*stag_step+stag_len, :] for idx in range((len(sig)-stag_len)//stag_step)])
        return stag_sigs

    def label_augment(self, label, span, offset=0):
        label_offset = np.zeros(np.shape(label))
        for idx in range(len(label)-offset):
            label_offset[idx+offset] = label[idx]
        label = label_offset
        label_ex = np.zeros(np.shape(label))
        idx = 0
        while idx < len(label):
            if label[idx] == 1:
                for idx2 in range(idx - span, idx, 1):
                    if 0 <= idx2 <= len(label) - 1:
                        label_ex[idx2] = 1

                for idx2 in range(idx, idx + span, 1):
                    if 0 <= idx2 <= len(label) - 1:
                        label_ex[idx2] = 1

            idx += 1

        return label_ex

    def label_ext_bisect(self, label, offset=0):
        label_ex = np.zeros(np.shape(label))
        '''by assumption, label sequence with first-half of heart beat'''
        idx_r_prev = -1
        idx_r_next = -1
        for idx in range(len(label)):
            if label[idx] == 1:
                if idx_r_prev < 0:
                    idx_r_prev = idx
                else:
                    idx_r_next = idx

                    idx_r_mid = (idx_r_next + idx_r_prev) // 2

                    for idx_2 in range(idx_r_prev, idx_r_mid):
                        label_ex[idx_2] = 1

                    idx_r_prev = idx_r_next

        return label_ex


    '''common filtering before processing'''
    def single_preprocess(self, x):
        x = _filtering(x, self.hyper_params)
        return x


    def preprocess(self, x, y):
        x, y = super().preprocess(x, y)
        print('finish filtering')
        sigs = x

        y = [self.label_augment(lb, span=40, offset=0) for lb in y]
        labels = np.array(y)

        return sigs, labels


def preload():
    pp_data_dir_name = gen_pp_data_dir_name()
    preload_name = gen_preload_name()

    train_names = []
    offsets = []
    if os.path.exists(os.path.join(pp_data_dir_name, preload_name)):
        train_names, offsets = pickle.load(open(os.path.join(pp_data_dir_name, preload_name), 'rb'))
        return train_names, offsets
    else:
        names = pickle.load(open(os.path.join(pp_data_dir_name, 'file_names.dat'), 'rb'))
        offset = hyper_params['preload_offset']
        for name in names:
            train_sigs, trainl_labels, pre_train_sigs, pre_train_labels = pickle.load(open(os.path.join(pp_data_dir_name, name + '.dat'), 'rb'))
            len_sig = len(pre_train_sigs)
            idx = 0
            while idx+hyper_params['crop_len'] <= len_sig:
                sig = pre_train_sigs[idx:idx+hyper_params['crop_len']]
                if np.isnan(sig).any():
                    pass
                else:
                    train_names.append(name)
                    offsets.append(idx)
                idx += offset

        pickle.dump((train_names, offsets), open(os.path.join(pp_data_dir_name, preload_name), 'wb'))

        return train_names, offsets



def load_kfold_names():
    pp_data_dir_name = gen_pp_data_dir_name()
    fold_name = gen_fold_name()
    if os.path.exists(os.path.join(pp_data_dir_name, fold_name)):
        fold_idx, train_names = pickle.load(open(os.path.join(pp_data_dir_name, fold_name), 'rb'))
    else:
        train_names = pickle.load(open(os.path.join(pp_data_dir_name, 'file_names.dat'), 'rb'))
        '''shuffling'''
        np.random.shuffle(train_names)
        '''k-fold generate'''
        fold_idx = [(train, val) for (train, val) in kfold(len(train_names), n_split=hyper_params['kfold'], shuffle=False)]
        pickle.dump((fold_idx, train_names), open(os.path.join(pp_data_dir_name, fold_name), 'wb'))

    return fold_idx, train_names

if __name__ == '__main__':
    '''load data and labels'''
    pp_data_dir_name = gen_pp_data_dir_name()
    if not os.path.isdir(pp_data_dir_name):

        ib_names, ib_train_sigs, ib_train_labels = load_icbeb2019(db_dir='dat\\icbeb2019')
        mdb_name_list, mdb_data_list, mdb_anno_list, mdb_anno_typ_list = load_mitdb(db_dir='dat\\wfdb')
        mdb_data_list = np.array(mdb_data_list)
        # adb_name_list, adb_data_list, adb_anno_list, adb_anno_typ_list = load_aha(db_dir='dat/wfdb')
        # adb_data_list = np.array(adb_data_list)

        '''merge data'''
        names = []
        train_sigs = []
        train_labels = []

        for idx in range(len(ib_names)):
            names.append('icbeb_'+ib_names[idx])
            train_sigs.append(ib_train_sigs[idx])
            train_labels.append(ib_train_labels[idx])
        for idx in range(len(mdb_name_list)):
            names.append('mitdb_'+mdb_name_list[idx]+'_ch0')
            train_sigs.append(mdb_data_list[idx][:,0])
            train_labels.append(mdb_anno_list[idx])

            names.append('mitdb_'+mdb_name_list[idx]+'_ch1')
            train_sigs.append(mdb_data_list[idx][:,1])
            train_labels.append(mdb_anno_list[idx])
        #
        # for idx in range(len(adb_name_list)):
        #     names.append('aha_'+adb_name_list[idx]+'_ch0')
        #     train_sigs.append(adb_data_list[idx][:,0])
        #     train_labels.append(adb_anno_list[idx])
        #
        #     names.append('aha_'+adb_name_list[idx]+'_ch1')
        #     train_sigs.append(adb_data_list[idx][:,1])
        #     train_labels.append(adb_anno_list[idx])

        '''preprocessing'''
        preprocessor = PlainPreprocessor(hyper_params)
        (pre_train_sigs, pre_train_labels) = preprocessor.preprocess(train_sigs, train_labels)

        os.makedirs(pp_data_dir_name)

        for idx in range(len(names)):
            pickle.dump((train_sigs[idx], train_labels[idx], pre_train_sigs[idx], pre_train_labels[idx]), open(os.path.join(pp_data_dir_name, names[idx]+'.dat'), 'wb'))
        pickle.dump((names), open(os.path.join(pp_data_dir_name, 'file_names.dat'), 'wb'))
