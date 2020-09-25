import scipy.io as sio
import numpy as np
import os


def load_icbeb2018(training_set=[1,2,3], db_dir='icbeb2018'):
    root_dir = os.path.join(os.getcwd(), db_dir)

    file_list = []
    dat_list = []
    label_list = []
    for idx in training_set:
        if idx == 1:
            training_set_dir = os.path.join(root_dir, 'TrainingSet1')
        elif idx == 2:
            training_set_dir = os.path.join(root_dir, 'TrainingSet2')
        elif idx == 3:
            training_set_dir = os.path.join(root_dir, 'TrainingSet3')

        for root, dirs, files in os.walk(training_set_dir):
            [file_list.append(f) for f in files]

        for f in file_list:
            mat_data = sio.matlab.loadmat(os.path.join(training_set_dir, f))
            dat_list.append(mat_data['dat'])
            label_list.append(mat_data['label'])

def load_icbeb2019_label(db_dir='icbeb2019'):
    dat_dir = os.path.join(os.getcwd(), db_dir, 'data')
    ref_dir = os.path.join(os.getcwd(), db_dir, 'ref')
    dat_file_list = []
    for root, dirs, files in os.walk(dat_dir):
        [dat_file_list.append(f) for f in files]

    dat_name = [dat_file.split('.')[0].split('_')[1] for dat_file in dat_file_list]
    ref_file_list = []
    for root, dirs, files in os.walk(ref_dir):
        [ref_file_list.append(f) for f in files]
    name_list = []
    label_list = []
    for name in dat_name:
        ref_file = 'R_' + name
        mat_label = sio.matlab.loadmat(os.path.join(ref_dir, ref_file))
        r_list = mat_label['R_peak']
        label = np.zeros((5000,1))
        for r in r_list:
            label[r - 1] = 1

        name_list.append(name)
        label_list.append(label)

    return name_list, label_list

def load_icbeb2019(db_dir='icbeb2019'):
    dat_dir = os.path.join(os.getcwd(), db_dir, 'data')
    ref_dir = os.path.join(os.getcwd(), db_dir, 'ref')
    dat_file_list = []
    for root, dirs, files in os.walk(dat_dir):
        [dat_file_list.append(f) for f in files]

    dat_name = [dat_file.split('.')[0].split('_')[1] for dat_file in dat_file_list]

    ref_file_list = []
    for root, dirs, files in os.walk(ref_dir):
        [ref_file_list.append(f) for f in files]

    sig_list = []
    label_list = []
    name_list = []
    for name in dat_name:
        print('loading data ', name)
        dat_file = 'data_' + name

        mat_data = sio.matlab.loadmat(os.path.join(dat_dir, dat_file))
        ecg = mat_data['ecg']

        ref_file = 'R_' + name
        mat_label = sio.matlab.loadmat(os.path.join(ref_dir, ref_file))
        r_list = mat_label['R_peak']
        label = np.zeros(np.shape(ecg))
        for r in r_list:
            label[r-1] = 1

        name_list.append(name)
        sig_list.append(ecg)
        label_list.append(label)

    return name_list, sig_list, label_list
