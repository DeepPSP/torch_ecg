import numpy as np
np.set_printoptions(threshold=np.inf)
import os
import re
import scipy.io as sio
import math
from CPSC2019_challenge import *
from tqdm import tqdm
import keras.backend.tensorflow_backend as KTF
from keras.models import model_from_yaml,load_model
import tensorflow as tf

from net import build_model
from keras.layers import Input
from keras.models import Model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config = config)
# 设置session
KTF.set_session(sess)

model1 = model_from_yaml(open('model_architecture_1.yaml').read())
model1.load_weights('model_weights_1.h5')

model2 = model_from_yaml(open('model_architecture_2.yaml').read())
model2.load_weights('model_weights_2.h5')

input_layer = Input((1280, 1))
output_layer = build_model(input_layer=input_layer,start_neurons=16)
model3 = Model(input_layer, output_layer)
model3.load_weights('model_weights_3.h5')

model4 = model_from_yaml(open('model_architecture_4.yaml').read())
model4.load_weights('model_weights_4.h5')

model = []
model.append(model1)
model.append(model2)
model.append(model3)
model.append(model4)


def load_ans(data_path_, rpos_path_, fs_):
    '''
    Please modify this function when you have to load model or any other parameters in CPSC2019_challenge()
    '''
    def is_mat(l):
        return l.endswith('.mat')
    ecg_files = list(filter(is_mat, os.listdir(data_path_)))
    rpos_files = list(filter(is_mat, os.listdir(rpos_path_)))
    HR_ref = []
    R_ref = []
    HR_ans = []
    R_ans = []
    for rpos_file in tqdm(rpos_files):
        index = re.split('[_.]', rpos_file)[1]
        ecg_file = 'data_' + index + '.mat'

        ref_path = os.path.join(rpos_path_, rpos_file)
        ecg_path = os.path.join(data_path_, ecg_file)

        ecg_data = np.transpose(sio.loadmat(ecg_path)['ecg'])[0]
        ecg_data = signal.resample(ecg_data, fs_*10)
        ecg_data = med_filt_1d(ecg_data)

        r_ref = sio.loadmat(ref_path)['R_peak'].flatten()
        r_ref = (r_ref/500*fs_).astype(int)
        r_ref = r_ref[(r_ref >= 0.5*fs_) & (r_ref <= 9.5*fs_)]
        
        r_hr = np.array([loc for loc in r_ref if (loc > 5.5 * fs_ and loc < len(ecg_data) - 0.5 * fs_)])
        hr_ans, r_ans = CPSC2019_challenge(model,ecg_data,fs_)

        HR_ref.append(round( 60 * fs_ / np.mean(np.diff(r_hr))))
        R_ref.append(r_ref)
        HR_ans.append(hr_ans)
        R_ans.append(r_ans)

    return R_ref, HR_ref, R_ans, HR_ans

def score(r_ref, hr_ref, r_ans, hr_ans, fs_, thr_):
    HR_score = 0
    record_flags = np.ones(len(r_ref))
    for i in range(len(r_ref)):
        FN = 0
        FP = 0
        TP = 0

        if math.isnan(hr_ans[i]):
            hr_ans[i] = 0
        hr_der = abs(int(hr_ans[i]) - int(hr_ref[i]))
        if hr_der <= 0.02 * hr_ref[i]:
            HR_score = HR_score + 1
        elif hr_der <= 0.05 * hr_ref[i]:
            HR_score = HR_score + 0.75
        elif hr_der <= 0.1 * hr_ref[i]:
            HR_score = HR_score + 0.5
        elif hr_der <= 0.2 * hr_ref[i]:
            HR_score = HR_score + 0.25

        for j in range(len(r_ref[i])):
            loc = np.where(np.abs(r_ans[i] - r_ref[i][j]) <= thr_*fs_)[0]
            if j == 0:
                err = np.where((r_ans[i] >= 0.5*fs_ + thr_*fs_) & (r_ans[i] <= r_ref[i][j] - thr_*fs_))[0]
            elif j == len(r_ref[i])-1:
                err = np.where((r_ans[i] >= r_ref[i][j]+thr_*fs_) & (r_ans[i] <= 9.5*fs_ - thr_*fs_))[0]
            else:
                err = np.where((r_ans[i] >= r_ref[i][j]+thr_*fs_) & (r_ans[i] <= r_ref[i][j+1]-thr_*fs_))[0]

            FP = FP + len(err)
            if len(loc) >= 1:
                TP += 1
                FP = FP + len(loc) - 1
            elif len(loc) == 0:
                FN += 1

        if FN + FP > 1:
            record_flags[i] = 0
        elif FN == 1 and FP == 0:
            record_flags[i] = 0.3
        elif FN == 0 and FP == 1:
            record_flags[i] = 0.7

    rec_acc = round(np.sum(record_flags) / len(r_ref), 4)
    hr_acc = round(HR_score / len(r_ref), 4)

    print( 'QRS_acc: {}'.format(rec_acc))
    print('HR_acc: {}'.format(hr_acc))
    print('Scoring complete.')

    # print("Score 0   :",len(record_flags[record_flags==0.0]))
    # print("Score 0.3 :",len(record_flags[record_flags==0.3]))
    # print("Score 0.7 :",len(record_flags[record_flags==0.7]))
    # print("Score 1   :",len(record_flags[record_flags==1]))

    return rec_acc, hr_acc

if __name__ == '__main__':
    FS1 = 500
    FS2 = 128
    THR = 0.075

    DATA_PATH = './data/'
    RPOS_PATH = './ref/'

    DATA_PATH='..\\train\\data\\'
    RPOS_PATH='..\\train\\ref\\'


    R_ref, HR_ref, R_ans, HR_ans = load_ans(DATA_PATH, RPOS_PATH, FS2)
    rec_acc, hr_acc = score(R_ref, HR_ref, R_ans, HR_ans, FS2, THR)

    with open('score.txt', 'w') as score_file:
        print('Total File Number: %d\n' %(np.shape(HR_ans)[0]), file=score_file)
        print('R Detection Acc: %0.4f' %rec_acc, file=score_file)
        print('HR Detection Acc: %0.4f' %hr_acc, file=score_file)

        score_file.close()
