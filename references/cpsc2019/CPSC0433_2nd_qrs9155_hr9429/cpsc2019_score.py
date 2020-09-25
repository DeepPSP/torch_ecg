from CPSC2019_challenge import CPSC2019_challenge
import numpy as np
import math
np.set_printoptions(threshold=np.inf)
import numpy as np
import tensorflow as tf
np.set_printoptions(threshold=np.inf)
import os
import re
from scipy import io
from keras.models import Model, load_model
import numpy as np
import scipy.signal as signal
import scipy.io
import keras.backend as K

def load_ans(data_path_, rpos_path_, fs_,model1,model2,model3,model4,model5,model6,model7):
    '''
    Please modify this function when you have to load model or any other parameters in CPSC2019_challenge()
    '''
    def is_mat(l):
        return l.endswith('.mat')
    ecg_files = list(filter(is_mat, os.listdir(data_path_)))
    rpos_files = list(filter(is_mat, os.listdir(rpos_path_)))
#    print(ecg_files)
    HR_ref = []
    R_ref = []
    HR_ans = []
    R_ans = []

    ecg_data=np.zeros((len(rpos_files),5000))
    templabel=[]
    tempref=[]
    temprh=[]
    n=0
    for rpos_file in rpos_files:
        index = re.split('[_.]', rpos_file)[1]
        ecg_file = 'data_' + index + '.mat'

        ref_path = os.path.join(rpos_path_, rpos_file)
        ecg_path = os.path.join(data_path_, ecg_file)

        ecg_data[n] = np.transpose(io.loadmat(ecg_path)['ecg'])[0]
        r_ref = io.loadmat(ref_path)['R_peak'].flatten()
#        print(r_ref)
        r_ref = r_ref[(r_ref >= 0.5*fs_) & (r_ref <= 9.5*fs_)]

        r_hr = np.array([loc for loc in r_ref if ((loc > 5.5 * fs_) and (loc < len(ecg_data[0]) - 0.5 * fs_))])

        
        temprh.append(r_hr)
        tempref.append(r_ref)
        

        n=n+1
        print(n)

    hr_ans, r_ans,result = CPSC2019_challenge(ecg_data,model1,model2,model3,model4,model5,model6,model7)

    for i in range(len(rpos_files)):
        HR_ref.append(round( 60 * fs_ / np.mean(np.diff(temprh[i]))))
        R_ref.append(tempref[i])
    


    HR_ans=hr_ans
    R_ans=r_ans

    print('finish')
    

    
    return R_ref, HR_ref, R_ans, HR_ans,result

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

    return rec_acc, hr_acc

if __name__ == '__main__':
    FS = 500
    THR = 0.075
    DATA_PATH = 'D:/心梗/2019生理参数竞赛/评分更新/train/data/'
    RPOS_PATH = 'D:/心梗/2019生理参数竞赛/评分更新/train/ref/'

    
    
    model1=load_model('D:/心梗/2019生理参数竞赛/改竞赛/终极上传——小波与之前模型的融合/最终融合模型/challenge9_21.h5')
    model2=load_model('D:/心梗/2019生理参数竞赛/改竞赛/终极上传——小波与之前模型的融合/最终融合模型/challenge9_27.h5')
    
    model3=load_model('D:/心梗/2019生理参数竞赛/改竞赛/终极上传——小波与之前模型的融合/最终融合模型/val_acc0.97177xiaobo没去噪 0.9585，0.9812.h5')
    model4=load_model('D:/心梗/2019生理参数竞赛/改竞赛/终极上传——小波与之前模型的融合/最终融合模型/xiaobo40.963，0.9712.h5')
    model5=load_model('D:/心梗/2019生理参数竞赛/改竞赛/终极上传——小波与之前模型的融合/最终融合模型/xiaoboseed4 0.9643，0.9688.h5')
    model6=load_model('D:/心梗/2019生理参数竞赛/改竞赛/终极上传——小波与之前模型的融合/最终融合模型/xiaoboseed8 0.9845，0.9912.h5')
    model7=load_model('D:/心梗/2019生理参数竞赛/改竞赛/终极上传——小波与之前模型的融合/最终融合模型/xiaoboseed9 0.964，0.98.h5')
    
    
    
    R_ref, HR_ref, R_ans, HR_ans,result = load_ans(DATA_PATH, RPOS_PATH, FS,model1,model2,model3,model4,model5,model6,model7)
    rec_acc, hr_acc = score(R_ref, HR_ref, R_ans, HR_ans, FS, THR)
