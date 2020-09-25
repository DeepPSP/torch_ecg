import keras
'''lib loading error prevention'''
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from keras.utils import to_categorical
from keras.models import load_model
import scipy as sp

"""========================"""
"""tensorflow configuration"""
""""======================="""

import tensorflow as tf
from keras import backend as K
num_cores = 48

num_CPU = 1
num_GPU = 1

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores,
                        allow_soft_placement=True,
                        device_count={'CPU': num_CPU, 'GPU': num_GPU})
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True

session = tf.Session(config=config)
K.set_session(session)
init = tf.global_variables_initializer()
session.run(init)

from sig_tool import normalize
from plain_model import PlainModel, hyper_params, gen_ckpt_prefix, gen_pp_data_dir_name
from plain_data_make import PlainPreprocessor, load_kfold_names, preload
from icbeb_tool import  load_icbeb2019_label
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt

from scipy import io
from sig_tool import  med_smooth

'''global variables'''
fold = 5
fs = 500 # sampling frequency of the data
preprocessor = PlainPreprocessor(hyper_params)


def calc(seg):
    seg = med_smooth(seg, 10)
    qrs_interval = []
    qrs = []

    qrs_start = -1
    qrs_stop = -1
    for idx in range(len(seg)):
        if seg[idx] == 1:
            if qrs_start == -1:
                # new qrs segmentation
                qrs_start = idx
            else:
                continue
        else:
            if qrs_start >= 0:
                qrs_stop = idx
                qrs_interval.append((qrs_start, qrs_stop))

                qrs_start = -1
                qrs_stop = -1
            else:
                continue
    idx = 0
    while idx < len(qrs_interval):
        # searching for
        interval = qrs_interval[idx]
        central = (interval[1] + interval[0]) // 2
        idx += 1

        # if interval[1]-interval[0] < 20:
        #     # qrs glitch
        #     # searching for next qrs
        #     idx_next = idx + 1
        #     if idx_next == len(qrs_interval):
        #         central = (interval[1]+interval[0])//2
        #         idx += 1
        #     else:
        #         while idx_next < len(qrs_interval):
        #             interval_next = qrs_interval[idx_next]
        #             if interval_next[1]-interval_next[0] < 20:
        #                 if interval_next[1]-interval[0] < 160:
        #                     idx_next += 1
        #                     if idx_next == len(qrs_interval):
        #                         central = (interval_next[1]+interval[0])//2
        #                         idx = idx_next
        #                         break
        #                 else:
        #                     central = (qrs_interval[idx_next-1][1] + interval[0])//2
        #                     idx = idx_next
        #                     break
        #             else:
        #                 central = (interval_next[1] + interval[0])//2
        #                 idx = idx_next + 1
        #                 break
        # else:
        #     central = (interval[1]+interval[0])//2
        #     idx += 1

        '''calibrate central with -20 ms if fir applied'''
        qrs.append(central)

    fs_ = 500
    r_hr = np.array([loc for loc in qrs if ((loc > 5.5 * fs_) and (loc < 5000 - 0.5 * fs_))])
    # rr = []
    # r_prev = -1
    # for r in qrs:
    #     if r < 5.5 * 500 or r > 5000 - 0.5 * 500:
    #         continue
    #
    #     if r_prev < 0:
    #         r_prev = r
    #     else:
    #         rr.append(r - r_prev)
    #         r_prev = r

    if len(r_hr) == 0:
        hr = 60
    else:
        # hr = 60 / (np.mean(rr) / 500)
        hr = np.round( 60 * fs_ / np.mean(np.diff(r_hr)))

    # plt.plot(ECG[0])
    # plt.plot(seg)
    # rwaves = np.zeros(len(seg))
    # for r in qrs:
    #     rwaves[r] = 1
    # plt.plot(rwaves)
    # plt.show()

    # fs_ = 500
    # sub_qrs_end = np.where((np.array(qrs) <= 9.575 * fs_))[0]
    # sub_qrs_start = np.where((np.array(qrs) >= 0.425 * fs_))[0]
    # if len(sub_qrs_start) >= 1 and len(sub_qrs_end) >= 1:
    #     qrs_inrange = qrs[sub_qrs_start[0]:sub_qrs_end[-1]]
    # elif len(sub_qrs_start) >= 1:
    #     qrs_inrange = qrs[sub_qrs_start[0]:]
    # elif len(sub_qrs_end) >= 1:
    #     qrs_inrange = qrs[:sub_qrs_end[-1]]
    # else:
    #     qrs_inrange = qrs[:]

    return hr, qrs

def score(r_ref, hr_ref, r_ans, hr_ans, fs_, thr_):
    HR_score = 0
    record_flags = np.ones(len(r_ref))
    for i in range(len(r_ref)):
        FN = 0
        FP = 0
        TP = 0
        hr_der = abs(int(hr_ans[i]) - int(hr_ref[i]))
        if hr_der <= 0.02 * hr_ref[i]:
            HR_score = HR_score + 1
        elif hr_der <= 0.05 * hr_ref[i]:
            HR_score = HR_score + 0.75
        elif hr_der <= 0.1 * hr_ref[i]:
            HR_score = HR_score + 0.5
        elif hr_der <= 0.2 * hr_ref[i]:
            HR_score = HR_score + 0.25

        ref_count = 0
        for j in range(len(r_ref[i])):
            # if r_ref[i][j] < 0.5 * fs_ or r_ref[i][j] > 9.5 * fs_:
            #     continue
            # ref_count += 1

            loc = np.where(np.abs(r_ans[i] - r_ref[i][j]) <= thr_ * fs_)[0]
            if j == 0:
                err = np.where((r_ans[i] >= 0.5 * fs_) & (r_ans[i] <= r_ref[i][j] - thr_ * fs_))[0]
            elif j == len(r_ref[i]) - 1:
                err = np.where((r_ans[i] >= r_ref[i][j] + thr_ * fs_) & (r_ans[i] <= 9.5 * fs_))[0]
            else:
                err = np.where((r_ans[i] >= r_ref[i][j] + thr_ * fs_) & (r_ans[i] <= r_ref[i][j + 1] - thr_ * fs_))[0]

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

    return record_flags, rec_acc, hr_acc

if __name__ == '__main__':
    train_paired, val_paired = pickle.load(open('shuffle_names.dat', 'rb'))

    models = []
    # model = load_model('models/rematch_ckpt_plain_merge_40_96086_0_012_0.0262_0.0317_0.9919_0.9882.h5')
    # model = load_model('models/rematch_ckpt_plain_merge_40_31093_2_006_0.1236_0.1534_0.9930_0.9779.h5')
    # model = load_model('models/rematch_ckpt_plain_merge_40_55824_0_015_0.0262_0.0234_0.9946_0.9952.h5')
    # model = load_model('models/rematch_ckpt_plain_merge_40_38633_1_053_0.0560_0.0956_0.9823_0.9647.h5')
    # model = load_model('models/rematch_ckpt_plain_40_50937_0_043_0.0603_0.0837_0.9705.h5')

    # model = load_model('models/rematch_ckpt_plain_rev2_40_98498_1_581_0.0652_0.0754_0.9738_0.9712.h5')
    # models.append(model)
    # model = load_model('models/rematch_ckpt_plain_rev2_40_28799_0_055_0.0439_0.0861_0.9847_0.9712.h5')
    # models.append(model)
    # model = load_model('models/rematch_ckpt_plain_rev2_40_90826_0_063_0.0573_0.1105_0.9821_0.9652.h5')
    # models.append(model)
    # model = load_model('models/rematch_ckpt_plain_rev3_40_32153_1_466_0.0743_0.0687_0.9706_0.9720.h5')
    # models.append(model)
    # model = load_model('models/rematch_ckpt_plain_rev3_40_54196_0_147_0.0746_0.0738_0.9700_0.9710.h5')
    # models.append(model)
    # model = load_model('models/rematch_ckpt_plain_rev3_40_27053_0_409_0.0679_0.0659_0.9729_0.9729.h5')
    # models.append(model)



    '''entry v2'''
    # model = load_model('models/rematch_ckpt_plain_rev4_40_19031_0_031_0.0587_0.0711_0.9763_0.9722.h5')
    # models.append(model)
    # model = load_model('models/rematch_ckpt_plain_rev4_40_92098_0_047_0.0654_0.0733_0.9740_0.9720.h5')
    # models.append(model)
    # model = load_model('models/rematch_ckpt_plain_rev4_40_11085_0_037_0.0669_0.0644_0.9729_0.9741.h5')
    # models.append(model)

    '''test entry'''
    # # Unet++ models
    # # model = load_model('models/rematch_ckpt_plain_rev4_40_90350_1_040_0.0583_0.0603_0.9771_0.9757.h5')
    # # models.append(model)
    # model = load_model('models/rematch_ckpt_plain_rev4_40_42101_0_038_0.0582_0.0607_0.9773_0.9769.h5')
    # models.append(model)
    # # model = load_model('models/rematch_ckpt_plain_rev4_40_52454_2_030_0.0667_0.0619_0.9747_0.9765.h5')
    # # models.append(model)
    #
    # # stacked lstm with attention
    # model = load_model('models/rematch_ckpt_plain_rev4_40_77525_0_036_0.0606_0.0622_0.9754_0.9751.h5')
    # models.append(model)
    #
    # # U-net conv
    # # model = load_model('models/rematch_ckpt_plain_rev4_40_81909_0_055_0.0734_0.0633_0.9710_0.9751.h5')
    # # models.append(model)
    # # model = load_model('models/rematch_ckpt_plain_rev4_50_11185_0_047_0.0909_0.0733_0.9648_0.9717.h5')
    # # models.append(model)
    #
    # # U-net LSTM
    # model = load_model('models/rematch_ckpt_plain_rev4_30_91712_0_061_0.0610_0.0576_0.9751_0.9770.h5')
    # models.append(model)

    '''v.2.3 finetune'''
    #
    # model = load_model('models/rematch_ckpt_plain_rev4_30_96625_0_001_0.0510_0.0587_0.9789_0.9771.h5') # U-net LSTM
    # models.append(model)
    # model = load_model('models/rematch_ckpt_plain_rev4_30_52643_0_001_0.0562_0.0622_0.9772_0.9746.h5') # stacked LSTM
    # models.append(model)
    # model = load_model('models/rematch_ckpt_plain_rev4_30_59847_0_001_0.0532_0.0606_0.9792_0.9768.h5') # U-net++ LSTM
    # models.append(model)

    '''v.2.4 finetune'''

    # model = load_model('models/rematch_ckpt_plain_rev4_30_69622_0_002_0.0498_0.0590_0.9793_0.9770.h5') # U-net LSTM
    # models.append(model)
    # model = load_model('models/rematch_ckpt_plain_rev4_30_91612_0_004_0.0511_0.0613_0.9792_0.9761.h5') # stacked LSTM
    # model = load_model('models/rematch_ckpt_plain_rev4_30_52643_0_001_0.0562_0.0622_0.9772_0.9746.h5') # stacked LSTM
    # models.append(model)
    # model = load_model('models/rematch_ckpt_plain_rev4_30_17321_2_002_0.0517_0.0608_0.9799_0.9768.h5') # U-net++ LSTM
    # models.append(model)

    '''v.2.5 with sig0'''
    # model = load_model('models/rematch_ckpt_plain_rev4_30_sig0_88967_0_055_0.0613_0.0627_0.9750_0.9741.h5') # HR-net LSTM
    # models.append(model)
    # model = load_model('models/rematch_ckpt_plain_rev4_30_sig0_66164_1_053_0.0634_0.0619_0.9742_0.9755.h5') # U-net LSTM
    # models.append(model)
    # model = load_model('models/rematch_ckpt_plain_rev4_30_sig0_80957_0_194_0.0514_0.0851_0.9849_0.9729.h5') # U-net++ LSTM
    # models.append(model)
    # model = load_model('models/rematch_ckpt_plain_rev4_30_sig0_35157_0_001_0.0534_0.0613_0.9781_0.9744.h5') # stacked LSTM
    # models.append(model)
    # model = load_model('models/rematch_ckpt_plain_rev4_30_sig0_30705_0_001_0.0564_0.0603_0.9770_0.9763.h5') #U-net++ LSTM
    # models.append(model)
    # model = load_model('models/rematch_ckpt_plain_rev4_30_sig0_87538_0_001_0.0587_0.0618_0.9776_0.9768.h5') #U-net LSTM
    # models.append(model)
    # model = load_model('models/rematch_ckpt_plain_rev4_30_sig0_23347_0_001_0.0518_0.0633_0.9787_0.9743.h5') #stacked LSTM
    # models.append(model)
    # model = load_model('models/rematch_ckpt_plain_rev4_30_sig0_63087_0_001_0.0498_0.0639_0.9795_0.9749.h5') #HRnet LSTM
    # models.append(model)


    '''v.3 test'''
    # model = load_model('models/rematch_ckpt_plain_rev4_30_sig0_37092_0_081_0.0609_0.0626_0.9751_0.9750.h5') #HRnet LSTM
    # models.append(model)
    # model = load_model('models/ckpt_ex_rev5_lstmunet_9014_0_30_082_0.0314_0.0828_0.9869_0.9748. h5') #HRnet LSTM
    # models.append(model)
    # model = load_model('models/rematch_ckpt_plain_rev4_30_unetconvlstm_sig71698_0_189_0.2243_0.2186_0.9737_0.9761.h5') #HRnet LSTM
    # models.append(model)

    '''v.3 kfold bagging lstm unet'''
    model = load_model('models/tune_fold10_30_netlstm_sig1_89383_4_002_0.0553_0.0452_0.9776_0.9811.h5')
    models.append(model)
    model = load_model('models/tune_fold10_30_netlstm_sig1_99576_3_007_0.0579_0.0484_0.9765_0.9801.h5')
    models.append(model)
    model = load_model('models/tune_fold10_30_netlstm_sig1_59925_7_003_0.0641_0.0546_0.9744_0.9781.h5')
    models.append(model)
    # model = load_model('models/tune_fold10_30_netlstm_sig1_24285_9_004_0.0651_0.0704_0.9740_0.9715.h5')
    # models.append(model)
    model = load_model('models/tune_fold10_30_netlstm_sig1_64082_1_007_0.0628_0.0658_0.9746_0.9732.h5')
    models.append(model)

    data_dir = gen_pp_data_dir_name()

    batch_x = []
    batch_x0 = []
    batch_y = []

    cnt = 0
    ref_hr = []
    ref_r = []
    # val = val_paired[:]
    # val = val_paired[:1900]
    val = val_paired[600:800]
    for name, offset in val:
        train_sig, train_label, pre_train_sig, pre_train_label = pickle.load(open(os.path.join(data_dir, name+'.dat'), 'rb'))

        RPOS_PATH = 'dat/icbeb2019/ref/'
        ref_path = os.path.join(RPOS_PATH, 'R_'+ str.split(name,'_')[1])
        r = io.loadmat(ref_path)['R_peak'].flatten()

        # mitdb label and sig extraction
        # label = train_label[offset:offset+hyper_params['crop_len']]
        # pre_train_label = pre_train_label[offset:offset+hyper_params['crop_len']]
        #
        # r = np.array([i for i in np.where(label == 1)])
        # pre_train_sig = pre_train_sig[offset:offset+hyper_params['crop_len']]
        # plt.plot(pre_train_sig[:,1])
        # plt.plot(pre_train_label)
        # plt.show()

        r = r[(r >= 0.5*500) & (r <= 9.5*500)]

        ref_r.append(r)
        r_hr = np.array([loc for loc in r if ((loc > 5.5 * 500) and (loc < 5000 - 0.5 * 500))])
        hr = round( 60 * 500 / np.mean(np.diff(r_hr)))
        ref_hr.append(hr)

        sig = pre_train_sig
        # sig = normalize(sig)
        if np.isnan(sig).any():
            continue

        # plt.plot(train_sig)
        # plt.plot(sig[:,0])
        # plt.plot(sig[:,1])
        # plt.plot(pre_train_label)
        # plt.legend(['raw', 'sig', 'diff', 'label'])
        # plt.show()

        # batch_x.append(np.transpose(sig))
        batch_x.append(pre_train_sig[:,1])
        batch_x0.append(pre_train_sig[:,0])
        batch_y.append(pre_train_label)

    batch_x = np.reshape(batch_x, newshape=(len(batch_x), hyper_params['crop_len'], 1))
    batch_x0 = np.reshape(batch_x0, newshape=(len(batch_x), hyper_params['crop_len'], 1))
    segs = []
    for model in models:
        seg = model.predict(np.array(batch_x))
        segs.append(seg)
    # seg = models[0].predict(np.array(batch_x))
    # segs.append(seg)
    # seg = models[1].predict(np.array(batch_x0))
    # segs.append(seg)
    # segs.append(seg[1])
    # seg = models[1].predict(np.array(batch_x))
    # segs.append(seg)
    # seg = models[3].predict(np.array(batch_x0))
    # segs.append(seg)
    # seg = models[4].predict(np.array(batch_x0))
    # segs.append(seg)
    # seg = models[5].predict(np.array(batch_x0))
    # segs.append(seg)


    # store the data and seg into mat
    sp.io.savemat('smooth_demo.mat', {'x':batch_x, 'y':batch_y, 'seg':segs, 'ref_r':ref_r})

    FS = 500
    THR = 0.075

    # score for individual networks
    ss_esti = []
    for seg in segs:
        ss = np.argmax(seg, axis=2)

        hr_ans = []
        r_ans = []
        for s in ss:
            hr, qrs = calc(s)
            hr_ans.append(hr)
            r_ans.append(np.array(qrs))

        record_flags, _, _ = score(ref_r, ref_hr, r_ans, hr_ans, FS, THR)

        ss_esti.append(ss)


    # score for ensemble networks

    # linear ensemble
    seg = np.average(segs, axis=0)
    seg = np.argmax(seg, axis=2)

    #voting
    # ss_sum = np.sum(ss_esti, axis=0)
    # seg = []
    # for ss in ss_sum:
    #     seg.append([0 if s <= 2 else 1 for s in ss])

    hr_ans = []
    r_ans = []
    for s in seg:
        hr, qrs = calc(s)
        hr_ans.append(hr)
        r_ans.append(np.array(qrs))

    record_flags, _, _ = score(ref_r, ref_hr, r_ans, hr_ans, FS, THR)


    for idx in range(len(val)):
        if record_flags[idx] >= 0.9:
            continue
        r_seq = np.zeros(5000)
        for r in r_ans[idx]:
            r_seq[r] = 1
        r_seq_ref = np.zeros(5000)
        for r in ref_r[idx]:
            r_seq_ref[r] = 1

        plt.figure(figsize=(16,16))
        plt.title(str(val_paired[idx][0])+' '+ str(record_flags[idx]))
        plt.plot(batch_x[idx]+1)
        plt.plot(seg[idx])
        plt.plot(r_seq)
        plt.plot(batch_y[idx]-1)
        plt.plot(r_seq_ref-1)
        plt.legend(['sig', 'seg', 'label'])
        plt.show()






