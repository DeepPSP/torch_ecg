from scipy.io import loadmat
import numpy as np
import math
from config import opt
import torch
import scipy.signal as ss
import matplotlib.pyplot as plt
import os

def get_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
        lr =lr[0]

    return lr

def get_val_ref(val_dir,val_bs,batch_id):
    HR_ref = []
    R_ref = []
    val_names = open(val_dir,'r').readlines()
    start_pos = val_bs * batch_id
    end_pos = val_bs * (batch_id + 1) if val_bs * (batch_id + 1) <len(val_names) else None
    val_names = val_names[start_pos:end_pos]
    for name in val_names:
        name= name.split()[0]
        r_ref = loadmat('./train/ref/R_' + name[-9:])['R_peak'].flatten()
        r_ref = r_ref[(r_ref >= 0.5 * 500) & (r_ref <= 9.5 * 500)]
        r_hr = np.array([loc for loc in r_ref if (loc > 5.5 * 500 and loc < 5000 - 0.5 * 500)])
        HR_ref.append(round(60 * 500 / np.mean(np.diff(r_hr))))
        R_ref.append(r_ref)
        # ref.append(label)
    return R_ref, HR_ref

def get_val_ref_names(val_dir,val_bs,batch_id):
    HR_ref = []
    R_ref = []
    val_names = open(val_dir,'r').readlines()
    start_pos = val_bs * batch_id
    end_pos = val_bs * (batch_id + 1) if val_bs * (batch_id + 1) <len(val_names) else None
    val_names = val_names[start_pos:end_pos]
    for name in val_names:
        name= name.split()[0]
        r_ref = loadmat('./train/ref/R_' + name[-9:])['R_peak'].flatten()
        r_ref = r_ref[(r_ref >= 0.5 * 500) & (r_ref <= 9.5 * 500)]
        r_hr = np.array([loc for loc in r_ref if (loc > 5.5 * 500 and loc < 5000 - 0.5 * 500)])
        HR_ref.append(round(60 * 500 / np.mean(np.diff(r_hr))))
        R_ref.append(r_ref)
        # ref.append(label)
    return R_ref, HR_ref , val_names

def get_sig(name):
    name = name.split()[0]
    sig = loadmat('./train/data/'+name)['ecg'].squeeze()
    signal = ss.medfilt(sig, 3)
    lowpass = ss.butter(2, 40.0 / (500 / 2.0), 'low')  # 40-45都可以，解决工频干扰
    signal_bp = ss.filtfilt(*lowpass, x=signal)
    lowpass = ss.butter(2, 2.0 / (500 / 2.0), 'low')  # 1.5-2.5都可以，计算基线
    baseline = ss.filtfilt(*lowpass, x=signal_bp)
    sbp = signal_bp - baseline
    sbp = np.array((sbp - sbp.mean()) / (sbp.std()))
    sbp = sbp / (sbp.max())
    return sbp



def analy_draw(r_ref, hr_ref, r_ans, r_ans_dbug,hr_ans, fs_, thr_,names):
    HR_score = 0
    record_flags = np.ones(len(r_ref))
    for i in range(len(r_ref)):
        FN_lst = []
        FP_lst = []
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
            loc = np.where(np.abs(r_ans[i] - r_ref[i][j]) <= thr_ * fs_)[0]
            if j == 0:
                err = np.where((r_ans[i] >= 0.5 * fs_ + thr_ * fs_) & (r_ans[i] <= r_ref[i][j] - thr_ * fs_))[0]
                if len(err)>0:
                    FP_lst.append(r_ans[i][err])
            elif j == len(r_ref[i]) - 1:
                err = np.where((r_ans[i] >= r_ref[i][j] + thr_ * fs_) & (r_ans[i] <= 9.5 * fs_ - thr_ * fs_))[0]
                if len(err) > 0:
                    FP_lst.append(r_ans[i][err])
            else:
                err = np.where((r_ans[i] >= r_ref[i][j] + thr_ * fs_) & (r_ans[i] <= r_ref[i][j + 1] - thr_ * fs_))[0]
                if len(err) > 0:
                    FP_lst.append(r_ans[i][err])

            FP = FP + len(err)
            if len(loc) >= 1:
                TP += 1
                FP = FP + len(loc) - 1
            elif len(loc) == 0:
                FN_lst.append(r_ref[i][j])
                FN += 1

        if FN + FP > 1:
            if not os.path.exists('./val_fig/0_fig/'):
                os.makedirs('./val_fig/0_fig/')
            record_flags[i] = 0
            sig = get_sig(names[i])
            plt.plot(sig)
            plt.plot(r_ref[i], sig[r_ref[i]], 'go')

            x_N = np.array(FN_lst).astype('int')
            y_N = sig[x_N]
            plt.plot(x_N, y_N, 'r+',markersize=10)
            x_ = np.array(r_ans_dbug[i]).astype('int')
            y_ = sig[x_]
            plt.plot(x_, y_, 'bo')

            if len(FP_lst)>0:
                x_P = np.concatenate(FP_lst,0).astype('int')
                # x_P = np.array(FP_lst).astype('int')
                y_P = sig[x_P]
                plt.plot(x_P,y_P, 'yo')

            # x = r_ref[i]
            # y = sig[x]
            # plt.plot(r_ref[i],sig[r_ref[i]],'go')
            # plt.plot(r_ans[i], sig[r_ans[i].astype('int')+20], 'ro')
            plt.title('FP:{:d},FN:{:d}'.format(FP,FN))
            plt.savefig('./val_fig/0_fig/'+names[i][:-4]+'png')
            plt.close()

        elif FN == 1 and FP == 0:
            if not os.path.exists('./val_fig/0.3_fig/'):
                os.makedirs('./val_fig/0.3_fig/')
            record_flags[i] = 0.3
            sig = get_sig(names[i])
            plt.plot(sig)
            plt.plot(r_ref[i], sig[r_ref[i]], 'go')

            x = np.array(FN_lst).astype('int')
            y = sig[x]
            plt.plot(x,y,'r+',markersize=40)

            x_ = np.array(r_ans_dbug[i]).astype('int')
            y_ = sig[x_]
            plt.plot(x_, y_, 'bo')
            # plt.plot(r_ans[i], sig[r_ans[i].astype('int')+20], 'ro')
            # plt.plot(np.array(FN_lst),sig[np.array(FN_lst)],'ro')
            plt.title('FP:{:d},FN:{:d}'.format(FP,FN))
            plt.savefig('./val_fig/0.3_fig/'+names[i][:-4]+'png')
            plt.close()
        elif FN == 0 and FP == 1:
            if not os.path.exists('./val_fig/0.7_fig/'):
                os.makedirs('./val_fig/0.7_fig/')
            record_flags[i] = 0.7
            sig = get_sig(names[i])
            plt.plot(sig)
            plt.plot(r_ref[i], sig[r_ref[i]], 'go')


            # tt = np.array(FP_lst).squeeze()
            x = np.array(FP_lst).astype('int')
            y = sig[x]
            # tt = np.int(r_ans[i])
            # plt.plot(r_ans[i], sig[r_ans[i].astype('int')+20], 'ro')
            plt.plot(x,y, 'yo')
            plt.title('FP:{:d},FN:{:d}'.format(FP,FN))
            plt.savefig('./val_fig/0.7_fig/'+names[i][:-4]+'png')
            plt.close()
    rec_acc = round(np.sum(record_flags) / len(r_ref), 4)
    hr_acc = round(HR_score / len(r_ref), 4)

        # print( 'QRS_acc: {}'.format(rec_acc))
        # print('HR_acc: {}'.format(hr_acc))
        # print('Scoring complete.')

    return rec_acc, hr_acc

def analy_draw_v2(r_ref, hr_ref, r_ans,hr_ans, fs_, thr_,names):
    HR_score = 0
    record_flags = np.ones(len(r_ref))
    for i in range(len(r_ref)):
        FN_lst = []
        FP_lst = []
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
            loc = np.where(np.abs(r_ans[i] - r_ref[i][j]) <= thr_ * fs_)[0]
            if j == 0:
                err = np.where((r_ans[i] >= 0.5 * fs_ + thr_ * fs_) & (r_ans[i] <= r_ref[i][j] - thr_ * fs_))[0]
                if len(err)>0:
                    FP_lst.append(r_ans[i][err])
            elif j == len(r_ref[i]) - 1:
                err = np.where((r_ans[i] >= r_ref[i][j] + thr_ * fs_) & (r_ans[i] <= 9.5 * fs_ - thr_ * fs_))[0]
                if len(err) > 0:
                    FP_lst.append(r_ans[i][err])
            else:
                err = np.where((r_ans[i] >= r_ref[i][j] + thr_ * fs_) & (r_ans[i] <= r_ref[i][j + 1] - thr_ * fs_))[0]
                if len(err) > 0:
                    FP_lst.append(r_ans[i][err])

            FP = FP + len(err)
            if len(loc) >= 1:
                TP += 1
                FP = FP + len(loc) - 1
            elif len(loc) == 0:
                FN_lst.append(r_ref[i][j])
                FN += 1

        if FN + FP > 1:
            if not os.path.exists('./val_fig_v2/0_fig/'):
                os.makedirs('./val_fig_v2/0_fig/')
            record_flags[i] = 0
            sig = get_sig(names[i])
            plt.plot(sig)
            plt.plot(r_ref[i], sig[r_ref[i]], 'go')

            if len(FN_lst)>0:
                x_N = np.array(FN_lst).astype('int')
                y_N = sig[x_N]
                plt.plot(x_N, y_N, 'r+',markersize=10)

            if len(FP_lst)>0:
                x_P = np.concatenate(FP_lst, 0).astype('int')
                # x_P = np.array(FP_lst).astype('int')
                y_P = sig[x_P]
                plt.plot(x_P,y_P, 'yo')

            # x = r_ref[i]
            # y = sig[x]
            # plt.plot(r_ref[i],sig[r_ref[i]],'go')
            # plt.plot(r_ans[i], sig[r_ans[i].astype('int')+20], 'ro')
            plt.title('FP:{:d},FN:{:d}'.format(FP,FN))
            plt.savefig('./val_fig_v2/0_fig/'+names[i][:-4]+'png')
            plt.close()

        elif FN == 1 and FP == 0:
            if not os.path.exists('./val_fig_v2/0.3_fig/'):
                os.makedirs('./val_fig_v2/0.3_fig/')
            record_flags[i] = 0.3
            sig = get_sig(names[i])
            plt.plot(sig)
            plt.plot(r_ref[i], sig[r_ref[i]], 'go')

            x = np.array(FN_lst).astype('int')
            y = sig[x]
            plt.plot(x,y,'r+',markersize=40)


            # plt.plot(r_ans[i], sig[r_ans[i].astype('int')+20], 'ro')
            # plt.plot(np.array(FN_lst),sig[np.array(FN_lst)],'ro')
            plt.title('FP:{:d},FN:{:d}'.format(FP,FN))
            plt.savefig('./val_fig_v2/0.3_fig/'+names[i][:-4]+'png')
            plt.close()
        elif FN == 0 and FP == 1:
            if not os.path.exists('./val_fig_v2/0.7_fig/'):
                os.makedirs('./val_fig_v2/0.7_fig/')
            record_flags[i] = 0.7
            sig = get_sig(names[i])
            plt.plot(sig)
            plt.plot(r_ref[i], sig[r_ref[i]], 'go')


            # tt = np.array(FP_lst).squeeze()
            x = np.array(FP_lst).astype('int')
            y = sig[x]
            # tt = np.int(r_ans[i])
            # plt.plot(r_ans[i], sig[r_ans[i].astype('int')+20], 'ro')
            plt.plot(x,y, 'yo')
            plt.title('FP:{:d},FN:{:d}'.format(FP,FN))
            plt.savefig('./val_fig_v2/0.7_fig/'+names[i][:-4]+'png')
            plt.close()
    rec_acc = round(np.sum(record_flags) / len(r_ref), 4)
    hr_acc = round(HR_score / len(r_ref), 4)

    # print( 'QRS_acc: {}'.format(rec_acc))
    # print('HR_acc: {}'.format(hr_acc))
    # print('Scoring complete.')

    return rec_acc, hr_acc

def score(r_ref, hr_ref, r_ans, hr_ans, fs_, thr_):
    HR_score = 0
    record_flags = np.ones(len(r_ref))
    for i in range(len(r_ref)):

        if math.isnan(hr_ans[i]):
            record_flags[i] = 0
            continue
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

        for j in range(len(r_ref[i])):
            loc = np.where(np.abs(r_ans[i] - r_ref[i][j]) <= thr_*fs_)[0]
            if j == 0:
                err = np.where((r_ans[i] >= 0.5*fs_) & (r_ans[i] <= r_ref[i][j] - thr_*fs_))[0]
            elif j == len(r_ref[i])-1:
                err = np.where((r_ans[i] >= r_ref[i][j]+thr_*fs_) & (r_ans[i] <= 9.5*fs_))[0]
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

    # print( 'QRS_acc: {}'.format(rec_acc))
    # print('HR_acc: {}'.format(hr_acc))
    # print('Scoring complete.')

    return rec_acc, hr_acc

def cal_dis(sig1,sig2):
    distance = torch.abs(sig1[0]-sig2[:,0])
    return distance

def predict(out,conf_thr):
    out = torch.cat(out,1)
    # out = out.detach().cpu()
    hr_ans = []
    r_ans = []
    for i_sig,pred in enumerate(out):     ## nms
        conf_mask = pred[...,1]>conf_thr
        pred = pred[conf_mask]
        if not pred.size(0):
            hr_ans.append(math.nan)
            r_ans.append(np.array([]))
            continue
        _,conf_sort_idex = torch.sort(pred[:,1],descending=True)
        pred = pred[conf_sort_idex]
        max_pred = []
        while pred.size(0):
            max_pred.append(pred[0])
            if len(pred)==1:
                break
            dis = cal_dis(max_pred[-1],pred[1:])
            pred = pred[1:][dis>opt.nms_thres]
        max_pred = torch.cat(max_pred,0).view(-1,2)
        _,point_sort_index = torch.sort(max_pred[:,0])
        max_pred = np.array(max_pred[point_sort_index])
        idx = (max_pred[:,0] >= 0.5 * 500) & (max_pred[:,0] <= 9.5 * 500)
        r_peak = max_pred[idx,0]
        r_hr = np.array([loc for loc in r_peak if (loc > 5.5 * 500 and loc < 5000 - 0.5 * 500)])
        hr = round(60 * 500 / np.mean(np.diff(r_hr)))
        hr_ans.append(hr)
        r_ans.append(r_peak)
    return r_ans,hr_ans