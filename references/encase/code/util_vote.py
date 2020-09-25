# -*- coding: utf-8 -*-

'''

'''

import numpy as np
from collections import Counter

def get_voted_proba_each_1(pre):
    return np.mean(pre, axis=0)

def get_voted_proba_each(pre):
    y_pre = [0. for j in range(4)]
    y_sec_pre = [0. for j in range(4)]
    y_third_pre = [0. for j in range(4)]
    y_pre = np.array(y_pre, dtype=np.float32)
    y_sec_pre = np.array(y_sec_pre, dtype=np.float32)
    y_third_pre = np.array(y_third_pre, dtype=np.float32)
    max_p = 0
    max_sec_p = 0
    max_third_p = 0
    sec_p = 0
    sec_sec_p = 0
    sec_third_p = 0
    
    for j in range(len(pre)):
        i_pred = np.array(pre[j], dtype=np.float32)
        
        cur_max_p = i_pred[np.argmax(i_pred)]
        cur_sec_p = 0
        for k in range(len(i_pred)):
            if i_pred[k] == cur_max_p:
                continue
            if i_pred[k] > cur_sec_p:
                cur_sec_p = i_pred[k]
        
        if (cur_max_p - cur_sec_p) > (max_p - sec_p):
            y_third_pre = y_sec_pre
            y_sec_pre = y_pre
            y_pre = i_pred
            max_p = cur_max_p
            sec_p = cur_sec_p
        elif len(pre) >= 2 and (cur_max_p - cur_sec_p) > (max_sec_p - sec_sec_p):
            y_third_pre = y_sec_pre
            y_sec_pre = i_pred
        elif len(pre) >= 3 and (cur_max_p - cur_sec_p) > (max_third_p - sec_third_p):
            y_third_pre = i_pred
            
    
    labels = [0. for j in range(4)]
    pred_1 = np.argmax(y_pre)
    labels[pred_1] +=1
    pred_2 = pred_3 = 0
    if len(pre) >= 2:
        pred_2 = np.argmax(y_sec_pre)
        labels[pred_2] +=1
    if len(pre) >= 3:
        pred_3 = np.argmax(y_third_pre)
        labels[pred_3] +=1

    # if pred_1 == 2:# and (abs(y_pre[k][np.argmax(labels)] - y_pre[k][2])/y_pre[k][np.argmax(labels)] <= 0.2):
    #     pass
    # elif pred_2 == 2:# and (abs(y_pre[k][np.argmax(labels)] - y_sec_pre[k][2])/y_pre[k][np.argmax(labels)] <= 0.2):
    #     y_pre = y_sec_pre
    # elif pred_3 == 2:# and (abs(y_pre[k][np.argmax(labels)] - y_third_pre[k][2])/y_pre[k][np.argmax(labels)] <= 0.2):
    #     y_pre = y_third_pre
    # elif pred_1 != np.argmax(labels):
    #     if pred_2 == np.argmax(labels):
    #         y_pre = y_sec_pre
    
    if pred_1 != np.argmax(labels):
        if pred_2 == np.argmax(labels):
            y_pre = y_sec_pre
            
    return y_pre

def get_voted_proba(set_proba, out_pid):
    unique_pids = sorted(list(set(out_pid)))
    seq_proba = []
    gt = []
    proba_dic = {}
    for i in range(len(out_pid)):
        if out_pid[i] in proba_dic:
            proba_dic[out_pid[i]].append(set_proba[i])
        else:
            proba_dic[out_pid[i]] = [set_proba[i]]
    for pid in unique_pids:
        seq_proba.append(get_voted_proba_each(proba_dic[pid]))
    
    return seq_proba

def group_gt(gt, pids):
    unique_pids = sorted(list(set(pids)))
    gt_dic = {k: [] for k in unique_pids}
    final_gt = []
    for i in range(len(pids)):
        gt_dic[pids[i]].append(gt[i])
    for k, v in gt_dic.items():
        final_gt.append(Counter(v).most_common(1)[0][0])
    return final_gt
