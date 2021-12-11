#!/usr/bin/env python3

import numpy as np
import json
import os
import sys

import scipy.io as sio
import wfdb

"""
Written by:  Xingyao Wang, Chengyu Liu
             School of Instrument Science and Engineering
             Southeast University, China
             chengyu@seu.edu.cn
"""

R = np.array([[1, -1, -.5], [-2, 1, 0], [-1, 0, 1]])

class RefInfo():
    def __init__(self, sample_path):
        self.sample_path = sample_path
        self.fs, self.len_sig, self.beat_loc, self.af_starts, self.af_ends, self.class_true = self._load_ref()
        self.endpoints_true = np.dstack((self.af_starts, self.af_ends))[0, :, :]
        # self.endpoints_true = np.concatenate((self.af_starts, self.af_ends), axis=-1)

        if self.class_true == 1 or self.class_true == 2:
            self.onset_score_range, self.offset_score_range = self._gen_endpoint_score_range()
        else:
            self.onset_score_range, self.offset_score_range = None, None

    def _load_ref(self):
        sig, fields = wfdb.rdsamp(self.sample_path)
        ann_ref = wfdb.rdann(self.sample_path, 'atr')

        fs = fields['fs']
        length = len(sig)
        sample_descrip = fields['comments']

        beat_loc = np.array(ann_ref.sample) # r-peak locations
        ann_note = np.array(ann_ref.aux_note) # rhythm change flag

        af_start_scripts = np.where((ann_note=='(AFIB') | (ann_note=='(AFL'))[0]
        af_end_scripts = np.where(ann_note=='(N')[0]

        if 'non atrial fibrillation' in sample_descrip:
            class_true = 0
        elif 'persistent atrial fibrillation' in sample_descrip:
            class_true = 1
        elif 'paroxysmal atrial fibrillation' in sample_descrip:
            class_true = 2
        else:
            print('Error: the recording is out of range!')

            return -1

        return fs, length, beat_loc, af_start_scripts, af_end_scripts, class_true
    
    def _gen_endpoint_score_range(self):
        """

        """
        onset_range = np.zeros((self.len_sig, ),dtype=np.float)
        offset_range = np.zeros((self.len_sig, ),dtype=np.float)
        for i, af_start in enumerate(self.af_starts):
            if self.class_true == 2:
                if max(af_start-1, 0) == 0:
                    onset_range[: self.beat_loc[af_start+2]] += 1
                elif max(af_start-2, 0) == 0:
                    onset_range[self.beat_loc[af_start-1]: self.beat_loc[af_start+2]] += 1
                    onset_range[: self.beat_loc[af_start-1]] += .5
                else:
                    onset_range[self.beat_loc[af_start-1]: self.beat_loc[af_start+2]] += 1
                    onset_range[self.beat_loc[af_start-2]: self.beat_loc[af_start-1]] += .5
                onset_range[self.beat_loc[af_start+2]: self.beat_loc[af_start+3]] += .5
            elif self.class_true == 1:
                onset_range[: self.beat_loc[af_start+2]] += 1
                onset_range[self.beat_loc[af_start+2]: self.beat_loc[af_start+3]] += .5
        for i, af_end in enumerate(self.af_ends):
            if self.class_true == 2:
                if min(af_end+1, len(self.beat_loc)-1) == len(self.beat_loc)-1:
                    offset_range[self.beat_loc[af_end-2]: ] += 1
                elif min(af_end+2, len(self.beat_loc)-1) == len(self.beat_loc)-1:
                    offset_range[self.beat_loc[af_end-2]: self.beat_loc[af_end+1]] += 1
                    offset_range[self.beat_loc[af_end+1]: ] += 0.5
                else:
                    offset_range[self.beat_loc[af_end-2]: self.beat_loc[af_end+1]] += 1
                    offset_range[self.beat_loc[af_end+1]: min(self.beat_loc[af_end+2], self.len_sig-1)] += .5
                offset_range[self.beat_loc[af_end-3]: self.beat_loc[af_end-2]] += .5 
            elif self.class_true == 1:
                offset_range[self.beat_loc[af_end-2]: ] += 1
                offset_range[self.beat_loc[af_end-3]: self.beat_loc[af_end-2]] += .5
        
        return onset_range, offset_range
    
def load_ans(ans_file):
    endpoints_pred = []
    if ans_file.endswith('.json'):
        json_file = open(ans_file, "r")
        ans_dic = json.load(json_file)
        endpoints_pred = np.array(ans_dic['predict_endpoints'])

    elif ans_file.endswith('.mat'):
        ans_struct = sio.loadmat(ans_file)
        endpoints_pred = ans_struct['predict_endpoints']-1

    return endpoints_pred

def ue_calculate(endpoints_pred, endpoints_true, onset_score_range, offset_score_range):
    score = 0
    ma = len(endpoints_true)
    mr = len(endpoints_pred)

    if mr == 0:
        score = 0
    
    else:
        for [start, end] in endpoints_pred:
            score += onset_score_range[int(start)]
            score += offset_score_range[int(end)]
    
    score *= (ma / max(ma, mr))

    return score

def ur_calculate(class_true, class_pred):
    score = R[int(class_true), int(class_pred)]

    return score

def score(data_path, ans_path):
    # AF burden estimation
    SCORE = []

    def is_mat_or_json(file):
        return (file.endswith('.json')) + (file.endswith('.mat'))
    ans_set = filter(is_mat_or_json, os.listdir(ans_path))
    # test_set = open(os.path.join(data_path, 'RECORDS'), 'r').read().splitlines()
    for i, ans_sample in enumerate(ans_set):
        sample_nam = ans_sample.split('.')[0]
        sample_path = os.path.join(data_path, sample_nam)
            
        endpoints_pred = load_ans(os.path.join(ans_path, ans_sample))
        TrueRef = RefInfo(sample_path)

        if len(endpoints_pred) == 0:
            class_pred = 0
        elif len(endpoints_pred) == 1 and np.diff(endpoints_pred)[-1] == TrueRef.len_sig - 1:
            class_pred = 1
        else:
            class_pred = 2

        ur_score = ur_calculate(TrueRef.class_true, class_pred)

        if TrueRef.class_true == 1 or TrueRef.class_true == 2:
            ue_score = ue_calculate(endpoints_pred, TrueRef.endpoints_true, TrueRef.onset_score_range, TrueRef.offset_score_range)
        else:
            ue_score = 0

        u = ur_score + ue_score
        SCORE.append(u)

    score_avg = np.mean(SCORE)

    return score_avg

if __name__ == '__main__':
    TESTSET_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
    score_avg = score(TESTSET_PATH, RESULT_PATH)
    print('AF Endpoints Detection Performance: %0.4f' %score_avg)

    with open(os.path.join(RESULT_PATH, 'score.txt'), 'w') as score_file:
        print('AF Endpoints Detection Performance: %0.4f' %score_avg, file=score_file)

        score_file.close()
