import glob
import numpy as np
import os
import re

import scipy.io as sio

from .CPSC2020_challenge import *


FS = 400
THR = 0.15
DATA_PATH = '../data/'
REF_PATH = '../ref/'

def load_ans():
    """
    Function for loading the detection results and references
    Input:
    Ouput:
        S_refs: position references for S
        V_refs: position references for V
        S_results: position results for S
        V_results: position results for V
    """
    def is_mat(l):
        return l.endswith('.mat')

    data_files = list(filter(is_mat, os.listdir(DATA_PATH)))

    S_refs = []
    V_refs = []
    S_results = []
    V_results = []
    for i, data_file in enumerate(data_files):
        index = re.split('[.]', data_file)[0][1: ]
        data_file = os.path.join(DATA_PATH, data_file)
        ref_file = os.path.join(REF_PATH, 'R{}.mat'.format(index))
        # load ecg file
        ecg_data = sio.loadmat(data_file)['ecg'].squeeze()
        # load answer
        s_ref = sio.loadmat(ref_file)['ref']['S_ref'][0, 0].flatten()
        v_ref = sio.loadmat(ref_file)['ref']['V_ref'][0, 0].flatten()
        # process ecg and conduct event detection using your algorithm
        s_pos, v_pos = CPSC2020_challenge(ecg_data, FS)
        S_refs.append(s_ref)
        V_refs.append(v_ref)
        S_results.append(s_pos)
        V_results.append(v_pos)

    return S_refs, V_refs, S_results, V_results

def CPSC2020_score(S_refs, V_refs, S_results, V_results):
    """
    Score Function
    Input:
        S_refs, V_refs, S_results, V_results
    Output:
        Score1: score for S
        Score2: score for V
    """
    s_score = np.zeros([len(S_refs), ])
    v_score = np.zeros([len(S_refs), ])
    ## Scoring ##
    for i, s_ref in enumerate(S_refs):
        v_ref = V_refs[i]
        s_pos = S_results[i]
        v_pos = V_results[i]
        s_tp = 0
        s_fp = 0
        s_fn = 0
        v_tp = 0
        v_fp = 0
        v_fn = 0
        if s_ref.size == 0:
            s_fp = len(s_pos)
        else:
            for m, ans in enumerate(s_ref):
                s_pos_cand = np.where(abs(s_pos-ans) <= THR*FS)[0]
                if s_pos_cand.size == 0:
                    s_fn += 1
                else:
                    s_tp += 1
        s_fp += (len(s_pos) - s_tp)

        if v_ref.size == 0:
            v_fp = len(v_pos)
        else:
            for m, ans in enumerate(v_ref):
                v_pos_cand = np.where(abs(v_pos-ans) <= THR*FS)[0]
                if v_pos_cand.size == 0:
                    v_fn += 1
                else:
                    v_tp += 1
        v_fp += (len(v_pos) - v_tp)

        # calculate the score
        s_score[i] = s_fp * (-1) + s_fn * (-5)
        v_score[i] = v_fp * (-1) + v_fn * (-5)
    Score1 = np.sum(s_score)
    Score2 = np.sum(v_score)

    return Score1, Score2

if __name__ == '__main__':
    S_refs, V_refs, S_results, V_results = load_ans()
    S1, S2 = CPSC2020_score(S_refs, V_refs, S_results, V_results)

    print ("S_score: {}".format(S1))
    print ("V_score: {}".format(S2))

    with open('score.txt', 'w') as score_file:
        print('S_score: {}'.format(S1), file=score_file)
        print('V_score: {}'.format(S2), file=score_file)

        score_file.close()
