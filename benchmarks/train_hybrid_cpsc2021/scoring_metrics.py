#!/usr/bin/env python3

import numpy as np
import json
import os
import sys
from typing import Union, Optional, List, Tuple, Sequence, NoReturn

import scipy.io as sio
import wfdb


__all__ = [
    "compute_challenge_metric",
    "gen_endpoint_score_mask", "gen_endpoint_score_range",
]


###########################################
# methods from the file score_2021.py 
# of the official repository
###########################################

"""
Written by:  Xingyao Wang, Chengyu Liu
             School of Instrument Science and Engineering
             Southeast University, China
             chengyu@seu.edu.cn
"""

R = np.array([[1, -1, -.5], [-2, 1, 0], [-1, 0, 1]])  # scoring matrix for classification

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
    
    def _gen_endpoint_score_range(self, verbose=0):
        """

        """
        onset_range = np.zeros((self.len_sig, ),dtype=np.float)
        offset_range = np.zeros((self.len_sig, ),dtype=np.float)
        for i, af_start in enumerate(self.af_starts):
            if self.class_true == 2:
                if max(af_start-1, 0) == 0:
                    onset_range[: self.beat_loc[af_start+2]] += 1
                    if verbose > 0:
                        print(f"official --- onset (c_ind, score 1): 0 --- {af_start+2}")
                        print(f"official --- onset (sample, score 1): 0 --- {self.beat_loc[af_start+2]}")
                elif max(af_start-2, 0) == 0:
                    onset_range[self.beat_loc[af_start-1]: self.beat_loc[af_start+2]] += 1
                    if verbose > 0:
                        print(f"official --- onset (c_ind, score 1): {af_start-1} --- {af_start+2}")
                        print(f"official --- onset (sample, score 1): {self.beat_loc[af_start-1]} --- {self.beat_loc[af_start+2]}")
                    onset_range[: self.beat_loc[af_start-1]] += .5
                    if verbose > 0:
                        print(f"official --- onset (c_ind, score 0.5): 0 --- {af_start-1}")
                        print(f"official --- onset (sample, score 0.5): 0 --- {self.beat_loc[af_start-1]}")
                else:
                    onset_range[self.beat_loc[af_start-1]: self.beat_loc[af_start+2]] += 1
                    if verbose > 0:
                        print(f"official --- onset (c_ind, score 1): {af_start-1} --- {af_start+2}")
                        print(f"official --- onset (sample, score 1): {self.beat_loc[af_start-1]} --- {self.beat_loc[af_start+2]}")
                    onset_range[self.beat_loc[af_start-2]: self.beat_loc[af_start-1]] += .5
                    if verbose > 0:
                        print(f"official --- onset (c_ind, score 0.5): {af_start-2} --- {af_start-1}")
                        print(f"official --- onset (sample, score 0.5): {self.beat_loc[af_start-2]} --- {self.beat_loc[af_start-1]}")
                onset_range[self.beat_loc[af_start+2]: self.beat_loc[af_start+3]] += .5
                if verbose > 0:
                    print(f"official --- onset (c_ind, score 0.5): {af_start+2} --- {af_start+3}")
                    print(f"official --- onset (sample, score 0.5): {self.beat_loc[af_start+2]} --- {self.beat_loc[af_start+3]}")
            elif self.class_true == 1:
                onset_range[: self.beat_loc[af_start+2]] += 1
                if verbose > 0:
                    print(f"official --- onset (c_ind, score 1): 0 --- {af_start+2}")
                    print(f"official --- onset (sample, score 1): 0 --- {self.beat_loc[af_start+2]}")
                onset_range[self.beat_loc[af_start+2]: self.beat_loc[af_start+3]] += .5
                if verbose > 0:
                    print(f"official --- onset (c_ind, score 0.5): {af_start+2} --- {af_start+3}")
                    print(f"official --- onset (sample, score 0.5): {self.beat_loc[af_start+2]} --- {self.beat_loc[af_start+3]}")
        for i, af_end in enumerate(self.af_ends):
            if self.class_true == 2:
                if min(af_end+1, len(self.beat_loc)-1) == len(self.beat_loc)-1:
                    offset_range[self.beat_loc[af_end-2]: ] += 1
                    if verbose > 0:
                        print(f"official --- offset (c_ind, score 1): {af_end-2} --- -1")
                        print(f"official --- offset (sample, score 1): {self.beat_loc[af_end-2]} --- -1")
                elif min(af_end+2, len(self.beat_loc)-1) == len(self.beat_loc)-1:
                    offset_range[self.beat_loc[af_end-2]: self.beat_loc[af_end+1]] += 1
                    if verbose > 0:
                        print(f"official --- offset (c_ind, score 1): {af_end-2} --- {af_end+1}")
                        print(f"official --- offset (sample, score 1): {self.beat_loc[af_end-2]} --- {self.beat_loc[af_end+1]}")
                    offset_range[self.beat_loc[af_end+1]: ] += 0.5
                    if verbose > 0:
                        print(f"official --- offset (c_ind, score 0.5): {af_end+1} --- -1")
                        print(f"official --- offset (sample, score 0.5): {self.beat_loc[af_end+1]} --- -1")
                else:
                    offset_range[self.beat_loc[af_end-2]: self.beat_loc[af_end+1]] += 1
                    if verbose > 0:
                        print(f"official --- offset (c_ind, score 1): {af_end-2} --- {af_end+1}")
                        print(f"official --- offset (sample, score 1): {self.beat_loc[af_end-2]} --- {self.beat_loc[af_end+1]}")
                    offset_range[self.beat_loc[af_end+1]: min(self.beat_loc[af_end+2], self.len_sig-1)] += .5
                    if verbose > 0:
                        print(f"official --- offset (c_ind, score 0.5): {af_end+1} --- -1")
                        print(f"official --- offset (sample, score 0.5): {self.beat_loc[af_end+1]} --- {min(self.beat_loc[af_end+2], self.len_sig-1)}")
                offset_range[self.beat_loc[af_end-3]: self.beat_loc[af_end-2]] += .5
                if verbose > 0:
                    print(f"official --- offset (c_ind, score 0.5): {af_end-3} --- {af_end-2}")
                    print(f"official --- offset (sample, score 0.5): {self.beat_loc[af_end-3]} --- {self.beat_loc[af_end-2]}")
            elif self.class_true == 1:
                offset_range[self.beat_loc[af_end-2]: ] += 1
                if verbose > 0:
                    print(f"official --- offset (c_ind, score 1): {af_end-2} --- -1")
                    print(f"official --- offset (sample, score 1): {self.beat_loc[af_end-2]} --- -1")
                offset_range[self.beat_loc[af_end-3]: self.beat_loc[af_end-2]] += .5
                if verbose > 0:
                    print(f"official --- offset (c_ind, score 0.5): {af_end-3} --- {af_end-2}")
                    print(f"official --- offset (sample, score 0.5): {self.beat_loc[af_end-3]} --- {self.beat_loc[af_end-2]}")
        
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



###################################################################
# custom metric computing function
###################################################################

def compute_challenge_metric(class_true:int,
                             class_pred:int,
                             endpoints_true:Sequence[Sequence[int]],
                             endpoints_pred:Sequence[Sequence[int]],
                             onset_score_range:Sequence[float],
                             offset_score_range:Sequence[float]) -> float:
    """ finished, checked,

    compute challenge metric for a single record

    Parameters
    ----------
    class_true: int,
        labelled for the record
    class_pred: int,
        predicted class for the record
    endpoints_true: sequence of intervals,
        labelled intervals of AF episodes
    endpoints_pred: sequence of intervals,
        predicted intervals of AF episodes
    onset_score_range: sequence of float,
        scoring mask for the AF onset predictions
    offset_score_range: sequence of float,
        scoring mask for the AF offset predictions

    Returns
    -------
    u: float,
        the final score for the prediction
    """
    ur_score = ur_calculate(class_true, class_pred)
    ue_score = ue_calculate(endpoints_pred, endpoints_true, onset_score_range, offset_score_range)
    u = ur_score + ue_score
    return u


def gen_endpoint_score_mask(siglen:int,
                            critical_points:Sequence[int],
                            af_intervals:Sequence[Sequence[int]],
                            bias:dict={1:1, 2:0.5},
                            verbose:int=0) -> Tuple[np.ndarray, np.ndarray]:
    """ finished, checked,

    generate the scoring mask for the onsets and offsets of af episodes,

    Parameters
    ----------
    siglen: int,
        length of the signal
    critical_points: sequence of int,
        locations (indices in the signal) of the critical points,
        including R peaks, rhythm annotations, etc,
        which are stored in the `sample` fields of an wfdb annotation file
        (corr. beat ann, rhythm ann are in the `symbol`, `aux_note` fields)
    af_intervals: sequence of intervals,
        intervals of the af episodes in terms of indices in `critical_points`
    bias: dict, default {1:1, 2:0.5},
        keys are bias (with ±) in terms of number of rpeaks
        values are corresponding scores
    verbose: int, default 0,
        log verbosity

    Returns
    -------
    (onset_score_mask, offset_score_mask): 2-tuple of ndarray,
        scoring mask for the onset and offsets predictions of af episodes

    NOTE
    ----
    1. the onsets in `af_intervals` are 0.15s ahead of the corresponding R peaks,
    while the offsets in `af_intervals` are 0.15s behind the corresponding R peaks.
    2. for records [data_39_4,data_48_4,data_68_23,data_98_5,data_101_5,data_101_7,data_101_8,data_104_25,data_104_27],
    the official `RefInfo._gen_endpoint_score_range` slightly expands the scoring intervals at heads or tails of the records,
    which strictly is incorrect as defined in the `Scoring` section of the official webpage (http://www.icbeb.org/CPSC2021)
    """
    _critical_points = list(critical_points)
    if 0 not in _critical_points:
        _critical_points.insert(0, 0)
        _af_intervals = [[itv[0]+1, itv[1]+1] for itv in af_intervals]
        if verbose >= 2:
            print(f"0 added to _critical_points, len(_critical_points): {len(_critical_points)-1} ==> {len(_critical_points)}")
    else:
        _af_intervals = [[itv[0], itv[1]] for itv in af_intervals]
    # records with AFf mostly have `_critical_points` ending with `siglen-1`
    # but in some rare case ending with `siglen`
    if siglen-1 in _critical_points:
        _critical_points[-1] = siglen
        if verbose >= 2:
            print(f"in _critical_points siglen-1 (={siglen-1}) changed to siglen (={siglen})")
    elif siglen in _critical_points:
        pass
    else:
        _critical_points.append(siglen)
        if verbose >= 2:
            print(f"siglen (={siglen}) appended to _critical_points, len(_critical_points): {len(_critical_points)-1} ==> {len(_critical_points)}")
    onset_score_mask, offset_score_mask = np.zeros((siglen,)), np.zeros((siglen,))
    for b, v in bias.items():
        mask_onset, mask_offset = np.zeros((siglen,)), np.zeros((siglen,))
        for itv in _af_intervals:
            onset_start = _critical_points[max(0, itv[0]-b)]
            # note that the onsets and offsets in `_af_intervals` already occupy positions in `_critical_points`
            onset_end = _critical_points[min(itv[0]+1+b, len(_critical_points)-1)]
            if verbose > 0:
                print(f"custom --- onset (c_ind, score {v}): {max(0, itv[0]-b)} --- {min(itv[0]+1+b, len(_critical_points)-1)}")
                print(f"custom --- onset (sample, score {v}): {_critical_points[max(0, itv[0]-b)]} --- {_critical_points[min(itv[0]+1+b, len(_critical_points)-1)]}")
            mask_onset[onset_start: onset_end] = v
            # note that the onsets and offsets in `af_intervals` already occupy positions in `_critical_points`
            offset_start = _critical_points[max(0, itv[1]-1-b)]
            offset_end = _critical_points[min(itv[1]+b, len(_critical_points)-1)]
            if verbose > 0:
                print(f"custom --- offset (c_ind, score {v}): {max(0, itv[1]-1-b)} --- {min(itv[1]+b, len(_critical_points)-1)}")
                print(f"custom --- offset (sample, score {v}): {_critical_points[max(0, itv[1]-1-b)]} --- {_critical_points[min(itv[1]+b, len(_critical_points)-1)]}")
            mask_offset[offset_start: offset_end] = v
        onset_score_mask = np.maximum(onset_score_mask, mask_onset)
        offset_score_mask = np.maximum(offset_score_mask, mask_offset)
    return onset_score_mask, offset_score_mask


gen_endpoint_score_range = gen_endpoint_score_mask  # alias



if __name__ == '__main__':
    TESTSET_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
    score_avg = score(TESTSET_PATH, RESULT_PATH)
    print('AF Endpoints Detection Performance: %0.4f' %score_avg)

    with open(os.path.join(RESULT_PATH, 'score.txt'), 'w') as score_file:
        print('AF Endpoints Detection Performance: %0.4f' %score_avg, file=score_file)

        score_file.close()
