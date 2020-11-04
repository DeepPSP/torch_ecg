"""
utility functions, most can be found in https://github.com/wenh06/utils
"""
import os, sys, re, logging
import time, datetime
from io import StringIO
from functools import reduce
from copy import deepcopy
from itertools import repeat
from numbers import Real, Number
from typing import Union, Optional, List, Tuple, Dict, Sequence, NoReturn

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from easydict import EasyDict as ED


__all__ = [
    "dict_to_str",
    "str2bool",
    "get_optimal_covering",
    "intervals_union",
    "intervals_intersection",
    "in_interval",
    "in_generalized_interval",
    "plot_single_lead_ecg",
    "class_weight_to_sample_weight",
    "pred_to_indices",
    "get_date_str",
    "mask_to_intervals",
    "list_sum",
    "compute_local_average",
    "gen_gaussian_noise", "gen_sinusoidal_noise", "gen_baseline_wander",
    "get_record_list_recursive3",
    "init_logger",
]


EMPTY_SET = []
Interval = Union[List[Real], Tuple[Real], type(EMPTY_SET)]
GeneralizedInterval = Union[List[Interval], Tuple[Interval], type(EMPTY_SET)]


def intervals_union(interval_list:GeneralizedInterval, join_book_endeds:bool=True) -> GeneralizedInterval:
    """ finished, checked,

    find the union (ordered and non-intersecting) of all the intervals in `interval_list`,
    which is a list of intervals in the form [a,b], where a,b need not be ordered

    Parameters:
    -----------
    interval_list: GeneralizedInterval,
        the list of intervals to calculate their union
    join_book_endeds: bool, default True,
        join the book-ended intervals into one (e.g. [[1,2],[2,3]] into [1,3]) or not
    
    Returns:
    --------
    processed: GeneralizedInterval,
        the union of the intervals in `interval_list`
    """
    interval_sort_key = lambda i: i[0]
    # list_add = lambda list1, list2: list1+list2
    processed = [item for item in interval_list if len(item) > 0]
    for item in processed:
        item.sort()
    processed.sort(key=interval_sort_key)
    # end_points = reduce(list_add, processed)
    merge_flag = True
    while merge_flag:
        merge_flag = False
        new_intervals = []
        if len(processed) == 1:
            return processed
        for idx, interval in enumerate(processed[:-1]):
            this_start, this_end = interval
            next_start, next_end = processed[idx + 1]
            # it is certain that this_start <= next_start
            if this_end < next_start:
                # the case where two consecutive intervals are disjoint
                new_intervals.append([this_start, this_end])
                if idx == len(processed) - 2:
                    new_intervals.append([next_start, next_end])
            elif this_end == next_start:
                # the case where two consecutive intervals are book-ended
                # concatenate if `join_book_endeds` is True, 
                # or one interval degenerates (to a single point)
                if (this_start == this_end or next_start == next_end) or join_book_endeds:
                    new_intervals.append([this_start, max(this_end, next_end)])
                    new_intervals += processed[idx + 2:]
                    merge_flag = True
                    processed = new_intervals
                    break
                else:
                    new_intervals.append([this_start, this_end])
                    if idx == len(processed) - 2:
                        new_intervals.append([next_start, next_end])
            else:
                new_intervals.append([this_start, max(this_end, next_end)])
                new_intervals += processed[idx + 2:]
                merge_flag = True
                processed = new_intervals
                break
        processed = new_intervals
    return processed


def get_optimal_covering(total_interval:Interval, to_cover:list, min_len:Real, split_threshold:Real, traceback:bool=False, **kwargs) -> Tuple[GeneralizedInterval,list]:
    """ finished, checked,

    compute an optimal covering (disjoint union of intervals) that covers `to_cover` such that
    each interval in the covering is of length at least `min_len`,
    and any two intervals in the covering have distance at least `split_threshold`

    Parameters:
    -----------
    total_interval: Interval,
        the total interval that the covering is picked from
    to_cover: list,
        a list of intervals to cover
    min_len: real number,
        minimun length of the intervals of the covering
    split_threshold: real number,
        minumun distance of intervals of the covering
    traceback: bool, default False,
        if True, a list containing the list of indices of the intervals in the original `to_cover`,
        that each interval in the covering covers

    Raises:
    -------
    if any of the intervals in `to_cover` exceeds the range of `total_interval`,
    ValueError will be raised

    Returns:
    --------
    (ret, ret_traceback)
        ret: GeneralizedInterval,
            the covering that satisfies the given conditions
        ret_traceback: list，
            contains the list of indices of the intervals in the original `to_cover`,
            that each interval in the covering covers
    """
    start_time = time.time()
    verbose = kwargs.get('verbose', 0)
    tmp = sorted(total_interval)
    tot_start, tot_end = tmp[0], tmp[-1]

    if verbose >= 1:
        print(f'total_interval = {total_interval}, with_length = {tot_end-tot_start}')

    if tot_end - tot_start < min_len:
        ret = [[tot_start, tot_end]]
        ret_traceback = [list(range(len(to_cover)))] if traceback else []
        return ret, ret_traceback
    to_cover_intervals = []
    for item in to_cover:
        if isinstance(item, list):
            to_cover_intervals.append(item)
        else:
            to_cover_intervals.append(
                [max(tot_start, item-min_len//2), min(tot_end, item+min_len//2)]
            )
    if traceback:
        replica_for_traceback = deepcopy(to_cover_intervals)

    if verbose >= 2:
        print(f'to_cover_intervals after all converted to intervals = {to_cover_intervals}')

        # elif isinstance(item, int):
        #     to_cover_intervals.append([item, item+1])
        # else:
        #     raise ValueError(f"{item} is not an integer or an interval")
    # to_cover_intervals = interval_union(to_cover_intervals)

    for interval in to_cover_intervals:
        interval.sort()
    
    interval_sort_key = lambda i: i[0]
    to_cover_intervals.sort(key=interval_sort_key)

    if verbose >= 2:
        print(f'to_cover_intervals after sorted = {to_cover_intervals}')

    # if to_cover_intervals[0][0] < tot_start or to_cover_intervals[-1][-1] > tot_end:
    #     raise IndexError("some item in to_cover list exceeds the range of total_interval")
    # these cases now seen normal, and treated as follows:
    for item in to_cover_intervals:
        item[0] = max(item[0], tot_start)
        item[-1] = min(item[-1], tot_end)
    # to_cover_intervals = [item for item in to_cover_intervals if item[-1] > item[0]]

    # ensure that the distance from the first interval to `tot_start` is at least `min_len`
    to_cover_intervals[0][-1] = max(to_cover_intervals[0][-1], tot_start + min_len)
    # ensure that the distance from the last interval to `tot_end` is at least `min_len`
    to_cover_intervals[-1][0] = min(to_cover_intervals[-1][0], tot_end - min_len)

    if verbose >= 2:
        print(f'`to_cover_intervals` after two tails adjusted to {to_cover_intervals}')

    # merge intervals whose distances (might be negative) are less than `split_threshold`
    merge_flag = True
    while merge_flag:
        merge_flag = False
        new_intervals = []
        if len(to_cover_intervals) == 1:
            break
        for idx, item in enumerate(to_cover_intervals[:-1]):
            this_start, this_end = item
            next_start, next_end = to_cover_intervals[idx + 1]
            if next_start - this_end >= split_threshold:
                if split_threshold == (next_start - next_end) == 0 or split_threshold == (this_start - this_end) == 0:
                    # the case where split_threshold ==0 and the degenerate case should be dealth with separately
                    new_intervals.append([this_start, max(this_end, next_end)])
                    new_intervals += to_cover_intervals[idx + 2:]
                    merge_flag = True
                    to_cover_intervals = new_intervals
                    break
                else:
                    new_intervals.append([this_start, this_end])
                    if idx == len(to_cover_intervals) - 2:
                        new_intervals.append([next_start, next_end])
            else:
                new_intervals.append([this_start, max(this_end, next_end)])
                new_intervals += to_cover_intervals[idx + 2:]
                merge_flag = True
                to_cover_intervals = new_intervals
                break
    if verbose >= 2:
        print(f'`to_cover_intervals` after merging intervals whose gaps < split_threshold are {to_cover_intervals}')

    # currently, distance between any two intervals in `to_cover_intervals` are larger than `split_threshold`
    # but any interval except the head and tail might has length less than `min_len`
    ret = []
    ret_traceback = []
    if len(to_cover_intervals) == 1:
        # NOTE: here, there's only one `to_cover_intervals`,
        # whose length should be at least `min_len`
        mid_pt = (to_cover_intervals[0][0]+to_cover_intervals[0][-1]) // 2
        half_len = min_len // 2
        if mid_pt - tot_start < half_len:
            ret_start = tot_start
            ret_end = min(tot_end, max(tot_start+min_len, to_cover_intervals[0][-1]))
            ret = [[ret_start, ret_end]]
        else:
            ret_start = max(tot_start, min(to_cover_intervals[0][0], mid_pt-half_len))
            ret_end = min(tot_end, max(mid_pt-half_len+min_len, to_cover_intervals[0][-1]))
            ret = [[ret_start, ret_end]]

    start = min(to_cover_intervals[0][0], to_cover_intervals[0][-1]-min_len)

    for idx, item in enumerate(to_cover_intervals[:-1]):
        # print('item', item)
        this_start, this_end = item
        next_start, next_end = to_cover_intervals[idx + 1]
        potential_end = max(this_end, start + min_len)
        # print(f'start = {start}')
        # print('potential_end', potential_end)
        # if distance from `potential_end` to `next_start` is not enough
        # and has not reached the end of `to_cover_intervals`
        # continue to the next loop
        if next_start - potential_end < split_threshold:
            if idx < len(to_cover_intervals) - 2:
                continue
            else:
                # now, idx==len(to_cover_intervals)-2
                # distance from `next_start` (hence `start`) to `tot_end` is at least `min_len`
                ret.append([start, max(start + min_len, next_end)])
        else:
            ret.append([start, potential_end])
            start = next_start
            if idx == len(to_cover_intervals) - 2:
                ret.append([next_start, max(next_start + min_len, next_end)])
        # print(f'ret = {ret}')
    if traceback:
        for item in ret:
            record = []
            for idx, item_prime in enumerate(replica_for_traceback):
                itc = intervals_intersection([item, item_prime])
                len_itc = itc[-1] - itc[0] if len(itc) > 0 else -1
                if len_itc > 0 or (len_itc == 0 and item_prime[-1] - item_prime[0] == 0):
                    record.append(idx)
            ret_traceback.append(record)
    
    if verbose >= 1:
        print(f'the final result of get_optimal_covering is ret = {ret}, ret_traceback = {ret_traceback}, the whole process used {time.time()-start_time} second(s)')
    
    return ret, ret_traceback


def intervals_intersection(interval_list:GeneralizedInterval, drop_degenerate:bool=True) -> Interval:
    """ finished, checked,

    calculate the intersection of all intervals in interval_list

    Parameters:
    -----------
    interval_list: GeneralizedInterval,
        the list of intervals to yield intersection
    drop_degenerate: bool, default True,
        whether or not drop the degenerate intervals, i.e. intervals with length 0
    
    Returns:
    --------
    its: Interval,
        the intersection of all intervals in `interval_list`
    """
    if [] in interval_list:
        return []
    for item in interval_list:
        item.sort()
    potential_start = max([item[0] for item in interval_list])
    potential_end = min([item[-1] for item in interval_list])
    if (potential_end > potential_start) or (potential_end == potential_start and not drop_degenerate):
        its = [potential_start, potential_end]
    else:
        its = []
    return its


def in_interval(val:Real, interval:Interval, left_closed:bool=True, right_closed:bool=False) -> bool:
    """ finished, checked,

    check whether val is inside interval or not

    Parameters:
    -----------
    val: real number,
    interval: Interval,
    left_closed: bool, default True,
    right_closed: bool, default False,

    Returns:
    --------
    is_in: bool,
    """
    itv = sorted(interval)
    if left_closed:
        is_in = (itv[0] <= val)
    else:
        is_in = (itv[0] < val)
    if right_closed:
        is_in = is_in and (val <= itv[-1])
    else:
        is_in = is_in and (val < itv[-1])
    return is_in


def in_generalized_interval(val:Real, generalized_interval:GeneralizedInterval, left_closed:bool=True, right_closed:bool=False) -> bool:
    """ finished, checked,

    check whether val is inside generalized_interval or not

    Parameters:
    -----------
    val: real number,
    generalized_interval: union of `Interval`s,
    left_closed: bool, default True,
    right_closed: bool, default False,

    Returns:
    --------
    is_in: bool,
    """
    is_in = False
    for interval in generalized_interval:
        if in_interval(val, interval, left_closed, right_closed):
            is_in = True
            break
    return is_in


def plot_single_lead_ecg(s:np.ndarray, fs:Real, use_idx:bool=False, **kwargs) -> NoReturn:
    """ not finished

    single lead ECG plot,

    Parameters:
    -----------
    s: array_like,
        the single lead ECG signal
    fs: real,
        sampling frequency of `s`
    use_idx: bool, default False,
        use idx instead of time for the x-axis
    kwargs: dict,
        keyword arguments, including
        - "waves": Dict[str, np.ndarray], consisting of
            "ppeaks", "qpeaks", "rpeaks", "speaks", "tpeaks",
            "ponsets", "poffsets", "qonsets", "soffsets", "tonsets", "toffsets"

    contributors: Jeethan, and WEN Hao
    """
    if 'plt' not in dir():
        import matplotlib.pyplot as plt
    default_fig_sz = 120
    line_len = fs * 25  # 25 seconds
    nb_lines, residue = divmod(len(s), line_len)
    waves = ED(kwargs.get("waves", ED()))
    if residue > 0:
        nb_lines += 1
    for idx in range(nb_lines):
        idx_start = idx*line_len
        idx_end = min((idx+1)*line_len, len(s))
        c = s[idx_start:idx_end]
        secs = np.arange(idx_start, idx_end)
        if not use_idx:
            secs = secs / fs
        mvs = np.array(c) * 0.001
        fig_sz = int(round(default_fig_sz * (idx_end-idx_start)/line_len))
        fig, ax = plt.subplots(figsize=(fig_sz, 6))
        ax.plot(secs, mvs, c='black')

        ax.axhline(y=0, linestyle='-', linewidth='1.0', color='red')
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.04))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        if waves:
            for w, w_indices in waves.items():
                epoch_w = [wi-idx_start for wi in w_indices if idx_start <= wi < idx_end]
                for wi in epoch_w:
                    ax.axvline(wi, linestyle='dashed', linewidth=0.7, color='magenta')
        ax.set_xlim(secs[0], secs[-1])
        ax.set_ylim(-1.5, 1.5)
        if use_idx:
            plt.xlabel('Samples')
        else:
            plt.xlabel('Time [s]')
        plt.ylabel('Voltage [mV]')
        plt.show()


def class_weight_to_sample_weight(y:np.ndarray, class_weight:Union[str,List[float],np.ndarray,dict]='balanced') -> np.ndarray:
    """ finished, checked,

    transform class weight to sample weight

    Parameters:
    -----------
    y: ndarray,
        the label (class) of each sample
    class_weight: str, or list, or ndarray, or dict, default 'balanced',
        the weight for each sample class,
        if is 'balanced', the class weight will automatically be given by 
        if `y` is of string type, then `class_weight` should be a dict,
        if `y` is of numeric type, and `class_weight` is array_like,
        then the labels (`y`) should be continuous and start from 0
    """
    if not class_weight:
        sample_weight = np.ones_like(y, dtype=float)
        return sample_weight
    
    try:
        sample_weight = y.copy().astype(int)
    except:
        sample_weight = y.copy()
        assert isinstance(class_weight, dict) or class_weight.lower()=='balanced', \
            "if `y` are of type str, then class_weight should be 'balanced' or a dict"
    
    if isinstance(class_weight, str) and class_weight.lower() == 'balanced':
        classes = np.unique(y).tolist()
        cw = compute_class_weight('balanced', classes=classes, y=y)
        trans_func = lambda s: cw[classes.index(s)]
    else:
        trans_func = lambda s: class_weight[s]
    sample_weight = np.vectorize(trans_func)(sample_weight)
    sample_weight = sample_weight / np.max(sample_weight)
    return sample_weight


def pred_to_indices(y_pred:np.ndarray, rpeaks:np.ndarray, class_map:dict) -> Tuple[np.ndarray, np.ndarray]:
    """ finished, checked,

    Parameters:
    -----------
    y_pred: ndarray,
        array of model prediction
    rpeaks: ndarray,
        indices of rpeaks, and of `y_pred` in the corresponding ECG signal
    class_map: dict,
        mapping from classes of string type to int,
        if elements of `y_pred` is of string type, then this mapping will not be used

    Returns:
    --------
    S_pos, V_pos: ndarray,
        indices of SPB, PVC respectively
    """
    classes = ["S", "V"]
    if len(y_pred) == 0:
        S_pos, V_pos = np.array([]), np.array([])
        return S_pos, V_pos
    pred_arr = {}
    if isinstance(y_pred[0], Real):
        for c in classes:
            pred_arr[c] = rpeaks[np.where(y_pred==class_map[c])[0]]
    else:  # of string type
        for c in classes:
            pred_arr[c] = rpeaks[np.where(y_pred==c)[0]]
    S_pos, V_pos = pred_arr["S"], pred_arr["V"]
    return S_pos, V_pos


def dict_to_str(d:Union[dict, list, tuple], current_depth:int=1, indent_spaces:int=4) -> str:
    """ finished, checked,

    convert a (possibly) nested dict into a `str` of json-like formatted form,
    this nested dict might also contain lists or tuples of dict (and of str, int, etc.)

    Parameters:
    -----------
    d: dict, or list, or tuple,
        a (possibly) nested `dict`, or a list of `dict`
    current_depth: int, default 1,
        depth of `d` in the (possible) parent `dict` or `list`
    indent_spaces: int, default 4,
        the indent spaces of each depth

    Returns:
    --------
    s: str,
        the formatted string
    """
    assert isinstance(d, (dict, list, tuple))
    if len(d) == 0:
        s = f"{{}}" if isinstance(d, dict) else f"[]"
        return s
    # flat_types = (Number, bool, str,)
    flat_types = (Number, bool,)
    flat_sep = ", "
    s = "\n"
    unit_indent = " "*indent_spaces
    prefix = unit_indent*current_depth
    if isinstance(d, (list, tuple)):
        if all([isinstance(v, flat_types) for v in d]):
            len_per_line = 110
            current_len = len(prefix) + 1  # + 1 for a comma 
            val = []
            for idx, v in enumerate(d):
                add_v = f"\042{v}\042" if isinstance(v, str) else str(v)
                add_len = len(add_v) + len(flat_sep)
                if current_len + add_len > len_per_line:
                    val = ", ".join([item for item in val])
                    s += f"{prefix}{val},\n"
                    val = [add_v]
                    current_len = len(prefix) + 1 + len(add_v)
                else:
                    val.append(add_v)
                    current_len += add_len
            if len(val) > 0:
                val = ", ".join([item for item in val])
                s += f"{prefix}{val}\n"
        else:
            for v in d:
                if isinstance(v, (dict, list, tuple)):
                    s += f"{prefix}{dict_to_str(v, current_depth+1)}\n"
                else:
                    val = f"\042{v}\042" if isinstance(v, str) else v
                    s += f"{prefix}{val}\n"
    elif isinstance(d, dict):
        for k, v in d.items():
            key = f"\042{k}\042" if isinstance(k, str) else k
            if isinstance(v, (dict, list, tuple)):
                s += f"{prefix}{key}: {dict_to_str(v, current_depth+1)}\n"
            else:
                val = f"\042{v}\042" if isinstance(v, str) else v
                s += f"{prefix}{key}: {val}\n"
    s += unit_indent*(current_depth-1)
    s = f"{{{s}}}" if isinstance(d, dict) else f"[{s}]"
    return s


def str2bool(v:Union[str, bool]) -> bool:
    """ finished, checked,

    converts a 'boolean' value possibly in the format of str to bool

    Parameters:
    -----------
    v: str or bool,
        the 'boolean' value

    Returns:
    --------
    b: bool,
        `v` in the format of bool

    References:
    -----------
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
       b = v
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        b = True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        b = False
    else:
        raise ValueError('Boolean value expected.')
    return b


def get_date_str(fmt:Optional[str]=None):
    """
    """
    now = datetime.datetime.now()
    _fmt = fmt or '%Y-%m-%d-%H-%M-%S'
    ds = now.strftime(_fmt)
    return ds


def mask_to_intervals(mask:np.ndarray, vals:Optional[Union[int,Sequence[int]]]=None) -> Union[list, dict]:
    """ finished, checked,

    Parameters:
    -----------
    mask: ndarray,
        1d mask
    vals: int or sequence of int, optional,
        values in `mask` to obtain intervals

    Returns:
    --------
    intervals: dict or list,
        the intervals corr. to each value in `vals` if `vals` is `None` or `Sequence`;
        or the intervals corr. to `vals` if `vals` is int.
        each interval is of the form `[a,b]`, left inclusive, right exclusive
    """
    if vals is None:
        _vals = list(set(mask))
    elif isinstance(vals, int):
        _vals = [vals]
    else:
        _vals = vals
    # assert set(_vals) & set(mask) == set(_vals)

    intervals = {v:[] for v in _vals}
    for v in _vals:
        valid_inds = np.where(np.array(mask)==v)[0]
        if len(valid_inds) == 0:
            continue
        split_indices = np.where(np.diff(valid_inds)>1)[0]
        split_indices = split_indices.tolist() + (split_indices+1).tolist()
        split_indices = sorted([0] + split_indices + [len(valid_inds)-1])
        for idx in range(len(split_indices)//2):
            intervals[v].append(
                [valid_inds[split_indices[2*idx]], valid_inds[split_indices[2*idx+1]]+1]
            )
    
    if isinstance(vals, int):
        intervals = intervals[vals]

    return intervals


def list_sum(l:Sequence[list]) -> list:
    """ finished, checked,
    """
    return reduce(lambda a,b: a+b, l, [])


def compute_local_average(arr:Union[Sequence,np.ndarray], radius:int) -> np.ndarray:
    """ finished, checked,

    Parameters:
    -----------
    arr: sequence,
        1d array
    radius: int,
        radius for computing average
    
    Returns:
    --------
    res: ndarray,
    """
    _arr = np.array(arr)
    assert _arr.ndim == 1 and radius >= 1
    if radius >= len(_arr) - 1:
        res = np.full(_arr.shape, fill_value=np.mean(_arr))
        return res
    window = 2*radius + 1
    if window >= len(_arr):
        head = np.array([np.mean(_arr[:i+radius+1]) for i in range(radius)])
        tail = np.array([np.mean(_arr[i-radius:]) for i in range(radius,len(_arr))])
        res = np.concatenate((head, tail))
        return res
    body = np.vstack(
        [np.concatenate((np.zeros((i,)), _arr, np.zeros((window-1-i,)))) for i in range(window)]
    )
    body = np.mean(body,axis=0)[2*radius:-2*radius]
    head = np.array([np.mean(_arr[:i+radius+1]) for i in range(radius)])
    tail = np.array([np.mean(_arr[i-2*radius:]) for i in range(radius)])
    res = np.concatenate((head, body, tail))
    return res


def gen_gaussian_noise(siglen:int, mean:Real=0, std:Real=0) -> np.ndarray:
    """ finished, checked,

    generate 1d Gaussian noise of given length, mean, and standard deviation

    Parameters:
    -----------
    siglen: int,
        length of the noise signal
    mean: real number, default 0,
        mean of the noise
    std: real number, default 0,
        standard deviation of the noise

    Returns:
    --------
    gn: ndarray,
        the gaussian noise of given length, mean, and standard deviation
    """
    gn = np.random.normal(mean, std, siglen)
    return gn


def gen_sinusoidal_noise(siglen:int, start_phase:Real, end_phase:Real, amplitude:Real, amplitude_mean:Real=0, amplitude_std:Real=0) -> np.ndarray:
    """ finished, checked,

    generate 1d sinusoidal noise of given length, amplitude, start phase, and end phase

    Parameters:
    -----------
    siglen: int,
        length of the (noise) signal
    start_phase: real number,
        start phase, with units in degrees
    end_phase: real number,
        end phase, with units in degrees
    amplitude: real number,
        amplitude of the sinusoidal curve
    amplitude_mean: real number,
        mean amplitude of an extra Gaussian noise
    amplitude_std: real number, default 0,
        standard deviation of an extra Gaussian noise

    Returns:
    --------
    sn: ndarray,
        the sinusoidal noise of given length, amplitude, start phase, and end phase
    """
    sn = np.linspace(start_phase, end_phase, siglen)
    sn = amplitude * np.sin(np.pi * sn / 180)
    sn += gen_gaussian_noise(siglen, amplitude_mean, amplitude_std)
    return sn


def gen_baseline_wander(siglen:int, fs:Real, bw_fs:Union[Real,Sequence[Real]], amplitude:Union[Real,Sequence[Real]], amplitude_mean:Real=0, amplitude_std:Real=0) -> np.ndarray:
    """ finished, checked,

    generate 1d baseline wander of given length, amplitude, and frequency

    Parameters:
    -----------
    siglen: int,
        length of the (noise) signal
    fs: real number,
        sampling frequency of the original signal
    bw_fs: real number, or list of real numbers,
        frequency (frequencies) of the baseline wander
    amplitude: real number, or list of real numbers,
        amplitude of the baseline wander (corr. to each frequency band)
    amplitude_mean: real number, default 0,
        mean amplitude of an extra Gaussian noise
    amplitude_std: real number, default 0,
        standard deviation of an extra Gaussian noise

    Returns:
    --------
    bw: ndarray,
        the baseline wander of given length, amplitude, frequency

    Example:
    --------
    >>> gen_baseline_wander(4000, 400, [0.4,0.1,0.05], [0.1,0.2,0.4])
    """
    bw = gen_gaussian_noise(siglen, amplitude_mean, amplitude_std)
    if isinstance(bw_fs, Real):
        _bw_fs = [bw_fs]
    else:
        _bw_fs = bw_fs
    if isinstance(amplitude, Real):
        _amplitude = list(repeat(amplitude, len(_bw_fs)))
    else:
        _amplitude = amplitude
    assert len(_bw_fs) == len(_amplitude)
    duration = (siglen / fs)
    for bf, a in zip(_bw_fs, _amplitude):
        start_phase = np.random.randint(0,360)
        end_phase = duration * bf * 360 + start_phase
        bw += gen_sinusoidal_noise(siglen, start_phase, end_phase, a, 0, 0)
    return bw


def get_record_list_recursive3(db_dir:str, rec_patterns:Union[str,Dict[str,str]]) -> Union[List[str], Dict[str, List[str]]]:
    """ finished, checked,

    get the list of records in `db_dir` recursively,
    for example, there are two folders 'patient1', 'patient2' in `db_dir`,
    and there are records 'A0001', 'A0002', ... in 'patient1'; 'B0001', 'B0002', ... in 'patient2',
    then the output would be 'patient1{sep}A0001', ..., 'patient2{sep}B0001', ...,
    sep is determined by the system

    Parameters:
    -----------
    db_dir: str,
        the parent (root) path of the whole database
    rec_patterns: str or dict,
        pattern of the record filenames, e.g. "A(?:\d+).mat",
        or patterns of several subsets, e.g. `{"A": "A(?:\d+).mat"}`

    Returns:
    --------
    res: list of str,
        list of records, in lexicographical order
    """
    if isinstance(rec_patterns, str):
        res = []
    elif isinstance(rec_patterns, dict):
        res = {k:[] for k in rec_patterns.keys()}
    db_dir = os.path.join(db_dir, "tmp").replace("tmp", "")  # make sure `db_dir` ends with a sep
    roots = [db_dir]
    while len(roots) > 0:
        new_roots = []
        for r in roots:
            tmp = [os.path.join(r, item) for item in os.listdir(r)]
            # res += [item for item in tmp if os.path.isfile(item)]
            if isinstance(rec_patterns, str):
                res += list(filter(re.compile(rec_patterns).search, tmp))
            elif isinstance(rec_patterns, dict):
                for k in rec_patterns.keys():
                    res[k] += list(filter(re.compile(rec_patterns[k]).search, tmp))
            new_roots += [item for item in tmp if os.path.isdir(item)]
        roots = deepcopy(new_roots)
    if isinstance(rec_patterns, str):
        res = [os.path.splitext(item)[0].replace(db_dir, "") for item in res]
        res = sorted(res)
    elif isinstance(rec_patterns, dict):
        for k in rec_patterns.keys():
            res[k] = [os.path.splitext(item)[0].replace(db_dir, "") for item in res[k]]
            res[k] = sorted(res[k])
    return res


def init_logger(log_dir:str, log_file:Optional[str]=None, mode:str='a', verbose:int=0) -> logging.Logger:
    """ finished, checked,

    Parameters:
    -----------
    log_dir: str,
        directory of the log file
    log_file: str, optional,
        name of the log file
    mode: str, default 'a',
        mode of writing the log file, can be one of 'a', 'w'
    verbose: int, default 0,
        log verbosity

    Returns:
    --------
    logger: Logger
    """
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = f'log_{get_date_str()}.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    print(f'log file path: {log_file}')

    logger = logging.getLogger('ECG-CRNN')

    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_file)

    if verbose >= 2:
        print("levels of c_handler and f_handler are set DEBUG")
        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    elif verbose >= 1:
        print("level of c_handler is set INFO, level of f_handler is set DEBUG")
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        print("levels of c_handler and f_handler are set WARNING")
        c_handler.setLevel(logging.WARNING)
        f_handler.setLevel(logging.WARNING)
        logger.setLevel(logging.WARNING)

    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


CPSC_STATS = pd.read_csv(StringIO("""rec,AF,len_h,N_beats,V_beats,S_beats,total_beats
A01,No,25.89,109062,0,24,109086
A02,Yes,22.83,98936,4554,0,103490
A03,Yes,24.70,137249,382,0,137631
A04,No,24.51,77812,19024,3466,100302
A05,No,23.57,94614,1,25,9440
A06,No,24.59,77621,0,6,77627
A07,No,23.11,73325,15150,3481,91956
A08,Yes,25.46,115518,2793,0,118311
A09,No,25.84,88229,2,1462,89693
A10,No,23.64,72821,169,9071,82061"""))


# columns truth, rows pred
OFFICIAL_LOSS_DF = pd.read_csv(StringIO(""",N_true,S_true,V_true
N_pred,0,5,5
S_pred,1,0,5
V_pred,1,5,0"""), index_col=0)
OFFICIAL_LOSS_MAT = OFFICIAL_LOSS_DF.values
