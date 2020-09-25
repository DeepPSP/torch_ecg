from sklearn.preprocessing import StandardScaler, scale
from keras.preprocessing import sequence
import unet
import scipy.io as sio
import os
import numpy as np
import h5py
import math

def QRS_score(ref_Rpeaks, detected, thres):
    fn = 0
    for r in ref_Rpeaks:
        miss = np.sum(np.logical_and(detected>=(r-thres), detected<=(r+thres)).astype(int))
        if miss == 0:
            fn = fn+1
            
    tp = len(ref_Rpeaks)-fn        
    fp = len(detected) - tp
    
    if fp+fn == 0:
        score = 1
    elif fn == 0 and fp == 1:
        score = 0.7
    elif fn == 1 and fp == 0:
        score = 0.3
    else:
        score = 0
        
    return score


def QRS_score_new(r_ref, r_ans, thr_, fs_):

    FN = 0
    FP = 0
    TP = 0
    for j in range(len(r_ref)):
        loc = np.where(np.abs(r_ans - r_ref[j]) <= thr_*fs_)[0]
        if j == 0:
            err = np.where((r_ans >= 0.5*fs_) & (r_ans <= r_ref[j] - thr_*fs_))[0]
        elif j == len(r_ref)-1:
            err = np.where((r_ans >= r_ref[j]+thr_*fs_) & (r_ans <= 9.5*fs_))[0]
        
        #if j < len(r_ref)-1:   
        else:
            err = np.where((r_ans >= r_ref[j]+thr_*fs_) & (r_ans <= r_ref[j+1]-thr_*fs_))[0]

        FP = FP + len(err)
        if len(loc) >= 1:
            TP += 1
            FP = FP + len(loc) - 1
        elif len(loc) == 0:
            FN += 1

    if FN + FP > 1:
        score = 0
    elif FN == 1 and FP == 0:
        score = 0.3
    elif FN == 0 and FP == 1:
        score = 0.7
    else:
        score = 1
        
    return score


def  QRS_score_official(r_ref, r_ans, thr_, fs_):

    FN = 0
    FP = 0
    TP = 0

    r_ref = r_ref[(r_ref >= 0.5*fs_) & (r_ref <= 9.5*fs_)]
    for j in range(len(r_ref)):
        loc = np.where(np.abs(r_ans - r_ref[j]) <= thr_*fs_)[0]
        if j == 0:
            err = np.where((r_ans >= 0.5*fs_ +thr_*fs_) & (r_ans <= r_ref[j] - thr_*fs_))[0]
        elif j == len(r_ref)-1:
            err = np.where((r_ans >= r_ref[j]+thr_*fs_) & (r_ans <= 9.5*fs_ - thr_*fs_))[0]
        else:
            err = np.where((r_ans >= r_ref[j]+thr_*fs_) & (r_ans <= r_ref[j+1] - thr_*fs_))[0]

        FP = FP + len(err)
        if len(loc) >= 1:
            TP += 1
            FP = FP + len(loc) - 1
        elif len(loc) == 0:
            FN += 1

    if FN + FP > 1:
        score = 0
    elif FN == 1 and FP == 0:
        score = 0.3
    elif FN == 0 and FP == 1:
        score = 0.7
    else:
        score = 1

    return score


def findpeaks(data, spacing=72, limit=None):
    """
    Janko Slavic peak detection algorithm and implementation.
    https://github.com/jankoslavic/py-tools/tree/master/findpeaks
    Finds peaks in `data` which are of `spacing` width and >=`limit`.
    :param ndarray data: data
    :param float spacing: minimum spacing to the next peak (should be 1 or more)
    :param float limit: peaks should have value greater or equal
    :return array: detected peaks indexes array
    """
    len = data.size
    x = np.zeros(int(len + 2 * spacing))
    x[:int(spacing)] = data[0] - 1.e-6
    x[int(-spacing):] = data[-1] - 1.e-6
    x[int(spacing):int(spacing) + len] = data
    peak_candidate = np.zeros(len)
    peak_candidate[:] = True
    for s in range(int(spacing)):
        start = spacing - s - 1
        h_b = x[start: start + len]  # before
        start = spacing
        h_c = x[start: start + len]  # central
        start = spacing + s + 1
        h_a = x[start: start + len]  # after
        peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c >= h_b, h_c > h_a))

    ind = np.argwhere(peak_candidate)
    ind = ind.reshape(ind.size)
    if limit is not None:
        ind = ind[data[ind] > limit]
    else:
        limit = np.mean(data[ind])/2
        ind = ind[data[ind] > limit]

    return ind

def find_peaks_PT(pred_array, thres, spacing=100):

    peaks = findpeaks(pred_array, spacing=spacing, limit=thres)
    # peaks = list(peaks)
    # average_RR = np.mean(np.diff(peaks))
    # for i in range(len(peaks)-1):
    #     if (peaks[i+1] - peaks[i]) > 1.5*average_RR:
    #         region_begin = peaks[i]-20 if peaks[i]-20>=0 else 0
    #         region_end =peaks[i+1]+20 if peaks[i+1]+20<len(pred_array) else len(pred_array)
    #         # print(region_begin)
    #         # print(region_end)
    #         FN_peaks = findpeaks(pred_array[int(region_begin):int(region_end)], spacing=90, limit=0.5*thres)
    #         FN_peaks = int(region_begin) + FN_peaks
    #         peaks.extend(FN_peaks[1:-1])        
    # peaks.sort()
    # peaks = np.array(peaks, dtype=np.float32)

    return peaks

def find_peaks(pred_array, thres):
#     fig1, ax1 = plt.subplots()
#     ax1.plot(pred_array)
#     plt.show()
    candidate_regions = (pred_array.squeeze()>thres).astype(int)
    padded_candidate_regions = np.pad(candidate_regions, (1,1), 'constant', constant_values=(0, 0))
    padded_candidate_regions_diff = np.diff(padded_candidate_regions)
#     print(np.unique(padded_candidate_regions_diff))
    regions_begins = np.where(padded_candidate_regions_diff==1)[0]-1
#     print('regions_begins:', regions_begins)
    regions_ends = np.where(padded_candidate_regions_diff==-1)[0]-1
#     print('regions_ends:', regions_ends)
    peaks = np.rint((regions_ends+regions_begins)/2)
    return peaks

def score_model2(model, input_dir, ref_dir, thres):
    #recordpaths = glob.glob(os.path.join(input_dir, '*.mat'))
    fs = 500
    total_score = 0
    for i in range(1600,2000):
        record_name = str(int(i+1)).zfill(5)

        # load data
        data_mat = sio.loadmat(os.path.join(input_dir, 'data_'+record_name+'.mat'))
        data = data_mat['ecg']
        data = np.array(data, dtype=np.float32)        
        # normalize the data
        #data = scale(data).astype(np.float32)
        # normalize the data
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        signal_length = data.shape[0]
        # pad data to 16's times   
    #     pad_length = 16 - signal_length%16
    #     data = np.pad(data.squeeze(), (0, pad_length), 'edge')
        # truncate data to 16's times
        truncated_length = signal_length - signal_length%16
        data = data.squeeze()[0:truncated_length]

        # predict
        data = np.expand_dims(data,0)
        data = np.expand_dims(data,-1)
        pred_array = model.predict(data).squeeze()
    #     pred_array = pred_array.squeeze()[0:signal_length]
        pred_peaks = find_peaks(pred_array, thres)
        pred_peaks = pred_peaks[np.logical_and(pred_peaks>(0.5*fs-0.075*fs), pred_peaks<(9.5*fs+0.075*fs))]
    #     print('pred_peaks', pred_peaks)

        # load reference
        ref_mat = sio.loadmat(os.path.join(ref_dir, 'R_'+record_name+'.mat'))
        R_peaks = ref_mat['R_peak']
        R_peaks = np.array(R_peaks, dtype=np.int32).squeeze()
        R_peaks = R_peaks[np.logical_and(R_peaks>(0.5*fs-0.075*fs), R_peaks<(9.5*fs+0.075*fs))]
    #     print('R_peaks', R_peaks)

        score = QRS_score(R_peaks, pred_peaks, 0.075*fs)
        print('%d : %f' % (i+1, score))
        total_score = total_score+score

    mean_score = total_score/400
    print(mean_score)
    return mean_score

def score_model_2channel(model, input_dir, ref_dir, thres):
    #recordpaths = glob.glob(os.path.join(input_dir, '*.mat'))
    fs = 500
    total_score = 0
    for i in range(1600,2000):
        record_name = str(int(i+1)).zfill(5)

        # load data
        data_mat = sio.loadmat(os.path.join(input_dir, 'data_'+record_name+'.mat'))
        data = data_mat['ecg']
        data = np.array(data, dtype=np.float32)        
        # normalize the data
        #data = scale(data).astype(np.float32)
        # normalize the data
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        signal_length = data.shape[0]
        # pad data to 16's times   
    #     pad_length = 16 - signal_length%16
    #     data = np.pad(data.squeeze(), (0, pad_length), 'edge')
        # truncate data to 16's times
        truncated_length = signal_length - signal_length%16
        data = data.squeeze()[0:truncated_length]

        # predict
        data = np.expand_dims(data,0)
        data = np.expand_dims(data,-1)
        data_rev = -1*data
        data = np.concatenate((data, data_rev), axis=-1)
        pred_array = model.predict(data).squeeze()
    #     pred_array = pred_array.squeeze()[0:signal_length]
        pred_peaks = find_peaks(pred_array, thres)
        pred_peaks = pred_peaks[np.logical_and(pred_peaks>(0.5*fs-0.075*fs), pred_peaks<(9.5*fs+0.075*fs))]
    #     print('pred_peaks', pred_peaks)

        # load reference
        ref_mat = sio.loadmat(os.path.join(ref_dir, 'R_'+record_name+'.mat'))
        R_peaks = ref_mat['R_peak']
        R_peaks = np.array(R_peaks, dtype=np.int32).squeeze()
        R_peaks = R_peaks[np.logical_and(R_peaks>(0.5*fs-0.075*fs), R_peaks<(9.5*fs+0.075*fs))]
    #     print('R_peaks', R_peaks)

        score = QRS_score(R_peaks, pred_peaks, 0.075*fs)
        print('%d : %f' % (i+1, score))
        total_score = total_score+score

    mean_score = total_score/400
    print(mean_score)
    return mean_score

def score_model(model, X, Y, thres, verbose=False):
    #recordpaths = glob.glob(os.path.join(input_dir, '*.mat'))
    fs = 500
    total_score = 0
    signal_length = X.shape[1]
    # truncate data to 16's times
    truncated_length = signal_length - signal_length%16
    X = X[:, 0:truncated_length]

    Y_pred = model.predict(X).squeeze()

    for i in range(X.shape[0]):
        
        pred_peaks = find_peaks_PT(Y_pred[i], thres)
        # pred_peaks = pred_peaks[np.logical_and(pred_peaks>(0.5*fs-0.075*fs), pred_peaks<(9.5*fs+0.075*fs))]
    #     print('pred_peaks', pred_peaks)

        R_peaks = Y[i].squeeze()
        # R_peaks = R_peaks[np.logical_and(R_peaks>(0.5*fs-0.075*fs), R_peaks<(9.5*fs+0.075*fs))]
    #     print('R_peaks', R_peaks)

        score = QRS_score_new(R_peaks, pred_peaks, 0.075, 500)
        if verbose:
            print('%d : %f' % (i+1, score))
        total_score = total_score+score

    mean_score = total_score/X.shape[0]
    print(mean_score)
    return mean_score



def score_model_new(model, X, Y, thres, verbose=False, seg_length=2000, step_length=500, padding_length=2048, fs=500):
    #recordpaths = glob.glob(os.path.join(input_dir, '*.mat')) 
    total_score = 0
    signal_length = X.shape[1]

    for i in range(X.shape[0]):

        x_seg = [] 

        for j in range(0, signal_length-seg_length+1, step_length):
            x_seg.append(X[i,j:j+seg_length])

        x_seg = np.array(x_seg, dtype = np.float32)

        x_seg = sequence.pad_sequences(x_seg, maxlen=padding_length, dtype='float32', padding='post')

        y_pred = model.predict(x_seg).squeeze()

        whole_pred = np.zeros((X.shape[1],), dtype=np.float32)
        for k in range(0,10,1):
            original_pred = []
            for seg_id in range(k-3, k+1):
                if seg_id >= 0 and seg_id < len(x_seg):
                    original_pred.append(y_pred[seg_id, int((k-seg_id)*fs):int((k-seg_id+1)*fs)])

            original_pred = np.array(original_pred, dtype=np.float32)
            whole_pred[int(k*fs): int((k+1)*fs)] = original_pred.mean(axis=0).squeeze()

        pred_peaks = find_peaks_PT(whole_pred, thres, spacing=int(fs*0.2))
        # pred_peaks = pred_peaks[np.logical_and(pred_peaks>(0.5*fs-0.075*fs), pred_peaks<(9.5*fs+0.075*fs))]
    #     print('pred_peaks', pred_peaks)

        R_peaks = Y[i].squeeze()
        # R_peaks = R_peaks[np.logical_and(R_peaks>(0.5*fs-0.075*fs), R_peaks<(9.5*fs+0.075*fs))]
    #     print('R_peaks', R_peaks)

        score = QRS_score_official(R_peaks, pred_peaks, 0.075, fs)
        if verbose:
            print('%d : %f' % (i+1, score))
        total_score = total_score+score

    mean_score = total_score/X.shape[0]
    print(mean_score)
    return mean_score