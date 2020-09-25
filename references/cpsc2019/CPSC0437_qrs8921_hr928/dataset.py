from keras.utils import plot_model, np_utils
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
import unet_lstm
import scipy.io as sio
import os
import numpy as np
import h5py
import glob
from sklearn.utils import shuffle
from operator import itemgetter 
import smooth
import pywt
import math
from matplotlib import pyplot as plt
from scipy.signal import resample

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def smooth_ref(signal_length, peaks, sigmoid):
    new_ref = np.zeros(signal_length)
    half_window_size = 120
    x_values = np.arange(-half_window_size, half_window_size)
    peak_distribution = gaussian(x_values, 0, sigmoid)
    for p in peaks: 
        if p >= half_window_size and p+half_window_size<signal_length:   
            new_ref[p-half_window_size:p+half_window_size] = peak_distribution
        elif p >= half_window_size:
            new_ref[p-half_window_size:] = peak_distribution[0:half_window_size+signal_length-p]
        else:
            new_ref[0:p+half_window_size] = peak_distribution[half_window_size-p:int(half_window_size*2)]
        
    return new_ref


def binary_ref(signal_length, peaks, one_length):
    new_ref = np.zeros(signal_length)
    half_window_size = int(one_length/2)
    for p in peaks: 
        if p >= half_window_size and p+half_window_size<signal_length:   
            new_ref[p-half_window_size:p-half_window_size+one_length] = 1
        elif p >= half_window_size:
            new_ref[p-half_window_size:] = 1
        else:
            new_ref[0:p+half_window_size] = 1
        
    return new_ref


# load data
def loaddata(input_dir, ref_dir, records_num):
    X = []
    Y = []
    QRS = []

    #recordpaths = glob.glob(os.path.join(input_dir, '*.mat'))
    for i in range(records_num):

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
        
        # load reference
        ref_mat = sio.loadmat(os.path.join(ref_dir, 'R_'+record_name+'.mat'))
        R_peaks = ref_mat['R_peak']
        R_peaks = np.array(R_peaks, dtype=np.int32)
        ref = smooth_ref(len(data), R_peaks, sigmoid=5)
        ref = np.expand_dims(ref, axis=-1)
        
        X.append(data)
        Y.append(ref)
        QRS.append(R_peaks)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    
    # remove duplicate rows
    X,indexes = np.unique(X,  return_index=True, axis=0)
    print(X.shape)
    Y = Y[indexes]
    QRS = itemgetter(*indexes)(QRS)
    
    # shuffle 
    X, Y, QRS = shuffle(X, Y, QRS, random_state=0) 
    
    return X,Y,QRS


# load data
def loaddata2(input_dir, ref_dir, sigmoid=5, do_normalize=True, denoise=False, fs = 360):
    X = []
    Y = []
    QRS = []
    

    datapaths = glob.glob(os.path.join(input_dir, '*.mat'))
    datapaths.sort()
    for data_path in datapaths:

        #print(data_path)
        
        # load data
        data_mat = sio.loadmat(data_path)
        data = data_mat['ecg']
        data = np.array(data, dtype=np.float32)   
#         print(data.shape)
        
        
        for i in range(data.shape[1]):
            smoothed_signal = smooth.smooth(data[:, i], window_len=int(fs), window='flat')
            data[:, i] = data[:, i] - smoothed_signal

        # denoise ECG
        for i in range(data.shape[1]):
            # DWT
            coeffs = pywt.wavedec(data[:, i], 'db4', level=3)
            # compute threshold
            noiseSigma = 0.01
            threshold = noiseSigma* math.sqrt(2 * math.log2(data[:, i].size))
            # apply threshold
            newcoeffs = coeffs
            for j in range(len(newcoeffs)):
                newcoeffs[j] = pywt.threshold(coeffs[j], threshold, mode='soft')

            # IDWT
            data[:, i] = pywt.waverec(newcoeffs, 'db4')[0:len(data)]
            
        # normalize the data
        # data = scale(data).astype(np.float32)

        # normalize the data
        if do_normalize:
            scaler = StandardScaler()
            scaler.fit(data)
            data = scaler.transform(data).astype(np.float32)
        
        X.append(data)

    X = np.array(X, dtype=np.float32)
    print('X.shape', X.shape)

    refpaths = glob.glob(os.path.join(ref_dir, '*.mat'))
    refpaths.sort()
    for ref_path in refpaths:
        
        # load reference
        ref_mat = sio.loadmat(ref_path)
        R_peaks = ref_mat['R_peak']
        R_peaks = np.array(R_peaks, dtype=np.int32).squeeze()
        ref = smooth_ref(len(data), R_peaks, sigmoid=sigmoid)
        ref = np.expand_dims(ref, axis=-1)  
        Y.append(ref)
        QRS.append(R_peaks)

    
    Y = np.array(Y, dtype=np.float32)

    print('Y.shape', Y.shape)
    
    # remove duplicate rows
    #X,indexes = np.unique(X,  return_index=True, axis=0)
    #print(X.shape)
    # Y = Y[indexes]
    # QRS = itemgetter(*indexes)(QRS)
    
    # shuffle 
#     X, Y, QRS = shuffle(X, Y, QRS, random_state=0) 
    
    return X,Y,QRS


def loaddata2_wavelet(input_dir, ref_dir, sigmoid=5, signal_length=2500, do_normalize=False):
    X = []
    Y = []
    QRS = []

    datapaths = glob.glob(os.path.join(input_dir, '*.mat'))
    datapaths.sort()
    for data_path in datapaths:

        # print(data_path)
        
        # load data
        data_mat = sio.loadmat(data_path)
        data = data_mat['ecg']
        data = np.array(data, dtype=np.float32).squeeze()  
        data_length = len(data)
        # normalize the data
        # data = scale(data).astype(np.float32)

        # normalize the data
        # scaler = StandardScaler()
        # scaler.fit(data)
        # data = scaler.transform(data).astype(np.float32)
        data = pywt.wavedec(data, 'db4', level=5)
        data_new = np.zeros((signal_length, len(data)), dtype=np.float32)
        for i in range(len(data)):
            data_new[:, i] = resample(data[i], signal_length)
        
        if do_normalize:
            data_new = scale(data_new, axis=0)
        X.append(data_new)

    X = np.array(X, dtype=np.float32)
    print('X.shape', X.shape)

    refpaths = glob.glob(os.path.join(ref_dir, '*.mat'))
    refpaths.sort()
    for ref_path in refpaths:
        
        # load reference
        ref_mat = sio.loadmat(ref_path)
        R_peaks = ref_mat['R_peak']
        R_peaks = np.array(R_peaks, dtype=np.int32).squeeze()
        ref = smooth_ref(data_length, R_peaks, sigmoid=sigmoid)
        ref = resample(ref, signal_length)
        ref = np.expand_dims(ref, axis=-1)  
        Y.append(ref)
        QRS.append(R_peaks)

    
    Y = np.array(Y, dtype=np.float32)

    print('Y.shape', Y.shape)
    
    # remove duplicate rows
    #X,indexes = np.unique(X,  return_index=True, axis=0)
    #print(X.shape)
    # Y = Y[indexes]
    # QRS = itemgetter(*indexes)(QRS)
    
    # shuffle 
    X, Y, QRS = shuffle(X, Y, QRS, random_state=0) 
    
    return X,Y,QRS




# load data
def loaddata_binary_label(input_dir, ref_dir, label_length=5):
    X = []
    Y = []
    QRS = []

    datapaths = glob.glob(os.path.join(input_dir, '*.mat'))
    datapaths.sort()
    for data_path in datapaths:

        # print(data_path)
        
        # load data
        data_mat = sio.loadmat(data_path)
        data = data_mat['ecg']
        data = np.array(data, dtype=np.float32)      
        # normalize the data
        # data = scale(data).astype(np.float32)
        # normalize the data
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data).astype(np.float32)
        X.append(data)

    X = np.array(X, dtype=np.float32)
    print('X.shape', X.shape)

    refpaths = glob.glob(os.path.join(ref_dir, '*.mat'))
    refpaths.sort()
    for ref_path in refpaths:
        
        # load reference
        ref_mat = sio.loadmat(ref_path)
        R_peaks = ref_mat['R_peak']
        R_peaks = np.array(R_peaks, dtype=np.int32).squeeze()
        ref = binary_ref(len(data), R_peaks, one_length=label_length)
        ref = np.expand_dims(ref, axis=-1)  
        Y.append(ref)
        QRS.append(R_peaks)

    
    Y = np.array(Y, dtype=np.float32)

    print('Y.shape', Y.shape)
    
    # remove duplicate rows
    #X,indexes = np.unique(X,  return_index=True, axis=0)
    #print(X.shape)
    # Y = Y[indexes]
    # QRS = itemgetter(*indexes)(QRS)
    
    # shuffle 
    X, Y, QRS = shuffle(X, Y, QRS, random_state=0) 
    
    return X,Y,QRS


def loaddata_denoise(input_dir, ref_dir, signal_length=5000, sigmoid=5, fs=320):
    X = []
    Y = []
    QRS = []

    datapaths = glob.glob(os.path.join(input_dir, '*.mat'))
    datapaths.sort()
    for data_path in datapaths:

        # print(data_path)
        
        # load data
        data_mat = sio.loadmat(data_path)
        data = data_mat['ecg']
        data = np.array(data, dtype=np.float32)      
        signal = data
        # plt.subplot(2,1,1)
        # plt.plot(signal[0:2000])

        # normalize the data
        # data = scale(data).astype(np.float32)
        # normalize the data
        for i in range(signal.shape[1]):
            smoothed_signal = smooth.smooth(signal[:, i], window_len=int(fs*0.886), window='flat')
            signal[:, i] = signal[:, i] - smoothed_signal

        # denoise ECG
        for i in range(signal.shape[1]):
            # DWT
            coeffs = pywt.wavedec(signal[:, i], 'db4', level=3)
            # compute threshold
            noiseSigma = 0.01
            threshold = noiseSigma* math.sqrt(2 * math.log2(signal[:, i].size))
            # apply threshold
            newcoeffs = coeffs
            for j in range(len(newcoeffs)):
                newcoeffs[j] = pywt.threshold(coeffs[j], threshold, mode='soft')

            # IDWT
            signal[:, i] = pywt.waverec(newcoeffs, 'db4')[0:len(signal)]

        # resample
        signal_new = np.zeros((signal_length, signal.shape[1]))
        for i in range(signal.shape[1]):
            signal_new[:, i] = resample(signal[:, i], signal_length)

        signal = signal_new

        # scaler = StandardScaler()
        # scaler.fit(signal)
        # signal = scaler.transform(signal).astype(np.float32)
        # plt.subplot(2,1,2)
        # plt.plot(signal[0:2000])
        # plt.show()
        X.append(signal)

    X = np.array(X, dtype=np.float32)
    print('X.shape', X.shape)

    refpaths = glob.glob(os.path.join(ref_dir, '*.mat'))
    refpaths.sort()
    for ref_path in refpaths:
        
        # load reference
        ref_mat = sio.loadmat(ref_path)
        R_peaks = ref_mat['R_peak']
        R_peaks = np.array(R_peaks, dtype=np.int32).squeeze()
        R_peaks = (R_peaks*(signal_length/len(data))).astype(np.int32)
        ref = smooth_ref(signal_length, R_peaks, sigmoid=sigmoid)
        ref = np.expand_dims(ref, axis=-1)  
        Y.append(ref)
        QRS.append(R_peaks)

    Y = np.array(Y, dtype=np.float32)

    print('Y.shape', Y.shape)

    # remove duplicate rows
    #X,indexes = np.unique(X,  return_index=True, axis=0)
    #print(X.shape)
    # Y = Y[indexes]
    # QRS = itemgetter(*indexes)(QRS)

    # shuffle 
    X, Y, QRS = shuffle(X, Y, QRS, random_state=0) 

    return X,Y,QRS