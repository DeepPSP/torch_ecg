from torch.utils.data import Dataset
import torch
import numpy as np
import scipy.signal as ss
from scipy.io import loadmat


class QRS_dataset(Dataset):
    def __init__(self,data_name):
        self.data_lst = open(data_name,'r').readlines()

    def __getitem__(self, idx):
        name = self.data_lst[idx].split()[0]
        sig = loadmat('./train/data/'+name)['ecg'].squeeze()
        ref = loadmat('./train/ref/R_' + name[-9:])['R_peak']

        signal = ss.medfilt(sig, 3)
        lowpass = ss.butter(2, 40.0 / (500 / 2.0), 'low')  # 40-45都可以，解决工频干扰
        signal_bp = ss.filtfilt(*lowpass, x=signal)
        lowpass = ss.butter(2, 2.0 / (500 / 2.0), 'low')  # 1.5-2.5都可以，计算基线
        baseline = ss.filtfilt(*lowpass, x=signal_bp)
        signal_bp = signal_bp - baseline
        sig = self.trans_data(signal_bp)

        ref = (ref.astype('float') / 5000).squeeze()

        filled_target = np.zeros(38)
        filled_target[range(len(ref))[:38]] = ref[:38]

        return sig.unsqueeze(0),filled_target

    def __len__(self):
        return len(self.data_lst)

    def trans_data(self, v):
        v = np.array((v - v.mean()) / (v.std()))
        v = v / (v.max())
        return torch.FloatTensor(v)



class QRS_val_dataset(Dataset):
    def __init__(self,data_name):
        self.data_lst = open(data_name,'r').readlines()

    def __getitem__(self, idx):
        name = self.data_lst[idx].split()[0]
        sig = loadmat('./train/data/'+name)['ecg'].squeeze()
        signal = ss.medfilt(sig, 3)
        lowpass = ss.butter(2, 40.0 / (500 / 2.0), 'low')  # 40-45都可以，解决工频干扰
        signal_bp = ss.filtfilt(*lowpass, x=signal)
        lowpass = ss.butter(2, 2.0 / (500 / 2.0), 'low')  # 1.5-2.5都可以，计算基线
        baseline = ss.filtfilt(*lowpass, x=signal_bp)
        signal_bp = signal_bp - baseline
        sig = self.trans_data(signal_bp)


        return sig.unsqueeze(0)

    def __len__(self):
        return len(self.data_lst)

    def trans_data(self, v):
        v = np.array((v - v.mean()) / (v.std()))
        v = v / (v.max())
        return torch.FloatTensor(v)

class QRS_val_dataset_mul(Dataset):
    def __init__(self,data_name):
        self.data_lst = open(data_name,'r').readlines()

    def __getitem__(self, idx):
        name = self.data_lst[idx].split()[0]
        sig = loadmat('./train/data/'+name)['ecg'].squeeze()
        # signal = ss.medfilt(sig, 3)
        # lowpass = ss.butter(2, 40.0 / (500 / 2.0), 'low')  # 40-45都可以，解决工频干扰
        # signal_bp = ss.filtfilt(*lowpass, x=signal)
        # lowpass = ss.butter(2, 2.0 / (500 / 2.0), 'low')  # 1.5-2.5都可以，计算基线
        # baseline = ss.filtfilt(*lowpass, x=signal_bp)
        # signal_bp = signal_bp - baseline
        sig_all_bp = self.filt(sig)
        sig_all_bp = self.trans_data(sig_all_bp)

        sig_fh = sig[:2500]
        sig_fh = ss.resample(sig_fh,5000)
        sig_fh_bp = self.filt(sig_fh)
        sig_fh_bp = self.trans_data(sig_fh_bp)

        sig_sh = sig[2500:]
        sig_sh = ss.resample(sig_sh,5000)
        sig_sh_bp = self.filt(sig_sh)
        sig_sh_bp = self.trans_data(sig_sh_bp)



        return (sig_all_bp.unsqueeze(0),sig_fh_bp.unsqueeze(0),sig_sh_bp.unsqueeze(0))

    def __len__(self):
        return len(self.data_lst)

    def trans_data(self, v):
        v = np.array((v - v.mean()) / (v.std()))
        v = v / (v.max())
        return torch.FloatTensor(v)

    def filt(self,sig):
        signal = ss.medfilt(sig, 3)
        lowpass = ss.butter(2, 40.0 / (500 / 2.0), 'low')  # 40-45都可以，解决工频干扰
        signal_bp = ss.filtfilt(*lowpass, x=signal)
        lowpass = ss.butter(2, 2.0 / (500 / 2.0), 'low')  # 1.5-2.5都可以，计算基线
        baseline = ss.filtfilt(*lowpass, x=signal_bp)
        signal_bp = signal_bp - baseline
        return signal_bp




