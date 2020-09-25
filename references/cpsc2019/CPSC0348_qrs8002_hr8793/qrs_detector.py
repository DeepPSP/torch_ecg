import numpy as np
from scipy.signal import resample_poly

import torch
import torch.nn.functional as F

from networks import CNN_en1D, seq2seq
from processor import preprocess, postprocess

class qrs_detector(object):
    def __init__(self, model_path, model_type='fcn'):
        assert model_type in ['fcn', 'seq2seq']

        self.model_type = model_type
        self.fs = 500
        self.L = 5000
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if self.model_type=='fcn':
            self.model = CNN_en1D(L=self.L).to(self.device)
        else:
            self.model = seq2seq(device=self.device).to(self.device)

        print('Loading model from {}.'.format(model_path))

        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        # print('Epoch: {}，Se: {}， pre: {}, F1: {}, acc:{}'.
        #       format(checkpoint['epoch'], checkpoint['se'], checkpoint['pre'], checkpoint['f1'],
        #              checkpoint['cpsc_acc']))

    def _detection(self, signal_tensor): # signal_tensor: (B=1) x (C=1) x L
        if type(signal_tensor) is np.ndarray:
            signal_tensor = torch.from_numpy(signal_tensor).view(1, 1, -1).float().to(self.device)

        if self.model_type=='fcn':
            outputs = self.model(signal_tensor)  # outputs: 1 x 1 x beats
            beat_locs2 = list(outputs.squeeze().detach().cpu().numpy())
            beat_locs = postprocess(beat_locs2, margin=12)
        else:
            outputs = self.model.sample(signal_tensor)  # outputs: 1 x 1 x beats
            beat_locs = np.rint(outputs.view(-1).detach().cpu().numpy()).astype(np.int) if outputs \
                                                                                           is not None else np.array([])
        return beat_locs

    def run(self, ecg_signal, fs, lead_list=None, fusion=False):
        if len(ecg_signal.shape)==1:
            ecg_signal = ecg_signal[:, np.newaxis]
        lead_num = ecg_signal.shape[1]

        if np.sum(np.isnan(ecg_signal))>0:
            nan_ls = np.argwhere(np.isnan(ecg_signal))
            for id1, id2 in nan_ls:
                ecg_signal[id1, id2] = 0.

        if fs != self.fs:
            signal_list = []
            for lead_idx in range(lead_num):
                signal_list.append(resample_poly(ecg_signal[:, lead_idx], self.fs, fs)) # resample
            signal = np.stack(signal_list, 1)
        else:
            signal = ecg_signal

        assert signal.shape[0] >= self.L

        for lead_idx in range(lead_num):
            signal[:, lead_idx] = preprocess(signal[:, lead_idx], fs=self.fs) # 滤波

        multilead_beat_locs = [np.array([]) for _ in range(lead_num)]
        slice_num, remain_length = divmod(signal.shape[0], self.L)
        signal_tensor = torch.from_numpy(signal).permute(1, 0).unsqueeze(0).float().to(self.device)

        for slice_idx in range(slice_num): # 切分成10s信号进行处理
            slice_signal = signal_tensor[:, :, slice_idx*self.L:(slice_idx+1)*self.L]
            input_list = torch.split(slice_signal, split_size_or_sections=1, dim=1)

            for idx in range(lead_num):
                beat_locs = self._detection(input_list[idx])
                multilead_beat_locs[idx] = np.append(multilead_beat_locs[idx], beat_locs+slice_idx*self.L)

        if remain_length > 0:
            remain_signal = signal_tensor[:, :, self.L-remain_length: self.L]
            remain_signal = F.pad(remain_signal, ((0, self.L-remain_length)), 'constant', value=0)
            input_list = torch.split(remain_signal, split_size_or_sections=1, dim=1)

            for idx in range(lead_num):
                beat_locs = self._detection(input_list[idx])
                multilead_beat_locs[idx] = np.append(multilead_beat_locs[idx], beat_locs+slice_num*self.L)

        if fs != self.fs:   # 检波结果要转换回原采样率的结果
            for idx in range(lead_num):
                multilead_beat_locs[idx] = np.rint(multilead_beat_locs[idx] / (self.fs / fs)).astype(np.int)

        if lead_num > 1 and fusion:
            fusion_solver = qrs_fusion()
            qrs_locs = fusion_solver.run(ecg_signal, fs, multilead_beat_locs, lead_list)
        else:
            qrs_locs = multilead_beat_locs[0]

        return np.array(qrs_locs), multilead_beat_locs


class qrs_fusion(object):
    def __init__(self):
        pass

    def run(self, signal, fs, beat_loca_list, lead_list):
        pass


if __name__ == '__main__':
    import os
    import sys
    import logging
    from glob import glob
    from multiprocessing import Pool

    import utils.ECG_IO as ECG_IO
    import utils.ECG_calc_beat_label as ECG_calc_beat_label
    import utils.ECG_bxb as ECG_bxb

    from dataload import DS1, DS2

    record_dir = "/data_4t/intern1/Documents/MIT_BIH_Arrhythmia/"
    bxb_window_ts = 0.15
    experiment_name = 'test'
    print_log_func = print #logging.info

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(f'{experiment_name}.log'),
                            logging.StreamHandler(sys.stdout)
                        ])

    model_path = './runs/fcn_finetune_batch_size_32_lr_0.001/checkpoints/model_best.pth'
    detector = qrs_detector(model_path=model_path, model_type='fcn')

    record_list = [x.split('/')[-1].replace('.dat', '') for x in glob(record_dir + '/*.dat')]
    record_list.sort()
    record_list = [it for it in record_list if it in DS1]

    print_log_func('total files :{}'.format(len(record_list)))

    def test(record):
        record_path = os.path.join(record_dir, record)
        data = ECG_IO.load_mit_data(record_path)
        ann = ECG_IO.load_mit_ann(record_path, "atr")

        if ann is not None:
            ann_locs = ann["sample"]
            ann_types = ann["symbol"]
            ref_locs, _ = ECG_calc_beat_label.anns2mitlabel(ann_types, ann_locs)
        else:
            ann_locs = None
            ann_types = None
            ref_locs = None

        signal = data["signal"]
        fs = data["fs"]
        lead_num = data["lead_num"]

        beat_locs, _ = detector.run(signal, fs)
        print_log_func('beat_locs:{}, ann_locs:{}'.format(len(beat_locs), len(ann_locs)))
        bxb_result, _ = ECG_bxb.bxb(beat_locs, ann_locs, ann_types, bxb_window=bxb_window_ts*fs)
        confusion_mat = ECG_bxb.bxb_calc_confusion_matrix(bxb_result)

        se = confusion_mat[0][0] / (confusion_mat[0][0] + confusion_mat[0][1])
        pre = confusion_mat[0][0] / (confusion_mat[0][0] + confusion_mat[1][0])
        f1 = 2*se*pre/(se + pre)
        print_log_func("fname %s se %f pre %f F1 %f" % (record, se, pre, f1))
        return confusion_mat

    confusion_list = []
    for record in record_list:
        confusion_list.append(test(record))

    total_confusion_mat = sum(confusion_list)

    se = total_confusion_mat[0][0] / (total_confusion_mat[0][0] + total_confusion_mat[0][1])
    pre = total_confusion_mat[0][0] / (total_confusion_mat[0][0] + total_confusion_mat[1][0])
    f1 = 2 * se * pre / (se + pre)
    print_log_func("all se %f pre %f F1 %f" % (se, pre, f1))