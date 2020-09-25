import os
import json
import logging
import time
import shutil
import torch

from glob import glob
import numpy as np
from scipy.signal import resample_poly

from processor import add_noise
import utils.ECG_IO as ECG_IO


def create_folder(runs_folder, experiment_name):
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)
    this_run_folder = os.path.join(runs_folder, f'{experiment_name}_{time.strftime("%Y.%m.%d_%H-%M-%S")}')

    os.makedirs(this_run_folder)
    os.makedirs(os.path.join(this_run_folder, 'checkpoints'))

    return this_run_folder

def save_checkpoint(state, is_best, dir='.'):
    filename = dir + '/checkpoints/checkpoint.pth'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, dir + '/checkpoints/model_best.pth')

def log_progress(losses_accu):
    log_print_helper(losses_accu, logging.info)


def print_progress(losses_accu):
    log_print_helper(losses_accu, print)


def log_print_helper(losses_accu, log_or_print_func):
    max_len = max([len(loss_name) for loss_name in losses_accu])
    for loss_name, loss_value in losses_accu.items():
        log_or_print_func(loss_name.ljust(max_len + 4) + '{:.4f}'.format(loss_value.avg))


if __name__ == '__main__':
    from multiprocessing import Pool
    # slice_ECG_WithNoise()
    # slice_ECG()
    # test_slice()
    record_dir = "/data_4t/intern1/Documents/AHA2MIT/"
    slice_dirs = ["/home/intern1/Documents/wanlinhong/data/AHA_data/slice10s_em6/",
                 "/home/intern1/Documents/wanlinhong/data/AHA_data/slice10s_em0/",
                 "/home/intern1/Documents/wanlinhong/data/AHA_data/slice10s_em06/",
                 "/home/intern1/Documents/wanlinhong/data/AHA_data/slice10s_em12/"]
    for dir in slice_dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    label_len = 1
    slice_len = 3600
    pad = (label_len - 1) // 2
    record_list = [x.split('/')[-1].replace('.dat', '') for x in glob(record_dir + '/*.dat')]
    print('total files :{}'.format(len(record_list)))

    noise_path = "/data_4t/intern1/Documents/Noise_For_NST/em"
    noise_data = ECG_IO.load_mit_data(noise_path)
    noise = noise_data["signal"]
    db = [-6, 0, 6, 12]

    for db_id in range(len(slice_dirs)):
        def pool_function(record):
            record_path = os.path.join(record_dir, record)
            slice_dir = slice_dirs[db_id]

            data = ECG_IO.load_mit_data(record_path)
            ann = ECG_IO.load_mit_ann(record_path, "atr")

            ann_locs = ann["sample"]
            ann_types = ann["symbol"]
            print(len(ann_types), len(ann_locs))

            signal = data["signal"]
            fs = data["fs"]
            lead_num = data["lead_num"]

            if np.sum(np.isnan(signal)) > 0:
                nan_ls = np.argwhere(np.isnan(signal))
                for id1, id2 in nan_ls:
                    signal[id1, id2] = 0.

            if fs != 360:
                signal_list = []
                for lead_idx in range(lead_num):
                    signal_list.append(resample_poly(signal[:, lead_idx], 360, fs))
                signal = np.stack(signal_list, 1)
                ann_locs = np.rint(ann_locs * (360. / fs)).astype(np.int)

            for i in range(signal.shape[0] // 650000):
                for lead_idx in range(lead_num):
                    signal[650000 * i:650000 * (i + 1), lead_idx] = \
                        add_noise(signal[650000 * i:650000 * (i + 1), lead_idx], noise[:, lead_idx%2], db[db_id])

            remainder = signal.shape[0] - (signal.shape[0] // 650000) * 650000
            if remainder > 0:
                for lead_idx in range(lead_num):
                    signal[signal.shape[0] - remainder:, lead_idx] = \
                        add_noise(signal[signal.shape[0] - remainder:, lead_idx], noise[:remainder, lead_idx%2],
                                  db[db_id])

            ann_ls = encode_ann(ann_locs, ann_types, signal.shape[0], pad=pad)

            for idx in range(len(ann_ls)):
                ann_ls[idx] = ann_ls[idx][ann_locs[0] - 50:ann_locs[-1] + 50]
            signal = signal[ann_locs[0] - 50:ann_locs[-1] + 50]

            data = encode_signal(signal, lead_num, ann_locs, split2lead=False)

            # for j in range(signal.shape[0] // slice_len):
            #     for num in range(len(data_ls)):
            #         data = data_ls[num]
            #         slice_data = data[j * slice_len:(j + 1) * slice_len]
            #
            #         slice_ann = [ann_ls[0][j * slice_len:(j + 1) * slice_len],
            #                      ann_ls[1][j * slice_len:(j + 1) * slice_len],
            #                      ann_ls[2][j * slice_len:(j + 1) * slice_len]]
            #
            #         np.save(os.path.join(slice_dir, record + '_' + str(j) + 'l'+ str(num) + '.npy'), slice_data)
            #
            #         f = open(os.path.join(slice_dir, record + '_' + str(j) + 'l'+ str(num) + '.txt'), 'w')
            #         f.writelines(json.dumps(slice_ann))
            #         f.close()

            for j in range(signal.shape[0] // slice_len):
                slice_data = data[j * slice_len:(j + 1) * slice_len]

                slice_ann = [ann_ls[0][j * slice_len:(j + 1) * slice_len],
                             ann_ls[1][j * slice_len:(j + 1) * slice_len],
                             ann_ls[2][j * slice_len:(j + 1) * slice_len]]

                np.save(os.path.join(slice_dir, record + '_' + str(j) + '.npy'), slice_data)

                f = open(os.path.join(slice_dir, record + '_' + str(j) + '.txt'), 'w')
                f.writelines(json.dumps(slice_ann))
                f.close()

        with Pool(6) as p:
            p.map(pool_function, record_list)