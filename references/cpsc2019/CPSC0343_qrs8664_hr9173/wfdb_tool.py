import wfdb
from wfdb import processing
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from sig_tool import resample, resample_idx
import shutil
import os
from sig_tool import interpolate

from concurr_tool import MultiTask
def detect_qrs(sig, fs):
    # qrs_idx = processing.xqrs_detect(sig, fs) # using xqrs algorithm
    qrs_idx = processing.qrs.gqrs_detect(sig, fs) # using gqrs algorithm
    return qrs_idx

"""
format annotations
"""
def format_anno(wfdb_data, wfdb_label):
    return None

def _gen_default_dbdir():
    return os.path.join(os.getcwd(), 'wfdb')

def dl_all_db(db_dir=None):
    if not db_dir:
        db_dir = _gen_default_dbdir()

    dbs = wfdb.get_dbs()
    for db in dbs:
        db_name = db[0]
        print('downloading db: ', db[0], ' ', db[1])
        wfdb.dl_database(db_name+'/', dl_dir=os.path.join(db_dir, db_name))


def dl_dbs(dbs, db_dir=None):
    if not db_dir:
        db_dir = _gen_default_dbdir()

    for db in dbs:
        wfdb.dl_database(db+'/', dl_dir=os.path.join(db_dir, db))


def load_database(database, db_dir):

    dir = os.path.join(db_dir, database)

    file_list = []
    for root, dirs, files in os.walk(dir):
        [file_list.append(f) for f in files]

    dat_list = [a.split('.')[0] for a in filter(lambda x:x.split('.')[1]=='dat', file_list)]
    hea_list = [a.split('.')[0] for a in filter(lambda x:x.split('.')[1]=='hea', file_list)]
    atr_list = [a.split('.')[0] for a in filter(lambda x:x.split('.')[1]=='atr', file_list)]

    record_list = [dat for dat in filter(lambda x:x in hea_list and x in atr_list, dat_list)]

    name_list = []
    data_list = []
    anno_list = []
    anno_typ_list = []
    for rec in record_list:
        print('loading data', rec)
        data = wfdb.rdrecord(os.path.join(dir, rec))
        anno = wfdb.rdann(os.path.join(dir, rec), 'atr')

        sig = data.p_signal
        anno_idx = [s for s in anno.sample]
        anno_typ = [s for s in anno.symbol]

        name_list.append(rec)
        data_list.append(sig)

        anno_list.append(anno_idx)
        anno_typ_list.append(anno_typ)


    return name_list, data_list, anno_list, anno_typ_list

def delete_non_beat(symbol, sample):
    """删除非心拍的标注符号"""
    AAMI_MIT_MAP = {'N': 'Nfe/jnBLR',#将19类信号分为五大类
                    'S': 'SAJa',
                    'V': 'VEr',
                    'F': 'F',
                    'Q': 'Q?'}
    MIT2AAMI = {c: k for k in AAMI_MIT_MAP.keys() for c in AAMI_MIT_MAP[k]}
    mit_beat_codes = list(MIT2AAMI.keys())
    symbol = np.array(symbol)#symbol对应标签,sample为R峰所在位置，sig为R峰值
    isin = np.isin(symbol, mit_beat_codes)
    symbol = symbol[isin]#去除19类之外的非心拍标注信号
    symbol = np.array([MIT2AAMI[x] for x in symbol])#将19类信号划分为五类
    return symbol, np.copy(sample[isin])#sample对应R峰采样点位置，利用R峰值判断AAMI五种类型

def is_anno_beat(anno_symb):
    AAMI_MIT_MAP = {'N': 'Nfe/jnBLR',#将19类信号分为五大类
                    'S': 'SAJa',
                    'V': 'VEr',
                    'F': 'F',
                    'Q': 'Q?'}
    MIT2AAMI = {c: k for k in AAMI_MIT_MAP.keys() for c in AAMI_MIT_MAP[k]}
    mit_beat_codes = list(MIT2AAMI.keys())
    if anno_symb in mit_beat_codes:
        return True
    else:
        return False

def load_mitdb(db_dir='wfdb', database='mitdb'):
    (name_list, data_list, anno_list, anno_typ_list) = load_database(database, db_dir)

    ''' resample before splitting'''
    print('start resampling')
    cnt = 0

    multiTask = MultiTask(pool_size=40, queue_size=5000)
    for d in data_list:
        multiTask.submit(cnt, resample, (d, 360, 500, 'linear'))
        cnt += 1
    rs_data = [d for d in multiTask.subscribe()]

    rs_anno = [resample_idx(anno, 360, 500) for anno in anno_list]
    rs_anno_typ = [typ for typ in anno_typ_list]

    return name_list, rs_data, rs_anno, rs_anno_typ


def load_aha(db_dir='wfdb', database='aha'):
    dir = os.path.join(db_dir, database)
    file_list = []
    for root, dirs, files in os.walk(dir):
        [file_list.append(f) for f in files]

    dat_list = [a.split('.')[0] for a in filter(lambda x:x.split('.')[1] == 'dat', file_list)]
    hea_list = [a.split('.')[0] for a in filter(lambda x:x.split('.')[1] == 'hea', file_list)]
    atr_list = [a.split('.')[0] for a in filter(lambda x:x.split('.')[1] == 'atr', file_list)]

    record_list = [dat for dat in filter(lambda x:x in hea_list and x in atr_list, dat_list)]

    name_list = []
    data_list = []
    anno_list = []
    anno_typ_list = []

    for rec in record_list:
        print('loading data', rec)
        data = wfdb.rdrecord(os.path.join(dir, rec))
        anno = wfdb.rdann(os.path.join(dir, rec), 'atr')

        sig = data.p_signal
        '''aha: local peaks to trace the R peak'''

        '''nan handling by interpolation'''
        idx_start = -1
        idx_stop = -1
        for idx in range(len(sig)):
            if np.isnan(sig[idx,0]):
                if idx_start < 0:
                    idx_start = idx - 1
            else:
                if idx_start >= 0:
                    idx_stop = idx

            if idx_start >= 0 and idx_stop >= 0:
                print('ch0 interpolating from ', idx_start, ' to ', idx_stop)
                sig_interp = interpolate(sig[idx_start,0], sig[idx_stop,0], idx_stop-idx_start+1)
                for idx in range(len(sig_interp)):
                    sig[idx_start+idx,0] = sig_interp[idx]
                idx_start = -1
                idx_stop = -1

        idx_start = -1
        idx_stop = -1
        for idx in range(len(sig)):
            if np.isnan(sig[idx,1]):
                if idx_start < 0:
                    idx_start = idx - 1
            else:
                if idx_start >= 0:
                    idx_stop = idx

            if idx_start >= 0 and idx_stop >= 0:
                print('ch1 interpolating from ', idx_start, ' to ', idx_stop)
                sig_interp = interpolate(sig[idx_start,1], sig[idx_stop,1], idx_stop-idx_start+1)
                for idx in range(len(sig_interp)):
                    sig[idx_start+idx,1] = sig_interp[idx]

                idx_start = -1
                idx_stop = -1

        peaks = processing.find_local_peaks(sig[:,0], 10)

        idx2 = 0
        anno_r = []
        anno_typ = []
        for idx in range(len(anno.sample)):
            idx_start = idx2
            for idx_peak in peaks[idx_start:]:
                idx2 += 1
                '''aha: find the nearest R peak after the QRS onset'''
                if idx_peak > anno.sample[idx]:
                    anno_r.append(idx_peak)
                    anno_typ.append(anno.symbol[idx])
                    break

        anno_r_idx = [r for r in anno_r]

        anno_typ_idx = [typ for typ in anno_typ]

        #trunc data
        sig = sig[anno_r[0]-200:]
        anno_r_idx = [r-anno_r[0] for r in anno_r_idx]

        data_list.append(sig)
        anno_list.append(anno_r_idx)
        anno_typ_list.append(anno_typ_idx)

        name_list.append(rec)

    ''' resample before splitting'''
    print('start resampling')
    cnt = 0
    multiTask = MultiTask(pool_size=40, queue_size=5000)
    for d in data_list:
        multiTask.submit(cnt, resample, (d, 250, 500, 'linear'))
        cnt += 1
    rs_data = [d for d in multiTask.subscribe()]

    rs_anno = [resample_idx(anno, 250, 500) for anno in anno_list]
    rs_anno_typ = [typ for typ in anno_typ_list]

    return name_list, rs_data, rs_anno, rs_anno_typ
