import random
from collections import Counter
import numpy as np
import sys
#from DTW import DTW
import FeatureExtract
from sklearn import neighbors
import MyEval
import dill
import pickle
import os.path


def read_expanded(fname):
    '''
    saved array should be read one by one, 
    change manually
    
    example 
    write: 
        fout = open('../../data1/expanded_three_part_window_6000_stride_299.bin', 'wb')
        np.save(fout, a1)
        np.save(fout, a2)
        fout.close()
        print('save done')
    
    read:
        fin = open('../../data1/expanded_three_part_window_6000_stride_299.bin', 'rb')
        a1 = np.load(fin)
        a2 = np.load(fin)
        fout.close()
    
    '''
    
    fin = open(fname, 'rb')
    train_data_out = np.load(fin)
    train_label_out = np.load(fin)
    val_data_out = np.load(fin)
    val_label_out = np.load(fin)
    test_data_out = np.load(fin)
    test_label_out = np.load(fin)
    test_data_pid_out = np.load(fin)
    fout.close()
    
    return train_data_out, train_label_out, val_data_out, val_label_out, test_data_out, test_label_out, test_data_pid_out
    
    
def resample_unequal(ts, length):
    resampled = [0.0] * length
    resampled_idx = list(np.linspace(0.0, len(ts)-1, length))
    for i in range(length):
        idx_i = resampled_idx[i]
        low_idx = int(np.floor(idx_i))
        low_weight = abs(idx_i - np.ceil(idx_i))
        high_idx = int(np.ceil(idx_i))
        high_weight = abs(idx_i - np.floor(idx_i))
        resampled[i] = low_weight * ts[low_idx] + high_weight * ts[high_idx]
#        print(idx_i, resampled[i], low_weight, high_weight)
#        break
    return resampled

def read_centerwave(fin_name = '../../data1/centerwave_raw.csv'):        
    '''
    data format: 
    [pid], [ts vector]\n
    '''
    my_pid = []
    my_data = []

    with open(fin_name) as fin:
        for line in fin:
            content = line.split(',')
            pid = content[0]
            data = [float(i) for i in content[1:]]
                
            my_pid.append(pid)
            my_data.append(data)
        
    print("read_centerwave DONE, data shape: {0}".format(len(my_data)))
    return my_data

def shrink_set_to_seq(pid_list, label):
    '''
    convert set of pids and labels to seq of pids and labels
    
    labels are chars. ['N', 'A', 'O', '~']
    now using vote, if same, choose last one
    '''
    label_char = ['N', 'A', 'O', '~']
    out_label = []
    label = np.array(label)
    pid_list = np.array([ii.split('_')[0] for ii in pid_list])
    pid_set = sorted(list(set([ii.split('_')[0] for ii in pid_list])))
    for pid in pid_set:
        tmp_label = label[pid_list == pid]
        cnt = [0, 0, 0, 0]
        for ii in tmp_label:
            cnt[label_char.index(ii)] += 1
            
        max_num = max(cnt)
        if cnt[2] == max_num and cnt[1] == max_num and cnt[0] == max_num:
            out_label.append('O')
        elif cnt[2] == max_num and cnt[0] == max_num:
            out_label.append('O')
        elif cnt[1] == max_num and cnt[0] == max_num:
            out_label.append('A')
        elif cnt[2] == max_num and cnt[1] == max_num:
            out_label.append('A')
        else:
            out_label.append(label_char[cnt.index(max_num)])
#        print(tmp_label)
#        print(Counter(tmp_label))
#        print(out_label)
#        break
    return list(pid_set), out_label

def read_expanded(fname = '../../data1/expanded.pkl'):
    with open(fname, 'rb') as fin:
        out_data = pickle.load(fin)
        out_label = pickle.load(fin)
    return out_data, out_label
    

def DeleteNoiseData(pid, data, label):
    """
    only retain 3 classes
    """
    pass


def PreProcessData(data, label, max_seq_len):
    """
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    out_data = []
    out_label = []
    out_seqlen = []
    
    my_len = len(data)
    for i in range(my_len):
        line = data[i]
        if len(line) > max_seq_len:
            out_data.append(line[:max_seq_len])
            out_label.append(label[i])
            out_seqlen.append(max_seq_len)
        else:
            append_ts = [0] *( max_seq_len - len(line))
            out_data.append(line+append_ts)
            out_label.append(label[i])
            out_seqlen.append(len(line))
            
    out_label = Label2OneHot(out_label)
    out_data = np.array(out_data, dtype=np.float)
    out_seqlen = np.array(out_seqlen, dtype=np.int64)
        
    return out_data, out_label, out_seqlen


def GenValidation( fin_name, ratio, my_seed ):
    """
    split patient ids into 2 groups, for offline validation
    
    TODO: now using random to split, should use cross validation
    """
    random.seed(my_seed)
    all_pid = []
    train_pid = []
    val_pid = []
    
    fin = open(fin_name)
    for line in fin.readlines():
        content = line.split(',')
        pid = content[0]
        all_pid.append(pid)

        rnd = random.uniform(0.0, 1.0)
        if rnd > ratio:
            train_pid.append(pid)
        else:
            val_pid.append(pid)
            
    fin.close()
    
    print('GenValidation DONE')

    return all_pid, train_pid, val_pid
    

def SplitDataByPid(pid, data, label, train_pid, val_pid):
    '''
    split data based on pid
    
    pid should have same rows with data
    '''
    my_train_data = []
    my_train_label = []
    my_val_data = []
    my_val_label = []

    my_len = len(pid)
    for i in range(my_len):
        tmp_pid = pid[i]
        tmp_data = data[i]
        tmp_label = label[i]
        if tmp_pid in train_pid:
            my_train_data.append(tmp_data)
            my_train_label.append(tmp_label)
        else:
            my_val_data.append(tmp_data)
            my_val_label.append(tmp_label)
            
    print("SplitDataByPid DONE")
    
    return my_train_data, my_train_label, my_val_data, my_val_label

def SplitDataByPid1(pid, data, label, train_pid, val_pid):
    '''
    split data based on pid
    
    also return pid
    
    pid should have same rows with data
    '''
    my_train_data = []
    my_train_label = []
    my_train_pid = []
    my_val_data = []
    my_val_label = []
    my_val_pid = []

    my_len = len(pid)
    for i in range(my_len):
        tmp_pid = pid[i]
        tmp_data = data[i]
        tmp_label = label[i]
        if tmp_pid in train_pid:
            my_train_data.append(tmp_data)
            my_train_label.append(tmp_label)
            my_train_pid.append(tmp_pid)
        else:
            my_val_data.append(tmp_data)
            my_val_label.append(tmp_label)
            my_val_pid.append(tmp_pid)
            
    print("SplitDataByPid DONE")
    
    return my_train_data, my_train_label, my_train_pid, my_val_data, my_val_label, my_val_pid
 

def ReadData( fin_name):
    '''
    data format: 
    [pid], [label], [ts vector]\n
    '''
    my_pid = []
    my_data = []
    my_label = []
    c = 0
    co =0 
    cn = 0
    with open(fin_name) as fin:
        for line in fin:
            content = line.strip().split(',')
            pid = content[0]
            label = content[1]
            # data = [float(i) for i in content[2:]]


            try:
                data = [float(i) for i in content[2:]]
            except:
                print(line)
                break

#            if label == 'A' or label == '~':
#                continue
#            if label == 'N':
#                c += 1
#                if c > 2456:
#                    continue
#                cn += 1
#            if label == 'O':
#                co +=1
                
            my_pid.append(pid)
            my_data.append(data)
            my_label.append(label)
            
#            print(my_data)
#            break
        
    print("Read data DONE")
    print(len(my_data))
#    print (co)
#    print (cn)
    
            
    return my_pid, my_data, my_label


def ReadSmall():
    """
    read a small dataset for quick test
    
    2 classes
    """
    all_pid, train_pid, val_pid = GenValidation( '../../REFERENCE.csv', 0.2, 1 )
    QRS_pid, QRS_data, QRS_label = ReadData( '../../data1/QRSinfo.csv' )
    train_QRS_data, train_QRS_label, val_QRS_data, val_QRS_label = SplitDataByPid(QRS_pid, QRS_data, QRS_label, train_pid, val_pid)
    QRS_train_feature = FeatureExtract.GetQRSFeature(train_QRS_data)
    QRS_val_feature = FeatureExtract.GetQRSFeature(val_QRS_data)
    for ii in range(len(train_QRS_label)):
        if train_QRS_label[ii] != '~':
            train_QRS_label[ii] = 'X'
    for ii in range(len(val_QRS_label)):
        if val_QRS_label[ii] != '~':
            val_QRS_label[ii] = 'X'
    return np.array(QRS_train_feature), train_QRS_label, np.array(QRS_val_feature), val_QRS_label


def ReadTestData( fin_name ):
    '''
    Read data without ID and label

    data format: 
    [pid], [label], [ts vector]\n
    '''
    with open(fin_name) as fin:
        for line in fin:
            content = line.split(',')
            data = [float(i) for i in content]
        
    print("Read data DONE")
            
    return data

def Label2OneHot(labels):
#    label_enum = ['N', 'O']#, 'A', '~']
    label_enum = ['N', 'A', 'O', '~']

    my_len = len(labels)
    out = np.zeros([my_len, len(label_enum)])
    for i in range(my_len):
        label = labels[i]
        out[i, label_enum.index(label)] = 1.0
    
    return out
    
def OneHot2Label(out):
    label_enum = ['N', 'A', 'O', '~']
    labels = []
    my_len = out.shape[0]
    for ii in range(my_len):
        idx = list(out[ii,:]).index(1)
        labels.append(label_enum[idx])
    
    return labels
    
def Index2Label(out):
    label_enum = ['N', 'A', 'O', '~']
    labels = []
    for ii in out:
        labels.append(label_enum[ii])
    
    return labels

def Label2Index(out):
    label_enum = ['N', 'A', 'O', '~']
    labels = []
    for ii in out:
        labels.append(label_enum.index(ii))
    
    return labels

def LabelTo2(labels, pos_label):
    out = []
    for label in labels:
        if label == pos_label:
            out.append(1)
        else:
            out.append(0)
    return out
        
def ProcessDataForNA(data, label):
    out_data = []
    out_label = []
    
    for i in range(len(label)):        
        if label[i] == 'N':
            out_data.append(data[i])
            out_label.append(0)
        elif label[i] == 'A':
            out_data.append(data[i])
            out_label.append(1)
            
    return out_data, out_label
    
def ProcessDataForNoise(data, label):
    out_data = []
    out_label = []
    
    for i in range(len(label)):        
        if label[i] == '~':
            out_data.append(data[i])
            out_label.append(1)
        else:
            out_data.append(data[i])
            out_label.append(0)
            
    return out_data, out_label

def TestFeatureQuality():
    '''
    test if nan inf in features
    '''

    with open('../../data2/features_all_v1.3.pkl', 'rb') as my_input:
        all_pid = dill.load(my_input)
        all_feature = dill.load(my_input)
        all_label = dill.load(my_input)
        
    all_feature = np.array(all_feature)
    print('nan: ', sum(sum(np.isnan(all_feature))))
    print('isposinf: ', sum(sum(np.isposinf(all_feature))))
    print('isneginf: ', sum(sum(np.isneginf(all_feature))))
    
    return 
        
        
if __name__ == '__main__':
    shrink_set_to_seq(test_pid, test_label)


