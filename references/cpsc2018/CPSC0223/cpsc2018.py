import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import random
import argparse
import csv
import glob
import scipy.io as sio
import numpy as np
from keras import layers
from keras import Input
from keras.models import Sequential,Model,load_model,model_from_json
'''
cspc2018_challenge score
Written by:  Xingyao Wang, Feifei Liu, Chengyu Liu
             School of Instrument Science and Engineering
             Southeast University, China
             chengyu@seu.edu.cn
'''

'''
Save prdiction answers to answers.csv in local path, the first column is recording name and the second
column is prediction label, for example:
Recoding    Result
B0001       1
.           .
.           .
.           .
'''

'''
Modified by:    Wenjie Cai
                University of Shanghai for Science and Technology
                Shanghai, China
                wenjiecai@aliyun.com
                2018/9/22
'''
model0 = model_from_json(open('train0905.json').read())
model0.load_weights('train0905.h5')
model1 = model_from_json(open('train0906naa.json').read())
model1.load_weights('train0922naa.h5')
model2 = model_from_json(open('train0906bbb.json').read())
model2.load_weights('train0922bbb.h5')
model3 = model_from_json(open('train0906st.json').read())
model3.load_weights('train0922st.h5')
model4 = model_from_json(open('train0906pa.json').read())
model4.load_weights('train0922pa.h5')
model = model_from_json(open('LSTM0922.json').read())
model.load_weights('LSTM0922.h5')

def tosamples0(ecg):
    if len(ecg[0])<625:
        new = np.concatenate((ecg,np.zeros((12,625-len(ecg[0])))))
    else:
        i = 0
        new = []
        while i+625<len(ecg[0]):
            new.append(ecg[:,i:i+625])
            i += 312
        new.append(ecg[:,-625:])
    samples = np.array(new)
    return samples
def tosamples(ecg):
    if len(ecg[0])<625:
        new = np.concatenate((ecg,np.zeros((12,625-len(ecg[0])))))
    else:
        i = 0
        new = []
        while i+625<len(ecg[0]):
            new.append(ecg[:,i:i+625])
            i += 156
        new.append(ecg[:,-625:])
    samples = np.array(new)
    return samples
def remove_outlier(ecg):
    if (ecg>3).any():
        for i in range(12):
            b = np.argwhere(np.diff(ecg[i])>3)
            if b.shape[0]>0:
                for k in b[:,0]:
                    ecg[i][k+1] = ecg[i][k] 
    if (ecg<-3).any():
        for i in range(12):
            b = np.argwhere(np.diff(ecg[i])<-3)
            if b.shape[0]>0:
                for k in b[:,0]:
                    ecg[i][k+1] = ecg[i][k]
    return ecg
def remove_noise(ecg):
    le = ecg.shape[1]
    for j in range(12):
        noise = []
        b = np.argwhere(np.abs(ecg[j])>2.5)
        b = b[:,0]
        c = np.diff(b)
        count = 0
        pn = 0
        for k in range(len(c)):
            if c[k]<2:
                count += 1
                if count>=8:
                    noise.append(b[k+1])
            elif c[k]>1 and c[k]<8:
                count = 0
                pn += 1
                if pn>1:
                    noise.append(b[k+1])
            elif c[k]<25:
                count = 0
                pn = 0
                noise.append(b[k+1])
            else:
                count = 0
                pn = 0
        if len(noise)>0:
            pre = -1
            for l in range(len(noise)):
                if pre>=0 and noise[l]-pre<200:
                    be = noise[l-1]
                else:
                    be = max(0,noise[l]-60)
                en = min(le,noise[l]+60)
                pre = noise[l]
                ecg[j][be:en] = 0
    return ecg
def getsamples0(ecg):
    length = ecg.shape[1]
    ecg = ecg[:,:int(length//4*4)]
    ecg = ecg.reshape(ecg.shape[0],-1,4)
    ecg = np.average(ecg,axis=2)
    ecg = remove_outlier(ecg)
    ecg = remove_noise(ecg)
    samples = tosamples0(ecg)
    return samples.reshape(-1,12,625,1)
def getsamples(ecg):
    length = ecg.shape[1]
    ecg = ecg[:,:int(length//4*4)]
    ecg = ecg.reshape(ecg.shape[0],-1,4)
    ecg = np.average(ecg,axis=2)
    ecg = remove_outlier(ecg)
    ecg = remove_noise(ecg)
    samples = tosamples(ecg)
    return samples.reshape(-1,12,625,1)

def cpsc2018(record_base_path):
    # ecg = scipy.io.loadmat(record_path)
    ###########################INFERENCE PART################################

    ## Please process the ecg data, and output the classification result.
    ## result should be an integer number in [1, 9].
    with open('answers.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        # column name
        writer.writerow(['Recording', 'Result'])
        for mat_item in os.listdir(record_base_path):
            if mat_item.endswith('.mat') and (not mat_item.startswith('._')):
                data = sio.loadmat(record_base_path+mat_item)
                ecg = data['ECG'][0][0][2]
                ecg0 = ecg.copy()
                samples0 = getsamples0(ecg0)
                samples = getsamples(ecg)
                result0 = model0.predict(samples0)
                result1 = model1.predict(samples)
                result2 = model2.predict(samples)
                result3 = model3.predict(samples)
                result4 = model4.predict(samples)
                results = np.hstack((result1,result2,result3,result4))
                l = len(results)
                if l<45:
                    z = np.zeros((45-l,16))
                    results = np.vstack((z,results))
                else:
                    results = results[:45,:]
                result = model.predict(results.reshape(1,45,16))[0]
                result = np.argmax(result)+1
                p = np.max(result0,axis=0)
                r = np.argmax(p)
                if (r==5 or r==6) and (p[1]>0.999 or p[2]>0.999 or p[3]>0.999 or p[4]>0.999 or p[7]>0.999 or p[8]>0.999):
                    p[5] = 0
                    p[6] = 0
                    r = np.argmax(p[1:])+1
                if r==8:
                    result = 9
                ## If the classification result is an invalid number, the result will be determined as normal(1).
                if result > 9 or result < 1 or not(str(result).isdigit()):
                    result = 1
                record_name = mat_item.rstrip('.mat')
                answer = [record_name, result]
                # write result
                writer.writerow(answer)

        csvfile.close()

    ###########################INFERENCE PART################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--recording_path',
                        help='path saving test record file')

    args = parser.parse_args()

    result = cpsc2018(record_base_path=args.recording_path)
