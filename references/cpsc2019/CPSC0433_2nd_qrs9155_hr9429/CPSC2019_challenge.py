import numpy as np
import numpy
import scipy.io
import keras
import tensorflow as tf
from pylab import *
from keras.layers import SimpleRNN,Permute,Reshape,CuDNNLSTM,LSTM,Dropout,Input, Add, Dense,\
 Activation,ZeroPadding1D, BatchNormalization, Flatten, Conv1D, Conv2D,AveragePooling1D,MaxPooling1D,MaxPooling2D,GlobalMaxPooling1D\
,UpSampling1D,concatenate
from keras.models import Model, load_model
import numpy as np
import scipy.signal as signal
from scipy import io
from scipy import signal
import pywt

result=[]

def CPSC2019_challenge(ECG,model1,model2,model3,model4,model5,model6,model7):
    def wavelet_data(data):#基础方法
        w='db5'
        a = data
        ca = []#近似分量
        cd = []#细节分量
        mode = pywt.Modes.smooth
        for i in range(7):
            (a, d) = pywt.dwt(a, w,mode)#进行7阶离散小波变换
            ca.append(a)
            cd.append(d)
        rec   = [] 
        rec_a = []
        rec_d = []
        for i, coeff in enumerate(ca):
            coeff_list = [coeff, None] + [None] * i
            rec_a.append(pywt.waverec(coeff_list, w))#重构
        for i, coeff in enumerate(cd):
            coeff_list = [None, coeff] + [None] * i
            rec_d.append(pywt.waverec(coeff_list, w))
        
        rec_a=np.array(rec_a)
        rec_d=np.array(rec_d)
        rec=np.concatenate((rec_a,rec_d))
        rec=rec.T
        return rec
    
    def wavelet_all_data(data):#基础方法
        rec_all=[]
        for i in range(len(data)):
                rec=wavelet_data(data[i])
                rec_all.append(rec)
        rec_all=np.array(rec_all)
        return rec_all
    def smooth(x,window_len=11,window='hanning'):
        if x.ndim != 1:
                raise(ValueError, "smooth only accepts 1 dimension arrays.")
        if x.size < window_len:
                raise( ValueError, "Input vector needs to be bigger than window size.")
        if window_len<3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise( ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        s=numpy.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=numpy.ones(window_len,'d')
        else:  
                w=eval('numpy.'+window+'(window_len)')
        y=numpy.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]
    
    ecgdata=np.zeros(ECG.shape)#小波的
    ecgdata2=np.zeros(ECG.shape)#之前上传的
    for i in range(len(ECG)):
        beat=ECG[i]
        print(i)
        E=beat
        E=E.reshape((1,5000))
        ecgdata[i]=E
         
        
        b=[0.0564484622607365,0.112896924521473,0.0564484622607365]
        a=[1,-1.22465158101310,0.450445430056041]
        y1 = signal.filtfilt(b, a, beat,method='pad')
        m=signal.medfilt(y1, 251)
        m=signal.medfilt(m, 251)
      
        E1=y1-m
        E1=E1.reshape((1,5000))
        ecgdata2[i]=E1
    ecgdata=ecgdata.reshape(len(ecgdata),5000)
    ecgdata=wavelet_all_data(ecgdata)
    ecgdata=ecgdata.reshape(ecgdata.shape[0],ecgdata.shape[1],14)
    
    ecgdata2=ecgdata2.reshape(len(ecgdata2),5000,1)
    
    result1=model1.predict(ecgdata2)#之前上传的
    print('model1')
    result2=model2.predict(ecgdata2)
    print('model2')
    result3=model3.predict(ecgdata)#小波的
    print('model3')
    result4=model4.predict(ecgdata)
    print('model4')
    result5=model5.predict(ecgdata)
    print('model5')
    result6=model6.predict(ecgdata)
    print('model6')
    result7=model7.predict(ecgdata)
    print('model7')
    
    result=(result1+result2+result3+result4+result5+result6+result7)/7

    result=np.array(result)
    result=result.reshape(len(result),5000)

    
    peakidx=[]
    
    
    hr=[]

    for i in range(len(result)):
        print(i)
        result0=result[i].reshape((5000,))
        result0=smooth(result0,window_len=30,window='flat')

        result0=smooth(result0,window_len=30,window='flat')

        result0=smooth(result0,window_len=30,window='flat')

        temp = signal.find_peaks(result0,distance=99,height=0.4)
        peakidx.append(temp)
        
# =============================================================================
        R_1=[]
        tt=list(temp[0])
        for  j in list(temp[0]):
            if j<0.5*500-0.075*500:
                tt.remove(j)
            if j>9.5*500+0.075*500:
                tt.remove(j)
        for j in tt:
            if j>5.5*500 and j<5000-0.5*500:
                R_1.append(j)
        if len(R_1)<=1:
            HR=500
        else:
            HR=round( 60 * 500 / np.mean(np.diff(R_1)))
# =============================================================================
        hr.append(HR)

    qrs=[]
    for i in range(len(peakidx)):
        qrs.append(peakidx[i][0])

    return hr, qrs,result