import scipy.signal as ss
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt

names = os.listdir('./data/')

name = names[0]


sig = loadmat('./data/'+name)['ecg'].squeeze()
anno = loadmat('./ref/R_'+name[-9:])['R_peak']

plt.plot(sig)
plt.plot(anno,sig[anno],'ro')
plt.show()


sig_re = ss.resample(sig,5120)
anno_re = (anno/5000)*5120
anno_re = anno_re.astype('int')
plt.plot(sig_re)
plt.plot(anno_re,sig_re[anno_re],'ro')
plt.show()

print(1)