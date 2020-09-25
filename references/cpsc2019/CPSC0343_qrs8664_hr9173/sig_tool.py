import numpy as np
import scipy as sp
import scipy.signal as signal
import matplotlib.pyplot as plt
import pywt

'''array resampling
sig: 1D array 
fs: original sampling rate
fs_new: resampled rate
method: interpolation method, available with linear and label
'''
def resample_idx(indice, fs, fs_new, method='linear'):
    idx_resample = []
    for idx in indice:
        i_resample = int(idx/fs*fs_new)
        idx_resample.append(i_resample)
    return idx_resample

def resample(sig, fs, fs_new, method='linear'):
    if len(np.shape(sig)) > 1:
        t_sig = np.transpose(sig)
        t_rs_sig = []
        for s in t_sig:
            re_sig = np.zeros((int)(len(s)/fs*fs_new))
            if method == 'linear':
                idx0 = 0
                for idx in range(len(re_sig)-1):
                    # by default align the first data
                    t = idx/fs_new
                    if (idx0+1)/fs > t >= idx0/fs:
                        dt_prev = t - idx0/fs
                        dt_next = (idx0+1)/fs - t
                        re_sig[idx] = (s[idx0]*dt_next+s[idx0+1]*dt_prev)*fs
                    else:
                        while t < idx0/fs or t >= (idx0+1)/fs:
                            idx0 += 1
                            if idx0 >= len(s):
                                break
                        if idx0 >= len(s)-1:
                            break
                        else:
                            dt_prev = t - idx0/fs
                            dt_next = (idx0+1)/fs - t
                            re_sig[idx] = (s[idx0]*dt_next+s[idx0+1]*dt_prev)*fs
            elif method == 'label':
                for idx in range(len(s)-1):
                    re_sig[(int)(idx/fs*fs_new)] = s[idx]
            elif method == 'label_str':
                re_sig = ['']*((int)(len(s)/fs*fs_new))
                for idx in range(len(s)-1):
                    re_sig[(int)(idx/fs*fs_new)] = s[idx]

            t_rs_sig.append(re_sig)

        return np.transpose(t_rs_sig)
    else:
        re_sig = np.zeros((int)(len(sig)/fs*fs_new))
        if method == 'linear':
            idx0 = 0
            for idx in range(len(re_sig)-1):
                # by default align the first data
                t = idx/fs_new
                if (idx0+1)/fs > t >= idx0/fs:
                    dt_prev = t - idx0/fs
                    dt_next = (idx0+1)/fs - t
                    re_sig[idx] = (sig[idx0]*dt_next+sig[idx0+1]*dt_prev)*fs
                else:
                    while t < idx0/fs or t >= (idx0+1)/fs:
                        idx0 += 1
                        if idx0 >= len(sig):
                            break
                    if idx0 >= len(sig)-1:
                        break
                    else:
                        dt_prev = t - idx0/fs
                        dt_next = (idx0+1)/fs - t
                        re_sig[idx] = (sig[idx0]*dt_next+sig[idx0+1]*dt_prev)*fs
        elif method == 'label':
            for idx in range(len(sig)-1):
                re_sig[(int)(idx/fs*fs_new)] = sig[idx]
        elif method == 'label_str':
            re_sig = ['']*((int)(len(sig)/fs*fs_new))
            for idx in range(len(sig)-1):
                re_sig[(int)(idx/fs*fs_new)] = sig[idx]

        return re_sig



'''fir filter'''
def fir(sig, h):
    return sp.signal.lfilter(h, [1.0], sig)


def iir(sig, b, a):
    return sp.signal.filtfilt(sig, b, a)


def interpolate(sig_start, sig_stop, num_point, method='linear'):
    sig_interp = np.zeros(num_point)
    if method == 'linear':
        diff = sig_stop - sig_start
        for idx in range(num_point):
            sig_interp[idx] = sig_start + diff / num_point * idx
    return sig_interp


'''moving median to remove pulse noise'''
def med_smooth(sig, med_len):
    sig_pad = []
    for idx in range(med_len):
        sig_pad.append(sig[0])
    for s in sig:
        sig_pad.append(s)
    for idx in range(med_len):
        sig_pad.append(sig[-1])
    med_sig = np.zeros(len(sig))
    for idx in range(len(sig)):
        try:
            med_sig[idx] = np.median(sig_pad[idx:idx+2*med_len])
        except RuntimeWarning:
            print(sig_pad[idx:idx+2*med_len])

    return med_sig

'''median filtering for baseline removal'''
def med_filter(sig, med_len):
    sig_pad = []
    for idx in range(med_len):
        sig_pad.append(sig[0])
    for s in sig:
        sig_pad.append(s)
    for idx in range(med_len):
        sig_pad.append(sig[-1])
    med_sig = np.zeros(len(sig))
    for idx in range(len(sig)):
        try:
            med_sig[idx] = sig_pad[idx+med_len] - np.median(sig_pad[idx:idx+2*med_len])
        except RuntimeWarning:
            print(sig_pad[idx:idx+2*med_len])
    # med_sig = [sig_pad[idx] - np.median(sig_pad[idx-med_len:idx+med_len]) for idx in range(med_len, len(sig)+med_len, 1)]

    return med_sig

def diff(sig, first=0):
    diff = np.diff(sig)
    dsig = []
    dsig.append(0)
    for d in diff:
        dsig.append(d)
    return dsig


def normalize(sig, method='norm'):
    if method == 'norm':
        mu = np.mean(sig)
        sigma = np.std(sig)
        return (sig-mu) / (sigma+np.finfo(float).eps)
    elif method == 'minmax':
        max_s = np.max(sig)
        min_s = np.min(sig)
        range_s = max_s-min_s
        return (sig-min_s) / (range_s + np.finfo(float).eps)
    else:
        sig = sig - np.mean(sig)
        min_s = np.min(sig)
        max_s = np.max(sig)
        return (sig-min_s) / (max_s-min_s)

'''Distance matrix computation (deprecated)'''


def rd_mat(sig, r=0.4):
    shape = np.shape(sig)
    dmat = np.zeros((shape[1]//10, shape[1]//10))
    for m in range(shape[1]//10):
        for n in range(shape[1]//10):
            dmat[m,n] = dist(sig[:,m*10], sig[:,n*10])
    max_d = np.max(np.max(dmat))
    dm = dmat / max_d
    rmat = np.zeros((shape[1], shape[1]))
    for m in range(shape[1]):
        for n in range(shape[1]):
            if dm[m, n] <= r:
                rmat[m, n] = 1
    return dmat[:,::10], rmat[:,::10]


'''compute the entry distance of tensor sig'''


def dmat(sig):
    shape = np.shape(sig)
    mat = np.zeros((shape[1], shape[1]))
    for idx1 in range(shape[1]):
        for idx2 in range(shape[1]):
            mat = dist(sig[:, idx1], sig[:, idx2])
    return mat


'''FFT spectrum'''


def fft_spec(x):
    shape = np.shape(x)
    xff = []
    for idx in range(shape[0]):
        xf = np.fft.rfft(x[idx]) / len(x)
        xff.append(np.abs(xf))

    return xff

'''STFT spectrum'''
'''return spectrum in dimension [time, freq, channel]'''
def stft_spec(x, fft_len, step=10):
    shape = np.shape(x)
    spec = []
    for idx in range((shape[0]-fft_len)//step):
        if idx*10+fft_len > shape[0]:
            break

        xff = []
        for m in range(shape[1]):
            s = x[idx*10:idx*10+fft_len, m]
            xf = np.abs(np.fft.rfft(s)/ fft_len)
            xff.append(xf)
        xff = np.transpose(xff)
        spec.append(xff)

    return spec

'''stagging sequence by padding the front and end of the sequence with x[0] and x[-1]'''
def stag(x, stag_len):
    x_stag = []
    x_ex = []
    for idx in range(stag_len):
        x_ex.append(x[0])
    for idx in range(len(x)):
        x_ex.append(x[idx])
    for idx in range(stag_len):
        x_ex.append(x[-1])
    for idx in range(len(x)):
        x_stag.append(x_ex[idx:idx+stag_len*2+1])

    return x_stag

def dist(x,y):
    return norm(x-y)

def norm(x):
    return sp.sqrt(np.sum(np.multiply(x,x)))

'''correlation without normalization'''
def corr(x,y):
    return np.sum(np.multiply(x,y))

'''
cross-correlation by eq: r(tau) = sum(x(t)*y(t-tau))/norm(x)/norm(y)
'''
def xcorr(x,y, tau=0):
    if tau > 0:
        x = x[tau:]
        y = y[:-tau]
    elif tau < 0:
        x = x[:tau]
        y = y[-tau:]
    else:
        pass
    if len(x) > len(y):
        x = x[:len(y)]
    else:
        y = y[:len(x)]

    x = x - np.mean(x)
    y = y - np.mean(y)

    n_x = norm(x)
    n_y = norm(y)
    if n_x == 0 or n_y == 0:
        print('zero norm')

    return corr(x,y) / (n_x*n_y)

# import matplotlib.pyplot as plt
# if __name__ == '__main__':
#     # 50 Hz notch
#     # h = [0.0310644135763970, 0.0255087415970111, 0.00964873935499283, \
#     #      -0.0107972599767222, -0.0280025955114245, -0.0350470722033277, -0.0287122166518437,
#     #      -0.0109229397856761, 0.0118018231781266, 0.030750882324595, 0.0383854331233733,
#     #      0.0313572393761393, 0.0119645268699271, -0.0125839951030895, -0.0328689785256390,
#     #      -0.0409053551544674, -0.0333100849578491, -0.0127265482002507, 0.0130947338610567,
#     #      0.0342400897997490, 0.0424733523115399, 0.0344720247413389, 0.0131765421523021,
#     #     -0.0132986704694056, -0.0347859290186206, 0.956994447933343, -0.0347859290186206,
#     #      -0.0132986704694056, 0.0131765421523021, 0.0344720247413389, 0.0424733523115399,
#     #     0.0342400897997490, 0.0130947338610567, -0.0127265482002507, -0.0333100849578491,
#     #      -0.0409053551544674, -0.0328689785256390, -0.0125839951030895, 0.0119645268699271,
#     #     0.0313572393761393, 0.0383854331233733, 0.0307508823245951, 0.0118018231781266,
#     #     -0.0109229397856761, -0.0287122166518437, -0.0350470722033277, -0.0280025955114245,
#     #     -0.0107972599767222, 0.00964873935499283, 0.0255087415970111, 0.0310644135763965]
#
#     #45 Hz lp
#     h = [0.00807419063902593, 0.00460762719195534, -0.00186027941903536, -0.00962803488996367,
#          -0.0164248379932052, -0.0200415912761841, -0.0190047817159963, -0.0130992987903060,
#          -0.00357318687269947, 0.00706596579689111, 0.0156450203946957, 0.0191800554231676,
#          0.0157782283534493, 0.00534163969593757, -0.0101385852058141, -0.0268167476178944,
#          -0.0397430549506140, -0.0439810848993720, -0.0358373400586266, -0.0139022855979554,
#          0.0203685151361154, 0.0627047106738502, 0.106726915344643, 0.145172687287144,
#          0.171376670378794, 0.180670093069694, 0.171376670378794, 0.145172687287144,
#          0.106726915344643, 0.0627047106738502, 0.0203685151361154, -0.0139022855979554,
#          -0.0358373400586266, -0.0439810848993720, -0.0397430549506140, -0.0268167476178944,
#          -0.0101385852058141, 0.00534163969593757, 0.0157782283534493, 0.0191800554231676,
#          0.0156450203946957, 0.00706596579689111, -0.00357318687269947, -0.0130992987903060,
#          -0.0190047817159963, -0.0200415912761841, -0.0164248379932052, -0.00962803488996367,
#          -0.00186027941903536, 0.00460762719195534, 0.00807419063902593]
#     w, resp = sp.signal.freqz(h,[1.0])
#     plt.figure()
#     plt.plot(w/3.14, sp.log10(abs(resp)))
#     plt.plot(w/3.14, np.angle(resp))
#     plt.show()
#     x = np.random.random(500)
#     f = fir(x, h)
#     plt.figure()
#     plt.plot(x)
#     plt.plot(f)
#     plt.show()

'''wavelet decomp'''
def wad_rec(sig, wavename, level):
    # coefs = pywt.wavedec(sig, wavename, level=level)
    #
    # ca = coefs[0]
    # cd = list(reversed(coefs[1:]))
    #
    # sa = []
    # sd = []
    # N = len(sig)
    # for idx in range(level):
    #     d = pywt.upcoef('d', cd[idx-1], wavename, level=idx+1)[:N]
    #     sd.append(d)
    # a = pywt.upcoef('a', ca, wavename, level=5)[:N]
    # sa.append(a)
    #
    # return sa,sd

    a = sig
    ca = []
    cd = []
    for idx in range(level):
        a, d = pywt.dwt(a, wavename)
        ca.append(a)
        cd.append(d)
    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        coef_list = [coeff, None] + [None]*i
        rec_a.append(pywt.waverec(coef_list, wavename))
    for i, coeff in enumerate(cd):
        coef_list = [None, coeff] + [None]*i
        rec_d.append(pywt.waverec(coef_list, wavename))
    return rec_a, rec_d



