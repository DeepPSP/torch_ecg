import keras
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import *
from sig_tool import fir, iir, med_filter, normalize
from sklearn import preprocessing as skpreprocess
from sig_tool import diff
from sig_tool import med_smooth
from concurr_tool import MultiTask
import scipy.signal as signal
import math
import matplotlib.pyplot as plt


'''common filtering before processing'''
def _filtering(sig, hyper_params):
    # baseline filtering
    sig_raw = sig

    sig = med_smooth(sig, 1)
    sig1 = med_smooth(sig, 30)
    sig2 = med_smooth(sig, 40)
    sig3 = med_smooth(sig, 50)
    sig4 = med_smooth(sig, 75)
    sig_med = np.mean([sig1, sig2, sig3, sig4], axis=0)
    sig = sig - sig_med
    # sig = med_filter(sig, hyper_params.get('med_len', 20))
    # sig = med_filter(sig, hyper_params.get('med_len', 50))

    # for idx in range(len(sig)):
    #     if np.isnan(sig[idx]):
    #         print('found nan', idx)
    #         break

    '''skip the filtering'''
    # 45 Hz LP filtering
    h = [0.00807419063902593, 0.00460762719195534, -0.00186027941903536, -0.00962803488996367,
         -0.0164248379932052, -0.0200415912761841, -0.0190047817159963, -0.0130992987903060,
         -0.00357318687269947, 0.00706596579689111, 0.0156450203946957, 0.0191800554231676,
         0.0157782283534493, 0.00534163969593757, -0.0101385852058141, -0.0268167476178944,
         -0.0397430549506140, -0.0439810848993720, -0.0358373400586266, -0.0139022855979554,
         0.0203685151361154, 0.0627047106738502, 0.106726915344643, 0.145172687287144,
         0.171376670378794, 0.180670093069694, 0.171376670378794, 0.145172687287144,
         0.106726915344643, 0.0627047106738502, 0.0203685151361154, -0.0139022855979554,
         -0.0358373400586266, -0.0439810848993720, -0.0397430549506140, -0.0268167476178944,
         -0.0101385852058141, 0.00534163969593757, 0.0157782283534493, 0.0191800554231676,
         0.0156450203946957, 0.00706596579689111, -0.00357318687269947, -0.0130992987903060,
         -0.0190047817159963, -0.0200415912761841, -0.0164248379932052, -0.00962803488996367,
         -0.00186027941903536, 0.00460762719195534, 0.00807419063902593]
    sig_unfiltered = sig
    # sig = fir(sig, h)

    # normalize with standard scaler (x-mu)/std
    # normalize may cause nan issues for aha
    sig = normalize(sig)

    return sig


class Preprocessor:
    def __init__(self, hyper_params):
        self.hyper_params = hyper_params

    def crop(self, x, crop_len):
        crop_sigs = []
        for sig in x:
            if len(sig[0,:]) < crop_len:
                # padding here
                crop_s = np.pad(sig[:,:crop_len], ((0,0), (0, crop_len-len(sig[0,:]))), 'wrap')
                crop_sigs.append(crop_s)
            else:
                crop_sigs.append(sig[:, :crop_len])
        return crop_sigs

    def preprocess(self, x, y):
        # multitask = MultiTask(40, 10000)
        cnt = 0
        sigs = []
        for sig in x:
            # multitask.submit(cnt, _filtering, (sig, self.hyper_params))
            sig_filtered = _filtering(sig, self.hyper_params)
            sigs.append(np.transpose(sig_filtered))
            cnt += 1
        # sigs = [np.transpose(sig) for sig in multitask.subscribe()]
        return sigs, y


class BaseModel:
    def __init__(self, hyper_params):
        self.input = None

        self.layers = []

        self.model = None

        self.hyper_params = hyper_params

    def build(self):
        pass

    def train(self, train_x, train_y, val_x, val_y):
        batch_size = self.hyper_params.get('batch_size', 4)
        epochs = self.hyper_params.get('epochs', 100)
        verbose = self.hyper_params.get('verbose', 1)
        callbacks = self.hyper_params.get('callbacks',[])
        self.model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_data=(val_x, val_y), verbose=verbose
                       )

    def train_gen(self, train_iter, val_iter, class_weight):
        batch_size = self.hyper_params.get('batch_size', 4)
        epochs = self.hyper_params.get('epochs', 100)
        verbose = self.hyper_params.get('verbose', 1)
        callbacks = self.hyper_params.get('callbacks',[])
        train_len = self.hyper_params['train_len']
        val_len = self.hyper_params['val_len']
        return self.model.fit_generator(generator=train_iter,
                                        steps_per_epoch=math.ceil(train_len//batch_size),
                                        epochs=epochs,
                                        verbose=verbose,
                                        callbacks=callbacks,
                                        validation_data=val_iter,
                                        validation_steps=math.ceil(val_len//batch_size),
                                        )
