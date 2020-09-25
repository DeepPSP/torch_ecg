import keras
'''lib loading error prevention'''
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from keras.utils import to_categorical


"""========================"""
"""tensorflow configuration"""
""""======================="""

import tensorflow as tf
from keras import backend as K
num_cores = 48

num_CPU = 1
num_GPU = 1

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores,
                        allow_soft_placement=True,
                        device_count={'CPU': num_CPU, 'GPU': num_GPU})
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True

session = tf.Session(config=config)
K.set_session(session)
init = tf.global_variables_initializer()
session.run(init)

from sig_tool import normalize, diff, wad_rec
from plain_model import PlainModel, hyper_params, gen_ckpt_prefix, gen_pp_data_dir_name
from plain_data_make import PlainPreprocessor, load_kfold_names, preload
from icbeb_tool import load_icbeb2019_label
from sig_tool import stag
import pickle
import re

from model_tool import gen_kfold

from keras.models import load_model
'''global variables'''
fold = 5
fs = 500 # sampling frequency of the data
preprocessor = PlainPreprocessor(hyper_params)

(name_list, label_list) = load_icbeb2019_label('dat/icbeb2019')
label_list = [preprocessor.label_augment(label, span=40) for label in label_list]
labels = {}
for idx in range(len(name_list)):
    labels[name_list[idx]] = label_list[idx]


def val_static(paired):
    data_dir = gen_pp_data_dir_name()
    set_x = []
    set_y = []
    for idx in range(len(paired)):
        name, offset = paired[idx]
        train_sig, train_label, pre_train_sig, pre_train_label = pickle.load(open(os.path.join(data_dir, name+'.dat'), 'rb'))
        label = train_label[offset:offset+hyper_params['crop_len']]
        label = preprocessor.label_augment(label, span=hyper_params['label_expand'], offset=0)
        sig = pre_train_sig[offset:offset+hyper_params['crop_len']]

        #new wavelet decompos
        # sig = sig[:,0]
        # sa, sd = wad_rec(sig, 'db4', level=5)
        # sig = diff(sa[2][:5000])
        # wsig = []
        # wsig.append(sa[2][:5000])
        # wsig.append(sd[3][:5000])
        # wsig = np.transpose(sig)
        # set_x.append(sig)
        #wavelet decompos ends here

        set_x.append(sig[:,1])
        set_y.append(to_categorical(label, num_classes=2))
    set_x = np.reshape(set_x, newshape=(len(paired),5000,1))
    val_x = set_x[:]
    val_y = set_y[:]

    return val_x, val_y

def train_static(paired):
    data_dir = gen_pp_data_dir_name()
    set_x = []
    set_y = []
    for idx in range(len(paired)):
        name, offset = paired[idx]
        train_sig, train_label, pre_train_sig, pre_train_label = pickle.load(open(os.path.join(data_dir, name+'.dat'), 'rb'))

        label = train_label[offset:offset+hyper_params['crop_len']]
        label = preprocessor.label_augment(label, span=hyper_params['label_expand'], offset=0)

        sig = pre_train_sig[offset:offset+hyper_params['crop_len']]
        # sig = np.log10(np.abs(sig[4:4996,0])+1)


        # new wavelet decompos
        # sig = sig[:,0]
        # sa, sd = wad_rec(sig, 'db4', level=5)
        # sig = diff(sa[2][:5000])
        # wsig = []
        # wsig.append(sa[2][:5000])
        # wsig.append(sd[3][:5000])
        # wsig = np.transpose(wsig)
        # set_x.append(sig)
        #wavelet decompos ends here

        set_x.append(sig[:,1])
        set_y.append(to_categorical(label, num_classes=2))
    set_x = np.reshape(set_x, newshape=(len(paired),5000,1))

    return set_x, set_y


def train_gen(paired):
    batch_size = hyper_params['batch_size']
    data_len = len(paired)
    data_dir = gen_pp_data_dir_name()
    while True:
        np.random.shuffle(paired)
        for idx in range(data_len // batch_size):
            batch_x = []
            batch_y = []
            batch_pair = [pair for pair in paired[idx*batch_size:idx*batch_size+batch_size]]

            for (name, offset) in batch_pair:
                train_sig, train_label, pre_train_sig, pre_train_label = pickle.load(open(os.path.join(data_dir, name+'.dat'), 'rb'))
                sig = pre_train_sig[offset:offset+hyper_params['crop_len']]
                label = train_label[offset:offset+hyper_params['crop_len']]
                label = preprocessor.label_augment(label, span=30, offset=0)

                # batch_x.append(stag(sig[:,0], 10))
                batch_x.append(sig[:,0])
                batch_y.append(to_categorical(label, num_classes=2))

            batch_x = np.array(batch_x)
            batch_x = np.reshape(batch_x, newshape=(batch_size,5000,1))

            yield batch_x, np.array(batch_y)


if __name__ == '__main__':

    '''load data buff'''
    # if True:
    if not os.path.exists('shuffle_names.v.2.3.dat'):
        names, offsets = preload()
        train_paired = []
        val_paired = []

        train_names = []
        train_offsets = []
        val_names = []
        val_offsets = []
        for idx in range(len(names)):
            if re.match('^icbeb', names[idx]):
                val_names.append(names[idx])
                val_offsets.append(offsets[idx])
            else:
                # if re.match('^mitdb.*ch0$', names[idx]):
                # if re.match('^mitdb', names[idx]):
                train_names.append(names[idx])
                train_offsets.append(offsets[idx])

        for idx in range(len(train_names)):
            train_paired.append((train_names[idx], train_offsets[idx]))
        np.random.shuffle(train_paired)
        for idx in range(len(val_names)):
            val_paired.append((val_names[idx], val_offsets[idx]))
        np.random.shuffle(val_paired)

        '''re-arrange icbeb into training data set'''
        # for idx in range(len(val_names)//4*3):
        #     train_names.append(val_names[idx])
        #     train_offsets.append(val_offsets[idx])
        # val_names = val_names[len(val_names)//4*3:]
        # val_offsets = val_offsets[len(val_names)//4*3:]
        #
        # train_names = train_names[-6000:]
        # train_offsets = train_offsets[-6000:]
        #
        # train_paired = []
        # val_paired = []
        # for idx in range(len(train_names)):
        #     train_paired.append((train_names[idx], train_offsets[idx]))
        # np.random.shuffle(train_paired)
        # for idx in range(len(val_names)):
        #     val_paired.append((val_names[idx], val_offsets[idx]))
        # np.random.shuffle(val_paired)


        pickle.dump((train_paired, val_paired), open('shuffle_names.dat', 'wb'))

    train_paired, val_paired = pickle.load(open('shuffle_names.v.2.3.dat', 'rb'))

    '''training'''
    cycle = 0
    ckpt_prefix = gen_ckpt_prefix()
    fold_set = [(train_set, val_set) for (train_set, val_set) in gen_kfold(val_paired, 10)]
    while True:
        '''set random seed'''
        seed = np.random.randint(10000,99999)
        tf.set_random_seed(seed)

        '''checkpoint & earlystoping'''
        ckpt = keras.callbacks.ModelCheckpoint(
            filepath='models/'+'_fold10_'+str(hyper_params['label_expand'])+'_netlstm_sig1_'+str(seed)+'_'+str(cycle)+'_{epoch:03d}_{loss:.4f}_{val_loss:.4f}_{acc:.4f}_{val_acc:.4f}.h5',
            monitor='val_acc', save_best_only=True, verbose=1)
        es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
        board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True)

        '''build model'''
        hyper_params['callbacks'] = [ckpt, es]
        model = PlainModel(hyper_params)
        model.build()

        '''fine tuning'''
        # model.model = load_model('models/rematch_ckpt_plain_rev4_40_42101_0_038_0.0582_0.0607_0.9773_0.9769.h5')
        # model.model = load_model('models/rematch_ckpt_plain_rev4_30_91712_0_061_0.0610_0.0576_0.9751_0.9770.h5')
        # model.model = load_model('models/rematch_ckpt_plain_rev4_40_77525_0_036_0.0606_0.0622_0.9754_0.9751.h5')
        # model.model = load_model('models/rematch_ckpt_plain_rev4_30_99552_0_041_0.0582_0.0596_0.9763_0.9757.h5') # test stacked lstm model not working

        # model.model = load_model('models/rematch_ckpt_plain_rev4_30_sig0_22015_0_041_0.0591_0.0610_0.9758_0.9744.h5') # test for sig0

        # model.model = load_model('models/rematch_ckpt_plain_rev4_30_sig0_47302_0_048_0.0579_0.0618_0.9760_0.9744.h5') # stacked lstm, 3.1m
        # model.model = load_model('models/rematch_ckpt_plain_rev4_30_sig0_76361_2_093_0.0685_0.0597_0.9727_0.9766.h5') # U-net conv
        # model.model = load_model('models/rematch_ckpt_plain_rev4_30_sig0_81549_3_101_0.0665_0.0604_0.9735_0.9760.h5') # U-net conv 16.1m
        # model.model = load_model('models/rematch_ckpt_plain_rev4_30_sig0_28983_0_031_0.0656_0.0619_0.9748_0.9764.h5') # U-net lstm 6.9m
        # model.model = load_model('models/rematch_ckpt_plain_rev4_30_sig0_53267_1_043_0.0601_0.0607_0.9754_0.9761.h5') # U-net++ lstm 11.1m
        # model.model = load_model('models/rematch_ckpt_plain_rev4_30_sig0_42074_1_067_0.0528_0.0626_0.9782_0.9749.h5') # HRnet lstm 8.6m

        # model.model = load_model('models/_fold10_30_netlstm_sig1_88693_0_175_0.0622_0.0656_0.9750_0.9750.h5')
        # model.model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
        #                    metrics=hyper_params.get('metrics', ['acc', 'mae']),
        #                    loss='binary_crossentropy')

        # model.model.load_weights('models/rematch_ckpt_plain_rev4_40_42101_0_038_0.0582_0.0607_0.9773_0.9769.h5')
        # model.model.load_weights('models/rematch_ckpt_plain_rev4_30_91712_0_061_0.0610_0.0576_0.9751_0.9770.h5')
        # model.model.load_weights('models/rematch_ckpt_plain_rev4_40_77525_0_036_0.0606_0.0622_0.9754_0.9751.h5')
        # model.model.load_weights('models/rematch_ckpt_plain_rev4_30_99552_0_041_0.0582_0.0596_0.9763_0.9757.h5') # test stacked lstm model not working

        # model.model.load_weights('models/rematch_ckpt_plain_rev4_30_sig0_22015_0_041_0.0591_0.0610_0.9758_0.9744.h5') # test for sig0


        # model.model.load_weights('models/rematch_ckpt_plain_rev4_30_sig0_47302_0_048_0.0579_0.0618_0.9760_0.9744.h5') # stacked lstm
        # model.model = load_model('models/rematch_ckpt_plain_rev4_30_sig0_76361_2_093_0.0685_0.0597_0.9727_0.9766.h5') # U-net conv
        # model.model.load_weights('models/rematch_ckpt_plain_rev4_30_sig0_81549_3_101_0.0665_0.0604_0.9735_0.9760.h5') # U-net conv
        # model.model.load_weights('models/rematch_ckpt_plain_rev4_30_sig0_28983_0_031_0.0656_0.0619_0.9748_0.9764.h5') # U-net lstm
        # model.model.load_weights('models/rematch_ckpt_plain_rev4_30_sig0_53267_1_043_0.0601_0.0607_0.9754_0.9761.h5') # U-net++ lstm
        # model.model.load_weights('models/1rematch_ckpt_plain_rev4_30_sig0_42074_1_067_0.0528_0.0626_0.9782_0.9749.h5') # HRnet lstm
        # model.model.load_weights('models/_fold10_30_netlstm_sig1_88693_0_175_0.0622_0.0656_0.9750_0.9750.h5') # HRnet lstm

        '''training'''
        # train_p = val_paired[:1900]
        # val_p = val_paired[1900:]
        # train_iter = train_gen(train_p)
        # val_iter = train_gen(val_p)
        #
        # hyper_params['train_len'] = len(train_p)
        # hyper_params['val_len'] = len(val_p)
        # print('training on ', len(train_p), ' validate on ', len(val_p))
        # hist = model.train_gen(train_iter, val_iter, None)

        train_p = val_paired[:1800]
        val_p = val_paired[1800:]

        # train_p, val_p = fold_set[cycle]
        train_x, train_y = train_static(train_p)
        val_x, val_y = val_static(val_p)

        # weights = np.ones(shape=[1900,5000])
        # for idxs in range(len(train_y)):
        #     for l in range(5000):
        #         if train_y[idxs][l][1] == 1:
        #             weights[idxs, l] = 4
        hist = model.train(np.array(train_x), np.array(train_y), np.array(val_x), np.array(val_y))
        # hist = model.train(np.array(train_x), [np.array(train_y),np.array(train_y)], np.array(val_x), [np.array(val_y), np.array(val_y)])

        cycle += 1
        cycle = cycle % 10

