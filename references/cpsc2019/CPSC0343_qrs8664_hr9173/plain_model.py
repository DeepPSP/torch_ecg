from model import BaseModel, Preprocessor
from keras.layers import *
from keras.models import Model
import keras

from model_tool import to_multilabel
import tensorflow as tf

'''Model of plain signal stagging'''
hyper_params = {
    'file_prefix': 'plain_rev4',
    'crop_len': 5000,
    'label_expand': 30,
    'med_len': 151,
    'stag_len': 500,
    'stag_step': 100,
    'kernel_size': 5,
    'filter_size': 36,
    'batch_size': 50,
    'epochs': 5000,
    'kfold': 5,
    'preload_offset':3000
}


def gen_ckpt_prefix():
    return 'rematch_ckpt_' + hyper_params['file_prefix']

def gen_pp_data_dir_name():
    return 'dat/ppdata_' + hyper_params['file_prefix']

def gen_fold_name():
    return 'kfold_' + str(hyper_params['kfold']) + '.dat'

def gen_preload_name():
    return 'preload_' + str(hyper_params['preload_offset']) + '.dat'


'''======================
Plain model
input shape = (5000, ch)
   ======================'''


def weighted_loss(y_true, y_pred):
    '''not used'''
    return K.binary_crossentropy(y_true, y_pred)


class PlainModel(BaseModel):

    def build(self):
        kernel_size = hyper_params['kernel_size']
        filter_size = hyper_params['filter_size']
        stag_len = hyper_params['stag_len']
        stag_step = hyper_params['stag_step']
        crop_len = hyper_params['crop_len']
        '''input size: [num_samples, time, stag_len, 12]'''
        self.input = Input(shape=(5000, 1))
        o = self.input


        '''googlenet+U-net'''
        # layer_down = []
        # for idx in range(4):
        #     scale = idx+1
        #     o1 = Conv1D(filter_size*scale, 1, strides=1, padding='same')(o)
        #     o1 = Conv1D(filter_size*scale, 3, strides=1, padding='same')(o1)
        #     o1 = BatchNormalization()(o1)
        #     o1 = Activation('relu')(o1)
        #
        #     o2 = Conv1D(filter_size*scale, 1, strides=1, padding='same')(o)
        #     o2 = Conv1D(filter_size*scale, 5, strides=1, padding='same')(o2)
        #     o2 = BatchNormalization()(o2)
        #     o2 = Activation('relu')(o2)
        #
        #     o3 = Conv1D(filter_size*scale, 1, strides=1, padding='same')(o)
        #     o3 = BatchNormalization()(o3)
        #     o3 = Activation('relu')(o3)
        #
        #     o = Concatenate()([o1, o2, o3])
        #
        #     # o = MaxPooling1D(2)(o)
        #     layer_down.append(o)
        #
        # scale = 4
        # o1 = Conv1D(filter_size*scale, 1, strides=1, padding='same')(o)
        # o1 = Conv1D(filter_size*scale, 3, strides=1, padding='same')(o1)
        # o1 = BatchNormalization()(o1)
        # o1 = Activation('relu')(o1)
        #
        # o2 = Conv1D(filter_size*scale, 1, strides=1, padding='same')(o)
        # o2 = Conv1D(filter_size*scale, 5, strides=1, padding='same')(o2)
        # o2 = BatchNormalization()(o2)
        # o2 = Activation('relu')(o2)
        #
        # o3 = Conv1D(filter_size*scale, 1, strides=1, padding='same')(o)
        # o3 = BatchNormalization()(o3)
        # o3 = Activation('relu')(o3)
        #
        # o = Concatenate()([o1, o2, o3])
        #
        # for idx in range(4):
        #     scale = 4-idx
        #     # o = UpSampling1D(size=2)(o)
        #     o1 = Conv1D(filter_size*scale, 1, strides=1, padding='same')(o)
        #     o1 = Conv1D(filter_size*scale, 3, strides=1, padding='same')(o1)
        #     o1 = BatchNormalization()(o1)
        #     o1 = Activation('relu')(o1)
        #
        #     o2 = Conv1D(filter_size*scale, 1, strides=1, padding='same')(o)
        #     o2 = Conv1D(filter_size*scale, 5, strides=1, padding='same')(o2)
        #     o2 = BatchNormalization()(o2)
        #     o2 = Activation('relu')(o2)
        #
        #     o3 = Conv1D(filter_size*scale, 1, strides=1, padding='same')(o)
        #     o3 = BatchNormalization()(o3)
        #     o3 = Activation('relu')(o3)
        #
        #     o = Concatenate()([o1, o2, o3])
        #
        #     #====#
        #     o = Concatenate()([layer_down[3-idx], o])
        #     #====#
        #
        #     o1 = Conv1D(filter_size*scale, 1, strides=1, padding='same')(o)
        #     o1 = Conv1D(filter_size*scale, 3, strides=1, padding='same')(o1)
        #     o1 = BatchNormalization()(o1)
        #     o1 = Activation('relu')(o1)
        #
        #     o2 = Conv1D(filter_size*scale, 1, strides=1, padding='same')(o)
        #     o2 = Conv1D(filter_size*scale, 5, strides=1, padding='same')(o2)
        #     o2 = BatchNormalization()(o2)
        #     o2 = Activation('relu')(o2)
        #
        #     o3 = Conv1D(filter_size*scale, 1, strides=1, padding='same')(o)
        #     o3 = BatchNormalization()(o3)
        #     o3 = Activation('relu')(o3)
        #
        #     o = Concatenate()([o1, o2, o3])
        #
        #
        # classifier = Dense(2, activation='sigmoid')(o)


        '''stacked LSTM-Conv'''
        # o = self.input
        # filter_size = 16

        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # # layers = []
        # # layers.append(o)
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # # layers.append(o)
        # #
        # # for m in range(12):
        # #     o = Add()(layers)
        # #     o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # #     layers.append(o)
        # #
        # # o = Add()(layers)
        #
        # ocnn = []
        # o1 = Conv1D(filter_size, 1, strides=1, padding='same')(o)
        # o2 = Conv1D(filter_size, 3, strides=1, padding='same')(o)
        # o3 = Conv1D(filter_size, 3, strides=1, padding='same')(o)
        # o = Concatenate()([o1, o2, o3])
        # o = BatchNormalization()(o)
        # o = Activation('relu')(o)
        # ocnn.append(o)
        #
        # o1 = Conv1D(filter_size*2, 1, strides=1, padding='same')(o)
        # o2 = Conv1D(filter_size*2, 3, strides=1, padding='same')(o)
        # o3 = Conv1D(filter_size*2, 3, strides=1, padding='same')(o)
        # o = Concatenate()([o1, o2, o3])
        # o = BatchNormalization()(o)
        # o = Activation('relu')(o)
        # ocnn.append(o)
        #
        # o1 = Conv1D(filter_size*4, 1, strides=1, padding='same')(o)
        # o2 = Conv1D(filter_size*4, 3, strides=1, padding='same')(o)
        # o3 = Conv1D(filter_size*4, 3, strides=1, padding='same')(o)
        # o = Concatenate()([o1, o2, o3])
        # o = BatchNormalization()(o)
        # o = Activation('relu')(o)
        # ocnn.append(o)
        #
        # o = Concatenate()(ocnn)
        #

        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # l1 = o
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # l2 = o
        #
        # o = Concatenate()([l1, l2])
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # l3 = o
        #
        # o = Concatenate()([l1, l2, l3])
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # l4 = o
        #
        # o = Concatenate()([l1, l2, l3, l4])
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        #
        # oa = TimeDistributed(Dense(filter_size*2))(o)
        # oa = TimeDistributed(BatchNormalization())(oa)
        # oa = TimeDistributed(Activation('relu'))(oa)
        # o = Multiply()([o,oa])
        #
        # olstm = o
        # classifier = Dense(2, activation='sigmoid')(olstm)


        '''U-net Conv'''
        # filter_size = 32
        # # ==encoder==
        # o = self.input
        #
        # o1 = Conv1D(filter_size, kernel_size, strides=1, padding='same')(o)
        # o1 = Dropout(0.4)(o1)
        # o1 = Conv1D(filter_size, kernel_size, strides=1, padding='same')(o1)
        # o1 = BatchNormalization()(o1)
        # o1 = Activation('relu')(o1)
        # o1_norm = o1
        # o1 = MaxPooling1D(pool_size=2)(o1)
        #
        # o2 = Conv1D(filter_size * 2, kernel_size, strides=1, padding='same')(o1)
        # o2 = Dropout(0.4)(o2)
        # o2 = Conv1D(filter_size * 2, kernel_size, strides=1, padding='same')(o2)
        # o2 = BatchNormalization()(o2)
        # o2 = Activation('relu')(o2)
        # o2_norm = o2
        # o2 = MaxPooling1D(pool_size=2)(o2)
        #
        # o3 = Conv1D(filter_size * 4, kernel_size, strides=1, padding='same')(o2)
        # o3 = Dropout(0.4)(o3)
        # o3 = Conv1D(filter_size * 4, kernel_size, strides=1, padding='same')(o3)
        # o3 = BatchNormalization()(o3)
        # o3 = Activation('relu')(o3)
        # o3_norm = o3
        # o3 = MaxPooling1D(pool_size=2)(o3)
        #
        # #==U bottom layer==
        # btm = Conv1D(filter_size * 8, kernel_size, strides=1, padding='same')(o3)
        # btm = Dropout(0.4)(btm)
        # btm = Conv1D(filter_size * 8, kernel_size, strides=1, padding='same')(btm)
        # btm = BatchNormalization()(btm)
        # btm = Activation('relu')(btm)
        #
        # #==decoder==
        #
        # o4 = UpSampling1D(size=2)(btm)
        # o4 = Conv1D(filter_size * 4, kernel_size, strides=1, padding='same')(o4)
        # o4 = Dropout(0.4)(o4)
        # o4 = Conv1D(filter_size * 4, kernel_size, strides=1, padding='same')(o4)
        # o4 = BatchNormalization()(o4)
        # o4 = Activation('relu')(o4)
        #
        # o4 = Concatenate()([o3_norm, o4])
        # o4 = Conv1D(filter_size * 4, kernel_size, strides=1, padding='same')(o4)
        # o4 = Dropout(0.4)(o4)
        # o4 = Conv1D(filter_size * 4, kernel_size, strides=1, padding='same')(o4)
        # o4 = BatchNormalization()(o4)
        # o4 = Activation('relu')(o4)
        #
        # o5 = UpSampling1D(size=2)(o4)
        # o5 = Conv1D(filter_size * 4, kernel_size, strides=1, padding='same')(o5)
        # o5 = Dropout(0.4)(o5)
        # o5 = Conv1D(filter_size * 2, kernel_size, strides=1, padding='same')(o5)
        # o5 = BatchNormalization()(o5)
        # o5 = Activation('relu')(o5)
        #
        # o5 = Concatenate()([o2_norm, o5])
        # o5 = Conv1D(filter_size * 2, kernel_size, strides=1, padding='same')(o5)
        # o5 = Dropout(0.4)(o5)
        # o5 = Conv1D(filter_size * 2, kernel_size, strides=1, padding='same')(o5)
        # o5 = BatchNormalization()(o5)
        # o5 = Activation('relu')(o5)
        #
        # o6 = UpSampling1D(size=2)(o5)
        # o6 = Conv1D(filter_size, kernel_size, strides=1, padding='same')(o6)
        # o6 = Dropout(0.4)(o6)
        # o6 = Conv1D(filter_size, kernel_size, strides=1, padding='same')(o6)
        # o6 = BatchNormalization()(o6)
        # o6 = Activation('relu')(o6)
        #
        # o6 = Concatenate()([o1_norm, o6])
        # o6 = Conv1D(filter_size, kernel_size, strides=1, padding='same')(o6)
        # o6 = Dropout(0.4)(o6)
        # o6 = Conv1D(filter_size, kernel_size, strides=1, padding='same')(o6)
        # o6 = BatchNormalization()(o6)
        # o6 = Activation('relu')(o6)
        #
        # ocnn = o6
        #
        # classifier = ocnn
        #
        # # classifier = Conv1D(2, 1, strides=1, padding='same')(classifier)
        # # classifier = BatchNormalization()(classifier)
        # # classifier = Activation('sigmoid')(classifier)
        #
        # # classifier = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(classifier)
        # # classifier = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(classifier)
        # classifier = Dense(2, activation='sigmoid')(classifier)

        '''U-net concatenate LSTM + CNN'''
        # '''===LSTM U-net==='''
        #     # ==encoder==
        # o = self.input
        # o1 = Bidirectional(CuDNNLSTM(filter_size//2, return_sequences=True))(o)
        # o1_norm = o1
        # o1 = MaxPooling1D(pool_size=2)(o1)
        #
        # o2 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o1)
        # o2_norm = o2
        # o2 = MaxPooling1D(pool_size=2)(o2)
        #
        # o3 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o2)
        # o3_norm = o3
        # o3 = MaxPooling1D(pool_size=2)(o3)
        #
        # #==U bottom layer==
        # btm = Bidirectional(CuDNNLSTM(filter_size*4, return_sequences=True))(o3)
        #
        # #==decoder==
        #
        # o4 = UpSampling1D(size=2)(btm)
        # o4 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o4)
        #
        # o4 = Concatenate()([o3_norm, o4])
        # o4 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o4)
        #
        # o5 = UpSampling1D(size=2)(o4)
        # o5 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o5)
        #
        # o5 = Concatenate()([o2_norm, o5])
        # o5 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o5)
        #
        # o6 = UpSampling1D(size=2)(o5)
        # o6 = Bidirectional(CuDNNLSTM(filter_size//2, return_sequences=True))(o6)
        #
        # o6 = Concatenate()([o1_norm, o6])
        # o6 = Bidirectional(CuDNNLSTM(filter_size//2, return_sequences=True))(o6)
        #
        # olstm = o6
        #
        # '''===CNN==='''
        # filter_size = 32
        #  # ==encoder==
        # o = self.input
        #
        # o1 = Conv1D(filter_size, kernel_size, strides=1, padding='same')(o)
        # o1 = Dropout(0.4)(o1)
        # o1 = Conv1D(filter_size, kernel_size, strides=1, padding='same')(o1)
        # o1 = BatchNormalization()(o1)
        # o1 = Activation('relu')(o1)
        # o1_norm = o1
        # o1 = MaxPooling1D(pool_size=2)(o1)
        #
        # o2 = Conv1D(filter_size * 2, kernel_size, strides=1, padding='same')(o1)
        # o2 = Dropout(0.4)(o2)
        # o2 = Conv1D(filter_size * 2, kernel_size, strides=1, padding='same')(o2)
        # o2 = BatchNormalization()(o2)
        # o2 = Activation('relu')(o2)
        # o2_norm = o2
        # o2 = MaxPooling1D(pool_size=2)(o2)
        #
        # o3 = Conv1D(filter_size * 4, kernel_size, strides=1, padding='same')(o2)
        # o3 = Dropout(0.4)(o3)
        # o3 = Conv1D(filter_size * 4, kernel_size, strides=1, padding='same')(o3)
        # o3 = BatchNormalization()(o3)
        # o3 = Activation('relu')(o3)
        # o3_norm = o3
        # o3 = MaxPooling1D(pool_size=2)(o3)
        #
        # #==U bottom layer==
        # btm = Conv1D(filter_size * 8, kernel_size, strides=1, padding='same')(o3)
        # btm = Dropout(0.4)(btm)
        # btm = Conv1D(filter_size * 8, kernel_size, strides=1, padding='same')(btm)
        # btm = BatchNormalization()(btm)
        # btm = Activation('relu')(btm)
        #
        # #==decoder==
        #
        # o4 = UpSampling1D(size=2)(btm)
        # o4 = Conv1D(filter_size * 4, kernel_size, strides=1, padding='same')(o4)
        # o4 = Dropout(0.4)(o4)
        # o4 = Conv1D(filter_size * 4, kernel_size, strides=1, padding='same')(o4)
        # o4 = BatchNormalization()(o4)
        # o4 = Activation('relu')(o4)
        #
        # o4 = Concatenate()([o3_norm, o4])
        # o4 = Conv1D(filter_size * 4, kernel_size, strides=1, padding='same')(o4)
        # o4 = Dropout(0.4)(o4)
        # o4 = Conv1D(filter_size * 4, kernel_size, strides=1, padding='same')(o4)
        # o4 = BatchNormalization()(o4)
        # o4 = Activation('relu')(o4)
        #
        # o5 = UpSampling1D(size=2)(o4)
        # o5 = Conv1D(filter_size * 4, kernel_size, strides=1, padding='same')(o5)
        # o5 = Dropout(0.4)(o5)
        # o5 = Conv1D(filter_size * 2, kernel_size, strides=1, padding='same')(o5)
        # o5 = BatchNormalization()(o5)
        # o5 = Activation('relu')(o5)
        #
        # o5 = Concatenate()([o2_norm, o5])
        # o5 = Conv1D(filter_size * 2, kernel_size, strides=1, padding='same')(o5)
        # o5 = Dropout(0.4)(o5)
        # o5 = Conv1D(filter_size * 2, kernel_size, strides=1, padding='same')(o5)
        # o5 = BatchNormalization()(o5)
        # o5 = Activation('relu')(o5)
        #
        # o6 = UpSampling1D(size=2)(o5)
        # o6 = Conv1D(filter_size, kernel_size, strides=1, padding='same')(o6)
        # o6 = Dropout(0.4)(o6)
        # o6 = Conv1D(filter_size, kernel_size, strides=1, padding='same')(o6)
        # o6 = BatchNormalization()(o6)
        # o6 = Activation('relu')(o6)
        #
        # o6 = Concatenate()([o1_norm, o6])
        # o6 = Conv1D(filter_size, kernel_size, strides=1, padding='same')(o6)
        # o6 = Dropout(0.4)(o6)
        # o6 = Conv1D(filter_size, kernel_size, strides=1, padding='same')(o6)
        # o6 = BatchNormalization()(o6)
        # o6 = Activation('relu')(o6)
        #
        # ocnn = o6
        #
        # classifier = Concatenate()([olstm, ocnn])
        #
        # classifier = Dropout(0.4)(classifier)
        #
        # classifier = Dense(2, activation='sigmoid')(classifier)

        '''HRnet '''
        # def cnn_block(filter_size, kernel_size, inlayer):
        #     o = Conv1D(filter_size, kernel_size, strides=1, padding='same')(inlayer)
        #     o = Dropout(0.4)(o)
        #     o = Conv1D(filter_size, kernel_size, strides=1, padding='same')(o)
        #     o = BatchNormalization()(o)
        #     o = Activation('relu')(o)
        #     return o
        #
        #
        # o = self.input
        # o1 = cnn_block(filter_size, kernel_size, o)
        #
        # o2 = MaxPooling1D(2)(o1)
        # o2 = cnn_block(filter_size*2, kernel_size, o2)
        # o1d1 = cnn_block(filter_size*2, kernel_size, o1)
        #
        # o3 = MaxPooling1D(2)(o2)
        # # o3 = Bidirectional(CuDNNLSTM(filter_size*4, return_sequences=True))(o3)
        # o3 = Concatenate()([MaxPooling1D(4)(o1d1), o3])
        # o3 = cnn_block(filter_size*4, kernel_size, o3)
        #
        # # o2d1 = Bidirectional(CuDNNLSTM(filter_size*4, return_sequences=True))(o2)
        # o2d1 = Concatenate()([MaxPooling1D(2)(o1d1), o2])
        # o2d1 = cnn_block(filter_size*4, kernel_size, o2d1)
        #
        # # o1d2 = Bidirectional(CuDNNLSTM(filter_size*4, return_sequences=True))(o1d1)
        # o1d2 = Concatenate()([UpSampling1D(2)(o2), o1d1])
        # o1d2 = cnn_block(filter_size*4, kernel_size, o1d2)
        #
        # o4 = MaxPooling1D(2)(o3)
        # # o4 = Bidirectional(CuDNNLSTM(filter_size*8, return_sequences=True))(o4)
        # o4 = Concatenate()([MaxPooling1D(8)(o1d2), MaxPooling1D(4)(o2d1), o4])
        # o4 = cnn_block(filter_size*8, kernel_size, o4)
        #
        # # o3d1 = Bidirectional(CuDNNLSTM(filter_size*8, return_sequences=True))(o3)
        # o3d1 = Concatenate()([MaxPooling1D(4)(o1d2), MaxPooling1D(2)(o2d1), o3])
        # o3d1 = cnn_block(filter_size*8, kernel_size, o3d1)
        #
        # # o2d2 = Bidirectional(CuDNNLSTM(filter_size*8, return_sequences=True))(o2d1)
        # o2d2 = Concatenate()([UpSampling1D(2)(o3), o2d1, MaxPooling1D(2)(o1d2)])
        # o2d2 = cnn_block(filter_size*8, kernel_size, o2d2)
        #
        # # o1d3 = Bidirectional(CuDNNLSTM(filter_size*8, return_sequences=True))(o1d2)
        # o1d3 = Concatenate()([UpSampling1D(4)(o3), UpSampling1D(2)(o2d1), o1d2])
        # o1d3 = cnn_block(filter_size*8, kernel_size, o1d3)
        #
        # # o = Bidirectional(CuDNNLSTM(filter_size*16, return_sequences=True))(o1d3)
        # o = Concatenate()([UpSampling1D(8)(o4), UpSampling1D(4)(o3d1), UpSampling1D(2)(o2d2), o1d3])
        # o = cnn_block(filter_size*16, kernel_size, o)
        #
        # classifier = o
        #
        # classifier = Dense(2, activation='sigmoid')(classifier)


        '''HRnet LSTM'''
        # o = self.input
        # filter_size = 16
        # o1 = Bidirectional(CuDNNLSTM(filter_size//2, return_sequences=True))(o)
        #
        # o2 = MaxPooling1D(2)(o1)
        # o2 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o2)
        # o1d1 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o1)
        #
        # o3 = MaxPooling1D(2)(o2)
        # # o3 = Bidirectional(CuDNNLSTM(filter_size*4, return_sequences=True))(o3)
        # o3 = Concatenate()([MaxPooling1D(4)(o1d1), o3])
        # o3 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o3)
        #
        # # o2d1 = Bidirectional(CuDNNLSTM(filter_size*4, return_sequences=True))(o2)
        # o2d1 = Concatenate()([MaxPooling1D(2)(o1d1), o2])
        # o2d1 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o2d1)
        #
        # # o1d2 = Bidirectional(CuDNNLSTM(filter_size*4, return_sequences=True))(o1d1)
        # o1d2 = Concatenate()([UpSampling1D(2)(o2), o1d1])
        # o1d2 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o1d2)
        #
        # o4 = MaxPooling1D(2)(o3)
        # # o4 = Bidirectional(CuDNNLSTM(filter_size*8, return_sequences=True))(o4)
        # o4 = Concatenate()([MaxPooling1D(8)(o1d2), MaxPooling1D(4)(o2d1), o4])
        # o4 = Bidirectional(CuDNNLSTM(filter_size*4, return_sequences=True))(o4)
        #
        # # o3d1 = Bidirectional(CuDNNLSTM(filter_size*8, return_sequences=True))(o3)
        # o3d1 = Concatenate()([MaxPooling1D(4)(o1d2), MaxPooling1D(2)(o2d1), o3])
        # o3d1 = Bidirectional(CuDNNLSTM(filter_size*4, return_sequences=True))(o3d1)
        #
        # # o2d2 = Bidirectional(CuDNNLSTM(filter_size*8, return_sequences=True))(o2d1)
        # o2d2 = Concatenate()([UpSampling1D(2)(o3), o2d1, MaxPooling1D(2)(o1d2)])
        # o2d2 = Bidirectional(CuDNNLSTM(filter_size*4, return_sequences=True))(o2d2)
        #
        # # o1d3 = Bidirectional(CuDNNLSTM(filter_size*8, return_sequences=True))(o1d2)
        # o1d3 = Concatenate()([UpSampling1D(4)(o3), UpSampling1D(2)(o2d1), o1d2])
        # o1d3 = Bidirectional(CuDNNLSTM(filter_size*4, return_sequences=True))(o1d3)
        #
        # # o = Bidirectional(CuDNNLSTM(filter_size*16, return_sequences=True))(o1d3)
        # o = Concatenate()([UpSampling1D(8)(o4), UpSampling1D(4)(o3d1), UpSampling1D(2)(o2d2), o1d3])
        # o = Bidirectional(CuDNNLSTM(filter_size*8, return_sequences=True))(o)
        #
        # classifier = o
        #
        # classifier = Dense(2, activation='sigmoid')(classifier)



        '''U-net ++'''
        # o = self.input
        # # transformer
        # o = Conv1D(8, 3, strides=1, padding='same')(o)
        # o = BatchNormalization()(o)
        # o = Activation('relu')(o)
        # oo = o
        # o = Conv1D(16, 5, strides=1, padding='same')(o)
        # o = BatchNormalization()(o)
        # o = Activation('relu')(o)
        # oa = o
        # o = Concatenate()([oo, oa])
        #
        # #unet++
        # o1 = Bidirectional(CuDNNLSTM(filter_size//2, return_sequences=True))(o)
        #
        # o2 = MaxPooling1D(2)(o1)
        # o2 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o2)
        #
        # o3 = MaxPooling1D(2)(o2)
        # o3 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o3)
        #
        # o4 = MaxPooling1D(2)(o3)
        # o4 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o4)
        #
        # o1d1 = Concatenate()([o1, UpSampling1D(2)(o2)])
        # o1d1 = Bidirectional(CuDNNLSTM(filter_size//2, return_sequences=True))(o1d1)
        #
        # o2d1 = Concatenate()([UpSampling1D(2)(o3), o2])
        # o2d1 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o2d1)
        #
        # o1d2 = Concatenate()([o1, o1d1, UpSampling1D(2)(o2d1)])
        # o1d2 = Bidirectional(CuDNNLSTM(filter_size//2, return_sequences=True))(o1d2)
        #
        # o3d1 = Concatenate()([o3, UpSampling1D(2)(o4)])
        # o3d1 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o3d1)
        #
        # o2d2 = Concatenate()([o2, o2d1, UpSampling1D(2)(o3d1)])
        # o2d2 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o2d2)
        #
        # o1d3 = Concatenate()([o1, o1d1, o1d2, UpSampling1D(2)(o2d2)])
        # o1d3 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o1d3)
        #
        #
        # classifier = o1d3
        # classifier = Dense(2)(classifier)
        # classifier = BatchNormalization()(classifier)
        # classifier = Activation('sigmoid')(classifier)

        '''U-net full LSTM'''
        o = self.input

        # # transformer
        # o = Conv1D(8,3,strides=1,padding='same')(o)
        # o = BatchNormalization()(o)
        # o = Activation('relu')(o)
        # oo = o
        # o = Conv1D(16,5,strides=1,padding='same')(o)
        # o = BatchNormalization()(o)
        # o = Activation('relu')(o)
        # oa = o
        # o = Concatenate()([oo, oa])
        # ==encoder==
        o1 = Bidirectional(CuDNNLSTM(filter_size//2, return_sequences=True))(o)
        o1_norm = o1
        o1 = MaxPooling1D(pool_size=2)(o1)

        o2 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o1)
        o2_norm = o2
        o2 = MaxPooling1D(pool_size=2)(o2)

        o3 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o2)
        o3_norm = o3
        o3 = MaxPooling1D(pool_size=2)(o3)

        # ==U bottom layer==
        btm = Bidirectional(CuDNNLSTM(filter_size*4, return_sequences=True))(o3)

        # ==decoder==

        o4 = UpSampling1D(size=2)(btm)
        o4 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o4)

        o4 = Concatenate()([o3_norm, o4])
        o4 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o4)

        o5 = UpSampling1D(size=2)(o4)
        o5 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o5)

        o5 = Concatenate()([o2_norm, o5])
        o5 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o5)

        o6 = UpSampling1D(size=2)(o5)
        o6 = Bidirectional(CuDNNLSTM(filter_size//2, return_sequences=True))(o6)

        o6 = Concatenate()([o1_norm, o6])
        o6 = Bidirectional(CuDNNLSTM(filter_size//2, return_sequences=True))(o6)

        ocnn = o6

        # classifier = Concatenate()([olstm, ocnn])
        classifier = ocnn
        classifier = Dense(2, activation='sigmoid')(classifier)

        '''U-net full LSTM with conv for downsampling/upsampling'''
        # # ==encoder==
        # o = self.input
        # o1 = Bidirectional(CuDNNLSTM(filter_size//2, return_sequences=True))(o)
        # o1_norm = o1
        # o1 = Conv1D(filter_size, kernel_size=3, strides=2, padding='same')(o1)
        # o1 = BatchNormalization()(o1)
        # o1 = Activation('relu')(o1)
        #
        # o2 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o1)
        # o2_norm = o2
        # o2 = Conv1D(filter_size*2, kernel_size=3, strides=2, padding='same')(o2)
        # o2 = BatchNormalization()(o2)
        # o2 = Activation('relu')(o2)
        #
        # o3 = Bidirectional(CuDNNLSTM(filter_size*4, return_sequences=True))(o2)
        # o3_norm = o3
        # o3 = Conv1D(filter_size*2, kernel_size=3, strides=2, padding='same')(o3)
        # o3 = BatchNormalization()(o3)
        # o3 = Activation('relu')(o3)
        #
        # # ==U bottom layer==
        # btm = Bidirectional(CuDNNLSTM(filter_size*4, return_sequences=True))(o3)
        #
        # # ==decoder==
        # o4 = Conv1D(filter_size*4, kernel_size=3, strides=1, padding='same')(btm)
        # o4 = BatchNormalization()(o4)
        # o4 = Activation('relu')(o4)
        # o4 = UpSampling1D(size=2)(o4)
        # o4 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o4)
        #
        # o4 = Concatenate()([o3_norm, o4])
        # o4 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o4)
        #
        # o5 = Conv1D(filter_size*2, kernel_size=3, strides=1, padding='same')(o4)
        # o5 = BatchNormalization()(o5)
        # o5 = Activation('relu')(o5)
        # o5 = UpSampling1D(size=2)(o5)
        # o5 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o5)
        #
        # o5 = Concatenate()([o2_norm, o5])
        # o5 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o5)
        #
        # o6 = Conv1D(filter_size*2, kernel_size=3, strides=1, padding='same')(o5)
        # o6 = BatchNormalization()(o6)
        # o6 = Activation('relu')(o6)
        # o6 = UpSampling1D(size=2)(o6)
        # o6 = Bidirectional(CuDNNLSTM(filter_size//2, return_sequences=True))(o6)
        #
        # o6 = Concatenate()([o1_norm, o6])
        # o6 = Bidirectional(CuDNNLSTM(filter_size//2, return_sequences=True))(o6)
        #
        #
        # classifier = o6
        # classifier = Dense(2, activation='sigmoid')(classifier)

        '''dense LSTM ensemble'''
        # o = self.input
        # filter_size = 64
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # oa = TimeDistributed(Dense(filter_size * 2, activation='relu'))(o)
        # o = Multiply()([o, oa])
        # c1 = Dense(2,activation='sigmoid')(o)
        #
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # oa = TimeDistributed(Dense(filter_size * 2, activation='relu'))(o)
        # o = Multiply()([o, oa])
        # c2 = Dense(2, activation='sigmoid')(o)
        #
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # oa = TimeDistributed(Dense(filter_size * 2, activation='relu'))(o)
        # o = Multiply()([o, oa])
        # c3 = Dense(2, activation='sigmoid')(o)
        #
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # oa = TimeDistributed(Dense(filter_size * 2, activation='relu'))(o)
        # o = Multiply()([o, oa])
        #
        # c4 = Dense(2, activation='sigmoid')(o)
        #
        # classifier = Average()([c1, c2, c3, c4])

        '''multitask'''
        # o = self.input
        #
        # # transformer
        # o = Conv1D(8,3,strides=1,padding='same')(o)
        # o = BatchNormalization()(o)
        # o = Activation('relu')(o)
        # oo = o
        # o = Conv1D(16,5,strides=1,padding='same')(o)
        # o = BatchNormalization()(o)
        # o = Activation('relu')(o)
        # oa = o
        # o = Concatenate()([oo, oa])
        #
        # ot = o
        #
        # # ==encoder==
        # o1 = Bidirectional(CuDNNLSTM(filter_size//2, return_sequences=True))(ot)
        # o1_norm = o1
        # o1 = MaxPooling1D(pool_size=2)(o1)
        #
        # o2 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o1)
        # o2_norm = o2
        # o2 = MaxPooling1D(pool_size=2)(o2)
        #
        # o3 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o2)
        # o3_norm = o3
        # o3 = MaxPooling1D(pool_size=2)(o3)
        #
        # # ==U bottom layer==
        # btm = Bidirectional(CuDNNLSTM(filter_size*4, return_sequences=True))(o3)
        #
        # # ==decoder==
        #
        # o4 = UpSampling1D(size=2)(btm)
        # o4 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o4)
        #
        # o4 = Concatenate()([o3_norm, o4])
        # o4 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o4)
        #
        # o5 = UpSampling1D(size=2)(o4)
        # o5 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o5)
        #
        # o5 = Concatenate()([o2_norm, o5])
        # o5 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o5)
        #
        # o6 = UpSampling1D(size=2)(o5)
        # o6 = Bidirectional(CuDNNLSTM(filter_size//2, return_sequences=True))(o6)
        #
        # o6 = Concatenate()([o1_norm, o6])
        # o6 = Bidirectional(CuDNNLSTM(filter_size//2, return_sequences=True))(o6)
        #
        # ocnn = o6
        #
        # # classifier = Concatenate()([olstm, ocnn])
        # classifier = ocnn
        # classifier1 = Dense(2, activation='sigmoid')(classifier)
        #
        #
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(ot)
        # l1 = o
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # l2 = o
        #
        # o = Concatenate()([l1, l2])
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # l3 = o
        #
        # o = Concatenate()([l1, l2, l3])
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # l4 = o
        #
        # o = Concatenate()([l1, l2, l3, l4])
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        #
        # oa = TimeDistributed(Dense(filter_size*2))(o)
        # oa = TimeDistributed(BatchNormalization())(oa)
        # oa = TimeDistributed(Activation('relu'))(oa)
        # o = Multiply()([o,oa])
        #
        # olstm = o
        # classifier2 = Dense(2, activation='sigmoid')(olstm)
        #
        # self.model = Model(self.input, [classifier1, classifier2])

        '''Unet convlstm'''
        # o = self.input
        #
        # # transformer
        # o = Conv1D(8, 3, strides=1, padding='same')(o)
        # o = BatchNormalization()(o)
        # o = Activation('relu')(o)
        # oa = o
        # o = Conv1D(16, 5, strides=1, padding='same')(o)
        # o = BatchNormalization()(o)
        # o = Activation('relu')(o)
        # o = Concatenate()([oa, o])
        # ot = o
        #
        # # ==encoder==
        # o1 = Bidirectional(CuDNNLSTM(filter_size // 2, return_sequences=True))(ot)
        # o1_norm = o1
        # o1 = MaxPooling1D(pool_size=2)(o1)
        #
        # o2 = Conv1D(filter_size*2, kernel_size,strides=1, padding='same')(o1)
        # # o2 = BatchNormalization()(o2)
        # # o2 = Activation('relu')(o2)
        # o2 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o2)
        # o2_norm = o2
        # o2 = MaxPooling1D(pool_size=2)(o2)
        #
        # o3 = Conv1D(filter_size*4, kernel_size,strides=1, padding='same')(o2)
        # # o3 = BatchNormalization()(o3)
        # # o3 = Activation('relu')(o3)
        # o3 = Bidirectional(CuDNNLSTM(filter_size * 2, return_sequences=True))(o3)
        # o3_norm = o3
        # o3 = MaxPooling1D(pool_size=2)(o3)
        #
        # # ==U bottom layer==
        # btm = Conv1D(filter_size*8, kernel_size,strides=1, padding='same')(o3)
        # # btm = BatchNormalization()(btm)
        # # btm = Activation('relu')(btm)
        # btm = Bidirectional(CuDNNLSTM(filter_size * 4, return_sequences=True))(btm)
        #
        # # ==decoder==
        #
        # o4 = UpSampling1D(size=2)(btm)
        # o4 = Conv1D(filter_size*4, kernel_size,strides=1, padding='same')(o4)
        # # o4 = BatchNormalization()(o4)
        # # o4 = Activation('relu')(o4)
        # o4 = Bidirectional(CuDNNLSTM(filter_size * 2, return_sequences=True))(o4)
        #
        # o4 = Concatenate()([o3_norm, o4])
        # o4 = Conv1D(filter_size*4, kernel_size,strides=1, padding='same')(o4)
        # # o4 = BatchNormalization()(o4)
        # # o4 = Activation('relu')(o4)
        # o4 = Bidirectional(CuDNNLSTM(filter_size * 2, return_sequences=True))(o4)
        #
        # o5 = UpSampling1D(size=2)(o4)
        # o5 = Conv1D(filter_size*2, kernel_size,strides=1, padding='same')(o5)
        # # o5 = BatchNormalization()(o5)
        # # o5 = Activation('relu')(o5)
        # o5 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o5)
        #
        # o5 = Concatenate()([o2_norm, o5])
        # o5 = Conv1D(filter_size*2, kernel_size,strides=1, padding='same')(o5)
        # # o5 = BatchNormalization()(o5)
        # # o5 = Activation('relu')(o5)
        # o5 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o5)
        #
        # o6 = UpSampling1D(size=2)(o5)
        # o6 = Conv1D(filter_size, kernel_size,strides=1, padding='same')(o6)
        # # o6 = BatchNormalization()(o6)
        # # o6 = Activation('relu')(o6)
        # o6 = Bidirectional(CuDNNLSTM(filter_size // 2, return_sequences=True))(o6)
        #
        # o6 = Concatenate()([o1_norm, o6])
        # o6 = Conv1D(filter_size, kernel_size,strides=1, padding='same')(o6)
        # # o6 = BatchNormalization()(o6)
        # # o6 = Activation('relu')(o6)
        # o6 = Bidirectional(CuDNNLSTM(filter_size // 2, return_sequences=True))(o6)
        #
        # classifier = Dense(2, activation='sigmoid')(o6)

        '''Unet Conv-LSTM parallel'''
        # o = self.input
        #
        # # transformer
        # o = Conv1D(8,3,strides=1,padding='same')(o)
        # o = BatchNormalization()(o)
        # o = Activation('relu')(o)
        # oa = o
        # o = Conv1D(16,5,strides=1,padding='same')(o)
        # o = BatchNormalization()(o)
        # o = Activation('relu')(o)
        # o = Concatenate()([oa, o])
        # ot = o
        #
        # # ==encoder==
        # o1 = Bidirectional(CuDNNLSTM(filter_size//2, return_sequences=True))(ot)
        # o1_norm = o1
        # o1 = MaxPooling1D(pool_size=2)(o1)
        #
        # o2 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o1)
        # o2_norm = o2
        # o2 = MaxPooling1D(pool_size=2)(o2)
        #
        # o3 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o2)
        # o3_norm = o3
        # o3 = MaxPooling1D(pool_size=2)(o3)
        #
        # # ==U bottom layer==
        # btm = Bidirectional(CuDNNLSTM(filter_size*4, return_sequences=True))(o3)
        #
        # # ==decoder==
        #
        # o4 = UpSampling1D(size=2)(btm)
        # o4 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o4)
        #
        # o4 = Concatenate()([o3_norm, o4])
        # o4 = Bidirectional(CuDNNLSTM(filter_size*2, return_sequences=True))(o4)
        #
        # o5 = UpSampling1D(size=2)(o4)
        # o5 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o5)
        #
        # o5 = Concatenate()([o2_norm, o5])
        # o5 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o5)
        #
        # o6 = UpSampling1D(size=2)(o5)
        # o6 = Bidirectional(CuDNNLSTM(filter_size//2, return_sequences=True))(o6)
        #
        # o6 = Concatenate()([o1_norm, o6])
        # o6 = Bidirectional(CuDNNLSTM(filter_size//2, return_sequences=True))(o6)
        #
        # olstm = o6
        #
        # # ==encoder==
        # o = ot
        #
        # o1 = Conv1D(filter_size, kernel_size, strides=1, padding='same')(o)
        # o1 = BatchNormalization()(o1)
        # o1 = Activation('relu')(o1)
        # o1_norm = o1
        # o1 = MaxPooling1D(pool_size=2)(o1)
        #
        # o2 = Conv1D(filter_size * 2, kernel_size, strides=1, padding='same')(o1)
        # o2 = BatchNormalization()(o2)
        # o2 = Activation('relu')(o2)
        # o2_norm = o2
        # o2 = MaxPooling1D(pool_size=2)(o2)
        #
        # o3 = Conv1D(filter_size * 4, kernel_size, strides=1, padding='same')(o2)
        # o3 = BatchNormalization()(o3)
        # o3 = Activation('relu')(o3)
        # o3_norm = o3
        # o3 = MaxPooling1D(pool_size=2)(o3)
        #
        # #==U bottom layer==
        # btm = Conv1D(filter_size * 8, kernel_size, strides=1, padding='same')(o3)
        # btm = BatchNormalization()(btm)
        # btm = Activation('relu')(btm)
        #
        # #==decoder==
        #
        # o4 = UpSampling1D(size=2)(btm)
        # o4 = Conv1D(filter_size * 4, kernel_size, strides=1, padding='same')(o4)
        # o4 = BatchNormalization()(o4)
        # o4 = Activation('relu')(o4)
        #
        # o4 = Concatenate()([o3_norm, o4])
        # o4 = Conv1D(filter_size * 4, kernel_size, strides=1, padding='same')(o4)
        # o4 = BatchNormalization()(o4)
        # o4 = Activation('relu')(o4)
        #
        # o5 = UpSampling1D(size=2)(o4)
        # o5 = Conv1D(filter_size * 2, kernel_size, strides=1, padding='same')(o5)
        # o5 = BatchNormalization()(o5)
        # o5 = Activation('relu')(o5)
        #
        # o5 = Concatenate()([o2_norm, o5])
        # o5 = Conv1D(filter_size * 2, kernel_size, strides=1, padding='same')(o5)
        # o5 = BatchNormalization()(o5)
        # o5 = Activation('relu')(o5)
        #
        # o6 = UpSampling1D(size=2)(o5)
        # o6 = Conv1D(filter_size, kernel_size, strides=1, padding='same')(o6)
        # o6 = BatchNormalization()(o6)
        # o6 = Activation('relu')(o6)
        #
        # o6 = Concatenate()([o1_norm, o6])
        # o6 = Conv1D(filter_size, kernel_size, strides=1, padding='same')(o6)
        # o6 = BatchNormalization()(o6)
        # o6 = Activation('relu')(o6)
        #
        # ocnn = o6
        #
        # classifier = Concatenate()([olstm, ocnn])
        # classifier = Dense(2, activation='sigmoid')(classifier)

        '''multi-scale LSTM'''
        # filter_size = 24
        # o = self.input
        #
        #
        # # transformer
        # o = Conv1D(8,3,strides=1,padding='same')(o)
        # o = BatchNormalization()(o)
        # o = Activation('relu')(o)
        # oa = o
        # o = Conv1D(16,5,strides=1,padding='same')(o)
        # o = BatchNormalization()(o)
        # o = Activation('relu')(o)
        # o = Concatenate()([oa, o])
        # ot = o
        #
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(ot)
        # l1 = o
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # l2 = o
        #
        # o = Concatenate()([l1, l2])
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # l3 = o
        #
        # o = Concatenate()([l1, l2, l3])
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # l4 = o
        #
        # o = Concatenate()([l1, l2, l3, l4])
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        #
        # oa = TimeDistributed(Dense(filter_size*2))(o)
        # oa = TimeDistributed(BatchNormalization())(oa)
        # oa = TimeDistributed(Activation('relu'))(oa)
        # o1 = Multiply()([o,oa])
        #
        # o = MaxPooling1D(2)(o1)
        # # transformer
        # # o = Conv1D(8,3,strides=1,padding='same')(o)
        # # o = BatchNormalization()(o)
        # # o = Activation('relu')(o)
        # # oa = o
        # # o = Conv1D(16,5,strides=1,padding='same')(o)
        # # o = BatchNormalization()(o)
        # # o = Activation('relu')(o)
        # # o = Concatenate()([oa, o])
        # ot = o
        #
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # l1 = o
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # l2 = o
        #
        # o = Concatenate()([l1, l2])
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # l3 = o
        #
        # o = Concatenate()([l1, l2, l3])
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # l4 = o
        #
        # o = Concatenate()([l1, l2, l3, l4])
        # o2 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        #
        # # oa = TimeDistributed(Dense(filter_size*2))(o)
        # # oa = TimeDistributed(BatchNormalization())(oa)
        # # oa = TimeDistributed(Activation('relu'))(oa)
        # # o2 = Multiply()([o,oa])
        #
        # o = MaxPooling1D(2)(o2)
        # # transformer
        # # o = Conv1D(8,3,strides=1,padding='same')(o)
        # # o = BatchNormalization()(o)
        # # o = Activation('relu')(o)
        # # oa = o
        # # o = Conv1D(16,5,strides=1,padding='same')(o)
        # # o = BatchNormalization()(o)
        # # o = Activation('relu')(o)
        # # o = Concatenate()([oa, o])
        #
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # l1 = o
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # l2 = o
        #
        # o = Concatenate()([l1, l2])
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # l3 = o
        #
        # o = Concatenate()([l1, l2, l3])
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # l4 = o
        #
        # o = Concatenate()([l1, l2, l3, l4])
        # o3 = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        #
        # # oa = TimeDistributed(Dense(filter_size*2))(o)
        # # oa = TimeDistributed(BatchNormalization())(oa)
        # # oa = TimeDistributed(Activation('relu'))(oa)
        # # o3 = Multiply()([o,oa])
        #
        # o = Concatenate()([UpSampling1D(4)(o3), UpSampling1D(2)(o2), o1])
        # o = Bidirectional(CuDNNLSTM(filter_size, return_sequences=True))(o)
        # # oa = TimeDistributed(Dense(filter_size*2))(o)
        # # oa = TimeDistributed(BatchNormalization())(oa)
        # # oa = TimeDistributed(Activation('relu'))(oa)
        # # o = Multiply()([o,oa])
        #
        # classifier = Dense(2)(o)
        # classifier = BatchNormalization()(classifier)
        # classifier = Activation('sigmoid')(classifier)


        '''Dense LSTM Unet'''
        # o = self.input
        #
        # # transformer
        # filter_size=16
        # # o = Conv1D(8,3,strides=1,padding='same')(o)
        # # o = BatchNormalization()(o)
        # # o = Activation('relu')(o)
        # # oa = o
        # # o = Conv1D(16,5,strides=1,padding='same')(o)
        # # o = BatchNormalization()(o)
        # # o = Activation('relu')(o)
        # # o = Concatenate()([oa, o])
        # # ot = o
        #
        # o1 = Bidirectional(CuDNNLSTM(filter_size,return_sequences=True))(o)
        #
        # o2 = MaxPooling1D(2)(o1)
        # # o2 = Bidirectional(CuDNNLSTM(filter_size,return_sequences=True))(o2)
        # # o2 = Concatenate()([MaxPooling1D(2)(o1), o2])
        # o2 = Bidirectional(CuDNNLSTM(filter_size*2,return_sequences=True))(o2)
        #
        # o3 = MaxPooling1D(2)(o2)
        # # o3 = Bidirectional(CuDNNLSTM(filter_size*2,return_sequences=True))(o3)
        # o3 = Concatenate()([MaxPooling1D(4)(o1), o3])
        # o3 = Bidirectional(CuDNNLSTM(filter_size*4,return_sequences=True))(o3)
        #
        # o4 = MaxPooling1D(2)(o3)
        # # o4 = Bidirectional(CuDNNLSTM(filter_size*4,return_sequences=True))(o4)
        # o4 = Concatenate()([MaxPooling1D(8)(o1), MaxPooling1D(4)(o2), o4])
        # o4 = Bidirectional(CuDNNLSTM(filter_size*8,return_sequences=True))(o4)
        #
        # o5 = UpSampling1D(2)(o4)
        # # o5 = Bidirectional(CuDNNLSTM(filter_size*4,return_sequences=True))(o5)
        # o5 = Concatenate()([MaxPooling1D(4)(o1), MaxPooling1D(2)(o2), o3, o5])
        # o5 = Bidirectional(CuDNNLSTM(filter_size*4,return_sequences=True))(o5)
        #
        # o6 = UpSampling1D(2)(o5)
        # # o6 = Bidirectional(CuDNNLSTM(filter_size*2,return_sequences=True))(o6)
        # o6 = Concatenate()([MaxPooling1D(2)(o1), o2, UpSampling1D(2)(o3), UpSampling1D(4)(o4), o6])
        # o6 = Bidirectional(CuDNNLSTM(filter_size*2,return_sequences=True))(o6)
        #
        # o7 = UpSampling1D(2)(o6)
        # # o7 = Bidirectional(CuDNNLSTM(filter_size,return_sequences=True))(o7)
        # o7 = Concatenate()([o1, UpSampling1D(2)(o2), UpSampling1D(4)(o3), UpSampling1D(8)(o4), UpSampling1D(4)(o5), o7])
        # o7 = Bidirectional(CuDNNLSTM(filter_size,return_sequences=True))(o7)
        #
        # oa = Dense(filter_size*2)(o7)
        # oa = BatchNormalization()(oa)
        # oa = Activation('relu')(oa)
        #
        # o7 = Multiply()([o7, oa])
        #
        # classifier = Dense(2)(o7)
        # classifier = BatchNormalization()(classifier)
        # classifier = Activation('sigmoid')(classifier)

        self.model = Model(self.input, classifier)

        self.model.compile(optimizer=keras.optimizers.Adam(lr=0.0001,),
                           metrics=self.hyper_params.get('metrics', ['acc', 'mae']),
                           loss='binary_crossentropy')
        self.model.summary()
