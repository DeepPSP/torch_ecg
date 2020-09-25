from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Lambda, Input, average, Reshape, UpSampling1D, Multiply,Concatenate
from keras.layers import Conv1D, Flatten, Dense, Add, AveragePooling1D
from keras.layers import ZeroPadding1D, Cropping1D, BatchNormalization, MaxPooling1D
from keras import backend as K
from keras.layers import *
from keras import losses
from keras.utils.np_utils import to_categorical
import numpy as np
from keras import regularizers
def crop(tensors):
    '''
    :param tensors: List of two tensors, the second tensor having larger spatial dims
    :return:
    '''
    h_dims, w_dims = [], []
    for t in tensors:
        b, h, w, d = K.get_variable_shape( t )
        h_dims.append( h )
        w_dims.append( w )
    crop_h, crop_w = (h_dims[1] - h_dims[0]), (w_dims[1] - w_dims[0])
    rem_h = crop_h % 2
    rem_w = crop_w % 2
    crop_h_dims = (crop_h // 2, crop_h // 2 + rem_h)
    crop_w_dims = (crop_w // 2, crop_w // 2 + rem_w)
    cropped = Cropping1D( cropping=(crop_h_dims, crop_w_dims) )( tensors[1] )
    return cropped


def dice_loss(y_true, y_pred):
    eps = 1e-5
    intersection = K.sum(y_true * y_pred, axis=-1) + eps
    summation = K.sum(y_true, axis=-1) + K.sum(y_pred,axis=-1) + eps
    dice_loss = 1. - (2. * intersection/summation)
    return dice_loss

# m: input
# dim: the num of channel
# res: controls the res connection
# drop: controls the dropout layer
# initpara: initial parameters
def convblock(m, dim, layername, res=0, drop=0.5, **kwargs):
    n = Conv1D(filters=dim, name= layername + '_conv1', **kwargs)(m)
    n = BatchNormalization(momentum=0.95, epsilon=0.001)(n)
    n = Dropout(drop)(n) if drop else n
    n = Conv1D(filters=dim, name= layername + '_conv2', **kwargs)(n)
    n = BatchNormalization(momentum=0.95, epsilon=0.001)(n)

    # m = Conv1D(filters=dim, name= layername + '_conv3', kernel_size=1, padding='same', activation='relu')(m)
    # m = BatchNormalization(momentum=0.95, epsilon=0.001)(m)
    return Concatenate()([m, n]) if res else n

def unet(input_shape, num_classes, lr, kernel_size=3, filter_num=32, res=0, maxpool=True, weights=None, drop_rate=0.5, use_lstm=True, loss_func='mse'):
    '''initialization'''
    kwargs = dict(
        kernel_size=kernel_size,
        strides=1,
        activation='relu',
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',  # Xavier均匀初始化

        #kernel_initializer='he_normal',
        bias_initializer='zeros',
        bias_regularizer=None,
        activity_regularizer=None,  # 施加在输出上的正则项
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,  # 权值是否更新
    )

    kwargs2 = dict(
        kernel_size=1,
        strides=1,
        activation='relu',
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',  # Xavier均匀初始化

        #kernel_initializer='he_normal',
        bias_initializer='zeros',
        bias_regularizer=None,
        activity_regularizer=None,  # 施加在输出上的正则项
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,  # 权值是否更新
    )
    
    num_classes = num_classes
    data = Input(shape=input_shape, dtype='float', name='data')
    # encoder
    enconv1 = convblock(data, dim=filter_num, res=res, drop=drop_rate, layername='block1', **kwargs)
    pool1 = MaxPooling1D(pool_size=3, strides=2,padding='same',name='pool1')(enconv1) if maxpool \
        else Conv1D(filters=filter_num, strides=2, name='pool1')(enconv1)

    enconv2 = convblock(pool1, dim=filter_num, res=res, drop=drop_rate, layername='block2', **kwargs)
    pool2 = MaxPooling1D(pool_size=3, strides=2, padding='same', name='pool2')(enconv2) if maxpool \
        else Conv1D(filters=filter_num, strides=2, name='pool2')(enconv2)

    enconv3 = convblock(pool2, dim=2*filter_num, res=res, drop=drop_rate, layername='block3', **kwargs)
    pool3 = MaxPooling1D(pool_size=3, strides=2, padding='same', name='pool3')(enconv3) if maxpool \
        else Conv1D( filters=filter_num, strides=2, name='pool3')(enconv3)

    enconv4 = convblock(pool3, dim=2*filter_num, res=res, drop=drop_rate, layername='block4', **kwargs)
    pool4 = MaxPooling1D(pool_size=3, strides=2, padding='same', name='pool4')(enconv4) if maxpool \
        else Conv1D(filters=filter_num, strides=2, name='pool4')(enconv4)

    enconv5 = convblock(pool4, dim=4*filter_num, res=res, drop=drop_rate, layername='block5', **kwargs)
    pool5 = MaxPooling1D(pool_size=3, strides=2, padding='same', name='pool5')(enconv5) if maxpool \
        else Conv1D(filters=filter_num, strides=2, name='pool5')(enconv5)

    enconv6 = convblock(pool5, dim=4*filter_num, res=res, drop=drop_rate, layername='block6', **kwargs)
    pool6 = MaxPooling1D(pool_size=3, strides=2, padding='same', name='pool6')(enconv6) if maxpool \
        else Conv1D(filters=filter_num, strides=2, name='pool6')(enconv6)

    enconv7 = convblock(pool6, dim=8*filter_num, res=res, drop=drop_rate, layername='block7', **kwargs)
    pool7 = MaxPooling1D(pool_size=3, strides=2, padding='same', name='pool7')(enconv7) if maxpool \
        else Conv1D(filters=filter_num, strides=2, name='pool7')(enconv7)

    enconv8 = convblock(pool7, dim=8*filter_num, res=res, drop=drop_rate, layername='block8', **kwargs)
    pool8 = MaxPooling1D(pool_size=3, strides=2, padding='same', name='pool8')(enconv8) if maxpool \
        else Conv1D(filters=filter_num, strides=2, name='pool8')(enconv8)

    enconv9 = convblock(pool8, dim=16*filter_num, res=res, drop=drop_rate, layername='block9', **kwargs)
    pool9 = MaxPooling1D(pool_size=3, strides=2, padding='same', name='pool9')(enconv9) if maxpool \
        else Conv1D(filters=filter_num, strides=2, name='pool9')(enconv9)


    enconv10 = convblock(pool9, dim=16*filter_num, res=res, drop=drop_rate, layername='block10', **kwargs)

    if use_lstm:
        # LSTM
        lstm1 = Bidirectional(LSTM(8*filter_num, dropout=0.2, recurrent_dropout=0.2,
                 return_state=False, return_sequences=True), merge_mode = 'concat')(enconv10)

        # decoder
        up9 = Conv1D(filters=16*filter_num, kernel_size=1, padding='same', activation='relu',
                     name='up9')(UpSampling1D(size=2)(lstm1))
    else:
        up9 = Conv1D(filters=16*filter_num, kernel_size=1, padding='same', activation='relu',
                     name='up9')(UpSampling1D(size=2)(enconv10))

    merge9 = Concatenate()([up9, enconv9])
    deconv9 = convblock(merge9, dim=8*filter_num, res=res, drop=drop_rate, layername='deconv9', **kwargs2)

    up8 = Conv1D(filters=8*filter_num, kernel_size=1, padding='same', activation='relu',
                 name='up8')(UpSampling1D(size=2)(deconv9))
    merge8 = Concatenate()([up8, enconv8])
    deconv8 = convblock(merge8, dim=8*filter_num, res=res, drop=drop_rate, layername='deconv8', **kwargs2)

    up7 = Conv1D(filters=8*filter_num, kernel_size=1, padding='same', activation='relu',
                 name='up7')(UpSampling1D(size=2)(deconv8))
    merge7 = Concatenate()([up7,enconv7])
    deconv7 = convblock(merge7, dim=8*filter_num, res=res, drop=drop_rate, layername='deconv7', **kwargs2)

    up6 = Conv1D(filters=4*filter_num, kernel_size=1, padding='same', activation='relu',
                 name='up6')(UpSampling1D(size=2)(deconv7))
    merge6 = Concatenate()([up6,enconv6])
    deconv6 = convblock(merge6, dim=filter_num, res=res, drop=drop_rate, layername='deconv6', **kwargs2)

    up5 = Conv1D(filters=4*filter_num, kernel_size=1, padding='same', activation='relu',
                 name='up5')(UpSampling1D(size=2)(deconv6))
    merge5 = Concatenate()([up5, enconv5])
    deconv5 = convblock(merge5, dim=filter_num, res=res, drop=drop_rate, layername='deconv5', **kwargs2)

    up4 = Conv1D(filters=2*filter_num, kernel_size=1, padding='same', activation='relu',
                 name='up4')(UpSampling1D(size=2)(deconv5))
    merge4 = Concatenate()([up4, enconv4])
    deconv4 = convblock(merge4, dim=filter_num, res=res, drop=drop_rate, layername='deconv4', **kwargs2)

    up3 = Conv1D(filters=2*filter_num, kernel_size=1, padding='same', activation='relu',
                 name='up3')(UpSampling1D(size=2)(deconv4))
    merge3 = Concatenate()([up3, enconv3])
    deconv3 = convblock(merge3, dim=filter_num, res=res, drop=drop_rate, layername='deconv3', **kwargs2)

    up2 = Conv1D(filters=filter_num, kernel_size=1, padding='same', activation='relu',
                 name='up2')(UpSampling1D(size=2)(deconv3))
    merge2 = Concatenate()([up2, enconv2])
    deconv2 = convblock(merge2, dim=filter_num, res=res, drop=drop_rate, layername='deconv2', **kwargs2)  

    up1 = Conv1D(filters=filter_num, kernel_size=1, padding='same', activation='relu',
                 name='up1')(UpSampling1D(size=2)(deconv2))
    merge1 = Concatenate()([up1, enconv1])
    deconv1 = convblock(merge1, dim=filter_num, res=res, drop=drop_rate, layername='deconv1', **kwargs2)  
    
    conv10 = Conv1D( filters=num_classes, kernel_size=1, padding='same', activation='relu',
                     name='conv10')(deconv1)

    predictions = Conv1D(filters=num_classes, kernel_size=1, activation='sigmoid',
                          padding='same', name='predictions')(conv10)

    model = Model(inputs=data, outputs= predictions)
    #model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)
    if weights is not None:
        model.load_weights(weights,by_name=True)
    sgd = optimizers.Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=sgd, loss=loss_func)

    return model

if __name__ == '__main__':
    model = unet((2500, 1), 1, 0.001, maxpool=True, weights=None)

