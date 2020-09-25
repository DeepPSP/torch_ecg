from keras.models import Model, load_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add,core
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D, AveragePooling1D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.core import Dense, Lambda
from keras.layers.core import Activation
from keras.layers import Input
from keras.layers.merge import concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K

ACTIVATION = "relu"
def __grouped_convolution_block(blockInput, grouped_channels, cardinality, strides, weight_decay=5e-4):
    ''' Adds a grouped convolution block. It is an equivalent block from the paper
    Args:
        input: input tensor
        grouped_channels: grouped number of filters
        cardinality: cardinality factor describing the number of groups
        strides: performs strided convolution for downscaling if > 1
        weight_decay: weight decay term
    Returns: a keras tensor
    '''
    init = blockInput
    group_list = []
    
    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv1D(grouped_channels, 16, padding='same', use_bias=False, strides=(strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    for c in range(cardinality):
#         x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels]
#         if K.image_data_format() == 'channels_last' else
#         lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :])(input)
        x =  Lambda(lambda z: z[:, :, c * grouped_channels:(c + 1) * grouped_channels])(blockInput)
    
        x = Conv1D(grouped_channels, 16, padding='same', use_bias=False, strides=(strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)

        group_list.append(x)

    group_merge = concatenate(group_list)#axis=channel_axis
    x = BatchNormalization()(group_merge)
    x = Activation('relu')(x)

    return x

def resnext_bottleneck_block(blockInput, filters=64, cardinality=8, strides=1, weight_decay=5e-4):
    ''' Adds a bottleneck block
    Args:
        input: input tensor
        filters: number of output filters
        cardinality: cardinality factor described number of
            grouped convolutions
        strides: performs strided convolution for downsampling if > 1
        weight_decay: weight decay factor
    Returns: a keras tensor
    '''
    init = blockInput

    grouped_channels = int(filters / cardinality)
#     channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

#     # Check if input number of filters is same as 16 * k, else create convolution2d for this input
#     if K.image_data_format() == 'channels_first':
#         if init._keras_shape[1] != 2 * filters:
#             init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
#                           use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
#             init = BatchNormalization(axis=channel_axis)(init)
#     else:
#         if init._keras_shape[-1] != 2 * filters:
#             init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
#                           use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
#             init = BatchNormalization(axis=channel_axis)(init)

    init = Conv1D(filters * 2, 1, padding='same', strides=(strides),
                  use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
    init = BatchNormalization()(init)
    
    x = Conv1D(filters, 1, padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(blockInput)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = __grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)

    x = Conv1D(filters * 2, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)

    x = add([init, x])
    x = Activation('relu')(x)

    return x

def convolution_block(x, filters, filter_size, strides=1, padding='same', activation=True):
    x = Conv1D(filters, filter_size, strides=strides, padding=padding,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = Activation(ACTIVATION)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = Activation(ACTIVATION)(blockInput)
    x = BatchNormalization()(x)
    x = convolution_block(x, num_filters, 16 )
    x = convolution_block(x, num_filters, 16, activation=False)
    x = Add()([x, blockInput])
    return x

def bottleneck(blockInput, num_filters=16, block="resnext"):
#     if block == "resnet":
    conv = residual_block(blockInput,num_filters)
    conv = residual_block(blockInput,num_filters)
#     elif block == "resnext":
    # conv = resnext_bottleneck_block(blockInput,num_filters)
    return conv
    
# Build model
def build_model(input_layer, start_neurons,block="resnext", DropoutRatio = 0.5,filter_size=32,nClasses=2,weight_decay=1e-4):
    # 101 -> 50
    conv1 = Conv1D(start_neurons * 1, filter_size, activation=None, padding="same",
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input_layer)
    
    conv1 = bottleneck(conv1,start_neurons * 1, block)
    
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling1D((2))(conv1)
    pool1 = Dropout(DropoutRatio/2)(pool1)

    # 50 -> 25
    conv2 = Conv1D(start_neurons * 2, filter_size, activation=None, padding="same",
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(pool1)
    
    conv2 = bottleneck(conv2,start_neurons * 2, block)
    
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling1D((2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv1D(start_neurons * 4, filter_size, activation=None, padding="same",
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(pool2)

    conv3 = bottleneck(conv3,start_neurons * 4, block)
    
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling1D((2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv1D(start_neurons * 8, filter_size, activation=None, padding="same",
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(pool3)
#     conv4 = residual_block(conv4,start_neurons * 8)
#     conv4 = residual_block(conv4,start_neurons * 8)
    conv4 = bottleneck(conv4,start_neurons * 8, block)
    
    conv4 = Activation(ACTIVATION)(conv4)
    pool4 = MaxPooling1D((2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv1D(start_neurons * 16, filter_size, activation=None, padding="same",
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(pool4)
#     convm = residual_block(convm,start_neurons * 16)
#     convm = residual_block(convm,start_neurons * 16)
    convm = bottleneck(convm,start_neurons * 16, block)
    
    convm = Activation(ACTIVATION)(convm)
    
    # 6 -> 12
    #deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    deconv4 = Conv1D(start_neurons * 8, filter_size,activation='relu', padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)
                     )(UpSampling1D(size=2)(convm))#kernel_initializer='he_normal'
    
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)
    
    uconv4 = Conv1D(start_neurons * 8, filter_size, activation=None, padding="same",
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(uconv4)
#     uconv4 = residual_block(uconv4,start_neurons * 8)
#     uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = bottleneck(uconv4,start_neurons * 8, block)
    
    uconv4 = Activation(ACTIVATION)(uconv4)
    
    # 12 -> 25
    #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    deconv3 = Conv1D(start_neurons * 4, filter_size, activation='relu', padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)
                     )(UpSampling1D(size=2)(uconv4))#kernel_initializer='he_normal'
    uconv3 = concatenate([deconv3, conv3])    
    uconv3 = Dropout(DropoutRatio)(uconv3)
    
    uconv3 = Conv1D(start_neurons * 4, filter_size, activation=None, padding="same",
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(uconv3)
#     uconv3 = residual_block(uconv3,start_neurons * 4)
#     uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = bottleneck(uconv3,start_neurons * 4, block)

    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    #deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    deconv2 = Conv1D(start_neurons * 2, filter_size, activation='relu', padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)
                     )(UpSampling1D(size=2)(uconv3))#kernel_initializer='he_normal'
    uconv2 = concatenate([deconv2, conv2])
        
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv1D(start_neurons * 2, filter_size, activation=None, padding="same",
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(uconv2)
#     uconv2 = residual_block(uconv2,start_neurons * 2)
#     uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = bottleneck(uconv2,start_neurons * 2, block)

    uconv2 = Activation(ACTIVATION)(uconv2)
    
    # 50 -> 101
    #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    deconv1 = Conv1D(start_neurons * 1, filter_size, activation='relu', padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)
                     )(UpSampling1D(size=2)(uconv2))#kernel_initializer='he_normal'
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv1D(start_neurons * 1, filter_size, activation=None, padding="same",
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(uconv1)
#     uconv1 = residual_block(uconv1,start_neurons * 1)
#     uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = bottleneck(uconv1,start_neurons * 1, block)
    
    uconv1 = Activation(ACTIVATION)(uconv1)
    
    uconv1 = Dropout(DropoutRatio/2)(uconv1)
    
    #******************* Deep Super Vision ******************#
    hypercolumn = concatenate(
        [
            uconv1,
            Conv1D(start_neurons * 2, filter_size, activation='relu', padding='same',use_bias=False,
                kernel_regularizer=l2(weight_decay))(UpSampling1D(size=2)(uconv2)),
            Conv1D(start_neurons * 4, filter_size, activation='relu', padding='same',use_bias=False,
                kernel_regularizer=l2(weight_decay))(UpSampling1D(size=4)(uconv3)),
            Conv1D(start_neurons * 8, filter_size, activation='relu', padding='same',use_bias=False,
                kernel_regularizer=l2(weight_decay))(UpSampling1D(size=8)(uconv4))#kernel_initializer='he_normal',
#             Lambda(lambda image: ktf.image.resize_images(image, (img_size_target, img_size_target)))(uconv2),
#             Lambda(lambda image: ktf.image.resize_images(image, (img_size_target, img_size_target)))(uconv3),
#             Lambda(lambda image: ktf.image.resize_images(image, (img_size_target, img_size_target)))(uconv4)
        ]
    )
    hypercolumn = Dropout(0.5)(hypercolumn)
    hypercolumn = Conv1D(start_neurons * 1, filter_size, padding="same", activation='relu',use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(hypercolumn)
    output_layer_noActi = Conv1D(1, 1, padding="same", activation=None,use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(hypercolumn)
    output_layer =  Activation('sigmoid', name='seg_output')(output_layer_noActi)  
        
#     #output_layer = Conv1D(1, 1, padding="same", activation="sigmoid")(uconv1)
#     output_layer = Conv1D(nClasses, 1, activation='relu', padding='same')(uconv1)#kernel_initializer='he_normal'
#     output_layer = core.Reshape((nClasses, input_length))(output_layer)
#     output_layer = core.Permute((2, 1))(output_layer)
#     output_layer = core.Activation('softmax')(output_layer)
#     #model = Model(inputs=inputs, outputs=conv9)
    
    return output_layer
def main():
    input_layer = Input((2560, 1))
    output_layer = build_model(input_layer=input_layer,block="resnext",start_neurons=16,DropoutRatio=0.5,filter_size=32,nClasses=2)
    model = Model(input_layer, output_layer)
    print(model.summary())
    # yaml_string = model_resnextD.to_yaml()
    # open('./unet_model/resnext_unet_deep_supervision_model_architecture_allt.yaml', 'w').write(yaml_string)
if __name__ == '__main__':
    main()
