from keras import layers
from keras import Input
from keras.models import Sequential, Model, load_model
from keras.layers import (
    Layer,
    LSTM, GRU,
    TimeDistributed, Bidirectional,
    ReLU, LeakyReLU,
    BatchNormalization,
    Dense, Dropout, Activation, Flatten, 
    Input, Reshape, GRU, CuDNNGRU,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D, AveragePooling1D,
    concatenate, add,
)
from keras.initializers import he_normal, he_uniform, Orthogonal
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
import argparse

from .const import SEED, model_input_len, batch_size, all_labels, nb_leads


class TI_CNN(Sequential):
    """
    """
    def __init__(self, classes:list, input_len:int, cnn:str, bidirectional:bool=True):
        """
        """
        super(Sequential, self).__init__(name='TI_CNN')
        self.classes = classes
        self.nb_classes = len(classes)
        self.input_len = input_len
        self.bidirectional = bidirectional
        self.cnn = cnn.lower()

        if self.cnn == 'vgg':
            self._build_vgg_model()
        elif self.cnn == 'resnet':
            self._build_resnet_model()
        elif self.cnn == 'xception':
            self._build_xception_model()

    def _build_vgg_model(self):
        """
        """
        self.add(
            Input(shape=(self.input_len, nb_leads)), name='input',
        )
        self.add(
            Conv1D(
                filters=64, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv1_1',
            )
        )
        self.add(
            BatchNormalization(name='bn1_1',)
        )
        self.add(
            ReLU(name='relu1_1',)
        )
        self.add(
            Conv1D(
                filters=64, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv1_2',
            )
        )
        self.add(
            BatchNormalization(name='bn1_2',)
        )
        self.add(
            ReLU(name='relu1_2',)
        )
        self.add(
            MaxPooling1D(
                pool_size=3, strides=3,
                name='maxpooling1',
            )
        )
        self.add(
            Conv1D(
                filters=128, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv2_1',
            )
        )
        self.add(
            BatchNormalization(name='bn2_1',)
        )
        self.add(
            ReLU(name='relu2_1',)
        )
        self.add(
            Conv1D(
                filters=128, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv2_2',
            )
        )
        self.add(
            BatchNormalization(name='conv2_2',)
        )
        self.add(
            ReLU(name='relu2_2',)
        )
        self.add(
            MaxPooling1D(
                pool_size=3, strides=3,
                name='maxpooling2',
            )
        )
        self.add(
            Conv1D(
                filters=256, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv3_1',
            )
        )
        self.add(
            BatchNormalization(name='bn3_1',)
        )
        self.add(
            ReLU(name='relu3_1',)
        )
        self.add(
            Conv1D(
                filters=256, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv3_2',
            )
        )

        self.add(
            BatchNormalization(name='bn3_2',)
        )
        self.add(
            ReLU(name='relu3_2',)
        )
        self.add(
            Conv1D(
                filters=256, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv3_3',
            )
        )
        self.add(
            BatchNormalization(name='bn3_3',)
        )
        self.add(
            ReLU(name='relu3_3',)
        )
        self.add(
            MaxPooling1D(
                pool_size=3, strides=3,
                name='maxpooling3',
            )
        )
        self.add(
            Conv1D(
                filters=512, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv4_1',
            )
        )
        self.add(
            BatchNormalization(name='bn4_1',)
        )
        self.add(
            ReLU(name='relu4_1',)
        )
        self.add(
            Conv1D(
                filters=512, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv4_2',
            )
        )

        self.add(
            BatchNormalization(name='bn4_2',)
        )
        self.add(
            ReLU(name='relu4_2',)
        )
        self.add(
            Conv1D(
                filters=512, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv4_3',
            )
        )
        self.add(
            BatchNormalization(name='bn4_3',)
        )
        self.add(
            ReLU(name='relu4_3',)
        )
        self.add(
            MaxPooling1D(
                pool_size=3, strides=3,
                name='maxpooling4',
            )
        )
        self.add(
            Conv1D(
                filters=512, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv5_1',
            )
        )
        self.add(
            BatchNormalization(name='bn5_1',)
        )
        self.add(
            ReLU(name='relu5_1',)
        )
        self.add(
            Conv1D(
                filters=512, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv5_2',
            )
        )

        self.add(
            BatchNormalization(name='bn5_2',)
        )
        self.add(
            ReLU(name='relu5_2',)
        )
        self.add(
            Conv1D(
                filters=512, kernel_size=3, strides=1, padding='same',
                kernel_initializer=he_normal(SEED),
                name='conv5_3',
            )
        )
        self.add(
            BatchNormalization(name='bn5_3',)
        )
        self.add(
            ReLU(name='relu5_3',)
        )
        self.add(
            MaxPooling1D(
                pool_size=3, strides=3,
                name='maxpooling5',
            )
        )

        self.add(
            Bidirectional(LSTM(
                128, kernel_initializer=Orthogonal(seed=SEED),
                return_sequences=True,
                name='bd_lstm1',
            ))
        )

        self.add(
            Bidirectional(LSTM(
                32, kernel_initializer=Orthogonal(seed=SEED),
                return_sequences=True,
                name='bd_lstm2',
            ))
        )

        self.add(
            Bidirectional(LSTM(
                9, kernel_initializer=Orthogonal(seed=SEED),
                return_sequences=False,
                name='bd_lstm3',
            ))
        )

        self.add(
            Dense(
                self.nb_classes,activation='sigmoid',
                name='prediction',
            )
        )

        self.compile(loss='binary_crossentropy', optimizer=Adam(0.0001))

        return self


    def _build_resnet_model(self):
        """
        # ref. the Stanford model
        """
        
        self.compile(loss='binary_crossentropy', optimizer=Adam(0.0001))

        return self


    def _build_xception_model(self):
        """
        #
        """
        
        self.compile(loss='binary_crossentropy', optimizer=Adam(0.0001))

        return self


# -----------------------------------------
# from resnet50 of keras_applications

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 2
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv1D(
        filters1, 1,
        kernel_initializer=he_normal(seed=SEED),
        name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv1D(
        filters2, kernel_size,
        padding='same',
        kernel_initializer=he_normal(seed=SEED),
        name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv1D(
        filters3, 1,
        kernel_initializer=he_normal(seed=SEED),
        name=conv_name_base + '2c'
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=2):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 2
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv1D(
        filters1, 1, strides=strides,
        kernel_initializer=he_normal(seed=SEED),
        name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv1D(
        filters2, kernel_size, padding='same',
        kernel_initializer=he_normal(seed=SEED),
        name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv1D(
        filters3, 1,
        kernel_initializer=he_normal(seed=SEED),
        name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv1D(
        filters3, 1, strides=strides,
        kernel_initializer=he_normal(seed=SEED),
        name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x
    
# -----------------------------------------


if __name__ == '__main__':
    # model = get_model()
    checkpointer = ModelCheckpoint(filepath='./ckpt/weights.hdf5', verbose=1, monitor='val_acc', save_best_only=True)
    csv_logger = CSVLogger('./ckpt/logger.csv')

    # model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, validation_data=(X_test, Y_test), callbacks=[checkpointer])

    # TODO: add the following callbacks:
    # LearningRateScheduler, CSVLogger
