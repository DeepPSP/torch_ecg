"""
legacy of the initial versions of ECG_CRNN models
"""
from copy import deepcopy

from keras import layers
from keras import Input
from keras.models import Sequential, Model, load_model
from keras.layers import (
    LSTM, GRU,
    TimeDistributed, Bidirectional,
    ReLU, LeakyReLU,
    BatchNormalization,
    Dense, Dropout, Activation, Flatten, 
    Input, Reshape, GRU, CuDNNGRU,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D, AveragePooling1D,
    concatenate,
)
from keras.initializers import he_normal, he_uniform, Orthogonal
from easydict import EasyDict as ED

from model_configs.cnn import vgg16, vgg_block_basic, vgg_block_mish, vgg_block_swish


SEED = 42


def get_model(config:dict):
    """
    """
    cfg = ED(deepcopy(config))

    model = Sequential(name='TI_CNN')

    vgg_block_cfg = deepcopy(vgg_block_basic)

    for block_idx, (num_convs, filters) in enumerate(zip(vgg16.num_convs, vgg16.num_filters)):
        for idx in range(num_convs):
            if block_idx == idx == 0:
                model.add(
                    Conv1D(
                        input_shape=(cfg.input_len, 12),
                        filters=filters,
                        kernel_size=vgg_block_cfg.filter_length,
                        strides=vgg_block_cfg.subsample_length,
                        padding='same',
                        kernel_initializer=he_normal(SEED),
                    )
                )
            else:
                model.add(
                    Conv1D(
                        filters=filters,
                        kernel_size=vgg_block_cfg.filter_length,
                        strides=vgg_block_cfg.subsample_length,
                        padding='same',
                        kernel_initializer=he_normal(SEED),
                    )
                )
            model.add(
                BatchNormalization()
            )
            model.add(
                ReLU()
            )
        model.add(
            MaxPooling1D(
                pool_size=vgg_block_cfg.pool_size,
                strides=vgg_block_cfg.pool_size,
            )
        )

    if cfg.tranches_for_training:
        nb_classes = len(cfg.tranche_classes[cfg.tranches_for_training])
    else:
        nb_classes = len(cfg.classes)

    for units in [256, 64]:
        model.add(
            Bidirectional(LSTM(
                units, kernel_initializer=Orthogonal(seed=SEED),
                return_sequences=True,
            ))
        )
    model.add(
        Bidirectional(LSTM(
            nb_classes, kernel_initializer=Orthogonal(seed=SEED),
            return_sequences=False,
        ))
    )

    model.add(
        Dense(nb_classes, activation='sigmoid')
    )

    return model
