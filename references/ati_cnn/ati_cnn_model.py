from keras import backend as K
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
    concatenate,
)
from keras import initializers, regularizers, constraints
from keras.initializers import he_normal, he_uniform, Orthogonal
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger

from .const import SEED, model_input_len, batch_size, all_labels, nb_leads


class ATI_CNN(Sequential):
    """
    """
    def __init__(self, classes:list, input_len:int, bidirectional:bool=True):
        """
        """
        super(Sequential, self).__init__(name='ATI_CNN')
        self.classes = classes
        self.nb_classes = len(classes)
        self.input_len = input_len
        self.bidirectional = bidirectional
    
        self._build_model()

    def _build_model(self):
        """
        """
        self.add(
            Conv1D(
                input_shape = (self.input_len, nb_leads),
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


class Attention(Layer):
    """
    """
    def __init__(self,):
        """
        """
        pass


class AttentionWithContext(Layer):
    """
    from CPSC0236
    """
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(
            shape=(input_shape[-1], input_shape[-1],),
            initializer=self.init,
            name='{}_W'.format(self.name),
            regularizer=self.W_regularizer,
            constraint=self.W_constraint
        )
        if self.bias:
            self.b = self.add_weight(
                shape=(input_shape[-1],),
                initializer='zero',
                name='{}_b'.format(self.name),
                regularizer=self.b_regularizer,
                constraint=self.b_constraint
            )
            self.u = self.add_weight(
                shape=(input_shape[-1],),
                initializer=self.init,
                name='{}_u'.format(self.name),
                regularizer=self.u_regularizer,
                constraint=self.u_constraint
            )
        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = self.dot_product(x, self.W)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)
        ait = self.dot_product(uit, self.u)
        a = K.exp(ait)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def dot_product(self, x, kernel):
        """
        """
        if K.backend() == 'tensorflow':
            return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
        else:
            return K.dot(x, kernel)


if __name__ == '__main__':
    # model = get_model()
    checkpointer = ModelCheckpoint(filepath='./ckpt/weights.hdf5', verbose=1, monitor='val_acc', save_best_only=True)
    csv_logger = CSVLogger('./ckpt/logger.csv')

    # model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, validation_data=(X_test, Y_test), callbacks=[checkpointer])

    # TODO: add the following callbacks:
    # LearningRateScheduler, CSVLogger
