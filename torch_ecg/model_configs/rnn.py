"""
the modules that follows CNN feature extractor,
mainly RNN, but can also be attention, and linears with non-linear activations
"""
from copy import deepcopy

from easydict import EasyDict as ED


__all__ = [
    "lstm",
    "attention",
]


lstm = ED()
lstm.bias = True
lstm.dropouts = 0.2
lstm.bidirectional = True
lstm.retseq = False
lstm.hidden_sizes = [12*24, 12*6]


attention = ED()
# almost the same with lstm, but the last layer is an attention layer
attention.head_num = 12
attention.bias = True
attention.dropouts = 0.2
attention.bidirectional = True
attention.hidden_sizes = [12*24, 12*6]
attention.nonlinearity = "tanh"


# previously, if rnn is set "none",
# then cnn is followed by only ONE linear layer to make predictions
# split this linear layer into several and adding non-linear activation function
# might be able to let the model learn better classifying hyper-surfaces
linear = ED()
linear.out_channels = [
    256, 64,
]
linear.bias = True
linear.dropouts = 0.2
linear.activation = "mish"
