"""
multi-layer perceptron,
heads for making predictions
"""

from easydict import EasyDict as ED


__all__ = [
    "linear",
]


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

