import math
import numpy as np


def normalize_data(data):
    """
    Normalize such that the mean of the input is 0 and the sample variance is 1

    :param data: The data set, expressed as a flat list of floats.
    :type data: list

    :return: The normalized data set, as a flat list of floats.
    :rtype: list
    """

    mean = np.mean(data)
    var = 0

    for _ in data:
        data[data.index(_)] = _ - mean

    for _ in data:
        var += math.pow(_, 2)

    var = math.sqrt(var / float(len(data)))

    for _ in data:
        data[data.index(_)] = _ / var

    return data
