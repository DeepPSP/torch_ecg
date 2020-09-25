import numpy as np

from .ti_cnn_model import get_model as ti_cnn


SEED = 42
x = np.load("/mnt/wenhao71/data/cinc2020_data/cpsc_x.npy")
y = np.load("/mnt/wenhao71/data/cinc2020_data/cpsc_y.npy")

if __name__ == '__main__':
    pass
