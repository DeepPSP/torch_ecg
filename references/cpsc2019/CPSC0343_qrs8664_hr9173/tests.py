import numpy as np
import re
#
# # while True:
# #     a = np.random.rand(1000)
# #     try:
# #         np.median(a)
# #     except RuntimeWarning:
# #         print(a)
# a = 'mitdb_data1003_ch1.dat'
# if re.match('^mitdb.*ch0.dat$', a):
#     print('matched')
# else:
#     print('unmatched')

a = [i for i in range(10)]
b = np.diff(a, axis=0)
print(b)

import pickle
import os
import re

# file_list = []
# dir = 'dat/ppdata_plain_merge'
# for root, dirs, files in os.walk(dir):
#     [file_list.append(f) for f in files]
#
# names = [f.split('.')[0] for f in file_list]
# names = [n for n in filter(lambda x: not x == 'file_names' and not re.match('^preload', x), names)]
# pickle.dump((names), open(os.path.join('dat/ppdata_plain_merge', 'file_names.dat'), 'wb'))
