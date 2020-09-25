# -*- coding: utf-8 -*-

'''
split long seq into small sub_seq, 
feed sub_seq to lstm

!!!!!!!!!!!deprecated
'''


import numpy as np
import ReadData
from BasicCLF import MyLR
import dill
import MyEval

def read_data_online():
    with open('../data/mimic_data_online_v3.bin', 'rb') as fin:
        _ = np.load(fin)
        mimic_all_feats = np.load(fin)
        all_label = np.load(fin)
    print('mimic_all_feats', mimic_all_feats.shape)
    return mimic_all_feats, all_label
    
def read_data_online_pkl():
    with open('../data/mimic_data_online_v1.pkl', 'rb') as fin:
        data_dict = dill.load(fin)
        mimic_all_feats = data_dict['mimic_all_feats']
        all_label = data_dict['all_label']
    print('mimic_all_feats', mimic_all_feats.shape)
    return mimic_all_feats, all_label
    
def read_data_offline():
    with open('../data/mimic_data_online_v1.pkl', 'rb') as fin:
        data_dict = dill.load(fin)
        mimic_all_feats = data_dict['mimic_all_feats']
        all_label = data_dict['all_label']
    print('mimic_all_feats', mimic_all_feats.shape)
    return mimic_all_feats, all_label
    
if __name__ == '__main__':
#     mimic_all_feats, all_label = read_data_online()
    
#     clf = MyLR()
#     clf.fit(mimic_all_feats, np.array(ReadData.OneHot2Label(all_label)))
#     print(clf.clf.coef_)
#     pred_train = clf.predict(mimic_all_feats)
#     MyEval.F1Score3(pred_train, np.array(ReadData.OneHot2Label(all_label)))

#     with open('../model/mimic/mimic_online_LR_v1.1.pkl', 'wb') as fout:
#         dill.dump(clf, fout)
#     print('done')

    with open('../model/mimic/mimic_online_LR_v1.1.pkl', 'rb') as fin:
        m1 = dill.load(fin)
    print(m1.clf.coef_)
    with open('../model/mimic/mimic_online_LR.pkl', 'rb') as fin:
        m2 = dill.load(fin)
    print(m2.clf.coef_)


