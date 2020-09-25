#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:08:47 2017

@author: shenda
"""

import numpy as np
from collections import Counter

def F1_table():
    '''
        # N A O P
    #       Normal	AF	Other	Noisy	Total
    #Normal	Nn	Na	No	Np	∑N
    #   AF	An	Aa	Ao	Ap	∑A
    #Other	On	Oa	Oo	Op	∑O
    #Noisy	Pn	Pa	Po	Pp	∑P
    #Total	∑n	∑a	∑o	∑p	
    '''
    
    # re_table = np.array([[461, 4, 15, 2],
    #                      [4, 54, 6, 0],
    #                      [57, 12, 165, 6],
    #                      [5, 0, 1, 7]])    
    
    re_table = np.array([[3676, 36, 199, 25],
                         [178, 3247, 559, 18],
                         [1049, 229, 2767, 86],
                         [531, 0, 379, 1417]])

                
    F1_N = 2*re_table[0,0] / (sum(re_table[:,0]) + sum(re_table[0,:]))
    F1_A = 2*re_table[1,1] / (sum(re_table[:,1]) + sum(re_table[1,:]))
    F1_O = 2*re_table[2,2] / (sum(re_table[:,2]) + sum(re_table[2,:]))
    F1_P = 2*re_table[3,3] / (sum(re_table[:,3]) + sum(re_table[3,:]))
    
    F1 = (F1_N + F1_A + F1_O + F1_P)/4
        
    return [F1_N, F1_A, F1_O, F1_P, F1]


def F14Exp(pred, groundtruth):
    '''
        # N A O P
    #       Normal	AF	Other	Noisy	Total
    #Normal	Nn	Na	No	Np	∑N
    #   AF	An	Aa	Ao	Ap	∑A
    #Other	On	Oa	Oo	Op	∑O
    #Noisy	Pn	Pa	Po	Pp	∑P
    #Total	∑n	∑a	∑o	∑p	
    '''
    
    pred_len = len(pred)
    re_table =np.zeros([4,4])
    
    for i in range(pred_len):
        
        i_pred = pred[i]
        i_groundtruth = groundtruth[i]

        if i_pred == 'N':
            if i_groundtruth == 'N':
                re_table[0,0] += 1
            elif i_groundtruth == 'A':
                re_table[1,0] += 1
            elif i_groundtruth == 'O':
                re_table[2,0] += 1
            elif i_groundtruth == '~':
                re_table[3,0] += 1
            else:
                print(i_groundtruth, 'wrong label')

        if i_pred == 'A':
            if i_groundtruth == 'N':
                re_table[0,1] += 1
            elif i_groundtruth == 'A':
                re_table[1,1] += 1
            elif i_groundtruth == 'O':
                re_table[2,1] += 1
            elif i_groundtruth == '~':
                re_table[3,1] += 1
            else:
                print(i_groundtruth, 'wrong label')
                
        if i_pred == 'O':
            if i_groundtruth == 'N':
                re_table[0,2] += 1
            elif i_groundtruth == 'A':
                re_table[1,2] += 1
            elif i_groundtruth == 'O':
                re_table[2,2] += 1
            elif i_groundtruth == '~':
                re_table[3,2] += 1
            else:
                print(i_groundtruth, 'wrong label')
                
        if i_pred == '~':
            if i_groundtruth == 'N':
                re_table[0,3] += 1
            elif i_groundtruth == 'A':
                re_table[1,3] += 1
            elif i_groundtruth == 'O':
                re_table[2,3] += 1
            elif i_groundtruth == '~':
                re_table[3,3] += 1
            else:
                print(i_groundtruth, 'wrong label')
                
                
    F1_N = 2*re_table[0,0] / (sum(re_table[:,0]) + sum(re_table[0,:]))
    F1_A = 2*re_table[1,1] / (sum(re_table[:,1]) + sum(re_table[1,:]))
    F1_O = 2*re_table[2,2] / (sum(re_table[:,2]) + sum(re_table[2,:]))
    F1_P = 2*re_table[3,3] / (sum(re_table[:,3]) + sum(re_table[3,:]))
    
    F1 = (F1_N + F1_A + F1_O + F1_P)/4
        
    return [F1_N, F1_A, F1_O, F1_P, F1]

def F1Score4(pred, groundtruth):
    """
        # N A O P
    #       Normal	AF	Other	Noisy	Total
    #Normal	Nn	Na	No	Np	∑N
    #   AF	An	Aa	Ao	Ap	∑A
    #Other	On	Oa	Oo	Op	∑O
    #Noisy	Pn	Pa	Po	Pp	∑P
    #Total	∑n	∑a	∑o	∑p	
    """
    pred_len = len(pred)
    re_table =np.zeros([4,4])
    
    for i in range(pred_len):
        
        i_pred = pred[i]
        i_groundtruth = groundtruth[i]

        if i_pred == 'N':
            if i_groundtruth == 'N':
                re_table[0,0] += 1
            elif i_groundtruth == 'A':
                re_table[1,0] += 1
            elif i_groundtruth == 'O':
                re_table[2,0] += 1
            elif i_groundtruth == '~':
                re_table[3,0] += 1
            else:
                print('wrong label')

        if i_pred == 'A':
            if i_groundtruth == 'N':
                re_table[0,1] += 1
            elif i_groundtruth == 'A':
                re_table[1,1] += 1
            elif i_groundtruth == 'O':
                re_table[2,1] += 1
            elif i_groundtruth == '~':
                re_table[3,1] += 1
            else:
                print('wrong label')
                
        if i_pred == 'O':
            if i_groundtruth == 'N':
                re_table[0,2] += 1
            elif i_groundtruth == 'A':
                re_table[1,2] += 1
            elif i_groundtruth == 'O':
                re_table[2,2] += 1
            elif i_groundtruth == '~':
                re_table[3,2] += 1
            else:
                print('wrong label')
                
        if i_pred == '~':
            if i_groundtruth == 'N':
                re_table[0,3] += 1
            elif i_groundtruth == 'A':
                re_table[1,3] += 1
            elif i_groundtruth == 'O':
                re_table[2,3] += 1
            elif i_groundtruth == '~':
                re_table[3,3] += 1
            else:
                print('wrong label')
                
                
    F1_N = 2*re_table[0,0] / (sum(re_table[:,0]) + sum(re_table[0,:]))
    F1_A = 2*re_table[1,1] / (sum(re_table[:,1]) + sum(re_table[1,:]))
    F1_O = 2*re_table[2,2] / (sum(re_table[:,2]) + sum(re_table[2,:]))
    F1_P = 2*re_table[3,3] / (sum(re_table[:,3]) + sum(re_table[3,:]))
    
    F1 = (F1_N + F1_A + F1_O + F1_P)/4
    
    np.set_printoptions(suppress=True)
    print('N', 'A', 'O', 'P')
    print(re_table)
    print(F1_N, F1_A, F1_O, F1_P)
    print(F1)
    
    return F1

def F1Score3_num(pred, groundtruth):
    """
        # N A O P
    #       Normal	AF	Other	Noisy	Total
    #Normal	Nn	Na	No	Np	∑N
    #   AF	An	Aa	Ao	Ap	∑A
    #Other	On	Oa	Oo	Op	∑O
    #Noisy	Pn	Pa	Po	Pp	∑P
    #Total	∑n	∑a	∑o	∑p	
    """

    gt_list = ["N", "A", "O", "~"]
    pred_1 = [gt_list[ii] for ii in np.argmax(pred, 1)]
    groundtruth_1 = [gt_list[ii] for ii in np.argmax(groundtruth, 1)]

    return F1Score3(pred_1, groundtruth_1)

def F1Score3(pred, groundtruth, return_re=True):
    '''
        # N A O P
    #       Normal	AF	Other	Noisy	Total
    #Normal	Nn	Na	No	Np	∑N
    #   AF	An	Aa	Ao	Ap	∑A
    #Other	On	Oa	Oo	Op	∑O
    #Noisy	Pn	Pa	Po	Pp	∑P
    #Total	∑n	∑a	∑o	∑p	
    '''
    
    pred_len = len(pred)
    re_table =np.zeros([4,4])
    
    for i in range(pred_len):
        
        i_pred = pred[i]
        i_groundtruth = groundtruth[i]

        if i_pred == 'N':
            if i_groundtruth == 'N':
                re_table[0,0] += 1
            elif i_groundtruth == 'A':
                re_table[1,0] += 1
            elif i_groundtruth == 'O':
                re_table[2,0] += 1
            elif i_groundtruth == '~':
                re_table[3,0] += 1
            else:
                print(i_groundtruth, 'wrong label')

        if i_pred == 'A':
            if i_groundtruth == 'N':
                re_table[0,1] += 1
            elif i_groundtruth == 'A':
                re_table[1,1] += 1
            elif i_groundtruth == 'O':
                re_table[2,1] += 1
            elif i_groundtruth == '~':
                re_table[3,1] += 1
            else:
                print(i_groundtruth, 'wrong label')
                
        if i_pred == 'O':
            if i_groundtruth == 'N':
                re_table[0,2] += 1
            elif i_groundtruth == 'A':
                re_table[1,2] += 1
            elif i_groundtruth == 'O':
                re_table[2,2] += 1
            elif i_groundtruth == '~':
                re_table[3,2] += 1
            else:
                print(i_groundtruth, 'wrong label')
                
        if i_pred == '~':
            if i_groundtruth == 'N':
                re_table[0,3] += 1
            elif i_groundtruth == 'A':
                re_table[1,3] += 1
            elif i_groundtruth == 'O':
                re_table[2,3] += 1
            elif i_groundtruth == '~':
                re_table[3,3] += 1
            else:
                print(i_groundtruth, 'wrong label')
                
                
    F1_N = 2*re_table[0,0] / (sum(re_table[:,0]) + sum(re_table[0,:]))
    F1_A = 2*re_table[1,1] / (sum(re_table[:,1]) + sum(re_table[1,:]))
    F1_O = 2*re_table[2,2] / (sum(re_table[:,2]) + sum(re_table[2,:]))
    
    F1 = (F1_N + F1_A + F1_O)/3
    acc = (re_table[0,0] + re_table[1,1] + re_table[2,2] + re_table[3,3]) / sum(sum(re_table))
    
    np.set_printoptions(suppress=True)
#    print('N', 'A', 'O')
    print(re_table)
#    print(F1_N, F1_A, F1_O)
    print('F1:', F1)
    print('acc:', acc)
    
    if return_re:
        return F1, re_table
    else:
        return F1
    
def F1Score2(pred, groundtruth):
    pred_len = len(pred)
    re_table =np.zeros([2,2])
    
    for i in range(pred_len):
        
        i_pred = pred[i]
        i_groundtruth = groundtruth[i]

        if i_pred == 'N':
            if i_groundtruth == 'N':
                re_table[0,0] += 1
            elif i_groundtruth == 'O':
                re_table[1,0] += 1
            else:
                print('wrong label')
                
        if i_pred == 'O':
            if i_groundtruth == 'N':
                re_table[0,1] += 1
            elif i_groundtruth == 'O':
                re_table[1,1] += 1
            else:
                print('wrong label')
                
                
    F1_X = 2*re_table[0,0] / (sum(re_table[:,0]) + sum(re_table[0,:]))
    F1_P = 2*re_table[1,1] / (sum(re_table[:,1]) + sum(re_table[1,:]))
    
    F1 = (F1_X + F1_P)/2
    
    np.set_printoptions(suppress=True)
    print(re_table)
    print(F1_X, F1_P)
    print(F1)
    return F1
    
def WrongStat(i_fold, pred, groundtruth, test_pid):
    out = []
    pred_len = len(pred)
    re_table =np.zeros([4,4])
    
    for i in range(pred_len):
        
        i_pred = pred[i]
        i_groundtruth = groundtruth[i]

        if i_pred == 'N':
            if i_groundtruth == 'N':
                re_table[0,0] += 1
            elif i_groundtruth == 'A':
                re_table[1,0] += 1
                out.append([i_fold, test_pid[i], 'A', 'N'])
            elif i_groundtruth == 'O':
                re_table[2,0] += 1
                out.append([i_fold, test_pid[i], 'O', 'N'])
            elif i_groundtruth == '~':
                re_table[3,0] += 1
            else:
                print('wrong label')

        if i_pred == 'A':
            if i_groundtruth == 'N':
                re_table[0,1] += 1
                out.append([i_fold, test_pid[i], 'N', 'A'])
            elif i_groundtruth == 'A':
                re_table[1,1] += 1
            elif i_groundtruth == 'O':
                re_table[2,1] += 1
                out.append([i_fold, test_pid[i], 'O', 'A'])
            elif i_groundtruth == '~':
                re_table[3,1] += 1
            else:
                print('wrong label')
                
        if i_pred == 'O':
            if i_groundtruth == 'N':
                re_table[0,2] += 1
                out.append([i_fold, test_pid[i], 'N', 'O'])
            elif i_groundtruth == 'A':
                re_table[1,2] += 1
                out.append([i_fold, test_pid[i], 'A', 'O'])
            elif i_groundtruth == 'O':
                re_table[2,2] += 1
            elif i_groundtruth == '~':
                re_table[3,2] += 1
            else:
                print('wrong label')
                
        if i_pred == '~':
            if i_groundtruth == 'N':
                re_table[0,3] += 1
            elif i_groundtruth == 'A':
                re_table[1,3] += 1
            elif i_groundtruth == 'O':
                re_table[2,3] += 1
            elif i_groundtruth == '~':
                re_table[3,3] += 1
            else:
                print('wrong label')
                
    return out

def F1ScoreNoise(pred, groundtruth):
    pred_len = len(pred)
    re_table =np.zeros([2,2])
    
    for i in range(pred_len):
        
        i_pred = pred[i]
        i_groundtruth = groundtruth[i]

        if i_pred == 1:
            if i_groundtruth == 1:
                re_table[0,0] += 1
            elif i_groundtruth == 0:
                re_table[1,0] += 1
            else:
                print('wrong label')
                
        if i_pred == 0:
            if i_groundtruth == 1:
                re_table[0,1] += 1
            elif i_groundtruth == 0:
                re_table[1,1] += 1
            else:
                print('wrong label')
                
                
    F1_NotP = 2*re_table[0,0] / (sum(re_table[:,0]) + sum(re_table[0,:]))
    F1_P = 2*re_table[1,1] / (sum(re_table[:,1]) + sum(re_table[1,:]))
    
    F1 = (F1_NotP + F1_P)/2
    
    np.set_printoptions(suppress=True)
    print(re_table)
    print(F1_NotP, F1_P)
    print(F1)
    
    
    
if __name__ == '__main__':
    print(F1_table())