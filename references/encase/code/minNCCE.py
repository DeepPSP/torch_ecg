import numpy as np
from collections import defaultdict
import math

def ampspaceIndex(x_value,xmin,xmax,phasenums = 4):
    ampinterval = (xmax - xmin) / phasenums
    index = 1
    while xmin < xmax:
        if x_value >= xmin and x_value < xmin + ampinterval:
            return index
        index += 1
        xmin += ampinterval
    return -1

def minimumNCCE(line,phasenums = 4):
    '''
    input: QSR info(R-R interval series) , the number of amplitude part
    method:minimum of the corrected conditional entropy of RR interval sequence
    paper:
    1.Measuring regularity by means of a corrected conditional entropy in sympathetic outflow
    2.Assessment of the dynamics of atrial signals and local atrial period
    series during atrial fibrillation: effects of isoproterenol
    administration(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC529297/)

    :param datas: QSR info from QSRinfo.csv
           phasenums: (xmax - xmin) / phasenums
    :return: [min_NCCE_value,min_L_Index]
    '''
        # normalize
    line = np.array(line, np.float)
    line = line[1:-1]
    # not enough information
    if line.size == 0:
        return [1, -1]
    line = (line - np.mean(line)) / (np.std(line) + 1e-5)
    # useful param
    N = line.size
    xmin = np.min(line)
    xmax = np.max(line)

    L = 1
    E = defaultdict(float)
    CCE = defaultdict(float)
    NCCE = defaultdict(float)
    while L <= N:
        notsingle_c = defaultdict(int)
        single_c = 0
        for index in range(0, N - L + 1):
            index1 = ampspaceIndex(line[index], xmin, xmax, phasenums)
            index2 = ampspaceIndex(line[index + L - 1], xmin, xmax, phasenums)
            if index1 == index2:
                # not single
                notsingle_c[index1] += 1
            else:
                # single
                single_c += 1

        notsingle_array = np.array(list(notsingle_c.values()), np.float)
        notsingle_value = np.dot(1 / notsingle_array, np.log(notsingle_array))

        single_value = single_c / N * math.log(N - L + 1)
        E[L] = single_value + notsingle_value
        EcL = single_c / N * E[1]
        CCE[L] = E[L] - E[L - 1] + EcL
        NCCE[L] = CCE[L] / (E[1] + 1e-5)
        L += 1
    CCE_values = np.array(list(CCE.values()))
    minCCE = CCE_values.min()
    minCCEI = CCE_values.argmin()
    NCCE_values = np.array(list(NCCE.values()))
    minNCCE = NCCE_values.min()
    return [minNCCE, minCCEI]