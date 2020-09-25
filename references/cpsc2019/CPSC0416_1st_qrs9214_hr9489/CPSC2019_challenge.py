import numpy as np

def CPSC2019_challenge(result):
    pos=np.argwhere(result>0.5).flatten()
    rpos = []
    pre = 0
    last = len(pos)
    for j in np.where(np.diff(pos)>2)[0]:
        if j-pre>2:
            rpos.append((pos[pre]+pos[j])*4)
        pre = j+1
    rpos.append((pos[pre]+pos[last-1])*4)
    qrs = np.array(rpos)
    qrs_diff = np.diff(qrs)
    check = True
    while check:
        qrs_diff = np.diff(qrs)
        for r in range(len(qrs_diff)):
            if qrs_diff[r]<100:
                if result[int(qrs[r]/8)]>result[int(qrs[r+1]/8)]:
                    qrs = np.delete(qrs,r+1)
                    check = True
                    break
                else:
                    qrs = np.delete(qrs,r)
                    check = True
                    break
            check = False
    hr = np.array([loc for loc in qrs if (loc > 2750 and loc < 4750)])
    if len(hr)>1:
        hr = round( 60 * 500 / np.mean(np.diff(hr)))
    else:
        hr = 80
    return hr, qrs
