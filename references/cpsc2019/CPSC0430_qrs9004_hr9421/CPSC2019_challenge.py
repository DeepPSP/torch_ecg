import numpy as np
from scipy import ndimage, misc, signal


def med_filt_1d(ecg):
    first_filtered = ndimage.median_filter(ecg, size=7)
    second_filtered =  ndimage.median_filter(first_filtered, 215)
    ecg_deno = ecg - second_filtered
    return ecg_deno


def predict_record(model,ecg_data,fs_):
    ecg = ecg_data
    ecg[ecg>10] = 0
    
    ecg = ecg.reshape(1,fs_*10,1)
    
    if len(model) == 4:
        pred_1 = model[0].predict(ecg)
        pred_2 = model[1].predict(ecg)
        pred_3 = model[2].predict(ecg)
        pred_4 = model[3].predict(ecg)

        t =   0.2*pred_1[0,:,1] \
            + 0.2*pred_2[0,:,0] \
            + 0.4*pred_3[0,:,0] \
            + 0.2*(1-0.5*(pred_4[2][0,:,0]+pred_4[3][0,:,0]))
    else:
        print("model length should be 4")

    pred_y = t    

    for l in range(pred_y.shape[0]):
        if(pred_y[l]>=0.5):
            pred_y[l]=1
        else:
            pred_y[l]=0
    
    y_pred = ndimage.median_filter(pred_y, size=3)
    
    return y_pred


def CPSC2019_challenge(model,ecg_data,fs_):
	#调用预测函数
	y_pred = predict_record(model,ecg_data,fs_)
	count=0
	r_ans=[]

	for i in range(0,len(y_pred)):
		if (y_pred[i]>0.5):
			count = count+1
		else:
			if (count>=fs_*0.1*0.5):#60
				ind = i- round(count*0.5)
				r_ans.append(ind)

			count=0

	if (count>=fs_*0.1*0.5):#60
		ind = len(y_pred)- round(count*0.5)
		r_ans.append(ind)

	#r_ans =np.array(r_ans)
	#print("r_ans :",r_ans)

	#r_rect1 = detect_leak(r_ans, y_pred)
	#r_rect2 = delete_extra(ecg_data, r_rect1)

	r_rect2 = np.array(r_ans)
	r_rect2 = r_rect2[(r_rect2 >= 0.5*fs_ - 0.075*fs_) & (r_rect2 <= 9.5*fs_ + 0.075*fs_)]

	# 计算心率值      
	r_hr = np.array([loc for loc in r_rect2 if ((loc > 5.5 * fs_) and (loc < len(ecg_data) - 0.5 * fs_))])
	hr_ans = round( 60 * fs_ / np.mean(np.diff(r_hr)))

	if(np.isnan(hr_ans)):
		hr_ans = 0
	return hr_ans, r_rect2

def CPSC2019_challengeNew(ECG):
    '''
    This function is your detection function, the input parameters is self-definedself.

    INPUT:
    ECG: single ecg data for 10 senonds
    .
    .
    .

    OUTPUT:
    hr: heart rate calculated based on the ecg data
    qrs: R peak location detected beased on the ecg data and your algorithm

    '''
    hr = 10
    qrs = np.arange(1, 5000, 500)

    return hr, qrs
