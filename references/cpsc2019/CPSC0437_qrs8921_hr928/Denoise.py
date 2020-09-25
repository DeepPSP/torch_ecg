import smooth
import pywt


def load_whole_multilabel_remove_BW_WaveletDenoise(datapath, classes, leads = None):
# remove baseline wander
		for i in range(signal.shape[1]):
			smoothed_signal = smooth.smooth(signal[:, i], window_len=250, window='flat')
			signal[:, i] = signal[:, i] - smoothed_signal

		# denoise ECG
		for i in range(signal.shape[1]):
			# DWT
			coeffs = pywt.wavedec(signal[:, i], 'db4', level=5)
			# compute threshold
			noiseSigma = 0.01
			threshold = noiseSigma* math.sqrt(2 * math.log2(signal[:, i].size))
			# apply threshold
			newcoeffs = coeffs
			for j in range(len(newcoeffs)):
				newcoeffs[j] = pywt.threshold(coeffs[j], threshold, mode='soft')

			# IDWT
			signal[:, i] = pywt.waverec(newcoeffs, 'db4')[0:len(signal)]