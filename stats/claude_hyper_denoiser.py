import pywt
import numpy as np

def denoise_spectrum(spectrum, wavelet='db8', threshold_mode='soft'):
    diff = np.diff(spectrum)
    sigma = np.std(diff) / np.sqrt(2)
    coeffs = pywt.wavedec(spectrum, wavelet)
    threshold = sigma * np.sqrt(2 * np.log(len(spectrum)))
    coeffs_thresh = [pywt.threshold(c, threshold, threshold_mode) for c in coeffs]
    return pywt.waverec(coeffs_thresh, wavelet)

