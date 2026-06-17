import numpy as np
from scipy import signal

def detrend_signal(arr):
    return signal.detrend(arr, axis=0)

def normalize_columns(arr):
    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, axis=0)
    stds[stds == 0] = 1
    return (arr - means) / stds
