import numpy as np
from scipy import stats

def features_from_array(arr):
    feats = {}
    ncols = arr.shape[1]

    for i in range(ncols):
        col = arr[:, i]

        feats[f"c{i}_mean"] = float(np.nanmean(col))
        feats[f"c{i}_std"] = float(np.nanstd(col))
        feats[f"c{i}_median"] = float(np.nanmedian(col))
        feats[f"c{i}_max"] = float(np.nanmax(col))
        feats[f"c{i}_min"] = float(np.nanmin(col))

        try:
            feats[f"c{i}_skew"] = float(stats.skew(col))
        except:
            feats[f"c{i}_skew"] = np.nan

        try:
            feats[f"c{i}_kurt"] = float(stats.kurtosis(col))
        except:
            feats[f"c{i}_kurt"] = np.nan

        feats[f"c{i}_energy"] = float(np.sum(col**2) / len(col))

    return feats
