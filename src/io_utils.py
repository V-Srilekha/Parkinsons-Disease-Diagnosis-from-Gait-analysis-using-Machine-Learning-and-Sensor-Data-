import os
import numpy as np

def list_signal_files(root, exts=(".txt", ".csv", ".dat")):
    files = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(exts):
                files.append(os.path.join(dp, f))
    return sorted(files)


def load_numeric_file(path):
    """
    SAFEST loader possible.
    Reads file line by line and extracts ONLY numeric rows.
    Ignores corrupted lines, missing values, strings, symbols.
    """

    cleaned_rows = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            # split by ANY whitespace
            parts = line.strip().replace(",", " ").split()

            # convert tokens to floats (skip row if ANY token is not numeric)
            numeric_row = []
            for p in parts:
                try:
                    numeric_row.append(float(p))
                except:
                    numeric_row = []   # mark row as invalid
                    break

            if len(numeric_row) > 0:
                cleaned_rows.append(numeric_row)

    if len(cleaned_rows) == 0:
        return None  # unreadable file

    # make rectangular matrix (pad missing cols with NaN)
    max_len = max(len(r) for r in cleaned_rows)
    final = []

    for r in cleaned_rows:
        if len(r) < max_len:
            r = r + [np.nan] * (max_len - len(r))
        final.append(r)

    return np.array(final, dtype=float)
