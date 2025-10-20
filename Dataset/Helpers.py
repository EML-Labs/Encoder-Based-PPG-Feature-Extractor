import numpy as np

def min_max_scaler(array):
    min_val = np.min(array)
    max_val = np.max(array)
    scaled = 2 * (array - min_val) / (max_val - min_val) - 1
    return scaled