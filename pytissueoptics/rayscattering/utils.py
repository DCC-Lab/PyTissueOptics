import numpy as np


def logNorm(data, eps=1e-6):
    data /= np.max(data)
    data = np.log(data + eps)
    data -= np.min(data)
    data /= np.max(data)
    return data
