import numpy as np

import warnings
warnings.formatwarning = lambda msg, *args, **kwargs: f'{msg}\n'
warn = warnings.warn


def logNorm(data, eps=1e-6):
    data /= np.max(data)
    data = np.log(data + eps)
    data -= np.min(data)
    data /= np.max(data)
    return data
