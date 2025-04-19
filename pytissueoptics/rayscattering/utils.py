import warnings
from typing import List

import numpy as np

warnings.formatwarning = lambda msg, *args, **kwargs: f'{msg}\n'
warn = warnings.warn


def logNorm(data, eps=1e-6):
    data /= np.max(data)
    data = np.log(data + eps)
    data -= np.min(data)
    data /= np.max(data)
    return data


def labelsEqual(label1: str, label2: str) -> bool:
    if label1 is None and label2 is None:
        return True
    if label1 is None or label2 is None:
        return False
    return label1.lower() == label2.lower()


def labelContained(label: str, labels: List[str]) -> bool:
    if label is None:
        return False
    return any(labelsEqual(label, l) for l in labels)
