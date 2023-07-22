import warnings


def noProgressBar(iterable, *args, **kwargs):
    warnings.warn("Package 'tqdm' not found. Progress bar will not be shown.")
    return iterable


try:
    from tqdm import tqdm as progressBar
except ImportError:
    progressBar = noProgressBar
