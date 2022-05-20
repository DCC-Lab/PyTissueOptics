import warnings

try:
    from tqdm import tqdm as progressBar
except ImportError:
    def progressBar(iterable, *args, **kwargs):
        warnings.warn("Package 'tqdm' not found. Progress bar will not be shown.")
        return iterable
