import warnings
try:
    from tqdm import tqdm
except ImportError:
    def mock_tqdm(iterable, *args, **kwargs):
        warnings.warn("Package 'tqdm' not found. Progress bar will not be shown.")
        return iterable
    tqdm = mock_tqdm
