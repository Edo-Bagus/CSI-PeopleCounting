"""
Feature extraction utilities including windowing and statistical features.
"""

from typing import List, Tuple

import numpy as np

from .config import WINDOW_SIZE, WINDOW_STRIDE


def window_indices(
    n_time: int,
    window_size: int = None,
    stride: int = None
) -> List[Tuple[int, int]]:
    """
    Generate (start, end) indices untuk window sliding di domain waktu.
    
    Parameters
    ----------
    n_time : int
        Jumlah time steps total
    window_size : int, optional
        Ukuran window. Default: WINDOW_SIZE dari config
    stride : int, optional
        Stride antar window. Default: WINDOW_STRIDE dari config
        
    Returns
    -------
    List[Tuple[int, int]]
        List berisi tuple (start, end) untuk setiap window
    """
    if window_size is None:
        window_size = WINDOW_SIZE
    if stride is None:
        stride = WINDOW_STRIDE
        
    indices = []
    start = 0
    while start + window_size <= n_time:
        end = start + window_size
        indices.append((start, end))
        start += stride
    return indices


def extract_features_from_window(mag_win: np.ndarray) -> np.ndarray:
    """
    Ekstrak fitur statistik dari satu window (magnitude saja).

    Fitur: statistik per subcarrier (mean, std, min, max, median) untuk magnitude.
    
    Parameters
    ----------
    mag_win : np.ndarray
        Magnitude window berukuran (n_subcarrier, window_size)
        
    Returns
    -------
    np.ndarray
        Vektor fitur 1D (flatten) siap untuk model ML
    """
    stats_funcs = [
        (np.mean, "mean"),
        (np.std, "std"),
        (np.min, "min"),
        (np.max, "max"),
        (np.median, "median"),
    ]

    def compute_stats(arr: np.ndarray) -> np.ndarray:
        feats = []
        for func, _ in stats_funcs:
            feats.append(func(arr, axis=1))  # hasil shape (n_subcarrier,)
        return np.stack(feats, axis=1)  # (n_subcarrier, n_stats)

    mag_feats = compute_stats(mag_win)

    # Flatten jadi 1D
    return mag_feats.ravel()
