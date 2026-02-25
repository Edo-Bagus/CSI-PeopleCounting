"""
CSI preprocessing utilities including filtering and transformations.
"""

from typing import Tuple

import numpy as np
from scipy.signal import savgol_filter

from .config import BASE_SAVGOL_WINDOW, SAVGOL_POLYORDER


def impute_missing_per_subcarrier(csi: np.ndarray) -> np.ndarray:
    """
    Imputasi sederhana: ganti NaN/Inf per-subcarrier dengan median baris tersebut.
    
    Parameters
    ----------
    csi : np.ndarray
        CSI matrix kompleks berukuran (n_subcarrier, n_time)
        
    Returns
    -------
    np.ndarray
        CSI matrix kompleks dengan missing values terimputasi
    """
    X = csi.copy()
    real = np.real(X)
    imag = np.imag(X)

    for comp in (real, imag):
        # Ganti Inf dengan NaN agar seragam
        comp[~np.isfinite(comp)] = np.nan
        # Imputasi per baris (subcarrier)
        for i in range(comp.shape[0]):
            row = comp[i]
            if np.all(np.isnan(row)):
                # Jika seluruh baris NaN, isi dengan 0
                comp[i] = 0.0
            else:
                median = np.nanmedian(row)
                # Ganti NaN dengan median
                nan_mask = np.isnan(row)
                if np.any(nan_mask):
                    row[nan_mask] = median
                    comp[i] = row

    return real + 1j * imag


def apply_savgol_per_subcarrier(
    csi: np.ndarray,
    window_length: int = None,
    polyorder: int = None
) -> np.ndarray:
    """
    Terapkan Savitzky-Golay filter di sepanjang sumbu waktu untuk real & imag.
    
    Parameters
    ----------
    csi : np.ndarray
        CSI matrix kompleks berukuran (n_subcarrier, n_time)
    window_length : int, optional
        Panjang window untuk Savitzky-Golay. Default: BASE_SAVGOL_WINDOW dari config
    polyorder : int, optional
        Order polynomial untuk Savitzky-Golay. Default: SAVGOL_POLYORDER dari config
        
    Returns
    -------
    np.ndarray
        CSI matrix kompleks ter-filter
    """
    if window_length is None:
        window_length = BASE_SAVGOL_WINDOW
    if polyorder is None:
        polyorder = SAVGOL_POLYORDER
        
    real = np.real(csi)
    imag = np.imag(csi)
    n_time = real.shape[1]

    # Sesuaikan window length agar valid
    window_length = min(window_length, n_time)
    # window_length harus ganjil dan > polyorder
    if window_length % 2 == 0:
        window_length = max(1, window_length - 1)
    if window_length <= polyorder:
        # Jika terlalu pendek, skip filtering
        return csi

    real_filt = savgol_filter(real, window_length=window_length, polyorder=polyorder, axis=1)
    imag_filt = savgol_filter(imag, window_length=window_length, polyorder=polyorder, axis=1)

    return real_filt + 1j * imag_filt


def csi_to_mag_phase(csi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Konversi CSI kompleks ke magnitude dan phase.
    
    Parameters
    ----------
    csi : np.ndarray
        CSI matrix kompleks berukuran (n_subcarrier, n_time)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple berisi (magnitude, phase), masing-masing shape (n_subcarrier, n_time)
    """
    mag = np.abs(csi)
    phase = np.angle(csi)
    return mag, phase
