"""
CSI preprocessing utilities including filtering and transformations.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

from .config import (
    BASE_SAVGOL_WINDOW,
    ENABLE_PHASE_DETRENDING,
    HAMPEL_N_SIGMAS,
    HAMPEL_WINDOW_SIZE,
    PHASE_UNWRAP_THRESHOLD_RAD,
    SAVGOL_POLYORDER,
)


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


def _hampel_filter_real_matrix(
    input_matrix: np.ndarray,
    window_size: int,
    n_sigmas: float,
) -> np.ndarray:
    """
    Hampel filter pada matrix real-valued dengan shape (n_subcarrier, n_time).
    """
    work = np.array(input_matrix, dtype=float, copy=True)
    work[~np.isfinite(work)] = np.nan

    n_time = work.shape[1]
    filtered = work.copy()
    k = 1.4826  # scale factor untuk distribusi Gaussian

    for ti in range(n_time):
        start_time = max(0, ti - window_size)
        end_time = min(n_time, ti + window_size + 1)
        local_window = work[:, start_time:end_time]

        x0 = np.nanmedian(local_window, axis=1)
        s0 = k * np.nanmedian(np.abs(local_window - x0[:, None]), axis=1)

        diff = np.abs(work[:, ti] - x0)
        outlier_mask = diff > (n_sigmas * s0)

        # Jika MAD nol, hindari koreksi agresif ke nilai median.
        outlier_mask[s0 == 0] = False
        filtered[:, ti] = np.where(outlier_mask, x0, work[:, ti])

    for i in range(filtered.shape[0]):
        row = filtered[i]
        if np.all(np.isnan(row)):
            filtered[i] = 0.0
            continue
        row_median = np.nanmedian(row)
        row[np.isnan(row)] = row_median
        filtered[i] = row

    return filtered


def hampel_filter_per_subcarrier(
    csi: np.ndarray,
    window_size: Optional[int] = None,
    n_sigmas: Optional[float] = None,
) -> np.ndarray:
    """
    Terapkan Hampel filter sepanjang sumbu waktu per-subcarrier.

    Parameter
    ---------
    csi : np.ndarray
        CSI matrix shape (n_subcarrier, n_time), real atau kompleks.
    window_size : int, optional
        Radius window Hampel (+/- window_size). Default dari config.
    n_sigmas : float, optional
        Ambang deteksi outlier berbasis MAD. Default dari config.

    Returns
    -------
    np.ndarray
        CSI matrix dengan outlier temporal terganti median lokal.
    """
    if csi.ndim != 2:
        raise ValueError("csi harus 2D dengan shape (n_subcarrier, n_time)")

    if window_size is None:
        window_size = HAMPEL_WINDOW_SIZE
    if n_sigmas is None:
        n_sigmas = HAMPEL_N_SIGMAS

    if window_size < 1:
        return csi.copy()

    real_filt = _hampel_filter_real_matrix(np.real(csi), window_size=window_size, n_sigmas=n_sigmas)
    if np.iscomplexobj(csi):
        imag_filt = _hampel_filter_real_matrix(np.imag(csi), window_size=window_size, n_sigmas=n_sigmas)
        return real_filt + 1j * imag_filt
    return real_filt


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


def unwrap_phase_with_discontinuity_correction(
    csi: np.ndarray,
    threshold_rad: Optional[float] = None,
) -> np.ndarray:
    """
    Unwrap phase pada sumbu waktu + koreksi diskontinuitas antar-subcarrier.

    Parameter
    ---------
    csi : np.ndarray
        CSI matrix kompleks berukuran (n_subcarrier, n_time).
    threshold_rad : float, optional
        Ambang lompatan fase untuk koreksi diskontinuitas (radian).

    Returns
    -------
    np.ndarray
        Phase matrix shape (n_subcarrier, n_time).
    """
    if csi.ndim != 2:
        raise ValueError("csi harus 2D dengan shape (n_subcarrier, n_time)")

    if threshold_rad is None:
        threshold_rad = PHASE_UNWRAP_THRESHOLD_RAD

    phase = np.unwrap(np.angle(csi), axis=1)
    n_subcarrier, n_time = phase.shape

    if n_subcarrier < 2:
        return phase

    for t in range(n_time):
        for _ in range(4):
            diff_sc = np.diff(phase[:, t])
            jump_idx = np.where(np.abs(diff_sc) > threshold_rad)[0]
            if jump_idx.size == 0:
                break

            correction = np.zeros(n_subcarrier, dtype=float)
            for idx in jump_idx:
                correction[idx + 1:] -= 2.0 * np.pi * np.sign(diff_sc[idx])
            phase[:, t] += correction

    return phase


def detrend_phase_by_linear_fit(phase: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Hilangkan tren linear phase antar-subcarrier untuk setiap time index.

    Parameters
    ----------
    phase : np.ndarray
        Phase matrix shape (n_subcarrier, n_time).

    Returns
    -------
    Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        (phase_detrended, (slope_per_time, intercept_per_time)).
    """
    if phase.ndim != 2:
        raise ValueError("phase harus 2D dengan shape (n_subcarrier, n_time)")

    n_subcarrier, n_time = phase.shape
    x = np.arange(n_subcarrier, dtype=float)

    detrended = np.zeros_like(phase, dtype=float)
    slopes = np.full(n_time, np.nan, dtype=float)
    intercepts = np.full(n_time, np.nan, dtype=float)

    for t in range(n_time):
        y = phase[:, t]
        finite_mask = np.isfinite(y)

        if np.sum(finite_mask) < 2:
            detrended[:, t] = y
            continue

        x_fit = x[finite_mask]
        y_fit = y[finite_mask]
        A = np.column_stack((x_fit, np.ones_like(x_fit)))
        slope, intercept = np.linalg.lstsq(A, y_fit, rcond=None)[0]

        trend = slope * x + intercept
        detrended[:, t] = y - trend
        slopes[t] = slope
        intercepts[t] = intercept

    return detrended, (slopes, intercepts)


def csi_to_mag_phase(
    csi: np.ndarray,
    apply_phase_sanitization: bool = False,
    hampel_window_size: Optional[int] = None,
    hampel_n_sigmas: Optional[float] = None,
    unwrap_threshold_rad: Optional[float] = None,
    detrend_phase: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Konversi CSI kompleks ke magnitude dan phase.
    
    Parameters
    ----------
    csi : np.ndarray
        CSI matrix kompleks berukuran (n_subcarrier, n_time)
    apply_phase_sanitization : bool, optional
        Jika True, jalankan alur sanitization phase (Hampel + unwrap correction + optional detrend).
    hampel_window_size : int, optional
        Radius window Hampel. Default dari config.
    hampel_n_sigmas : float, optional
        Ambang outlier Hampel. Default dari config.
    unwrap_threshold_rad : float, optional
        Ambang koreksi diskontinuitas phase. Default dari config.
    detrend_phase : bool, optional
        Jika True, hilangkan tren linear phase antar-subcarrier. Default dari config.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple berisi (magnitude, phase), masing-masing shape (n_subcarrier, n_time)
    """
    mag = np.abs(csi)

    if not apply_phase_sanitization:
        phase = np.unwrap(np.angle(csi), axis=1)
        return mag, phase

    csi_for_phase = hampel_filter_per_subcarrier(
        csi,
        window_size=hampel_window_size,
        n_sigmas=hampel_n_sigmas,
    )
    phase = unwrap_phase_with_discontinuity_correction(
        csi_for_phase,
        threshold_rad=unwrap_threshold_rad,
    )

    if detrend_phase is None:
        detrend_phase = ENABLE_PHASE_DETRENDING
    if detrend_phase:
        phase, _ = detrend_phase_by_linear_fit(phase)

    return mag, phase
