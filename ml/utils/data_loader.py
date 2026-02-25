"""
Data loading utilities for CSI .mat files.
"""

import os
import re
from glob import glob
from typing import List

import numpy as np
from scipy.io import loadmat

from .config import DATASET_ROOT, PC_FOLDERS, DELETE_SUBCARRIER_INDICES


def extract_label_from_filename(filepath: str) -> int:
    """
    Ekstrak label jumlah orang dari nama file, misal *_n04.mat -> 4.
    
    Parameters
    ----------
    filepath : str
        Path ke file .mat
        
    Returns
    -------
    int
        Label jumlah orang
        
    Raises
    ------
    ValueError
        Jika tidak bisa ekstrak label dari nama file
    """
    filename = os.path.basename(filepath)
    match = re.search(r"_n(\d+)\.mat$", filename)
    if not match:
        raise ValueError(f"Tidak bisa ekstrak label dari nama file: {filename}")
    return int(match.group(1))


def load_csi_matrix(
    mat_path: str,
    delete_idxs: np.ndarray = None
) -> np.ndarray:
    """
    Load matrix CSI kompleks dari file .mat dan menghapus subcarrier control signal.

    - Mengambil key pertama non-meta ('__*__')
    - Memastikan data bertipe kompleks
    - Memastikan bentuk (n_subcarrier, n_time)
    - Menghapus subcarrier pada index tertentu (control / unused)

    Parameters
    ----------
    mat_path : str
        Path ke file .mat
    delete_idxs : np.ndarray, optional
        Index subcarrier yang akan dihapus. Default: DELETE_SUBCARRIER_INDICES dari config

    Returns
    -------
    np.ndarray
        CSI matrix kompleks berukuran (n_subcarrier_clean, n_time)
        
    Raises
    ------
    ValueError
        Jika tidak ditemukan variabel usable atau data tidak 2D
    """
    if delete_idxs is None:
        delete_idxs = np.asarray(DELETE_SUBCARRIER_INDICES, dtype=int)
    
    data = loadmat(mat_path)
    usable = {k: v for k, v in data.items() if not k.startswith("__")}
    if not usable:
        raise ValueError(f"Tidak ditemukan variabel usable di {mat_path}")

    key, value = next(iter(usable.items()))
    arr = np.asarray(value)

    if arr.ndim < 2:
        raise ValueError(
            f"CSI di {mat_path} (key={key}) tidak 2D, shape={arr.shape}"
        )

    # Konversi ke kompleks jika perlu
    if not np.iscomplexobj(arr):
        arr = arr.astype(np.complex128)

    # Pastikan bentuk (n_subcarrier, n_time)
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T

    n_subcarrier = arr.shape[0]

    # Filter index yang valid saja
    delete_idxs = delete_idxs[delete_idxs < n_subcarrier]

    # Hapus subcarrier control signal
    arr = np.delete(arr, delete_idxs, axis=0)

    return arr


def collect_all_mat_files() -> List[str]:
    """
    Kumpulkan semua path file .mat dari folder PC-1a ... PC-4a.
    
    Returns
    -------
    List[str]
        List path file .mat yang terurut
    """
    all_files: List[str] = []
    for pc in PC_FOLDERS:
        folder_path = os.path.join(DATASET_ROOT, pc)
        pattern = os.path.join(folder_path, "*.mat")
        files = sorted(glob(pattern))
        all_files.extend(files)
    return all_files
