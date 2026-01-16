"""
Utils package for CSI-based people counting pipeline.

This package provides modular functions for:
- Configuration and constants
- Data loading from .mat files
- CSI preprocessing and filtering
- Feature extraction and windowing
- Model definitions
"""

from . import config
from .data_loader import (
    extract_label_from_filename,
    load_csi_matrix,
    collect_all_mat_files,
)
from .preprocessing import (
    impute_missing_per_subcarrier,
    apply_savgol_per_subcarrier,
    csi_to_mag_phase,
)
from .feature_extraction import (
    window_indices,
    extract_features_from_window,
)
from .models import (
    make_xgboost_pipeline,
    make_xgboost_regression_pipeline,
    make_knn_pipeline,
)

__all__ = [
    # Config module
    "config",
    # Data loader functions
    "extract_label_from_filename",
    "load_csi_matrix",
    "collect_all_mat_files",
    # Preprocessing functions
    "impute_missing_per_subcarrier",
    "apply_savgol_per_subcarrier",
    "csi_to_mag_phase",
    # Feature extraction functions
    "window_indices",
    "extract_features_from_window",
    # Model functions
    "make_xgboost_pipeline",
    "make_xgboost_regression_pipeline",
    "make_knn_pipeline",
]
