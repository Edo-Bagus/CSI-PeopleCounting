"""
Configuration and constants for CSI-based people counting pipeline.
"""

import os

# ================================
# Dataset Configuration
# ================================
DATASET_ROOT = os.path.join("dataset")
PC_FOLDERS = ["PC-1a", "PC-2a", "PC-3a", "PC-4a"]

# ================================
# Preprocessing Parameters
# ================================
BASE_SAVGOL_WINDOW = 11  # akan disesuaikan jika sinyal terlalu pendek
SAVGOL_POLYORDER = 3

# Index subcarrier yang akan dihapus (control signal)
DELETE_SUBCARRIER_INDICES = [0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255]

# ================================
# Feature Extraction Parameters
# ================================
WINDOW_SIZE = 200   # jumlah step waktu per window (heuristik)
WINDOW_STRIDE = 100  # stride antar window (heuristik)

# ================================
# Model Training Parameters
# ================================
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # proporsi terhadap total data (bukan terhadap train saja)

# ================================
# XGBoost Parameters (Classification)
# ================================
XGBOOST_PARAMS = {
    "objective": "multi:softprob",
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "n_jobs": -1,
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
}

# ================================
# XGBoost Parameters (Ordinal Regression)
# ================================
XGBOOST_REG_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "n_jobs": -1,
    "eval_metric": "rmse",
    "random_state": RANDOM_STATE,
}
