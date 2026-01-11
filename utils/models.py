"""
Model definitions and pipelines for CSI-based people counting.
"""

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import XGBOOST_PARAMS


def make_xgboost_pipeline():
    """
    Buat pipeline XGBoost untuk klasifikasi multi-class.
    
    Returns
    -------
    Pipeline
        Sklearn Pipeline berisi XGBoost classifier
    """
    clf = XGBClassifier(**XGBOOST_PARAMS)
    
    pipe = Pipeline([
        ("clf", clf),   # TANPA StandardScaler
    ])
    return pipe
