"""Model definitions and pipelines for CSI-based people counting."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor

from .config import XGBOOST_PARAMS, XGBOOST_REG_PARAMS


def make_xgboost_pipeline() -> Pipeline:
    """Buat pipeline XGBoost untuk klasifikasi multi-class."""
    clf = XGBClassifier(**XGBOOST_PARAMS)

    pipe = Pipeline([
        ("clf", clf),  # TANPA StandardScaler (tree-based)
    ])
    return pipe


def make_xgboost_regression_pipeline() -> Pipeline:
    """Buat pipeline XGBoost untuk regresi ordinal (jumlah orang)."""

    reg = XGBRegressor(**XGBOOST_REG_PARAMS)

    pipe = Pipeline([
        ("regressor", reg),  # TANPA StandardScaler (tree-based)
    ])
    return pipe


def make_knn_pipeline(
    n_neighbors: int = 5,
    weights: str = "distance",
    metric: str = "minkowski",
) -> Pipeline:
    """Buat pipeline k-Nearest Neighbors dengan StandardScaler.

    Parameters
    ----------
    n_neighbors : int, default=5
        Jumlah tetangga terdekat.
    weights : {"uniform", "distance"}, default="distance"
        Skema pembobotan untuk tetangga.
    metric : str, default="minkowski"
        Metrik jarak untuk KNN.
    """

    clf = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
    return pipe
