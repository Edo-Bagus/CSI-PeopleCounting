"""Model definitions and pipelines for CSI-based people counting."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor

from .config import XGBOOST_PARAMS, XGBOOST_REG_PARAMS, RANDOM_STATE


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


def make_random_forest_pipeline(
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
) -> Pipeline:
    """Buat pipeline Random Forest untuk klasifikasi multi-class."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    pipe = Pipeline([
        ("clf", clf),
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
