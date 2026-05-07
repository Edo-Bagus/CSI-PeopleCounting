"""
Hyperparameter tuning RandomForest berdasarkan hasil CV dari `train_v2_rf_cv.py`.

Membaca `cv_result_<task>.joblib` yang berisi info best fold (train/val file lists),
lalu menjalankan random search: setiap kandidat di-fit sekali pada best fold's train split
dan di-score pada best fold's val split (tanpa CV internal).

Run dari direktori `ml/`:
    python dataset_4-5-26/train_v2_rf_tune.py                 # default: keduanya, 20 iter
    python dataset_4-5-26/train_v2_rf_tune.py --task clf      # hanya klasifikasi
    python dataset_4-5-26/train_v2_rf_tune.py --task reg      # hanya regresi
    python dataset_4-5-26/train_v2_rf_tune.py --n-iter 50     # lebih banyak kandidat
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils import (
    apply_savgol_per_subcarrier,
    csi_to_mag_phase,
    extract_features_from_window,
    impute_missing_per_subcarrier,
    make_random_forest_pipeline,
    make_random_forest_regression_pipeline,
    window_indices,
)
from utils import config

WINDOW_SIZE = 50
WINDOW_STRIDE = 25
DATA_DIR = Path(__file__).resolve().parent
MERGED_PATH = DATA_DIR / "merged.npz"
DELETE_SUBCARRIER_INDICES = np.asarray(config.DELETE_SUBCARRIER_INDICES, dtype=int)
ALL_LABELS = list(range(7))

RF_PARAM_DIST = {
    "n_estimators": [200, 300, 500],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}


def featurize_all_files(
    merged,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Featurize semua file sekali (one-pass). Return (X, y_total, file_per_window)."""
    csi_all = merged["csi"]
    nama_all = merged["nama_file"]
    total_all = merged["total_orang"]

    unique_files = np.unique(nama_all)
    print(f"Featurize {len(unique_files)} files (window={WINDOW_SIZE}/{WINDOW_STRIDE})")

    X_list: List[np.ndarray] = []
    yt_list: List[int] = []
    file_list: List[str] = []

    t0 = time.time()
    for i, fname in enumerate(unique_files, start=1):
        mask = nama_all == fname
        rows = csi_all[mask]

        total_vals = total_all[mask]
        assert np.all(total_vals == total_vals[0]), f"total_orang tidak konstan di {fname}"
        total_label = int(total_vals[0])

        rows_f = rows.astype(np.float64)
        csi_complex = (rows_f[:, 0::2] + 1j * rows_f[:, 1::2]).T

        valid_del = DELETE_SUBCARRIER_INDICES[
            DELETE_SUBCARRIER_INDICES < csi_complex.shape[0]
        ]
        csi_complex = np.delete(csi_complex, valid_del, axis=0)

        csi_complex = impute_missing_per_subcarrier(csi_complex)
        csi_complex = apply_savgol_per_subcarrier(csi_complex)
        mag, phase = csi_to_mag_phase(csi_complex)

        idx_list = window_indices(mag.shape[1], WINDOW_SIZE, WINDOW_STRIDE)
        if not idx_list:
            continue

        for s, e in idx_list:
            feat = extract_features_from_window(mag[:, s:e], phase[:, s:e])
            X_list.append(feat)
            yt_list.append(total_label)
            file_list.append(str(fname))

        if i % 5 == 0 or i == len(unique_files):
            elapsed = time.time() - t0
            print(
                f"  [{i:>2}/{len(unique_files)}] {fname} "
                f"(total={total_label}) n_time={mag.shape[1]} "
                f"windows={len(idx_list)} | elapsed {elapsed:.1f}s"
            )

    X = np.vstack(X_list).astype(np.float32)
    y_total = np.asarray(yt_list, dtype=np.int32)
    file_per_window = np.asarray(file_list)

    print(
        f"Featurize done. X={X.shape}, y_total={y_total.shape}, "
        f"total time={time.time() - t0:.1f}s"
    )
    return X, y_total, file_per_window


def build_feature_names(feat_dim: int) -> List[str]:
    n_stats = 5
    n_subcarrier = feat_dim // (2 * n_stats)
    feat_names: List[str] = []
    for domain in ("mag", "phase"):
        for sc in range(n_subcarrier):
            for stat in ("mean", "std", "min", "max", "median"):
                feat_names.append(f"{domain}_sc{sc:03d}_{stat}")
    if len(feat_names) != feat_dim:
        feat_names = [f"feature_{i}" for i in range(feat_dim)]
    return feat_names


def run_tune(
    task: str,
    X: np.ndarray,
    y: np.ndarray,
    file_per_window: np.ndarray,
    n_iter: int = 20,
) -> None:
    """Load CV result, reconstruct best fold split, run hyperparameter tuning."""
    assert task in {"clf", "reg"}
    is_clf = task == "clf"
    factory = make_random_forest_pipeline if is_clf else make_random_forest_regression_pipeline
    step_name = "clf" if is_clf else "regressor"
    task_label = "Classification" if is_clf else "Regression"
    metric_name = "accuracy" if is_clf else "MAE"

    print("\n" + "#" * 64)
    print(f"# TASK: {task_label}  ({task})  — Hyperparameter Tuning")
    print("#" * 64)

    cv_path = DATA_DIR / f"cv_result_{task}.joblib"
    if not cv_path.exists():
        raise FileNotFoundError(
            f"CV result tidak ditemukan: {cv_path}\n"
            f"Jalankan train_v2_rf_cv.py --task {task} terlebih dahulu."
        )
    cv = joblib.load(cv_path)
    print(f"  loaded CV result from {cv_path}")
    print(f"  best fold: {cv['best_fold_id']}  ({metric_name}={cv['best_fold_score']:.4f})")
    print(f"  best fold train files: {len(cv['best_train_files'])}")
    print(f"  best fold val files  : {len(cv['best_val_files'])}")

    label_encoder = cv["label_encoder"]
    best_train_files = set(cv["best_train_files"])
    best_val_files = set(cv["best_val_files"])

    y_work = y
    if is_clf:
        assert label_encoder is not None
        y_work = label_encoder.transform(y).astype(np.int32)

    train_mask = np.fromiter(
        (f in best_train_files for f in file_per_window),
        dtype=bool,
        count=len(file_per_window),
    )
    val_mask = np.fromiter(
        (f in best_val_files for f in file_per_window),
        dtype=bool,
        count=len(file_per_window),
    )
    print(f"  train windows: {train_mask.sum()}  val windows: {val_mask.sum()}")
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise RuntimeError("Best fold split menghasilkan empty train/val — cek cv_result file.")

    X_tr, y_tr = X[train_mask], y_work[train_mask]
    X_va, y_va = X[val_mask], y_work[val_mask]

    print(f"\n[Final-{task}] Hyperparameter tuning on best fold (fold {cv['best_fold_id']}) ...")
    print(f"  candidates: n_iter={n_iter} (fit-once each on train, score on val)")
    print(f"  search space: {RF_PARAM_DIST}")

    rng = np.random.RandomState(config.RANDOM_STATE)

    def _sample_params() -> dict:
        return {k: v[int(rng.randint(len(v)))] for k, v in RF_PARAM_DIST.items()}

    best_tune_score: float = -np.inf if is_clf else np.inf
    best_params: dict = {}
    final_model = None

    t0 = time.time()
    for i in range(n_iter):
        params = _sample_params()
        candidate = factory()
        candidate.named_steps[step_name].set_params(**params)
        print(f"  [{i + 1:>3}/{n_iter}] trying params={params}")
        candidate.fit(X_tr, y_tr)
        y_va_pred = candidate.predict(X_va)

        if is_clf:
            score: float = float(accuracy_score(y_va, y_va_pred))
            is_tune_better = score > best_tune_score
        else:
            score = float(mean_absolute_error(y_va, y_va_pred))
            is_tune_better = score < best_tune_score

        print(f"  [{i + 1:>3}/{n_iter}] {metric_name}={score:.4f}")

        if is_tune_better:
            best_tune_score = score
            best_params = params
            final_model = candidate

    elapsed = time.time() - t0
    print(f"  tuning done in {elapsed:.1f}s")
    print(f"  best params  : {best_params}")
    print(f"  best val {metric_name}: {best_tune_score:.4f}")

    out_path = DATA_DIR / f"rf_total_{task}_final.joblib"
    joblib.dump(
        {
            "model": final_model,
            "task": "classification" if is_clf else "regression",
            "labels": ALL_LABELS,
            "label_encoder": label_encoder,
            "window_size": WINDOW_SIZE,
            "window_stride": WINDOW_STRIDE,
            "delete_subcarrier_indices": DELETE_SUBCARRIER_INDICES.tolist(),
            "feature_dim": int(X.shape[1]),
            "outputs": ["total"],
            "best_params": best_params,
            "best_tune_score": float(best_tune_score),
            "tune_metric": metric_name,
        },
        out_path,
    )
    print(f"  saved final model to {out_path}")

    feat_names = build_feature_names(X.shape[1])
    importances = np.asarray(
        final_model.named_steps[step_name].feature_importances_, dtype=float
    )
    df_imp = pd.DataFrame({"feature": feat_names, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False)
    csv_path = DATA_DIR / f"feature_importance_rf_total_{task}.csv"
    df_imp.to_csv(csv_path, index=False)
    top5 = df_imp.head(5)["feature"].tolist()
    print(f"  feature importance ({task}): top-5 = {top5} -> {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning RandomForest dari hasil CV."
    )
    parser.add_argument(
        "--task",
        choices=["clf", "reg", "both"],
        default="both",
        help="Pilih varian model: 'clf', 'reg', atau 'both'. Default: both.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Jumlah kandidat hyperparameter yang dicoba. Default: 20.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = ["clf", "reg"] if args.task == "both" else [args.task]

    print("=" * 64)
    print("RandomForest Hyperparameter Tuning (single-output)")
    print(f"  merged.npz : {MERGED_PATH}")
    print(f"  window     : {WINDOW_SIZE} / {WINDOW_STRIDE}")
    print(f"  delete sc  : {DELETE_SUBCARRIER_INDICES.tolist()}")
    print(f"  run tasks  : {tasks}")
    print(f"  tuning     : fit-once per candidate, n_iter={args.n_iter}")
    print("=" * 64)

    print("\n[1] Loading merged.npz ...")
    merged = np.load(MERGED_PATH, allow_pickle=False)
    print(f"  csi          : {merged['csi'].shape}  {merged['csi'].dtype}")
    print(f"  unique files : {len(np.unique(merged['nama_file']))}")
    print(f"  total uniq   : {np.unique(merged['total_orang']).tolist()}")

    print("\n[2] Featurize all files (one-pass) ...")
    X, y_total, file_per_window = featurize_all_files(merged)
    print(f"  total dist (per-window): {dict(zip(*np.unique(y_total, return_counts=True)))}")

    for task in tasks:
        run_tune(task, X, y_total, file_per_window, n_iter=args.n_iter)

    print("\nDone.")


if __name__ == "__main__":
    main()
