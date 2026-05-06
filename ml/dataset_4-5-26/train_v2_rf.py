"""
Training RandomForest Total People Counting (single-output) berbasis CSI WiFi.

Varian dari `train_v2.py` (XGBoost) yang mengganti model dengan RandomForest:
  - Klasifikasi : RandomForestClassifier  (multi-class)
  - Regresi    : RandomForestRegressor   (hasil di-round ke int)

Flow:
  1. Load merged.npz, baca array `total_orang` langsung.
  2. Group per file -> reconstruct complex CSI (256, n_time)
     -> hapus control subcarriers (256 -> 242).
  3. Preprocessing per file: impute -> Savitzky-Golay -> csi_to_mag_phase.
  4. Windowing 50/25 + extract_features_from_window -> fitur 2420-dim per window.
  5. Split berbasis folds_by_file/fold_{1..K} (split per-file = no leakage).
  6. K-fold CV pakai RF classifier + regressor -> metrik per fold.
  7. Hyperparameter tuning (RandomizedSearchCV) di seluruh data, simpan best model.

Run dari direktori `ml/`:
    python dataset_4-5-26/train_v2_rf.py                      # default: keduanya, 20 iter, 5-fold tune
    python dataset_4-5-26/train_v2_rf.py --task clf           # hanya klasifikasi
    python dataset_4-5-26/train_v2_rf.py --task reg           # hanya regresi
    python dataset_4-5-26/train_v2_rf.py --task both          # keduanya (eksplisit)
    python dataset_4-5-26/train_v2_rf.py --n-iter 50          # lebih banyak iterasi tuning
    python dataset_4-5-26/train_v2_rf.py --tune-cv 3          # jumlah fold CV saat tuning
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

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
FOLDS_DIR = DATA_DIR / "folds_by_file"
DELETE_SUBCARRIER_INDICES = np.asarray(config.DELETE_SUBCARRIER_INDICES, dtype=int)
ALL_LABELS = list(range(7))  # total range 0..6 (label 5 absen di data, tetap dimasukkan)


def discover_fold_ids(folds_dir: Path) -> List[int]:
    """Return sorted fold ids ditemukan dari subfolder `fold_<n>` di `folds_dir`."""
    if not folds_dir.is_dir():
        raise FileNotFoundError(f"Folds dir tidak ditemukan: {folds_dir}")
    ids: List[int] = []
    for child in folds_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("fold_"):
            continue
        suffix = name[len("fold_"):]
        if suffix.isdigit():
            ids.append(int(suffix))
    if not ids:
        raise RuntimeError(f"Tidak ada subfolder `fold_<n>` di {folds_dir}")
    return sorted(ids)


def parse_fold_filelist(path: Path) -> List[str]:
    """Return list of file keys from a fold txt (toleran BOM dan format <idx>\\t<fname>)."""
    files: List[str] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            if "\t" in line:
                _, _, fname = line.partition("\t")
            else:
                fname = line
            fname = fname.strip()
            if fname:
                files.append(fname)
    return files


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
        rows = csi_all[mask]  # (n_packet, 512) int16

        total_vals = total_all[mask]
        assert np.all(total_vals == total_vals[0]), f"total_orang tidak konstan di {fname}"
        total_label = int(total_vals[0])

        rows_f = rows.astype(np.float64)
        csi_complex = (rows_f[:, 0::2] + 1j * rows_f[:, 1::2]).T  # (256, n_time)

        valid_del = DELETE_SUBCARRIER_INDICES[
            DELETE_SUBCARRIER_INDICES < csi_complex.shape[0]
        ]
        csi_complex = np.delete(csi_complex, valid_del, axis=0)  # (242, n_time)

        csi_complex = impute_missing_per_subcarrier(csi_complex)
        csi_complex = apply_savgol_per_subcarrier(csi_complex)
        mag, phase = csi_to_mag_phase(csi_complex)

        idx_list = window_indices(mag.shape[1], WINDOW_SIZE, WINDOW_STRIDE)
        if not idx_list:
            print(
                f"  [{i:>2}/{len(unique_files)}] {fname}: no windows "
                f"(n_time={mag.shape[1]}) — skip"
            )
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


def report_classification(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    acc = accuracy_score(y_true, y_pred)
    print(f"\n[{name}] total accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, labels=ALL_LABELS, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=ALL_LABELS)
    print(f"Confusion matrix (rows=true, cols=pred; labels={ALL_LABELS}):")
    print(cm)
    return acc


def report_regression(
    name: str, y_true: np.ndarray, y_pred_raw: np.ndarray
) -> Tuple[float, float, float]:
    """Return (mae, rmse, exact_match) dan print confusion matrix int-rounded."""
    mae = float(mean_absolute_error(y_true, y_pred_raw))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_raw)))
    y_pred_int = np.clip(np.round(y_pred_raw), 0, 6).astype(int)
    exact = float((y_pred_int == y_true).mean())

    print(f"\n[{name}] regression metrics: MAE={mae:.4f}  RMSE={rmse:.4f}  exact-match={exact:.4f}")
    cm = confusion_matrix(y_true, y_pred_int, labels=ALL_LABELS)
    print(f"Confusion matrix int-rounded (rows=true, cols=pred; labels={ALL_LABELS}):")
    print(cm)
    return mae, rmse, exact


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


# Hyperparameter search space untuk RandomForest (shared clf & reg).
# Keys di-prefix dengan step name saat search (lihat run_task).
RF_PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500, 700],
    "max_depth": [None, 10, 20, 30, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.2, 0.3, 0.5],
}


def run_task(
    task: str,
    X: np.ndarray,
    y: np.ndarray,
    file_per_window: np.ndarray,
    n_iter: int = 20,
    tune_cv: int = 5,
) -> None:
    """task in {'clf', 'reg'}. Jalankan K-fold CV + hyperparameter tuning + final model."""
    assert task in {"clf", "reg"}
    is_clf = task == "clf"
    factory = make_random_forest_pipeline if is_clf else make_random_forest_regression_pipeline
    step_name = "clf" if is_clf else "regressor"
    task_label = "Classification" if is_clf else "Regression"

    fold_ids = discover_fold_ids(FOLDS_DIR)
    n_folds = len(fold_ids)

    print("\n" + "#" * 64)
    print(f"# TASK: {task_label}  ({task})")
    print("#" * 64)

    # RandomForestClassifier mendukung label integer arbitrary, tidak perlu LabelEncoder.
    # Namun tetap di-encode untuk konsistensi pelaporan dengan ALL_LABELS [0..6].
    label_encoder: LabelEncoder | None = None
    y_work = y
    if is_clf:
        label_encoder = LabelEncoder()
        y_work = label_encoder.fit_transform(y).astype(np.int32)
        print(
            f"  label encoder classes: {label_encoder.classes_.tolist()} "
            f"-> encoded 0..{len(label_encoder.classes_) - 1}"
        )

    accs: List[float] = []
    maes: List[float] = []
    rmses: List[float] = []
    exacts: List[float] = []

    for k in fold_ids:
        print(f"\n----- [{task}] Fold {k}/{n_folds} -----")
        train_files = set(parse_fold_filelist(FOLDS_DIR / f"fold_{k}" / "train_files.txt"))
        val_files = set(parse_fold_filelist(FOLDS_DIR / f"fold_{k}" / "val_files.txt"))
        print(f"  train files: {len(train_files)}  val files: {len(val_files)}")

        train_mask = np.fromiter(
            (f in train_files for f in file_per_window),
            dtype=bool,
            count=len(file_per_window),
        )
        val_mask = np.fromiter(
            (f in val_files for f in file_per_window),
            dtype=bool,
            count=len(file_per_window),
        )
        print(f"  train windows: {train_mask.sum()}  val windows: {val_mask.sum()}")
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            raise RuntimeError(
                f"Fold {k}: empty split — verifikasi format nama_file vs train/val_files.txt"
            )

        X_tr, y_tr = X[train_mask], y_work[train_mask]
        X_va, y_va = X[val_mask], y_work[val_mask]

        model = factory()
        t0 = time.time()
        model.fit(X_tr, y_tr)
        print(f"  fit done in {time.time() - t0:.1f}s")

        y_pred = model.predict(X_va)
        if is_clf:
            assert label_encoder is not None
            y_va_orig = label_encoder.inverse_transform(y_va)
            y_pred_orig = label_encoder.inverse_transform(y_pred.astype(int))
            acc = report_classification(f"[{task}] Fold {k} val", y_va_orig, y_pred_orig)
            accs.append(acc)
        else:
            mae, rmse, exact = report_regression(f"[{task}] Fold {k} val", y_va, y_pred)
            maes.append(mae)
            rmses.append(rmse)
            exacts.append(exact)

    print(f"\n========== [{task}] {n_folds}-Fold CV Summary ==========")
    if is_clf:
        a = np.asarray(accs)
        print(
            f"accuracy   : mean={a.mean():.4f}  std={a.std(ddof=1):.4f}  "
            f"per-fold={a.round(4).tolist()}"
        )
    else:
        m = np.asarray(maes)
        r = np.asarray(rmses)
        e = np.asarray(exacts)
        print(
            f"MAE        : mean={m.mean():.4f}  std={m.std(ddof=1):.4f}  "
            f"per-fold={m.round(4).tolist()}"
        )
        print(
            f"RMSE       : mean={r.mean():.4f}  std={r.std(ddof=1):.4f}  "
            f"per-fold={r.round(4).tolist()}"
        )
        print(
            f"exact-match: mean={e.mean():.4f}  std={e.std(ddof=1):.4f}  "
            f"per-fold={e.round(4).tolist()}"
        )

    # ------------------------------------------------------------------ #
    #  Hyperparameter tuning via RandomizedSearchCV on full dataset      #
    # ------------------------------------------------------------------ #
    print(f"\n[Final-{task}] Hyperparameter tuning on full dataset ...")
    print(f"  RandomizedSearchCV: n_iter={n_iter}, cv={tune_cv}")

    # Prefix param keys with pipeline step name (e.g. "clf__n_estimators")
    param_dist = {f"{step_name}__{k}": v for k, v in RF_PARAM_DIST.items()}
    scoring = "accuracy" if is_clf else "neg_mean_absolute_error"

    search = RandomizedSearchCV(
        factory(),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=tune_cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        verbose=1,
        refit=True,  # refit best estimator on entire X, y_work
        return_train_score=False,
    )

    t0 = time.time()
    search.fit(X, y_work)
    elapsed = time.time() - t0
    print(f"  tuning done in {elapsed:.1f}s")
    print(f"  best params  : {search.best_params_}")
    print(f"  best CV score: {search.best_score_:.4f}  (metric: {scoring})")

    final_model = search.best_estimator_

    out_path = DATA_DIR / f"rf_total_{task}_final.joblib"
    joblib.dump(
        {
            "model": final_model,
            "task": "classification" if is_clf else "regression",
            "labels": ALL_LABELS,
            "label_encoder": label_encoder,  # None untuk regresi; LabelEncoder untuk clf
            "window_size": WINDOW_SIZE,
            "window_stride": WINDOW_STRIDE,
            "delete_subcarrier_indices": DELETE_SUBCARRIER_INDICES.tolist(),
            "feature_dim": int(X.shape[1]),
            "outputs": ["total"],
            "best_params": search.best_params_,
            "best_cv_score": float(search.best_score_),
            "tune_scoring": scoring,
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
        description="Train RandomForest untuk total people counting (clf, reg, atau keduanya)."
    )
    parser.add_argument(
        "--task",
        choices=["clf", "reg", "both"],
        default="both",
        help="Pilih varian model: 'clf' (klasifikasi), 'reg' (regresi), 'both' keduanya. Default: both.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Jumlah iterasi RandomizedSearchCV saat tuning final model. Default: 20.",
    )
    parser.add_argument(
        "--tune-cv",
        type=int,
        default=5,
        help="Jumlah fold CV di dalam RandomizedSearchCV. Default: 5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = ["clf", "reg"] if args.task == "both" else [args.task]

    print("=" * 64)
    print("RandomForest Total People Counting (single-output)")
    print(f"  merged.npz : {MERGED_PATH}")
    print(f"  folds dir  : {FOLDS_DIR}")
    print(f"  window     : {WINDOW_SIZE} / {WINDOW_STRIDE}")
    print(f"  delete sc  : {DELETE_SUBCARRIER_INDICES.tolist()}")
    print(f"  task target: total_orang (range {ALL_LABELS[0]}..{ALL_LABELS[-1]})")
    print(f"  run tasks  : {tasks}")
    print(f"  tuning     : RandomizedSearchCV n_iter={args.n_iter}  cv={args.tune_cv}")
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
        run_task(task, X, y_total, file_per_window, n_iter=args.n_iter, tune_cv=args.tune_cv)

    print("\nDone.")


if __name__ == "__main__":
    main()
