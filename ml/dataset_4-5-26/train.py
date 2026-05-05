"""
Training XGBoost Multi-Output untuk klasifikasi jumlah orang per region
(orang_a, orang_b) berbasis CSI WiFi.

Flow:
  1. Load merged.npz (CSI per-packet untuk 46 file, 794k packets).
  2. Group per file -> reconstruct complex CSI (256, n_time)
     -> hapus control subcarriers (256 -> 242).
  3. Preprocessing per file: impute -> Savitzky-Golay -> csi_to_mag_phase.
  4. Windowing 50/25 + extract_features_from_window -> fitur 2420-dim per window.
  5. Split berbasis folds_by_file/fold_{1..5} (split per-file = no leakage).
  6. 5-fold CV pakai MultiOutputClassifier(XGBoost) -> akurasi per output + exact-match.
  7. Train model final di seluruh data, simpan ke xgb_multioutput_final.joblib.

Run dari direktori `ml/`:
    python dataset_4-5-26/train.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.multioutput import MultiOutputClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils import (
    apply_savgol_per_subcarrier,
    csi_to_mag_phase,
    extract_features_from_window,
    impute_missing_per_subcarrier,
    make_xgboost_pipeline,
    window_indices,
)
from utils import config

WINDOW_SIZE = 50
WINDOW_STRIDE = 25
DATA_DIR = Path(__file__).resolve().parent
MERGED_PATH = DATA_DIR / "merged.npz"
FOLDS_DIR = DATA_DIR / "folds_by_file"
DELETE_SUBCARRIER_INDICES = np.asarray(config.DELETE_SUBCARRIER_INDICES, dtype=int)
ALL_LABELS = list(range(7))  # orang_a, orang_b range 0..6


def parse_fold_filelist(path: Path) -> List[str]:
    """Return list of file keys (e.g. '4/A_4orang_Duduk Cluster.npz') from a fold txt.

    Mendukung dua format baris:
      - "<filename>"               (mis. "0/0orang_a.npz")
      - "<idx>\\t<filename>"        (mis. "1\\t0/0orang_a.npz")
    Nama file boleh mengandung spasi internal (mis. "Duduk Cluster.npz").
    """
    files: List[str] = []
    with open(path, "r", encoding="utf-8-sig") as f:  # utf-8-sig: tolerate optional BOM
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Featurize semua file sekali (one-pass). Return (X, y_a, y_b, file_per_window)."""
    csi_all = merged["csi"]
    nama_all = merged["nama_file"]
    a_all = merged["orang_a"]
    b_all = merged["orang_b"]

    unique_files = np.unique(nama_all)
    print(f"Featurize {len(unique_files)} files (window={WINDOW_SIZE}/{WINDOW_STRIDE})")

    X_list: List[np.ndarray] = []
    ya_list: List[int] = []
    yb_list: List[int] = []
    file_list: List[str] = []

    t0 = time.time()
    for i, fname in enumerate(unique_files, start=1):
        mask = nama_all == fname
        rows = csi_all[mask]  # (n_packet, 512) int16

        a_vals = a_all[mask]
        b_vals = b_all[mask]
        assert np.all(a_vals == a_vals[0]), f"orang_a tidak konstan di {fname}"
        assert np.all(b_vals == b_vals[0]), f"orang_b tidak konstan di {fname}"
        a_label = int(a_vals[0])
        b_label = int(b_vals[0])

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
            ya_list.append(a_label)
            yb_list.append(b_label)
            file_list.append(str(fname))

        if i % 5 == 0 or i == len(unique_files):
            elapsed = time.time() - t0
            print(
                f"  [{i:>2}/{len(unique_files)}] {fname} "
                f"(a={a_label}, b={b_label}) n_time={mag.shape[1]} "
                f"windows={len(idx_list)} | elapsed {elapsed:.1f}s"
            )

    X = np.vstack(X_list).astype(np.float32)
    y_a = np.asarray(ya_list, dtype=np.int32)
    y_b = np.asarray(yb_list, dtype=np.int32)
    file_per_window = np.asarray(file_list)

    print(
        f"Featurize done. X={X.shape}, y_a={y_a.shape}, y_b={y_b.shape}, "
        f"total time={time.time() - t0:.1f}s"
    )
    return X, y_a, y_b, file_per_window


def report_split(name: str, y_true: np.ndarray, y_pred: np.ndarray, label_name: str) -> None:
    acc = accuracy_score(y_true, y_pred)
    print(f"\n[{name}] {label_name} accuracy: {acc:.4f}")
    print(
        classification_report(
            y_true, y_pred, labels=ALL_LABELS, zero_division=0
        )
    )
    cm = confusion_matrix(y_true, y_pred, labels=ALL_LABELS)
    print(f"Confusion matrix ({label_name}; rows=true, cols=pred; labels={ALL_LABELS}):")
    print(cm)


def main() -> None:
    print("=" * 64)
    print("XGBoost Multi-Output People Counting per Region (orang_a, orang_b)")
    print(f"  merged.npz : {MERGED_PATH}")
    print(f"  folds dir  : {FOLDS_DIR}")
    print(f"  window     : {WINDOW_SIZE} / {WINDOW_STRIDE}")
    print(f"  delete sc  : {DELETE_SUBCARRIER_INDICES.tolist()}")
    print("=" * 64)

    print("\n[1] Loading merged.npz ...")
    merged = np.load(MERGED_PATH, allow_pickle=False)
    print(f"  csi          : {merged['csi'].shape}  {merged['csi'].dtype}")
    print(f"  unique files : {len(np.unique(merged['nama_file']))}")
    print(f"  orang_a uniq : {np.unique(merged['orang_a']).tolist()}")
    print(f"  orang_b uniq : {np.unique(merged['orang_b']).tolist()}")

    print("\n[2] Featurize all files (one-pass) ...")
    X, y_a, y_b, file_per_window = featurize_all_files(merged)
    Y = np.column_stack([y_a, y_b])
    print(f"  Y shape: {Y.shape}")
    print(f"  orang_a dist: {dict(zip(*np.unique(y_a, return_counts=True)))}")
    print(f"  orang_b dist: {dict(zip(*np.unique(y_b, return_counts=True)))}")

    print("\n[3] 5-Fold Cross Validation (split per-file via folds_by_file)")
    accs_a: List[float] = []
    accs_b: List[float] = []
    accs_exact: List[float] = []

    for k in range(1, 6):
        print(f"\n----- Fold {k} -----")
        train_files = set(parse_fold_filelist(FOLDS_DIR / f"fold_{k}" / "train_files.txt"))
        val_files = set(parse_fold_filelist(FOLDS_DIR / f"fold_{k}" / "val_files.txt"))
        print(f"  train files: {len(train_files)}  val files: {len(val_files)}")

        train_mask = np.fromiter(
            (f in train_files for f in file_per_window), dtype=bool, count=len(file_per_window)
        )
        val_mask = np.fromiter(
            (f in val_files for f in file_per_window), dtype=bool, count=len(file_per_window)
        )
        print(f"  train windows: {train_mask.sum()}  val windows: {val_mask.sum()}")
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            raise RuntimeError(
                f"Fold {k}: empty split — verifikasi format nama_file vs train/val_files.txt"
            )

        X_tr, Y_tr = X[train_mask], Y[train_mask]
        X_va, Y_va = X[val_mask], Y[val_mask]

        model = MultiOutputClassifier(make_xgboost_pipeline(), n_jobs=1)
        t0 = time.time()
        model.fit(X_tr, Y_tr)
        print(f"  fit done in {time.time() - t0:.1f}s")

        Y_pred = model.predict(X_va)
        acc_a = accuracy_score(Y_va[:, 0], Y_pred[:, 0])
        acc_b = accuracy_score(Y_va[:, 1], Y_pred[:, 1])
        exact = float((Y_va == Y_pred).all(axis=1).mean())
        print(f"  acc orang_a : {acc_a:.4f}")
        print(f"  acc orang_b : {acc_b:.4f}")
        print(f"  exact match : {exact:.4f}")

        report_split(f"Fold {k} val", Y_va[:, 0], Y_pred[:, 0], "orang_a")
        report_split(f"Fold {k} val", Y_va[:, 1], Y_pred[:, 1], "orang_b")

        accs_a.append(acc_a)
        accs_b.append(acc_b)
        accs_exact.append(exact)

    a_arr = np.asarray(accs_a)
    b_arr = np.asarray(accs_b)
    e_arr = np.asarray(accs_exact)
    print("\n========== 5-Fold CV Summary ==========")
    print(
        f"orang_a    : mean={a_arr.mean():.4f}  std={a_arr.std(ddof=1):.4f}  "
        f"per-fold={a_arr.round(4).tolist()}"
    )
    print(
        f"orang_b    : mean={b_arr.mean():.4f}  std={b_arr.std(ddof=1):.4f}  "
        f"per-fold={b_arr.round(4).tolist()}"
    )
    print(
        f"exact match: mean={e_arr.mean():.4f}  std={e_arr.std(ddof=1):.4f}  "
        f"per-fold={e_arr.round(4).tolist()}"
    )

    print("\n[4] Train final model on full dataset ...")
    final_model = MultiOutputClassifier(make_xgboost_pipeline(), n_jobs=1)
    t0 = time.time()
    final_model.fit(X, Y)
    print(f"  fit done in {time.time() - t0:.1f}s")

    out_path = DATA_DIR / "xgb_multioutput_final.joblib"
    joblib.dump(
        {
            "model": final_model,
            "labels": ALL_LABELS,
            "window_size": WINDOW_SIZE,
            "window_stride": WINDOW_STRIDE,
            "delete_subcarrier_indices": DELETE_SUBCARRIER_INDICES.tolist(),
            "feature_dim": int(X.shape[1]),
            "outputs": ["orang_a", "orang_b"],
        },
        out_path,
    )
    print(f"  saved final model to {out_path}")

    # Feature importance per output
    feat_dim = X.shape[1]
    n_stats = 5
    n_subcarrier = feat_dim // (2 * n_stats)
    feat_names: List[str] = []
    for domain in ("mag", "phase"):
        for sc in range(n_subcarrier):
            for stat in ("mean", "std", "min", "max", "median"):
                feat_names.append(f"{domain}_sc{sc:03d}_{stat}")
    if len(feat_names) != feat_dim:
        feat_names = [f"feature_{i}" for i in range(feat_dim)]

    for out_idx, out_name in enumerate(["orang_a", "orang_b"]):
        pipe = final_model.estimators_[out_idx]  # cloned & fitted Pipeline
        importances = np.asarray(
            pipe.named_steps["clf"].feature_importances_, dtype=float
        )
        df_imp = pd.DataFrame({"feature": feat_names, "importance": importances})
        df_imp = df_imp.sort_values("importance", ascending=False)
        csv_path = DATA_DIR / f"feature_importance_{out_name}.csv"
        df_imp.to_csv(csv_path, index=False)
        top5 = df_imp.head(5)["feature"].tolist()
        print(f"  feature importance ({out_name}): top-5 = {top5} -> {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
