#!/usr/bin/env python3
"""Create stratified k-fold train/val splits by file name."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_name_to_indices(names: np.ndarray) -> dict[str, np.ndarray]:
    unique_names, inverse = np.unique(names, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_inverse)) + 1, len(order)]

    name_to_indices: dict[str, np.ndarray] = {}
    for i, name in enumerate(unique_names):
        start = boundaries[i]
        end = boundaries[i + 1]
        name_to_indices[str(name)] = order[start:end]

    return name_to_indices


def build_file_labels(
    name_to_indices: dict[str, np.ndarray],
    labels: np.ndarray,
) -> dict[str, int]:
    file_labels: dict[str, int] = {}
    for name, idx in name_to_indices.items():
        values = labels[idx]
        if values.size == 0:
            raise ValueError(f"File {name} has no labels")
        label = int(values[0])
        if not np.all(values == label):
            unique = np.unique(values)
            raise ValueError(
                f"File {name} has mixed labels: {unique.tolist()}"
            )
        file_labels[name] = label
    return file_labels


def group_files_by_label(file_labels: dict[str, int]) -> dict[int, list[str]]:
    groups: dict[int, list[str]] = {}
    for name, label in file_labels.items():
        groups.setdefault(label, []).append(name)
    return groups


def make_round_robin_folds(
    groups: dict[int, list[str]],
    n_splits: int,
    shuffle: bool,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds: list[list[str]] = [[] for _ in range(n_splits)]

    for label in sorted(groups):
        group = list(groups[label])
        if shuffle:
            rng.shuffle(group)

        for i, name in enumerate(group):
            folds[i % n_splits].append(name)

    return [np.asarray(fold) for fold in folds]


def write_fold(
    fold_dir: Path,
    fold_index: int,
    val_names: np.ndarray,
    all_names: np.ndarray,
    name_to_indices: dict[str, np.ndarray],
) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)

    val_mask = np.isin(all_names, val_names)
    train_names = all_names[~val_mask]

    val_idx = np.concatenate([name_to_indices[str(n)] for n in val_names], axis=0)
    train_idx = np.concatenate([name_to_indices[str(n)] for n in train_names], axis=0)

    val_idx = np.sort(val_idx)
    train_idx = np.sort(train_idx)

    np.savez_compressed(
        fold_dir / "indices.npz",
        train_idx=train_idx.astype(np.int64, copy=False),
        val_idx=val_idx.astype(np.int64, copy=False),
    )

    (fold_dir / "val_files.txt").write_text(
        "\n".join(val_names.astype(str)), encoding="utf-8"
    )
    (fold_dir / "train_files.txt").write_text(
        "\n".join(train_names.astype(str)), encoding="utf-8"
    )

    print(
        f"fold {fold_index}: files train={len(train_names)}, val={len(val_names)} | "
        f"records train={len(train_idx)}, val={len(val_idx)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create k-fold train/val splits by file name"
    )
    parser.add_argument(
        "input_npz",
        type=Path,
        help="Path to merged dataset (.npz) with nama_file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("splitting") / "folds_by_file",
        help="Output folder for folds (default: splitting/folds_by_file)",
    )
    parser.add_argument(
        "--names-key",
        type=str,
        default="nama_file",
        help="Key for file names in the .npz (default: nama_file)",
    )
    parser.add_argument(
        "--label-key",
        type=str,
        default="total_orang",
        help="Key for labels to stratify (default: total_orang)",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of folds (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling of file names before splitting",
    )

    args = parser.parse_args()

    if args.splits < 2:
        raise ValueError("--splits must be at least 2")

    input_path = args.input_npz.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    with np.load(input_path) as data:
        if args.names_key not in data:
            raise KeyError(f"Missing key: {args.names_key}")
        if args.label_key not in data:
            raise KeyError(f"Missing key: {args.label_key}")
        names = np.asarray(data[args.names_key])
        labels = np.asarray(data[args.label_key])

    if names.ndim != 1:
        raise ValueError(f"{args.names_key} must be 1D, got {names.shape}")
    if labels.ndim != 1:
        raise ValueError(f"{args.label_key} must be 1D, got {labels.shape}")
    if len(labels) != len(names):
        raise ValueError(
            f"{args.label_key} length {len(labels)} != {len(names)}"
        )

    unique_names = np.unique(names)
    print(f"total records: {len(names)}")
    print(f"unique files: {len(unique_names)}")

    name_to_indices = build_name_to_indices(names)
    file_labels = build_file_labels(name_to_indices, labels)
    groups = group_files_by_label(file_labels)
    min_files = min(len(group) for group in groups.values())
    effective_splits = args.splits
    if min_files < args.splits:
        effective_splits = min_files
        print(
            f"warning: reducing folds from {args.splits} to {effective_splits} "
            f"because a label has only {min_files} files"
        )

    folds = make_round_robin_folds(
        groups=groups,
        n_splits=effective_splits,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, val_names in enumerate(folds, start=1):
        fold_dir = output_dir / f"fold_{i}"
        write_fold(
            fold_dir=fold_dir,
            fold_index=i,
            val_names=val_names,
            all_names=unique_names,
            name_to_indices=name_to_indices,
        )

    print(f"done -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
