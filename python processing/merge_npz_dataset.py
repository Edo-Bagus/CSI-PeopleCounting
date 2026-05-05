#!/usr/bin/env python3
"""Merge per-file CSI .npz outputs into a single .npz dataset."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


REQUIRED_KEYS = ("msg_id", "csi")


@dataclass
class Entry:
    path: Path
    rel_name: str
    csi: np.ndarray
    msg_id: np.ndarray
    orang_a: int
    orang_b: int
    total_orang: int


def parse_count_numbers(text: str) -> List[int]:
    match = re.search(r"([0-9]+(?:-[0-9]+)*)\s*orang", text, re.IGNORECASE)
    if match:
        numbers = match.group(1)
    else:
        fallback = re.search(r"([0-9]+(?:-[0-9]+)*)", text)
        if not fallback:
            raise ValueError(f"Cannot parse count from: {text}")
        numbers = fallback.group(1)

    return [int(x) for x in numbers.split("-") if x]


def parse_labels(stem: str) -> Tuple[int, int, int]:
    parts = stem.split("_")
    if not parts:
        raise ValueError(f"Empty filename: {stem}")

    region = parts[0].upper()
    if region == "AB":
        count_part = parts[1] if len(parts) > 1 else ""
        nums = parse_count_numbers(count_part)
        if len(nums) >= 2:
            orang_a, orang_b = nums[0], nums[1]
        elif len(nums) == 1:
            orang_a, orang_b = nums[0], 0
        else:
            raise ValueError(f"Invalid AB count: {stem}")
        total_orang = orang_a + orang_b
        return orang_a, orang_b, total_orang

    if region in {"A", "B"}:
        count_part = parts[1] if len(parts) > 1 else ""
        nums = parse_count_numbers(count_part)
        total_orang = sum(nums)
        if region == "A":
            return total_orang, 0, total_orang
        return 0, total_orang, total_orang

    # Fallback for names like 0orang_a
    try:
        nums = parse_count_numbers(parts[0])
        total_orang = sum(nums)
        return 0, 0, total_orang
    except ValueError:
        pass

    raise ValueError(f"Unrecognized label format: {stem}")


def iter_npz_files(src_dir: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*.npz" if recursive else "*.npz"
    for path in src_dir.glob(pattern):
        if path.is_file() and path.suffix.lower() == ".npz":
            yield path


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(f"Missing keys {missing}")
        csi = np.asarray(data["csi"], dtype=np.int16)
        msg_id = np.asarray(data["msg_id"], dtype=np.uint32)

    if csi.ndim != 2:
        raise ValueError(f"csi must be 2D, got shape {csi.shape}")
    if msg_id.shape[0] != csi.shape[0]:
        raise ValueError(
            f"msg_id length {msg_id.shape[0]} != csi rows {csi.shape[0]}"
        )

    return csi, msg_id


def collect_entries(
    input_dir: Path,
    recursive: bool,
    output_path: Path,
) -> Tuple[List[Entry], int, int, int]:
    entries: List[Entry] = []
    max_width = 0
    total_records = 0
    max_name_len = 1

    for path in sorted(iter_npz_files(input_dir, recursive)):
        if path.resolve() == output_path.resolve():
            continue

        rel_name = path.relative_to(input_dir).as_posix()
        try:
            orang_a, orang_b, total_orang = parse_labels(path.stem)
        except ValueError as exc:
            print(f"[skip] {rel_name} -> {exc}")
            continue

        try:
            csi, msg_id = load_npz(path)
        except ValueError as exc:
            print(f"[skip] {rel_name} -> {exc}")
            continue

        entries.append(
            Entry(
                path=path,
                rel_name=rel_name,
                csi=csi,
                msg_id=msg_id,
                orang_a=orang_a,
                orang_b=orang_b,
                total_orang=total_orang,
            )
        )
        max_width = max(max_width, csi.shape[1])
        total_records += csi.shape[0]
        max_name_len = max(max_name_len, len(rel_name))

    return entries, max_width, total_records, max_name_len


def merge_entries(
    entries: List[Entry],
    max_width: int,
    total_records: int,
    max_name_len: int,
) -> dict:
    csi_out = np.zeros((total_records, max_width), dtype=np.int16)
    msg_id_out = np.zeros(total_records, dtype=np.uint32)
    total_out = np.zeros(total_records, dtype=np.uint16)
    orang_a_out = np.zeros(total_records, dtype=np.uint16)
    orang_b_out = np.zeros(total_records, dtype=np.uint16)
    name_out = np.empty(total_records, dtype=f"<U{max_name_len}")

    offset = 0
    for entry in entries:
        count = entry.csi.shape[0]
        csi_out[offset : offset + count, : entry.csi.shape[1]] = entry.csi
        msg_id_out[offset : offset + count] = entry.msg_id
        total_out[offset : offset + count] = entry.total_orang
        orang_a_out[offset : offset + count] = entry.orang_a
        orang_b_out[offset : offset + count] = entry.orang_b
        name_out[offset : offset + count] = entry.rel_name
        offset += count

    return {
        "csi": csi_out,
        "msg_id": msg_id_out,
        "total_orang": total_out,
        "orang_a": orang_a_out,
        "orang_b": orang_b_out,
        "nama_file": name_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge multiple CSI .npz files into a single dataset"
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Folder containing .npz files (default: current folder)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .npz file (default: <input_dir>/merged.npz)",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan subfolders for .npz files (default: true)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input folder not found: {input_dir}", file=sys.stderr)
        return 1

    output_path = args.output or (input_dir / "merged.npz")
    output_path = output_path.resolve()
    if output_path.exists() and not args.overwrite:
        print(f"Output already exists: {output_path}", file=sys.stderr)
        print("Use --overwrite to replace it.")
        return 1

    entries, max_width, total_records, max_name_len = collect_entries(
        input_dir=input_dir,
        recursive=args.recursive,
        output_path=output_path,
    )

    if not entries:
        print("No valid .npz files found.")
        return 1

    merged = merge_entries(entries, max_width, total_records, max_name_len)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **merged)

    print(
        f"Done. Files: {len(entries)}, records: {total_records}, "
        f"output: {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
