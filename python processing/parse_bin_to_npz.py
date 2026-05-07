#!/usr/bin/env python3
"""Parse CSI .bin log(s) into .npz files."""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np


MAGIC = 0xDEADBEEF
HEADER_FMT = "<IBhBbbBBB6sIHBH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
MAX_CSI_LEN = 512


@dataclass
class CsiRecord:
    msg_id: int
    tx_id: int
    rssi: int
    rate: int
    noise_floor: int
    fft_gain: int
    agc_gain: int
    channel: int
    first_word_invalid: int
    mac: str
    timestamp: int
    sig_len: int
    rx_state: int
    csi_len: int
    csi: List[int]


def parse_records(data: bytes, strict: bool = False) -> Iterable[CsiRecord]:
    offset = 0

    if len(data) >= 4:
        (prefix,) = struct.unpack_from("<I", data, 0)
        if prefix == MAGIC:
            offset = 4

    while offset < len(data):
        remaining = len(data) - offset
        if remaining < HEADER_SIZE:
            msg = f"Trailing {remaining} byte(s) ignored at offset {offset}"
            if strict:
                raise ValueError(msg)
            print(f"[WARN] {msg}", file=sys.stderr)
            break

        header = struct.unpack_from(HEADER_FMT, data, offset)
        (
            msg_id,
            tx_id,
            rssi,
            rate,
            noise_floor,
            fft_gain,
            agc_gain,
            channel,
            first_word_invalid,
            mac_raw,
            timestamp,
            sig_len,
            rx_state,
            csi_len,
        ) = header
        offset += HEADER_SIZE

        if csi_len > MAX_CSI_LEN:
            msg = (
                f"Invalid csi_len={csi_len} at record offset {offset - HEADER_SIZE}; "
                f"expected <= {MAX_CSI_LEN}"
            )
            if strict:
                raise ValueError(msg)
            print(f"[WARN] {msg}", file=sys.stderr)
            break

        csi_bytes = csi_len * 2
        if len(data) - offset < csi_bytes:
            msg = (
                f"Incomplete CSI payload at offset {offset}; "
                f"need {csi_bytes} byte(s), have {len(data) - offset}"
            )
            if strict:
                raise ValueError(msg)
            print(f"[WARN] {msg}", file=sys.stderr)
            break

        csi = list(struct.unpack_from(f"<{csi_len}h", data, offset)) if csi_len else []
        offset += csi_bytes

        mac = ":".join(f"{b:02x}" for b in mac_raw)
        yield CsiRecord(
            msg_id=msg_id,
            tx_id=tx_id,
            rssi=rssi,
            rate=rate,
            noise_floor=noise_floor,
            fft_gain=fft_gain,
            agc_gain=agc_gain,
            channel=channel,
            first_word_invalid=first_word_invalid,
            mac=mac,
            timestamp=timestamp,
            sig_len=sig_len,
            rx_state=rx_state,
            csi_len=csi_len,
            csi=csi,
        )


def build_arrays(records: List[CsiRecord]):
    count = len(records)
    max_len = max((r.csi_len for r in records), default=0)

    csi = np.zeros((count, max_len), dtype=np.int16)
    for i, r in enumerate(records):
        if r.csi_len:
            csi[i, : r.csi_len] = np.asarray(r.csi, dtype=np.int16)

    msg_id_raw = np.asarray([r.msg_id for r in records], dtype=np.int64)
    if msg_id_raw.size:
        msg_id_raw -= msg_id_raw.min()

    return {
        "msg_id": msg_id_raw.astype(np.uint32),
        "tx_id": np.asarray([r.tx_id for r in records], dtype=np.uint8),
        "rssi": np.asarray([r.rssi for r in records], dtype=np.int16),
        "rate": np.asarray([r.rate for r in records], dtype=np.uint8),
        "noise_floor": np.asarray([r.noise_floor for r in records], dtype=np.int8),
        "fft_gain": np.asarray([r.fft_gain for r in records], dtype=np.int8),
        "agc_gain": np.asarray([r.agc_gain for r in records], dtype=np.uint8),
        "channel": np.asarray([r.channel for r in records], dtype=np.uint8),
        "first_word_invalid": np.asarray(
            [r.first_word_invalid for r in records], dtype=np.uint8
        ),
        "mac": np.asarray([r.mac for r in records], dtype="U17"),
        "timestamp": np.asarray([r.timestamp for r in records], dtype=np.uint32),
        "sig_len": np.asarray([r.sig_len for r in records], dtype=np.uint16),
        "rx_state": np.asarray([r.rx_state for r in records], dtype=np.uint8),
        "csi_len": np.asarray([r.csi_len for r in records], dtype=np.uint16),
        "csi": csi,
    }


def write_npz(records: List[CsiRecord], output_path: Path) -> None:
    arrays = build_arrays(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def iter_bin_files(src_dir: Path, recursive: bool):
    pattern = "**/*.bin" if recursive else "*.bin"
    for path in src_dir.glob(pattern):
        if path.is_file() and path.suffix.lower() == ".bin":
            yield path


def resolve_output_path(
    input_path: Path, src_root: Path, output_root: Path
) -> Path:
    relative = input_path.relative_to(src_root)
    return output_root / relative.with_suffix(".npz")


def process_one_file(
    input_path: Path,
    output_path: Path,
    strict: bool,
    overwrite: bool,
) -> bool:
    if output_path.exists() and not overwrite:
        print(f"[skip] {input_path.name} -> {output_path} exists")
        return False

    data = input_path.read_bytes()
    records = list(parse_records(data, strict=strict))
    trimmed = records[2000:-2000] if len(records) > 4000 else []
    write_npz(trimmed, output_path)
    print(f"[ok] {input_path.name} -> {output_path} | rows={len(trimmed)} (trimmed from {len(records)})")
    return True

def main() -> int:
    parser = argparse.ArgumentParser(description="Parse CSI .bin into .npz files")
    parser.add_argument("input", type=Path, help="Input .bin file or folder")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .npz for single file input",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output root for folder input (default: input folder)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan subfolders for .bin files",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on truncated/corrupted tail instead of warning",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npz files",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    processed = 0
    if input_path.is_file():
        output_path = args.output or input_path.with_suffix(".npz")
        processed += int(
            process_one_file(
                input_path=input_path,
                output_path=output_path,
                strict=args.strict,
                overwrite=args.overwrite,
            )
        )
    else:
        output_root = (args.output_dir or input_path).resolve()
        for bin_path in iter_bin_files(input_path, args.recursive):
            output_path = resolve_output_path(bin_path, input_path, output_root)
            processed += int(
                process_one_file(
                    input_path=bin_path,
                    output_path=output_path,
                    strict=args.strict,
                    overwrite=args.overwrite,
                )
            )

    print(f"Done. Written: {processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
