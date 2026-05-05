import argparse
import re
import shutil
from pathlib import Path


def parse_total_people(stem: str) -> int:
    parts = stem.split("_")
    if len(parts) < 1:
        raise ValueError(f"Filename has no count part: {stem}")

    count_part = parts[1] if len(parts) > 1 else ""
    match = re.match(r"([0-9]+(?:-[0-9]+)*)\s*orang", count_part, re.IGNORECASE)
    if match:
        numbers = match.group(1)
    else:
        # Fallback: try to extract numbers from any part (handles 0orang_*.bin)
        fallback = re.search(r"([0-9]+(?:-[0-9]+)*)", count_part) or re.search(
            r"([0-9]+(?:-[0-9]+)*)", parts[0]
        )
        if not fallback:
            raise ValueError(f"Cannot parse count part: {count_part}")
        numbers = fallback.group(1)

    return sum(int(x) for x in numbers.split("-") if x)


def unique_target_path(dst_dir: Path, name: str) -> Path:
    target = dst_dir / name
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    for i in range(1, 10000):
        candidate = dst_dir / f"{stem}_dup{i}{suffix}"
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Too many duplicates for {name}")


def iter_bin_files(src_dir: Path, recursive: bool):
    pattern = "**/*.bin" if recursive else "*.bin"
    for path in src_dir.glob(pattern):
        if path.is_file() and path.suffix.lower() == ".bin":
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Group .bin files into folders named by total people count."
    )
    parser.add_argument(
        "src",
        nargs="?",
        default=".",
        help="Source folder containing .bin files (default: current folder)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan subfolders for .bin files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions without copying/moving files",
    )

    args = parser.parse_args()
    src_dir = Path(args.src).resolve()

    if not src_dir.exists() or not src_dir.is_dir():
        raise SystemExit(f"Source folder not found: {src_dir}")

    moved = 0
    errors = 0

    for file_path in iter_bin_files(src_dir, args.recursive):
        try:
            total = parse_total_people(file_path.stem)
        except ValueError as exc:
            errors += 1
            print(f"[skip] {file_path.name} -> {exc}")
            continue

        dst_dir = src_dir / str(total)
        dst_dir.mkdir(parents=True, exist_ok=True)
        target = unique_target_path(dst_dir, file_path.name)

        if args.dry_run:
            action = "COPY" if args.copy else "MOVE"
            print(f"[{action}] {file_path.name} -> {target}")
            moved += 1
            continue

        if args.copy:
            shutil.copy2(file_path, target)
        else:
            shutil.move(file_path, target)
        moved += 1

    print(f"Done. Processed: {moved}, skipped: {errors}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
