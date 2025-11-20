#!/usr/bin/env python3
"""
Split processed scene folders into train/val/test subsets via symlinks.

Example:
    python split_dataset.py \
        --input-dir data/pdb_processed \
        --output-dir data/pdb_split \
        --ratios 80 10 10
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split processed dataset folders.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Source root.")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Destination root."
    )
    parser.add_argument(
        "--ratios",
        type=int,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(80, 10, 10),
        help="Split ratios (default: 80 10 10).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling folders."
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy folders instead of creating symlinks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it exists.",
    )
    return parser.parse_args()


def list_subfolders(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def prepare_output(root: Path, overwrite: bool) -> None:
    if root.exists():
        if overwrite:
            shutil.rmtree(root)
        else:
            raise FileExistsError(
                f"{root} already exists. Use --overwrite to recreate it."
            )
    for split in ("train", "val", "test"):
        (root / split).mkdir(parents=True, exist_ok=True)


def assign_splits(total: int, ratios: List[int]) -> List[int]:
    ratio_sum = sum(ratios)
    base_counts = [total * r // ratio_sum for r in ratios]
    remainder = total - sum(base_counts)
    for i in range(remainder):
        base_counts[i % len(base_counts)] += 1
    return base_counts


def place_folder(src: Path, dst_root: Path, split: str, copy: bool) -> None:
    dst = dst_root / split / src.name
    if copy:
        shutil.copytree(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main() -> None:
    args = parse_args()
    folders = list_subfolders(args.input_dir)
    if not folders:
        raise RuntimeError(f"No subfolders found in {args.input_dir}")

    random.seed(args.seed)
    random.shuffle(folders)

    prepare_output(args.output_dir, overwrite=args.overwrite)
    metadata = args.input_dir / "atom_types.json"
    if metadata.exists():
        shutil.copy(metadata, args.output_dir / "atom_types.json")

    counts = assign_splits(len(folders), list(args.ratios))
    split_names = ["train", "val", "test"]

    idx = 0
    for split, count in zip(split_names, counts):
        for _ in range(count):
            place_folder(folders[idx], args.output_dir, split, args.copy)
            idx += 1

    print(
        "Split summary: "
        + ", ".join(f"{split}={count}" for split, count in zip(split_names, counts))
    )


if __name__ == "__main__":
    main()
