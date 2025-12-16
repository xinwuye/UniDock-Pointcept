#!/usr/bin/env python3
"""
Recursively find structure files and create symlinks into a flat output directory.

Modes:
- protein: match **/*_protein*.pdb, dest filename removes first '_protein'
  Example:
    /path/to/1a0q/1a0q_protein.pdb -> <output_dir>/1a0q.pdb (symlink)
- ligand: match **/*.sdf, dest filename removes first '_ligand' (if present)
  Example:
    /path/to/1a0q/1a0q_ligand.sdf -> <output_dir>/1a0q.sdf (symlink)

Usage:
  python link_protein_pdbs.py --kind protein --input-dir data/pdbbind2020r1/P-L --output-dir data/pdbbind2020r1/protein
  python link_protein_pdbs.py --kind ligand  --input-dir data/pdbbind2020r1/P-L --output-dir data/pdbbind2020r1/ligand
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from tqdm import tqdm


def _dest_name_for(src: Path, *, kind: str) -> str:
    """Return destination filename for a given source file."""
    stem = src.stem
    if kind == "protein":
        # Remove the first occurrence of '_protein' from the stem, keep extension.
        if "_protein" not in stem:
            raise ValueError(f"Source does not contain '_protein' in stem: {src.name}")
        new_stem = stem.replace("_protein", "", 1)
        return f"{new_stem}{src.suffix}"
    if kind == "ligand":
        # Remove the first occurrence of '_ligand' from the stem if present, keep extension.
        new_stem = stem.replace("_ligand", "", 1)
        return f"{new_stem}{src.suffix}"
    raise ValueError(f"Unknown kind: {kind}")


def _safe_symlink(src: Path, dest: Path, *, overwrite: bool, relative: bool) -> str:
    """
    Create dest -> src symlink.
    Returns: 'created' | 'skipped' (already correct) | raises on conflict.
    """
    if dest.exists() or dest.is_symlink():
        # If it's already the correct symlink, skip.
        if dest.is_symlink():
            try:
                existing_target = dest.resolve(strict=True)
            except FileNotFoundError:
                existing_target = None
            if existing_target == src.resolve():
                return "skipped"

        if not overwrite:
            raise FileExistsError(f"Destination exists: {dest}")
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)

    target = os.path.relpath(src, start=dest.parent) if relative else str(src)
    dest.symlink_to(target)
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively link structure files into a flat output directory."
    )
    parser.add_argument(
        "--kind",
        choices=["protein", "ligand"],
        default="protein",
        help="Which files to link: protein=*_protein*.pdb, ligand=*.sdf (default: protein).",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Root directory to search recursively.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to create symlinks in (flat).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destinations if they conflict.",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Create absolute symlinks (default: relative symlinks).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without creating symlinks.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    if args.kind == "protein":
        pattern = "*.pdb"
        candidates = [
            p for p in input_dir.rglob(pattern) if p.is_file() and "_protein" in p.stem
        ]
        empty_msg = f"No matching files found under {input_dir} (pattern: **/*_protein*.pdb)"
    else:
        pattern = "*.sdf"
        candidates = [p for p in input_dir.rglob(pattern) if p.is_file()]
        empty_msg = f"No matching files found under {input_dir} (pattern: **/*.sdf)"

    if not candidates:
        print(empty_msg)
        return

    created = 0
    skipped = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for src in tqdm(sorted(candidates), desc="Linking", unit="file"):
        dest_name = _dest_name_for(src, kind=args.kind)
        dest = output_dir / dest_name

        if args.dry_run:
            link_target = (
                os.path.relpath(src, start=output_dir) if not args.absolute else str(src)
            )
            action = "SKIP" if dest.exists() and not args.overwrite else "LINK"
            print(f"{action}: {dest} -> {link_target}")
            continue

        try:
            result = _safe_symlink(
                src, dest, overwrite=args.overwrite, relative=not args.absolute
            )
        except FileExistsError as e:
            # Name collision is likely when different subfolders share the same basename.
            raise FileExistsError(
                f"{e}\nName collision? Two sources may map to the same dest name: {dest.name}\n"
                "Consider using --overwrite (dangerous) or adjust naming logic."
            ) from e

        if result == "created":
            created += 1
        else:
            skipped += 1

    print(
        f"Done. Found {len(candidates)} sources. Created {created} links, skipped {skipped}."
    )


if __name__ == "__main__":
    main()


