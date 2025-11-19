#!/usr/bin/env python3
"""
Convert ligand SDF files into Pointcept-compatible scene folders.

Each atom is treated as a point. For each .sdf file we write:
    - coord.npy: (N, 3) float32 positions
    - atom_type.npy: (N, C) float32 one-hot encoding of atom species

Usage:
    python preprocess_sdf.py --input-dir path/to/sdf --output-dir data/ligands
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm


def split_molecules(lines: Sequence[str]) -> List[List[str]]:
    """Split raw SDF content into molecule blocks."""
    blocks: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if line.strip() == "$$$$":
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)
    return blocks


def parse_counts_line(line: str) -> int:
    """Extract the atom count from the counts line."""
    # Try fixed-width format first, then fallback to token parsing.
    token = line[0:3].strip()
    if token.isdigit():
        return int(token)
    parts = line.split()
    if not parts:
        raise ValueError("Invalid counts line in SDF file.")
    return int(parts[0])


def parse_atoms_from_block(block: Sequence[str]) -> Tuple[List[Tuple[float, float, float]], List[str]]:
    """Parse coordinates and atom symbols from a single molecule block."""
    if len(block) < 4:
        raise RuntimeError("Incomplete SDF block encountered.")
    counts_line = block[3]
    num_atoms = parse_counts_line(counts_line)
    atom_lines = block[4 : 4 + num_atoms]

    coords: List[Tuple[float, float, float]] = []
    atoms: List[str] = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            x, y, z = map(float, parts[:3])
        except ValueError:
            continue
        coords.append((x, y, z))
        atoms.append(parts[3].upper())

    if len(coords) != num_atoms:
        raise RuntimeError("Mismatch between atom count and parsed atoms.")
    return coords, atoms


def collect_atom_types(sdf_files: Sequence[Path]) -> List[str]:
    """Scan all SDF files to build the atom vocabulary."""
    atom_types = set()
    for sdf_file in tqdm(sdf_files, desc="Scanning atom types"):
        lines = sdf_file.read_text().splitlines()
        blocks = split_molecules(lines)
        if not blocks:
            continue
        coords, atoms = parse_atoms_from_block(blocks[0])
        atom_types.update(atoms)
    if not atom_types:
        raise RuntimeError("No atoms detected in the provided SDF files.")
    return sorted(atom_types)


def process_file(
    sdf_file: Path,
    out_dir: Path,
    atom_to_idx: Dict[str, int],
) -> None:
    """Convert a single SDF file into coord.npy and atom_type.npy."""
    lines = sdf_file.read_text().splitlines()
    blocks = split_molecules(lines)
    if not blocks:
        raise RuntimeError(f"No molecule block found in {sdf_file}")

    coords, atoms = parse_atoms_from_block(blocks[0])
    if not coords:
        raise RuntimeError(f"No atoms parsed in {sdf_file}")

    coords_arr = np.asarray(coords, dtype=np.float32)
    atom_type_arr = np.zeros((len(coords), len(atom_to_idx)), dtype=np.float32)
    atom_indices = [atom_to_idx[a] for a in atoms]
    atom_type_arr[np.arange(len(coords)), atom_indices] = 1.0

    scene_dir = out_dir / sdf_file.stem
    scene_dir.mkdir(parents=True, exist_ok=True)
    np.save(scene_dir / "coord.npy", coords_arr)
    np.save(scene_dir / "atom_type.npy", atom_type_arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="SDF to Pointcept converter.")
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing .sdf files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Destination directory for processed ligands.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if it exists.",
    )
    args = parser.parse_args()

    sdf_files = sorted(args.input_dir.glob("*.sdf"))
    if not sdf_files:
        raise FileNotFoundError(f"No .sdf files found in {args.input_dir}")

    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} already exists. Use --overwrite to reuse it."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    atom_types = collect_atom_types(sdf_files)
    atom_to_idx = {atom: i for i, atom in enumerate(atom_types)}

    for sdf_file in tqdm(sdf_files, desc="Processing SDF files"):
        process_file(sdf_file, args.output_dir, atom_to_idx)

    with (args.output_dir / "atom_types.json").open("w") as handle:
        json.dump(atom_types, handle, indent=2)

    print(
        f"Processed {len(sdf_files)} files. "
        f"Atom types ({len(atom_types)}): {', '.join(atom_types)}"
    )


if __name__ == "__main__":
    main()
