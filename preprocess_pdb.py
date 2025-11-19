#!/usr/bin/env python3
"""
Convert a directory of PDB files into Pointcept-compatible scene folders.

Each atom is treated as a point. For every PDB file we extract:
    - coord.npy: (N, 3) float32 array of XYZ coordinates
    - atom_type.npy: (N, C) float32 one-hot encoding over atom species

Usage:
    python tools/preprocess_pdb.py --input-dir path/to/pdbs --output-dir data/pdb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm


def parse_atom_type(line: str) -> str | None:
    """Return the normalized atom type contained in a PDB line."""
    record = line[0:6].strip().upper()
    if record not in {"ATOM", "HETATM"}:
        return None
    element = line[76:78].strip()
    if not element:
        # Fall back to atom name if element column is empty
        atom_name = line[12:16].strip()
        element = "".join([c for c in atom_name if c.isalpha()])
    return element.upper() if element else None


def parse_atom_coord(line: str) -> Tuple[float, float, float] | None:
    """Return XYZ coordinates encoded in a PDB ATOM/HETATM line."""
    record = line[0:6].strip().upper()
    if record not in {"ATOM", "HETATM"}:
        return None
    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError:
        return None
    return x, y, z


def collect_atom_types(pdb_files: Sequence[Path]) -> List[str]:
    """Scan files once to discover the global atom vocabulary."""
    atom_types = set()
    for pdb_file in tqdm(pdb_files, desc="Scanning atom types"):
        with pdb_file.open("r") as handle:
            for line in handle:
                atom = parse_atom_type(line)
                if atom:
                    atom_types.add(atom)
    if not atom_types:
        raise RuntimeError("No atoms found; please check the input directory.")
    return sorted(atom_types)


def process_file(
    pdb_file: Path,
    out_dir: Path,
    atom_to_idx: Dict[str, int],
) -> None:
    """Convert one pdb file into Pointcept-friendly npy blobs."""
    coords: List[Tuple[float, float, float]] = []
    atom_ids: List[int] = []
    with pdb_file.open("r") as handle:
        for line in handle:
            coord = parse_atom_coord(line)
            atom = parse_atom_type(line)
            if coord is None or atom is None:
                continue
            coords.append(coord)
            atom_ids.append(atom_to_idx[atom])

    if not coords:
        raise RuntimeError(f"No valid atoms found in {pdb_file}")

    coords_arr = np.asarray(coords, dtype=np.float32)
    atom_type_arr = np.zeros(
        (len(coords), len(atom_to_idx)), dtype=np.float32
    )
    atom_type_arr[np.arange(len(coords)), atom_ids] = 1.0

    scene_dir = out_dir / pdb_file.stem
    scene_dir.mkdir(parents=True, exist_ok=True)
    np.save(scene_dir / "coord.npy", coords_arr)
    np.save(scene_dir / "atom_type.npy", atom_type_arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="PDB to Pointcept converter.")
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing .pdb files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Destination directory for processed scenes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if it exists.",
    )
    args = parser.parse_args()

    pdb_files = sorted(args.input_dir.glob("*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No .pdb files found in {args.input_dir}")

    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} already exists. Use --overwrite to reuse it."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    atom_types = collect_atom_types(pdb_files)
    atom_to_idx = {atom: i for i, atom in enumerate(atom_types)}

    for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
        process_file(pdb_file, args.output_dir, atom_to_idx)

    with (args.output_dir / "atom_types.json").open("w") as handle:
        json.dump(atom_types, handle, indent=2)

    print(
        f"Processed {len(pdb_files)} files. "
        f"Atom types ({len(atom_types)}): {', '.join(atom_types)}"
    )


if __name__ == "__main__":
    main()
