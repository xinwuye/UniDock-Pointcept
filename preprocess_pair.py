#!/usr/bin/env python3
"""
Convert paired protein PDB + ligand SDF files into Pointcept-compatible scene folders.

Each atom is treated as a point. For every (protein, ligand) pair we write:
  - coord.npy:     (N, 3) float32 XYZ coordinates (protein atoms then ligand atoms)
  - atom_type.npy: (N, C) float32 one-hot encoding over the UNION atom vocabulary
  - identity.npy:  (N, 2) float32 one-hot encoding: protein=[1,0], ligand=[0,1]

Pairing rule:
  - protein file: <stem>.pdb
  - ligand file:  <stem>.sdf
  - a pair exists if both are present (same stem).

Expected input structure:
  This script looks for files directly under the provided directories (non-recursive):
    --protein-dir/*.pdb
    --ligand-dir/*.sdf
  (You can use the symlink script to flatten a recursive dataset first.)

Usage:
  python preprocess_pair.py --protein-dir data/pdbbind2020r1/pdb --ligand-dir data/pdbbind2020r1/sdf --output-dir pdbbind2020r1/protein-ligand
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm


# -----------------------
# PDB parsing (protein)
# -----------------------


def parse_pdb_atom_type(line: str) -> str | None:
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


def parse_pdb_atom_coord(line: str) -> Tuple[float, float, float] | None:
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


def read_pdb_atoms(pdb_file: Path) -> Tuple[List[Tuple[float, float, float]], List[str]]:
    """Read coordinates and atom symbols from a PDB file."""
    coords: List[Tuple[float, float, float]] = []
    atoms: List[str] = []
    with pdb_file.open("r") as handle:
        for line in handle:
            coord = parse_pdb_atom_coord(line)
            atom = parse_pdb_atom_type(line)
            if coord is None or atom is None:
                continue
            coords.append(coord)
            atoms.append(atom)
    if not coords:
        raise RuntimeError(f"No valid atoms found in {pdb_file}")
    return coords, atoms


# -----------------------
# SDF parsing (ligand)
# -----------------------


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
    token = line[0:3].strip()
    if token.isdigit():
        return int(token)
    parts = line.split()
    if not parts:
        raise ValueError("Invalid counts line in SDF file.")
    return int(parts[0])


def parse_atoms_from_block(
    block: Sequence[str],
) -> Tuple[List[Tuple[float, float, float]], List[str]]:
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


def read_sdf_atoms(sdf_file: Path) -> Tuple[List[Tuple[float, float, float]], List[str]]:
    """Read coordinates and atom symbols from the first molecule in an SDF file."""
    lines = sdf_file.read_text().splitlines()
    blocks = split_molecules(lines)
    if not blocks:
        raise RuntimeError(f"No molecule block found in {sdf_file}")
    coords, atoms = parse_atoms_from_block(blocks[0])
    if not coords:
        raise RuntimeError(f"No atoms parsed in {sdf_file}")
    return coords, atoms


# -----------------------
# Pairing + processing
# -----------------------


def build_pairs(protein_files: Sequence[Path], ligand_files: Sequence[Path]) -> List[Tuple[str, Path, Path]]:
    """Pair by shared stem; returns list of (stem, protein_path, ligand_path)."""
    prot_by_stem = {p.stem: p for p in protein_files}
    lig_by_stem = {p.stem: p for p in ligand_files}
    common = sorted(set(prot_by_stem).intersection(lig_by_stem))
    return [(stem, prot_by_stem[stem], lig_by_stem[stem]) for stem in common]


def collect_union_atom_types(pairs: Sequence[Tuple[str, Path, Path]]) -> List[str]:
    """Scan all pairs to build the union atom vocabulary."""
    atom_types = set()
    for _, prot_path, lig_path in tqdm(pairs, desc="Scanning atom types"):
        _, prot_atoms = read_pdb_atoms(prot_path)
        _, lig_atoms = read_sdf_atoms(lig_path)
        atom_types.update(prot_atoms)
        atom_types.update(lig_atoms)
    if not atom_types:
        raise RuntimeError("No atoms found across provided pairs; please check inputs.")
    return sorted(atom_types)


def process_pair(
    *,
    stem: str,
    protein_path: Path,
    ligand_path: Path,
    out_dir: Path,
    atom_to_idx: Dict[str, int],
) -> None:
    """Convert one protein+ligand pair into Pointcept-friendly npy blobs."""
    prot_coords, prot_atoms = read_pdb_atoms(protein_path)
    lig_coords, lig_atoms = read_sdf_atoms(ligand_path)

    coords = prot_coords + lig_coords
    atoms = prot_atoms + lig_atoms

    coords_arr = np.asarray(coords, dtype=np.float32)
    atom_type_arr = np.zeros((len(coords), len(atom_to_idx)), dtype=np.float32)
    atom_indices = [atom_to_idx[a] for a in atoms]
    atom_type_arr[np.arange(len(coords)), atom_indices] = 1.0

    identity = np.zeros((len(coords), 2), dtype=np.float32)
    identity[: len(prot_coords), 0] = 1.0
    identity[len(prot_coords) :, 1] = 1.0

    scene_dir = out_dir / stem
    scene_dir.mkdir(parents=True, exist_ok=True)
    np.save(scene_dir / "coord.npy", coords_arr)
    np.save(scene_dir / "atom_type.npy", atom_type_arr)
    np.save(scene_dir / "identity.npy", identity)


def main() -> None:
    parser = argparse.ArgumentParser(description="Protein+ligand pair converter (PDB+SDF).")
    parser.add_argument(
        "--protein-dir",
        required=True,
        type=Path,
        help="Directory containing protein .pdb files (non-recursive).",
    )
    parser.add_argument(
        "--ligand-dir",
        required=True,
        type=Path,
        help="Directory containing ligand .sdf files (non-recursive).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Destination directory for processed paired scenes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing scene folders if they exist.",
    )
    args = parser.parse_args()

    protein_files = sorted(args.protein_dir.glob("*.pdb"))
    ligand_files = sorted(args.ligand_dir.glob("*.sdf"))
    if not protein_files:
        raise FileNotFoundError(f"No .pdb files found in {args.protein_dir}")
    if not ligand_files:
        raise FileNotFoundError(f"No .sdf files found in {args.ligand_dir}")

    pairs = build_pairs(protein_files, ligand_files)
    missing_protein = sorted(set(p.stem for p in ligand_files) - set(p.stem for p in protein_files))
    missing_ligand = sorted(set(p.stem for p in protein_files) - set(p.stem for p in ligand_files))

    if not pairs:
        raise RuntimeError(
            "No pairs found (matching stems). "
            "Ensure protein files are <stem>.pdb and ligand files are <stem>.sdf."
        )

    if missing_protein:
        print(f"Warning: {len(missing_protein)} ligands have no matching protein (example: {missing_protein[:5]}).")
    if missing_ligand:
        print(f"Warning: {len(missing_ligand)} proteins have no matching ligand (example: {missing_ligand[:5]}).")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    atom_types = collect_union_atom_types(pairs)
    atom_to_idx = {atom: i for i, atom in enumerate(atom_types)}

    skipped = 0
    for stem, prot_path, lig_path in tqdm(pairs, desc="Processing pairs"):
        scene_dir = args.output_dir / stem
        if scene_dir.exists():
            if args.overwrite:
                shutil.rmtree(scene_dir)
            else:
                skipped += 1
                continue
        process_pair(
            stem=stem,
            protein_path=prot_path,
            ligand_path=lig_path,
            out_dir=args.output_dir,
            atom_to_idx=atom_to_idx,
        )

    with (args.output_dir / "atom_types.json").open("w") as handle:
        json.dump(atom_types, handle, indent=2)

    print(
        f"Processed {len(pairs) - skipped} pairs (skipped {skipped}). "
        f"Union atom types ({len(atom_types)}): {', '.join(atom_types)}"
    )


if __name__ == "__main__":
    main()


