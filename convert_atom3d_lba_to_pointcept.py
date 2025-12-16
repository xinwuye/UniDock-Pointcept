#!/usr/bin/env python3
"""
Convert Atom3D LBA (Ligand Binding Affinity) LMDB dataset into the same folder structure
as produced by preprocess_pair.py, plus an affinity label file per sample.

Input (Atom3D LBA):
  <lba_root>/data/{train,val,test}/data.mdb

Each LMDB entry (per Atom3D docs) contains keys like:
  - atoms_protein: pandas.DataFrame
  - atoms_ligand: pandas.DataFrame
  - scores: dict, including 'neglog_aff' (pK)
  - id: str (PDB code)

Output:
  <output_dir>/
    atom_types.json
    train/<sample_id>/
      coord.npy        float32 (N,3) protein then ligand
      atom_type.npy    float32 (N,C) one-hot over UNION vocabulary across ALL splits
      identity.npy     float32 (N,2) protein=[1,0], ligand=[0,1]
      affinity.npy     float32 (1,) label = scores['neglog_aff']
    val/<sample_id>/...
    test/<sample_id>/...

Usage:
  python convert_atom3d_lba_to_pointcept.py \
    --lba-root data/lba/split-by-sequence-identity-30 \
    --output-dir data/lba_pointcept/split-by-seq30
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm


def _find_col(df, candidates: Sequence[str]) -> str:
    cols = set(map(str, df.columns))
    for c in candidates:
        if c in cols:
            return c
    # try case-insensitive match
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(f"Could not find any of columns {list(candidates)} in dataframe columns: {list(df.columns)}")


def _extract_coords_and_atoms(df) -> Tuple[np.ndarray, List[str]]:
    """
    Extract coordinates (N,3) and atom symbols (list[str]) from an Atom3D atoms dataframe.
    Expected columns (typical): x,y,z and element.
    """
    xcol = _find_col(df, ["x", "X"])
    ycol = _find_col(df, ["y", "Y"])
    zcol = _find_col(df, ["z", "Z"])
    ecol = _find_col(df, ["element", "elem", "Element", "symbol"])

    coords = df[[xcol, ycol, zcol]].to_numpy(dtype=np.float32, copy=True)
    atoms_raw = df[ecol].astype(str).tolist()
    atoms = [a.strip().upper() for a in atoms_raw]
    return coords, atoms


def _iter_split(ds) -> Iterable[dict]:
    # LMDBDataset supports __len__ and __getitem__
    for i in range(len(ds)):
        yield ds[i]


def _collect_union_atom_types(datasets: Dict[str, object]) -> List[str]:
    atom_types = set()
    for split, ds in datasets.items():
        for item in tqdm(_iter_split(ds), desc=f"Scanning atom types ({split})", total=len(ds)):
            prot_df = item.get("atoms_protein")
            lig_df = item.get("atoms_ligand")
            if prot_df is None or lig_df is None:
                raise KeyError("Expected keys 'atoms_protein' and 'atoms_ligand' in LMDB entry.")
            _, prot_atoms = _extract_coords_and_atoms(prot_df)
            _, lig_atoms = _extract_coords_and_atoms(lig_df)
            atom_types.update(prot_atoms)
            atom_types.update(lig_atoms)

    if not atom_types:
        raise RuntimeError("No atoms detected across dataset.")
    return sorted(atom_types)


def _make_one_hot(atom_indices: Sequence[int], num_classes: int) -> np.ndarray:
    arr = np.zeros((len(atom_indices), num_classes), dtype=np.float32)
    arr[np.arange(len(atom_indices)), list(atom_indices)] = 1.0
    return arr


def _safe_scene_dir(base: Path, sample_id: str, used: set[str]) -> Path:
    """
    Ensure per-split unique folder name. If id repeats, append -{n}.
    """
    name = sample_id
    if name in used:
        n = 1
        while f"{sample_id}-{n}" in used:
            n += 1
        name = f"{sample_id}-{n}"
    used.add(name)
    return base / name


def _process_item(
    item: dict,
    *,
    out_dir: Path,
    atom_to_idx: Dict[str, int],
) -> None:
    prot_df = item["atoms_protein"]
    lig_df = item["atoms_ligand"]

    prot_coords, prot_atoms = _extract_coords_and_atoms(prot_df)
    lig_coords, lig_atoms = _extract_coords_and_atoms(lig_df)

    coords = np.concatenate([prot_coords, lig_coords], axis=0).astype(np.float32, copy=False)
    atoms = prot_atoms + lig_atoms

    atom_indices = [atom_to_idx[a] for a in atoms]
    atom_type = _make_one_hot(atom_indices, num_classes=len(atom_to_idx))

    identity = np.zeros((coords.shape[0], 2), dtype=np.float32)
    identity[: prot_coords.shape[0], 0] = 1.0
    identity[prot_coords.shape[0] :, 1] = 1.0

    scores = item.get("scores", {})
    if not isinstance(scores, dict) or "neglog_aff" not in scores:
        raise KeyError("Expected item['scores']['neglog_aff'] for affinity label.")
    affinity = np.asarray([float(scores["neglog_aff"])], dtype=np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "coord.npy", coords)
    np.save(out_dir / "atom_type.npy", atom_type)
    np.save(out_dir / "identity.npy", identity)
    np.save(out_dir / "affinity.npy", affinity)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Atom3D LBA LMDB to Pointcept-style npy folders.")
    ap.add_argument(
        "--lba-root",
        required=True,
        type=Path,
        help="Path like data/lba/split-by-sequence-identity-30 (contains data/train|val|test/data.mdb).",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output root directory.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-sample folders if they exist.",
    )
    args = ap.parse_args()

    try:
        from atom3d.datasets import LMDBDataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "atom3d is required. Install it in your environment, e.g. `pip install atom3d`."
        ) from e

    split_to_path = {
        "train": args.lba_root / "data" / "train",
        "val": args.lba_root / "data" / "val",
        "test": args.lba_root / "data" / "test",
    }

    datasets: Dict[str, object] = {}
    for split, p in split_to_path.items():
        if not p.exists():
            raise FileNotFoundError(f"Split directory not found: {p}")
        datasets[split] = LMDBDataset(str(p))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1) global union atom vocabulary across ALL splits
    atom_types = _collect_union_atom_types(datasets)
    atom_to_idx = {a: i for i, a in enumerate(atom_types)}
    with (args.output_dir / "atom_types.json").open("w") as f:
        json.dump(atom_types, f, indent=2)

    # 2) process splits
    for split, ds in datasets.items():
        split_out = args.output_dir / split
        split_out.mkdir(parents=True, exist_ok=True)
        used_names: set[str] = set()

        for item in tqdm(_iter_split(ds), desc=f"Converting ({split})", total=len(ds)):
            sample_id = str(item.get("id", "unknown"))
            scene_dir = _safe_scene_dir(split_out, sample_id, used_names)
            if scene_dir.exists():
                if args.overwrite:
                    shutil.rmtree(scene_dir)
                else:
                    continue
            _process_item(item, out_dir=scene_dir, atom_to_idx=atom_to_idx)

    print(
        f"Done. Wrote Pointcept-style folders to {args.output_dir}. "
        f"Union atom types: {len(atom_types)}"
    )


if __name__ == "__main__":
    main()


