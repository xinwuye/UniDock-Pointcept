import os
import sys
import numpy as np

# Add project root to sys.path to allow importing pointcept
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse
from functools import partial
from pointcept.models.utils.serialization.bwms import get_bwms_coder

def process_sample(sample_path, coder):
    coord_path = os.path.join(sample_path, "coord.npy")
    atom_type_path = os.path.join(sample_path, "atom_type.npy")
    out_path = os.path.join(sample_path, "bwms_order.npy")
    
    # Skip if already exists
    if os.path.exists(out_path):
        return
        
    try:
        coord = np.load(coord_path)
        atom_type = np.load(atom_type_path)
        
        # BWMS computation
        # Note: coder.encode returns 'code' (ranks). 
        # But for pre-calculation, we want to match how PTv3 uses it.
        # PTv3's serialization() expects encode() to return a 'code' tensor.
        code = coder.encode(coord, atom_type)
        
        np.save(out_path, code.astype(np.int64))
    except Exception as e:
        print(f"Error processing {sample_path}: {e}")

def process_sample_wrapper(path, data_root):
    # Re-get coder inside process to avoid pickle issues
    atom_types_json = os.path.join(data_root, "atom_types.json")
    coder = get_bwms_coder(atom_types_json)
    process_sample(path, coder)

def main():
    parser = argparse.ArgumentParser(description="Precompute BWMS serialization orders for a dataset.")
    parser.add_argument("--data-root", type=str, required=True, help="Path to the dataset root directory.")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count(), help="Number of worker processes.")
    args = parser.parse_args()

    data_root = args.data_root
    atom_types_json = os.path.join(data_root, "atom_types.json")
    
    if not os.path.exists(atom_types_json):
        raise FileNotFoundError(f"Could not find atom_types.json at {atom_types_json}")

    splits = ["train", "val", "test"]
    all_sample_paths = []
    
    for split in splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir): 
            print(f"Skipping split {split}, directory not found.")
            continue
        samples = os.listdir(split_dir)
        for s in samples:
            sample_path = os.path.join(split_dir, s)
            if os.path.isdir(sample_path):
                all_sample_paths.append(sample_path)
            
    print(f"Found {len(all_sample_paths)} samples in {data_root}. Starting pre-computation with {args.num_workers} workers...")
    
    # Use partial to pass data_root to the wrapper
    worker_fn = partial(process_sample_wrapper, data_root=data_root)
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        list(tqdm(executor.map(worker_fn, all_sample_paths), total=len(all_sample_paths)))

if __name__ == "__main__":
    main()
